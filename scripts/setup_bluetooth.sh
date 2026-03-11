#!/bin/bash
# ===========================================================================
# CyberRaccoon Bluetooth HID Setup Script
#
# Configures the Raspberry Pi as a Bluetooth HID device:
#   1. Disables BlueZ input plugin (prevents HID profile hijacking)
#   2. Restarts bluetooth service
#   3. Sets device class to Keyboard+Mouse combo (0x002540)
#   4. Sets device name to "CyberRaccoon"
#   5. Enables discoverable and pairable mode
#
# Run once before using Bluetooth HID:
#   sudo scripts/setup_bluetooth.sh
#
# After running this script, pair from the target computer's
# Bluetooth settings to connect.
# ===========================================================================

set -euo pipefail

DEVICE_NAME="CyberRaccoon"
DEVICE_CLASS="0x002540"   # Keyboard + Mouse combo

echo "[INFO] Setting up CyberRaccoon Bluetooth HID..."

# ---------------------------------------------------------------------------
# Step 1: Disable BlueZ input plugin
# ---------------------------------------------------------------------------
# The BlueZ "input" plugin hijacks HID Profile registration, preventing
# our custom L2CAP socket server from binding to PSM 17/19.
# We need to disable it so our code can manage HID connections directly.

BTSERVICE="/lib/systemd/system/bluetooth.service"

if [ -f "$BTSERVICE" ]; then
    # Check if already patched
    if grep -q "\-P input" "$BTSERVICE" 2>/dev/null; then
        echo "[INFO] BlueZ input plugin already disabled."
    else
        echo "[INFO] Disabling BlueZ input plugin..."
        # Add -P input flag to ExecStart line
        sed -i 's|ExecStart=.*/bluetoothd.*|& -P input|' "$BTSERVICE"
        systemctl daemon-reload
        echo "[OK] BlueZ input plugin disabled."
    fi
else
    echo "[WARN] bluetooth.service not found at $BTSERVICE"
    echo "       Trying alternative path..."

    # Try alternative location
    BTSERVICE="/usr/lib/systemd/system/bluetooth.service"
    if [ -f "$BTSERVICE" ]; then
        if ! grep -q "\-P input" "$BTSERVICE" 2>/dev/null; then
            sed -i 's|ExecStart=.*/bluetoothd.*|& -P input|' "$BTSERVICE"
            systemctl daemon-reload
            echo "[OK] BlueZ input plugin disabled (alternative path)."
        fi
    else
        echo "[ERROR] Cannot find bluetooth.service. Manual setup needed."
        echo "        Add '-P input' to the bluetoothd ExecStart line."
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
# Step 2: Restart Bluetooth service
# ---------------------------------------------------------------------------
echo "[INFO] Restarting bluetooth service..."
systemctl restart bluetooth
sleep 2

# Verify service is running
if systemctl is-active --quiet bluetooth; then
    echo "[OK] Bluetooth service running."
else
    echo "[ERROR] Bluetooth service failed to start."
    systemctl status bluetooth --no-pager | head -10
    exit 1
fi

# ---------------------------------------------------------------------------
# Step 3: Configure adapter
# ---------------------------------------------------------------------------
echo "[INFO] Configuring Bluetooth adapter..."

# Power on
hciconfig hci0 up 2>/dev/null || true

# Set device class (Keyboard + Mouse combo)
hciconfig hci0 class "$DEVICE_CLASS"
echo "[OK] Device class set to $DEVICE_CLASS"

# Set device name
hciconfig hci0 name "$DEVICE_NAME"
echo "[OK] Device name set to '$DEVICE_NAME'"

# Make discoverable and pairable
hciconfig hci0 piscan
echo "[OK] Device is now discoverable and pairable."

# ---------------------------------------------------------------------------
# Step 4: Install persistent pairing agent
# ---------------------------------------------------------------------------
# A "NoInputNoOutput" D-Bus agent must be running persistently so that
# BlueZ uses "Just Works" pairing (no 6-digit passkey prompt).  The old
# approach of piping commands to bluetoothctl didn't work because the
# agent died as soon as the heredoc finished.
#
# Solution: install a small Python D-Bus agent as a systemd service.

AGENT_SCRIPT="/usr/local/bin/cyberraccoon-pair-agent.py"
AGENT_SERVICE="/etc/systemd/system/cyberraccoon-pair-agent.service"

echo "[INFO] Installing persistent pairing agent..."

cat > "$AGENT_SCRIPT" << 'PYEOF'
#!/usr/bin/env python3
"""CyberRaccoon Bluetooth NoInputNoOutput pairing agent.

Runs as a D-Bus service, auto-accepting all pairing requests so the
target computer can pair without a 6-digit passkey prompt.
"""
import dbus
import dbus.service
import dbus.mainloop.glib
from gi.repository import GLib

AGENT_PATH = "/cyberraccoon/pair_agent"

class PairAgent(dbus.service.Object):
    @dbus.service.method("org.bluez.Agent1", in_signature="", out_signature="")
    def Release(self):
        pass

    @dbus.service.method("org.bluez.Agent1", in_signature="os", out_signature="")
    def AuthorizeService(self, device, uuid):
        pass

    @dbus.service.method("org.bluez.Agent1", in_signature="o", out_signature="")
    def RequestAuthorization(self, device):
        pass

    @dbus.service.method("org.bluez.Agent1", in_signature="o", out_signature="u")
    def RequestPasskey(self, device):
        return dbus.UInt32(0)

    @dbus.service.method("org.bluez.Agent1", in_signature="ouq", out_signature="")
    def DisplayPasskey(self, device, passkey, entered):
        pass

    @dbus.service.method("org.bluez.Agent1", in_signature="ou", out_signature="")
    def RequestConfirmation(self, device, passkey):
        pass  # auto-confirm

    @dbus.service.method("org.bluez.Agent1", in_signature="", out_signature="")
    def Cancel(self):
        pass

if __name__ == "__main__":
    dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)
    bus = dbus.SystemBus()
    agent = PairAgent(bus, AGENT_PATH)
    manager = dbus.Interface(
        bus.get_object("org.bluez", "/org/bluez"),
        "org.bluez.AgentManager1",
    )
    manager.RegisterAgent(AGENT_PATH, "NoInputNoOutput")
    manager.RequestDefaultAgent(AGENT_PATH)
    GLib.MainLoop().run()
PYEOF
chmod +x "$AGENT_SCRIPT"

cat > "$AGENT_SERVICE" << 'UNITEOF'
[Unit]
Description=CyberRaccoon Bluetooth Pairing Agent
After=bluetooth.service
Requires=bluetooth.service

[Service]
ExecStart=/usr/bin/python3 /usr/local/bin/cyberraccoon-pair-agent.py
Restart=on-failure
RestartSec=3

[Install]
WantedBy=multi-user.target
UNITEOF

systemctl daemon-reload
systemctl enable --now cyberraccoon-pair-agent.service
sleep 1

if systemctl is-active --quiet cyberraccoon-pair-agent.service; then
    echo "[OK] Persistent pairing agent running."
else
    echo "[WARN] Pairing agent failed to start. Check: systemctl status cyberraccoon-pair-agent"
fi

# ---------------------------------------------------------------------------
# Step 5: bluetoothctl settings (adapter power/discoverable/pairable)
# ---------------------------------------------------------------------------
bluetoothctl <<EOF 2>/dev/null
power on
discoverable on
pairable on
EOF

# ---------------------------------------------------------------------------
# Step 6: Grant Python CAP_NET_BIND_SERVICE + CAP_NET_ADMIN
# ---------------------------------------------------------------------------
# Binding to L2CAP PSM 17/19 (< 0x1001) requires CAP_NET_BIND_SERVICE,
# and configuring the HCI adapter requires CAP_NET_ADMIN.
# We apply these to the real Python binary so the server can run as a
# normal user instead of root.
echo "[INFO] Granting Bluetooth capabilities to Python..."

# Resolve the real binary (setcap does not follow symlinks)
PYTHON_BIN=$(readlink -f "$(which python3)")

if [ -z "$PYTHON_BIN" ] || [ ! -f "$PYTHON_BIN" ]; then
    echo "[WARN] Could not locate Python binary; skipping setcap."
    echo "       Run manually: sudo setcap 'cap_net_bind_service+eip cap_net_admin+eip' /usr/bin/python3.X"
else
    setcap 'cap_net_bind_service+eip cap_net_admin+eip' "$PYTHON_BIN"
    echo "[OK] Capabilities set on $PYTHON_BIN"
    echo "     (cap_net_bind_service: L2CAP PSM 17/19, cap_net_admin: hciconfig)"
fi

echo ""
echo "==========================================="
echo "  CyberRaccoon Bluetooth HID Ready"
echo "==========================================="
echo "  Device name: $DEVICE_NAME"
echo "  Device class: $DEVICE_CLASS"
echo ""
echo "  Next steps:"
echo "  1. On the target computer, open Bluetooth settings"
echo "  2. Look for '$DEVICE_NAME' and click Connect/Pair"
echo "  3. Run the CyberRaccoon executor with --transport bt"
echo "==========================================="
