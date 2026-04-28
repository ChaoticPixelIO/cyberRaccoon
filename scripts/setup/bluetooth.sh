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
# Run via: sudo scripts/setup.sh --bt
#
# After running this script, pair from the target computer's
# Bluetooth settings to connect.
# ===========================================================================

set -euo pipefail

DEVICE_NAME="CyberRaccoon"
DEVICE_CLASS="0x002540"   # Keyboard + Mouse combo

echo "[INFO] Setting up CyberRaccoon Bluetooth HID..."

# ---------------------------------------------------------------------------
# Step 0: Install required system packages
# ---------------------------------------------------------------------------
echo "[INFO] Checking system dependencies..."

PACKAGES_NEEDED=""
dpkg -s bluez &>/dev/null          || PACKAGES_NEEDED="$PACKAGES_NEEDED bluez"
dpkg -s libcap2-bin &>/dev/null    || PACKAGES_NEEDED="$PACKAGES_NEEDED libcap2-bin"
dpkg -s python3-dbus &>/dev/null   || PACKAGES_NEEDED="$PACKAGES_NEEDED python3-dbus"
dpkg -s python3-gi &>/dev/null     || PACKAGES_NEEDED="$PACKAGES_NEEDED python3-gi"

if [ -n "$PACKAGES_NEEDED" ]; then
    echo "[INFO] Installing missing packages:$PACKAGES_NEEDED"
    apt-get update -qq
    apt-get install -y $PACKAGES_NEEDED
    echo "[OK] System packages installed."
else
    echo "[OK] All system packages present."
fi

# ---------------------------------------------------------------------------
# Step 1: Disable BlueZ plugins that pollute the HID SDP record
# ---------------------------------------------------------------------------
# We need bluetoothd to advertise ONLY HID + base profiles. Without this,
# disabling just the "input" plugin is not enough -- BlueZ's audio plugins
# still register Audio Source/Sink + AVRCP UUIDs, and macOS sees a
# multi-profile device and prioritizes audio over HID, then aborts before
# opening the HID L2CAP channels. (Concrete failure: Mac sends
# AuthorizeService for AVRCP UUID 0x110e, link is torn down before the HID
# handshake; kernel logs "ACL packet for unknown connection handle".)
#
# Plugins disabled:
#   input         BlueZ's HID host plugin would claim HID profile
#                 registration and block our L2CAP server on PSM 17/19.
#   a2dp,avrcp    Audio profiles -- the source of the SDP pollution
#                 described above. Removing them lets macOS go straight
#                 to HID.
#   sap           SIM Access Profile -- irrelevant on a Pi.
#   network,health  PAN and Health Device -- also irrelevant for HID-only.
#   gap           Generic Access Profile plugin -- the GAP UUID (0x1800)
#                 is still exposed by core BlueZ even with this plugin off.
#
# We use a systemd drop-in so package upgrades don't revert the override.
# The empty `ExecStart=` clears whatever the package shipped, then the
# second `ExecStart=` sets ours.

DISABLE_PLUGINS="input,a2dp,avrcp,sap,network,health,gap"

# Resolve bluetoothd path from whichever unit file the package shipped.
PKG_BT_UNIT=""
for candidate in /lib/systemd/system/bluetooth.service /usr/lib/systemd/system/bluetooth.service; do
    [ -f "$candidate" ] && PKG_BT_UNIT="$candidate" && break
done

if [ -z "$PKG_BT_UNIT" ]; then
    echo "[ERROR] Cannot find bluetooth.service. Is bluez installed?"
    exit 1
fi

# Extract the bluetoothd binary path from the package's ExecStart line.
# Strip any prior `-P ...` argument (legacy in-place edits added "-P input").
BLUETOOTHD_BIN=$(grep -E '^ExecStart=' "$PKG_BT_UNIT" \
    | head -1 | sed 's|^ExecStart=||' | awk '{print $1}')

if [ -z "$BLUETOOTHD_BIN" ] || [ ! -x "$BLUETOOTHD_BIN" ]; then
    echo "[ERROR] Could not resolve bluetoothd binary from $PKG_BT_UNIT"
    exit 1
fi

BT_DROPIN_DIR="/etc/systemd/system/bluetooth.service.d"
BT_DROPIN_FILE="$BT_DROPIN_DIR/10-cyberraccoon-disable-plugins.conf"
BT_DROPIN_CONTENT="# Managed by scripts/setup/bluetooth.sh -- re-run to refresh.
# Disables BlueZ plugins that pollute the SDP record with non-HID UUIDs.
# See script comments for the full rationale.
[Service]
ExecStart=
ExecStart=$BLUETOOTHD_BIN -P $DISABLE_PLUGINS
"

mkdir -p "$BT_DROPIN_DIR"
TMP_DROPIN="$(mktemp)"
printf '%s' "$BT_DROPIN_CONTENT" > "$TMP_DROPIN"
if [ ! -f "$BT_DROPIN_FILE" ] || ! cmp -s "$TMP_DROPIN" "$BT_DROPIN_FILE"; then
    install -m 644 "$TMP_DROPIN" "$BT_DROPIN_FILE"
    systemctl daemon-reload
    echo "[OK] Wrote $BT_DROPIN_FILE"
    echo "     Plugins disabled: $DISABLE_PLUGINS"
else
    echo "[INFO] $BT_DROPIN_FILE already current"
fi
rm -f "$TMP_DROPIN"

# Revert any legacy in-place edits of the package unit file. Earlier
# versions of this script appended `-P input` directly to the package's
# ExecStart line. The drop-in above supersedes that, but we revert the
# in-place edit so the package file matches upstream and won't fight
# package upgrades.
for legacy in /lib/systemd/system/bluetooth.service /usr/lib/systemd/system/bluetooth.service; do
    if [ -f "$legacy" ] && grep -qE '^ExecStart=.* -P input *$' "$legacy" 2>/dev/null; then
        sed -i 's| -P input *$||' "$legacy"
        systemctl daemon-reload
        echo "[OK] Reverted legacy in-place '-P input' edit in $legacy"
    fi
done

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
echo "[INFO] Granting Bluetooth capabilities to Python venv interpreter..."

# Resolve $REPO_ROOT/venv/bin/python3 first (visible in logs); fall back to
# system python3 if the venv is absent. Both resolve to the same inode
# (/usr/bin/python3.13) on a normal Pi setup, so setcap operates on the same
# file either way -- but the explicit venv path makes the script's intent clear.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_PYTHON="$REPO_ROOT/venv/bin/python3"

if [ -e "$VENV_PYTHON" ]; then
    PYTHON_BIN="$(readlink -f "$VENV_PYTHON")"
    echo "[INFO] Resolved $VENV_PYTHON -> $PYTHON_BIN"
else
    echo "[WARN] $VENV_PYTHON not found (venv not yet created?). Falling back to system python3."
    PYTHON_BIN="$(readlink -f "$(which python3)")"
fi

if [ -z "$PYTHON_BIN" ] || [ ! -f "$PYTHON_BIN" ]; then
    echo "[WARN] Could not locate Python binary; skipping setcap."
    echo "       After creating the venv, re-run: sudo scripts/setup.sh --bt"
else
    # KEEP BOTH CAPS -- cap_net_bind_service is required for L2CAP PSM 17/19
    # bind (see bluetooth_device.py:381-386); setcap REPLACES rather than
    # merges, so dropping one would silently break socket creation.
    setcap 'cap_net_bind_service+eip cap_net_admin+eip' "$PYTHON_BIN"
    echo "[OK] Capabilities set on $PYTHON_BIN"
    echo "     cap_net_bind_service: L2CAP PSM 17/19 bind"
    echo "     cap_net_admin:        hciconfig adapter config"
    # Verify
    ACTUAL=$(getcap "$PYTHON_BIN" 2>/dev/null || true)
    echo "[VERIFY] $ACTUAL"
    echo "[NOTE]   Capabilities live on the inode of $PYTHON_BIN."
    echo "         Recreating the venv with the same Python version preserves them."
    echo "         If you upgrade or reinstall system Python, re-run: sudo scripts/setup.sh --bt"
    echo "[NOTE]   These caps apply to ALL Python processes using this interpreter."
    echo "         On a single-purpose Pi this is acceptable; document if multi-tenant."
fi

# ---------------------------------------------------------------------------
# Step 7: Disable PipeWire/wireplumber BlueZ monitor (persistent)
# ---------------------------------------------------------------------------
# Why: wireplumber's bluez monitor registers Audio Source/Sink/AVRCP/Handsfree
# UUIDs on hci0 whenever it starts. macOS' SDP discovery during pair sees an
# audio device and caches that -- no HID UUID 0x1124 in the bond record, so
# macOS will never open PSM 17/19. Disable ONLY the bluez monitor; ALSA,
# v4l2, and AirPlay (uxplay -> GStreamer -> ALSA) are unaffected.
#
# Targets wireplumber 0.5+ (Debian 13 default uses SPA-JSON drop-ins).
echo "[INFO] Configuring wireplumber to disable BlueZ monitor at /etc/wireplumber/wireplumber.conf.d/51-disable-bluez.conf..."

WP_CONF_DIR="/etc/wireplumber/wireplumber.conf.d"
WP_CONF_FILE="$WP_CONF_DIR/51-disable-bluez.conf"
WP_CONF_CONTENT='# Disable PipeWire BlueZ monitor to prevent audio UUIDs from competing with
# our HID profile on hci0. Other monitors (alsa, v4l2) are unaffected, so
# AirPlay audio (uxplay -> GStreamer -> ALSA) keeps working.
#
# Managed by scripts/setup/bluetooth.sh -- re-run to refresh.
wireplumber.profiles = {
  main = {
    monitor.bluez = disabled
    monitor.bluez-midi = disabled
  }
}
'

mkdir -p "$WP_CONF_DIR"
# Idempotency via byte-exact compare (codex HIGH concern fix):
# The previous form `[ "$(cat ...)" != "$WP_CONF_CONTENT" ]` was broken
# because command substitution strips trailing newlines, so a byte-identical
# file compared unequal and the script rewrote every run. Use `cmp -s`
# against a `mktemp` tempfile and `install -m 644` for an atomic copy
# that sets perms in one operation.
TMP_CONF="$(mktemp)"
printf '%s' "$WP_CONF_CONTENT" > "$TMP_CONF"
if [ ! -f "$WP_CONF_FILE" ] || ! cmp -s "$TMP_CONF" "$WP_CONF_FILE"; then
    install -m 644 "$TMP_CONF" "$WP_CONF_FILE"
    echo "[OK] Wrote $WP_CONF_FILE"
else
    echo "[INFO] $WP_CONF_FILE already current"
fi
rm -f "$TMP_CONF"

# Best-effort runtime apply (cold boot will load it cleanly anyway).
# The setup script runs as root via sudo; wireplumber runs in the login
# user's systemd-user instance. Use --machine=$LOGIN_USER@.host to target
# the user bus from a root shell. Failure is non-fatal -- cold reboot picks
# up the config file naturally on next user login.
#
# Codex MEDIUM concern fix: distinguish "transport unavailable" (no user
# bus / systemd-machined missing) from "unit simply inactive" via a
# `show-environment` probe before the is-active check.
LOGIN_USER="${SUDO_USER:-$(logname 2>/dev/null || echo '')}"
if [ -n "$LOGIN_USER" ]; then
    # Probe transport first -- distinguishes "user bus missing" from "unit inactive"
    if ! systemctl --user --machine="${LOGIN_USER}@.host" --no-pager show-environment >/dev/null 2>&1; then
        echo "[WARN] Cannot reach user systemd for $LOGIN_USER (systemd-machined or user bus unavailable)."
        echo "       Cold boot will load the wireplumber config naturally."
    elif systemctl --user --machine="${LOGIN_USER}@.host" is-active wireplumber >/dev/null 2>&1; then
        if systemctl --user --machine="${LOGIN_USER}@.host" restart wireplumber 2>/dev/null; then
            echo "[OK] Restarted wireplumber for $LOGIN_USER (config applied to current session)"
        else
            echo "[WARN] Transport reachable but restart failed. Cold boot will apply."
        fi
    else
        echo "[INFO] wireplumber not active for $LOGIN_USER (will load on next login)."
    fi
else
    echo "[INFO] Could not determine login user; runtime restart skipped (cold boot will apply)."
fi

# ---------------------------------------------------------------------------
# Step 8: Persist HID device class in /etc/bluetooth/main.conf
# ---------------------------------------------------------------------------
# Why: BTHID-01 originally relied on `setcap cap_net_admin+eip` on the venv
# Python interpreter so `_configure_adapter` could call `hciconfig hci0 class
# 0x002540` at runtime. That design failed on real hardware (cycle-9
# checkpoint, see .planning/phases/04-harden-bluetooth-hid-setup/04-03-SUMMARY.md
# Step 8): file caps DO NOT propagate to subprocess'd hciconfig because the
# parent's CapInh=0 / CapAmb=0 zero out NewPermitted in the child. By
# persisting the class in main.conf, bluetoothd applies it on its own start
# (it already holds CAP_NET_ADMIN), so no runtime privileged op is required.
#
# Idempotency: mirror Step 7 (cmp -s + mktemp + install -m 644). Unlike
# Step 7 (which owns the entire wireplumber drop-in), main.conf is shared
# with stock BlueZ defaults. We must preserve other [General] keys
# (Name, DiscoverableTimeout, etc.) — so we render the merged content
# in a tempfile and only install if it differs byte-exact from the live file.
BT_CONF_FILE="/etc/bluetooth/main.conf"
TARGET_CLASS="0x002540"

if [ ! -f "$BT_CONF_FILE" ]; then
    echo "[WARN] $BT_CONF_FILE not found — skipping Class persistence (is bluez installed?)"
else
    echo "[INFO] Ensuring [General] Class = $TARGET_CLASS in $BT_CONF_FILE..."

    TMP_CONF="$(mktemp)"
    # awk merge:
    #   - if a Class= line exists anywhere in [General], replace it in place
    #   - otherwise insert `Class = 0x002540` at the end of [General]
    #   - all other sections (e.g. [Policy], [GATT]) and keys are preserved
    #   - if [General] does not exist, append it at EOF with the Class line
    awk -v target="$TARGET_CLASS" '
        BEGIN { in_general = 0; found_general = 0; class_set = 0 }
        /^\[General\]/ {
            in_general = 1; found_general = 1; print; next
        }
        /^\[/ && !/^\[General\]/ {
            if (in_general && !class_set) {
                print "Class = " target
                class_set = 1
            }
            in_general = 0; print; next
        }
        in_general && /^[[:space:]]*Class[[:space:]]*=/ {
            if (!class_set) {
                print "Class = " target
                class_set = 1
            }
            next
        }
        { print }
        END {
            if (found_general && !class_set) {
                print "Class = " target
            } else if (!found_general) {
                print ""
                print "[General]"
                print "Class = " target
            }
        }
    ' "$BT_CONF_FILE" > "$TMP_CONF"

    if ! cmp -s "$TMP_CONF" "$BT_CONF_FILE"; then
        install -m 644 "$TMP_CONF" "$BT_CONF_FILE"
        echo "[OK] Wrote $BT_CONF_FILE (Class = $TARGET_CLASS)"
        # Restart bluetoothd once so the new class takes effect immediately.
        # Cold reboot would also pick it up; this avoids a reboot in the
        # interactive setup flow.
        if systemctl restart bluetooth 2>/dev/null; then
            echo "[OK] Restarted bluetooth.service (class change applied)"
        else
            echo "[WARN] systemctl restart bluetooth failed; cold reboot will apply"
        fi
    else
        echo "[INFO] $BT_CONF_FILE already current (Class = $TARGET_CLASS)"
    fi
    rm -f "$TMP_CONF"
fi

# ---------------------------------------------------------------------------
# Step 9: Warn if a cyberraccoon web server is already running
# ---------------------------------------------------------------------------
# File capabilities apply only on execve(), so a server started BEFORE this
# script ran will still have CapEff=0 even though /usr/bin/python3.13 now
# carries cap_net_bind_service+cap_net_admin. Without this warning the user
# clicks Connect and sees a misleading "run sudo scripts/setup.sh --bt" error
# (which they just did). Tell them to restart instead.
# Filter on `comm` so only the actual python interpreter matches. Bare
# pgrep -f would also match the parent shell whose argv literally contains
# the search regex (e.g. the `pgrep` invocation inside a `bash -c '…'`).
RUNNING_PIDS=$(ps -eo pid=,comm=,args= \
    | awk '$2 ~ /^python/ && $0 ~ /-m cyberraccoon([[:space:]]|$)/ && $0 ~ /--web([[:space:]]|$)/ {print $1}' \
    | tr '\n' ' ' | xargs)

if [ -n "$RUNNING_PIDS" ]; then
    echo ""
    echo "[WARN] cyberraccoon --web is currently running (PID(s): $RUNNING_PIDS)."
    echo "       File capabilities apply only on process start, so the running"
    echo "       server still has no CAP_NET_BIND_SERVICE / CAP_NET_ADMIN."
    echo "       Restart it before clicking Connect:"

    # If the cyberraccoon.service unit owns one of the running pids, show
    # `systemctl restart`. Otherwise show the manual kill+relaunch path.
    SERVICE_PID=""
    if systemctl list-unit-files cyberraccoon.service >/dev/null 2>&1; then
        SERVICE_PID=$(systemctl show -p MainPID --value cyberraccoon.service 2>/dev/null || echo "")
    fi
    if [ -n "$SERVICE_PID" ] && [ "$SERVICE_PID" != "0" ] \
       && echo " $RUNNING_PIDS " | grep -q " $SERVICE_PID "; then
        echo "         sudo systemctl restart cyberraccoon"
    else
        echo "         # If you launched via systemd:"
        echo "         sudo systemctl restart cyberraccoon"
        echo "         # If you launched manually (e.g. via nohup / foreground):"
        echo "         kill $RUNNING_PIDS"
        echo "         # ...then re-run your launch command:"
        echo "         cd $REPO_ROOT && source ~/.apikeys && venv/bin/python3 -m cyberraccoon --web --host 0.0.0.0 --port 8000"
    fi
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
