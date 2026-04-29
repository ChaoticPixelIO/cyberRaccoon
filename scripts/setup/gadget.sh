#!/bin/bash
# ===========================================================================
# CyberRaccoon USB Gadget Setup Script
#
# Creates a single-interface USB HID device with a combined report descriptor
# using Report IDs:
#   - Report ID 1: Keyboard (9-byte reports: 1 ID + 8 data)
#   - Report ID 2: Absolute-coordinate Mouse (8-byte reports: 1 ID + 7 data)
#
# A single HID interface is used instead of two separate ones because macOS
# only matches the HID driver to interfaces with bInterfaceSubClass=0, and
# silently skips the boot keyboard interface (subclass=1). Combining both
# into one interface with Report IDs works reliably across macOS and Linux.
#
# Device file: /dev/hidg0 (shared by keyboard and mouse)
#
# Run via: sudo scripts/setup.sh --gadget
# Idempotent: skips if gadget already configured.
# ===========================================================================

set -euo pipefail

# Note: do NOT early-exit if gadget is up. We still install/refresh the systemd unit and helper script so a repo upgrade picks up new versions. The helper script (cyberraccoon-usb-gadget-create) is itself idempotent.

echo "[INFO] Setting up CyberRaccoon USB Gadget..."

# ---------------------------------------------------------------------------
# Pi 5 cable-topology heads-up.
# The Pi 5 USB-C port is the only one that can act as a USB device — it's
# also the power input. A known dwc2 kernel bug (raspberrypi/linux#6289)
# means single-cable OTG (Pi powered by the target via the same USB-C cable)
# leaves the UDC "not attached". Documented workaround: USB power/data
# splitter — external power to the Pi, data cable from the Pi USB-C to
# the target.
# ---------------------------------------------------------------------------
if [ -f /proc/device-tree/model ] && grep -q "Raspberry Pi 5" /proc/device-tree/model; then
    echo ""
    echo "[NOTE] Raspberry Pi 5 cable check:"
    echo "       - Use the Pi USB-C port (the same one used for power) as the"
    echo "         data link to the target. Other USB ports on the Pi are"
    echo "         host-only and cannot act as a USB device."
    echo "       - Single-cable USB-C OTG (Pi powered by the target) hits a"
    echo "         known dwc2 kernel bug. If you are not already using a USB"
    echo "         power/data splitter cable, set one up: external power to"
    echo "         the Pi, data cable from Pi USB-C to the target."
    echo ""
fi

# ---------------------------------------------------------------------------
# Ensure the dwc2 overlay is enabled in /boot/firmware/config.txt under a
# section that applies to this Pi model. Without this the SoC's USB OTG
# controller never initialises in device mode, so /sys/class/udc stays empty
# regardless of cables or target power. We saw this in the wild: an existing
# `dtoverlay=dwc2,dr_mode=host` line nested under [cm5] never applied to a
# Pi 5 Model B, leaving USB Gadget setup permanently broken with no clue why.
#
# Parser respects [pi5] / [pi4] / [cm5] / [all] section headers — only lines
# in sections that match this model count. A `# CyberRaccoon:` sentinel
# comment makes the auto-add idempotent across re-runs.
# ---------------------------------------------------------------------------
CONFIG_TXT=""
for p in /boot/firmware/config.txt /boot/config.txt; do
    if [ -f "$p" ]; then
        CONFIG_TXT="$p"
        break
    fi
done

if [ -z "$CONFIG_TXT" ]; then
    echo "[ERROR] Could not find config.txt at /boot/firmware/config.txt or /boot/config.txt"
    exit 1
fi

# Determine model filter ("pi5" / "pi4" / etc.) for section matching
MODEL_FILTER="pi5"
if [ -f /proc/device-tree/model ]; then
    if grep -q "Raspberry Pi 4" /proc/device-tree/model; then
        MODEL_FILTER="pi4"
    elif grep -q "Raspberry Pi 3" /proc/device-tree/model; then
        MODEL_FILTER="pi3"
    fi
fi

# Returns one of: missing | host | peripheral | otg | default | <other>
DWC2_STATE=$(awk -v want="$MODEL_FILTER" '
    BEGIN { section = ""; result = "missing" }
    /^[[:space:]]*$/ { next }
    /^[[:space:]]*#/ { next }
    /^\[.*\]$/ {
        s = $0
        gsub(/^[[:space:]]*\[|\][[:space:]]*$/, "", s)
        section = tolower(s)
        next
    }
    {
        if (section != "" && section != "all" && section != want) next
        if ($0 !~ /^[[:space:]]*dtoverlay=dwc2($|[[:space:],])/) next
        mode = "default"
        n = split($0, parts, ",")
        for (i = 2; i <= n; i++) {
            if (parts[i] ~ /^[[:space:]]*dr_mode=/) {
                sub(/^[[:space:]]*dr_mode=/, "", parts[i])
                m = parts[i]
                gsub(/[[:space:]]/, "", m)
                mode = tolower(m)
            }
        }
        # Last applied line wins, mirroring firmware behaviour
        result = mode
    }
    END { print result }
' "$CONFIG_TXT")

# ---------------------------------------------------------------------------
# Install persistence artefacts BEFORE the dwc2 overlay state check.
#
# These steps are safe regardless of dwc2 state — the systemd unit's
# ConditionPathExistsGlob=/sys/class/udc/* guard means it harmlessly skips
# at boot when dwc2 isn't loaded yet. Installing them upfront means a
# fresh Pi that needs the dwc2 overlay added (the missing-overlay path
# below, which exits 0 after editing config.txt) self-heals on the next
# boot: dwc2 loads, the UDC appears, the unit fires, /dev/hidg0 is
# created — without the user having to re-run setup.sh --gadget.
#
# Doing this before the dwc2 check is critical: if we did it after, the
# missing-overlay early-exit would skip the install entirely and the
# post-reboot boot would have a loaded UDC but no unit to bind to it.
# ---------------------------------------------------------------------------

# Disable competing gadget drivers (e.g. g_ether) that hold the UDC.
# Only one gadget driver can bind to the UDC at a time.
# Raspberry Pi OS ships g_ether + rpi-usb-gadget-ics for USB networking;
# we must disable them permanently so HID gadget works on every boot.
BLACKLIST_CONF="/etc/modprobe.d/cyberraccoon-no-gether.conf"
if [ ! -f "$BLACKLIST_CONF" ]; then
    echo "[INFO] Blacklisting competing gadget modules..."
    cat > "$BLACKLIST_CONF" <<'CONF'
blacklist g_ether
blacklist g_serial
blacklist g_mass_storage
blacklist g_multi
blacklist g_webcam
CONF
fi

# Remove g_ether from modules-load if present
MODULES_CONF="/etc/modules-load.d/usb-gadget.conf"
if [ -f "$MODULES_CONF" ] && grep -q '^g_ether' "$MODULES_CONF"; then
    echo "[INFO] Removing g_ether from $MODULES_CONF"
    sed -i '/^g_ether/d' "$MODULES_CONF"
fi

# Disable the Raspberry Pi USB gadget ICS service
if systemctl is-enabled rpi-usb-gadget-ics.service &>/dev/null; then
    echo "[INFO] Disabling rpi-usb-gadget-ics.service"
    systemctl disable --now rpi-usb-gadget-ics.service 2>/dev/null || true
fi

# Unload any competing modules currently loaded
for mod in g_ether g_serial g_mass_storage g_multi g_webcam; do
    if lsmod | grep -q "^${mod} "; then
        echo "[INFO] Unloading competing gadget module: $mod"
        modprobe -r "$mod" 2>/dev/null || true
    fi
done

# Install persistent gadget creator + systemd unit.
# configfs gadgets live in tmpfs (/sys/kernel/config) and disappear on every
# reboot. The standalone helper script + systemd oneshot unit recreate the
# gadget on every boot. Mirrors the persistence pattern already used by
# cyberraccoon-pair-agent.service (see scripts/setup/bluetooth.sh).
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HELPER_SRC="$REPO_ROOT/scripts/lib/cyberraccoon-usb-gadget-create.sh"
HELPER_DST="/usr/local/sbin/cyberraccoon-usb-gadget-create"
UNIT_PATH="/etc/systemd/system/cyberraccoon-usb-gadget.service"

if [ ! -f "$HELPER_SRC" ]; then
    echo "[ERROR] Missing $HELPER_SRC — repo layout is broken."
    exit 1
fi

echo "[INFO] Installing gadget creator helper to $HELPER_DST..."
install -m 0755 "$HELPER_SRC" "$HELPER_DST"

echo "[INFO] Installing systemd unit at $UNIT_PATH..."
cat > "$UNIT_PATH" << 'UNITEOF'
[Unit]
Description=CyberRaccoon USB HID Gadget (configfs)
After=sys-kernel-config.mount systemd-modules-load.service
Requires=sys-kernel-config.mount
ConditionPathExistsGlob=/sys/class/udc/*

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/usr/local/sbin/cyberraccoon-usb-gadget-create
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
UNITEOF

systemctl daemon-reload
systemctl enable cyberraccoon-usb-gadget.service

# ---------------------------------------------------------------------------
# Now handle the dwc2 overlay state. On the missing-overlay path we exit 0
# after writing config.txt; on reboot the unit installed above fires and
# creates /dev/hidg0 automatically.
# ---------------------------------------------------------------------------
case "$DWC2_STATE" in
    peripheral|otg|default)
        echo "[INFO] dwc2 overlay already enabled in $CONFIG_TXT (mode: $DWC2_STATE)"
        ;;
    host)
        echo "[ERROR] dwc2 overlay in $CONFIG_TXT is set to host mode."
        echo "        With dr_mode=host the Pi can't act as a USB device."
        echo "        Edit $CONFIG_TXT and change dr_mode=host to"
        echo "        dr_mode=peripheral (or remove dr_mode= for OTG default),"
        echo "        then reboot before re-running this script."
        exit 1
        ;;
    missing)
        echo "[INFO] dwc2 overlay not enabled for this Pi in $CONFIG_TXT — adding it."
        BACKUP="${CONFIG_TXT}.cyberraccoon-bak"
        if [ ! -f "$BACKUP" ]; then
            cp "$CONFIG_TXT" "$BACKUP"
            echo "[INFO] Backup saved to $BACKUP"
        fi
        cat >> "$CONFIG_TXT" <<'CFG'

# CyberRaccoon: enable dwc2 in peripheral mode so the Pi USB-C port can
# act as a USB device (HID gadget). Required for /dev/hidg0.
[all]
dtoverlay=dwc2,dr_mode=peripheral
CFG
        echo ""
        echo "[REBOOT REQUIRED] dwc2 overlay added to $CONFIG_TXT."
        echo "                  The kernel must reboot before /sys/class/udc"
        echo "                  appears and /dev/hidg0 can be created."
        echo "                  cyberraccoon-usb-gadget.service is already"
        echo "                  installed and enabled — it will fire on the"
        echo "                  next boot once dwc2 is loaded."
        echo ""
        echo "                  Run:  sudo reboot"
        # Signal to setup.sh / install.sh that a reboot is required.
        # World-writable so the unprivileged installer can clean it up later.
        touch /tmp/cyberraccoon-needs-reboot 2>/dev/null && chmod 666 /tmp/cyberraccoon-needs-reboot 2>/dev/null || true
        exit 0
        ;;
    *)
        echo "[WARN] Unrecognised dwc2 dr_mode=$DWC2_STATE in $CONFIG_TXT — continuing anyway."
        ;;
esac

# Run the helper once now to create the gadget for the current boot.
# Same outcome as the old inline creation, but routed through the unit so
# logs go to the journal and the path matches future boots exactly.
# Only reached when dwc2 is already active — on the missing-overlay path
# the early exit above skips this step (a UDC won't exist until reboot).
echo "[INFO] Starting cyberraccoon-usb-gadget.service for the current boot..."
if systemctl restart cyberraccoon-usb-gadget.service; then
    if [ -e /dev/hidg0 ]; then
        echo "[OK] CyberRaccoon USB Gadget active — /dev/hidg0 ready."
        echo "     The unit is enabled and will recreate the gadget on every boot."
    else
        echo "[WARN] cyberraccoon-usb-gadget.service started but /dev/hidg0 is missing."
        echo "       Run 'systemctl status cyberraccoon-usb-gadget' for details."
        echo "       Most common cause on Pi 5: target unplugged or single-cable USB-C bug."
    fi
else
    echo "[WARN] cyberraccoon-usb-gadget.service did not start cleanly."
    echo "       Run 'sudo systemctl status cyberraccoon-usb-gadget' and"
    echo "       'journalctl -u cyberraccoon-usb-gadget' for details."
fi
