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

GADGET_DIR="/sys/kernel/config/usb_gadget/cyber_raccoon"

# Check if already configured AND device file exists
if [ -d "$GADGET_DIR" ] && [ -e /dev/hidg0 ]; then
    echo "[INFO] USB Gadget already configured at $GADGET_DIR — skipping."
    exit 0
fi

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
        echo ""
        echo "                  Run:  sudo reboot"
        echo "                  Then re-run: sudo scripts/setup.sh --gadget"
        exit 0
        ;;
    *)
        echo "[WARN] Unrecognised dwc2 dr_mode=$DWC2_STATE in $CONFIG_TXT — continuing anyway."
        ;;
esac

# ---------------------------------------------------------------------------
# Disable competing gadget drivers (e.g. g_ether) that hold the UDC.
# Only one gadget driver can bind to the UDC at a time.
# Raspberry Pi OS ships g_ether + rpi-usb-gadget-ics for USB networking;
# we must disable them permanently so HID gadget works on every boot.
# ---------------------------------------------------------------------------
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

# Clean up stale configfs gadget (dir exists but /dev/hidg0 missing)
if [ -d "$GADGET_DIR" ] && [ ! -e /dev/hidg0 ]; then
    echo "[INFO] Cleaning up stale gadget configuration..."
    echo "" > "$GADGET_DIR/UDC" 2>/dev/null || true
    rm -f "$GADGET_DIR/configs/c.1/hid.combo" 2>/dev/null || true
    rmdir "$GADGET_DIR/configs/c.1/strings/0x409" 2>/dev/null || true
    rmdir "$GADGET_DIR/configs/c.1" 2>/dev/null || true
    rmdir "$GADGET_DIR/functions/hid.combo" 2>/dev/null || true
    rmdir "$GADGET_DIR/strings/0x409" 2>/dev/null || true
    rmdir "$GADGET_DIR" 2>/dev/null || true
fi

# Load composite module
modprobe libcomposite

# ---------------------------------------------------------------------------
# Verify UDC is available.
#   - Pi 4B: needs dtoverlay=dwc2 in /boot/firmware/config.txt + reboot.
#   - Pi 5: dwc2 is loaded automatically, but single-cable USB-C OTG is
#     affected by a kernel bug (raspberrypi/linux#6289) — UDC stays
#     "not attached". A USB power/data splitter cable is the documented
#     workaround. Bluetooth HID (--bt) avoids USB entirely.
# ---------------------------------------------------------------------------
if [ -z "$(ls /sys/class/udc 2>/dev/null)" ]; then
    echo "[ERROR] The Pi cannot present itself as a USB device right now."
    echo "        (Technical detail: no USB Device Controller is exposed by"
    echo "         the kernel — /sys/class/udc is empty.)"
    echo ""
    if [ -f /proc/device-tree/model ] && grep -q "Raspberry Pi 5" /proc/device-tree/model; then
        echo "        On Pi 5 the USB-C port only switches to device mode when it"
        echo "        sees a real host on the other end. Common reasons it doesn't:"
        echo ""
        echo "        1. The target computer is powered off — turn it on, then"
        echo "           re-run this script."
        echo "        2. The cable plugged into the Pi USB-C port doesn't carry"
        echo "           data (some chargers / cables are power-only). Try a"
        echo "           known-good USB-C data cable to the target."
        echo "        3. You are powering the Pi through the same single USB-C"
        echo "           cable from the target. This hits a known dwc2 kernel"
        echo "           bug — use a USB power/data splitter so the Pi has its"
        echo "           own power and a separate data link to the target."
        echo ""
        echo "        Quick check (after fixing): ls /sys/class/udc — should"
        echo "        list a controller (e.g. xhci-hcd.0.auto)."
        echo "        Or skip USB entirely with Bluetooth HID:"
        echo "        sudo scripts/setup.sh --bt"
    else
        echo "        On this Pi model the dwc2 overlay must be enabled. Add"
        echo "        the following line to /boot/firmware/config.txt and reboot:"
        echo "            dtoverlay=dwc2"
    fi
    exit 1
fi

# Create gadget directory
mkdir -p "$GADGET_DIR"
cd "$GADGET_DIR"

# Device identifiers
echo 0x1d6b > idVendor      # Linux Foundation
echo 0x0104 > idProduct     # Multifunction Composite Gadget
echo 0x0100 > bcdDevice     # Device version
echo 0x0200 > bcdUSB        # USB 2.0

# Strings (English)
mkdir -p strings/0x409
echo "CyberRaccoon"       > strings/0x409/manufacturer
echo "AI HID Controller"  > strings/0x409/product
echo "CR00001"            > strings/0x409/serialnumber

# Configuration
mkdir -p configs/c.1/strings/0x409
echo "CyberRaccoon HID Config" > configs/c.1/strings/0x409/configuration
echo 250 > configs/c.1/MaxPower

# ---------------------------------------------------------------------------
# Combined HID Function (hidg0): Keyboard + Mouse via Report IDs
#
# Report ID 1 — Keyboard (9 bytes total):
#   report_id(1B) | modifier(1B) | reserved(1B) | keycodes(6B)
#
# Report ID 2 — Absolute Mouse (8 bytes total):
#   report_id(1B) | buttons(1B) | X(2B LE) | Y(2B LE) | wheel(1B) | pad(1B)
#
# Combined HID Report Descriptor (138 bytes)
# ---------------------------------------------------------------------------
mkdir -p functions/hid.combo
echo 0 > functions/hid.combo/protocol
echo 0 > functions/hid.combo/subclass
echo 9 > functions/hid.combo/report_length    # max(9, 8) = 9

# Keyboard collection (Report ID 1) + Mouse collection (Report ID 2)
echo -ne '\x05\x01\x09\x06\xa1\x01\x85\x01\x05\x07\x19\xe0\x29\xe7\x15\x00\x25\x01\x75\x01\x95\x08\x81\x02\x95\x01\x75\x08\x81\x01\x95\x05\x75\x01\x05\x08\x19\x01\x29\x05\x91\x02\x95\x01\x75\x03\x91\x01\x95\x06\x75\x08\x15\x00\x26\xff\x00\x05\x07\x19\x00\x29\xff\x81\x00\xc0\x05\x01\x09\x02\xa1\x01\x85\x02\x09\x01\xa1\x00\x05\x09\x19\x01\x29\x05\x15\x00\x25\x01\x75\x01\x95\x05\x81\x02\x95\x01\x75\x03\x81\x01\x05\x01\x09\x30\x09\x31\x16\x00\x00\x26\xff\x7f\x75\x10\x95\x02\x81\x02\x09\x38\x15\x81\x25\x7f\x75\x08\x95\x01\x81\x06\x95\x01\x75\x08\x81\x01\xc0\xc0' > functions/hid.combo/report_desc

ln -s functions/hid.combo configs/c.1/

# ---------------------------------------------------------------------------
# Activate gadget by binding to UDC
# ---------------------------------------------------------------------------
UDC=$(ls /sys/class/udc | head -1)
if [ -z "$UDC" ]; then
    echo "[ERROR] No UDC found. Ensure USB OTG is available."
    exit 1
fi
echo "$UDC" > UDC

# ---------------------------------------------------------------------------
# Make /dev/hidg0 writable by non-root users so the web server (running as
# the invoking user) can drive the gadget without sudo. The kernel creates
# /dev/hidg0 with mode 0600 root:root by default, which gives non-root
# processes EACCES on every write. World-rw is appropriate for a dedicated
# CyberRaccoon Pi — the threat model is the connected target machine, not
# other local users.
# ---------------------------------------------------------------------------
chmod 0666 /dev/hidg0 || echo "[WARN] chmod 0666 /dev/hidg0 failed — non-root processes may get permission denied."

echo "[OK] CyberRaccoon USB Gadget configured successfully."
echo "     Keyboard + Mouse: /dev/hidg0 (Report ID 1=keyboard, 2=mouse)"
echo "     UDC:              $UDC"
