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

# ---------------------------------------------------------------------------
# Pi 5 check: USB-C on Pi 5 is power-only (dwc3 host mode).
# USB Gadget requires a UDC, which only Pi 4B (and earlier) expose via dwc2.
# ---------------------------------------------------------------------------
if [ -f /proc/device-tree/model ] && grep -q "Raspberry Pi 5" /proc/device-tree/model; then
    echo "[ERROR] Raspberry Pi 5 detected."
    echo "        The USB-C port on Pi 5 is power-only and cannot act as a USB device."
    echo "        Use Bluetooth HID instead:  sudo scripts/setup.sh --bt"
    echo "        Then run with:              python -m cyberraccoon --transport bt"
    exit 1
fi

echo "[INFO] Setting up CyberRaccoon USB Gadget..."

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
# Verify UDC is available (Pi 4B exposes this via dwc2 in config.txt)
# ---------------------------------------------------------------------------
if [ -z "$(ls /sys/class/udc 2>/dev/null)" ]; then
    echo "[ERROR] No UDC found. Ensure dwc2 overlay is enabled in /boot/firmware/config.txt:"
    echo "        dtoverlay=dwc2"
    echo "        Then reboot."
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

echo "[OK] CyberRaccoon USB Gadget configured successfully."
echo "     Keyboard + Mouse: /dev/hidg0 (Report ID 1=keyboard, 2=mouse)"
echo "     UDC:              $UDC"
