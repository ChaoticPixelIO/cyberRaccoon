#!/bin/bash
# ===========================================================================
# CyberRaccoon USB Gadget Setup Script
#
# Creates a composite USB HID device with:
#   - hidg0: Boot Keyboard (8-byte reports)
#   - hidg1: Absolute-coordinate Mouse (7-byte reports)
#
# Run once at boot with: sudo scripts/setup_gadget.sh
# Idempotent: skips if gadget already configured.
# ===========================================================================

set -euo pipefail

GADGET_DIR="/sys/kernel/config/usb_gadget/cyber_raccoon"

# Check if already configured
if [ -d "$GADGET_DIR" ]; then
    echo "[INFO] USB Gadget already configured at $GADGET_DIR — skipping."
    exit 0
fi

echo "[INFO] Setting up CyberRaccoon USB Gadget..."

# Load composite module
modprobe libcomposite

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
# HID Function 1: Keyboard (hidg0)
# Boot Keyboard protocol, 8-byte reports
# ---------------------------------------------------------------------------
mkdir -p functions/hid.keyboard
echo 1 > functions/hid.keyboard/protocol        # Keyboard
echo 1 > functions/hid.keyboard/subclass        # Boot Interface
echo 8 > functions/hid.keyboard/report_length

# Standard Boot Keyboard HID Report Descriptor (63 bytes)
echo -ne '\x05\x01\x09\x06\xa1\x01\x05\x07\x19\xe0\x29\xe7\x15\x00\x25\x01\x75\x01\x95\x08\x81\x02\x95\x01\x75\x08\x81\x01\x95\x05\x75\x01\x05\x08\x19\x01\x29\x05\x91\x02\x95\x01\x75\x03\x91\x01\x95\x06\x75\x08\x15\x00\x26\xff\x00\x05\x07\x19\x00\x29\xff\x81\x00\xc0' > functions/hid.keyboard/report_desc

ln -s functions/hid.keyboard configs/c.1/

# ---------------------------------------------------------------------------
# HID Function 2: Absolute Mouse (hidg1)
# Absolute coordinate protocol, 7-byte reports
# ---------------------------------------------------------------------------
mkdir -p functions/hid.mouse
echo 0 > functions/hid.mouse/protocol           # None
echo 0 > functions/hid.mouse/subclass           # None
echo 7 > functions/hid.mouse/report_length

# Absolute-coordinate Mouse HID Report Descriptor (67 bytes)
echo -ne '\x05\x01\x09\x02\xa1\x01\x09\x01\xa1\x00\x05\x09\x19\x01\x29\x05\x15\x00\x25\x01\x75\x01\x95\x05\x81\x02\x95\x01\x75\x03\x81\x01\x05\x01\x09\x30\x09\x31\x16\x00\x00\x26\xff\x7f\x75\x10\x95\x02\x81\x02\x09\x38\x15\x81\x25\x7f\x75\x08\x95\x01\x81\x06\x95\x01\x75\x08\x81\x01\xc0\xc0' > functions/hid.mouse/report_desc

ln -s functions/hid.mouse configs/c.1/

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
echo "     Keyboard: /dev/hidg0"
echo "     Mouse:    /dev/hidg1"
echo "     UDC:      $UDC"
