#!/bin/bash
# ===========================================================================
# CyberRaccoon USB HID Gadget Creator (idempotent)
#
# Idempotent USB HID gadget creator. Invoked by `cyberraccoon-usb-gadget.service`
# at boot and by `scripts/setup/gadget.sh` at install time. Writes the
# cyber_raccoon configfs gadget and binds it to the first available UDC.
#
# Does NOT manage the dwc2 overlay or modprobe blacklists — that is
# `scripts/setup/gadget.sh`'s one-time job. This script assumes those
# preconditions are already met (a previous run of setup.sh --gadget +
# the post-overlay reboot).
#
# Why this script exists separately: configfs USB gadgets live in tmpfs
# (/sys/kernel/config) and disappear on every reboot. systemd invokes this
# helper on each boot via cyberraccoon-usb-gadget.service so /dev/hidg0
# reappears without the user re-running setup.sh.
#
# Combined HID interface uses Report IDs:
#   - Report ID 1: Keyboard (9-byte reports: 1 ID + 8 data)
#   - Report ID 2: Absolute-coordinate Mouse (8-byte reports: 1 ID + 7 data)
#
# Device file: /dev/hidg0 (shared by keyboard and mouse).
# ===========================================================================

set -euo pipefail

# Root check — mirrors scripts/setup.sh:check_root
if [ "$(id -u)" -ne 0 ]; then
    echo "[ERROR] Must run as root"
    exit 1
fi

GADGET_DIR="/sys/kernel/config/usb_gadget/cyber_raccoon"

# Idempotent fast path: if /dev/hidg0 already exists, nothing to do.
if [ -e /dev/hidg0 ]; then
    echo "[INFO] /dev/hidg0 already present — skipping gadget creation."
    exit 0
fi

# Load composite module (required before any configfs writes)
modprobe libcomposite

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

# ---------------------------------------------------------------------------
# Verify UDC is available.
#   - Pi 4B: needs dtoverlay=dwc2 in /boot/firmware/config.txt + reboot.
#   - Pi 5: dwc2 loads automatically. Recommended topology is a USB
#     power/data splitter (external power + separate data cable to target)
#     so you can swap targets without power-cycling the Pi. A single USB
#     cable from the Pi USB-C to the target also works (carries power +
#     data) but changing target requires powering off the Pi. If the UDC
#     stays "not attached", a known kernel bug (raspberrypi/linux#6289)
#     may be hitting a single-cable setup — switch to the splitter
#     topology or fall back to Bluetooth HID (--bt).
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
        echo "        3. If you're using a single USB cable (Pi powered by the"
        echo "           target via the same cable), you may be hitting a known"
        echo "           dwc2 kernel bug (raspberrypi/linux#6289). Switch to a"
        echo "           USB power/data splitter (external power to Pi +"
        echo "           separate data cable to target), or fall back to"
        echo "           Bluetooth HID."
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
