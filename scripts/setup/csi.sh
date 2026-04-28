#!/bin/bash
# ===========================================================================
# CyberRaccoon CSI HDMI Capture Setup
#
# Configures the TC358743 HDMI-to-CSI bridge on Raspberry Pi 5:
#   1. Installs v4l-utils (v4l2-ctl, media-ctl)
#   2. Adds dtoverlay=tc358743-pi5 to /boot/firmware/config.txt
#   3. Verifies hardware requirements
#
# Requires a reboot after first run to load the device tree overlay.
#
# Hardware:
#   - TC358743 HDMI-to-CSI bridge board
#   - Connected to CAM0 (2-lane, 720p) or CAM1 (4-lane, 1080p)
#   - HDMI cable from target computer to TC358743 input
# ===========================================================================

set -euo pipefail

echo "[INFO] Setting up CyberRaccoon CSI HDMI Capture (TC358743)..."

# ---------------------------------------------------------------------------
# 1. Install v4l-utils
# ---------------------------------------------------------------------------
echo ""
echo "[1/4] Checking v4l-utils..."

if command -v v4l2-ctl &>/dev/null && command -v media-ctl &>/dev/null; then
    echo "  [OK] v4l2-ctl and media-ctl already installed"
else
    echo "  Installing v4l-utils..."
    apt-get update -qq
    apt-get install -y v4l-utils
    echo "  [OK] v4l-utils installed"
fi

# ---------------------------------------------------------------------------
# 2. Configure device tree overlay
# ---------------------------------------------------------------------------
echo ""
echo "[2/4] Configuring device tree overlay..."

CONFIG_FILE="/boot/firmware/config.txt"
if [ ! -f "$CONFIG_FILE" ]; then
    # Older Pi OS uses /boot/config.txt
    CONFIG_FILE="/boot/config.txt"
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "  [ERROR] Cannot find config.txt at /boot/firmware/ or /boot/"
    exit 1
fi

# Check which CAM port to use
OVERLAY_LINE=""
NEEDS_REBOOT=false

if grep -q "^dtoverlay=tc358743" "$CONFIG_FILE"; then
    EXISTING=$(grep "^dtoverlay=tc358743" "$CONFIG_FILE")
    echo "  [OK] TC358743 overlay already configured: $EXISTING"
else
    echo ""
    echo "  Which CSI port is the TC358743 connected to?"
    echo ""
    echo "    [0] CAM0 — 2-lane CSI, max 720p"
    echo "        Fewer lanes limit bandwidth. The driver will force the HDMI"
    echo "        source to 1280x720@60Hz via EDID to stay within limits."
    echo ""
    echo "    [1] CAM1 — 4-lane CSI, max 1080p  (recommended)"
    echo "        Double the bandwidth allows full 1920x1080@60Hz capture."
    echo "        Sharper screenshots give the LLM more detail to work with."
    echo ""
    read -rp "  CAM port [0]: " CAM_CHOICE
    CAM_CHOICE="${CAM_CHOICE:-0}"

    if [ "$CAM_CHOICE" = "1" ]; then
        OVERLAY_LINE="dtoverlay=tc358743-pi5,4lane=1"
        echo "  Adding: $OVERLAY_LINE  (CAM1, 4-lane, 1080p capable)"
    else
        OVERLAY_LINE="dtoverlay=tc358743-pi5"
        echo "  Adding: $OVERLAY_LINE  (CAM0, 2-lane, 720p)"
        echo ""
        echo "  Tip: CAM1 (4-lane) gives 1080p capture for better LLM accuracy."
        echo "       You can switch later by changing the overlay line in $CONFIG_FILE."
    fi

    # Add overlay to config.txt
    echo "" >> "$CONFIG_FILE"
    echo "# CyberRaccoon: TC358743 HDMI-to-CSI bridge" >> "$CONFIG_FILE"
    echo "$OVERLAY_LINE" >> "$CONFIG_FILE"
    echo "  [OK] Overlay added to $CONFIG_FILE"
    NEEDS_REBOOT=true
    # Signal to setup.sh / install.sh that a reboot is required.
    # World-writable so the unprivileged installer can clean it up later.
    touch /tmp/cyberraccoon-needs-reboot 2>/dev/null && chmod 666 /tmp/cyberraccoon-needs-reboot 2>/dev/null || true
fi

# ---------------------------------------------------------------------------
# 3. Grant video device access
# ---------------------------------------------------------------------------
echo ""
echo "[3/4] Checking video device permissions..."

# /dev/video* nodes are owned by group 'video'. Add the invoking user
# (the real user behind sudo) so they can run capture without sudo.
REAL_USER="${SUDO_USER:-$(logname 2>/dev/null || echo '')}"
if [ -n "$REAL_USER" ] && [ "$REAL_USER" != "root" ]; then
    if id -nG "$REAL_USER" | grep -qw video; then
        echo "  [OK] User '$REAL_USER' already in 'video' group"
    else
        usermod -aG video "$REAL_USER"
        echo "  [OK] Added '$REAL_USER' to 'video' group"
        echo "  Note: log out and back in (or run 'newgrp video') for this to take effect"
    fi
else
    echo "  [WARN] Could not determine the real user — skipping group setup."
    echo "         Manually run: sudo usermod -aG video YOUR_USERNAME"
fi

# ---------------------------------------------------------------------------
# 4. Verify
# ---------------------------------------------------------------------------
echo ""
echo "[4/4] Verification..."

PASS=true

# Check v4l-utils
if command -v v4l2-ctl &>/dev/null; then
    echo "  [OK] v4l2-ctl found: $(which v4l2-ctl)"
else
    echo "  [FAIL] v4l2-ctl not found"
    PASS=false
fi

if command -v media-ctl &>/dev/null; then
    echo "  [OK] media-ctl found: $(which media-ctl)"
else
    echo "  [FAIL] media-ctl not found"
    PASS=false
fi

# Check config.txt
if grep -q "^dtoverlay=tc358743" "$CONFIG_FILE"; then
    echo "  [OK] Device tree overlay configured"
else
    echo "  [FAIL] Device tree overlay not in $CONFIG_FILE"
    PASS=false
fi

# Check if TC358743 is already visible (only after reboot with overlay)
# Scan /dev/media0 through /dev/media9 to match runtime discovery behavior
TC358743_FOUND=false
for i in $(seq 0 9); do
    if [ -e "/dev/media${i}" ] && media-ctl -d "/dev/media${i}" -p 2>/dev/null | grep -q "tc358743"; then
        echo "  [OK] TC358743 detected in media topology (/dev/media${i})"
        TC358743_FOUND=true
        break
    fi
done
if ! $TC358743_FOUND; then
    if $NEEDS_REBOOT; then
        echo "  [--] TC358743 not yet visible (reboot required to load overlay)"
    else
        echo "  [WARN] TC358743 not detected. Check:"
        echo "         1) Ribbon cable connection to the correct CAM port"
        echo "         2) Board is powered (check LED)"
        echo "         3) Reboot if overlay was just added"
    fi
fi

echo ""
if $PASS; then
    if $NEEDS_REBOOT; then
        echo "==========================================="
        echo "  Setup complete — REBOOT REQUIRED"
        echo ""
        echo "  The device tree overlay was just added."
        echo "  Run: sudo reboot"
        echo ""
        echo "  After reboot, test with:"
        echo "    python -m cyberraccoon.capture.cli --source csi --output csi.jpg"
        echo "==========================================="
    else
        echo "==========================================="
        echo "  CSI HDMI capture ready!"
        echo ""
        echo "  Test with:"
        echo "    python -m cyberraccoon.capture.cli --source csi --output csi.jpg"
        echo ""
        echo "  Full agent with CSI capture:"
        echo "    python -m cyberraccoon --task 'Open Notepad' --source csi --transport bt"
        echo "==========================================="
    fi
else
    echo "==========================================="
    echo "  Some checks failed. Review errors above."
    echo "==========================================="
    exit 1
fi
