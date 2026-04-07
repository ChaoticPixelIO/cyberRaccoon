#!/bin/bash
# ===========================================================================
# CyberRaccoon AirPlay Capture Setup Script
#
# Installs uxplay + GStreamer dependencies for AirPlay screen mirroring
# on Raspberry Pi OS (Debian/Ubuntu).
#
# After running this script, the Mac user can AirPlay mirror to the Pi.
# The AirPlayCapture class receives decoded frames via RTP + GStreamer.
#
# Run once with: sudo scripts/setup_airplay.sh
# ===========================================================================

set -euo pipefail

echo "==========================================="
echo "  CyberRaccoon AirPlay Capture Setup"
echo "==========================================="

# ---------------------------------------------------------------------------
# 1. Check we're running on a Debian-based system
# ---------------------------------------------------------------------------
if ! command -v apt-get &> /dev/null; then
    echo "[ERROR] apt-get not found. This script requires Debian/Ubuntu."
    exit 1
fi

# ---------------------------------------------------------------------------
# 2. Install uxplay (AirPlay receiver)
# ---------------------------------------------------------------------------
echo ""
echo "[1/4] Installing uxplay..."
if command -v uxplay &> /dev/null; then
    UXPLAY_VERSION=$(uxplay -h 2>&1 | head -1 || echo "unknown")
    echo "  uxplay already installed: $UXPLAY_VERSION"
else
    apt-get update -qq
    apt-get install -y uxplay
    echo "  uxplay installed successfully"
fi

# ---------------------------------------------------------------------------
# 3. Install GStreamer plugins
# ---------------------------------------------------------------------------
echo ""
echo "[2/4] Installing GStreamer plugins..."
apt-get install -y \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-libav \
    gstreamer1.0-tools

echo "  GStreamer plugins installed"

# ---------------------------------------------------------------------------
# 4. Install system OpenCV with GStreamer support
# ---------------------------------------------------------------------------
echo ""
echo "[3/4] Checking OpenCV + GStreamer..."
if python3 -c "import cv2; info = cv2.getBuildInformation(); assert 'GStreamer:                   YES' in info" 2>/dev/null; then
    echo "  OpenCV with GStreamer support: OK"
else
    echo "  Installing system python3-opencv (includes GStreamer support)..."
    apt-get install -y python3-opencv
    echo "  python3-opencv installed"
    echo ""
    echo "  NOTE: If using a venv, create it with --system-site-packages:"
    echo "    python3 -m venv --system-site-packages venv"
    echo "  pip-installed opencv-python does NOT include GStreamer."
fi

# ---------------------------------------------------------------------------
# 5. Install Avahi (mDNS) for AirPlay discovery
# ---------------------------------------------------------------------------
echo ""
echo "[4/4] Checking Avahi (mDNS) service..."
if systemctl is-active --quiet avahi-daemon; then
    echo "  Avahi daemon: running"
else
    apt-get install -y avahi-daemon
    systemctl enable avahi-daemon
    systemctl start avahi-daemon
    echo "  Avahi daemon installed and started"
fi

# ---------------------------------------------------------------------------
# 6. Verify installation
# ---------------------------------------------------------------------------
echo ""
echo "==========================================="
echo "  Verification"
echo "==========================================="

PASS=true

# Check uxplay
if command -v uxplay &> /dev/null; then
    echo "  [OK] uxplay found: $(which uxplay)"
else
    echo "  [FAIL] uxplay not found"
    PASS=false
fi

# Check GStreamer
if gst-inspect-1.0 avdec_h264 &> /dev/null; then
    echo "  [OK] GStreamer H.264 decoder (avdec_h264)"
else
    echo "  [FAIL] GStreamer H.264 decoder not found"
    PASS=false
fi

if gst-inspect-1.0 rtph264depay &> /dev/null; then
    echo "  [OK] GStreamer RTP H.264 depayloader"
else
    echo "  [FAIL] GStreamer RTP depayloader not found"
    PASS=false
fi

# Check OpenCV GStreamer
if python3 -c "import cv2; assert 'GStreamer:                   YES' in cv2.getBuildInformation()" 2>/dev/null; then
    echo "  [OK] OpenCV GStreamer support"
else
    echo "  [FAIL] OpenCV lacks GStreamer support"
    PASS=false
fi

# Check Avahi
if systemctl is-active --quiet avahi-daemon; then
    echo "  [OK] Avahi daemon running"
else
    echo "  [FAIL] Avahi daemon not running"
    PASS=false
fi

echo ""
if $PASS; then
    echo "==========================================="
    echo "  All checks passed!"
    echo ""
    echo "  Usage:"
    echo "    # Test AirPlay capture:"
    echo "    python -m capture.cli --source airplay --output airplay.jpg"
    echo ""
    echo "    # On Mac: System Settings > General > AirDrop & Handoff"
    echo "    # Or: Control Center > Screen Mirroring > CyberRaccoon"
    echo ""
    echo "    # Full agent with AirPlay:"
    echo "    python -m cyberraccoon --task 'Open Notepad' --source airplay"
    echo "==========================================="
else
    echo "==========================================="
    echo "  Some checks failed. Review errors above."
    echo "==========================================="
    exit 1
fi
