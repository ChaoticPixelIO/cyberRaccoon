#!/bin/bash
# ===========================================================================
# CyberRaccoon Installer
#
# One-command setup for Raspberry Pi. Installs system dependencies, clones
# the repo (if needed), creates a Python venv, installs the package, and
# sets up a systemd service for the Web UI.
#
# Usage:
#   curl -sSL https://raw.githubusercontent.com/ChaoticPixelIO/cyberRaccoon/main/install.sh -o install.sh
#   bash install.sh
#
# Or if you've already cloned the repo:
#   ./install.sh
#
# Do NOT run as root (sudo). The script will invoke sudo for the few
# operations that need it. The repo and venv must be owned by your user.
#
# After install, open http://<pi-hostname>:8000 in your browser.
# The Web UI will show you what hardware setup commands to run.
# ===========================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_URL="https://github.com/ChaoticPixelIO/cyberRaccoon.git"
DEFAULT_INSTALL_DIR="$HOME/cyberRaccoon"
SERVICE_NAME="cyberraccoon"
WEB_PORT=8000

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; }
step()  { echo -e "\n${BOLD}[$1/$TOTAL_STEPS]${NC} $2"; }

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------

TOTAL_STEPS=7

echo ""
echo -e "${BOLD}==========================================${NC}"
echo -e "${BOLD}  CyberRaccoon Installer${NC}"
echo -e "${BOLD}==========================================${NC}"
echo ""

# Must NOT be root — the repo and venv must be owned by the real user,
# otherwise the systemd service (running as that user) cannot read them.
if [ "$(id -u)" -eq 0 ]; then
    error "Do NOT run this installer as root (or via sudo)."
    error ""
    error "Run it as your normal user. The script will ask for sudo"
    error "only when needed (apt install, systemd setup)."
    error ""
    error "Exit, then re-run:  ./install.sh"
    exit 1
fi

# Must be Linux (Raspberry Pi OS / Debian)
if [[ "$(uname -s)" != "Linux" ]]; then
    error "This installer is for Raspberry Pi / Debian Linux."
    error "On macOS, use: git clone ... && pip install -e \".[dev]\""
    exit 1
fi

# Must have apt
if ! command -v apt-get &>/dev/null; then
    error "apt-get not found. This installer requires Debian/Raspberry Pi OS."
    exit 1
fi

# Warn if not Pi (but don't block — might be a Debian dev box)
if [ -f /proc/device-tree/model ]; then
    PI_MODEL=$(tr -d '\0' < /proc/device-tree/model)
    info "Detected: $PI_MODEL"
else
    warn "Not running on a Raspberry Pi. Proceeding anyway."
fi

# Cache sudo credentials upfront so prompts don't interrupt later steps.
# This requires a TTY — fails fast in pipe-to-bash invocations.
info "Requesting sudo access for apt install and systemd setup..."
if ! sudo -v; then
    error "Could not obtain sudo credentials."
    error ""
    error "If you ran via 'curl | bash', it can't prompt for a password."
    error "Download first, then run:"
    error "  curl -sSL https://raw.githubusercontent.com/ChaoticPixelIO/cyberRaccoon/main/install.sh -o install.sh"
    error "  bash install.sh"
    exit 1
fi

# Keep sudo credentials alive in the background (re-calls sudo -v every 50s)
(while true; do sudo -n true 2>/dev/null; sleep 50; done) &
SUDO_KEEPALIVE_PID=$!
trap "kill $SUDO_KEEPALIVE_PID 2>/dev/null || true" EXIT

# ---------------------------------------------------------------------------
# Step 1: Determine install directory
# ---------------------------------------------------------------------------

step 1 "Locating project directory..."

# Detect if we're already inside the repo
if [ -f "pyproject.toml" ] && grep -q "cyberraccoon" pyproject.toml 2>/dev/null; then
    INSTALL_DIR="$(pwd)"
    info "Already inside CyberRaccoon repo: $INSTALL_DIR"
    NEED_CLONE=false
elif [ -f "$DEFAULT_INSTALL_DIR/pyproject.toml" ] && grep -q "cyberraccoon" "$DEFAULT_INSTALL_DIR/pyproject.toml" 2>/dev/null; then
    INSTALL_DIR="$DEFAULT_INSTALL_DIR"
    info "Found existing install: $INSTALL_DIR"
    NEED_CLONE=false
else
    INSTALL_DIR="$DEFAULT_INSTALL_DIR"
    NEED_CLONE=true
fi

# ---------------------------------------------------------------------------
# Step 2: Install system packages
# ---------------------------------------------------------------------------

step 2 "Installing system packages..."

# Collect what's missing
PACKAGES=""

# Core Python tools
dpkg -s python3-venv &>/dev/null   || PACKAGES="$PACKAGES python3-venv"
dpkg -s python3-pip &>/dev/null    || PACKAGES="$PACKAGES python3-pip"
dpkg -s python3-dev &>/dev/null    || PACKAGES="$PACKAGES python3-dev"

# OpenCV with GStreamer (required for capture)
dpkg -s python3-opencv &>/dev/null || PACKAGES="$PACKAGES python3-opencv"

# Bluetooth HID dependencies
dpkg -s python3-dbus &>/dev/null   || PACKAGES="$PACKAGES python3-dbus"
dpkg -s python3-gi &>/dev/null     || PACKAGES="$PACKAGES python3-gi"

# Git (for cloning)
dpkg -s git &>/dev/null            || PACKAGES="$PACKAGES git"

if [ -n "$PACKAGES" ]; then
    info "Installing:$PACKAGES"
    sudo apt-get update -qq
    sudo apt-get install -y $PACKAGES
    info "System packages installed."
else
    info "All system packages already present."
fi

# ---------------------------------------------------------------------------
# Step 3: Clone repository
# ---------------------------------------------------------------------------

step 3 "Setting up repository..."

if $NEED_CLONE; then
    if [ -d "$INSTALL_DIR" ]; then
        error "$INSTALL_DIR already exists but doesn't look like CyberRaccoon."
        error "Remove it or choose a different location, then re-run."
        exit 1
    fi
    info "Cloning from $REPO_URL into $INSTALL_DIR"
    git clone "$REPO_URL" "$INSTALL_DIR"
    cd "$INSTALL_DIR"
else
    cd "$INSTALL_DIR"
    info "Using existing repo at $INSTALL_DIR"
    if [ -d .git ]; then
        info "Pulling latest changes..."
        git pull --ff-only origin main 2>/dev/null || warn "Could not fast-forward (local changes?). Continuing with current checkout."
    fi
fi

# ---------------------------------------------------------------------------
# Step 4: Create Python virtual environment
# ---------------------------------------------------------------------------

step 4 "Setting up Python environment..."

VENV_DIR="$INSTALL_DIR/venv"

if [ -d "$VENV_DIR" ] && [ -f "$VENV_DIR/bin/python3" ]; then
    info "Virtual environment already exists: $VENV_DIR"
else
    info "Creating venv with --system-site-packages (required for OpenCV + GStreamer)..."
    python3 -m venv --system-site-packages "$VENV_DIR"
    info "Virtual environment created."
fi

# Activate for remaining commands
VENV_PYTHON="$VENV_DIR/bin/python3"
VENV_PIP="$VENV_DIR/bin/pip"

# ---------------------------------------------------------------------------
# Step 5: Install CyberRaccoon package
# ---------------------------------------------------------------------------

step 5 "Installing CyberRaccoon..."

# Check for accidentally installed opencv-python (shadows system package)
if "$VENV_PIP" show opencv-python &>/dev/null 2>&1 || "$VENV_PIP" show opencv-python-headless &>/dev/null 2>&1; then
    warn "Removing pip-installed opencv-python (shadows system package, breaks GStreamer)..."
    "$VENV_PIP" uninstall -y opencv-python opencv-python-headless 2>/dev/null || true
fi

"$VENV_PIP" install --quiet -e .
info "CyberRaccoon installed."

# ---------------------------------------------------------------------------
# Step 6: Verify critical imports
# ---------------------------------------------------------------------------

step 6 "Verifying environment..."

VERIFY_OK=true

# Check OpenCV with GStreamer
if "$VENV_PYTHON" -c "
import cv2
info = cv2.getBuildInformation()
assert 'GStreamer:' in info and 'YES' in info.split('GStreamer:')[1].split('\n')[0]
" 2>/dev/null; then
    info "OpenCV with GStreamer: OK"
else
    # OpenCV might work without GStreamer (AirPlay won't, but others will)
    if "$VENV_PYTHON" -c "import cv2" 2>/dev/null; then
        warn "OpenCV available but WITHOUT GStreamer (AirPlay capture won't work)"
    else
        warn "OpenCV not importable — capture backends will fail"
        VERIFY_OK=false
    fi
fi

# Check dbus (needed for Bluetooth)
if "$VENV_PYTHON" -c "import dbus" 2>/dev/null; then
    info "python3-dbus: OK"
else
    warn "python3-dbus not importable (Bluetooth HID won't work)"
    warn "Fix: sudo apt install python3-dbus"
fi

# Check gi (needed for Bluetooth pairing agent)
if "$VENV_PYTHON" -c "from gi.repository import GLib" 2>/dev/null; then
    info "python3-gi (GLib): OK"
else
    warn "python3-gi not importable (Bluetooth pairing won't work)"
    warn "Fix: sudo apt install python3-gi"
fi

# Check the package itself imports
if "$VENV_PYTHON" -c "import cyberraccoon" 2>/dev/null; then
    info "cyberraccoon package: OK"
else
    error "Failed to import cyberraccoon package!"
    VERIFY_OK=false
fi

if ! $VERIFY_OK; then
    error "Some critical checks failed. Review warnings above."
    error "The Web UI may still start, but some features won't work."
fi

# ---------------------------------------------------------------------------
# Step 7: Install and start systemd service
# ---------------------------------------------------------------------------

step 7 "Setting up systemd service..."

SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
RUN_USER="$(whoami)"
RUN_HOME="$HOME"

cat << SERVICEEOF | sudo tee "$SERVICE_FILE" > /dev/null
[Unit]
Description=CyberRaccoon AI Computer Control — Web UI
After=network-online.target bluetooth.service
Wants=network-online.target

[Service]
Type=simple
User=$RUN_USER
WorkingDirectory=$INSTALL_DIR
ExecStart=$VENV_PYTHON -m cyberraccoon --web --host 0.0.0.0 --port $WEB_PORT
Restart=on-failure
RestartSec=5
Environment=HOME=$RUN_HOME
EnvironmentFile=-$RUN_HOME/.apikeys

[Install]
WantedBy=multi-user.target
SERVICEEOF

sudo systemctl daemon-reload
sudo systemctl enable "$SERVICE_NAME"
sudo systemctl restart "$SERVICE_NAME"

# Wait a moment for the service to start
sleep 2

if sudo systemctl is-active --quiet "$SERVICE_NAME"; then
    info "CyberRaccoon service started."
else
    warn "Service failed to start. Check: sudo journalctl -u $SERVICE_NAME -n 20"
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------

# Get the Pi's hostname/IP for the URL
PI_HOSTNAME=$(hostname -I 2>/dev/null | awk '{print $1}')
if [ -z "$PI_HOSTNAME" ]; then
    PI_HOSTNAME=$(hostname)
fi

echo ""
echo -e "${BOLD}==========================================${NC}"
echo -e "${GREEN}${BOLD}  CyberRaccoon installed successfully!${NC}"
echo -e "${BOLD}==========================================${NC}"
echo ""
echo -e "  ${BOLD}Web UI:${NC} http://${PI_HOSTNAME}:${WEB_PORT}"
echo ""
echo "  Next steps:"
echo "    1. Open the Web UI in your browser"
echo "    2. Go to the Status tab — it shows what hardware setup is needed"
echo "    3. Run the setup commands shown (e.g. sudo scripts/setup.sh --all)"
echo "    4. Set your API key in the Config tab"
echo "    5. Submit a task!"
echo ""
echo "  Useful commands:"
echo "    sudo systemctl status cyberraccoon    # check service"
echo "    sudo journalctl -u cyberraccoon -f    # view logs"
echo "    sudo systemctl restart cyberraccoon   # restart"
echo ""
echo -e "${BOLD}==========================================${NC}"
