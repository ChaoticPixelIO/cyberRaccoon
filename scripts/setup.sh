#!/bin/bash
# ===========================================================================
# CyberRaccoon Pi Hardware Setup
#
# Unified entry point for configuring Raspberry Pi hardware:
#   - Bluetooth HID (keyboard + mouse over Bluetooth)
#   - USB HID Gadget (keyboard + mouse over USB — Pi 4B only)
#   - AirPlay capture (screen mirroring from Mac/iPhone)
#   - CSI HDMI capture (TC358743 HDMI-to-CSI bridge)
#
# Usage:
#   sudo scripts/setup.sh                   # interactive
#   sudo scripts/setup.sh --bt              # Bluetooth HID only
#   sudo scripts/setup.sh --gadget          # USB HID Gadget only
#   sudo scripts/setup.sh --airplay         # AirPlay capture only
#   sudo scripts/setup.sh --csi             # CSI HDMI capture only
#   sudo scripts/setup.sh --bt --airplay    # multiple components
#   sudo scripts/setup.sh --all             # everything applicable
# ===========================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SETUP_DIR="$SCRIPT_DIR/setup"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m' # No Color

info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; }

is_pi5() {
    [ -f /proc/device-tree/model ] && grep -q "Raspberry Pi 5" /proc/device-tree/model
}

check_root() {
    if [ "$(id -u)" -ne 0 ]; then
        error "This script must be run as root: sudo scripts/setup.sh"
        exit 1
    fi
}

run_component() {
    local script="$1"
    local name="$2"

    echo ""
    echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BOLD}  Setting up: $name${NC}"
    echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""

    if [ ! -f "$script" ]; then
        error "Script not found: $script"
        return 1
    fi

    bash "$script"
}

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------

DO_BT=false
DO_GADGET=false
DO_AIRPLAY=false
DO_CSI=false
DO_INTERACTIVE=false

if [ $# -eq 0 ]; then
    DO_INTERACTIVE=true
else
    while [ $# -gt 0 ]; do
        case "$1" in
            --bt|--bluetooth)  DO_BT=true ;;
            --gadget|--usb)    DO_GADGET=true ;;
            --airplay)         DO_AIRPLAY=true ;;
            --csi)             DO_CSI=true ;;
            --all)
                DO_BT=true
                DO_GADGET=true
                DO_AIRPLAY=true
                DO_CSI=true
                ;;
            --help|-h)
                echo "Usage: sudo scripts/setup.sh [OPTIONS]"
                echo ""
                echo "Options:"
                echo "  --bt, --bluetooth   Set up Bluetooth HID"
                echo "  --gadget, --usb     Set up USB HID Gadget (Pi 4B only)"
                echo "  --airplay           Set up AirPlay capture"
                echo "  --csi               Set up CSI HDMI capture (TC358743)"
                echo "  --all               Set up all components"
                echo "  -h, --help          Show this help"
                echo ""
                echo "No arguments: interactive mode (asks what to configure)"
                exit 0
                ;;
            *)
                error "Unknown option: $1 (try --help)"
                exit 1
                ;;
        esac
        shift
    done
fi

# ---------------------------------------------------------------------------
# Root check
# ---------------------------------------------------------------------------

check_root

# ---------------------------------------------------------------------------
# Interactive mode
# ---------------------------------------------------------------------------

if $DO_INTERACTIVE; then
    echo ""
    echo -e "${BOLD}==========================================${NC}"
    echo -e "${BOLD}  CyberRaccoon — Pi Hardware Setup${NC}"
    echo -e "${BOLD}==========================================${NC}"
    echo ""

    # Detect Pi model
    MODEL="Raspberry Pi"
    [ -f /proc/device-tree/model ] && MODEL=$(tr -d '\0' < /proc/device-tree/model)
    info "Detected: $MODEL"
    echo ""
    echo "  CyberRaccoon needs at least ONE control method AND at least ONE capture"
    echo "  method to work. Pick from each group, or just type 'all' for everything."
    echo ""
    echo "    CONTROL  (pick at least one):  B = Bluetooth      G = USB Gadget"
    echo "    CAPTURE  (pick at least one):  A = AirPlay        C = CSI HDMI"
    echo ""
    echo "  Combine letters in any order. Examples:"
    echo "    BC      → Bluetooth + CSI HDMI    (recommended for wired-only setup)"
    echo "    GA      → USB Gadget + AirPlay    (recommended for Mac/iPhone wireless capture)"
    echo "    BGAC    → install everything (same as 'all')"
    echo "    all     → everything applicable   (simplest)"
    if is_pi5; then
        echo ""
        echo "  Note: On Pi 5, single-cable USB-C OTG hits a known dwc2 kernel bug."
        echo "        USB Gadget works with a USB power/data splitter cable."
    fi

    echo ""
    echo "  (type 'q' to quit without configuring anything)"
    echo ""

    # Loop until the user picks a valid combo or explicitly quits / confirms anyway.
    while true; do
        # Reset flags each iteration so re-picking starts clean.
        DO_BT=false
        DO_GADGET=false
        DO_AIRPLAY=false
        DO_CSI=false

        read -rp "  Components: " CHOICES
        CHOICES=$(echo "$CHOICES" | tr '[:lower:]' '[:upper:]')

        if [ "$CHOICES" = "Q" ] || [ "$CHOICES" = "QUIT" ] || [ "$CHOICES" = "EXIT" ]; then
            echo ""
            info "Quit. To set up specific components later, see:"
            info "  sudo $0 --help"
            exit 0
        fi

        if [ "$CHOICES" = "ALL" ]; then
            DO_BT=true
            DO_GADGET=true
            DO_AIRPLAY=true
            DO_CSI=true
        else
            [[ "$CHOICES" == *B* ]] && DO_BT=true
            [[ "$CHOICES" == *G* ]] && DO_GADGET=true
            [[ "$CHOICES" == *A* ]] && DO_AIRPLAY=true
            [[ "$CHOICES" == *C* ]] && DO_CSI=true
        fi

        if ! $DO_BT && ! $DO_GADGET && ! $DO_AIRPLAY && ! $DO_CSI; then
            echo ""
            warn "Nothing selected. Pick at least one letter (B, G, A, C) or 'all'. Type 'q' to quit."
            echo ""
            continue
        fi

        # Soft-validation: warn if user picked only CONTROL or only CAPTURE
        HAS_CONTROL=false
        HAS_CAPTURE=false
        $DO_BT      && HAS_CONTROL=true
        $DO_GADGET  && HAS_CONTROL=true
        $DO_AIRPLAY && HAS_CAPTURE=true
        $DO_CSI     && HAS_CAPTURE=true

        if $HAS_CONTROL && $HAS_CAPTURE; then
            break  # valid combo, proceed
        fi

        echo ""
        if ! $HAS_CONTROL; then
            warn "You picked CAPTURE but no CONTROL — the agent won't be able to send keyboard/mouse to the target."
        fi
        if ! $HAS_CAPTURE; then
            warn "You picked CONTROL but no CAPTURE — the agent won't be able to see the target's screen."
        fi
        echo ""
        echo "    y = yes, continue with this partial setup"
        echo "    r = re-pick components  (default)"
        echo "    q = quit setup"
        read -rp "  Choice [y/R/q]: " CONFIRM
        CONFIRM=$(echo "$CONFIRM" | tr '[:upper:]' '[:lower:]')
        case "$CONFIRM" in
            y|yes)
                break  # user confirmed partial setup
                ;;
            q|quit|exit)
                echo ""
                info "Quit. To set up specific components later, see:"
                info "  sudo $0 --help"
                exit 0
                ;;
            *)
                # blank, r, or anything else → loop back to the picker
                echo ""
                info "Let's pick again."
                echo ""
                ;;
        esac
    done
fi

# ---------------------------------------------------------------------------
# Run selected components
# ---------------------------------------------------------------------------

FAILED=0

if $DO_BT; then
    run_component "$SETUP_DIR/bluetooth.sh" "Bluetooth HID" || FAILED=$((FAILED + 1))
fi

if $DO_GADGET; then
    run_component "$SETUP_DIR/gadget.sh" "USB HID Gadget" || FAILED=$((FAILED + 1))
fi

if $DO_AIRPLAY; then
    run_component "$SETUP_DIR/airplay.sh" "AirPlay Capture" || FAILED=$((FAILED + 1))
fi

if $DO_CSI; then
    run_component "$SETUP_DIR/csi.sh" "CSI HDMI Capture (TC358743)" || FAILED=$((FAILED + 1))
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

echo ""
echo -e "${BOLD}==========================================${NC}"
if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}${BOLD}  Setup complete!${NC}"
else
    echo -e "${YELLOW}${BOLD}  Setup finished with $FAILED error(s)${NC}"
fi
echo -e "${BOLD}==========================================${NC}"
echo ""

if $DO_CSI; then
    warn "CSI HDMI requires a reboot to load the device tree overlay."
    echo "  Run: sudo reboot"
    echo ""
fi

if $DO_GADGET && is_pi5; then
    echo "  Pi 5 USB Gadget cable check:"
    echo "    - Use the Pi USB-C port (the one used for power) as the data link."
    echo "    - Single-cable USB-C OTG (Pi powered by the target) hits a known"
    echo "      dwc2 kernel bug. If you are not already using a USB power/data"
    echo "      splitter cable, set one up: external power to the Pi, data"
    echo "      cable from Pi USB-C to the target."
    echo ""
fi

if ! $DO_BT || ! $DO_GADGET || ! $DO_AIRPLAY || ! $DO_CSI; then
    echo "  Some components weren't configured this run. To set up the rest later:"
    echo "    sudo $0 --help"
    echo ""
fi

echo "  Next steps:"
echo "    Return to the CyberRaccoon Web UI to configure your API key,"
echo "    pick a screen capture source, and run a task."
echo "    (If the Web UI isn't running yet: python -m cyberraccoon --web)"
echo ""
