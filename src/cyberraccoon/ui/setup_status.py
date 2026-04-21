"""Hardware setup status checker for the Web UI.

Detects which hardware components are configured on the Pi without
requiring root privileges. The Web UI polls this to show a live
setup checklist with commands to fix missing components.

All checks are read-only: file existence, process status, device
nodes, and Python import tests. No sudo required.

Usage::

    from cyberraccoon.ui.setup_status import check_setup_status

    status = check_setup_status()
    # Returns dict with per-component status
"""

from __future__ import annotations

import importlib
import importlib.metadata
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any

logger = logging.getLogger("M5.setup_status")


# ---------------------------------------------------------------------------
# Cached OpenCV probe — shared by python_env and airplay checks
# ---------------------------------------------------------------------------

def _probe_opencv() -> dict[str, Any]:
    """Import cv2 once and return version + GStreamer support flag.

    Returns a dict: {"importable": bool, "version": str|None, "has_gstreamer": bool}
    """
    try:
        cv2 = importlib.import_module("cv2")
    except ImportError:
        return {"importable": False, "version": None, "has_gstreamer": False}

    version = getattr(cv2, "__version__", "unknown")
    try:
        info = cv2.getBuildInformation()
    except AttributeError:
        return {"importable": True, "version": version, "has_gstreamer": False}

    gstreamer_line = (
        info.split("GStreamer:")[1].split("\n")[0]
        if "GStreamer:" in info else ""
    )
    return {
        "importable": True,
        "version": version,
        "has_gstreamer": "YES" in gstreamer_line,
    }


# ---------------------------------------------------------------------------
# Component status values
# ---------------------------------------------------------------------------

READY = "ready"
NOT_CONFIGURED = "not_configured"
NOT_AVAILABLE = "not_available"
REBOOT_REQUIRED = "reboot_required"
PARTIAL = "partial"


def _is_raspberry_pi() -> bool:
    """True if running on any Raspberry Pi model."""
    try:
        model = Path("/proc/device-tree/model").read_text().strip("\x00")
        return "Raspberry Pi" in model
    except FileNotFoundError:
        return False


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def _check_bluetooth() -> dict[str, str]:
    """Check Bluetooth HID setup status.

    Checks:
    1. bluetooth.service is active
    2. BlueZ input plugin is disabled (-P input)
    3. Pairing agent service is running
    4. Python has CAP_NET_BIND_SERVICE capability
    """
    issues = []

    # 1. Bluetooth service running
    bt_active = _systemctl_is_active("bluetooth")
    if not bt_active:
        return {
            "status": NOT_CONFIGURED,
            "detail": "Bluetooth service not running",
        }

    # 2. BlueZ input plugin disabled
    input_disabled = False
    for path in [
        "/lib/systemd/system/bluetooth.service",
        "/usr/lib/systemd/system/bluetooth.service",
    ]:
        try:
            content = Path(path).read_text()
            if "-P input" in content:
                input_disabled = True
                break
        except (FileNotFoundError, PermissionError):
            continue

    if not input_disabled:
        issues.append("BlueZ input plugin not disabled")

    # 3. Pairing agent service
    agent_active = _systemctl_is_active("cyberraccoon-pair-agent")
    if not agent_active:
        issues.append("Pairing agent not running")

    # 4. Python capabilities (best-effort check)
    python_bin = shutil.which("python3")
    if python_bin:
        try:
            real_bin = Path(python_bin).resolve()
            result = subprocess.run(
                ["getcap", str(real_bin)],
                capture_output=True, text=True, timeout=5,
            )
            if "cap_net_bind_service" not in result.stdout:
                issues.append("Python missing network capabilities (setcap)")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass  # getcap not available, skip check

    if issues:
        return {
            "status": PARTIAL,
            "detail": "; ".join(issues),
        }

    return {"status": READY, "detail": "Bluetooth HID configured"}


def _check_usb_gadget() -> dict[str, str]:
    """Check USB HID Gadget setup status.

    On Pi 5, USB Gadget is not available (USB-C is power-only without
    a splitter cable, and dwc2 has known issues).
    On Pi 4B, checks /dev/hidg0 and libcomposite.
    """
    if not _is_raspberry_pi():
        return {"status": NOT_AVAILABLE, "detail": "Not running on a Raspberry Pi"}

    # Detect Pi model
    is_pi5 = False
    try:
        model = Path("/proc/device-tree/model").read_text().strip("\x00")
        is_pi5 = "Raspberry Pi 5" in model
    except FileNotFoundError:
        pass

    if is_pi5:
        # Check if gadget is somehow working (user has splitter)
        if Path("/dev/hidg0").exists():
            return {
                "status": READY,
                "detail": "/dev/hidg0 available (USB splitter detected)",
            }
        return {
            "status": NOT_AVAILABLE,
            "detail": (
                "Pi 5 USB-C is power-only. "
                "Use Bluetooth HID (--transport bt) or a USB power/data splitter cable."
            ),
        }

    # Pi 4B or other: check if gadget is configured
    if Path("/dev/hidg0").exists():
        return {"status": READY, "detail": "/dev/hidg0 available"}

    # Check if dwc2 overlay is in config.txt
    dwc2_configured = False
    for config_path in ["/boot/firmware/config.txt", "/boot/config.txt"]:
        try:
            content = Path(config_path).read_text()
            if "dtoverlay=dwc2" in content:
                dwc2_configured = True
                break
        except FileNotFoundError:
            continue

    if not dwc2_configured:
        return {
            "status": NOT_CONFIGURED,
            "detail": "dwc2 overlay not in config.txt and /dev/hidg0 missing",
        }

    return {
        "status": NOT_CONFIGURED,
        "detail": "/dev/hidg0 not present (run setup or reboot after config change)",
    }


def _check_csi() -> dict[str, str]:
    """Check CSI HDMI capture (TC358743) status.

    Checks:
    1. dtoverlay=tc358743 in config.txt
    2. TC358743 visible in media topology (requires reboot after overlay add)
    """
    if not _is_raspberry_pi():
        return {"status": NOT_AVAILABLE, "detail": "Not running on a Raspberry Pi"}

    # 1. Check config.txt for overlay
    overlay_configured = False
    overlay_line = ""
    for config_path in ["/boot/firmware/config.txt", "/boot/config.txt"]:
        try:
            for line in Path(config_path).read_text().splitlines():
                if line.strip().startswith("dtoverlay=tc358743"):
                    overlay_configured = True
                    overlay_line = line.strip()
                    break
            if overlay_configured:
                break
        except FileNotFoundError:
            continue

    if not overlay_configured:
        return {"status": NOT_CONFIGURED, "detail": "TC358743 overlay not in config.txt"}

    # 2. Check if TC358743 is visible in media topology
    tc_visible = False
    if shutil.which("media-ctl"):
        for i in range(10):
            dev = f"/dev/media{i}"
            if not Path(dev).exists():
                continue
            try:
                result = subprocess.run(
                    ["media-ctl", "-d", dev, "-p"],
                    capture_output=True, text=True, timeout=5,
                )
                if "tc358743" in result.stdout.lower():
                    tc_visible = True
                    break
            except (subprocess.TimeoutExpired, PermissionError):
                continue

    if not tc_visible:
        return {
            "status": REBOOT_REQUIRED,
            "detail": f"Overlay configured ({overlay_line}) but TC358743 not visible. Reboot required.",
        }

    return {"status": READY, "detail": f"TC358743 detected ({overlay_line})"}


def _check_airplay(opencv: dict[str, Any]) -> dict[str, str]:
    """Check AirPlay capture dependencies.

    Checks:
    1. uxplay installed
    2. GStreamer H.264 decoder available
    3. Avahi daemon running
    4. OpenCV GStreamer support (reused from _probe_opencv)
    """
    issues = []

    # 1. uxplay
    if not shutil.which("uxplay"):
        issues.append("uxplay not installed")

    # 2. GStreamer H.264
    if shutil.which("gst-inspect-1.0"):
        try:
            result = subprocess.run(
                ["gst-inspect-1.0", "avdec_h264"],
                capture_output=True, timeout=5,
            )
            if result.returncode != 0:
                issues.append("GStreamer H.264 decoder (avdec_h264) not found")
        except subprocess.TimeoutExpired:
            issues.append("GStreamer check timed out")
    else:
        issues.append("GStreamer tools not installed")

    # 3. Avahi
    if not _systemctl_is_active("avahi-daemon"):
        issues.append("Avahi daemon not running")

    # 4. OpenCV GStreamer (reuse probe result)
    if not opencv["importable"]:
        issues.append("OpenCV not importable")
    elif not opencv["has_gstreamer"]:
        issues.append("OpenCV missing GStreamer support")

    if issues:
        if len(issues) >= 3:
            return {"status": NOT_CONFIGURED, "detail": "; ".join(issues)}
        return {"status": PARTIAL, "detail": "; ".join(issues)}

    return {"status": READY, "detail": "uxplay, GStreamer, avahi all OK"}


def _check_python_env(opencv: dict[str, Any]) -> dict[str, str]:
    """Check Python environment health.

    Checks:
    1. cyberraccoon package importable
    2. OpenCV importable (reused from _probe_opencv)
    3. dbus importable (for Bluetooth)
    4. gi importable (for Bluetooth pairing)
    5. No shadowing opencv-python pip package
    """
    issues = []

    # 1. Package
    try:
        importlib.import_module("cyberraccoon")
    except ImportError:
        issues.append("cyberraccoon package not installed")

    # 2. OpenCV (reuse probe)
    if not opencv["importable"]:
        issues.append("OpenCV not importable")

    # 3. dbus
    try:
        importlib.import_module("dbus")
    except ImportError:
        issues.append("python3-dbus not importable")

    # 4. gi
    try:
        importlib.import_module("gi")
    except ImportError:
        issues.append("python3-gi not importable")

    # 5. Shadowing check — is pip-installed opencv-python present in this env?
    for pkg in ("opencv-python", "opencv-python-headless"):
        try:
            importlib.metadata.distribution(pkg)
            issues.append(f"{pkg} pip-installed (shadows system package)")
        except importlib.metadata.PackageNotFoundError:
            pass

    if issues:
        return {"status": PARTIAL, "detail": "; ".join(issues)}

    detail = f"cv2 {opencv['version']}"
    if opencv["has_gstreamer"]:
        detail += " (GStreamer: YES)"
    detail += ", dbus OK, gi OK"
    return {"status": READY, "detail": detail}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _systemctl_is_active(service: str) -> bool:
    """Check if a systemd service is active (running)."""
    try:
        result = subprocess.run(
            ["systemctl", "is-active", "--quiet", service],
            capture_output=True, timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check_setup_status() -> dict[str, Any]:
    """Check the status of all hardware/software components.

    Returns a dict with per-component status and a top-level setup command.
    All checks are non-root — safe to call from the web server.
    """
    opencv = _probe_opencv()

    components = {
        "python_env": _check_python_env(opencv),
        "bluetooth": _check_bluetooth(),
        "usb_gadget": _check_usb_gadget(),
        "csi_hdmi": _check_csi(),
        "airplay": _check_airplay(opencv),
    }

    # Count how many hardware components need setup
    hw_keys = ["bluetooth", "usb_gadget", "csi_hdmi", "airplay"]
    needs_setup = [
        k for k in hw_keys
        if components[k]["status"] in (NOT_CONFIGURED, PARTIAL)
    ]
    needs_reboot = [
        k for k in hw_keys
        if components[k]["status"] == REBOOT_REQUIRED
    ]

    # Build setup commands
    setup_commands = []
    if needs_setup:
        setup_commands.append("sudo scripts/setup.sh --all")
    if needs_reboot:
        setup_commands.append("sudo reboot")

    return {
        "components": components,
        "needs_setup": needs_setup,
        "needs_reboot": needs_reboot,
        "setup_commands": setup_commands,
    }
