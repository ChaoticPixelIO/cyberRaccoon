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

# Name of the systemd unit that recreates /dev/hidg0 on every boot
# (installed by scripts/setup/gadget.sh; runs scripts/lib/cyberraccoon-usb-gadget-create.sh).
_USB_GADGET_UNIT = "cyberraccoon-usb-gadget"


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

def _bluez_input_plugin_disabled() -> bool:
    """Return True if bluetoothd is configured to disable the input plugin.

    Two install patterns are accepted:

    1. **Drop-in (current setup)** — any file under
       ``/etc/systemd/system/bluetooth.service.d/`` containing an
       ``ExecStart=`` line with ``-P`` followed by a plugin list that
       includes ``input``. The current setup script also disables
       ``a2dp,avrcp,sap,network,health,gap`` to keep the SDP record
       HID-only, so the substring check ``"-P "`` + ``"input"`` is
       sufficient.

    2. **Legacy in-place edit** — older versions of the setup script
       appended ``-P input`` directly to the package unit file at
       ``/lib/systemd/system/bluetooth.service``. The current setup
       script reverts that edit, but pre-existing installs may still
       have it.

    Either pattern is functionally equivalent: systemd merges drop-ins
    over the package unit and bluetoothd ends up with the right ``-P``
    flag. We just need to confirm one of them is in place.
    """
    dropin_dir = Path("/etc/systemd/system/bluetooth.service.d")
    if dropin_dir.is_dir():
        for conf in dropin_dir.glob("*.conf"):
            try:
                for line in conf.read_text().splitlines():
                    stripped = line.lstrip()
                    if not stripped.startswith("ExecStart="):
                        continue
                    # Look for `-P <list>` where <list> contains `input`
                    parts = stripped.split()
                    for i, part in enumerate(parts):
                        if part == "-P" and i + 1 < len(parts):
                            if "input" in parts[i + 1].split(","):
                                return True
                        elif part.startswith("-P") and "input" in part[2:].split(","):
                            return True
            except (FileNotFoundError, PermissionError):
                continue

    for path in [
        "/lib/systemd/system/bluetooth.service",
        "/usr/lib/systemd/system/bluetooth.service",
    ]:
        try:
            if "-P input" in Path(path).read_text():
                return True
        except (FileNotFoundError, PermissionError):
            continue

    return False


def _check_bluetooth() -> dict[str, str]:
    """Check Bluetooth HID setup status.

    Checks:
    1. bluetooth.service is active
    2. BlueZ input plugin is disabled (via drop-in or legacy in-place edit)
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

    # 2. BlueZ input plugin disabled. Modern setup writes a drop-in at
    #    /etc/systemd/system/bluetooth.service.d/*.conf overriding ExecStart
    #    with `-P input,...` (and other plugins). Legacy installs added
    #    `-P input` in-place to the package unit file. Accept either.
    if not _bluez_input_plugin_disabled():
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


def _read_dwc2_overlay(config_paths: list[str], model_filter: str) -> dict[str, Any]:
    """Walk config.txt and return the active dwc2 overlay setting for this model.

    Section headers like ``[cm5]`` or ``[pi4]`` scope subsequent lines. Only
    sections that apply to ``model_filter`` (plus the implicit default scope
    and ``[all]``) are considered. Lines under non-matching sections (e.g.
    ``[cm5]`` on a Pi 5 Model B) are ignored.

    Returns:
        ``{"present": bool, "dr_mode": str | None, "config_path": str | None}``
        where ``dr_mode`` is the lowercase value (``"peripheral"``,
        ``"otg"``, ``"host"``, …) or ``None`` if the overlay is loaded
        without an explicit ``dr_mode=`` (which the firmware defaults to OTG).
        ``config_path`` is the file we actually parsed.
    """
    applied = {"", "all", model_filter}
    last: dict[str, Any] = {"present": False, "dr_mode": None, "config_path": None}
    for path in config_paths:
        try:
            content = Path(path).read_text()
        except FileNotFoundError:
            continue
        last["config_path"] = path
        section = ""
        for raw in content.splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("[") and line.endswith("]"):
                section = line[1:-1].lower()
                continue
            if section not in applied:
                continue
            if not line.startswith("dtoverlay=dwc2"):
                continue
            # Parse dr_mode= parameter if present
            dr_mode = None
            for part in line.split(",")[1:]:
                kv = part.strip()
                if kv.startswith("dr_mode="):
                    dr_mode = kv.split("=", 1)[1].strip().lower()
                    break
            # Later applied lines override earlier ones, just like the firmware
            last["present"] = True
            last["dr_mode"] = dr_mode
        return last  # only inspect the first config file that exists
    return last


def _dwc2_module_loaded() -> bool:
    """True if dwc2 is providing a USB Device Controller in the running kernel.

    Two signals, either is sufficient:

    1. ``/sys/module/dwc2`` exists — the conventional signal for "module
       loaded". Works when dwc2 is loaded as a module (the typical Pi
       4B / pre-Pi-5 case).

    2. ``/sys/class/udc/`` has at least one entry — the dwc2 driver has
       registered a USB Device Controller. This is the more reliable
       signal on Pi 5, where the dtoverlay-driven dwc2 path can leave
       ``/sys/module/dwc2`` absent even when the gadget stack is fully
       functional and ``/dev/hidg0`` is present.
    """
    if Path("/sys/module/dwc2").exists():
        return True
    udc_dir = Path("/sys/class/udc")
    if udc_dir.exists():
        try:
            if any(udc_dir.iterdir()):
                return True
        except OSError:
            pass
    return False


def _gadget_runtime_status(pi5_cable_note: str | None) -> dict[str, str]:
    """Decide the gadget status from runtime state.

    Called once the dwc2 overlay is present + sane. Inspects:
      - /dev/hidg0 (the device node we ultimately need)
      - /sys/class/udc (does the kernel have a USB Device Controller?)
      - cyberraccoon-usb-gadget.service install state (so we can tell
        "old install / pre-persistence" from "unit installed but disabled"
        from "unit enabled but the boot-time helper had no UDC").

    Pulled out so Pi 5 and Pi 4B paths share the same systemd-unit logic.
    ``pi5_cable_note`` is attached as a top-level "note" on Pi 5 only.
    """
    note_kv = {"note": pi5_cable_note} if pi5_cable_note else {}

    # Branch A: /dev/hidg0 exists → READY (regardless of unit state).
    if Path("/dev/hidg0").exists():
        return {"status": READY, "detail": "/dev/hidg0 available", **note_kv}

    udc_present = (
        any(Path("/sys/class/udc").iterdir())
        if Path("/sys/class/udc").exists() else False
    )
    unit_state = _systemctl_unit_state(_USB_GADGET_UNIT)

    # Branch B: unit not installed (pre-persistence install) — tell user to run setup.sh.
    if unit_state == "not-found":
        return {
            "status": NOT_CONFIGURED,
            "detail": (
                "/dev/hidg0 missing and persistent gadget service is not installed. "
                "Run: sudo scripts/setup.sh --gadget"
            ),
            **note_kv,
        }

    # Branch C: unit installed but disabled — re-run setup.sh OR enable directly.
    if unit_state == "disabled":
        return {
            "status": PARTIAL,
            "detail": (
                "Persistent gadget service is installed but disabled. "
                "Run: sudo systemctl enable --now cyberraccoon-usb-gadget "
                "(or re-run: sudo scripts/setup.sh --gadget)"
            ),
            **note_kv,
        }

    # Branch E: UDC empty — existing cable/power messaging, unchanged.
    if not udc_present:
        return {
            "status": NOT_CONFIGURED,
            "detail": (
                "Pi can't currently act as a USB device — usually because the "
                "target is powered off, or the cable to the target doesn't "
                "carry data. Power on the target and re-check."
            ),
            **note_kv,
        }

    # Branch D: unit looks healthy on paper but /dev/hidg0 still missing.
    # Likely the unit ran early, found UDC absent, and ConditionPathExistsGlob
    # silently skipped it. User reconnects + restarts the unit.
    return {
        "status": PARTIAL,
        "detail": (
            "Persistent gadget service is enabled but /dev/hidg0 is missing. "
            "Reconnect the target USB cable, then run: "
            "sudo systemctl restart cyberraccoon-usb-gadget. "
            "For details: sudo systemctl status cyberraccoon-usb-gadget"
        ),
        **note_kv,
    }


def _check_usb_gadget() -> dict[str, str]:
    """Check USB HID Gadget setup status.

    On Pi 5, walks /boot/firmware/config.txt to confirm the dwc2 overlay is
    enabled in a model-applicable section and not pinned to host mode —
    catching the most common reason ``/sys/class/udc`` is empty before
    blaming cables or target power state. After that, hands off to
    ``_gadget_runtime_status`` which knows about the persistent systemd
    unit (cyberraccoon-usb-gadget.service) installed by
    ``scripts/setup/gadget.sh``.
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

    config_paths = ["/boot/firmware/config.txt", "/boot/config.txt"]

    if is_pi5:
        pi5_cable_note = (
            "Pi 5 cable check: the Pi USB-C port (the same one used for power) "
            "must carry data to the target, and the target needs to be powered "
            "on — otherwise the Pi can't act as a USB device. Recommended: a "
            "USB power/data splitter (external power to Pi, separate data "
            "cable to target) — lets you swap the data cable to another target "
            "without power-cycling the Pi. A single USB cable from the Pi "
            "USB-C to the target also works (carries power + data; target side "
            "USB-C or USB-A) — under-voltage warnings are possible and "
            "changing target requires powering off the Pi. If the UDC stays "
            "\"not attached\" (a known dwc2 kernel bug on single-cable "
            "setups), switch to the splitter topology or use Bluetooth HID."
        )

        # 1. Verify config.txt enables dwc2 in a usable mode for Pi 5.
        overlay = _read_dwc2_overlay(config_paths, "pi5")
        if overlay["config_path"] is None:
            return {
                "status": NOT_CONFIGURED,
                "detail": "config.txt not found at /boot/firmware/config.txt or /boot/config.txt.",
                "note": pi5_cable_note,
            }
        if not overlay["present"]:
            return {
                "status": NOT_CONFIGURED,
                "detail": (
                    "dwc2 overlay not enabled in config.txt — Pi can't act as "
                    "a USB device until it is added. Run setup --gadget, then reboot."
                ),
                "note": pi5_cable_note,
            }
        if overlay["dr_mode"] == "host":
            return {
                "status": NOT_CONFIGURED,
                "detail": (
                    f"dwc2 overlay in {overlay['config_path']} is set to host "
                    "mode, so the Pi can't act as a USB device. Change "
                    "dr_mode=host to dr_mode=peripheral and reboot."
                ),
                "note": pi5_cable_note,
            }

        # Overlay is configured correctly but the running kernel hasn't picked
        # it up yet (e.g. immediately after `sudo scripts/setup.sh --gadget`).
        # Distinguish this from a runtime cable/power problem so the user is
        # told to reboot, not to re-run setup.
        if not _dwc2_module_loaded():
            return {
                "status": REBOOT_REQUIRED,
                "detail": (
                    f"dwc2 overlay is in {overlay['config_path']} but the "
                    "running kernel hasn't loaded the dwc2 module — reboot "
                    "to apply."
                ),
                "note": pi5_cable_note,
            }

        # 2. Overlay is fine — now check runtime state (incl. systemd unit).
        return _gadget_runtime_status(pi5_cable_note)

    # Pi 4B or other
    overlay = _read_dwc2_overlay(config_paths, "pi4")
    if not overlay["present"]:
        # Preserve "/dev/hidg0 missing" wording for the no-overlay path so
        # the message stays accurate even if the device node happens to exist
        # for some unrelated reason.
        if Path("/dev/hidg0").exists():
            return {"status": READY, "detail": "/dev/hidg0 available"}
        return {
            "status": NOT_CONFIGURED,
            "detail": "dwc2 overlay not in config.txt and /dev/hidg0 missing",
        }
    if overlay["dr_mode"] == "host":
        return {
            "status": NOT_CONFIGURED,
            "detail": (
                f"dwc2 overlay in {overlay['config_path']} is set to host "
                "mode. Change dr_mode=host to dr_mode=peripheral (or remove "
                "dr_mode=) and reboot."
            ),
        }
    if not _dwc2_module_loaded():
        return {
            "status": REBOOT_REQUIRED,
            "detail": (
                f"dwc2 overlay is in {overlay['config_path']} but the "
                "running kernel hasn't loaded the dwc2 module — reboot "
                "to apply."
            ),
        }

    # Overlay is fine on Pi 4B — share the same systemd-unit-aware runtime path.
    return _gadget_runtime_status(None)


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


def _systemctl_unit_state(unit: str) -> str:
    """Return the install state of a systemd unit.

    Wraps ``systemctl is-enabled <unit>``. Returns one of:
        - "enabled"     — unit installed and enabled
        - "disabled"    — unit installed but not enabled
        - "static"      — unit installed but cannot be enabled (no [Install])
        - "masked"      — unit masked
        - "not-found"   — unit file does not exist
        - "unknown"     — systemctl missing, timed out, or returned unrecognised state
    """
    try:
        result = subprocess.run(
            ["systemctl", "is-enabled", unit],
            capture_output=True, text=True, timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "unknown"
    state = result.stdout.strip()
    if state in {"enabled", "disabled", "static", "masked"}:
        return state
    # systemctl returns non-zero + "Failed to get unit file state for X: No such
    # file or directory" on stderr when the unit isn't installed.
    if "No such file" in result.stderr or state == "":
        return "not-found"
    return "unknown"


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
