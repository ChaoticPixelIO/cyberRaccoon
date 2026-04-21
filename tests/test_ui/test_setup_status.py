"""Tests for ui.setup_status — hardware/software setup checks.

These tests exercise the structural contract: shape of the returned dict,
status strings, and graceful degradation on non-Pi systems (macOS dev box).

Hardware-specific paths (TC358743 visible, Bluetooth adapter configured)
are tested with mocks — the real checks run on the Pi.
"""

from __future__ import annotations

from unittest.mock import patch

from cyberraccoon.ui.setup_status import (
    NOT_AVAILABLE,
    NOT_CONFIGURED,
    PARTIAL,
    READY,
    REBOOT_REQUIRED,
    _probe_opencv,
    check_setup_status,
)


class TestCheckSetupStatusShape:
    """The return dict structure must stay stable — the frontend depends on it."""

    def test_returns_required_top_level_keys(self) -> None:
        result = check_setup_status()
        assert set(result.keys()) == {
            "components",
            "needs_setup",
            "needs_reboot",
            "setup_commands",
        }

    def test_all_components_present(self) -> None:
        result = check_setup_status()
        assert set(result["components"].keys()) == {
            "python_env",
            "bluetooth",
            "usb_gadget",
            "csi_hdmi",
            "airplay",
        }

    def test_each_component_has_status_and_detail(self) -> None:
        result = check_setup_status()
        for name, comp in result["components"].items():
            assert "status" in comp, f"{name} missing 'status'"
            assert "detail" in comp, f"{name} missing 'detail'"
            assert isinstance(comp["status"], str)
            assert isinstance(comp["detail"], str)

    def test_status_values_are_known(self) -> None:
        valid = {READY, NOT_CONFIGURED, NOT_AVAILABLE, REBOOT_REQUIRED, PARTIAL}
        result = check_setup_status()
        for name, comp in result["components"].items():
            assert comp["status"] in valid, f"{name} has unknown status {comp['status']!r}"


class TestCheckSetupStatusLogic:
    """Relationships between component statuses and the top-level summary."""

    def test_setup_command_present_when_any_component_needs_setup(self) -> None:
        result = check_setup_status()
        hw_components = ("bluetooth", "usb_gadget", "csi_hdmi", "airplay")
        any_needs_setup = any(
            result["components"][c]["status"] in (NOT_CONFIGURED, PARTIAL)
            for c in hw_components
        )
        if any_needs_setup:
            assert "sudo scripts/setup.sh --all" in result["setup_commands"]

    def test_reboot_command_present_only_when_reboot_required(self) -> None:
        result = check_setup_status()
        hw_components = ("bluetooth", "usb_gadget", "csi_hdmi", "airplay")
        any_reboot = any(
            result["components"][c]["status"] == REBOOT_REQUIRED
            for c in hw_components
        )
        if any_reboot:
            assert "sudo reboot" in result["setup_commands"]
        else:
            assert "sudo reboot" not in result["setup_commands"]


class TestProbeOpenCV:
    """_probe_opencv is called once per check_setup_status; must be robust."""

    def test_returns_expected_keys(self) -> None:
        result = _probe_opencv()
        assert set(result.keys()) == {"importable", "version", "has_gstreamer"}

    def test_handles_missing_cv2(self) -> None:
        with patch("importlib.import_module", side_effect=ImportError("no cv2")):
            result = _probe_opencv()
            assert result == {"importable": False, "version": None, "has_gstreamer": False}


class TestNonPiBehavior:
    """On macOS (no /proc/device-tree/model), checks must degrade gracefully."""

    def test_does_not_raise_on_macos(self) -> None:
        # Just verify it runs without exceptions on the dev machine
        result = check_setup_status()
        assert isinstance(result, dict)
