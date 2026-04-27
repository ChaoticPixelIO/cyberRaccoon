"""Tests for ui.setup_status — hardware/software setup checks.

These tests exercise the structural contract: shape of the returned dict,
status strings, and graceful degradation on non-Pi systems (macOS dev box).

Hardware-specific paths (TC358743 visible, Bluetooth adapter configured)
are tested with mocks — the real checks run on the Pi.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from cyberraccoon.ui.setup_status import (
    NOT_AVAILABLE,
    NOT_CONFIGURED,
    PARTIAL,
    READY,
    REBOOT_REQUIRED,
    _check_usb_gadget,
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


class TestUsbGadgetRebootRequired:
    """Post-setup-pre-reboot state must report REBOOT_REQUIRED, not NOT_CONFIGURED.

    After ``sudo scripts/setup.sh --gadget`` writes the dwc2 overlay to
    config.txt, the kernel hasn't loaded the dwc2 module yet — so
    ``/sys/module/dwc2`` doesn't exist and ``/dev/hidg0`` is missing. The
    Status tab must tell the user to reboot, not to re-run setup.
    """

    def _path_exists_factory(self, missing: set[str]):
        """Build a Path.exists side_effect that returns False for paths in ``missing``."""
        real_exists = Path.exists

        def fake_exists(self_path: Path) -> bool:
            if str(self_path) in missing:
                return False
            return real_exists(self_path)

        return fake_exists

    def test_pi5_reboot_required_when_dwc2_module_not_loaded(self) -> None:
        overlay = {
            "present": True,
            "dr_mode": "peripheral",
            "config_path": "/boot/firmware/config.txt",
        }
        missing = {"/sys/module/dwc2", "/dev/hidg0"}

        with patch(
            "cyberraccoon.ui.setup_status._is_raspberry_pi", return_value=True
        ), patch(
            "cyberraccoon.ui.setup_status.Path.read_text",
            return_value="Raspberry Pi 5 Model B",
        ), patch(
            "cyberraccoon.ui.setup_status._read_dwc2_overlay", return_value=overlay
        ), patch(
            "cyberraccoon.ui.setup_status.Path.exists",
            new=self._path_exists_factory(missing),
        ):
            result = _check_usb_gadget()

        assert result["status"] == REBOOT_REQUIRED, result
        assert "reboot" in result["detail"].lower()

    def test_pi4b_reboot_required_when_dwc2_module_not_loaded(self) -> None:
        overlay = {
            "present": True,
            "dr_mode": "peripheral",
            "config_path": "/boot/firmware/config.txt",
        }
        missing = {"/sys/module/dwc2", "/dev/hidg0"}

        with patch(
            "cyberraccoon.ui.setup_status._is_raspberry_pi", return_value=True
        ), patch(
            "cyberraccoon.ui.setup_status.Path.read_text",
            return_value="Raspberry Pi 4 Model B",
        ), patch(
            "cyberraccoon.ui.setup_status._read_dwc2_overlay", return_value=overlay
        ), patch(
            "cyberraccoon.ui.setup_status.Path.exists",
            new=self._path_exists_factory(missing),
        ):
            result = _check_usb_gadget()

        assert result["status"] == REBOOT_REQUIRED, result
        assert "reboot" in result["detail"].lower()


class TestUsbGadgetSystemdUnit:
    """Post-reboot states distinguished by systemd unit install state.

    After dwc2 is loaded, /dev/hidg0 absence has multiple causes:
      - unit not installed (old setup, pre-persistence)
      - unit installed but disabled
      - unit enabled but UDC empty (cable/power)
      - unit enabled, UDC present, /dev/hidg0 still missing (skipped by Condition*)

    All cases are exercised on the Pi 5 path because Pi 4B shares the same
    `_gadget_runtime_status` helper.
    """

    @staticmethod
    def _path_exists_factory(missing: set[str], extra_present: set[str] = frozenset()):
        """Return a Path.exists replacement controlling specific paths.

        ``missing`` paths return False even if they exist on the dev box.
        ``extra_present`` paths return True even if they do not exist on
        the dev box (so /sys/class/udc can be faked "present" on macOS).
        Other paths fall through to the real Path.exists.
        """
        real_exists = Path.exists

        def fake_exists(self_path: Path) -> bool:
            s = str(self_path)
            if s in missing:
                return False
            if s in extra_present:
                return True
            return real_exists(self_path)

        return fake_exists

    @staticmethod
    def _iterdir_factory(udc_present: bool):
        """Return a Path.iterdir replacement that controls /sys/class/udc only.

        When asked for /sys/class/udc, returns a non-empty/empty iterator
        based on ``udc_present``. Other paths fall through to the real
        Path.iterdir (rare in these tests; we patch Path.exists for the
        rest of the world).
        """
        real_iterdir = Path.iterdir

        def fake_iterdir(self_path: Path):
            if str(self_path) == "/sys/class/udc":
                if udc_present:
                    return iter([Path("/sys/class/udc/dummy")])
                return iter([])
            return real_iterdir(self_path)

        return fake_iterdir

    def _run_pi5(
        self,
        *,
        hidg0_present: bool,
        udc_present: bool,
        unit_state: str,
    ) -> dict[str, str]:
        """Drive _check_usb_gadget on the Pi 5 path with the standard mocks.

        Mocks:
          - _is_raspberry_pi → True
          - /proc/device-tree/model → "Raspberry Pi 5 Model B"
          - _read_dwc2_overlay → overlay present, peripheral mode
          - Path.exists → controls /dev/hidg0 + /sys/class/udc + /sys/module/dwc2
          - Path.iterdir → controls /sys/class/udc contents
          - _systemctl_unit_state → caller-supplied
        """
        missing: set[str] = set()
        # Always say /sys/class/udc and /sys/module/dwc2 exist; the latter
        # ensures we get past the REBOOT_REQUIRED gate added in 260428-0v8.
        extra_present: set[str] = {"/sys/class/udc", "/sys/module/dwc2"}
        if not hidg0_present:
            missing.add("/dev/hidg0")
        else:
            extra_present.add("/dev/hidg0")

        overlay = {
            "present": True,
            "dr_mode": "peripheral",
            "config_path": "/boot/firmware/config.txt",
        }

        with patch("cyberraccoon.ui.setup_status._is_raspberry_pi", return_value=True), \
             patch("cyberraccoon.ui.setup_status.Path.read_text",
                   return_value="Raspberry Pi 5 Model B"), \
             patch("cyberraccoon.ui.setup_status._read_dwc2_overlay",
                   return_value=overlay), \
             patch("cyberraccoon.ui.setup_status.Path.exists",
                   new=self._path_exists_factory(missing, extra_present)), \
             patch("cyberraccoon.ui.setup_status.Path.iterdir",
                   new=self._iterdir_factory(udc_present)), \
             patch("cyberraccoon.ui.setup_status._systemctl_unit_state",
                   return_value=unit_state):
            return _check_usb_gadget()

    def test_unit_not_found_returns_not_configured(self) -> None:
        """Branch B: unit missing (old install) → NOT_CONFIGURED + setup.sh hint."""
        result = self._run_pi5(
            hidg0_present=False, udc_present=True, unit_state="not-found",
        )
        assert result["status"] == NOT_CONFIGURED
        assert "scripts/setup.sh --gadget" in result["detail"]

    def test_unit_disabled_returns_partial(self) -> None:
        """Branch C: unit installed but disabled → PARTIAL + enable hint."""
        result = self._run_pi5(
            hidg0_present=False, udc_present=True, unit_state="disabled",
        )
        assert result["status"] == PARTIAL
        # Either of these phrasings is acceptable per the plan
        assert (
            "systemctl enable" in result["detail"]
            or "scripts/setup.sh --gadget" in result["detail"]
        )

    def test_unit_enabled_but_no_hidg0_returns_partial(self) -> None:
        """Branch D: unit enabled, UDC present, hidg0 missing → PARTIAL + status hint."""
        result = self._run_pi5(
            hidg0_present=False, udc_present=True, unit_state="enabled",
        )
        assert result["status"] == PARTIAL
        assert (
            "systemctl status cyberraccoon-usb-gadget" in result["detail"]
            or "systemctl restart cyberraccoon-usb-gadget" in result["detail"]
        )

    def test_unit_enabled_with_hidg0_returns_ready(self) -> None:
        """Branch A: hidg0 present → READY (regardless of unit state)."""
        result = self._run_pi5(
            hidg0_present=True, udc_present=True, unit_state="enabled",
        )
        assert result["status"] == READY
        assert "/dev/hidg0" in result["detail"]

    def test_unit_enabled_but_udc_empty_falls_through_to_cable_message(self) -> None:
        """Branch E: UDC empty → existing cable/power messaging preserved."""
        result = self._run_pi5(
            hidg0_present=False, udc_present=False, unit_state="enabled",
        )
        assert result["status"] == NOT_CONFIGURED
        # Existing message language: target powered off / cable doesn't carry data
        assert (
            "target is powered off" in result["detail"]
            or "cable" in result["detail"]
            or "data" in result["detail"]
        )
