"""Tests for ui.wifi_manager — WiFiManager with mocked subprocess calls."""

from __future__ import annotations

import subprocess
from unittest.mock import patch, MagicMock

import pytest

from ui.exceptions import WiFiError
from ui.wifi_manager import WiFiManager, WiFiNetwork


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _cp(
    stdout: str = "",
    stderr: str = "",
    returncode: int = 0,
) -> subprocess.CompletedProcess[str]:
    """Shorthand for building CompletedProcess."""
    return subprocess.CompletedProcess(
        args=[], returncode=returncode, stdout=stdout, stderr=stderr,
    )


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

class TestBackendDetection:
    """WiFiManager._detect_backend selects nmcli or wpa_supplicant."""

    def test_prefers_nmcli(self) -> None:
        with patch("shutil.which", side_effect=lambda x: "/usr/bin/nmcli" if x == "nmcli" else None):
            assert WiFiManager._detect_backend() == "networkmanager"

    def test_falls_back_to_wpa(self) -> None:
        def _which(cmd: str) -> str | None:
            if cmd == "nmcli":
                return None
            if cmd in ("wpa_supplicant", "wpa_cli"):
                return f"/usr/sbin/{cmd}"
            return None

        with patch("shutil.which", side_effect=_which):
            assert WiFiManager._detect_backend() == "wpa_supplicant"

    def test_raises_when_no_backend(self) -> None:
        with patch("shutil.which", return_value=None):
            with pytest.raises(WiFiError, match="No Wi-Fi backend"):
                WiFiManager._detect_backend()


# ---------------------------------------------------------------------------
# Connect
# ---------------------------------------------------------------------------

class TestConnect:
    """WiFiManager.connect()."""

    def test_connect_success(self, mock_wifi_manager) -> None:
        wm, run_cmd = mock_wifi_manager
        run_cmd.return_value = _cp(stdout="Device 'wlan0' successfully activated")

        result = wm.connect("MyNetwork", "pass123")
        assert result is True

        # Verify nmcli was called with correct args
        args = run_cmd.call_args[0][0]
        assert "connect" in args
        assert "MyNetwork" in args
        assert "pass123" in args

    def test_connect_failure_raises(self, mock_wifi_manager) -> None:
        wm, run_cmd = mock_wifi_manager
        run_cmd.return_value = _cp(
            returncode=1, stderr="Error: No network with SSID 'Nope' found."
        )

        with pytest.raises(WiFiError, match="nmcli connect failed"):
            wm.connect("Nope", "pass")

    def test_connect_empty_ssid_raises(self, mock_wifi_manager) -> None:
        wm, _ = mock_wifi_manager
        with pytest.raises(WiFiError, match="SSID must not be empty"):
            wm.connect("", "pass")

    def test_connect_open_network(self, mock_wifi_manager) -> None:
        wm, run_cmd = mock_wifi_manager
        run_cmd.return_value = _cp(stdout="connected")

        wm.connect("OpenNet", "")

        # Password arg should NOT be in the command
        args = run_cmd.call_args[0][0]
        assert "password" not in args


# ---------------------------------------------------------------------------
# Scan
# ---------------------------------------------------------------------------

class TestScan:
    """WiFiManager.scan() parses nmcli output."""

    def test_scan_parses_networks(
        self, mock_wifi_manager, mock_nmcli_scan_output: str
    ) -> None:
        wm, run_cmd = mock_wifi_manager
        run_cmd.return_value = _cp(stdout=mock_nmcli_scan_output)

        networks = wm.scan()

        # "HomeWiFi" appears twice but should be deduplicated
        ssids = [n.ssid for n in networks]
        assert ssids.count("HomeWiFi") == 1

        # Hidden SSID (empty) should be excluded
        assert "" not in ssids

        # Should be sorted by signal (strongest first)
        signals = [n.signal_strength for n in networks]
        assert signals == sorted(signals, reverse=True)

        # Check connected flag
        home = next(n for n in networks if n.ssid == "HomeWiFi")
        assert home.connected is True
        assert home.security == "WPA2"

    def test_scan_failure_returns_empty(self, mock_wifi_manager) -> None:
        wm, run_cmd = mock_wifi_manager
        run_cmd.return_value = _cp(returncode=1, stderr="scan failed")

        networks = wm.scan()
        assert networks == []


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

class TestStatus:
    """WiFiManager.is_connected(), get_current_network(), get_ip_address()."""

    def test_is_connected_true(self, mock_wifi_manager) -> None:
        wm, run_cmd = mock_wifi_manager
        run_cmd.return_value = _cp(stdout="wifi:connected\nloopback:connected")

        assert wm.is_connected() is True

    def test_is_connected_false(self, mock_wifi_manager) -> None:
        wm, run_cmd = mock_wifi_manager
        run_cmd.return_value = _cp(stdout="wifi:disconnected\n")

        assert wm.is_connected() is False

    def test_get_current_network(self, mock_wifi_manager) -> None:
        wm, run_cmd = mock_wifi_manager
        run_cmd.return_value = _cp(stdout="yes:HomeWiFi\nno:Neighbor\n")

        assert wm.get_current_network() == "HomeWiFi"

    def test_get_current_network_none(self, mock_wifi_manager) -> None:
        wm, run_cmd = mock_wifi_manager
        run_cmd.return_value = _cp(stdout="no:HomeWiFi\nno:Neighbor\n")

        assert wm.get_current_network() is None

    def test_get_ip_address(self, mock_wifi_manager) -> None:
        wm, run_cmd = mock_wifi_manager
        run_cmd.return_value = _cp(
            stdout="IP4.ADDRESS[1]:192.168.1.42/24\n"
        )

        assert wm.get_ip_address() == "192.168.1.42"

    def test_get_ip_address_no_ip(self, mock_wifi_manager) -> None:
        wm, run_cmd = mock_wifi_manager
        run_cmd.return_value = _cp(returncode=1, stderr="error")

        assert wm.get_ip_address() is None


# ---------------------------------------------------------------------------
# Disconnect
# ---------------------------------------------------------------------------

class TestDisconnect:
    """WiFiManager.disconnect()."""

    def test_disconnect_success(self, mock_wifi_manager) -> None:
        wm, run_cmd = mock_wifi_manager
        run_cmd.return_value = _cp(stdout="Device 'wlan0' successfully disconnected.")

        assert wm.disconnect() is True

    def test_disconnect_failure(self, mock_wifi_manager) -> None:
        wm, run_cmd = mock_wifi_manager
        run_cmd.return_value = _cp(returncode=1, stderr="error")

        assert wm.disconnect() is False


# ---------------------------------------------------------------------------
# Saved networks
# ---------------------------------------------------------------------------

class TestHasSavedNetwork:
    """WiFiManager.has_saved_network()."""

    def test_has_saved_true(self, mock_wifi_manager) -> None:
        wm, run_cmd = mock_wifi_manager
        run_cmd.return_value = _cp(
            stdout="802-11-wireless:HomeWiFi\nethernet:Wired\n"
        )

        assert wm.has_saved_network() is True

    def test_has_saved_false(self, mock_wifi_manager) -> None:
        wm, run_cmd = mock_wifi_manager
        run_cmd.return_value = _cp(stdout="ethernet:Wired\n")

        assert wm.has_saved_network() is False


# ---------------------------------------------------------------------------
# Subprocess timeout
# ---------------------------------------------------------------------------

class TestSubprocessTimeout:
    """_run_cmd wraps TimeoutExpired in WiFiError."""

    def test_timeout_raises_wifi_error(self) -> None:
        with patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd=["nmcli"], timeout=30),
        ):
            with pytest.raises(WiFiError, match="timed out"):
                WiFiManager._run_cmd(["nmcli", "test"])

    def test_command_not_found_raises(self) -> None:
        with patch(
            "subprocess.run",
            side_effect=FileNotFoundError("nmcli"),
        ):
            with pytest.raises(WiFiError, match="Command not found"):
                WiFiManager._run_cmd(["nmcli", "test"])
