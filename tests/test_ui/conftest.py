"""Shared fixtures for M5 (ui) tests."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from config import AppConfig


@pytest.fixture()
def tmp_config_path(tmp_path: Path) -> Path:
    """Return a temporary path for config.yaml inside a temp dir."""
    return tmp_path / "config.yaml"


@pytest.fixture()
def sample_config() -> AppConfig:
    """Return an AppConfig with recognisable non-default values."""
    cfg = AppConfig()
    cfg.llm.model = "gpt-4o"
    cfg.llm.api_key = "sk-test-key"
    cfg.network.wifi_ssid = "TestNetwork"
    cfg.network.wifi_password = "secret123"
    cfg.network.web_port = 9999
    cfg.capture_source = "csi"
    return cfg


@pytest.fixture()
def mock_nmcli_scan_output() -> str:
    """Realistic nmcli wifi list output (terse format)."""
    return (
        "HomeWiFi:85:WPA2:yes\n"
        "Neighbor5G:72:WPA2:no\n"
        "OpenCafe::Open:no\n"        # note: signal is empty → parse error
        "Office:60:WPA3:no\n"
        ":45:WPA2:no\n"              # hidden SSID (empty)
        "HomeWiFi:80:WPA2:no\n"      # duplicate
    )


def _make_completed_process(
    stdout: str = "",
    stderr: str = "",
    returncode: int = 0,
) -> subprocess.CompletedProcess[str]:
    """Helper to build a subprocess.CompletedProcess."""
    return subprocess.CompletedProcess(
        args=[], returncode=returncode, stdout=stdout, stderr=stderr,
    )


@pytest.fixture()
def mock_wifi_manager():
    """Return a WiFiManager with _run_cmd and _detect_backend mocked.

    Usage in tests::

        def test_something(mock_wifi_manager):
            wm, run_cmd = mock_wifi_manager
            run_cmd.return_value = _make_completed_process(stdout="ok")
            assert wm.is_connected() is True
    """
    with patch(
        "ui.wifi_manager.WiFiManager._detect_backend",
        return_value="networkmanager",
    ), patch(
        "ui.wifi_manager.WiFiManager._run_cmd",
    ) as mock_run:
        from ui.wifi_manager import WiFiManager
        wm = WiFiManager()
        yield wm, mock_run
