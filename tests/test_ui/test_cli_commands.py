"""Tests for ui.cli.commands — CLI command handlers."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from config import AppConfig
from ui.app_controller import AppController
from ui.cli.commands import CommandHandler


@pytest.fixture()
def ctrl(tmp_path: Path) -> AppController:
    """AppController with a temp config path."""
    c = AppController(config_path=str(tmp_path / "cfg.yaml"))
    c.load_config()
    return c


@pytest.fixture()
def handler(ctrl: AppController) -> CommandHandler:
    return CommandHandler(ctrl)


# ---------------------------------------------------------------------------
# help
# ---------------------------------------------------------------------------

class TestHelp:
    def test_help_lists_commands(self, handler: CommandHandler) -> None:
        output = handler.execute("help", "")
        assert "config" in output
        assert "task" in output
        assert "wifi" in output
        assert "quit" in output

    def test_unknown_command(self, handler: CommandHandler) -> None:
        output = handler.execute("foobar", "")
        assert "Unknown command" in output


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

class TestConfigCommands:
    def test_config_show(self, handler: CommandHandler) -> None:
        output = handler.execute("config", "show")
        assert "llm" in output.lower()
        assert "capture" in output.lower()

    def test_config_set_and_show(self, handler: CommandHandler) -> None:
        handler.execute("config", "set llm.model custom-model")
        output = handler.execute("config", "show")
        assert "custom-model" in output

    def test_config_set_missing_args(self, handler: CommandHandler) -> None:
        output = handler.execute("config", "set llm.model")
        assert "Usage" in output

    def test_config_reset(self, handler: CommandHandler, ctrl: AppController) -> None:
        ctrl.update_config(**{"llm.model": "temporary"})
        handler.execute("config", "reset")
        config = ctrl.get_config()
        assert config.llm.model == AppConfig().llm.model

    def test_config_show_masks_secrets(self, handler: CommandHandler, ctrl: AppController) -> None:
        ctrl.update_config(**{"llm.api_key": "sk-very-secret-key"})
        output = handler.execute("config", "show")
        assert "sk-v..." in output
        assert "sk-very-secret-key" not in output


# ---------------------------------------------------------------------------
# task
# ---------------------------------------------------------------------------

class TestTaskCommands:
    def test_task_status_idle(self, handler: CommandHandler) -> None:
        output = handler.execute("task", "status")
        assert "idle" in output

    def test_task_run_without_modules(self, handler: CommandHandler) -> None:
        output = handler.execute("task", "run Open Notepad")
        assert "Cannot start task" in output or "Modules not init" in output

    def test_task_run_missing_goal(self, handler: CommandHandler) -> None:
        output = handler.execute("task", "run")
        assert "Usage" in output

    def test_task_abort(self, handler: CommandHandler) -> None:
        output = handler.execute("task", "abort")
        assert "Abort" in output

    def test_task_unknown_sub(self, handler: CommandHandler) -> None:
        output = handler.execute("task", "unknown")
        assert "Usage" in output


# ---------------------------------------------------------------------------
# capture
# ---------------------------------------------------------------------------

class TestCaptureCommands:
    def test_capture_test_without_modules(self, handler: CommandHandler) -> None:
        output = handler.execute("capture", "test")
        assert "not available" in output.lower() or "modules" in output.lower()

    def test_capture_test_with_mock(
        self, handler: CommandHandler, ctrl: AppController
    ) -> None:
        mock_result = MagicMock(width=1280, height=720, size_bytes=50000)
        with patch.object(ctrl, "capture_preview", return_value=mock_result):
            output = handler.execute("capture", "test")
            assert "1280" in output
            assert "720" in output

    def test_capture_unknown_sub(self, handler: CommandHandler) -> None:
        output = handler.execute("capture", "")
        assert "Usage" in output


# ---------------------------------------------------------------------------
# wifi
# ---------------------------------------------------------------------------

class TestWifiCommands:
    def test_wifi_unavailable(self, handler: CommandHandler, ctrl: AppController) -> None:
        with patch.object(ctrl, "get_wifi_manager", return_value=None):
            output = handler.execute("wifi", "scan")
            assert "not available" in output.lower()

    def test_wifi_scan(self, handler: CommandHandler, ctrl: AppController) -> None:
        from ui.wifi_manager import WiFiNetwork
        mock_wm = MagicMock()
        mock_wm.scan.return_value = [
            WiFiNetwork(ssid="Home", signal_strength=-45, security="WPA2", connected=True),
            WiFiNetwork(ssid="Guest", signal_strength=-70, security="Open", connected=False),
        ]
        with patch.object(ctrl, "get_wifi_manager", return_value=mock_wm):
            output = handler.execute("wifi", "scan")
            assert "Home" in output
            assert "Guest" in output

    def test_wifi_status_connected(self, handler: CommandHandler, ctrl: AppController) -> None:
        mock_wm = MagicMock()
        mock_wm.backend = "networkmanager"
        mock_wm.is_connected.return_value = True
        mock_wm.get_current_network.return_value = "HomeNet"
        mock_wm.get_ip_address.return_value = "192.168.1.42"
        with patch.object(ctrl, "get_wifi_manager", return_value=mock_wm):
            output = handler.execute("wifi", "status")
            assert "HomeNet" in output
            assert "192.168.1.42" in output

    def test_wifi_connect(self, handler: CommandHandler, ctrl: AppController) -> None:
        mock_wm = MagicMock()
        mock_wm.connect.return_value = True
        mock_wm.get_ip_address.return_value = "10.0.0.5"
        with patch.object(ctrl, "get_wifi_manager", return_value=mock_wm):
            output = handler.execute("wifi", "connect MySSID mypass")
            assert "Connected" in output
            mock_wm.connect.assert_called_once_with("MySSID", "mypass")

    def test_wifi_unknown_sub(self, handler: CommandHandler, ctrl: AppController) -> None:
        mock_wm = MagicMock()
        with patch.object(ctrl, "get_wifi_manager", return_value=mock_wm):
            output = handler.execute("wifi", "unknown")
            assert "Usage" in output


# ---------------------------------------------------------------------------
# logs
# ---------------------------------------------------------------------------

class TestLogCommands:
    def test_logs_empty(self, handler: CommandHandler) -> None:
        output = handler.execute("logs", "")
        assert "no logs" in output.lower()

    def test_logs_tail(self, handler: CommandHandler, ctrl: AppController) -> None:
        with patch.object(ctrl, "get_logs", return_value=["line1", "line2"]):
            output = handler.execute("logs", "tail 5")
            assert "line1" in output

    def test_logs_clear(self, handler: CommandHandler) -> None:
        output = handler.execute("logs", "clear")
        assert "cleared" in output.lower()


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

class TestStatusCommand:
    def test_status_output(self, handler: CommandHandler) -> None:
        output = handler.execute("status", "")
        assert "Modules ready" in output
        assert "Capture" in output
        assert "LLM" in output


# ---------------------------------------------------------------------------
# command_names
# ---------------------------------------------------------------------------

class TestCommandNames:
    def test_command_names_non_empty(self, handler: CommandHandler) -> None:
        names = handler.command_names()
        assert "config" in names
        assert "task run" in names
        assert "quit" in names
        assert len(names) > 10
