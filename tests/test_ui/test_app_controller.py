"""Tests for ui.app_controller — AppController lifecycle, events, tasks."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from config import AppConfig
from ui.app_controller import (
    AppController,
    AppEvent,
    AppEventType,
    LogCaptureHandler,
)
from ui.exceptions import TaskError


# ---------------------------------------------------------------------------
# LogCaptureHandler
# ---------------------------------------------------------------------------

class TestLogCaptureHandler:
    """Ring-buffer logging handler tests."""

    def test_captures_records(self) -> None:
        handler = LogCaptureHandler(max_records=10)
        logger = logging.getLogger("test.capture_handler")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        logger.info("hello")
        logger.warning("world")

        records = handler.get_records()
        assert len(records) == 2
        assert records[0].getMessage() == "hello"
        assert records[1].getMessage() == "world"

        logger.removeHandler(handler)

    def test_ring_buffer_overflow(self) -> None:
        handler = LogCaptureHandler(max_records=3)
        logger = logging.getLogger("test.overflow")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        for i in range(5):
            logger.info("msg_%d", i)

        records = handler.get_records()
        assert len(records) == 3
        assert records[0].getMessage() == "msg_2"
        assert records[2].getMessage() == "msg_4"

        logger.removeHandler(handler)

    def test_get_records_count(self) -> None:
        handler = LogCaptureHandler(max_records=10)
        logger = logging.getLogger("test.count")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        for i in range(5):
            logger.info("msg_%d", i)

        last_two = handler.get_records(2)
        assert len(last_two) == 2
        assert last_two[0].getMessage() == "msg_3"

        logger.removeHandler(handler)

    def test_clear(self) -> None:
        handler = LogCaptureHandler()
        logger = logging.getLogger("test.clear")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        logger.info("x")
        handler.clear()
        assert handler.get_records() == []

        logger.removeHandler(handler)

    def test_on_record_callback(self) -> None:
        callback = MagicMock()
        handler = LogCaptureHandler(on_record=callback)
        logger = logging.getLogger("test.callback")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        logger.info("event")
        callback.assert_called_once()
        assert callback.call_args[0][0].getMessage() == "event"

        logger.removeHandler(handler)


# ---------------------------------------------------------------------------
# Event system
# ---------------------------------------------------------------------------

class TestEventSystem:
    """AppController event add/remove/emit."""

    def test_add_and_emit(self, tmp_path: Path) -> None:
        ctrl = AppController(config_path=str(tmp_path / "cfg.yaml"))
        events: list[AppEvent] = []
        ctrl.add_listener(events.append)

        ctrl.load_config()

        assert len(events) == 1
        assert events[0].type == AppEventType.CONFIG_CHANGED

    def test_remove_listener(self, tmp_path: Path) -> None:
        ctrl = AppController(config_path=str(tmp_path / "cfg.yaml"))
        events: list[AppEvent] = []
        ctrl.add_listener(events.append)
        ctrl.remove_listener(events.append)

        ctrl.load_config()
        assert len(events) == 0

    def test_listener_error_does_not_crash(self, tmp_path: Path) -> None:
        ctrl = AppController(config_path=str(tmp_path / "cfg.yaml"))

        def bad_listener(event: AppEvent) -> None:
            raise RuntimeError("boom")

        ctrl.add_listener(bad_listener)
        ctrl.load_config()  # should not raise


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class TestConfig:
    """Config load/update/reset via AppController."""

    def test_load_config_returns_app_config(self, tmp_path: Path) -> None:
        ctrl = AppController(config_path=str(tmp_path / "cfg.yaml"))
        config = ctrl.load_config()
        assert isinstance(config, AppConfig)

    def test_get_config_lazy_loads(self, tmp_path: Path) -> None:
        ctrl = AppController(config_path=str(tmp_path / "cfg.yaml"))
        config = ctrl.get_config()
        assert isinstance(config, AppConfig)

    def test_update_config_persists(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "cfg.yaml"
        ctrl = AppController(config_path=str(cfg_path))
        ctrl.load_config()
        ctrl.update_config(**{"llm.model": "gpt-4o"})

        # Reload from file
        ctrl2 = AppController(config_path=str(cfg_path))
        config = ctrl2.load_config()
        assert config.llm.model == "gpt-4o"

    def test_update_config_emits_event(self, tmp_path: Path) -> None:
        ctrl = AppController(config_path=str(tmp_path / "cfg.yaml"))
        ctrl.load_config()

        events: list[AppEvent] = []
        ctrl.add_listener(events.append)
        ctrl.update_config(**{"capture_source": "csi"})

        config_events = [e for e in events if e.type == AppEventType.CONFIG_CHANGED]
        assert len(config_events) == 1
        assert config_events[0].data["source"] == "update"

    def test_reset_config(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "cfg.yaml"
        ctrl = AppController(config_path=str(cfg_path))
        ctrl.load_config()
        ctrl.update_config(**{"llm.model": "custom"})
        assert ctrl.get_config().llm.model == "custom"

        ctrl.reset_config()
        assert ctrl.get_config().llm.model == AppConfig().llm.model


# ---------------------------------------------------------------------------
# Module lifecycle (mocked)
# ---------------------------------------------------------------------------

class TestModuleLifecycle:
    """init_modules / close_modules with mocked hardware."""

    def _mock_modules(self):
        """Return patches for all hardware dependencies."""
        mock_capture = MagicMock()
        mock_capture.capture.return_value = MagicMock(
            width=1280, height=720, size_bytes=50000, base64_jpeg="abc",
        )

        patches = {
            "create_capture": patch(
                "ui.app_controller.create_capture",
                return_value=mock_capture,
            ),
            "create_protocol": patch(
                "ui.app_controller.create_protocol",
                return_value=MagicMock(),
            ),
            "ActionExecutor": patch(
                "ui.app_controller.ActionExecutor",
                return_value=MagicMock(),
            ),
            "BluetoothExecutor": patch(
                "ui.app_controller.BluetoothExecutor",
                return_value=MagicMock(),
            ),
            "VisionAgent": patch(
                "ui.app_controller.VisionAgent",
                return_value=MagicMock(),
            ),
        }
        return patches, mock_capture

    def test_init_and_close(self, tmp_path: Path) -> None:
        patches, _ = self._mock_modules()
        with patches["create_capture"], patches["create_protocol"], \
             patches["ActionExecutor"], patches["BluetoothExecutor"], \
             patches["VisionAgent"]:
            ctrl = AppController(config_path=str(tmp_path / "cfg.yaml"))
            ctrl.load_config()

            events: list[AppEvent] = []
            ctrl.add_listener(events.append)

            ctrl.init_modules()
            assert ctrl.modules_ready is True

            ready_events = [e for e in events if e.type == AppEventType.MODULES_READY]
            assert len(ready_events) >= 1

            ctrl.close_modules()
            assert ctrl.modules_ready is False

            closed_events = [e for e in events if e.type == AppEventType.MODULES_CLOSED]
            assert len(closed_events) >= 1

    def test_capture_preview(self, tmp_path: Path) -> None:
        patches, mock_capture = self._mock_modules()
        with patches["create_capture"], patches["create_protocol"], \
             patches["ActionExecutor"], patches["BluetoothExecutor"], \
             patches["VisionAgent"]:
            ctrl = AppController(config_path=str(tmp_path / "cfg.yaml"))
            ctrl.load_config()
            ctrl.init_modules()

            result = ctrl.capture_preview()
            assert result is not None
            assert result.width == 1280

            ctrl.close_modules()

    def test_capture_preview_none_without_modules(self, tmp_path: Path) -> None:
        ctrl = AppController(config_path=str(tmp_path / "cfg.yaml"))
        assert ctrl.capture_preview() is None


# ---------------------------------------------------------------------------
# Split module lifecycle (independent capture / executor)
# ---------------------------------------------------------------------------

class TestSplitModuleLifecycle:
    """init_capture / close_capture / init_executor / close_executor."""

    def _mock_modules(self):
        """Return patches for all hardware dependencies."""
        mock_capture = MagicMock()
        mock_capture.capture.return_value = MagicMock(
            width=1280, height=720, size_bytes=50000, base64_jpeg="abc",
        )

        patches = {
            "create_capture": patch(
                "ui.app_controller.create_capture",
                return_value=mock_capture,
            ),
            "create_protocol": patch(
                "ui.app_controller.create_protocol",
                return_value=MagicMock(),
            ),
            "ActionExecutor": patch(
                "ui.app_controller.ActionExecutor",
                return_value=MagicMock(),
            ),
            "BluetoothExecutor": patch(
                "ui.app_controller.BluetoothExecutor",
                return_value=MagicMock(),
            ),
            "VisionAgent": patch(
                "ui.app_controller.VisionAgent",
                return_value=MagicMock(),
            ),
        }
        return patches, mock_capture

    def test_init_capture_independently(self, tmp_path: Path) -> None:
        patches, _ = self._mock_modules()
        with patches["create_capture"], patches["create_protocol"], \
             patches["ActionExecutor"], patches["BluetoothExecutor"], \
             patches["VisionAgent"]:
            ctrl = AppController(config_path=str(tmp_path / "cfg.yaml"))
            ctrl.load_config()

            events: list[AppEvent] = []
            ctrl.add_listener(events.append)

            result = ctrl.init_capture()
            assert ctrl.capture_ready is True
            assert ctrl.executor_ready is False
            assert ctrl.modules_ready is False
            assert result is not None  # test frame

            capture_events = [e for e in events if e.type == AppEventType.CAPTURE_READY]
            assert len(capture_events) == 1

            # No MODULES_READY since executor not connected
            modules_events = [e for e in events if e.type == AppEventType.MODULES_READY]
            assert len(modules_events) == 0

    def test_init_executor_independently(self, tmp_path: Path) -> None:
        patches, _ = self._mock_modules()
        with patches["create_capture"], patches["create_protocol"], \
             patches["ActionExecutor"], patches["BluetoothExecutor"], \
             patches["VisionAgent"]:
            ctrl = AppController(config_path=str(tmp_path / "cfg.yaml"))
            ctrl.load_config()

            events: list[AppEvent] = []
            ctrl.add_listener(events.append)

            ctrl.init_executor()
            assert ctrl.executor_ready is True
            assert ctrl.capture_ready is False
            assert ctrl.modules_ready is False

            executor_events = [e for e in events if e.type == AppEventType.EXECUTOR_READY]
            assert len(executor_events) == 1

    def test_both_connected_means_modules_ready(self, tmp_path: Path) -> None:
        patches, _ = self._mock_modules()
        with patches["create_capture"], patches["create_protocol"], \
             patches["ActionExecutor"], patches["BluetoothExecutor"], \
             patches["VisionAgent"]:
            ctrl = AppController(config_path=str(tmp_path / "cfg.yaml"))
            ctrl.load_config()

            events: list[AppEvent] = []
            ctrl.add_listener(events.append)

            ctrl.init_capture()
            ctrl.init_executor()

            assert ctrl.modules_ready is True

            # MODULES_READY should be emitted when the second one connects
            modules_events = [e for e in events if e.type == AppEventType.MODULES_READY]
            assert len(modules_events) == 1

    def test_close_capture_while_executor_open(self, tmp_path: Path) -> None:
        patches, _ = self._mock_modules()
        with patches["create_capture"], patches["create_protocol"], \
             patches["ActionExecutor"], patches["BluetoothExecutor"], \
             patches["VisionAgent"]:
            ctrl = AppController(config_path=str(tmp_path / "cfg.yaml"))
            ctrl.load_config()

            ctrl.init_capture()
            ctrl.init_executor()
            assert ctrl.modules_ready is True

            events: list[AppEvent] = []
            ctrl.add_listener(events.append)

            ctrl.close_capture()
            assert ctrl.capture_ready is False
            assert ctrl.executor_ready is True  # executor intact
            assert ctrl.modules_ready is False

            closed_events = [e for e in events if e.type == AppEventType.CAPTURE_CLOSED]
            assert len(closed_events) == 1

            # MODULES_CLOSED should fire since we went from ready to not ready
            modules_closed = [e for e in events if e.type == AppEventType.MODULES_CLOSED]
            assert len(modules_closed) == 1

    def test_close_executor_while_capture_open(self, tmp_path: Path) -> None:
        patches, _ = self._mock_modules()
        with patches["create_capture"], patches["create_protocol"], \
             patches["ActionExecutor"], patches["BluetoothExecutor"], \
             patches["VisionAgent"]:
            ctrl = AppController(config_path=str(tmp_path / "cfg.yaml"))
            ctrl.load_config()

            ctrl.init_capture()
            ctrl.init_executor()
            assert ctrl.modules_ready is True

            events: list[AppEvent] = []
            ctrl.add_listener(events.append)

            ctrl.close_executor()
            assert ctrl.executor_ready is False
            assert ctrl.capture_ready is True  # capture intact
            assert ctrl.modules_ready is False

    def test_close_capture_clears_agent(self, tmp_path: Path) -> None:
        """Closing capture should clear the agent since it holds a reference."""
        patches, _ = self._mock_modules()
        with patches["create_capture"], patches["create_protocol"], \
             patches["ActionExecutor"], patches["BluetoothExecutor"], \
             patches["VisionAgent"]:
            ctrl = AppController(config_path=str(tmp_path / "cfg.yaml"))
            ctrl.load_config()
            ctrl.init_capture()
            ctrl.init_executor()

            # start_task would create agent; simulate by calling _ensure_agent
            ctrl._ensure_agent()
            assert ctrl._agent is not None

            ctrl.close_capture()
            assert ctrl._agent is None

    def test_ensure_agent_passes_skill_text(self, tmp_path: Path) -> None:
        """_ensure_agent should load skills and pass skill_text to create_protocol."""
        patches, _ = self._mock_modules()
        mock_create_protocol = MagicMock(return_value=MagicMock())
        patches["create_protocol"] = patch(
            "ui.app_controller.create_protocol",
            mock_create_protocol,
        )
        with patches["create_capture"], patches["create_protocol"], \
             patches["ActionExecutor"], patches["BluetoothExecutor"], \
             patches["VisionAgent"], \
             patch("agent.skills.load_skills", return_value="# Blender\nShortcuts") as mock_load:
            ctrl = AppController(config_path=str(tmp_path / "cfg.yaml"))
            ctrl.load_config()
            ctrl.get_config().agent.skills = ["blender"]
            ctrl.init_capture()
            ctrl.init_executor()

            ctrl._ensure_agent()

            mock_load.assert_called_once_with(["blender"])
            _, kwargs = mock_create_protocol.call_args
            assert kwargs["skill_text"] == "# Blender\nShortcuts"

    def test_ensure_agent_skill_load_failure_raises(self, tmp_path: Path) -> None:
        """_ensure_agent should raise TaskError when skill loading fails."""
        patches, _ = self._mock_modules()
        with patches["create_capture"], patches["create_protocol"], \
             patches["ActionExecutor"], patches["BluetoothExecutor"], \
             patches["VisionAgent"], \
             patch("agent.skills.load_skills", side_effect=FileNotFoundError("not found")):
            ctrl = AppController(config_path=str(tmp_path / "cfg.yaml"))
            ctrl.load_config()
            ctrl.get_config().agent.skills = ["missing"]
            ctrl.init_capture()
            ctrl.init_executor()

            with pytest.raises(TaskError, match="Failed to load skills"):
                ctrl._ensure_agent()

    def test_init_capture_returns_test_frame(self, tmp_path: Path) -> None:
        patches, _ = self._mock_modules()
        with patches["create_capture"], patches["create_protocol"], \
             patches["ActionExecutor"], patches["BluetoothExecutor"], \
             patches["VisionAgent"]:
            ctrl = AppController(config_path=str(tmp_path / "cfg.yaml"))
            ctrl.load_config()
            result = ctrl.init_capture()
            assert result is not None
            assert result.width == 1280
            assert result.base64_jpeg == "abc"

    def test_start_task_creates_agent_on_demand(self, tmp_path: Path) -> None:
        """start_task should create agent lazily when both modules ready."""
        from agent.vision_agent import TaskResult, TaskStatus

        mock_result = TaskResult(
            status=TaskStatus.COMPLETED, reason="done",
            total_steps=1, total_input_tokens=10,
            total_output_tokens=5, total_duration_s=1.0,
        )
        mock_agent = MagicMock()
        mock_agent.run.return_value = mock_result

        patches, _ = self._mock_modules()
        patches["VisionAgent"] = patch(
            "ui.app_controller.VisionAgent",
            return_value=mock_agent,
        )
        with patches["create_capture"], patches["create_protocol"], \
             patches["ActionExecutor"], patches["BluetoothExecutor"], \
             patches["VisionAgent"]:
            ctrl = AppController(config_path=str(tmp_path / "cfg.yaml"))
            ctrl.load_config()
            ctrl.init_capture()
            ctrl.init_executor()
            assert ctrl._agent is None  # not yet created

            ctrl.start_task("Test")

            # Wait for task thread
            for _ in range(50):
                if not ctrl.task_running:
                    break
                time.sleep(0.05)

            assert ctrl._agent is not None
            mock_agent.run.assert_called_once()

    def test_start_task_fails_without_both(self, tmp_path: Path) -> None:
        patches, _ = self._mock_modules()
        with patches["create_capture"], patches["create_protocol"], \
             patches["ActionExecutor"], patches["BluetoothExecutor"], \
             patches["VisionAgent"]:
            ctrl = AppController(config_path=str(tmp_path / "cfg.yaml"))
            ctrl.load_config()
            ctrl.init_capture()  # only capture

            with pytest.raises(TaskError, match="Modules not initialised"):
                ctrl.start_task("Test")

    def test_capture_ready_and_executor_ready_events(self, tmp_path: Path) -> None:
        patches, _ = self._mock_modules()
        with patches["create_capture"], patches["create_protocol"], \
             patches["ActionExecutor"], patches["BluetoothExecutor"], \
             patches["VisionAgent"]:
            ctrl = AppController(config_path=str(tmp_path / "cfg.yaml"))
            ctrl.load_config()

            events: list[AppEvent] = []
            ctrl.add_listener(events.append)

            ctrl.init_capture()
            ctrl.init_executor()
            ctrl.close_executor()
            ctrl.close_capture()

            event_types = [e.type for e in events]
            assert AppEventType.CAPTURE_READY in event_types
            assert AppEventType.EXECUTOR_READY in event_types
            assert AppEventType.EXECUTOR_CLOSED in event_types
            assert AppEventType.CAPTURE_CLOSED in event_types

    def test_get_status_includes_split_fields(self, tmp_path: Path) -> None:
        patches, _ = self._mock_modules()
        with patches["create_capture"], patches["create_protocol"], \
             patches["ActionExecutor"], patches["BluetoothExecutor"], \
             patches["VisionAgent"]:
            ctrl = AppController(config_path=str(tmp_path / "cfg.yaml"))
            ctrl.load_config()

            status = ctrl.get_status()
            assert status["capture_ready"] is False
            assert status["executor_ready"] is False
            assert status["modules_ready"] is False

            ctrl.init_capture()
            status = ctrl.get_status()
            assert status["capture_ready"] is True
            assert status["executor_ready"] is False
            assert status["modules_ready"] is False


# ---------------------------------------------------------------------------
# Task control (mocked)
# ---------------------------------------------------------------------------

class TestTaskControl:
    """start_task / abort_task / get_task_status."""

    def test_start_task_requires_modules(self, tmp_path: Path) -> None:
        ctrl = AppController(config_path=str(tmp_path / "cfg.yaml"))
        ctrl.load_config()

        with pytest.raises(TaskError, match="Modules not initialised"):
            ctrl.start_task("test")

    def test_task_lifecycle(self, tmp_path: Path) -> None:
        """Start a task, wait for it to finish, verify events."""
        from agent.vision_agent import TaskResult, TaskStatus

        mock_result = TaskResult(
            status=TaskStatus.COMPLETED,
            reason="Task done",
            total_steps=3,
            total_input_tokens=100,
            total_output_tokens=50,
            total_duration_s=5.0,
        )

        mock_agent = MagicMock()
        mock_agent.run.return_value = mock_result

        mock_capture = MagicMock()

        with patch("ui.app_controller.create_capture", return_value=mock_capture), \
             patch("ui.app_controller.create_protocol", return_value=MagicMock()), \
             patch("ui.app_controller.ActionExecutor", return_value=MagicMock()), \
             patch("ui.app_controller.BluetoothExecutor", return_value=MagicMock()), \
             patch("ui.app_controller.VisionAgent", return_value=mock_agent):

            ctrl = AppController(config_path=str(tmp_path / "cfg.yaml"))
            ctrl.load_config()
            ctrl.init_modules()

            events: list[AppEvent] = []
            ctrl.add_listener(events.append)

            ctrl.start_task("Open Notepad")

            # Wait for task thread to finish
            for _ in range(50):
                if not ctrl.task_running:
                    break
                time.sleep(0.05)

            assert not ctrl.task_running

            # Check events
            started = [e for e in events if e.type == AppEventType.TASK_STARTED]
            finished = [e for e in events if e.type == AppEventType.TASK_FINISHED]
            assert len(started) == 1
            assert started[0].data["goal"] == "Open Notepad"
            assert len(finished) == 1
            assert finished[0].data["status"] == "completed"

            # Check status
            status = ctrl.get_task_status()
            assert status["status"] == "completed"
            assert status["total_steps"] == 3

            ctrl.close_modules()

    def test_cannot_start_two_tasks(self, tmp_path: Path) -> None:
        """Starting a second task while one is running should raise."""
        # Make agent.run() block for a while
        def slow_run(*args, **kwargs):
            time.sleep(2)
            from agent.vision_agent import TaskResult, TaskStatus
            return TaskResult(
                status=TaskStatus.COMPLETED, reason="done",
                total_steps=0, total_input_tokens=0,
                total_output_tokens=0, total_duration_s=0,
            )

        mock_agent = MagicMock()
        mock_agent.run.side_effect = slow_run

        with patch("ui.app_controller.create_capture", return_value=MagicMock()), \
             patch("ui.app_controller.create_protocol", return_value=MagicMock()), \
             patch("ui.app_controller.ActionExecutor", return_value=MagicMock()), \
             patch("ui.app_controller.BluetoothExecutor", return_value=MagicMock()), \
             patch("ui.app_controller.VisionAgent", return_value=mock_agent):

            ctrl = AppController(config_path=str(tmp_path / "cfg.yaml"))
            ctrl.load_config()
            ctrl.init_modules()

            ctrl.start_task("Task 1")
            time.sleep(0.1)  # Let thread start

            with pytest.raises(TaskError, match="already running"):
                ctrl.start_task("Task 2")

            ctrl.abort_task()
            ctrl.close_modules()

    def test_idle_status(self, tmp_path: Path) -> None:
        ctrl = AppController(config_path=str(tmp_path / "cfg.yaml"))
        ctrl.load_config()
        status = ctrl.get_task_status()
        assert status["status"] == "idle"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

class TestLogging:
    """Log capture integration."""

    def test_get_logs(self, tmp_path: Path) -> None:
        ctrl = AppController(config_path=str(tmp_path / "cfg.yaml"))
        ctrl.install_log_handler("test.app_ctrl_log")

        test_logger = logging.getLogger("test.app_ctrl_log")
        test_logger.setLevel(logging.DEBUG)
        test_logger.info("test message")

        logs = ctrl.get_logs()
        assert any("test message" in line for line in logs)

        ctrl.remove_log_handler("test.app_ctrl_log")

    def test_clear_logs(self, tmp_path: Path) -> None:
        ctrl = AppController(config_path=str(tmp_path / "cfg.yaml"))
        ctrl.install_log_handler("test.app_ctrl_clear")

        test_logger = logging.getLogger("test.app_ctrl_clear")
        test_logger.setLevel(logging.DEBUG)
        test_logger.info("will be cleared")

        ctrl.clear_logs()
        assert ctrl.get_logs() == []

        ctrl.remove_log_handler("test.app_ctrl_clear")

    def test_log_event_emitted(self, tmp_path: Path) -> None:
        ctrl = AppController(config_path=str(tmp_path / "cfg.yaml"))
        ctrl.install_log_handler("test.app_ctrl_event")

        events: list[AppEvent] = []
        ctrl.add_listener(events.append)

        test_logger = logging.getLogger("test.app_ctrl_event")
        test_logger.setLevel(logging.DEBUG)
        test_logger.info("logged")

        log_events = [e for e in events if e.type == AppEventType.LOG_MESSAGE]
        assert len(log_events) >= 1
        assert "logged" in log_events[0].data["message"]

        ctrl.remove_log_handler("test.app_ctrl_event")


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

class TestStatus:
    """get_status() aggregation."""

    def test_status_basic(self, tmp_path: Path) -> None:
        ctrl = AppController(config_path=str(tmp_path / "cfg.yaml"))
        ctrl.load_config()

        status = ctrl.get_status()
        assert "modules_ready" in status
        assert "capture_source" in status
        assert "task" in status
        assert status["modules_ready"] is False
