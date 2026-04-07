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

    def test_ensure_agent_loads_skills_but_not_in_protocol(self, tmp_path: Path) -> None:
        """_ensure_agent loads skills but does NOT pass them to create_protocol.

        Skills go to the planner (plan-then-execute mode), not the executor's
        system prompt. Putting skills in the tool-enabled prompt causes the
        LLM to ignore them.
        """
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
            # skill_text should NOT be passed to the protocol —
            # skills are handled by the planner, not the executor
            _, kwargs = mock_create_protocol.call_args
            assert kwargs.get("skill_text") is None

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
        # Phase 4 (HIGH-1): AppController now forces workflow mode for all
        # tasks so the approval gate protects every task. This test mocks
        # run_workflow instead of run accordingly.
        from agent.workflow_runner import WorkflowResult

        mock_workflow_result = WorkflowResult(
            status="completed", reason="done",
            steps_completed=1, steps_total=1,
            total_duration_s=1.0,
        )
        mock_agent = MagicMock()
        mock_agent.run_workflow.return_value = mock_workflow_result

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
            mock_agent.run_workflow.assert_called_once()

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
        # Phase 4 (HIGH-1): AppController forces workflow mode so mock
        # run_workflow instead of run.
        from agent.workflow_runner import WorkflowResult

        mock_workflow_result = WorkflowResult(
            status="completed",
            reason="Task done",
            steps_completed=3,
            steps_total=3,
            total_duration_s=5.0,
        )

        mock_agent = MagicMock()
        mock_agent.run_workflow.return_value = mock_workflow_result

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
        # Phase 4 (HIGH-1): workflow mode forced — slow run_workflow.
        def slow_run_workflow(*args, **kwargs):
            time.sleep(2)
            from agent.workflow_runner import WorkflowResult
            return WorkflowResult(
                status="completed", reason="done",
                steps_completed=0, steps_total=0,
                total_duration_s=0,
            )

        mock_agent = MagicMock()
        mock_agent.run_workflow.side_effect = slow_run_workflow

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


# ---------------------------------------------------------------------------
# TestPlanDiscussion (Phase 4 — DISCUSS-02 + DISCUSS-05)
# ---------------------------------------------------------------------------

try:
    from ui.app_controller import PlanDiscussionState  # added in plan 04
    _PD_AVAILABLE = True
except ImportError:
    _PD_AVAILABLE = False

# Phase 5 — plan modification is added in plan 05-04.
try:
    # These symbols are added in plan 05-04 (PlanDiscussionState extension + methods)
    # and 05-02 (RewriteResult).
    from agent.planner import RewriteResult, PlanStep  # plan 05-02 / existing
    # Probe: does PlanDiscussionState have the Phase 5 fields yet?
    from ui.app_controller import PlanDiscussionState as _PDS
    import dataclasses as _dc
    _pds_field_names = {f.name for f in _dc.fields(_PDS)}
    _PLAN_MOD_AVAILABLE = (
        "previous_steps" in _pds_field_names
        and "pending_rewrite" in _pds_field_names
        and "edited_step_numbers" in _pds_field_names
    )
except (ImportError, TypeError):
    _PLAN_MOD_AVAILABLE = False


@pytest.mark.skipif(
    not _PD_AVAILABLE,
    reason="PlanDiscussionState not yet implemented (plan 04)",
)
class TestPlanDiscussion:
    """Tests for AppController plan discussion state and chat method."""

    def _ctrl(self, tmp_path: Path) -> AppController:
        c = AppController(config_path=str(tmp_path / "cfg.yaml"))
        c.load_config()
        return c

    def _plan_ready_payload(self) -> dict[str, Any]:
        return {
            "type": "plan_ready",
            "task_goal": "Search for weather",
            "screenshot_base64": "fake_screenshot_b64",
            "steps": [
                {
                    "number": 1,
                    "goal": "Open Chrome",
                    "reboot_expected": False,
                    "expected_actions": 2,
                    "expected_outcome": "Chrome visible",
                },
            ],
        }

    def test_cache_populated_on_plan_ready(self, tmp_path: Path) -> None:
        ctrl = self._ctrl(tmp_path)
        # Simulate a workflow event coming through the on_step bridge
        ctrl._on_step_bridge(self._plan_ready_payload())
        assert ctrl._plan_discussion is not None
        assert ctrl._plan_discussion.task_goal == "Search for weather"
        assert len(ctrl._plan_discussion.steps) == 1
        assert ctrl._plan_discussion.screenshot_base64 == "fake_screenshot_b64"
        assert ctrl._plan_discussion.chat_history == []

    def test_cache_cleared_on_finish(self, tmp_path: Path) -> None:
        ctrl = self._ctrl(tmp_path)
        ctrl._on_step_bridge(self._plan_ready_payload())
        assert ctrl._plan_discussion is not None
        # Emit task_finished via the internal event dispatcher
        ctrl._emit(AppEvent(
            type=AppEventType.TASK_FINISHED,
            data={"status": "completed"},
        ))
        # Give the on-finish hook a chance to run
        assert ctrl._plan_discussion is None

    def test_chat_uses_cached_context(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        ctrl = self._ctrl(tmp_path)
        ctrl._on_step_bridge(self._plan_ready_payload())

        captured: dict[str, Any] = {}

        class FakePlanner:
            def chat_about_plan(
                self,
                *,
                task_goal: str,
                steps: list,
                screenshot_base64: str,
                history: list,
                new_question: str,
                skill_text: str | None = None,
            ) -> str | None:
                captured["task_goal"] = task_goal
                captured["steps"] = steps
                captured["screenshot_base64"] = screenshot_base64
                captured["history"] = list(history)
                captured["new_question"] = new_question
                return "Because Chrome is the primary browser."

        # Force the controller's internal planner factory to return FakePlanner.
        # Plan 04 decides the exact wiring; these two hook paths are both
        # attempted so whichever one lands can be observed by this test.
        monkeypatch.setattr(
            "ui.app_controller._build_planner_for_chat",
            lambda cfg: FakePlanner(),
            raising=False,
        )
        # Alternate attachment path — if plan 04 stores the planner differently,
        # this test may need a follow-up nudge. Keeping both paths documented.
        ctrl._chat_planner_factory = lambda: FakePlanner()  # type: ignore[attr-defined]

        answer = ctrl.chat_about_plan("Why step 1?")
        assert answer == "Because Chrome is the primary browser."
        assert captured["task_goal"] == "Search for weather"
        assert captured["screenshot_base64"] == "fake_screenshot_b64"
        assert captured["history"] == []  # first turn
        assert "Why step 1?" in captured["new_question"]

        # Chat history now has 2 entries (user Q + assistant A)
        assert ctrl._plan_discussion is not None
        assert len(ctrl._plan_discussion.chat_history) == 2
        assert ctrl._plan_discussion.chat_history[0]["role"] == "user"
        assert ctrl._plan_discussion.chat_history[1]["role"] == "assistant"

    def test_chat_returns_none_when_no_plan_cached(self, tmp_path: Path) -> None:
        ctrl = self._ctrl(tmp_path)
        assert ctrl._plan_discussion is None
        result = ctrl.chat_about_plan("any question")
        assert result is None, (
            "chat_about_plan MUST return None when no plan is cached"
        )

    def test_reject_ends_with_aborted(self, tmp_path: Path) -> None:
        ctrl = self._ctrl(tmp_path)
        ctrl._on_step_bridge(self._plan_ready_payload())
        # Simulate reject → workflow task_finished with aborted status
        ctrl._emit(AppEvent(
            type=AppEventType.TASK_FINISHED,
            data={"status": "aborted", "reason": "Plan rejected by user"},
        ))
        # After aborted, the cached plan state is cleared
        assert ctrl._plan_discussion is None

    def test_skillless_task_completes_via_workflow_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """[REVIEWS: HIGH-1] Skillless task MUST go through workflow path.

        Forces an empty skills list and asserts the plan_ready event still
        reaches _on_step_bridge (so the approval gate engages) and the
        agent's run_workflow is invoked (not the direct run() path).
        This proves the AppController layer does not skip WorkflowRunner
        when skills are empty.
        """
        from agent.workflow_runner import WorkflowResult

        ctrl = self._ctrl(tmp_path)
        # Force config to have no skills — the regression scenario
        cfg = ctrl.get_config()
        cfg.agent.skills = []

        # Track events that the bridge emits
        events_received: list[dict[str, Any]] = []

        def listener(ev: AppEvent) -> None:
            if ev.type == AppEventType.WORKFLOW_EVENT:
                events_received.append(dict(ev.data))

        ctrl.add_listener(listener)

        # Inject a fake agent whose run_workflow emits plan_ready then
        # returns a completed result without any real HID output.
        class FakeAgent:
            def __init__(self) -> None:
                self.run_workflow_called = False
                self.executor_calls: list[Any] = []

            def run_workflow(
                self, *, task_goal: str, planner: Any,
                skill_text: str | None, on_step: Any,
            ) -> WorkflowResult:
                self.run_workflow_called = True
                # Emit plan_ready exactly as WorkflowRunner would
                on_step({
                    "type": "plan_ready",
                    "task_goal": task_goal,
                    "screenshot_base64": "fake_b64",
                    "steps": [{
                        "number": 1,
                        "goal": task_goal,
                        "reboot_expected": False,
                        "expected_actions": 1,
                        "expected_outcome": "",
                    }],
                })
                # Skillless fallback: single-step completion
                return WorkflowResult(
                    status="completed",
                    reason="",
                    steps_completed=1,
                    steps_total=1,
                    total_duration_s=0.1,
                )

        fake_agent = FakeAgent()
        with ctrl._lock:
            ctrl._agent = fake_agent  # type: ignore[assignment]

        # Stub the TaskPlanner so no real LLM call happens
        from agent import planner as planner_mod
        monkeypatch.setattr(
            planner_mod, "TaskPlanner",
            lambda **kw: object(),  # placeholder, FakeAgent doesn't use it
        )

        # Run the task synchronously (bypass the worker thread)
        ctrl._run_task("generic skillless task")

        # Assertions
        assert fake_agent.run_workflow_called, (
            "[REVIEWS: HIGH-1] FakeAgent.run_workflow MUST be called — "
            "skillless task bypassed the workflow path"
        )
        plan_ready_events = [
            e for e in events_received if e.get("type") == "plan_ready"
        ]
        assert len(plan_ready_events) == 1, (
            "[REVIEWS: HIGH-1] plan_ready MUST fire for skillless task"
        )
        # Note: the approval gate itself lives in WorkflowRunner, not the
        # FakeAgent; the real end-to-end gate is covered by
        # tests/test_agent/test_workflow_runner.py::TestWorkflowApprovalGate.
        # This test's contribution is proving the AppController layer
        # does not skip WorkflowRunner when skills are empty.

    def test_lifecycle_cleared_on_exception(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """[REVIEWS: MEDIUM-1] Finally block clears state even on exception.

        Pre-populates _current_skill_text and _plan_discussion, injects an
        agent that raises mid-run, then asserts both are None after
        _run_task returns.
        """
        from agent.planner import PlanStep

        ctrl = self._ctrl(tmp_path)
        # Pre-populate cache state
        with ctrl._lock:
            ctrl._current_skill_text = "pre-existing skill text"
            ctrl._plan_discussion = PlanDiscussionState(
                task_goal="prior",
                steps=[PlanStep(number=1, goal="prior")],
                screenshot_base64="prior_b64",
            )

        # Inject an agent that raises on run_workflow
        class RaisingAgent:
            def run_workflow(self, **kw: Any) -> Any:
                raise RuntimeError("boom")

        with ctrl._lock:
            ctrl._agent = RaisingAgent()  # type: ignore[assignment]

        from agent import planner as planner_mod
        monkeypatch.setattr(
            planner_mod, "TaskPlanner", lambda **kw: object(),
        )

        # Run — the exception is caught internally, TASK_FINISHED(failed)
        # is emitted, and the finally block clears state.
        ctrl._run_task("boom task")

        with ctrl._lock:
            assert ctrl._current_skill_text is None, (
                "[REVIEWS: MEDIUM-1] _current_skill_text MUST be cleared "
                "in finally after exception"
            )
            assert ctrl._plan_discussion is None, (
                "[REVIEWS: MEDIUM-1] _plan_discussion MUST be cleared in "
                "finally after exception"
            )


# ===========================================================================
# Phase 5: plan modification (DISCUSS-03, DISCUSS-04, DISCUSS-06)
# ===========================================================================


@pytest.mark.skipif(
    not _PLAN_MOD_AVAILABLE,
    reason="Plan modification state not yet implemented (plan 05-04)",
)
class TestPlanModification:
    """Tests for AppController plan modification methods (Phase 5)."""

    def _ctrl(self, tmp_path: Path) -> AppController:
        c = AppController(config_path=str(tmp_path / "cfg.yaml"))
        c.load_config()
        return c

    def _plan_ready_payload(self) -> dict[str, Any]:
        return {
            "type": "plan_ready",
            "task_goal": "Open Notepad",
            "screenshot_base64": "fake_b64",
            "steps": [
                {"number": 1, "goal": "Open Start menu",
                 "reboot_expected": False, "expected_actions": 1,
                 "expected_outcome": "Start menu visible"},
                {"number": 2, "goal": "Type notepad",
                 "reboot_expected": False, "expected_actions": 2,
                 "expected_outcome": "Notepad highlighted"},
                {"number": 3, "goal": "Press Enter",
                 "reboot_expected": False, "expected_actions": 1,
                 "expected_outcome": "Notepad opens"},
            ],
        }

    def _seed(self, ctrl: AppController) -> None:
        ctrl._on_step_bridge(self._plan_ready_payload())

    def _make_fake_planner(
        self, rewrite_result: Any,
    ) -> Any:
        class FakePlanner:
            def rewrite_plan(
                self, *, task_goal, current_steps, screenshot_base64,
                modification_request, skill_text=None,
                completed_step_numbers=None,
            ):
                return rewrite_result
        return FakePlanner()

    def test_request_rewrite_populates_pending_not_steps(
        self, tmp_path: Path,
    ) -> None:
        ctrl = self._ctrl(tmp_path)
        self._seed(ctrl)
        # Snapshot original steps for later comparison
        original_goals = [s.goal for s in ctrl._plan_discussion.steps]
        fake_new_steps = [
            PlanStep(number=1, goal="Press Win+R",
                     expected_actions=1, expected_outcome="Run dialog"),
            PlanStep(number=2, goal="Type notepad and Enter",
                     expected_actions=2, expected_outcome="Notepad opens"),
        ]
        fake_result = RewriteResult(
            action="rewrite",
            steps=fake_new_steps,
            summary="use Win+R shortcut",
        )
        ctrl._chat_planner_factory = (
            lambda: self._make_fake_planner(fake_result)
        )
        result = ctrl.request_plan_rewrite("use Win+R")
        assert result is not None
        assert result.action == "rewrite"
        # Steps UNCHANGED, pending_rewrite holds the new list
        assert [s.goal for s in ctrl._plan_discussion.steps] == original_goals
        assert ctrl._plan_discussion.pending_rewrite is not None
        assert len(ctrl._plan_discussion.pending_rewrite) == 2
        assert ctrl._plan_discussion.pending_rewrite[0].goal == "Press Win+R"

    def test_request_rewrite_no_change_leaves_state_untouched(
        self, tmp_path: Path,
    ) -> None:
        ctrl = self._ctrl(tmp_path)
        self._seed(ctrl)
        original_goals = [s.goal for s in ctrl._plan_discussion.steps]
        fake_result = RewriteResult(
            action="no_change",
            message="Your input is a question, not a change request.",
        )
        ctrl._chat_planner_factory = (
            lambda: self._make_fake_planner(fake_result)
        )
        result = ctrl.request_plan_rewrite("why step 1?")
        assert result is not None
        assert result.action == "no_change"
        # Nothing changed
        assert [s.goal for s in ctrl._plan_discussion.steps] == original_goals
        assert ctrl._plan_discussion.pending_rewrite is None

    def test_accept_rewrite_commits_pending_and_pushes_previous(
        self, tmp_path: Path,
    ) -> None:
        ctrl = self._ctrl(tmp_path)
        self._seed(ctrl)
        original_goals = [s.goal for s in ctrl._plan_discussion.steps]
        new_steps = [
            PlanStep(number=1, goal="Press Win+R",
                     expected_actions=1, expected_outcome="Run dialog"),
        ]
        fake_result = RewriteResult(
            action="rewrite", steps=new_steps, summary="shortcut",
        )
        ctrl._chat_planner_factory = (
            lambda: self._make_fake_planner(fake_result)
        )
        ctrl.request_plan_rewrite("use Win+R")
        ok = ctrl.accept_plan_rewrite()
        assert ok is True
        assert ctrl._plan_discussion.pending_rewrite is None
        assert len(ctrl._plan_discussion.steps) == 1
        assert ctrl._plan_discussion.steps[0].goal == "Press Win+R"
        assert ctrl._plan_discussion.previous_steps is not None
        assert (
            [s.goal for s in ctrl._plan_discussion.previous_steps]
            == original_goals
        )

    def test_discard_rewrite_clears_pending_no_step_mutation(
        self, tmp_path: Path,
    ) -> None:
        ctrl = self._ctrl(tmp_path)
        self._seed(ctrl)
        original_goals = [s.goal for s in ctrl._plan_discussion.steps]
        fake_result = RewriteResult(
            action="rewrite",
            steps=[PlanStep(number=1, goal="x")],
            summary="",
        )
        ctrl._chat_planner_factory = (
            lambda: self._make_fake_planner(fake_result)
        )
        ctrl.request_plan_rewrite("change")
        assert ctrl._plan_discussion.pending_rewrite is not None
        ok = ctrl.discard_plan_rewrite()
        assert ok is True
        assert ctrl._plan_discussion.pending_rewrite is None
        assert [s.goal for s in ctrl._plan_discussion.steps] == original_goals

    def test_edit_step_updates_goal_and_pushes_previous(
        self, tmp_path: Path,
    ) -> None:
        ctrl = self._ctrl(tmp_path)
        self._seed(ctrl)
        ok = ctrl.edit_plan_step(1, "Press Win+R instead")
        assert ok is True
        assert ctrl._plan_discussion.steps[0].goal == "Press Win+R instead"
        assert ctrl._plan_discussion.previous_steps is not None
        assert ctrl._plan_discussion.previous_steps[0].goal == "Open Start menu"
        assert 1 in ctrl._plan_discussion.edited_step_numbers

    def test_edit_step_returns_false_when_no_plan(
        self, tmp_path: Path,
    ) -> None:
        ctrl = self._ctrl(tmp_path)
        assert ctrl._plan_discussion is None
        assert ctrl.edit_plan_step(1, "x") is False

    def test_edit_step_returns_false_for_unknown_step(
        self, tmp_path: Path,
    ) -> None:
        ctrl = self._ctrl(tmp_path)
        self._seed(ctrl)
        assert ctrl.edit_plan_step(99, "x") is False

    def test_add_step_appends_blank_and_marks_edited(
        self, tmp_path: Path,
    ) -> None:
        ctrl = self._ctrl(tmp_path)
        self._seed(ctrl)
        ok = ctrl.add_plan_step()
        assert ok is True
        steps = ctrl._plan_discussion.steps
        assert len(steps) == 4
        assert steps[-1].number == 4
        assert steps[-1].goal == ""
        assert 4 in ctrl._plan_discussion.edited_step_numbers

    def test_delete_step_removes_and_renumbers(
        self, tmp_path: Path,
    ) -> None:
        ctrl = self._ctrl(tmp_path)
        self._seed(ctrl)
        ok = ctrl.delete_plan_step(2)
        assert ok is True
        steps = ctrl._plan_discussion.steps
        assert len(steps) == 2
        # Renumbered 1, 2 (not 1, 3)
        assert steps[0].number == 1
        assert steps[1].number == 2
        # The step that was formerly #3 is now #2
        assert steps[1].goal == "Press Enter"

    def test_delete_step_returns_false_for_unknown(
        self, tmp_path: Path,
    ) -> None:
        ctrl = self._ctrl(tmp_path)
        self._seed(ctrl)
        assert ctrl.delete_plan_step(99) is False

    def test_mutations_refused_during_preview(
        self, tmp_path: Path,
    ) -> None:
        ctrl = self._ctrl(tmp_path)
        self._seed(ctrl)
        fake_result = RewriteResult(
            action="rewrite",
            steps=[PlanStep(number=1, goal="new")],
            summary="",
        )
        ctrl._chat_planner_factory = (
            lambda: self._make_fake_planner(fake_result)
        )
        ctrl.request_plan_rewrite("change")
        assert ctrl._plan_discussion.pending_rewrite is not None
        # All mutation methods EXCEPT accept/discard refuse
        assert ctrl.edit_plan_step(1, "x") is False
        assert ctrl.add_plan_step() is False
        assert ctrl.delete_plan_step(1) is False
        # But accept still works
        assert ctrl.accept_plan_rewrite() is True

    def test_request_rewrite_refused_when_preview_active(
        self, tmp_path: Path,
    ) -> None:
        ctrl = self._ctrl(tmp_path)
        self._seed(ctrl)
        fake_result = RewriteResult(
            action="rewrite",
            steps=[PlanStep(number=1, goal="new")],
            summary="",
        )
        ctrl._chat_planner_factory = (
            lambda: self._make_fake_planner(fake_result)
        )
        ctrl.request_plan_rewrite("first change")
        assert ctrl._plan_discussion.pending_rewrite is not None
        # Second rewrite attempt while preview is active
        result = ctrl.request_plan_rewrite("second change")
        assert result is None

    def test_lifecycle_clears_modification_state(
        self, tmp_path: Path,
    ) -> None:
        """[Pitfall 2 regression] All Phase 5 fields reset on task boundary."""
        ctrl = self._ctrl(tmp_path)
        self._seed(ctrl)
        ctrl.edit_plan_step(1, "edited")
        assert ctrl._plan_discussion.previous_steps is not None
        assert ctrl._plan_discussion.edited_step_numbers == {1}
        # Emit task_finished
        ctrl._emit(AppEvent(
            type=AppEventType.TASK_FINISHED,
            data={"status": "completed"},
        ))
        # All Phase 5 state is gone because _plan_discussion itself is None
        assert ctrl._plan_discussion is None

    def test_approve_uses_current_steps_after_edit(
        self, tmp_path: Path,
    ) -> None:
        """[Pitfall 1 regression] Integration Strategy A.

        After a manual edit, approve_plan MUST push the modified step
        list into WorkflowRunner via set_current_plan() BEFORE calling
        approve_plan() on the runner (which sets the threading Event).
        """
        ctrl = self._ctrl(tmp_path)
        self._seed(ctrl)
        captured: dict[str, Any] = {"steps": None, "approved": False}

        class FakeWorkflowRunner:
            def set_current_plan(
                self, steps: list, plan_version: int = 0,
            ) -> None:
                captured["steps"] = list(steps)
            def approve_plan(self) -> None:
                captured["approved"] = True

        class FakeAgent:
            def __init__(self) -> None:
                self._workflow_runner = FakeWorkflowRunner()

        with ctrl._lock:
            ctrl._agent = FakeAgent()  # type: ignore[assignment]

        # Modify the plan manually
        ctrl.edit_plan_step(1, "Press Win+R")
        # Now approve — the MODIFIED steps must reach the runner
        ctrl.approve_plan()
        assert captured["approved"] is True
        assert captured["steps"] is not None
        assert len(captured["steps"]) == 3
        assert captured["steps"][0].goal == "Press Win+R"

    def test_plan_version_increments_on_mutation(
        self, tmp_path: Path,
    ) -> None:
        """[REVIEWS HIGH-1] plan_version counter must increment on every
        committed mutation to prevent stale-state consumption.
        """
        ctrl = self._ctrl(tmp_path)
        self._seed(ctrl)
        assert ctrl._plan_discussion.plan_version == 0
        # Manual edit increments
        ctrl.edit_plan_step(1, "new goal")
        assert ctrl._plan_discussion.plan_version == 1
        # Add increments
        ctrl.add_plan_step()
        assert ctrl._plan_discussion.plan_version == 2
        # Delete increments
        ctrl.delete_plan_step(4)
        assert ctrl._plan_discussion.plan_version == 3
        # Accept rewrite increments
        fake_result = RewriteResult(
            action="rewrite",
            steps=[PlanStep(number=1, goal="rewritten")],
            summary="r",
        )
        ctrl._chat_planner_factory = (
            lambda: self._make_fake_planner(fake_result)
        )
        ctrl.request_plan_rewrite("change")
        ctrl.accept_plan_rewrite()
        assert ctrl._plan_discussion.plan_version == 4

    def test_edited_markers_survive_renumber_after_delete(
        self, tmp_path: Path,
    ) -> None:
        """[REVIEWS HIGH-2] edited_step_numbers must remap correctly
        when a delete causes renumbering. Position-based step numbers
        are unstable identifiers; the remapping logic must track them.
        """
        ctrl = self._ctrl(tmp_path)
        self._seed(ctrl)
        # Edit step 3
        ctrl.edit_plan_step(3, "Modified press Enter")
        assert 3 in ctrl._plan_discussion.edited_step_numbers
        # Delete step 1 — old step 3 becomes step 2
        ctrl.delete_plan_step(1)
        # After renumber, the edited marker should remap to 2
        assert 2 in ctrl._plan_discussion.edited_step_numbers
        assert 3 not in ctrl._plan_discussion.edited_step_numbers

    def test_approve_pushes_plan_version_to_runner(
        self, tmp_path: Path,
    ) -> None:
        """[REVIEWS HIGH-1] Integration Strategy A must thread
        plan_version through set_current_plan for traceability.
        """
        ctrl = self._ctrl(tmp_path)
        self._seed(ctrl)
        captured: dict[str, Any] = {"steps": None, "version": None}

        class FakeWorkflowRunner:
            def set_current_plan(self, steps: list, plan_version: int = 0) -> None:
                captured["steps"] = list(steps)
                captured["version"] = plan_version
            def approve_plan(self) -> None:
                pass

        class FakeAgent:
            def __init__(self) -> None:
                self._workflow_runner = FakeWorkflowRunner()

        with ctrl._lock:
            ctrl._agent = FakeAgent()  # type: ignore[assignment]

        ctrl.edit_plan_step(1, "edited")
        ctrl.approve_plan()
        assert captured["steps"] is not None
        assert captured["version"] is not None
        assert captured["version"] >= 1


# ---------------------------------------------------------------------------
# TestPauseLifecycle (Phase 7 — CRUISE-03, CRUISE-04, CRUISE-05)
# ---------------------------------------------------------------------------

# Guard: pause methods added in plan 07-03.
_PAUSE_AVAILABLE = hasattr(AppController, "pause_task")


@pytest.mark.skipif(
    not _PAUSE_AVAILABLE,
    reason="pause_task() not yet implemented (plan 07-03)",
)
class TestPauseLifecycle:
    """Tests for AppController pause/resume/cancel (CRUISE-03, CRUISE-04, CRUISE-05)."""

    def _ctrl(self, tmp_path: Path) -> AppController:
        c = AppController(config_path=str(tmp_path / "cfg.yaml"))
        c.load_config()
        return c

    def _inject_agent(self, ctrl: AppController) -> MagicMock:
        """Wire a mock VisionAgent with a mock WorkflowRunner."""
        agent = MagicMock()
        runner = MagicMock()
        runner.resume = MagicMock()
        runner.set_current_plan = MagicMock()
        agent._workflow_runner = runner
        agent.pause = MagicMock()
        agent.abort = MagicMock()
        with ctrl._lock:
            ctrl._agent = agent
        return agent

    def _fire_task_paused(self, ctrl: AppController) -> None:
        """Simulate a task_paused workflow event through _on_step_bridge."""
        ctrl._on_step_bridge({
            "type": "task_paused",
            "steps": [
                {"number": 1, "goal": "Open Notepad", "status": "done"},
                {"number": 2, "goal": "Type Hello", "status": "done"},
                {"number": 3, "goal": "Save file", "status": "pending"},
                {"number": 4, "goal": "Close app", "status": "pending"},
            ],
            "steps_completed": 2,
            "screenshot_base64": "base64_paused_screenshot",
        })

    def test_pause_task_sets_pause_event(self, tmp_path: Path) -> None:
        """pause_task() calls agent.pause() and returns True."""
        ctrl = self._ctrl(tmp_path)
        agent = self._inject_agent(ctrl)
        result = ctrl.pause_task()
        assert result is True
        agent.pause.assert_called_once()

    def test_pause_task_uses_public_api(self, tmp_path: Path) -> None:
        """pause_task() calls agent.pause() not agent._pause_event.set() directly.
        Addresses review concern: HIGH-3 control ownership / layer leakage."""
        ctrl = self._ctrl(tmp_path)
        agent = self._inject_agent(ctrl)
        ctrl.pause_task()
        # Verify public API was called
        agent.pause.assert_called_once()
        # Verify source code does not directly access _pause_event in method body
        import inspect
        source = inspect.getsource(ctrl.pause_task)
        # Strip docstring (everything between first pair of triple-quotes)
        # to check only the implementation code
        assert "_pause_event.set()" not in source, \
            "pause_task must use agent.pause(), not _pause_event.set() directly"

    def test_pause_task_returns_false_when_no_agent(self, tmp_path: Path) -> None:
        """pause_task() returns False when no agent is running."""
        ctrl = self._ctrl(tmp_path)
        assert ctrl.pause_task() is False

    def test_resume_task_pushes_plan_and_unblocks(self, tmp_path: Path) -> None:
        """resume_task() calls set_current_plan with remaining steps + runner.resume()."""
        from agent.planner import PlanStep
        ctrl = self._ctrl(tmp_path)
        agent = self._inject_agent(ctrl)
        runner = agent._workflow_runner

        # Simulate paused state with completed steps
        self._fire_task_paused(ctrl)

        result = ctrl.resume_task()
        assert result is True

        # Verify set_current_plan was called with only remaining steps
        runner.set_current_plan.assert_called_once()
        call_args = runner.set_current_plan.call_args
        remaining_steps = call_args[0][0]
        # Steps 1 and 2 are done, so only steps 3 and 4 should be pushed
        assert len(remaining_steps) == 2
        assert remaining_steps[0].number == 3
        assert remaining_steps[1].number == 4

        # Verify resume was called
        runner.resume.assert_called_once()

    def test_resume_task_returns_false_when_no_agent(
        self, tmp_path: Path,
    ) -> None:
        """resume_task() returns False when no agent is running."""
        ctrl = self._ctrl(tmp_path)
        assert ctrl.resume_task() is False

    def test_cancel_paused_task_calls_abort(self, tmp_path: Path) -> None:
        """cancel_paused_task() delegates to abort_task."""
        ctrl = self._ctrl(tmp_path)
        agent = self._inject_agent(ctrl)
        ctrl.cancel_paused_task()
        agent.abort.assert_called_once()

    def test_plan_discussion_populated_on_pause(self, tmp_path: Path) -> None:
        """When task_paused event fires, PlanDiscussionState has remaining steps
        and screenshot."""
        ctrl = self._ctrl(tmp_path)
        self._fire_task_paused(ctrl)

        with ctrl._lock:
            state = ctrl._plan_discussion
        assert state is not None
        assert len(state.steps) == 4
        assert state.screenshot_base64 == "base64_paused_screenshot"
        assert state.completed_step_count == 2
        assert state.completed_step_numbers == {1, 2}

    def test_plan_discussion_cleared_on_task_finish(self, tmp_path: Path) -> None:
        """PlanDiscussionState is cleared when the task finishes after resume."""
        from ui.app_controller import AppEvent, AppEventType
        ctrl = self._ctrl(tmp_path)
        self._fire_task_paused(ctrl)

        # Verify state exists
        with ctrl._lock:
            assert ctrl._plan_discussion is not None

        # Fire task_finished event -- should clear state
        ctrl._emit(AppEvent(type=AppEventType.TASK_FINISHED, data={}))

        with ctrl._lock:
            assert ctrl._plan_discussion is None

    def test_completed_steps_not_editable_in_paused_state(
        self, tmp_path: Path,
    ) -> None:
        """edit_plan_step refuses to edit completed steps in paused state (CRUISE-04).
        Uses completed_step_numbers set, not index-based count.
        Addresses review concern: HIGH-2 step identity fragility."""
        ctrl = self._ctrl(tmp_path)
        self._fire_task_paused(ctrl)

        # Step 1 is completed -- should refuse edit
        assert ctrl.edit_plan_step(1, "New goal") is False
        # Step 2 is completed -- should refuse edit
        assert ctrl.edit_plan_step(2, "New goal") is False
        # Step 3 is pending -- should allow edit
        assert ctrl.edit_plan_step(3, "Save file as .txt") is True

    def test_completed_steps_not_deletable_in_paused_state(
        self, tmp_path: Path,
    ) -> None:
        """delete_plan_step refuses to delete completed steps in paused state.
        Addresses review concern: HIGH-2 step identity fragility."""
        ctrl = self._ctrl(tmp_path)
        self._fire_task_paused(ctrl)

        # Step 1 is completed -- should refuse delete
        assert ctrl.delete_plan_step(1) is False
        # Step 2 is completed -- should refuse delete
        assert ctrl.delete_plan_step(2) is False
        # Step 3 is pending -- should allow delete
        assert ctrl.delete_plan_step(3) is True

    def test_fresh_screenshot_in_paused_state(self, tmp_path: Path) -> None:
        """PlanDiscussionState.screenshot_base64 is set from task_paused event.
        Addresses review concern: HIGH-4/HIGH-5 fresh screenshot."""
        ctrl = self._ctrl(tmp_path)

        # Fire task_paused with a specific screenshot
        ctrl._on_step_bridge({
            "type": "task_paused",
            "steps": [
                {"number": 1, "goal": "Step A", "status": "done"},
            ],
            "steps_completed": 1,
            "screenshot_base64": "fresh_pause_screenshot_data",
        })

        with ctrl._lock:
            state = ctrl._plan_discussion
        assert state is not None
        assert state.screenshot_base64 == "fresh_pause_screenshot_data"


# ---------------------------------------------------------------------------
# Completed-Step Lock (Phase 8, 08-02)
# ---------------------------------------------------------------------------

# Feature-detect _validate_completed_step_lock
try:
    from ui.app_controller import _validate_completed_step_lock
    _LOCK_AVAILABLE = True
except ImportError:
    _LOCK_AVAILABLE = False


@pytest.mark.skipif(
    not _LOCK_AVAILABLE,
    reason="_validate_completed_step_lock not yet implemented (plan 08-02)",
)
class TestCompletedStepLock:
    """Tests for completed-step lock enforcement in paused-state rewrite."""

    def _ctrl(self, tmp_path: Path) -> AppController:
        c = AppController(config_path=str(tmp_path / "cfg.yaml"))
        c.load_config()
        return c

    def _fire_task_paused(
        self, ctrl: AppController,
        steps: list[dict[str, Any]] | None = None,
        completed_count: int = 2,
    ) -> None:
        """Simulate a task_paused event with completed steps."""
        if steps is None:
            steps = [
                {"number": 1, "goal": "Open Chrome", "status": "done",
                 "expected_actions": 2, "expected_outcome": "Chrome visible"},
                {"number": 2, "goal": "Navigate to site", "status": "done",
                 "expected_actions": 1, "expected_outcome": "Site loaded"},
                {"number": 3, "goal": "Click login", "status": "pending",
                 "expected_actions": 1, "expected_outcome": "Login form"},
            ]
        # Need a plan_ready first so task_goal is set
        ctrl._on_step_bridge({
            "type": "plan_ready",
            "task_goal": "Log into site",
            "screenshot_base64": "pre_exec_screenshot",
            "steps": steps,
        })
        ctrl._on_step_bridge({
            "type": "task_paused",
            "steps": steps,
            "steps_completed": completed_count,
            "screenshot_base64": "pause_screenshot",
        })

    def test_lock_passes_when_no_modifications(self) -> None:
        """Lock returns None when completed steps are preserved."""
        original = [
            PlanStep(number=1, goal="Open Chrome"),
            PlanStep(number=2, goal="Type URL"),
            PlanStep(number=3, goal="Click Go"),
        ]
        proposed = [
            PlanStep(number=1, goal="Open Chrome"),
            PlanStep(number=2, goal="Type URL"),
            PlanStep(number=3, goal="Click search instead"),
        ]
        result = _validate_completed_step_lock(original, proposed, {1})
        assert result is None

    def test_lock_rejects_modified_completed_goal(self) -> None:
        """Lock rejects when a completed step's goal is changed."""
        original = [
            PlanStep(number=1, goal="Open Chrome"),
            PlanStep(number=2, goal="Type URL"),
        ]
        proposed = [
            PlanStep(number=1, goal="Open Firefox"),  # changed!
            PlanStep(number=2, goal="Type URL"),
        ]
        result = _validate_completed_step_lock(original, proposed, {1})
        assert result is not None
        assert "Cannot modify completed steps" in result

    def test_lock_rejects_removed_completed_step(self) -> None:
        """Lock rejects when a completed step is removed."""
        original = [
            PlanStep(number=1, goal="Open Chrome"),
            PlanStep(number=2, goal="Type URL"),
            PlanStep(number=3, goal="Click Go"),
        ]
        proposed = [
            PlanStep(number=2, goal="Type URL"),
            PlanStep(number=3, goal="Click Go"),
        ]
        result = _validate_completed_step_lock(original, proposed, {1})
        assert result is not None
        assert "Cannot modify completed steps" in result

    def test_lock_rejects_renumbered_completed_step(self) -> None:
        """Lock rejects when completed steps are renumbered by insertion.
        Addresses review concern HIGH-1, edge case: renumbering."""
        original = [
            PlanStep(number=1, goal="Open Chrome"),
            PlanStep(number=2, goal="Type URL"),
            PlanStep(number=3, goal="Click Go"),
        ]
        # New step inserted at position 1, pushing old step 1 to number 2
        proposed = [
            PlanStep(number=1, goal="Launch browser first"),  # NEW step at 1
            PlanStep(number=2, goal="Open Chrome"),  # old step 1 renumbered to 2
            PlanStep(number=3, goal="Type URL"),
        ]
        # Step 1 was completed, but now position 1 has a different goal
        result = _validate_completed_step_lock(original, proposed, {1})
        assert result is not None
        assert "Cannot modify completed steps" in result

    def test_lock_passes_when_no_completed(self) -> None:
        """Lock returns None when completed_numbers is empty set."""
        original = [PlanStep(number=1, goal="Open Chrome")]
        proposed = [PlanStep(number=1, goal="Open Firefox")]
        result = _validate_completed_step_lock(original, proposed, set())
        assert result is None

    def test_lock_passes_with_empty_set(self) -> None:
        """Explicit empty set skips validation."""
        original = [PlanStep(number=1, goal="Open Chrome")]
        proposed = [PlanStep(number=1, goal="Something else")]
        result = _validate_completed_step_lock(original, proposed, set())
        assert result is None

    def test_paused_rewrite_passes_completed_numbers(
        self, tmp_path: Path,
    ) -> None:
        """request_plan_rewrite passes completed_step_numbers to planner."""
        ctrl = self._ctrl(tmp_path)
        self._fire_task_paused(ctrl)

        captured_kwargs: dict[str, Any] = {}

        class FakePlanner:
            def rewrite_plan(
                self, *, task_goal, current_steps, screenshot_base64,
                modification_request, skill_text=None,
                completed_step_numbers=None,
            ):
                captured_kwargs["completed_step_numbers"] = completed_step_numbers
                return RewriteResult(
                    action="rewrite",
                    steps=[
                        PlanStep(number=1, goal="Open Chrome"),
                        PlanStep(number=2, goal="Navigate to site"),
                        PlanStep(number=3, goal="Click login button"),
                    ],
                    summary="minor change",
                )

        ctrl._chat_planner_factory = lambda: FakePlanner()
        ctrl.request_plan_rewrite("change step 3")
        assert captured_kwargs.get("completed_step_numbers") == {1, 2}

    def test_pre_execution_rewrite_omits_completed_numbers(
        self, tmp_path: Path,
    ) -> None:
        """request_plan_rewrite passes None when no completed steps."""
        ctrl = self._ctrl(tmp_path)
        # Use plan_ready (pre-execution) -- no completed steps
        ctrl._on_step_bridge({
            "type": "plan_ready",
            "task_goal": "Open Notepad",
            "screenshot_base64": "fake_b64",
            "steps": [
                {"number": 1, "goal": "Open Start menu",
                 "expected_actions": 1, "expected_outcome": "Menu visible"},
            ],
        })

        captured_kwargs: dict[str, Any] = {}

        class FakePlanner:
            def rewrite_plan(
                self, *, task_goal, current_steps, screenshot_base64,
                modification_request, skill_text=None,
                completed_step_numbers=None,
            ):
                captured_kwargs["completed_step_numbers"] = completed_step_numbers
                return RewriteResult(
                    action="rewrite",
                    steps=[PlanStep(number=1, goal="Press Win+R")],
                    summary="shortcut",
                )

        ctrl._chat_planner_factory = lambda: FakePlanner()
        ctrl.request_plan_rewrite("use shortcut")
        assert captured_kwargs.get("completed_step_numbers") is None

    def test_paused_rewrite_rejects_modified_completed_step(
        self, tmp_path: Path,
    ) -> None:
        """Lock violation emits plan_rewrite_error and returns None."""
        ctrl = self._ctrl(tmp_path)
        self._fire_task_paused(ctrl)

        events: list[AppEvent] = []
        ctrl.add_listener(events.append)

        class FakePlanner:
            def rewrite_plan(
                self, *, task_goal, current_steps, screenshot_base64,
                modification_request, skill_text=None,
                completed_step_numbers=None,
            ):
                return RewriteResult(
                    action="rewrite",
                    steps=[
                        PlanStep(number=1, goal="Open Firefox"),  # MODIFIED completed step!
                        PlanStep(number=2, goal="Navigate to site"),
                        PlanStep(number=3, goal="New login step"),
                    ],
                    summary="changed browser",
                )

        ctrl._chat_planner_factory = lambda: FakePlanner()
        result = ctrl.request_plan_rewrite("change to Firefox")
        assert result is None

        # Should have emitted plan_rewrite_error
        error_events = [
            e for e in events
            if e.type == AppEventType.WORKFLOW_EVENT
            and e.data.get("type") == "plan_rewrite_error"
        ]
        assert len(error_events) == 1
        assert "Cannot modify completed steps" in error_events[0].data["message"]

    def test_lock_rejection_preserves_state(
        self, tmp_path: Path,
    ) -> None:
        """After lock rejection, pending_rewrite remains None.
        Addresses review concern MEDIUM: state preservation on rejection."""
        ctrl = self._ctrl(tmp_path)
        self._fire_task_paused(ctrl)

        # Verify no pending rewrite before
        assert ctrl._plan_discussion.pending_rewrite is None

        class FakePlanner:
            def rewrite_plan(
                self, *, task_goal, current_steps, screenshot_base64,
                modification_request, skill_text=None,
                completed_step_numbers=None,
            ):
                return RewriteResult(
                    action="rewrite",
                    steps=[
                        PlanStep(number=1, goal="CHANGED"),  # violates lock
                        PlanStep(number=2, goal="Navigate to site"),
                    ],
                    summary="bad rewrite",
                )

        ctrl._chat_planner_factory = lambda: FakePlanner()
        ctrl.request_plan_rewrite("change everything")

        # pending_rewrite should still be None
        assert ctrl._plan_discussion.pending_rewrite is None

    def test_accepted_rewrite_has_no_completed_markers(
        self, tmp_path: Path,
    ) -> None:
        """After accepting a paused rewrite, no steps have [COMPLETED] in goal.
        Addresses review concern MEDIUM-5: marker hygiene."""
        ctrl = self._ctrl(tmp_path)
        self._fire_task_paused(ctrl)

        class FakePlanner:
            def rewrite_plan(
                self, *, task_goal, current_steps, screenshot_base64,
                modification_request, skill_text=None,
                completed_step_numbers=None,
            ):
                return RewriteResult(
                    action="rewrite",
                    steps=[
                        PlanStep(number=1, goal="Open Chrome"),  # unchanged
                        PlanStep(number=2, goal="Navigate to site"),  # unchanged
                        PlanStep(number=3, goal="Enter credentials"),  # changed
                    ],
                    summary="updated login step",
                )

        ctrl._chat_planner_factory = lambda: FakePlanner()
        ctrl.request_plan_rewrite("update login step")

        # Accept the rewrite
        assert ctrl._plan_discussion.pending_rewrite is not None
        ctrl.accept_plan_rewrite()

        # No steps should have [COMPLETED] in their goals
        for step in ctrl._plan_discussion.steps:
            assert "[COMPLETED]" not in step.goal
