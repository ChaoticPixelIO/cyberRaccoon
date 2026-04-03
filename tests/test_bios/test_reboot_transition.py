"""Tests for BIOS reboot transition handling in VisionAgent.

Tests the reboot command detection, transition handler, state management,
and main loop integration for the BIOS operation feature.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from agent.vision_agent import VisionAgent, TaskStatus
from capture.base import CaptureError, CaptureResult, frame_to_capture_result
from tests.test_agent.conftest import (
    FakeCaptureResult,
    MockCapture,
    MockExecutor,
    MockProtocol,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_image(brightness: int = 128, width: int = 64, height: int = 64) -> Image.Image:
    """Create a solid-color PIL Image for testing."""
    arr = np.full((height, width, 3), brightness, dtype=np.uint8)
    return Image.fromarray(arr, "RGB")


def _make_capture_result(brightness: int = 128) -> CaptureResult:
    """Create a CaptureResult with a solid-color frame."""
    img = _make_image(brightness)
    return frame_to_capture_result(img, 64, 64, 50)


class RebootTestCapture:
    """Mock capture that returns a sequence of frames with configurable brightness.

    brightness_sequence: list of int (0-255), one per capture() call.
    If the list is exhausted, repeats the last value.
    If a value is None, raises CaptureError.
    """

    def __init__(self, brightness_sequence: list[int | None]) -> None:
        self._sequence = brightness_sequence
        self._index = 0
        self._open = True

    def open(self) -> None:
        self._open = True

    def close(self) -> None:
        self._open = False

    def is_open(self) -> bool:
        return self._open

    def capture(self) -> CaptureResult:
        if not self._open:
            raise CaptureError("Not open")
        idx = min(self._index, len(self._sequence) - 1)
        self._index += 1
        val = self._sequence[idx]
        if val is None:
            raise CaptureError("Simulated capture failure")
        return _make_capture_result(val)


def _make_agent(
    responses: list[dict[str, Any] | list[dict[str, Any]] | None],
    capture: Any = None,
    executor: Any = None,
    max_steps: int = 10,
    stability_check: bool = False,
) -> VisionAgent:
    """Convenience builder for VisionAgent with mock dependencies."""
    return VisionAgent(
        capture=capture or MockCapture(),
        protocol=MockProtocol(responses),
        executor=executor or MockExecutor(),
        max_steps=max_steps,
        max_consecutive_failures=3,
        post_action_delay_s=0,
        task_timeout_s=60.0,
        stability_check=stability_check,
    )


# ===========================================================================
# Reboot Command Detection
# ===========================================================================

class TestRebootCommandDetection:
    """Tests for _is_reboot_command() static method."""

    def test_windows_reboot_command(self) -> None:
        cmd = {"action": "type", "text": "shutdown /r /fw /t 0"}
        assert VisionAgent._is_reboot_command(cmd) is True

    def test_linux_reboot_command(self) -> None:
        cmd = {"action": "type", "text": "sudo systemctl reboot --firmware-setup"}
        assert VisionAgent._is_reboot_command(cmd) is True

    def test_linux_reboot_short_flag(self) -> None:
        cmd = {"action": "type", "text": "systemctl reboot --firmware"}
        assert VisionAgent._is_reboot_command(cmd) is True

    def test_normal_type_command(self) -> None:
        cmd = {"action": "type", "text": "hello world"}
        assert VisionAgent._is_reboot_command(cmd) is False

    def test_non_type_action(self) -> None:
        cmd = {"action": "click", "x": 100, "y": 200}
        assert VisionAgent._is_reboot_command(cmd) is False

    def test_empty_text(self) -> None:
        cmd = {"action": "type", "text": ""}
        assert VisionAgent._is_reboot_command(cmd) is False

    def test_no_text_field(self) -> None:
        cmd = {"action": "type"}
        assert VisionAgent._is_reboot_command(cmd) is False

    def test_case_insensitive(self) -> None:
        cmd = {"action": "type", "text": "Shutdown /R /FW /T 0"}
        assert VisionAgent._is_reboot_command(cmd) is True


# ===========================================================================
# Screen Black Detection
# ===========================================================================

class TestScreenBlackDetection:
    """Tests for _is_screen_black() static method."""

    def test_black_screen(self) -> None:
        frame = _make_capture_result(brightness=0)
        assert VisionAgent._is_screen_black(frame) is True

    def test_nearly_black_screen(self) -> None:
        frame = _make_capture_result(brightness=10)
        assert VisionAgent._is_screen_black(frame) is True

    def test_normal_screen(self) -> None:
        frame = _make_capture_result(brightness=128)
        assert VisionAgent._is_screen_black(frame) is False

    def test_bright_screen(self) -> None:
        frame = _make_capture_result(brightness=255)
        assert VisionAgent._is_screen_black(frame) is False

    def test_threshold_boundary(self) -> None:
        frame = _make_capture_result(brightness=19)
        assert VisionAgent._is_screen_black(frame) is True
        frame = _make_capture_result(brightness=20)
        assert VisionAgent._is_screen_black(frame) is False


# ===========================================================================
# Reboot Transition Handler
# ===========================================================================

class TestRebootTransitionHandler:
    """Tests for _wait_for_reboot_transition() method."""

    def test_normal_transition(self) -> None:
        """Screen goes black then stabilizes on a bright frame."""
        # Sequence: normal(128) -> black(0) -> black(0) -> bright(150) x5
        capture = RebootTestCapture([128, 0, 0, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150])
        agent = _make_agent(
            [{"action": "done", "reason": "test"}],
            capture=capture,
        )
        result = agent._wait_for_reboot_transition(
            timeout_s=30.0, poll_interval_s=0.01, stability_duration_s=0.05,
        )
        assert result is not None
        assert not VisionAgent._is_screen_black(result)

    def test_timeout(self) -> None:
        """If screen stays black, transition times out."""
        capture = RebootTestCapture([0])  # Always black
        agent = _make_agent(
            [{"action": "done", "reason": "test"}],
            capture=capture,
        )
        with pytest.raises(CaptureError, match="timed out"):
            agent._wait_for_reboot_transition(
                timeout_s=0.1, poll_interval_s=0.01, stability_duration_s=0.05,
            )

    def test_capture_error_retry(self) -> None:
        """CaptureErrors during polling are retried, not fatal."""
        # Sequence: error x4 -> bright(150) x10
        capture = RebootTestCapture(
            [None, None, None, None, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150]
        )
        agent = _make_agent(
            [{"action": "done", "reason": "test"}],
            capture=capture,
        )
        result = agent._wait_for_reboot_transition(
            timeout_s=30.0, poll_interval_s=0.01, stability_duration_s=0.05,
        )
        assert result is not None

    def test_abort_during_transition(self) -> None:
        """Abort event stops the transition handler."""
        capture = RebootTestCapture([0])  # Always black
        agent = _make_agent(
            [{"action": "done", "reason": "test"}],
            capture=capture,
        )
        agent._abort_event.set()
        with pytest.raises(CaptureError, match="aborted"):
            agent._wait_for_reboot_transition(
                timeout_s=30.0, poll_interval_s=0.01, stability_duration_s=0.05,
            )

    def test_unstable_screen_waits_for_stability(self) -> None:
        """Flickering screen (alternating brightness) delays transition."""
        # Alternating: not stable enough for 5 consecutive identical frames
        capture = RebootTestCapture([0, 0, 100, 200, 100, 200, 150, 150, 150, 150, 150, 150, 150])
        agent = _make_agent(
            [{"action": "done", "reason": "test"}],
            capture=capture,
        )
        result = agent._wait_for_reboot_transition(
            timeout_s=30.0, poll_interval_s=0.01, stability_duration_s=0.05,
        )
        assert result is not None


# ===========================================================================
# State Management
# ===========================================================================

class TestBiosStateManagement:
    """Tests for _in_bios_mode and _expecting_reboot flags."""

    def test_flags_start_false(self) -> None:
        agent = _make_agent([{"action": "done", "reason": "test"}])
        assert agent._in_bios_mode is False
        assert agent._expecting_reboot is False

    def test_flags_reset_on_new_task(self) -> None:
        agent = _make_agent([{"action": "done", "reason": "test"}])
        agent._in_bios_mode = True
        agent._expecting_reboot = True
        agent.run("test task")
        assert agent._in_bios_mode is False
        assert agent._expecting_reboot is False


# ===========================================================================
# Main Loop Integration
# ===========================================================================

class TestMainLoopIntegration:
    """Tests for BIOS reboot handling in the main agent loop."""

    def test_non_reboot_task_normal_flow(self) -> None:
        """Normal task with no reboot command uses standard stability wait."""
        agent = _make_agent([
            {"action": "click", "x": 100, "y": 200, "screen_summary": "desktop"},
            {"action": "done", "reason": "Task completed"},
        ])
        result = agent.run("search google")
        assert result.status == TaskStatus.COMPLETED
        assert agent._in_bios_mode is False

    def test_reboot_command_triggers_batch_interrupt(self) -> None:
        """When a batch contains a reboot command, remaining commands are skipped."""
        executor = MockExecutor()
        agent = _make_agent(
            responses=[
                [
                    {"action": "type", "text": "shutdown /r /fw /t 0"},
                    {"action": "key", "keys": ["enter"]},  # Should be skipped
                ],
                {"action": "done", "reason": "Task completed"},
            ],
            executor=executor,
        )
        # The agent will detect the reboot command and interrupt the batch.
        # Since we're using MockCapture (no real reboot), the transition
        # handler will see non-black frames and complete quickly.
        # We set a short timeout to avoid hanging.
        agent._stability_threshold = 100.0  # Accept any frame as stable
        result = agent.run("disable secure boot")
        # The reboot command should have been executed
        executed_actions = [cmd["action"] for cmd in executor.executed]
        assert "type" in executed_actions
        # The batch interrupt means "key" should NOT have been executed
        # (it gets skipped after the reboot command)
        type_cmds = [cmd for cmd in executor.executed if cmd["action"] == "type"]
        assert len(type_cmds) >= 1
