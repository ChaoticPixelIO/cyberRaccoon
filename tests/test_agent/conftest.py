"""Shared fixtures and mocks for M2 Vision Agent tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None  # type: ignore[assignment,misc]

from cyberraccoon.agent.protocols.base import ComputerUseProtocol, StepResult
from cyberraccoon.capture.screen_capture import CaptureError


# ---------------------------------------------------------------------------
# Test image helpers
# ---------------------------------------------------------------------------

def make_test_image(
    color: tuple[int, int, int] = (128, 128, 128),
    size: tuple[int, int] = (4, 4),
) -> Image.Image:
    """Create a solid-color PIL Image for frame-diff testing.

    Args:
        color: RGB tuple (default mid-gray).
        size:  (width, height) in pixels (default 4x4).

    Returns:
        A PIL Image in RGB mode filled with *color*.
    """
    if Image is None:  # pragma: no cover
        raise RuntimeError("Pillow is required for make_test_image")
    return Image.new("RGB", size, color)


# ---------------------------------------------------------------------------
# Mock M1 (Screen Capture)
# ---------------------------------------------------------------------------

@dataclass
class FakeCaptureResult:
    """Lightweight stand-in for CaptureResult.

    The *image* field defaults to a small solid-gray PIL Image so that
    ``compute_frame_diff`` works out-of-the-box in tests that need it.
    Existing tests that never inspect *image* are unaffected.
    """

    base64_jpeg: str = "fake_base64_data"
    width: int = 1280
    height: int = 720
    timestamp: float = 0.0
    size_bytes: int = 1000
    image: Any = field(default_factory=lambda: make_test_image())


class MockCapture:
    """Mock M1 that returns fake screenshots.

    Args:
        fail_at_step: Raise ``CaptureError`` on this call number (-1 = never).
        images: Optional list of PIL Images. Each ``capture()`` call pops the
            next image from the list (cycling the last one). When *None*,
            every call returns a default solid-gray image.
    """

    def __init__(
        self,
        fail_at_step: int = -1,
        images: list[Any] | None = None,
    ) -> None:
        self.call_count = 0
        self.fail_at_step = fail_at_step
        self._images = list(images) if images else None
        self._image_index = 0

    def capture(self) -> FakeCaptureResult:
        self.call_count += 1
        if self.call_count == self.fail_at_step:
            raise CaptureError("Mock capture failure")

        img: Any = None
        if self._images:
            idx = min(self._image_index, len(self._images) - 1)
            img = self._images[idx]
            self._image_index += 1

        return FakeCaptureResult(
            timestamp=self.call_count,
            image=img if img is not None else make_test_image(),
        )


# ---------------------------------------------------------------------------
# Mock M3 (Protocol)
# ---------------------------------------------------------------------------

class MockProtocol(ComputerUseProtocol):
    """Mock protocol that returns pre-scripted StepResult responses.

    Pass a list of dicts representing commands. Use None to simulate failure.
    A dict with ``{"action": "done", "reason": "..."}`` produces is_done=True.
    The last response is repeated if more calls are made than responses.
    """

    def __init__(
        self,
        responses: list[dict[str, Any] | None],
        input_tokens: int = 100,
        output_tokens: int = 10,
        detect_os_result: str | None = None,
    ) -> None:
        self.responses = responses
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.call_count = 0
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self.report_result_calls: list[tuple[bool, str | None]] = []
        self._detect_os_result = detect_os_result
        self.detect_os_calls: list[str] = []

    def step(self, screenshot_base64: str, task_goal: str) -> StepResult:
        idx = min(self.call_count, len(self.responses) - 1)
        self.call_count += 1
        cmd = self.responses[idx]

        self._total_input_tokens += self.input_tokens
        self._total_output_tokens += self.output_tokens

        if cmd is None:
            return StepResult(
                command=None,
                is_done=False,
                done_reason="",
                screen_summary="",
                raw_text="error",
                input_tokens=self.input_tokens,
                output_tokens=self.output_tokens,
                latency_ms=500,
                success=False,
                error="Mock protocol failure",
            )

        # Support batch responses: if cmd is a list, return as commands
        if isinstance(cmd, list):
            # Check if last item is done
            is_done = bool(cmd) and cmd[-1].get("action") == "done"
            done_reason = cmd[-1].get("reason", "Task completed") if is_done else ""
            completion_status = cmd[-1].get("status", "success") if is_done else "success"
            action_cmds = cmd[:-1] if is_done else cmd
            screen_summary = cmd[-1].get("screen_summary", "") if cmd else ""

            return StepResult(
                command=action_cmds[0] if action_cmds else None,
                is_done=is_done,
                done_reason=done_reason,
                screen_summary=screen_summary,
                raw_text=str(cmd),
                input_tokens=self.input_tokens,
                output_tokens=self.output_tokens,
                latency_ms=500,
                success=True,
                commands=action_cmds,
                completion_status=completion_status,
            )

        is_done = cmd.get("action") == "done"
        done_reason = cmd.get("reason", "Task completed") if is_done else ""
        completion_status = cmd.get("status", "success") if is_done else "success"
        screen_summary = cmd.get("screen_summary", "")

        return StepResult(
            command=None if is_done else cmd,
            is_done=is_done,
            done_reason=done_reason,
            screen_summary=screen_summary,
            raw_text=str(cmd),
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            latency_ms=500,
            success=True,
            completion_status=completion_status,
        )

    def report_result(self, success: bool, error: str | None = None) -> None:
        self.report_result_calls.append((success, error))

    def report_results(
        self, results: list[tuple[bool, str | None]],
    ) -> None:
        self.report_result_calls.extend(results)

    def reset(self) -> None:
        self.call_count = 0
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self.report_result_calls.clear()

    def get_usage_summary(self) -> dict[str, int]:
        return {
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
        }

    def detect_os(self, screenshot_base64: str) -> str | None:
        self.detect_os_calls.append(screenshot_base64)
        return self._detect_os_result


# ---------------------------------------------------------------------------
# Mock M4 (Action Executor)
# ---------------------------------------------------------------------------

class FailAtIndexExecutor:
    """Mock executor that fails on a specific command index within a batch.

    Used for testing batch execution failure handling.
    """

    def __init__(self, fail_at: int = 1) -> None:
        self.executed: list[dict[str, Any]] = []
        self.fail_at = fail_at

    def execute(self, command: dict[str, Any]) -> dict[str, Any]:
        idx = len(self.executed)
        self.executed.append(command)
        if idx == self.fail_at:
            return {
                "id": command.get("id"),
                "status": "error",
                "action": command["action"],
                "error": f"Mock exec failure at index {idx}",
            }
        return {
            "id": command.get("id"),
            "status": "ok",
            "action": command["action"],
            "duration_ms": 50,
        }


class MockExecutor:
    """Mock M4 that records executed commands.

    Set fail_actions to simulate execution failures for specific actions.
    """

    def __init__(self, fail_actions: set[str] | None = None) -> None:
        self.executed: list[dict[str, Any]] = []
        self.fail_actions = fail_actions or set()

    def execute(self, command: dict[str, Any]) -> dict[str, Any]:
        self.executed.append(command)
        if command.get("action") in self.fail_actions:
            return {
                "id": command.get("id"),
                "status": "error",
                "action": command["action"],
                "error": "Mock exec failure",
            }
        return {
            "id": command.get("id"),
            "status": "ok",
            "action": command["action"],
            "duration_ms": 50,
        }
