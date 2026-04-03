"""Tests for M2 Vision Agent — flow, termination, logging.

All tests use mock M1/M3/M4 (no hardware or API needed).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import numpy as np
from PIL import Image

from agent.protocols.base import StepResult
from agent.vision_agent import TaskResult, TaskStatus, VisionAgent
from capture.base import CaptureError, CaptureResult, frame_to_capture_result
from tests.test_agent.conftest import (
    FailAtIndexExecutor,
    MockCapture,
    MockExecutor,
    MockProtocol,
)


# ---------------------------------------------------------------------------
# Helper to create agent with test-friendly defaults
# ---------------------------------------------------------------------------

def _make_agent(
    responses: list[dict[str, Any] | list[dict[str, Any]] | None],
    capture: MockCapture | None = None,
    executor: MockExecutor | None = None,
    max_steps: int = 10,
    max_consecutive_failures: int = 3,
    task_timeout_s: float = 60.0,
    stability_check: bool = False,
) -> VisionAgent:
    """Convenience builder for VisionAgent with mock dependencies."""
    return VisionAgent(
        capture=capture or MockCapture(),
        protocol=MockProtocol(responses),
        executor=executor or MockExecutor(),
        max_steps=max_steps,
        max_consecutive_failures=max_consecutive_failures,
        post_action_delay_s=0,  # No sleep in tests
        task_timeout_s=task_timeout_s,
        stability_check=stability_check,
    )


# ===========================================================================
# Basic Flow
# ===========================================================================

class TestBasicFlow:
    """Tests for the normal happy path."""

    def test_simple_task_completes(self) -> None:
        """LLM returns two actions then done -> COMPLETED."""
        agent = _make_agent([
            {"action": "click", "x": 100, "y": 200, "screen_summary": "desktop"},
            {"action": "type", "text": "hello", "screen_summary": "notepad open"},
            {"action": "done", "reason": "Task completed"},
        ])
        result = agent.run("test task")

        assert result.status == TaskStatus.COMPLETED
        assert result.total_steps == 3
        assert "Task completed" in result.reason

    def test_immediate_done(self) -> None:
        """LLM says done on first step -> COMPLETED with step=1."""
        agent = _make_agent([
            {"action": "done", "reason": "Already done"},
        ])
        result = agent.run("check status")

        assert result.status == TaskStatus.COMPLETED
        assert result.total_steps == 1

    def test_done_without_reason(self) -> None:
        """Done command without reason should use default."""
        agent = _make_agent([
            {"action": "done"},
        ])
        result = agent.run("task")

        assert result.status == TaskStatus.COMPLETED
        assert "completed" in result.reason.lower()


# ===========================================================================
# Termination Conditions
# ===========================================================================

class TestTerminationConditions:
    """Tests for the four termination mechanisms."""

    def test_max_steps_exceeded(self) -> None:
        """Exceeding max_steps -> FAILED."""
        agent = _make_agent(
            [{"action": "click", "x": 100, "y": 100}],
            max_steps=3,
        )
        result = agent.run("infinite task")

        assert result.status == TaskStatus.FAILED
        assert "max steps" in result.reason.lower()
        assert result.total_steps == 3

    def test_consecutive_llm_failures_abort(self) -> None:
        """Consecutive LLM failures -> FAILED."""
        agent = _make_agent(
            [None, None, None],
            max_consecutive_failures=2,
        )
        result = agent.run("failing task")

        assert result.status == TaskStatus.FAILED
        assert "consecutive" in result.reason.lower()

    def test_consecutive_exec_failures_abort(self) -> None:
        """Consecutive executor failures -> FAILED."""
        executor = MockExecutor(fail_actions={"click"})
        agent = _make_agent(
            [{"action": "click", "x": 1, "y": 1}],
            executor=executor,
            max_consecutive_failures=2,
        )
        result = agent.run("exec fail task")

        assert result.status == TaskStatus.FAILED
        assert "consecutive" in result.reason.lower()

    def test_failure_counter_resets_on_success(self) -> None:
        """A successful step resets the consecutive failure counter."""
        agent = _make_agent(
            [
                None,  # LLM failure (count=1)
                {"action": "click", "x": 100, "y": 100},  # Success (reset to 0)
                None,  # LLM failure (count=1)
                {"action": "done", "reason": "ok"},  # Done
            ],
            max_consecutive_failures=2,
        )
        result = agent.run("flaky task")

        assert result.status == TaskStatus.COMPLETED

    def test_capture_failure_aborts(self) -> None:
        """M1 capture failure -> FAILED immediately."""
        agent = _make_agent(
            [{"action": "click", "x": 1, "y": 1}],
            capture=MockCapture(fail_at_step=1),
        )
        result = agent.run("capture fail task")

        assert result.status == TaskStatus.FAILED
        assert "capture" in result.reason.lower()

    def test_abort_requested(self) -> None:
        """Abort via on_step callback -> ABORTED after one step."""
        agent = _make_agent(
            [{"action": "click", "x": 100, "y": 100}],
        )

        def abort_after_first_action(info: dict) -> None:
            if info.get("step", 0) > 0:  # skip initial screenshot
                agent.abort()

        result = agent.run("aborted task", on_step=abort_after_first_action)

        assert result.status == TaskStatus.ABORTED
        assert result.total_steps == 1

    def test_abort_method(self) -> None:
        """abort() sets the flag correctly."""
        agent = _make_agent([{"action": "done", "reason": "ok"}])
        assert not agent._abort_event.is_set()
        agent.abort()
        assert agent._abort_event.is_set()


# ===========================================================================
# Step Log
# ===========================================================================

class TestStepLog:
    """Tests for execution logging."""

    def test_step_log_recorded(self) -> None:
        """Each step should be recorded in step_log."""
        agent = _make_agent([
            {"action": "click", "x": 100, "y": 200},
            {"action": "done", "reason": "ok"},
        ])
        result = agent.run("logged task")

        assert len(result.step_log) == 2
        assert result.step_log[0]["command"]["action"] == "click"
        assert result.step_log[0]["execute_status"] == "ok"
        assert result.step_log[1]["command"]["action"] == "done"
        assert result.step_log[1]["command"]["reason"] == "ok"
        assert result.step_log[1]["execute_status"] == "done"

    def test_failed_llm_step_logged(self) -> None:
        """LLM failure steps should also be logged."""
        agent = _make_agent([
            None,
            {"action": "done", "reason": "ok"},
        ])
        result = agent.run("task")

        assert result.step_log[0]["command"] is None
        assert result.step_log[0]["execute_status"] == "skipped"
        assert "llm_error" in result.step_log[0]

    def test_step_log_has_latency(self) -> None:
        """Step log entries should include LLM latency."""
        agent = _make_agent([
            {"action": "click", "x": 1, "y": 1},
            {"action": "done", "reason": "ok"},
        ])
        result = agent.run("task")

        for entry in result.step_log:
            assert "llm_latency_ms" in entry
            assert entry["llm_latency_ms"] > 0


# ===========================================================================
# Token Usage
# ===========================================================================

class TestTokenUsage:
    """Tests for cumulative token tracking."""

    def test_tokens_accumulated(self) -> None:
        """Token usage should accumulate across all steps."""
        agent = _make_agent([
            {"action": "click", "x": 1, "y": 1},
            {"action": "click", "x": 2, "y": 2},
            {"action": "done", "reason": "ok"},
        ])
        result = agent.run("token test")

        # MockProtocol returns 100 input + 10 output per call, 3 calls total
        assert result.total_input_tokens == 300
        assert result.total_output_tokens == 30

    def test_tokens_counted_on_failure(self) -> None:
        """Failed LLM calls should still count tokens."""
        agent = _make_agent(
            [None, {"action": "done", "reason": "ok"}],
        )
        result = agent.run("task")

        # 2 calls: 1 failed + 1 success
        assert result.total_input_tokens == 200
        assert result.total_output_tokens == 20

    def test_duration_recorded(self) -> None:
        """Total duration should be positive."""
        agent = _make_agent([
            {"action": "done", "reason": "ok"},
        ])
        result = agent.run("quick task")

        assert result.total_duration_s >= 0


# ===========================================================================
# Executor Integration
# ===========================================================================

class TestExecutorIntegration:
    """Tests for M4 executor interaction."""

    def test_commands_forwarded_to_executor(self) -> None:
        """Commands should be passed to executor with generated IDs."""
        executor = MockExecutor()
        agent = _make_agent(
            [
                {"action": "click", "x": 100, "y": 200},
                {"action": "type", "text": "hello"},
                {"action": "done", "reason": "ok"},
            ],
            executor=executor,
        )
        agent.run("exec test")

        assert len(executor.executed) == 2
        assert executor.executed[0]["action"] == "click"
        assert executor.executed[1]["action"] == "type"

    def test_command_id_generated(self) -> None:
        """M2 should generate command IDs for executor dedup."""
        executor = MockExecutor()
        agent = _make_agent(
            [
                {"action": "click", "x": 1, "y": 1},
                {"action": "done", "reason": "ok"},
            ],
            executor=executor,
        )
        agent.run("id test")

        assert "id" in executor.executed[0]
        assert executor.executed[0]["id"].startswith("step_1_")

    def test_done_action_not_sent_to_executor(self) -> None:
        """Done action should NOT be forwarded to executor."""
        executor = MockExecutor()
        agent = _make_agent(
            [{"action": "done", "reason": "ok"}],
            executor=executor,
        )
        agent.run("done only")

        assert len(executor.executed) == 0

    def test_skipped_status_not_counted_as_failure(self) -> None:
        """Executor returning 'skipped' should not increment failure counter."""
        class SkipExecutor:
            def execute(self, command: dict) -> dict:
                return {"id": command.get("id"), "status": "skipped",
                        "reason": "duplicate"}

        agent = VisionAgent(
            capture=MockCapture(),
            protocol=MockProtocol([
                {"action": "click", "x": 1, "y": 1},
                {"action": "click", "x": 2, "y": 2},
                {"action": "done", "reason": "ok"},
            ]),
            executor=SkipExecutor(),
            max_consecutive_failures=2,
            post_action_delay_s=0,
            stability_check=False,
        )
        result = agent.run("skip test")

        # Should complete, not fail due to consecutive failures
        assert result.status == TaskStatus.COMPLETED


# ===========================================================================
# Callback
# ===========================================================================

class TestOnStepCallback:
    """Tests for on_step callback."""

    def test_callback_invoked(self) -> None:
        """on_step should be called for each action step including done."""
        steps_received: list[dict] = []

        agent = _make_agent([
            {"action": "click", "x": 1, "y": 1},
            {"action": "type", "text": "hi"},
            {"action": "done", "reason": "ok"},
        ])
        result = agent.run("callback test", on_step=steps_received.append)

        assert result.status == TaskStatus.COMPLETED
        assert len(steps_received) == 4
        # Step 0: initial screenshot
        assert steps_received[0]["step"] == 0
        assert steps_received[0]["execute_status"] == "screenshot"
        # Steps 1-3: click, type, done
        assert steps_received[1]["command"]["action"] == "click"
        assert steps_received[2]["command"]["action"] == "type"
        assert steps_received[3]["command"]["action"] == "done"
        assert steps_received[3]["execute_status"] == "done"
        assert steps_received[3]["screenshot_base64"]

    def test_callback_error_does_not_crash(self) -> None:
        """Callback exceptions should be caught, not crash the loop."""
        def bad_callback(info: dict) -> None:
            raise ValueError("callback boom")

        agent = _make_agent([
            {"action": "click", "x": 1, "y": 1},
            {"action": "done", "reason": "ok"},
        ])
        result = agent.run("callback error test", on_step=bad_callback)

        # Should still complete despite callback error
        assert result.status == TaskStatus.COMPLETED


# ===========================================================================
# Result Structure
# ===========================================================================

class TestResultStructure:
    """Tests for TaskResult field correctness."""

    def test_completed_result_fields(self) -> None:
        """COMPLETED result should have all required fields."""
        agent = _make_agent([
            {"action": "done", "reason": "all good"},
        ])
        result = agent.run("field test")

        assert result.status == TaskStatus.COMPLETED
        assert result.reason == "all good"
        assert result.total_steps == 1
        assert result.total_input_tokens >= 0
        assert result.total_output_tokens >= 0
        assert result.total_duration_s >= 0
        assert isinstance(result.step_log, list)

    def test_failed_result_fields(self) -> None:
        """FAILED result should have descriptive reason."""
        agent = _make_agent(
            [{"action": "click", "x": 1, "y": 1}],
            max_steps=2,
        )
        result = agent.run("fail test")

        assert result.status == TaskStatus.FAILED
        assert len(result.reason) > 0


# ===========================================================================
# Wait Action
# ===========================================================================

class TestWaitAction:
    """Tests for the wait action."""

    def test_wait_action_does_not_terminate(self) -> None:
        """wait action should not terminate the loop like done does."""
        agent = _make_agent([
            {"action": "wait", "duration_s": 1.0, "screen_summary": "loading"},
            {"action": "done", "reason": "loaded"},
        ])
        result = agent.run("wait test")

        assert result.status == TaskStatus.COMPLETED
        assert result.total_steps == 2

    def test_wait_action_sent_to_executor(self) -> None:
        """wait action should be forwarded to executor."""
        executor = MockExecutor()
        agent = _make_agent(
            [
                {"action": "wait", "duration_s": 3.0, "screen_summary": "loading"},
                {"action": "done", "reason": "ok"},
            ],
            executor=executor,
        )
        agent.run("wait exec test")

        assert len(executor.executed) == 1
        assert executor.executed[0]["action"] == "wait"
        assert executor.executed[0]["duration_s"] == 3.0

    def test_wait_resets_consecutive_failures(self) -> None:
        """A successful wait should reset the consecutive failure counter."""
        agent = _make_agent(
            [
                None,  # LLM failure (count=1)
                {"action": "wait", "duration_s": 1.0},  # Success (reset)
                None,  # LLM failure (count=1)
                {"action": "done", "reason": "ok"},
            ],
            max_consecutive_failures=2,
        )
        result = agent.run("wait reset test")

        assert result.status == TaskStatus.COMPLETED


# ===========================================================================
# UI Stability Detection
# ===========================================================================

def _make_image(value: int = 0, width: int = 1280, height: int = 720) -> Image.Image:
    """Create a solid-color test image."""
    arr = np.full((height, width, 3), value, dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


class ImageCapture:
    """Mock capture that returns configurable PIL Images for stability testing."""

    def __init__(self, images: list[Image.Image]) -> None:
        self._images = images
        self._idx = 0
        self.call_count = 0

    def capture(self) -> CaptureResult:
        self.call_count += 1
        idx = min(self._idx, len(self._images) - 1)
        self._idx += 1
        return frame_to_capture_result(self._images[idx])


class TestUIStability:
    """Tests for automatic UI stability detection (frame diffing)."""

    @patch("agent.vision_agent.time.sleep")
    def test_stability_waits_for_stable_screen(self, mock_sleep: Any) -> None:
        """Agent should capture extra frames until the screen stabilizes."""
        # Frame sequence: initial (black), then changing, then stable
        images = [
            _make_image(0),    # initial capture
            _make_image(100),  # post-action capture A (after first action)
            _make_image(200),  # still changing
            _make_image(200),  # stable (same as previous)
        ]
        cap = ImageCapture(images)

        agent = VisionAgent(
            capture=cap,
            protocol=MockProtocol([
                {"action": "click", "x": 1, "y": 1},
                {"action": "done", "reason": "ok"},
            ]),
            executor=MockExecutor(),
            post_action_delay_s=0,
            stability_check=True,
            stability_threshold=2.0,
            stability_interval_s=0,  # no real sleep in tests
            stability_max_wait_s=10.0,
        )
        result = agent.run("stability test")

        assert result.status == TaskStatus.COMPLETED
        # 1 initial + 3 stability captures = 4 total
        assert cap.call_count == 4

    @patch("agent.vision_agent.time.sleep")
    def test_stability_timeout_proceeds(self, mock_sleep: Any) -> None:
        """If screen never stabilizes, agent should proceed after max_wait."""
        # All frames are different
        images = [_make_image(i * 30) for i in range(20)]
        cap = ImageCapture(images)

        agent = VisionAgent(
            capture=cap,
            protocol=MockProtocol([
                {"action": "click", "x": 1, "y": 1},
                {"action": "done", "reason": "ok"},
            ]),
            executor=MockExecutor(),
            post_action_delay_s=0,
            stability_check=True,
            stability_threshold=2.0,
            stability_interval_s=0,
            stability_max_wait_s=0,  # immediate timeout
        )
        result = agent.run("timeout test")

        assert result.status == TaskStatus.COMPLETED

    @patch("agent.vision_agent.time.sleep")
    def test_stability_disabled_single_capture(self, mock_sleep: Any) -> None:
        """With stability_check=False, only one capture per step."""
        images = [_make_image(0), _make_image(100)]
        cap = ImageCapture(images)

        agent = VisionAgent(
            capture=cap,
            protocol=MockProtocol([
                {"action": "done", "reason": "ok"},
            ]),
            executor=MockExecutor(),
            post_action_delay_s=0,
            stability_check=False,
        )
        result = agent.run("no stability test")

        assert result.status == TaskStatus.COMPLETED
        # Only 1 initial capture, no stability check (done on first step)
        assert cap.call_count == 1


# ===========================================================================
# CaptureError Edge Cases
# ===========================================================================

class _FailingImageCapture:
    """Mock capture that raises CaptureError on a specific call number."""

    def __init__(self, images: list[Image.Image], fail_at: int) -> None:
        self._images = images
        self._fail_at = fail_at
        self._idx = 0
        self.call_count = 0

    def capture(self) -> CaptureResult:
        self.call_count += 1
        if self.call_count == self._fail_at:
            raise CaptureError("Mock capture failure")
        idx = min(self._idx, len(self._images) - 1)
        self._idx += 1
        return frame_to_capture_result(self._images[idx])


# ===========================================================================
# report_result Integration
# ===========================================================================

class TestReportResultIntegration:
    """Tests for VisionAgent calling protocol.report_result() after execution."""

    def test_report_result_called_on_success(self) -> None:
        """Successful execution should call report_result(True)."""
        protocol = MockProtocol([
            {"action": "click", "x": 100, "y": 200},
            {"action": "done", "reason": "ok"},
        ])
        agent = VisionAgent(
            capture=MockCapture(),
            protocol=protocol,
            executor=MockExecutor(),
            post_action_delay_s=0,
            stability_check=False,
        )
        agent.run("test task")

        assert len(protocol.report_result_calls) == 1
        assert protocol.report_result_calls[0] == (True, None)

    def test_report_result_called_on_exec_failure(self) -> None:
        """Failed execution should call report_result(False, error_msg)."""
        protocol = MockProtocol([
            {"action": "click", "x": 100, "y": 200},
            {"action": "click", "x": 100, "y": 200},
            {"action": "done", "reason": "ok"},
        ])
        executor = MockExecutor(fail_actions={"click"})
        agent = VisionAgent(
            capture=MockCapture(),
            protocol=protocol,
            executor=executor,
            max_consecutive_failures=5,
            post_action_delay_s=0,
            stability_check=False,
        )
        agent.run("test task")

        # Two click failures then done (no report_result for done)
        assert len(protocol.report_result_calls) == 2
        for success, error in protocol.report_result_calls:
            assert success is False
            assert error is not None

    def test_report_result_called_on_skipped(self) -> None:
        """Skipped (duplicate) execution should call report_result(True)."""
        class SkipExecutor:
            def execute(self, command: dict) -> dict:
                return {"id": command.get("id"), "status": "skipped",
                        "reason": "duplicate"}

        protocol = MockProtocol([
            {"action": "click", "x": 1, "y": 1},
            {"action": "done", "reason": "ok"},
        ])
        agent = VisionAgent(
            capture=MockCapture(),
            protocol=protocol,
            executor=SkipExecutor(),
            post_action_delay_s=0,
            stability_check=False,
        )
        agent.run("skip test")

        assert len(protocol.report_result_calls) == 1
        assert protocol.report_result_calls[0] == (True, None)

    def test_report_result_not_called_for_done(self) -> None:
        """Done action should NOT call report_result (no execution happens)."""
        protocol = MockProtocol([
            {"action": "done", "reason": "already done"},
        ])
        agent = VisionAgent(
            capture=MockCapture(),
            protocol=protocol,
            executor=MockExecutor(),
            post_action_delay_s=0,
            stability_check=False,
        )
        agent.run("done only")

        assert len(protocol.report_result_calls) == 0


# ===========================================================================
# CaptureError Edge Cases
# ===========================================================================

class TestTruncateBase64InMessages:
    """Tests for VisionAgent._truncate_base64_in_messages."""

    def test_anthropic_image_block_truncated(self) -> None:
        long_b64 = "A" * 100
        messages = [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": long_b64}},
        ]}]
        result = VisionAgent._truncate_base64_in_messages(messages)
        data = result[0]["content"][0]["source"]["data"]
        assert "...(" in data
        assert "100 bytes base64" in data
        assert data.startswith("A" * 20)

    def test_openai_image_url_truncated(self) -> None:
        long_b64 = "B" * 200
        messages = [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{long_b64}"}},
        ]}]
        result = VisionAgent._truncate_base64_in_messages(messages)
        url = result[0]["content"][0]["image_url"]["url"]
        assert url.startswith("data:image/jpeg;base64,")
        assert "200 bytes base64" in url

    def test_tool_result_nested_image_truncated(self) -> None:
        long_b64 = "C" * 150
        messages = [{"role": "user", "content": [
            {"type": "tool_result", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": long_b64}},
            ]},
        ]}]
        result = VisionAgent._truncate_base64_in_messages(messages)
        nested = result[0]["content"][0]["content"][0]
        assert "150 bytes base64" in nested["source"]["data"]

    def test_string_content_unchanged(self) -> None:
        messages = [{"role": "user", "content": "Hello world"}]
        result = VisionAgent._truncate_base64_in_messages(messages)
        assert result[0]["content"] == "Hello world"

    def test_short_base64_not_truncated(self) -> None:
        short_b64 = "A" * 30
        messages = [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": short_b64}},
        ]}]
        result = VisionAgent._truncate_base64_in_messages(messages)
        assert result[0]["content"][0]["source"]["data"] == short_b64

    def test_original_messages_not_mutated(self) -> None:
        long_b64 = "D" * 100
        original_source = {"type": "base64", "media_type": "image/jpeg", "data": long_b64}
        messages = [{"role": "user", "content": [
            {"type": "image", "source": original_source},
        ]}]
        VisionAgent._truncate_base64_in_messages(messages)
        # Original should be untouched
        assert messages[0]["content"][0]["source"]["data"] == long_b64


class TestCaptureErrorEdgeCases:
    """Tests for CaptureError during stability check and re-capture paths."""

    @patch("agent.vision_agent.time.sleep")
    def test_capture_failure_during_stability_check(self, mock_sleep: Any) -> None:
        """CaptureError inside _wait_for_stable_screen should abort with FAILED."""
        # Capture 1: initial (ok), Capture 2: stability frame_a (ok),
        # Capture 3: stability frame_b (FAIL)
        images = [_make_image(0), _make_image(100)]
        cap = _FailingImageCapture(images, fail_at=3)

        agent = VisionAgent(
            capture=cap,
            protocol=MockProtocol([
                {"action": "click", "x": 1, "y": 1},
                {"action": "done", "reason": "ok"},
            ]),
            executor=MockExecutor(),
            post_action_delay_s=0,
            stability_check=True,
            stability_threshold=2.0,
            stability_interval_s=0,
            stability_max_wait_s=10.0,
        )
        result = agent.run("stability capture fail")

        assert result.status == TaskStatus.FAILED
        assert "capture" in result.reason.lower()

    def test_capture_failure_after_llm_failure(self) -> None:
        """CaptureError on re-capture after protocol failure should abort with FAILED."""
        # Capture 1: initial (ok), then protocol fails, re-capture (FAIL)
        cap = MockCapture(fail_at_step=2)

        agent = VisionAgent(
            capture=cap,
            protocol=MockProtocol([None, {"action": "done", "reason": "ok"}]),
            executor=MockExecutor(),
            post_action_delay_s=0,
            stability_check=False,
        )
        result = agent.run("re-capture fail")

        assert result.status == TaskStatus.FAILED
        assert "capture" in result.reason.lower()


# ===========================================================================
# Auto-detect Target OS
# ===========================================================================

class TestAutoDetectTargetOS:
    """Tests for auto-detecting target OS from initial screenshot."""

    def test_auto_detect_sets_target_os(self) -> None:
        """When executor._target_os is None, detect_os should be called and result applied."""
        from executor.clipboard_bridge import TargetOS

        protocol = MockProtocol(
            [{"action": "done", "reason": "ok"}],
            detect_os_result="macos",
        )
        executor = MockExecutor()
        executor._target_os = None  # simulate auto-detect mode

        agent = VisionAgent(
            capture=MockCapture(),
            protocol=protocol,
            executor=executor,
            post_action_delay_s=0,
            stability_check=False,
        )
        result = agent.run("test task")

        assert result.status == TaskStatus.COMPLETED
        assert len(protocol.detect_os_calls) == 1
        assert executor._target_os == TargetOS.MACOS

    def test_auto_detect_none_leaves_unset(self) -> None:
        """When detect_os returns None, _target_os should stay None."""
        protocol = MockProtocol(
            [{"action": "done", "reason": "ok"}],
            detect_os_result=None,
        )
        executor = MockExecutor()
        executor._target_os = None

        agent = VisionAgent(
            capture=MockCapture(),
            protocol=protocol,
            executor=executor,
            post_action_delay_s=0,
            stability_check=False,
        )
        result = agent.run("test task")

        assert result.status == TaskStatus.COMPLETED
        assert len(protocol.detect_os_calls) == 1
        assert executor._target_os is None

    def test_manual_target_os_skips_detect(self) -> None:
        """When _target_os is already set, detect_os should not be called."""
        from executor.clipboard_bridge import TargetOS

        protocol = MockProtocol(
            [{"action": "done", "reason": "ok"}],
            detect_os_result="linux",
        )
        executor = MockExecutor()
        executor._target_os = TargetOS.WINDOWS  # already set

        agent = VisionAgent(
            capture=MockCapture(),
            protocol=protocol,
            executor=executor,
            post_action_delay_s=0,
            stability_check=False,
        )
        result = agent.run("test task")

        assert result.status == TaskStatus.COMPLETED
        assert len(protocol.detect_os_calls) == 0
        assert executor._target_os == TargetOS.WINDOWS

    def test_detect_os_invalid_ignored(self) -> None:
        """detect_os returning invalid value (e.g. 'chromeos') is handled gracefully."""
        protocol = MockProtocol(
            [{"action": "done", "reason": "ok"}],
            detect_os_result="chromeos",
        )
        executor = MockExecutor()
        executor._target_os = None

        agent = VisionAgent(
            capture=MockCapture(),
            protocol=protocol,
            executor=executor,
            post_action_delay_s=0,
            stability_check=False,
        )
        result = agent.run("test task")

        assert result.status == TaskStatus.COMPLETED
        assert executor._target_os is None

    def test_no_target_os_attr_skips_detect(self) -> None:
        """Executors without _target_os attr should not trigger detect_os."""
        protocol = MockProtocol(
            [{"action": "done", "reason": "ok"}],
            detect_os_result="macos",
        )
        executor = MockExecutor()
        # MockExecutor doesn't have _target_os by default

        agent = VisionAgent(
            capture=MockCapture(),
            protocol=protocol,
            executor=executor,
            post_action_delay_s=0,
            stability_check=False,
        )
        result = agent.run("test task")

        assert result.status == TaskStatus.COMPLETED
        assert len(protocol.detect_os_calls) == 0


# ===========================================================================
# Batch (Multi-Action) Execution
# ===========================================================================

class TestBatchExecution:
    """Tests for multi-action per step execution."""

    def test_batch_of_two_actions_both_succeed(self) -> None:
        """Batch of 2 actions should execute both, step_count increments by 1."""
        executor = MockExecutor()
        agent = _make_agent(
            [
                [
                    {"action": "click", "x": 500, "y": 300},
                    {"action": "type", "text": "hello"},
                ],
                {"action": "done", "reason": "ok"},
            ],
            executor=executor,
        )
        result = agent.run("batch test")

        assert result.status == TaskStatus.COMPLETED
        assert result.total_steps == 2  # 1 batch step + 1 done step
        assert len(executor.executed) == 2
        assert executor.executed[0]["action"] == "click"
        assert executor.executed[1]["action"] == "type"

    def test_batch_second_action_fails(self) -> None:
        """Second action in batch fails -> third not executed."""
        executor = FailAtIndexExecutor(fail_at=1)
        agent = _make_agent(
            [
                [
                    {"action": "click", "x": 100, "y": 100},
                    {"action": "type", "text": "hi"},
                    {"action": "key", "keys": ["return"]},
                ],
                {"action": "done", "reason": "ok"},
            ],
            executor=executor,
            max_consecutive_failures=3,
        )
        result = agent.run("partial fail test")

        assert result.status == TaskStatus.COMPLETED
        # Only first 2 commands executed, third skipped
        assert len(executor.executed) == 2

    def test_batch_failure_increments_consecutive(self) -> None:
        """Batch failure should increment consecutive_failures."""
        executor = MockExecutor(fail_actions={"click"})
        agent = _make_agent(
            [
                [
                    {"action": "click", "x": 100, "y": 100},
                    {"action": "type", "text": "hi"},
                ],
            ],
            executor=executor,
            max_consecutive_failures=2,
        )
        result = agent.run("consecutive fail test")

        assert result.status == TaskStatus.FAILED
        assert "consecutive" in result.reason.lower()

    def test_batch_success_resets_consecutive(self) -> None:
        """A successful batch should reset consecutive_failures."""
        agent = _make_agent(
            [
                None,  # LLM failure (count=1)
                [
                    {"action": "click", "x": 100, "y": 100},
                    {"action": "type", "text": "hi"},
                ],  # Batch success (reset to 0)
                None,  # LLM failure (count=1)
                {"action": "done", "reason": "ok"},
            ],
            max_consecutive_failures=2,
        )
        result = agent.run("reset test")

        assert result.status == TaskStatus.COMPLETED

    def test_step_log_contains_batch_fields(self) -> None:
        """Step log should contain commands list and batch_size."""
        agent = _make_agent([
            [
                {"action": "click", "x": 100, "y": 200},
                {"action": "type", "text": "hello"},
            ],
            {"action": "done", "reason": "ok"},
        ])
        result = agent.run("log test")

        batch_entry = result.step_log[0]
        assert "commands" in batch_entry
        assert "batch_size" in batch_entry
        assert batch_entry["batch_size"] == 2
        assert len(batch_entry["commands"]) == 2
        assert batch_entry["command"]["action"] == "click"  # first command

    def test_on_step_called_once_per_batch(self) -> None:
        """on_step should be called once per batch, not per command."""
        steps_received: list[dict] = []

        agent = _make_agent([
            [
                {"action": "click", "x": 100, "y": 200},
                {"action": "type", "text": "hello"},
            ],
            {"action": "done", "reason": "ok"},
        ])
        result = agent.run("callback test", on_step=steps_received.append)

        assert result.status == TaskStatus.COMPLETED
        # Step 0: initial screenshot, Step 1: batch, Step 2: done
        assert len(steps_received) == 3
        assert steps_received[1]["batch_size"] == 2

    def test_single_action_via_get_commands(self) -> None:
        """Single command still works via get_commands() fallback."""
        executor = MockExecutor()
        agent = _make_agent(
            [
                {"action": "click", "x": 100, "y": 200},
                {"action": "done", "reason": "ok"},
            ],
            executor=executor,
        )
        result = agent.run("single test")

        assert result.status == TaskStatus.COMPLETED
        assert len(executor.executed) == 1
        # Step log should have batch_size 1 for single commands
        assert result.step_log[0]["batch_size"] == 1

    def test_batch_command_ids_generated(self) -> None:
        """Each command in batch should get a unique ID with index."""
        executor = MockExecutor()
        agent = _make_agent(
            [
                [
                    {"action": "click", "x": 100, "y": 200},
                    {"action": "type", "text": "hi"},
                ],
                {"action": "done", "reason": "ok"},
            ],
            executor=executor,
        )
        agent.run("id test")

        assert executor.executed[0]["id"].startswith("step_1_0_")
        assert executor.executed[1]["id"].startswith("step_1_1_")

    def test_report_results_called_for_batch(self) -> None:
        """protocol.report_results should be called with per-command results."""
        protocol = MockProtocol([
            [
                {"action": "click", "x": 100, "y": 200},
                {"action": "type", "text": "hi"},
            ],
            {"action": "done", "reason": "ok"},
        ])
        agent = VisionAgent(
            capture=MockCapture(),
            protocol=protocol,
            executor=MockExecutor(),
            post_action_delay_s=0,
            stability_check=False,
        )
        agent.run("test task")

        # 2 success results from the batch
        assert len(protocol.report_result_calls) == 2
        assert protocol.report_result_calls[0] == (True, None)
        assert protocol.report_result_calls[1] == (True, None)

    def test_batch_with_trailing_done_executes_commands(self) -> None:
        """Commands preceding 'done' in a batch must execute before completing."""
        executor = MockExecutor()
        agent = _make_agent(
            [
                [
                    {"action": "click", "x": 100, "y": 200},
                    {"action": "type", "text": "hi"},
                    {"action": "done", "reason": "typed"},
                ],
            ],
            executor=executor,
        )
        result = agent.run("batch+done test")

        assert result.status == TaskStatus.COMPLETED
        assert len(executor.executed) == 2
        assert executor.executed[0]["action"] == "click"
        assert executor.executed[1]["action"] == "type"

    def test_step_log_contains_exec_results(self) -> None:
        """Step log should include per-command exec_results."""
        executor = FailAtIndexExecutor(fail_at=1)
        agent = _make_agent(
            [
                [
                    {"action": "click", "x": 100, "y": 100},
                    {"action": "type", "text": "hi"},
                    {"action": "key", "keys": ["return"]},
                ],
                {"action": "done", "reason": "ok"},
            ],
            executor=executor,
            max_consecutive_failures=3,
        )
        result = agent.run("exec_results test")

        batch_entry = result.step_log[0]
        assert "exec_results" in batch_entry
        assert len(batch_entry["exec_results"]) == 3
        assert batch_entry["exec_results"][0] == (True, None)
        assert batch_entry["exec_results"][1][0] is False
        assert batch_entry["exec_results"][2] == (False, "Skipped due to earlier failure")


# ===========================================================================
# Completion Status Propagation
# ===========================================================================

class TestCompletionStatusDataclasses:
    """Tests for completion_status field on StepResult and TaskResult."""

    def test_step_result_default(self) -> None:
        """StepResult with no completion_status defaults to 'success'."""
        sr = StepResult(
            command=None, is_done=True, done_reason="done",
            screen_summary="", raw_text="", input_tokens=0,
            output_tokens=0, latency_ms=0, success=True,
        )
        assert sr.completion_status == "success"

    def test_step_result_gave_up(self) -> None:
        """StepResult with completion_status='gave_up' stores correctly."""
        sr = StepResult(
            command=None, is_done=True, done_reason="cannot find button",
            screen_summary="", raw_text="", input_tokens=0,
            output_tokens=0, latency_ms=0, success=True,
            completion_status="gave_up",
        )
        assert sr.completion_status == "gave_up"

    def test_task_result_default(self) -> None:
        """TaskResult with no completion_status defaults to 'success'."""
        tr = TaskResult(
            status=TaskStatus.COMPLETED, reason="ok",
            total_steps=1, total_input_tokens=0,
            total_output_tokens=0, total_duration_s=1.0,
        )
        assert tr.completion_status == "success"

    def test_task_result_stuck(self) -> None:
        """TaskResult with completion_status='stuck' stores correctly."""
        tr = TaskResult(
            status=TaskStatus.COMPLETED, reason="stuck on login",
            total_steps=1, total_input_tokens=0,
            total_output_tokens=0, total_duration_s=1.0,
            completion_status="stuck",
        )
        assert tr.completion_status == "stuck"


class TestCompletionStatusPropagation:
    """Tests for completion_status propagation through VisionAgent."""

    def test_default_status_propagation(self) -> None:
        """Agent done with default status produces TaskResult.completion_status == 'success'."""
        agent = _make_agent([
            {"action": "done", "reason": "Task completed"},
        ])
        result = agent.run("test task")
        assert result.status == TaskStatus.COMPLETED
        assert result.completion_status == "success"

    def test_gave_up_status_propagation(self) -> None:
        """Agent done with gave_up propagates to TaskResult.completion_status."""
        agent = _make_agent([
            {"action": "done", "reason": "Cannot find button", "status": "gave_up"},
        ])
        result = agent.run("test task")
        assert result.status == TaskStatus.COMPLETED
        assert result.completion_status == "gave_up"
