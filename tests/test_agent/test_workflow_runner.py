"""Tests for the workflow runner — step execution, retry, replan, abort."""

from __future__ import annotations

import threading
from typing import Any
from unittest.mock import MagicMock

import pytest

from agent.planner import PlanStep, TaskPlanner
from agent.vision_agent import TaskResult, TaskStatus, VisionAgent
from agent.workflow_runner import WorkflowResult, WorkflowRunner
from tests.test_agent.conftest import (
    FakeCaptureResult,
    MockCapture,
    MockExecutor,
    MockProtocol,
    make_test_image,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class MockPlanner(TaskPlanner):
    """Planner that returns pre-configured steps."""

    def __init__(
        self,
        plan_steps: list[PlanStep] | None = None,
        replan_steps: list[PlanStep] | None = None,
    ) -> None:
        super().__init__(provider="anthropic", model="test", api_key="test")
        self._plan_steps = plan_steps or [PlanStep(number=1, goal="test step")]
        self._replan_steps = replan_steps

    def plan(self, task_goal, screenshot_base64, skill_text=None):
        return self._plan_steps

    def replan(self, **kwargs):
        return self._replan_steps


def _make_agent(
    responses: list[dict[str, Any] | None],
    capture: Any = None,
) -> VisionAgent:
    """Build a VisionAgent with mock dependencies."""
    return VisionAgent(
        capture=capture or MockCapture(),
        protocol=MockProtocol(responses),
        executor=MockExecutor(),
        max_steps=10,
        max_consecutive_failures=3,
        post_action_delay_s=0,
        task_timeout_s=60.0,
        stability_check=False,
    )


# ===========================================================================
# Happy Path
# ===========================================================================

class TestWorkflowHappyPath:
    """Tests for successful workflow execution."""

    def test_single_step_completes(self) -> None:
        agent = _make_agent([{"action": "done", "reason": "Step done"}])
        planner = MockPlanner([PlanStep(number=1, goal="Do the thing")])
        runner = WorkflowRunner(agent, planner, auto_approve=True)

        result = runner.run("task", "fake_screenshot")
        assert result.status == "completed"
        assert result.steps_completed == 1

    def test_multi_step_completes(self) -> None:
        # Agent completes each step with "done"
        agent = _make_agent([{"action": "done", "reason": "Step done"}])
        planner = MockPlanner([
            PlanStep(number=1, goal="Step 1"),
            PlanStep(number=2, goal="Step 2"),
            PlanStep(number=3, goal="Step 3"),
        ])
        runner = WorkflowRunner(agent, planner, auto_approve=True)

        result = runner.run("task", "fake_screenshot")
        assert result.status == "completed"
        assert result.steps_completed == 3

    def test_progress_callback_called(self) -> None:
        agent = _make_agent([{"action": "done", "reason": "ok"}])
        planner = MockPlanner([PlanStep(number=1, goal="Step 1")])
        runner = WorkflowRunner(agent, planner, auto_approve=True)

        events: list[dict] = []
        result = runner.run("task", "fake_screenshot", on_progress=events.append)

        # Filter workflow-level events (have "type" key) from
        # per-action agent events (have "step" key but no "type").
        workflow_types = [e["type"] for e in events if "type" in e]
        assert "plan_ready" in workflow_types
        assert "step_start" in workflow_types
        assert "step_done" in workflow_types
        assert "workflow_done" in workflow_types


# ===========================================================================
# Retry
# ===========================================================================

class TestWorkflowRetry:
    """Tests for step retry logic."""

    def test_retry_then_succeed(self) -> None:
        # First call fails, second succeeds
        responses = [
            None,  # fail
            {"action": "done", "reason": "ok"},  # succeed on retry
        ]
        agent = _make_agent(responses)
        planner = MockPlanner([PlanStep(number=1, goal="Tricky step")])
        runner = WorkflowRunner(agent, planner, max_retries_per_step=2, auto_approve=True)

        result = runner.run("task", "fake_screenshot")
        assert result.status == "completed"
        assert result.steps_completed == 1


# ===========================================================================
# Replan
# ===========================================================================

class TestWorkflowReplan:
    """Tests for re-planning after step failure."""

    def test_replan_after_exhausted_retries(self) -> None:
        # MockProtocol.reset() resets call_count to 0, so each agent.run()
        # starts from the beginning of the responses list.
        # We need a protocol that fails on the first run but succeeds
        # on subsequent runs (after replan). Use a stateful mock.
        call_number = {"value": 0}
        original_responses = [None, None, None]  # fail
        success_responses = [{"action": "done", "reason": "ok"}]

        class ReplanProtocol(MockProtocol):
            def reset(self) -> None:
                call_number["value"] += 1
                if call_number["value"] >= 2:
                    # After replan, switch to success responses
                    self.responses = success_responses
                super().reset()

        protocol = ReplanProtocol(original_responses)
        agent = VisionAgent(
            capture=MockCapture(),
            protocol=protocol,
            executor=MockExecutor(),
            max_steps=10,
            max_consecutive_failures=3,
            post_action_delay_s=0,
            task_timeout_s=60.0,
            stability_check=False,
        )
        planner = MockPlanner(
            plan_steps=[PlanStep(number=1, goal="Original step")],
            replan_steps=[PlanStep(number=1, goal="New approach")],
        )
        runner = WorkflowRunner(
            agent, planner, max_retries_per_step=0, max_replans=2, auto_approve=True,
        )

        result = runner.run("task", "fake_screenshot")
        assert result.status == "completed"
        assert result.steps_completed == 1

    def test_abort_when_replan_fails(self) -> None:
        responses = [None, None, None]  # all fail
        agent = _make_agent(responses)
        planner = MockPlanner(
            plan_steps=[PlanStep(number=1, goal="Step")],
            replan_steps=None,  # replan returns None
        )
        runner = WorkflowRunner(agent, planner, max_retries_per_step=2, auto_approve=True)

        result = runner.run("task", "fake_screenshot")
        assert result.status == "failed"
        assert "re-planning" in result.reason.lower() or "re-plan" in result.reason.lower()

    def test_replan_preserves_completed_steps(self) -> None:
        """Re-plan after failure keeps completed steps and only replaces remaining."""
        # Step 1 succeeds, step 2 fails, replan produces new steps
        run_count = {"value": 0}

        class StepAwareProtocol(MockProtocol):
            def reset(self) -> None:
                run_count["value"] += 1
                if run_count["value"] <= 1:
                    # First agent.run (step 1) succeeds
                    self.responses = [{"action": "done", "reason": "ok"}]
                elif run_count["value"] == 2:
                    # Second agent.run (step 2) fails
                    self.responses = [None]
                else:
                    # After replan, succeed
                    self.responses = [{"action": "done", "reason": "ok"}]
                super().reset()

        protocol = StepAwareProtocol([{"action": "done", "reason": "ok"}])
        agent = VisionAgent(
            capture=MockCapture(),
            protocol=protocol,
            executor=MockExecutor(),
            max_steps=10,
            max_consecutive_failures=3,
            post_action_delay_s=0,
            task_timeout_s=60.0,
            stability_check=False,
        )
        planner = MockPlanner(
            plan_steps=[
                PlanStep(number=1, goal="Step A"),
                PlanStep(number=2, goal="Step B"),
                PlanStep(number=3, goal="Step C"),
            ],
            replan_steps=[
                PlanStep(number=1, goal="Step B revised"),
                PlanStep(number=2, goal="Step C revised"),
            ],
        )
        runner = WorkflowRunner(
            agent, planner, max_retries_per_step=0, max_replans=1,
            auto_approve=True,
        )

        events: list[dict] = []
        result = runner.run("task", "fake_screenshot", on_progress=events.append)

        assert result.status == "completed"
        # Step A completed + 2 replanned steps
        assert result.steps_completed == 3

        # Check replanned event preserves completed count
        replan_events = [e for e in events if e.get("type") == "replanned"]
        assert len(replan_events) == 1
        assert replan_events[0]["steps_completed"] == 1

        # New steps should be renumbered: 2 and 3 (after step 1)
        new_step_numbers = [s["number"] for s in replan_events[0]["new_steps"]]
        assert new_step_numbers == [2, 3]

    def test_max_replans_limit(self) -> None:
        # Keep failing even after replan
        responses = [None] * 20  # always fail
        agent = _make_agent(responses)
        planner = MockPlanner(
            plan_steps=[PlanStep(number=1, goal="Step")],
            replan_steps=[PlanStep(number=1, goal="New step")],  # replan works but step still fails
        )
        runner = WorkflowRunner(
            agent, planner, max_retries_per_step=0, max_replans=2, auto_approve=True,
        )

        result = runner.run("task", "fake_screenshot")
        assert result.status == "failed"
        assert "re-plan" in result.reason.lower()


# ===========================================================================
# Abort
# ===========================================================================

class TestWorkflowAbort:
    """Tests for abort handling."""

    def test_abort_between_steps(self) -> None:
        agent = _make_agent([{"action": "done", "reason": "ok"}])
        planner = MockPlanner([
            PlanStep(number=1, goal="Step 1"),
            PlanStep(number=2, goal="Step 2"),
        ])
        runner = WorkflowRunner(agent, planner, auto_approve=True)

        # Set abort after first step completes
        original_run = agent.run

        def run_with_abort(*args, **kwargs):
            result = original_run(*args, **kwargs)
            agent._abort_event.set()  # abort after this step
            return result

        agent.run = run_with_abort

        result = runner.run("task", "fake_screenshot")
        assert result.status == "aborted"
        assert result.steps_completed == 1


# ===========================================================================
# Reboot Tag
# ===========================================================================

class TestWorkflowRebootTag:
    """Tests for [REBOOT EXPECTED] step handling."""

    def test_reboot_step_calls_transition(self) -> None:
        agent = _make_agent([{"action": "done", "reason": "ok"}])
        planner = MockPlanner([
            PlanStep(number=1, goal="Type reboot command", reboot_expected=True),
        ])
        runner = WorkflowRunner(agent, planner, auto_approve=True)

        # Mock the transition handler
        agent._wait_for_reboot_transition = MagicMock()

        result = runner.run("task", "fake_screenshot")
        assert result.status == "completed"
        agent._wait_for_reboot_transition.assert_called_once()

    def test_reboot_transition_failure(self) -> None:
        from capture.base import CaptureError

        agent = _make_agent([{"action": "done", "reason": "ok"}])
        planner = MockPlanner([
            PlanStep(number=1, goal="Reboot", reboot_expected=True),
        ])
        runner = WorkflowRunner(agent, planner, auto_approve=True)

        # Transition handler fails
        agent._wait_for_reboot_transition = MagicMock(
            side_effect=CaptureError("Timeout"),
        )

        result = runner.run("task", "fake_screenshot")
        assert result.status == "failed"
        assert "transition" in result.reason.lower()


# ===========================================================================
# Single Step Plan (no-skill passthrough)
# ===========================================================================

class TestWorkflowSingleStep:
    """Tests for single-step plans (equivalent to normal agent.run)."""

    def test_single_step_works(self) -> None:
        agent = _make_agent([{"action": "done", "reason": "ok"}])
        planner = MockPlanner([PlanStep(number=1, goal="Just do the task")])
        runner = WorkflowRunner(agent, planner, auto_approve=True)

        result = runner.run("simple task", "fake_screenshot")
        assert result.status == "completed"
        assert result.steps_completed == 1
        assert result.steps_total == 1


# ===========================================================================
# Post-Step Validation
# ===========================================================================

class TestWorkflowValidation:
    """Tests for post-step validation (adaptive re-planning)."""

    def test_validation_continue(self) -> None:
        """When validation says CONTINUE, steps proceed normally."""
        agent = _make_agent([{"action": "done", "reason": "ok"}])
        planner = MockPlanner([
            PlanStep(number=1, goal="Step 1"),
            PlanStep(number=2, goal="Step 2"),
        ])
        # Mock validate_plan to return continue
        planner.validate_plan = lambda **kw: ("continue", None, "")
        runner = WorkflowRunner(agent, planner, auto_approve=True)

        result = runner.run("task", "fake_screenshot")
        assert result.status == "completed"
        assert result.steps_completed == 2

    def test_validation_replan(self) -> None:
        """When validation says REPLAN, remaining steps are replaced."""
        agent = _make_agent([{"action": "done", "reason": "ok"}])
        planner = MockPlanner([
            PlanStep(number=1, goal="Open Edge"),
            PlanStep(number=2, goal="Go to taobao"),
        ])
        # After step 1, validation says replan
        call_count = {"n": 0}
        def mock_validate(**kw):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return ("replan", [PlanStep(number=2, goal="Close Edge"), PlanStep(number=3, goal="Open Chrome")], "Wrong browser")
            return ("continue", None, "")
        planner.validate_plan = mock_validate
        runner = WorkflowRunner(agent, planner, auto_approve=True)

        result = runner.run("task", "fake_screenshot")
        assert result.status == "completed"
        # 1 (original step 1) + 2 (replanned steps) = 3
        assert result.steps_completed == 3

    def test_validation_escalate_then_resolve(self) -> None:
        """When validation says ESCALATE, workflow pauses until user resolves."""
        import threading
        agent = _make_agent([{"action": "done", "reason": "ok"}])
        planner = MockPlanner([
            PlanStep(number=1, goal="Open browser"),
            PlanStep(number=2, goal="Go to site"),
        ])
        call_count = {"n": 0}
        def mock_validate(**kw):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return ("escalate", None, "Login required")
            return ("continue", None, "")
        planner.validate_plan = mock_validate
        runner = WorkflowRunner(agent, planner, auto_approve=True)

        # Auto-resolve the escalation from another thread
        def resolve_soon():
            import time
            time.sleep(0.5)
            runner.resolve_escalation()
        t = threading.Thread(target=resolve_soon)
        t.start()

        result = runner.run("task", "fake_screenshot")
        t.join()
        assert result.status == "completed"
        assert result.steps_completed == 2

    def test_validation_escalate_then_abort(self) -> None:
        """When escalation is pending and user aborts, workflow aborts."""
        import threading
        agent = _make_agent([{"action": "done", "reason": "ok"}])
        planner = MockPlanner([
            PlanStep(number=1, goal="Step 1"),
            PlanStep(number=2, goal="Step 2"),
        ])
        def mock_validate(**kw):
            return ("escalate", None, "CAPTCHA detected")
        planner.validate_plan = mock_validate
        runner = WorkflowRunner(agent, planner, auto_approve=True)

        # Abort from another thread
        def abort_soon():
            import time
            time.sleep(0.5)
            agent._abort_event.set()
        t = threading.Thread(target=abort_soon)
        t.start()

        result = runner.run("task", "fake_screenshot")
        t.join()
        assert result.status == "aborted"

    def test_validation_runs_on_last_step(self) -> None:
        """Last step MUST trigger validation (D-04: always validate last step).

        Changed from test_no_validation_on_last_step: the new validation
        gate always validates the last step to confirm the task completed.
        """
        agent = _make_agent([{"action": "done", "reason": "ok"}])
        planner = MockPlanner([PlanStep(number=1, goal="Only step")])
        validate_called = {"called": False}
        def mock_validate(**kw):
            validate_called["called"] = True
            return ("continue", None, "")
        planner.validate_plan = mock_validate
        runner = WorkflowRunner(agent, planner, auto_approve=True)

        result = runner.run("task", "fake_screenshot")
        assert result.status == "completed"
        assert validate_called["called"]


# ===========================================================================
# Completion Status Detection (COMP-02)
# ===========================================================================

class TestCompletionStatusDetection:
    """Tests for WorkflowRunner using completion_status instead of keyword matching.

    Verifies that the structured completion_status field is used to detect
    gave_up/stuck instead of fragile keyword matching on reason text.
    """

    def test_gave_up_detected_as_failure(self) -> None:
        """Agent done with status='gave_up' is detected as failure."""
        agent = _make_agent([
            {"action": "done", "status": "gave_up", "reason": "Cannot find the button"},
        ])
        planner = MockPlanner([PlanStep(number=1, goal="Click button")])
        runner = WorkflowRunner(
            agent, planner, max_retries_per_step=0, max_replans=0,
            auto_approve=True,
        )

        result = runner.run("task", "fake_screenshot")
        assert result.status == "failed"
        assert "gave_up" in result.reason

    def test_stuck_detected_as_failure(self) -> None:
        """Agent done with status='stuck' is detected as failure."""
        agent = _make_agent([
            {"action": "done", "status": "stuck", "reason": "Screen is blank"},
        ])
        planner = MockPlanner([PlanStep(number=1, goal="Check screen")])
        runner = WorkflowRunner(
            agent, planner, max_retries_per_step=0, max_replans=0,
            auto_approve=True,
        )

        result = runner.run("task", "fake_screenshot")
        assert result.status == "failed"
        assert "stuck" in result.reason

    def test_success_status_treated_as_success(self) -> None:
        """Agent done with status='success' is treated as success."""
        agent = _make_agent([
            {"action": "done", "status": "success", "reason": "Task completed"},
        ])
        planner = MockPlanner([PlanStep(number=1, goal="Complete task")])
        runner = WorkflowRunner(agent, planner, auto_approve=True)

        result = runner.run("task", "fake_screenshot")
        assert result.status == "completed"

    def test_default_status_treated_as_success(self) -> None:
        """Agent done without status field defaults to success (D-07 fallback)."""
        agent = _make_agent([
            {"action": "done", "reason": "Task completed"},
        ])
        planner = MockPlanner([PlanStep(number=1, goal="Complete task")])
        runner = WorkflowRunner(agent, planner, auto_approve=True)

        result = runner.run("task", "fake_screenshot")
        assert result.status == "completed"

    def test_no_false_positive_on_success_with_negative_words(self) -> None:
        """Success with 'cannot' in reason is NOT falsely detected as failure.

        This is the key regression test: the word 'cannot' in the reason
        no longer triggers failure because we check completion_status,
        not keyword matching on reason text.
        """
        agent = _make_agent([
            {
                "action": "done",
                "status": "success",
                "reason": "Task completed, cannot find any remaining issues",
            },
        ])
        planner = MockPlanner([PlanStep(number=1, goal="Complete task")])
        runner = WorkflowRunner(agent, planner, auto_approve=True)

        result = runner.run("task", "fake_screenshot")
        assert result.status == "completed"

    def test_gave_up_triggers_replan(self) -> None:
        """gave_up on first attempt triggers replan, which then succeeds."""
        call_number = {"value": 0}

        class ReplanProtocol(MockProtocol):
            def reset(self) -> None:
                call_number["value"] += 1
                if call_number["value"] >= 2:
                    self.responses = [
                        {"action": "done", "status": "success", "reason": "Found it"},
                    ]
                super().reset()

        protocol = ReplanProtocol([
            {"action": "done", "status": "gave_up", "reason": "Cannot find element"},
        ])
        agent = VisionAgent(
            capture=MockCapture(),
            protocol=protocol,
            executor=MockExecutor(),
            max_steps=10,
            max_consecutive_failures=3,
            post_action_delay_s=0,
            task_timeout_s=60.0,
            stability_check=False,
        )
        planner = MockPlanner(
            plan_steps=[PlanStep(number=1, goal="Original step")],
            replan_steps=[PlanStep(number=1, goal="New approach")],
        )
        runner = WorkflowRunner(
            agent, planner, max_retries_per_step=0, max_replans=1,
            auto_approve=True,
        )

        result = runner.run("task", "fake_screenshot")
        assert result.status == "completed"
        assert result.steps_completed >= 1


# ===========================================================================
# Validation Gating (VALID-01 / VALID-02)
# ===========================================================================

class TestValidationGating:
    """Tests for conditional validation skip/always-validate logic.

    The gate decides whether to call validate_plan after a step completes:
      - SKIP when frame-diff > 5% AND actions < 50% budget
      - ALWAYS validate for reboot steps, high budget, gave_up/stuck, last step
      - Always-validate takes precedence over skip conditions
      - No budget data (expected_actions=None) => always validate
    """

    @staticmethod
    def _build(
        plan_steps: list[PlanStep],
        *,
        images: list | None = None,
        responses: list | None = None,
    ) -> tuple[VisionAgent, MockPlanner, WorkflowRunner, list[dict]]:
        """Build wired test objects.

        Returns (agent, planner, runner, events_list).
        ``planner._validate_calls`` records step numbers that triggered
        validation. ``planner._validate_called`` is True if any call
        was made.
        """
        capture = MockCapture(images=images)
        resp = responses or [{"action": "done", "reason": "ok"}]
        agent = _make_agent(resp, capture=capture)
        planner = MockPlanner(plan_steps)
        planner._validate_called = False
        planner._validate_calls: list[int] = []

        def tracking_validate(**kw):
            planner._validate_called = True
            step = kw.get("completed_step")
            if step is not None:
                planner._validate_calls.append(step.number)
            return ("continue", None, "")

        planner.validate_plan = tracking_validate
        runner = WorkflowRunner(agent, planner, auto_approve=True)
        events: list[dict] = []
        return agent, planner, runner, events

    # -- Test 1: skip when screen changed AND low budget --

    def test_skip_when_screen_changed_and_low_budget(self) -> None:
        """frame-diff >5% AND total_steps < expected_actions*0.5 => skip."""
        # Two distinct colors => large frame-diff
        img_before = make_test_image(color=(0, 0, 0))
        img_after = make_test_image(color=(255, 255, 255))
        # Images: pre-step capture, agent initial capture, val capture
        # MockCapture returns images in order; provide enough for all calls
        images = [img_before, img_before, img_after, img_after,
                  img_after, img_after, img_after, img_after]

        steps = [
            PlanStep(number=1, goal="Step 1", expected_actions=10),
            PlanStep(number=2, goal="Step 2", expected_actions=10),
        ]
        agent, planner, runner, events = self._build(steps, images=images)
        result = runner.run("task", "fake_screenshot", on_progress=events.append)

        assert result.status == "completed"
        # Step 1 validation should be SKIPPED (screen changed, low budget)
        # Step 2 is last step => always validated
        assert 1 not in planner._validate_calls
        # Check that validation_skipped event was emitted for step 1
        skip_events = [e for e in events if e.get("type") == "validation_skipped"]
        assert len(skip_events) >= 1

    # -- Test 2: no skip when screen unchanged --

    def test_no_skip_when_screen_unchanged(self) -> None:
        """frame-diff <5% => validate even if budget is low."""
        # Same color before/after => tiny frame-diff
        same_img = make_test_image(color=(128, 128, 128))
        images = [same_img] * 20  # all same

        steps = [
            PlanStep(number=1, goal="Step 1", expected_actions=10),
            PlanStep(number=2, goal="Step 2", expected_actions=10),
        ]
        agent, planner, runner, events = self._build(steps, images=images)
        result = runner.run("task", "fake_screenshot", on_progress=events.append)

        assert result.status == "completed"
        # Step 1 must be validated (screen unchanged)
        assert 1 in planner._validate_calls

    # -- Test 3: no skip when high budget --

    def test_no_skip_when_high_budget(self) -> None:
        """total_steps >= expected_actions*0.5 => validate even if screen changed."""
        img_before = make_test_image(color=(0, 0, 0))
        img_after = make_test_image(color=(255, 255, 255))
        images = [img_before, img_before, img_after, img_after,
                  img_after, img_after, img_after, img_after]

        # expected_actions=2, agent takes 1 step (done) => ratio=0.5 => NOT < 0.5 => validate
        steps = [
            PlanStep(number=1, goal="Step 1", expected_actions=2),
            PlanStep(number=2, goal="Step 2", expected_actions=2),
        ]
        agent, planner, runner, events = self._build(steps, images=images)
        result = runner.run("task", "fake_screenshot", on_progress=events.append)

        assert result.status == "completed"
        # Step 1 validated (budget ratio >= 0.5)
        assert 1 in planner._validate_calls

    # -- Test 4: always validate reboot --

    def test_always_validate_reboot(self) -> None:
        """reboot_expected=True => always validate regardless."""
        img_before = make_test_image(color=(0, 0, 0))
        img_after = make_test_image(color=(255, 255, 255))
        images = [img_before, img_before, img_after, img_after,
                  img_after, img_after, img_after, img_after]

        steps = [
            PlanStep(number=1, goal="Reboot", expected_actions=10,
                     reboot_expected=True),
            PlanStep(number=2, goal="Continue"),
        ]
        agent, planner, runner, events = self._build(steps, images=images)
        # Mock the transition handler since reboot_expected triggers it
        agent._wait_for_reboot_transition = MagicMock()
        result = runner.run("task", "fake_screenshot", on_progress=events.append)

        assert result.status == "completed"
        # Step 1 validated (reboot always-validate)
        assert 1 in planner._validate_calls

    # -- Test 5: always validate high budget (>80%) --

    def test_always_validate_high_budget(self) -> None:
        """>80% budget consumed => always validate."""
        img_before = make_test_image(color=(0, 0, 0))
        img_after = make_test_image(color=(255, 255, 255))
        images = [img_before, img_before, img_after, img_after,
                  img_after, img_after, img_after, img_after]

        # expected_actions=1, agent takes 1 step => ratio=1.0 > 0.8 => always validate
        steps = [
            PlanStep(number=1, goal="Quick step", expected_actions=1),
            PlanStep(number=2, goal="Next step"),
        ]
        agent, planner, runner, events = self._build(steps, images=images)
        result = runner.run("task", "fake_screenshot", on_progress=events.append)

        assert result.status == "completed"
        # Step 1 validated (>80% budget consumed)
        assert 1 in planner._validate_calls

    # -- Test 6: always validate gave_up --

    def test_always_validate_gave_up(self) -> None:
        """gave_up completion_status => always validate."""
        img_before = make_test_image(color=(0, 0, 0))
        img_after = make_test_image(color=(255, 255, 255))
        images = [img_before, img_before, img_after, img_after,
                  img_after, img_after, img_after, img_after]

        steps = [
            PlanStep(number=1, goal="Try something", expected_actions=10),
            PlanStep(number=2, goal="Fallback"),
        ]
        # Note: gave_up steps currently trigger failure and retry.
        # We test the _should_skip_validation method directly since
        # the workflow path for gave_up goes through retry/replan.
        agent, planner, runner, events = self._build(steps, images=images)
        result_mock = TaskResult(
            status=TaskStatus.COMPLETED,
            reason="gave up",
            total_steps=1,
            total_input_tokens=0,
            total_output_tokens=0,
            total_duration_s=0.0,
            completion_status="gave_up",
        )
        step = steps[0]
        should_skip = runner._should_skip_validation(
            step=step,
            result=result_mock,
            pre_step_image=img_before,
            post_step_image=img_after,
            is_last_step=False,
        )
        assert not should_skip

    # -- Test 7: always validate last step --

    def test_always_validate_last_step(self) -> None:
        """Last step always triggers validation (D-04)."""
        img_before = make_test_image(color=(0, 0, 0))
        img_after = make_test_image(color=(255, 255, 255))
        images = [img_before, img_before, img_after, img_after,
                  img_after, img_after, img_after, img_after]

        steps = [
            PlanStep(number=1, goal="Only step", expected_actions=10),
        ]
        agent, planner, runner, events = self._build(steps, images=images)
        result = runner.run("task", "fake_screenshot", on_progress=events.append)

        assert result.status == "completed"
        # For single-step plan, it IS the last step => validate must run
        assert planner._validate_called

    # -- Test 8: always validate when no budget data --

    def test_always_validate_when_no_budget_data(self) -> None:
        """expected_actions=None => always validate (D-03)."""
        img_before = make_test_image(color=(0, 0, 0))
        img_after = make_test_image(color=(255, 255, 255))
        images = [img_before, img_before, img_after, img_after,
                  img_after, img_after, img_after, img_after]

        steps = [
            PlanStep(number=1, goal="Step 1", expected_actions=None),
            PlanStep(number=2, goal="Step 2"),
        ]
        agent, planner, runner, events = self._build(steps, images=images)
        result = runner.run("task", "fake_screenshot", on_progress=events.append)

        assert result.status == "completed"
        # Step 1 validated (no budget data)
        assert 1 in planner._validate_calls

    # -- Test 9: always-validate before skip (D-05) --

    def test_always_validate_before_skip(self) -> None:
        """When skip AND always-validate both apply, always-validate wins."""
        img_before = make_test_image(color=(0, 0, 0))
        img_after = make_test_image(color=(255, 255, 255))

        step = PlanStep(number=1, goal="Reboot step", expected_actions=10,
                        reboot_expected=True)
        agent, planner, runner, events = self._build([step])
        result_mock = TaskResult(
            status=TaskStatus.COMPLETED,
            reason="ok",
            total_steps=1,  # <50% of 10 => skip condition met
            total_input_tokens=0,
            total_output_tokens=0,
            total_duration_s=0.0,
        )
        # Both skip conditions (screen changed, low budget) AND
        # always-validate (reboot) are true => should NOT skip
        should_skip = runner._should_skip_validation(
            step=step,
            result=result_mock,
            pre_step_image=img_before,
            post_step_image=img_after,
            is_last_step=False,
        )
        assert not should_skip

    # -- Test 10: progress reports validation_skipped --

    def test_progress_reports_validation_skipped(self) -> None:
        """When validation is skipped, on_progress has validation_skipped event."""
        img_before = make_test_image(color=(0, 0, 0))
        img_after = make_test_image(color=(255, 255, 255))
        images = [img_before, img_before, img_after, img_after,
                  img_after, img_after, img_after, img_after]

        steps = [
            PlanStep(number=1, goal="Step 1", expected_actions=10),
            PlanStep(number=2, goal="Step 2", expected_actions=10),
        ]
        agent, planner, runner, events = self._build(steps, images=images)
        result = runner.run("task", "fake_screenshot", on_progress=events.append)

        skip_events = [e for e in events if e.get("type") == "validation_skipped"]
        assert len(skip_events) >= 1
        assert skip_events[0]["step_number"] == 1

    # -- Test 11: progress reports validation ran (no validation_skipped) --

    def test_progress_reports_validation_ran(self) -> None:
        """When validation runs normally, no validation_skipped event for step 1."""
        same_img = make_test_image(color=(128, 128, 128))
        images = [same_img] * 20

        steps = [
            PlanStep(number=1, goal="Step 1", expected_actions=10),
            PlanStep(number=2, goal="Step 2", expected_actions=10),
        ]
        agent, planner, runner, events = self._build(steps, images=images)
        result = runner.run("task", "fake_screenshot", on_progress=events.append)

        skip_events = [e for e in events if e.get("type") == "validation_skipped"]
        # No validation_skipped for step 1 (screen unchanged => always validate)
        step1_skips = [e for e in skip_events if e.get("step_number") == 1]
        assert len(step1_skips) == 0
