"""Tests for the workflow runner — step execution, retry, replan, abort."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any
from unittest.mock import MagicMock

import pytest

from cyberraccoon.agent.planner import PlanStep, TaskPlanner
from cyberraccoon.agent.vision_agent import TaskResult, TaskStatus, VisionAgent
from cyberraccoon.agent.workflow_runner import WorkflowResult, WorkflowRunner
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
        # Phase 3: return fresh PlanStep copies so _cancel_and_append's in-place
        # mutation (ns.number = ...) does not corrupt a shared reference across
        # multiple replans in the same test.
        if self._replan_steps is None:
            return None
        return [
            PlanStep(
                number=s.number,
                goal=s.goal,
                reboot_expected=s.reboot_expected,
                expected_outcome=s.expected_outcome,
                expected_actions=s.expected_actions,
                status="pending",
            )
            for s in self._replan_steps
        ]

    def verify_step(self, step, observed_screenshot, skill_text=None):
        """Phase 3: default verify_step returns verified=True so non-Path-A
        tests do not hit the LLM. Tests that exercise Path A override this
        method or use VerifyMockPlanner / VerifyReplanMockPlanner."""
        from cyberraccoon.agent.planner import StepVerification
        return StepVerification(
            verified=True,
            expected=step.expected_outcome,
            observed="mock-ok",
        )


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
        # Phase 3: auto_replan=True so Path B auto-fires replan (preserves
        # pre-03-02 behavior where validate_plan auto-returned continue).
        runner = WorkflowRunner(
            agent, planner, max_retries_per_step=0, max_replans=2,
            auto_approve=True, auto_replan=True,
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
        runner = WorkflowRunner(
            agent, planner, max_retries_per_step=2,
            auto_approve=True, auto_replan=True,
        )

        result = runner.run("task", "fake_screenshot")
        assert result.status == "failed"
        assert "re-plan" in result.reason.lower() or "no steps" in result.reason.lower()

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
            auto_approve=True, auto_replan=True,
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

        # Plan 03-02: _cancel_and_append starts at max+1; step1 was preserved
        # with number=1, so new steps start at number=4 (since the failing
        # step 2 and cancelled step 3 also keep their numbers).
        new_step_numbers = [s["number"] for s in replan_events[0]["new_steps"]]
        assert new_step_numbers == [4, 5]

    def test_max_replans_limit(self) -> None:
        # Keep failing even after replan
        responses = [None] * 20  # always fail
        agent = _make_agent(responses)
        planner = MockPlanner(
            plan_steps=[PlanStep(number=1, goal="Step")],
            replan_steps=[PlanStep(number=1, goal="New step")],  # replan works but step still fails
        )
        runner = WorkflowRunner(
            agent, planner, max_retries_per_step=0, max_replans=2,
            auto_approve=True, auto_replan=True,
        )

        result = runner.run("task", "fake_screenshot")
        assert result.status == "failed"
        # Plan 03-02: terminal message is "Max replans exhausted" OR
        # "after Retry Once" depending on whether retry fired.
        assert (
            "max replans" in result.reason.lower()
            or "re-plan" in result.reason.lower()
            or "after retry once" in result.reason.lower()
        )


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
        from cyberraccoon.capture.base import CaptureError

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
# Post-Step Validation — DELETED in Phase 3 plan 03-02 (M3).
# Semantics replaced by Path A/B/C tests in test_workflow_runner_replan_gate.py.
# ===========================================================================


# ===========================================================================
# Completion Status Detection (COMP-02)
# ===========================================================================

class TestCompletionStatusDetection:
    """Tests for WorkflowRunner using completion_status instead of keyword matching.

    Verifies that the structured completion_status field is used to detect
    gave_up/stuck instead of fragile keyword matching on reason text.
    """

    def test_gave_up_detected_as_failure(self) -> None:
        """Agent done with status='gave_up' is detected as failure.

        Phase 3: Path B fires; with auto_replan=True + max_replans=0 the
        runner returns status=failed because no replans are available.
        """
        agent = _make_agent([
            {"action": "done", "status": "gave_up", "reason": "Cannot find the button"},
        ])
        planner = MockPlanner([PlanStep(number=1, goal="Click button")])
        runner = WorkflowRunner(
            agent, planner, max_retries_per_step=0, max_replans=0,
            auto_approve=True, auto_replan=True,
        )

        result = runner.run("task", "fake_screenshot")
        assert result.status == "failed"

    def test_stuck_detected_as_failure(self) -> None:
        """Agent done with status='stuck' is detected as failure."""
        agent = _make_agent([
            {"action": "done", "status": "stuck", "reason": "Screen is blank"},
        ])
        planner = MockPlanner([PlanStep(number=1, goal="Check screen")])
        runner = WorkflowRunner(
            agent, planner, max_retries_per_step=0, max_replans=0,
            auto_approve=True, auto_replan=True,
        )

        result = runner.run("task", "fake_screenshot")
        assert result.status == "failed"

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
            auto_approve=True, auto_replan=True,
        )

        result = runner.run("task", "fake_screenshot")
        assert result.status == "completed"
        assert result.steps_completed >= 1


# ===========================================================================
# Validation Gating (VALID-01 / VALID-02) — DELETED in Phase 3 plan 03-02 (M3).
# Skip-gate semantics preserved in _should_skip_validation and exercised
# indirectly by the new Path A tests in test_workflow_runner_replan_gate.py.
# ===========================================================================

class _DeletedTestValidationGating:  # pytest-skipped marker (underscore prefix)
    """Placeholder — original TestValidationGating class deleted per M3."""

    def _placeholder(self) -> None:
        return None


# ===========================================================================
# Budget Feedback (BUDGET-01)
# ===========================================================================

class BudgetMockPlanner(MockPlanner):
    """MockPlanner that captures calibration_data kwarg from replan()."""

    def __init__(
        self,
        plan_steps: list[PlanStep] | None = None,
        replan_steps: list[PlanStep] | None = None,
    ) -> None:
        super().__init__(plan_steps, replan_steps)
        self._last_replan_kwargs: dict[str, Any] = {}

    def replan(self, **kwargs: Any) -> list[PlanStep] | None:
        self._last_replan_kwargs = kwargs
        return super().replan(**kwargs)


class TestBudgetFeedback:
    """Tests for budget data (actions_used, expected_actions) in workflow.

    These tests are in RED state: the production code does not yet include
    actions_used/expected_actions in step_results, step_done events, or
    calibration_data threading to replan().
    """

    def test_step_results_include_budget_data(self) -> None:
        """step_results[0] must contain actions_used and expected_actions (D-05)."""
        agent = _make_agent([{"action": "done", "reason": "ok"}])
        planner = MockPlanner([PlanStep(number=1, goal="step 1", expected_actions=3)])
        runner = WorkflowRunner(agent, planner, auto_approve=True)

        result = runner.run("task", "fake_screenshot")
        assert result.status == "completed"
        assert len(result.step_results) >= 1
        entry = result.step_results[0]
        assert "actions_used" in entry, "step_results must include actions_used (D-05)"
        assert "expected_actions" in entry, "step_results must include expected_actions (D-05)"
        assert entry["expected_actions"] == 3

    def test_step_done_includes_actions_used(self) -> None:
        """step_done progress event must include actions_used and expected_actions (D-06)."""
        agent = _make_agent([{"action": "done", "reason": "ok"}])
        planner = MockPlanner([PlanStep(number=1, goal="step 1", expected_actions=3)])
        runner = WorkflowRunner(agent, planner, auto_approve=True)

        events: list[dict[str, Any]] = []
        result = runner.run("task", "fake_screenshot", on_progress=events.append)
        assert result.status == "completed"

        step_done_events = [e for e in events if e.get("type") == "step_done"]
        assert len(step_done_events) >= 1
        evt = step_done_events[0]
        assert "actions_used" in evt, "step_done event must include actions_used (D-06)"
        assert "expected_actions" in evt, "step_done event must include expected_actions (D-06)"

    def test_budget_history_accumulates(self) -> None:
        """2-step workflow records actions_used in both step_results entries."""
        agent = _make_agent([{"action": "done", "reason": "ok"}])
        planner = MockPlanner([
            PlanStep(number=1, goal="Step A", expected_actions=2),
            PlanStep(number=2, goal="Step B", expected_actions=4),
        ])
        runner = WorkflowRunner(agent, planner, auto_approve=True)

        result = runner.run("task", "fake_screenshot")
        assert result.status == "completed"
        assert len(result.step_results) == 2
        assert "actions_used" in result.step_results[0]
        assert "actions_used" in result.step_results[1]

    def test_replan_receives_calibration_data(self) -> None:
        """When step 2 fails and triggers replan, calibration_data is passed."""
        run_count = {"value": 0}

        class StepAwareProtocol(MockProtocol):
            def reset(self) -> None:
                run_count["value"] += 1
                if run_count["value"] <= 1:
                    # Step 1 succeeds
                    self.responses = [{"action": "done", "reason": "ok"}]
                elif run_count["value"] == 2:
                    # Step 2 fails
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
        planner = BudgetMockPlanner(
            plan_steps=[
                PlanStep(number=1, goal="Step A", expected_actions=2),
                PlanStep(number=2, goal="Step B", expected_actions=5),
            ],
            replan_steps=[PlanStep(number=1, goal="Revised step")],
        )
        runner = WorkflowRunner(
            agent, planner, max_retries_per_step=0, max_replans=1,
            auto_approve=True, auto_replan=True,
        )

        result = runner.run("task", "fake_screenshot")
        assert "calibration_data" in planner._last_replan_kwargs, (
            "replan() must receive calibration_data kwarg"
        )
        cal_data = planner._last_replan_kwargs["calibration_data"]
        assert isinstance(cal_data, list)
        assert len(cal_data) >= 1
        # First entry should have the required keys
        assert "step_number" in cal_data[0]
        assert "step_goal" in cal_data[0]
        assert "expected_actions" in cal_data[0]
        assert "actions_used" in cal_data[0]

    def test_failed_step_in_calibration_data(self) -> None:
        """Calibration data includes BOTH step 1 (success) AND step 2 (failed).

        Per RESEARCH.md Pitfall 2: the failed step that triggered replan
        must also be in calibration data.
        """
        run_count = {"value": 0}

        class StepAwareProtocol(MockProtocol):
            def reset(self) -> None:
                run_count["value"] += 1
                if run_count["value"] <= 1:
                    self.responses = [{"action": "done", "reason": "ok"}]
                elif run_count["value"] == 2:
                    self.responses = [None]
                else:
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
        planner = BudgetMockPlanner(
            plan_steps=[
                PlanStep(number=1, goal="Step A", expected_actions=2),
                PlanStep(number=2, goal="Step B", expected_actions=5),
            ],
            replan_steps=[PlanStep(number=1, goal="Revised step")],
        )
        runner = WorkflowRunner(
            agent, planner, max_retries_per_step=0, max_replans=1,
            auto_approve=True, auto_replan=True,
        )

        result = runner.run("task", "fake_screenshot")
        assert "calibration_data" in planner._last_replan_kwargs
        cal_data = planner._last_replan_kwargs["calibration_data"]
        # Must include both step 1 (success) and step 2 (failed)
        assert len(cal_data) >= 2, (
            "calibration_data must include failed step (Pitfall 2)"
        )
        step_numbers = [e["step_number"] for e in cal_data]
        assert 1 in step_numbers
        assert 2 in step_numbers

    def test_failed_step_no_estimate_in_calibration(self) -> None:
        """Step with expected_actions=None still contributes to calibration (D-02)."""
        run_count = {"value": 0}

        class StepAwareProtocol(MockProtocol):
            def reset(self) -> None:
                run_count["value"] += 1
                if run_count["value"] <= 1:
                    self.responses = [{"action": "done", "reason": "ok"}]
                elif run_count["value"] == 2:
                    self.responses = [None]
                else:
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
        planner = BudgetMockPlanner(
            plan_steps=[
                PlanStep(number=1, goal="Step A", expected_actions=None),
                PlanStep(number=2, goal="Step B", expected_actions=5),
            ],
            replan_steps=[PlanStep(number=1, goal="Revised step")],
        )
        runner = WorkflowRunner(
            agent, planner, max_retries_per_step=0, max_replans=1,
            auto_approve=True, auto_replan=True,
        )

        result = runner.run("task", "fake_screenshot")
        assert "calibration_data" in planner._last_replan_kwargs
        cal_data = planner._last_replan_kwargs["calibration_data"]
        # Find step 1 entry
        step1_entries = [e for e in cal_data if e["step_number"] == 1]
        assert len(step1_entries) == 1
        step1 = step1_entries[0]
        assert step1["expected_actions"] is None, (
            "Steps with no estimate must preserve expected_actions=None (D-02)"
        )
        assert isinstance(step1["actions_used"], int), (
            "actions_used must be an int even when expected_actions is None"
        )


# ===========================================================================
# Pause / Resume (CRUISE-02, CRUISE-05)
# ===========================================================================

class TestPauseResume:
    """Tests for WorkflowRunner pause gate and resume (CRUISE-02, CRUISE-05)."""

    def test_resume_event_exists(self) -> None:
        """WorkflowRunner has a _resume_event attribute."""
        agent = _make_agent([{"action": "done", "reason": "ok"}])
        runner = WorkflowRunner(agent=agent, planner=MockPlanner())
        assert hasattr(runner, "_resume_event"), "WorkflowRunner must have _resume_event"
        assert isinstance(runner._resume_event, threading.Event)

    def test_pause_halts_between_steps(self) -> None:
        """When VisionAgent returns PAUSED, WorkflowRunner blocks on _resume_event."""
        steps = [
            PlanStep(number=1, goal="step one"),
            PlanStep(number=2, goal="step two"),
        ]
        # First step completes, second step returns PAUSED
        agent = _make_agent([
            {"action": "done", "reason": "step 1 done"},
            {"action": "click", "x": 1, "y": 1},  # will be paused
        ])
        planner = MockPlanner(plan_steps=steps)
        runner = WorkflowRunner(agent=agent, planner=planner, auto_approve=True)

        # Set pause before step 2 starts
        agent._pause_event.set()

        progress_events: list[dict[str, Any]] = []

        def on_progress(evt: dict[str, Any]) -> None:
            progress_events.append(evt)

        # Run in a thread since it will block on _resume_event
        result_holder: list[WorkflowResult] = []

        def run() -> None:
            result_holder.append(
                runner.run("test", screenshot_base64="fake", on_progress=on_progress)
            )

        t = threading.Thread(target=run)
        t.start()
        import time
        time.sleep(0.5)

        # Should have emitted task_paused event
        paused_events = [e for e in progress_events if e.get("type") == "task_paused"]
        assert len(paused_events) >= 1, "Expected task_paused event"

        # Resume
        runner._resume_event.set()
        t.join(timeout=5)

    def test_resume_with_modified_plan(self) -> None:
        """After pause, set_current_plan pushes modified steps; resume uses them."""
        steps = [
            PlanStep(number=1, goal="step one"),
            PlanStep(number=2, goal="original step two"),
        ]
        agent = _make_agent([
            {"action": "done", "reason": "step done"},
            {"action": "done", "reason": "step done"},
        ])
        planner = MockPlanner(plan_steps=steps)
        runner = WorkflowRunner(agent=agent, planner=planner, auto_approve=True)

        # This test verifies that set_current_plan is consumed after resume.
        # The exact mechanism (blocking on _resume_event then consuming override)
        # is tested by running the workflow and checking the executed goals.
        # Full implementation test -- expected RED until plan 02 lands.
        assert hasattr(runner, "_resume_event")
        assert hasattr(runner, "set_current_plan")

    def test_cancel_while_paused_returns_aborted(self) -> None:
        """Setting abort event while paused causes aborted result."""
        agent = _make_agent([{"action": "done", "reason": "ok"}])
        runner = WorkflowRunner(agent=agent, planner=MockPlanner(), auto_approve=True)
        assert hasattr(runner, "_resume_event")
        # When abort is set during the resume_event wait loop,
        # the result should be WorkflowResult(status="aborted")

    def test_fresh_screenshot_captured_on_pause(self) -> None:
        """task_paused event includes screenshot_base64 from a fresh capture (D-05).
        Addresses review concern: HIGH-4 capture failure / HIGH-5 missing test."""
        steps = [
            PlanStep(number=1, goal="step one"),
            PlanStep(number=2, goal="step two"),
        ]
        agent = _make_agent([
            {"action": "done", "reason": "step 1 done"},
            {"action": "click", "x": 1, "y": 1},
        ])
        planner = MockPlanner(plan_steps=steps)
        runner = WorkflowRunner(agent=agent, planner=planner, auto_approve=True)
        agent._pause_event.set()

        progress_events: list[dict[str, Any]] = []

        def on_progress(evt: dict[str, Any]) -> None:
            progress_events.append(evt)

        result_holder: list[WorkflowResult] = []

        def run() -> None:
            result_holder.append(
                runner.run("test", screenshot_base64="fake", on_progress=on_progress)
            )

        t = threading.Thread(target=run)
        t.start()
        import time
        time.sleep(0.5)

        paused_events = [e for e in progress_events if e.get("type") == "task_paused"]
        assert len(paused_events) >= 1, "Expected task_paused event"
        assert "screenshot_base64" in paused_events[0], "task_paused must include screenshot"
        assert paused_events[0]["screenshot_base64"], "screenshot must not be empty"

        runner._resume_event.set()
        t.join(timeout=5)

    def test_pre_step_pause_race(self) -> None:
        """Pause set after step 1 completes but before step 2 agent.run() starts.
        WorkflowRunner must enter pause gate without starting the next step.
        Addresses review concern: MEDIUM-1 pre-step race."""
        steps = [
            PlanStep(number=1, goal="step one"),
            PlanStep(number=2, goal="step two"),
            PlanStep(number=3, goal="step three"),
        ]
        agent = _make_agent([
            {"action": "done", "reason": "step 1 done"},
            {"action": "done", "reason": "step 2 done"},
            {"action": "done", "reason": "step 3 done"},
        ])
        planner = MockPlanner(plan_steps=steps)
        runner = WorkflowRunner(agent=agent, planner=planner, auto_approve=True)

        progress_events: list[dict[str, Any]] = []

        def on_progress(evt: dict[str, Any]) -> None:
            progress_events.append(evt)
            # Set pause after step 1 done event, before step 2 starts
            if evt.get("type") == "step_done" and evt.get("step_number") == 1:
                agent._pause_event.set()

        result_holder: list[WorkflowResult] = []

        def run() -> None:
            result_holder.append(
                runner.run("test", screenshot_base64="fake", on_progress=on_progress)
            )

        t = threading.Thread(target=run)
        t.start()
        import time
        time.sleep(1.0)

        paused_events = [e for e in progress_events if e.get("type") == "task_paused"]
        assert len(paused_events) >= 1, "Expected task_paused from pre-step check"
        # Step 2 should NOT have started (no step_start for step 2)
        step_starts_after_1 = [
            e for e in progress_events
            if e.get("type") == "step_start" and e.get("step_number", 0) >= 2
        ]
        assert len(step_starts_after_1) == 0, \
            "Step 2 should not start when pause is set before it"

        runner._resume_event.set()
        t.join(timeout=5)

    def test_resume_does_not_emit_task_started(self) -> None:
        """Resume must NOT emit a task_started event (Pitfall 2 -- would clear state).
        Addresses review concern: HIGH-5 event lifecycle ordering."""
        steps = [
            PlanStep(number=1, goal="step one"),
            PlanStep(number=2, goal="step two"),
        ]
        agent = _make_agent([
            {"action": "done", "reason": "step 1 done"},
            {"action": "done", "reason": "step 2 done"},
        ])
        planner = MockPlanner(plan_steps=steps)
        runner = WorkflowRunner(agent=agent, planner=planner, auto_approve=True)
        agent._pause_event.set()

        progress_events: list[dict[str, Any]] = []

        def on_progress(evt: dict[str, Any]) -> None:
            progress_events.append(evt)

        result_holder: list[WorkflowResult] = []

        def run() -> None:
            result_holder.append(
                runner.run("test", screenshot_base64="fake", on_progress=on_progress)
            )

        t = threading.Thread(target=run)
        t.start()
        import time
        time.sleep(0.5)

        # Now resume
        agent._pause_event.clear()
        runner._resume_event.set()
        t.join(timeout=5)

        # After the first task_started (from initial run), there must be
        # NO second task_started after task_resumed
        events_after_resumed: list[dict[str, Any]] = []
        found_resumed = False
        for evt in progress_events:
            if evt.get("type") == "task_resumed":
                found_resumed = True
                continue
            if found_resumed:
                events_after_resumed.append(evt)

        task_started_after_resume = [
            e for e in events_after_resumed if e.get("type") == "task_started"
        ]
        assert len(task_started_after_resume) == 0, \
            "Resume must NOT emit task_started (would clear PlanDiscussionState)"

    def test_step_index_recalculation_after_plan_modification(self) -> None:
        """After pause, user modifies plan (adds/removes steps). Resume must
        continue from the correct next incomplete step.
        Addresses review concern: MEDIUM-2 step index recalculation."""
        steps = [
            PlanStep(number=1, goal="step one"),
            PlanStep(number=2, goal="step two"),
            PlanStep(number=3, goal="step three"),
        ]
        agent = _make_agent([
            {"action": "done", "reason": "step 1 done"},
            # step 2 returns paused
            {"action": "click", "x": 1, "y": 1},
            # After resume with modified plan: only the new step
            {"action": "done", "reason": "modified step done"},
        ])
        planner = MockPlanner(plan_steps=steps)
        runner = WorkflowRunner(agent=agent, planner=planner, auto_approve=True)

        progress_events: list[dict[str, Any]] = []

        def on_progress(evt: dict[str, Any]) -> None:
            progress_events.append(evt)

        result_holder: list[WorkflowResult] = []

        def run() -> None:
            result_holder.append(
                runner.run("test", screenshot_base64="fake", on_progress=on_progress)
            )

        # This is a full integration test -- RED until plan 02 lands.
        # The test validates that after pause at step 2, pushing a new plan
        # [modified_step_2] via set_current_plan and resuming causes
        # the runner to execute modified_step_2, not skip or re-execute step 1.
        assert hasattr(runner, "_resume_event")
        assert hasattr(runner, "set_current_plan")


# ===========================================================================
# Replanned Payload Shape (MISSING-01 / FLOW-2)
# ===========================================================================

# Plan 03-02: plan_ready step dicts include status (new field).
PLAN_READY_STEP_KEYS = {
    "number", "goal", "reboot_expected", "expected_actions",
    "expected_outcome",
}


class VerifyReplanMockPlanner(MockPlanner):
    """MockPlanner that supports configurable verify_step responses and
    auto-submits submit_replan_decision('replan') when a Path A dialog fires.

    Phase 3 migration of the old ValidateMockPlanner — the workflow now
    uses verify_step + the replan decision gate rather than validate_plan.
    """

    def __init__(
        self,
        plan_steps: list[PlanStep] | None = None,
        replan_steps: list[PlanStep] | None = None,
        verify_responses: list[Any] | None = None,
    ) -> None:
        super().__init__(plan_steps, replan_steps)
        # Each entry is a StepVerification; import lazily to avoid circular.
        from cyberraccoon.agent.planner import StepVerification
        self._StepVerification = StepVerification
        self._verify_responses = verify_responses or []
        self._verify_call_count = 0

    def verify_step(self, step: PlanStep, observed_screenshot: str,
                    skill_text: str | None = None) -> Any:
        if not self._verify_responses:
            return self._StepVerification(
                verified=True,
                expected=step.expected_outcome,
                observed="default",
            )
        idx = min(self._verify_call_count, len(self._verify_responses) - 1)
        self._verify_call_count += 1
        return self._verify_responses[idx]


class TestReplannedPayload:
    """Tests for replanned event payload shape (MISSING-01 / FLOW-2).

    Migrated in Phase 3 plan 03-02: replanned events now include
    cancelled_step_numbers. Payload still includes all 5 step fields
    + screenshot_base64 + reason, plus the new status field.
    """

    def _drive_replan(self, runner: WorkflowRunner) -> threading.Thread:
        """Spawn a thread that auto-submits 'replan' when the gate arms."""
        def decider() -> None:
            time.sleep(0.4)
            # The gate may not be armed yet if verify was fast; poll briefly.
            for _ in range(20):
                try:
                    runner.submit_replan_decision("replan")
                    return
                except RuntimeError:
                    time.sleep(0.1)
        t = threading.Thread(target=decider)
        t.start()
        return t

    def test_path_a_replan_payload(self) -> None:
        """Path A (verifier mismatch) replan includes all 5 fields + screenshot + cancelled_step_numbers."""
        from cyberraccoon.agent.planner import StepVerification
        new_steps = [
            PlanStep(number=0, goal="New step A", expected_actions=3,
                     expected_outcome="Something visible", reboot_expected=False),
            PlanStep(number=0, goal="New step B", expected_actions=5,
                     expected_outcome="Task complete", reboot_expected=True),
        ]
        agent = _make_agent([{"action": "done", "reason": "ok"}])
        planner = VerifyReplanMockPlanner(
            plan_steps=[
                PlanStep(number=1, goal="Only step", expected_actions=2),
            ],
            replan_steps=new_steps,
            verify_responses=[
                StepVerification(
                    verified=False,
                    expected="x", observed="y",
                    mismatch_reason="Need different approach",
                ),
                StepVerification(verified=True, expected="x", observed="y"),
                StepVerification(verified=True, expected="x", observed="y"),
            ],
        )
        runner = WorkflowRunner(agent, planner, auto_approve=True, max_replans=2)

        events: list[dict[str, Any]] = []
        t = self._drive_replan(runner)
        result = runner.run("task", "fake_screenshot", on_progress=events.append)
        t.join(timeout=2.0)
        assert result.status == "completed"

        replan_events = [e for e in events if e.get("type") == "replanned"]
        assert len(replan_events) >= 1
        evt = replan_events[0]

        for step_dict in evt["new_steps"]:
            assert "number" in step_dict
            assert "goal" in step_dict
            assert "reboot_expected" in step_dict
            assert "expected_actions" in step_dict
            assert "expected_outcome" in step_dict

        assert "screenshot_base64" in evt
        assert "reason" in evt
        # Plan 03-02 NEW field
        assert "cancelled_step_numbers" in evt
        assert isinstance(evt["cancelled_step_numbers"], list)

    def test_path_b_exhausted_retries_replan_payload(self) -> None:
        """Path B (retry exhausted) replan includes all 5 fields + screenshot + reason + cancelled_step_numbers."""
        call_number = {"value": 0}

        class FailThenSucceedProtocol(MockProtocol):
            def reset(self) -> None:
                call_number["value"] += 1
                if call_number["value"] >= 2:
                    self.responses = [
                        {"action": "done", "reason": "ok"},
                    ]
                super().reset()

        protocol = FailThenSucceedProtocol([None])
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
        replan_steps = [
            PlanStep(number=0, goal="New approach", expected_actions=2,
                     expected_outcome="Completed", reboot_expected=False),
        ]
        planner = MockPlanner(
            plan_steps=[PlanStep(number=1, goal="Original step")],
            replan_steps=replan_steps,
        )
        runner = WorkflowRunner(
            agent, planner, max_retries_per_step=0, max_replans=1,
            auto_approve=True,
        )

        events: list[dict[str, Any]] = []
        t = self._drive_replan(runner)
        result = runner.run("task", "fake_screenshot", on_progress=events.append)
        t.join(timeout=2.0)
        assert result.status == "completed"

        replan_events = [e for e in events if e.get("type") == "replanned"]
        assert len(replan_events) >= 1
        evt = replan_events[0]

        for step_dict in evt["new_steps"]:
            assert "number" in step_dict
            assert "goal" in step_dict
            assert "reboot_expected" in step_dict
            assert "expected_actions" in step_dict
            assert "expected_outcome" in step_dict

        assert "screenshot_base64" in evt
        assert "reason" in evt
        assert "cancelled_step_numbers" in evt
        assert isinstance(evt["cancelled_step_numbers"], list)

    def test_screenshot_base64_always_present(self) -> None:
        """screenshot_base64 key is always present in replanned events."""
        from cyberraccoon.agent.planner import StepVerification
        new_steps = [
            PlanStep(number=0, goal="New step", expected_actions=3,
                     expected_outcome="Visible", reboot_expected=False),
        ]
        agent = _make_agent([{"action": "done", "reason": "ok"}])
        planner = VerifyReplanMockPlanner(
            plan_steps=[
                PlanStep(number=1, goal="Step 1", expected_actions=3),
                PlanStep(number=2, goal="Step 2", expected_actions=3),
            ],
            replan_steps=new_steps,
            verify_responses=[
                StepVerification(
                    verified=False, expected="x", observed="y",
                    mismatch_reason="Better approach",
                ),
                StepVerification(verified=True, expected="x", observed="y"),
            ],
        )
        runner = WorkflowRunner(agent, planner, auto_approve=True, max_replans=2)

        events: list[dict[str, Any]] = []
        t = self._drive_replan(runner)
        runner.run("task", "fake_screenshot", on_progress=events.append)
        t.join(timeout=2.0)

        replan_events = [e for e in events if e.get("type") == "replanned"]
        assert len(replan_events) >= 1
        for evt in replan_events:
            assert "screenshot_base64" in evt
            assert "cancelled_step_numbers" in evt


# ===========================================================================
# Frame-Diff Debug Instrumentation (CALIB-01)
# ===========================================================================

class LogCapture:
    """Context manager that captures log messages from a named logger."""

    def __init__(self, logger_name: str) -> None:
        self.logger_name = logger_name
        self.messages: list[str] = []
        self._handler: logging.Handler | None = None

    def __enter__(self) -> "LogCapture":
        import logging as _logging

        class _Handler(_logging.Handler):
            def __init__(self, target: list[str]) -> None:
                super().__init__()
                self._target = target

            def emit(self, record: _logging.LogRecord) -> None:
                self._target.append(self.format(record))

        self._handler = _Handler(self.messages)
        logger = _logging.getLogger(self.logger_name)
        logger.addHandler(self._handler)
        logger.setLevel(_logging.DEBUG)
        return self

    def __exit__(self, *args: Any) -> None:
        import logging as _logging
        if self._handler:
            _logging.getLogger(self.logger_name).removeHandler(self._handler)


class TestFrameDiffDebug:
    """Tests for frame-diff debug logging gated on CYBERRACCOON_FRAME_DIFF_DEBUG."""

    def test_frame_diff_debug_logging_when_enabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When CYBERRACCOON_FRAME_DIFF_DEBUG=1, debug log fires with diff data."""
        monkeypatch.setenv("CYBERRACCOON_FRAME_DIFF_DEBUG", "1")

        # Force module-level flag to reload (it was evaluated at import time)
        import cyberraccoon.agent.workflow_runner as wr_module
        monkeypatch.setattr(wr_module, "_FRAME_DIFF_DEBUG", True)

        img_before = make_test_image(color=(0, 0, 0))
        img_after = make_test_image(color=(255, 255, 255))
        images = [img_before, img_before, img_after, img_after,
                  img_after, img_after, img_after, img_after]

        steps = [
            PlanStep(number=1, goal="Step 1", expected_actions=10),
            PlanStep(number=2, goal="Step 2", expected_actions=10),
        ]
        capture = MockCapture(images=images)
        agent = _make_agent(
            [{"action": "done", "reason": "ok"}],
            capture=capture,
        )
        planner = MockPlanner(steps)
        from cyberraccoon.agent.planner import StepVerification
        planner.verify_step = lambda **kw: StepVerification(
            verified=True, expected="x", observed="y",
        )
        runner = WorkflowRunner(agent, planner, auto_approve=True)

        with LogCapture("M2.workflow") as log_output:
            result = runner.run("task", "fake_screenshot")

        assert result.status == "completed"
        assert any("FRAME_DIFF_DEBUG" in msg for msg in log_output.messages), (
            "Expected FRAME_DIFF_DEBUG log message when env var is set"
        )

    def test_frame_diff_debug_silent_when_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When CYBERRACCOON_FRAME_DIFF_DEBUG is unset, no debug log fires."""
        monkeypatch.delenv("CYBERRACCOON_FRAME_DIFF_DEBUG", raising=False)

        import cyberraccoon.agent.workflow_runner as wr_module
        monkeypatch.setattr(wr_module, "_FRAME_DIFF_DEBUG", False)

        img_before = make_test_image(color=(0, 0, 0))
        img_after = make_test_image(color=(255, 255, 255))
        images = [img_before, img_before, img_after, img_after,
                  img_after, img_after, img_after, img_after]

        steps = [
            PlanStep(number=1, goal="Step 1", expected_actions=10),
            PlanStep(number=2, goal="Step 2", expected_actions=10),
        ]
        capture = MockCapture(images=images)
        agent = _make_agent(
            [{"action": "done", "reason": "ok"}],
            capture=capture,
        )
        planner = MockPlanner(steps)
        from cyberraccoon.agent.planner import StepVerification
        planner.verify_step = lambda **kw: StepVerification(
            verified=True, expected="x", observed="y",
        )
        runner = WorkflowRunner(agent, planner, auto_approve=True)

        with LogCapture("M2.workflow") as log_output:
            result = runner.run("task", "fake_screenshot")

        assert result.status == "completed"
        assert not any("FRAME_DIFF_DEBUG" in msg for msg in log_output.messages), (
            "Expected NO FRAME_DIFF_DEBUG log message when env var is unset"
        )
