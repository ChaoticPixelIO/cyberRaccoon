"""Tests for the WorkflowRunner replan-decision gate (Phase 3 — REPLAN-04/05/06).

Full end-to-end path tests (TestPathA / TestPathB / TestPathC /
TestLazyReplan / TestLoopSkipsCancelled / TestRetryStepOnce) are populated
after Task 3 wires the call sites.
"""
from __future__ import annotations

import threading
import time
from typing import Any

import pytest

try:
    from cyberraccoon.agent.planner import (
        PlanStep,
        StepVerification,
        TaskPlanner,
    )
    from cyberraccoon.agent.workflow_runner import (
        WorkflowRunner,
        _cancel_and_append,
    )
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _AVAILABLE,
    reason="Phase 3 plan 02 not yet landed",
)

# Reuse helpers from existing test_workflow_runner
from tests.test_agent.test_workflow_runner import MockPlanner, _make_agent
from tests.test_agent.conftest import MockCapture, MockExecutor, MockProtocol
from cyberraccoon.agent.vision_agent import VisionAgent


class CarryoverMockProtocol(MockProtocol):
    """MockProtocol that preserves call_count across run boundaries.

    VisionAgent.run() calls protocol.reset() at every invocation, which
    zeroes MockProtocol.call_count and would cause a multi-step Path C test
    (step N escalates; step N+1 needs the next response) to replay the
    escalate response instead of advancing. This subclass keeps call_count
    intact so each entry in ``responses`` is consumed exactly once across
    the full workflow, which is what Path C resume/replan tests need.
    """

    def reset(self) -> None:
        # Intentionally do NOT reset call_count — keep consuming responses
        # linearly across step boundaries.
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self.report_result_calls.clear()


def _make_agent_carryover(responses: list[dict] | None) -> VisionAgent:
    """Build a VisionAgent with a CarryoverMockProtocol."""
    return VisionAgent(
        capture=MockCapture(),
        protocol=CarryoverMockProtocol(responses),
        executor=MockExecutor(),
        max_steps=10,
        max_consecutive_failures=3,
        post_action_delay_s=0,
        task_timeout_s=60.0,
        stability_check=False,
    )


class VerifyMockPlanner(MockPlanner):
    """MockPlanner with verify_step + replan call counters (Phase 3)."""

    def __init__(
        self,
        plan_steps: list[PlanStep] | None = None,
        replan_steps: list[PlanStep] | None = None,
        verify_responses: list[StepVerification] | None = None,
    ) -> None:
        super().__init__(plan_steps=plan_steps, replan_steps=replan_steps)
        self._verify_responses = verify_responses or []
        self._verify_call_count = 0
        self._replan_call_count = 0

    def verify_step(
        self, step: PlanStep, observed_screenshot: str,
        skill_text: str | None = None,
    ) -> StepVerification:
        if not self._verify_responses:
            return StepVerification(
                verified=True,
                expected=step.expected_outcome,
                observed="default",
            )
        idx = min(self._verify_call_count, len(self._verify_responses) - 1)
        self._verify_call_count += 1
        return self._verify_responses[idx]

    def replan(self, **kwargs: Any) -> Any:
        self._replan_call_count += 1
        return super().replan(**kwargs)


class TestGateBlocking:
    """REPLAN-05: gate blocks indefinitely until decision or abort."""

    def test_submit_replan_decision_unblocks_gate(self) -> None:
        agent = _make_agent([{"action": "done", "reason": "ok"}])
        planner = VerifyMockPlanner([PlanStep(number=1, goal="step")])
        runner = WorkflowRunner(agent, planner, auto_approve=True)

        choices_seen: list[str] = []

        def caller_thread() -> None:
            choice = runner._await_replan_decision(
                path="A",
                payload={"step_number": 1, "step_goal": "x"},
                on_progress=lambda e: None,
            )
            choices_seen.append(choice)

        t = threading.Thread(target=caller_thread)
        t.start()
        time.sleep(0.4)
        assert t.is_alive(), "Gate should still be blocking after 0.4s"
        assert choices_seen == []
        runner.submit_replan_decision("replan")
        t.join(timeout=2.0)
        assert not t.is_alive(), "Gate did not unblock within 2s after decision"
        assert choices_seen == ["replan"]

    def test_gate_unblocks_on_abort(self) -> None:
        agent = _make_agent([{"action": "done", "reason": "ok"}])
        planner = VerifyMockPlanner([PlanStep(number=1, goal="step")])
        runner = WorkflowRunner(agent, planner, auto_approve=True)

        result_box: list[str] = []

        def caller_thread() -> None:
            choice = runner._await_replan_decision(
                path="B",
                payload={"step_number": 1, "step_goal": "x"},
                on_progress=lambda e: None,
            )
            result_box.append(choice)

        t = threading.Thread(target=caller_thread)
        t.start()
        time.sleep(0.3)
        agent._abort_event.set()
        t.join(timeout=2.0)
        assert not t.is_alive()
        assert result_box == ["abort"]

    def test_gate_returns_choice_string(self) -> None:
        agent = _make_agent([{"action": "done", "reason": "ok"}])
        planner = VerifyMockPlanner([PlanStep(number=1, goal="step")])
        runner = WorkflowRunner(agent, planner, auto_approve=True)
        # Path A allows continue/replan/abort
        for choice in ("continue", "replan", "abort"):
            box: list[str] = []

            def caller() -> None:
                box.append(runner._await_replan_decision(
                    path="A", payload={}, on_progress=None,
                ))
            t = threading.Thread(target=caller)
            t.start()
            time.sleep(0.1)
            runner.submit_replan_decision(choice)
            t.join(timeout=1.0)
            assert box == [choice]


class TestAutoReplan:
    """REPLAN-06: Auto Re-plan skips Path A/B dialog, never Path C. Also emits replan_dialog_resolved."""

    def test_auto_replan_skips_path_a_dialog(self) -> None:
        agent = _make_agent([{"action": "done", "reason": "ok"}])
        planner = VerifyMockPlanner([PlanStep(number=1, goal="step")])
        runner = WorkflowRunner(
            agent, planner, auto_approve=True, auto_replan=True,
        )
        events: list[dict] = []
        choice = runner._await_replan_decision(
            path="A",
            payload={"step_number": 1, "step_goal": "x"},
            on_progress=events.append,
        )
        assert choice == "replan"
        assert not any(e.get("type") == "replan_dialog" for e in events)
        assert any(e.get("type") == "replan_auto" for e in events)
        # H8 — replan_dialog_resolved also emitted in auto-replan so stale dialogs close.
        assert any(e.get("type") == "replan_dialog_resolved" for e in events)

    def test_auto_replan_skips_path_b_dialog(self) -> None:
        agent = _make_agent([{"action": "done", "reason": "ok"}])
        planner = VerifyMockPlanner([PlanStep(number=1, goal="step")])
        runner = WorkflowRunner(
            agent, planner, auto_approve=True, auto_replan=True,
        )
        events: list[dict] = []
        choice = runner._await_replan_decision(
            path="B",
            payload={"step_number": 1, "step_goal": "x"},
            on_progress=events.append,
        )
        assert choice == "replan"
        assert not any(e.get("type") == "replan_dialog" for e in events)
        assert any(e.get("type") == "replan_dialog_resolved" for e in events)

    def test_auto_replan_set_via_setter(self) -> None:
        agent = _make_agent([{"action": "done", "reason": "ok"}])
        planner = VerifyMockPlanner([PlanStep(number=1, goal="step")])
        runner = WorkflowRunner(agent, planner, auto_approve=True)
        assert runner._auto_replan is False
        runner.set_auto_replan(True)
        assert runner._auto_replan is True


class TestChoiceValidation:
    """H5 — submit_replan_decision rejects invalid choice + raises RuntimeError if no gate."""

    def test_rejects_invalid_choice_for_path_a(self) -> None:
        agent = _make_agent([{"action": "done", "reason": "ok"}])
        planner = VerifyMockPlanner([PlanStep(number=1, goal="step")])
        runner = WorkflowRunner(agent, planner, auto_approve=True)
        runner._active_gate = "replan_A"
        # Path A does not accept 'retry' or 'resume'
        with pytest.raises(ValueError, match="invalid choice"):
            runner.submit_replan_decision("retry")
        with pytest.raises(ValueError, match="invalid choice"):
            runner.submit_replan_decision("resume")
        with pytest.raises(ValueError, match="invalid choice"):
            runner.submit_replan_decision("garbage")

    def test_rejects_invalid_choice_for_path_b(self) -> None:
        agent = _make_agent([{"action": "done", "reason": "ok"}])
        planner = VerifyMockPlanner([PlanStep(number=1, goal="step")])
        runner = WorkflowRunner(agent, planner, auto_approve=True)
        runner._active_gate = "replan_B"
        # Path B does not accept 'continue' or 'resume'
        with pytest.raises(ValueError, match="invalid choice"):
            runner.submit_replan_decision("continue")
        with pytest.raises(ValueError, match="invalid choice"):
            runner.submit_replan_decision("resume")

    def test_rejects_invalid_choice_for_path_c(self) -> None:
        agent = _make_agent([{"action": "done", "reason": "ok"}])
        planner = VerifyMockPlanner([PlanStep(number=1, goal="step")])
        runner = WorkflowRunner(agent, planner, auto_approve=True)
        runner._active_gate = "escalation_C"
        # Path C does not accept 'continue' or 'retry'
        with pytest.raises(ValueError, match="invalid choice"):
            runner.submit_replan_decision("continue")
        with pytest.raises(ValueError, match="invalid choice"):
            runner.submit_replan_decision("retry")

    def test_accepts_valid_choices_per_gate(self) -> None:
        agent = _make_agent([{"action": "done", "reason": "ok"}])
        planner = VerifyMockPlanner([PlanStep(number=1, goal="step")])
        runner = WorkflowRunner(agent, planner, auto_approve=True)
        # Path A
        runner._active_gate = "replan_A"
        for choice in ("continue", "replan", "abort"):
            runner._replan_decision.clear()
            runner.submit_replan_decision(choice)
            assert runner._replan_choice == choice
        # Path B
        runner._active_gate = "replan_B"
        for choice in ("retry", "replan", "abort"):
            runner._replan_decision.clear()
            runner.submit_replan_decision(choice)
            assert runner._replan_choice == choice
        # Path C
        runner._active_gate = "escalation_C"
        for choice in ("resume", "replan", "abort"):
            runner._replan_decision.clear()
            runner.submit_replan_decision(choice)
            assert runner._replan_choice == choice

    def test_raises_runtime_error_when_no_gate_armed(self) -> None:
        agent = _make_agent([{"action": "done", "reason": "ok"}])
        planner = VerifyMockPlanner([PlanStep(number=1, goal="step")])
        runner = WorkflowRunner(agent, planner, auto_approve=True)
        assert runner._active_gate is None
        with pytest.raises(RuntimeError, match="no active replan gate"):
            runner.submit_replan_decision("replan")


class TestDecisionRaceCondition:
    """H3 — a fast submit between emit and wait must be captured, not lost."""

    def test_synchronous_submit_before_wait_is_captured(self) -> None:
        """Simulates the race: on_progress handler SYNCHRONOUSLY submits a decision
        from the same thread (e.g., a test harness) before _await_replan_decision
        reaches its wait-loop. H3 guarantees this decision is NOT lost."""
        agent = _make_agent([{"action": "done", "reason": "ok"}])
        planner = VerifyMockPlanner([PlanStep(number=1, goal="step")])
        runner = WorkflowRunner(agent, planner, auto_approve=True)

        def racing_on_progress(event: dict) -> None:
            # Submit IMMEDIATELY when the dialog emits — mimics a fast
            # network roundtrip where the decision arrives between
            # event emission and wait-loop entry.
            if event.get("type") == "replan_dialog":
                runner.submit_replan_decision("continue")

        # Pre-set the event and choice to simulate a stale prior set() that
        # the clear-before-emit logic must wipe:
        runner._replan_decision.set()
        runner._replan_choice = "STALE"

        choice = runner._await_replan_decision(
            path="A",
            payload={"step_number": 1, "step_goal": "x"},
            on_progress=racing_on_progress,
        )
        # Must see 'continue' (the fresh submit), not 'STALE' (the prior state).
        assert choice == "continue"

    def test_stale_event_does_not_unblock_new_dialog(self) -> None:
        """H3 — if _replan_decision was set by a prior cycle, the helper must
        clear it BEFORE emit so the new dialog does not immediately
        return the prior (now-irrelevant) choice."""
        agent = _make_agent([{"action": "done", "reason": "ok"}])
        planner = VerifyMockPlanner([PlanStep(number=1, goal="step")])
        runner = WorkflowRunner(agent, planner, auto_approve=True)
        # Seed stale state
        runner._replan_decision.set()
        runner._replan_choice = "PRIOR"

        box: list[str] = []

        def caller() -> None:
            box.append(runner._await_replan_decision(
                path="A",
                payload={"step_number": 1, "step_goal": "x"},
                on_progress=None,
            ))

        t = threading.Thread(target=caller)
        t.start()
        # If H3 is NOT implemented, the helper would immediately return
        # "PRIOR" because _replan_decision was still set. With H3,
        # the helper clears it before waiting and blocks.
        time.sleep(0.3)
        assert t.is_alive(), "H3 violation: helper returned stale choice without waiting"
        runner.submit_replan_decision("continue")
        t.join(timeout=1.0)
        assert box == ["continue"]


class TestDialogResolvedEmit:
    """H8 — replan_dialog_resolved fires on every choice (continue, retry, replan, resume, abort).

    Helper-level test: the helper does NOT emit replan_dialog_resolved itself
    (that's the runner's branch-site responsibility in Task 3). These tests
    verify the AUTO-REPLAN path does emit it (covered in TestAutoReplan).
    End-to-end emit on user decision is verified in TestPathA/B/C after
    Task 3 wires the branch-site emits.
    """

    def test_auto_replan_path_emits_resolved(self) -> None:
        agent = _make_agent([{"action": "done", "reason": "ok"}])
        planner = VerifyMockPlanner([PlanStep(number=1, goal="step")])
        runner = WorkflowRunner(
            agent, planner, auto_approve=True, auto_replan=True,
        )
        events: list[dict] = []
        runner._await_replan_decision(
            path="A",
            payload={"step_number": 1, "step_goal": "x"},
            on_progress=events.append,
        )
        resolved = [e for e in events if e.get("type") == "replan_dialog_resolved"]
        assert len(resolved) == 1
        assert resolved[0]["choice"] == "replan"


# ---------------------------------------------------------------------------
# End-to-end path tests (Task 3 — workflow wiring complete)
# ---------------------------------------------------------------------------


def _submit_after_delay(runner: WorkflowRunner, choice: str, delay: float = 0.4) -> threading.Thread:
    """Spawn a thread that polls until a gate is armed then submits choice.

    Using polling (not a one-shot sleep+submit) makes tests robust against
    variable verify_step latency. Caller .join()s the thread after runner.run().
    """
    def decider() -> None:
        time.sleep(delay)
        for _ in range(40):  # ~4 seconds of polling
            try:
                runner.submit_replan_decision(choice)
                return
            except RuntimeError:
                time.sleep(0.1)
    t = threading.Thread(target=decider)
    t.start()
    return t


class TestPathA:
    """REPLAN-01: verifier mismatch → replan_dialog (warning) → choice routing."""

    def test_path_a_emits_replan_dialog_with_warning_payload(self) -> None:
        agent = _make_agent([{"action": "done", "reason": "ok"}])
        planner = VerifyMockPlanner(
            plan_steps=[
                PlanStep(number=1, goal="Open Chrome",
                         expected_outcome="Chrome visible"),
                PlanStep(number=2, goal="Go to site"),
            ],
            replan_steps=[PlanStep(number=0, goal="recovered step")],
            verify_responses=[
                StepVerification(
                    verified=False,
                    expected="Chrome visible",
                    observed="Edge visible",
                    mismatch_reason="Wrong browser launched",
                    confidence=0.9,
                ),
                # Second verify returns True so step 2 doesn't re-trigger Path A.
                # Without this, VerifyMockPlanner repeats responses[-1] and the
                # test hangs because a second gate arms with no submitter.
                StepVerification(verified=True, expected="site loaded", observed="site visible"),
            ],
        )
        runner = WorkflowRunner(agent, planner, auto_approve=True)

        t = _submit_after_delay(runner, "continue", delay=0.4)
        events: list[dict] = []
        runner.run("task", "fake_screenshot", on_progress=events.append)
        t.join(timeout=2.0)

        dialog_events = [e for e in events if e.get("type") == "replan_dialog"]
        assert len(dialog_events) >= 1
        evt = dialog_events[0]
        assert evt["path"] == "A"
        assert evt["step_number"] == 1
        assert evt["step_goal"] == "Open Chrome"
        assert evt["expected"] == "Chrome visible"
        assert evt["observed"] == "Edge visible"
        assert evt["mismatch_reason"] == "Wrong browser launched"
        assert evt["failure_reason"] is None
        assert "screenshot_base64" in evt
        assert any(
            e.get("type") == "replan_dialog_resolved" and e.get("choice") == "continue"
            for e in events
        )

    def test_path_a_continue_choice_advances_without_replan_call(self) -> None:
        agent = _make_agent([
            {"action": "done", "reason": "ok"},
            {"action": "done", "reason": "ok"},
        ])
        planner = VerifyMockPlanner(
            plan_steps=[
                PlanStep(number=1, goal="step1"),
                PlanStep(number=2, goal="step2"),
            ],
            replan_steps=[PlanStep(number=0, goal="recovered")],
            verify_responses=[
                StepVerification(verified=False, expected="x",
                                 observed="y", mismatch_reason="z"),
                StepVerification(verified=True, expected="x", observed="y"),
            ],
        )
        runner = WorkflowRunner(agent, planner, auto_approve=True)

        t = _submit_after_delay(runner, "continue", delay=0.4)
        result = runner.run("task", "fake_screenshot")
        t.join(timeout=2.0)

        assert planner._replan_call_count == 0
        assert result.status == "completed"

    def test_path_a_replan_choice_calls_replan_and_marks_cancelled(self) -> None:
        agent = _make_agent([
            {"action": "done", "reason": "ok"},
            {"action": "done", "reason": "ok"},
        ])
        planner = VerifyMockPlanner(
            plan_steps=[
                PlanStep(number=1, goal="step1"),
                PlanStep(number=2, goal="step2"),
                PlanStep(number=3, goal="step3"),
            ],
            replan_steps=[PlanStep(number=0, goal="recovered")],
            verify_responses=[
                StepVerification(verified=False, expected="x",
                                 observed="y", mismatch_reason="z"),
                StepVerification(verified=True, expected="x", observed="y"),
            ],
        )
        runner = WorkflowRunner(agent, planner, auto_approve=True)

        t = _submit_after_delay(runner, "replan", delay=0.4)
        events: list[dict] = []
        runner.run("task", "fake_screenshot", on_progress=events.append)
        t.join(timeout=2.0)

        assert planner._replan_call_count == 1
        replanned = [e for e in events if e.get("type") == "replanned"]
        assert len(replanned) == 1
        assert "cancelled_step_numbers" in replanned[0]
        # Path A calls _cancel_and_append(steps, i, new_steps) where i is the
        # current step's index. Steps after i get cancelled.
        assert sorted(replanned[0]["cancelled_step_numbers"]) == [2, 3]
        assert any(
            e.get("type") == "replan_dialog_resolved" and e.get("choice") == "replan"
            for e in events
        )


class TestPathB:
    """REPLAN-02: step failure after retries → replan_dialog (error) → choice routing."""

    def test_path_b_emits_replan_dialog_with_error_payload(self) -> None:
        # Use None responses so VisionAgent fails immediately (via consecutive
        # failures), not via max_steps loops. This avoids the ThreadPoolExecutor
        # 0.5s-per-step overhead that would exceed the _submit_after_delay window.
        agent = _make_agent([None])
        planner = VerifyMockPlanner(
            plan_steps=[PlanStep(number=1, goal="Click Save")],
            replan_steps=[PlanStep(number=0, goal="recovered")],
        )
        runner = WorkflowRunner(agent, planner, auto_approve=True)

        t = _submit_after_delay(runner, "abort", delay=0.4)
        events: list[dict] = []
        result = runner.run("task", "fake_screenshot", on_progress=events.append)
        t.join(timeout=2.0)

        dialog_events = [e for e in events if e.get("type") == "replan_dialog"]
        assert any(e["path"] == "B" for e in dialog_events)
        b = next(e for e in dialog_events if e["path"] == "B")
        assert b["step_number"] == 1
        assert b["step_goal"] == "Click Save"
        assert b["failure_reason"] is not None
        assert b["expected"] is None
        assert b["observed"] is None
        assert "screenshot_base64" in b
        assert result.status == "aborted"
        assert any(
            e.get("type") == "replan_dialog_resolved" and e.get("choice") == "abort"
            for e in events
        )

    def test_path_b_replan_choice_calls_replan(self) -> None:
        # Use a protocol that switches to success after replan.
        from tests.test_agent.conftest import MockProtocol, MockCapture, MockExecutor
        from cyberraccoon.agent.vision_agent import VisionAgent as _VA
        run_count = {"value": 0}

        class FailThenSucceedProtocol(MockProtocol):
            def reset(self) -> None:
                run_count["value"] += 1
                if run_count["value"] >= 2:
                    self.responses = [{"action": "done", "reason": "ok"}]
                super().reset()

        agent = _VA(
            capture=MockCapture(),
            protocol=FailThenSucceedProtocol([None]),
            executor=MockExecutor(),
            max_steps=10,
            max_consecutive_failures=3,
            post_action_delay_s=0,
            task_timeout_s=60.0,
            stability_check=False,
        )
        planner = VerifyMockPlanner(
            plan_steps=[PlanStep(number=1, goal="step")],
            replan_steps=[PlanStep(number=0, goal="recovered")],
        )
        runner = WorkflowRunner(
            agent, planner, auto_approve=True,
            max_retries_per_step=0, max_replans=2,
        )

        t = _submit_after_delay(runner, "replan", delay=0.4)
        runner.run("task", "fake_screenshot")
        t.join(timeout=2.0)
        assert planner._replan_call_count == 1


class TestRetryStepOnce:
    """H4 — Path B 'retry' choice invokes _retry_step_once with full bookkeeping."""

    def test_retry_failure_auto_aborts_without_dialog_reemit(self) -> None:
        """Retry failed too — auto-abort (H4 decision (a)). Must NOT re-emit Path B."""
        # Use None responses so VisionAgent fails immediately (consecutive failures),
        # not via max_steps loops. This keeps the test fast and avoids
        # exceeding the _submit_after_delay polling window.
        agent = _make_agent([None])
        planner = VerifyMockPlanner(
            plan_steps=[PlanStep(number=1, goal="doomed step")],
            replan_steps=[PlanStep(number=0, goal="r")],
        )
        runner = WorkflowRunner(agent, planner, auto_approve=True)

        t = _submit_after_delay(runner, "retry", delay=0.4)
        events: list[dict] = []
        result = runner.run("task", "fake_screenshot", on_progress=events.append)
        t.join(timeout=2.0)

        assert result.status == "failed"
        assert "after Retry Once" in result.reason
        # Must NOT have re-emitted the Path B dialog
        dialog_events = [
            e for e in events
            if e.get("type") == "replan_dialog" and e.get("path") == "B"
        ]
        assert len(dialog_events) == 1, (
            "Path B dialog emitted more than once (H4 violation)"
        )


class TestPathC:
    """REPLAN-03: escalation gate routes through unified submit_replan_decision.

    Path C is triggered by VisionAgent completion_status='escalate' (new signal,
    Phase 3 deviation from the plan text — the original validate_plan escalation
    verdict is gone, and completion_status is the cleanest live trigger).
    """

    def test_path_c_resume_choice_unchanged_behavior(self) -> None:
        # Use carryover protocol so step 2's run() sees response index 1
        # (the plain "done" ok) instead of replaying the escalate.
        agent = _make_agent_carryover([
            {"action": "done", "status": "escalate", "reason": "Login required"},
            {"action": "done", "reason": "ok"},
        ])
        planner = VerifyMockPlanner(
            plan_steps=[PlanStep(number=1, goal="step1"),
                        PlanStep(number=2, goal="step2")],
            replan_steps=[PlanStep(number=0, goal="recovered")],
        )
        runner = WorkflowRunner(agent, planner, auto_approve=True)

        t = _submit_after_delay(runner, "resume", delay=0.5)
        events: list[dict] = []
        runner.run("task", "fake_screenshot", on_progress=events.append)
        t.join(timeout=3.0)
        assert planner._replan_call_count == 0
        assert any(
            e.get("type") == "replan_dialog_resolved" and e.get("choice") == "resume"
            for e in events
        )

    def test_path_c_replan_choice_calls_replan_and_emits_replanned(self) -> None:
        # Carryover protocol: step 1 escalates (idx 0); after replan, the
        # recovered step consumes idx 1 (plain "done" ok) instead of
        # replaying escalate.
        agent = _make_agent_carryover([
            {"action": "done", "status": "escalate", "reason": "Login required"},
            {"action": "done", "reason": "ok"},
        ])
        planner = VerifyMockPlanner(
            plan_steps=[PlanStep(number=1, goal="step1"),
                        PlanStep(number=2, goal="step2")],
            replan_steps=[PlanStep(number=0, goal="recovered")],
        )
        runner = WorkflowRunner(agent, planner, auto_approve=True, max_replans=2)

        t = _submit_after_delay(runner, "replan", delay=0.5)
        events: list[dict] = []
        runner.run("task", "fake_screenshot", on_progress=events.append)
        t.join(timeout=3.0)

        assert planner._replan_call_count == 1
        assert any(
            e.get("type") == "replanned" and "cancelled_step_numbers" in e
            for e in events
        )
        assert any(
            e.get("type") == "replan_dialog_resolved" and e.get("choice") == "replan"
            for e in events
        )

    def test_path_c_abort_terminates_workflow(self) -> None:
        agent = _make_agent([
            {"action": "done", "status": "escalate", "reason": "x"},
        ])
        planner = VerifyMockPlanner(
            plan_steps=[PlanStep(number=1, goal="s1"),
                        PlanStep(number=2, goal="s2")],
            replan_steps=[PlanStep(number=0, goal="r")],
        )
        runner = WorkflowRunner(agent, planner, auto_approve=True)

        t = _submit_after_delay(runner, "abort", delay=0.5)
        events: list[dict] = []
        result = runner.run("task", "fake_screenshot", on_progress=events.append)
        t.join(timeout=3.0)
        assert result.status == "aborted"
        assert any(
            e.get("type") == "replan_dialog_resolved" and e.get("choice") == "abort"
            for e in events
        )

    def test_auto_replan_does_not_bypass_path_c_escalation(self) -> None:
        """TC1 (review test-coverage gap): Auto Re-plan must NEVER bypass
        Path C — escalations exist precisely for CAPTCHA/2FA/credential
        situations that require human input. The bypass logic uses
        `path in ("A", "B")` and Path C never calls _await_replan_decision.

        Regression-guard: a refactor that loosens the path filter or
        unifies the gate sites under one auto-bypass would silently
        auto-resolve escalations. This test must FAIL if that happens."""
        agent = _make_agent_carryover([
            {"action": "done", "status": "escalate", "reason": "2FA prompt"},
            {"action": "done", "reason": "ok"},
        ])
        planner = VerifyMockPlanner(
            plan_steps=[PlanStep(number=1, goal="login"),
                        PlanStep(number=2, goal="next")],
            replan_steps=[PlanStep(number=0, goal="recovered")],
        )
        # Auto Re-plan ENABLED at construction
        runner = WorkflowRunner(
            agent, planner, auto_approve=True, auto_replan=True,
        )
        assert runner._auto_replan is True

        t = _submit_after_delay(runner, "resume", delay=0.5)
        events: list[dict] = []
        runner.run("task", "fake_screenshot", on_progress=events.append)
        t.join(timeout=3.0)

        # The escalate event MUST have been emitted (proving the gate
        # armed) — auto-replan must not have synthesized a bypass.
        escalate_events = [e for e in events if e.get("type") == "escalate"]
        assert len(escalate_events) == 1, (
            f"Auto Re-plan bypassed Path C — found {len(escalate_events)} "
            "escalate events instead of 1. CRITICAL safety regression."
        )
        # And NO replan_auto event for Path C (bypass marker).
        auto_events = [e for e in events if e.get("type") == "replan_auto"]
        for e in auto_events:
            assert e.get("path") != "C", (
                "Path C must never emit replan_auto — escalations require "
                "explicit user decision"
            )
        # Real user decision was honored.
        assert any(
            e.get("type") == "replan_dialog_resolved" and e.get("choice") == "resume"
            for e in events
        )


class TestRetrySuccessBookkeeping:
    """TC2 (review test-coverage gap): when Path B 'retry' SUCCEEDS, all
    bookkeeping that the normal-step success branch performs must also
    happen — budget_history, step_done event, step.status='done',
    step_results entry. Previously only the failure path was tested
    (test_retry_failure_auto_aborts_without_dialog_reemit)."""

    def test_retry_succeeds_mirrors_normal_step_bookkeeping(self) -> None:
        # Step 1 fails on first attempt (drives Path B), then succeeds on
        # retry. Step 2 immediately succeeds. Carryover protocol so the
        # response index advances across step boundaries.
        agent = _make_agent_carryover([
            None,                                      # idx 0: step 1 attempt 1 (fail)
            {"action": "done", "reason": "ok"},        # idx 1: retry succeeds
            {"action": "done", "reason": "ok"},        # idx 2: step 2
        ])
        plan_step_1 = PlanStep(
            number=1, goal="flaky", expected_actions=2,
        )
        plan_step_2 = PlanStep(
            number=2, goal="next", expected_actions=1,
        )
        planner = VerifyMockPlanner(
            plan_steps=[plan_step_1, plan_step_2],
            replan_steps=[PlanStep(number=0, goal="r")],
        )
        runner = WorkflowRunner(agent, planner, auto_approve=True)

        t = _submit_after_delay(runner, "retry", delay=0.4)
        events: list[dict] = []
        result = runner.run("task", "fake_screenshot", on_progress=events.append)
        t.join(timeout=3.0)

        # Retry succeeded → step marked done, workflow completed normally.
        assert plan_step_1.status == "done", (
            "retry success must set step.status='done' (was: "
            f"{plan_step_1.status!r})"
        )
        # step_done event was emitted for step 1 (the retried step) —
        # previously missing per review I1.
        step_done_events = [
            e for e in events
            if e.get("type") == "step_done" and e.get("step_number") == 1
        ]
        assert len(step_done_events) >= 1, (
            "step_done event must be emitted for retried step 1; was missing "
            "before review I1 fix"
        )
        # Workflow should not be reported as failed.
        assert result.status != "failed", (
            f"workflow status was {result.status!r}; reason={result.reason}"
        )


class TestGateStateRouting:
    """H2 — _active_gate is set when armed and cleared when branch runs."""

    def test_active_gate_cleared_after_decision(self) -> None:
        agent = _make_agent([{"action": "done", "reason": "ok"}])
        planner = VerifyMockPlanner(
            plan_steps=[PlanStep(number=1, goal="s")],
            replan_steps=[PlanStep(number=0, goal="r")],
            verify_responses=[
                StepVerification(verified=False, expected="x",
                                 observed="y", mismatch_reason="z"),
                StepVerification(verified=True, expected="x", observed="y"),
            ],
        )
        runner = WorkflowRunner(agent, planner, auto_approve=True)
        assert runner._active_gate is None

        t = _submit_after_delay(runner, "continue", delay=0.4)
        runner.run("task", "fake_screenshot")
        t.join(timeout=2.0)
        # After completion, gate cleared
        assert runner._active_gate is None


class TestLazyReplan:
    """REPLAN-04: planner.replan only fires after user/auto chooses replan."""

    def test_no_replan_call_when_choice_is_continue(self) -> None:
        agent = _make_agent([
            {"action": "done", "reason": "ok"},
            {"action": "done", "reason": "ok"},
        ])
        planner = VerifyMockPlanner(
            plan_steps=[PlanStep(number=1, goal="s1"),
                        PlanStep(number=2, goal="s2")],
            replan_steps=[PlanStep(number=0, goal="r")],
            verify_responses=[
                StepVerification(verified=False, expected="x",
                                 observed="y", mismatch_reason="z"),
                StepVerification(verified=True, expected="x", observed="y"),
            ],
        )
        runner = WorkflowRunner(agent, planner, auto_approve=True)

        t = _submit_after_delay(runner, "continue", delay=0.4)
        runner.run("task", "fake_screenshot")
        t.join(timeout=2.0)
        assert planner._replan_call_count == 0

    def test_replan_call_when_auto_replan_synthesizes_replan(self) -> None:
        agent = _make_agent([
            {"action": "done", "reason": "ok"},
            {"action": "done", "reason": "ok"},
        ])
        planner = VerifyMockPlanner(
            plan_steps=[PlanStep(number=1, goal="s1"),
                        PlanStep(number=2, goal="s2")],
            replan_steps=[PlanStep(number=0, goal="r")],
            verify_responses=[
                StepVerification(verified=False, expected="x",
                                 observed="y", mismatch_reason="z"),
                StepVerification(verified=True, expected="x", observed="y"),
            ],
        )
        runner = WorkflowRunner(
            agent, planner, auto_approve=True, auto_replan=True,
        )
        runner.run("task", "fake_screenshot")
        assert planner._replan_call_count == 1


class TestLoopSkipsCancelled:
    """STEPS-03: workflow loop skips steps with status=='cancelled'."""

    def test_loop_skips_cancelled_step(self) -> None:
        agent = _make_agent([
            {"action": "done", "reason": "ok"},
            {"action": "done", "reason": "ok"},
        ])
        planner = VerifyMockPlanner(
            plan_steps=[
                PlanStep(number=1, goal="step1"),
                PlanStep(number=2, goal="cancelled_step", status="cancelled"),
                PlanStep(number=3, goal="step3"),
            ],
            replan_steps=[PlanStep(number=0, goal="r")],
        )
        runner = WorkflowRunner(agent, planner, auto_approve=True)
        result = runner.run("task", "fake_screenshot")
        assert result.status == "completed"
        assert result.steps_completed == 2


class TestReplannedPayloadCancelledNumbers:
    """STEPS-04 backend: replanned event includes cancelled_step_numbers list."""

    def test_replanned_includes_cancelled_step_numbers(self) -> None:
        agent = _make_agent([
            {"action": "done", "reason": "ok"},
            {"action": "done", "reason": "ok"},
            {"action": "done", "reason": "ok"},
        ])
        planner = VerifyMockPlanner(
            plan_steps=[
                PlanStep(number=1, goal="s1"),
                PlanStep(number=2, goal="s2"),
                PlanStep(number=3, goal="s3"),
                PlanStep(number=4, goal="s4"),
            ],
            replan_steps=[PlanStep(number=0, goal="r")],
            verify_responses=[
                StepVerification(verified=False, expected="x",
                                 observed="y", mismatch_reason="z"),
                StepVerification(verified=True, expected="x", observed="y"),
                StepVerification(verified=True, expected="x", observed="y"),
            ],
        )
        runner = WorkflowRunner(agent, planner, auto_approve=True, max_replans=2)

        t = _submit_after_delay(runner, "replan", delay=0.4)
        events: list[dict] = []
        runner.run("task", "fake_screenshot", on_progress=events.append)
        t.join(timeout=2.0)
        replanned = [e for e in events if e.get("type") == "replanned"]
        assert len(replanned) >= 1
        evt = replanned[0]
        assert "cancelled_step_numbers" in evt
        assert isinstance(evt["cancelled_step_numbers"], list)
        assert sorted(evt["cancelled_step_numbers"]) == [2, 3, 4]


class TestValidatePlanShimDeleted:
    """H1 staging — after this plan, validate_plan is completely gone."""

    def test_no_validate_plan_in_planner(self) -> None:
        from cyberraccoon.agent import planner as p
        assert not hasattr(p.TaskPlanner, "validate_plan"), (
            "validate_plan shim must be deleted in plan 03-02 after "
            "workflow_runner.py is rewired"
        )

    def test_no_validate_plan_call_in_workflow_runner(self) -> None:
        import inspect
        from cyberraccoon.agent import workflow_runner as w
        source = inspect.getsource(w)
        # Only code comments are allowed to mention validate_plan (none expected).
        assert "validate_plan(" not in source, (
            "workflow_runner.py must not call validate_plan after Task 3 Edit F"
        )


class TestAutoReplanConstructorPropagation:
    """H6 — WorkflowRunner instantiation sites pass auto_replan config."""

    def test_vision_agent_instantiation_uses_config_auto_replan(self) -> None:
        """Grep the VisionAgent source to confirm the kwarg appears."""
        import inspect
        from cyberraccoon.agent import vision_agent as va
        source = inspect.getsource(va)
        assert "auto_replan=" in source, (
            "VisionAgent must pass auto_replan to WorkflowRunner (H6)"
        )
