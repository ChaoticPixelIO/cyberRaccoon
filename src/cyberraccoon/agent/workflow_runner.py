"""Workflow runner — executes a plan's steps one at a time through the agent.

Orchestrates the plan-then-execute loop:
  1. Receive steps from the planner
  2. Execute each step via VisionAgent.run() with a narrowly scoped goal
  3. Handle retries, re-planning, and transitions between steps

The runner is generic — it has no knowledge of specific applications (BIOS,
WeChat, etc.). All domain-specific behavior comes from the skill text and
the planner's step decomposition.

Usage::

    from cyberraccoon.agent.planner import TaskPlanner
    from cyberraccoon.agent.workflow_runner import WorkflowRunner

    planner = TaskPlanner(provider, model, api_key)
    runner = WorkflowRunner(agent, planner, capture)
    result = runner.run(task_goal, skill_text, on_progress=callback)
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from cyberraccoon.agent.planner import (
    LLMAuthError,
    LLMTransportError,
    PlanStep,
    StepVerification,
    TaskPlanner,
)
from cyberraccoon.agent.vision_agent import TaskResult, TaskStatus, VisionAgent
from cyberraccoon.capture.base import compute_frame_diff

logger = logging.getLogger("M2.workflow")

_FRAME_DIFF_DEBUG = os.environ.get(
    "CYBERRACCOON_FRAME_DIFF_DEBUG", ""
).lower() in ("1", "true")


# ---------------------------------------------------------------------------
# Phase 3 — STEPS-01/02: cancel-and-append step-list mutation helper
# ---------------------------------------------------------------------------

def _cancel_and_append(
    current_steps: list[PlanStep],
    completed_index: int,
    new_steps: list[PlanStep],
) -> tuple[list[PlanStep], list[int]]:
    """Mark unexecuted steps cancelled and append new steps with fresh numbers.

    Args:
        current_steps: full step list at the time of re-plan.
        completed_index: index ``i`` of the step that JUST completed
            (or, for path B, the index of the failed step minus 1 —
            cancellation must include the failed step itself).
            Steps at indices 0..completed_index are kept as-is.
            Steps at indices completed_index+1..end are flagged
            status='cancelled' (unless they were already 'done' or
            'cancelled').
        new_steps: list of fresh PlanStep objects whose .number
            values will be reassigned to start at max(existing) + 1.

    Returns:
        (mutated_full_list, cancelled_step_numbers)

    - Cancelled steps keep their original .number (STEPS-02).
    - new_steps mutate in place — their .number is reassigned.
    - Idempotent over already-cancelled tails: a step already at
      status='cancelled' is left alone, and its number is not
      re-emitted in cancelled_step_numbers (avoids double-marking
      across multiple re-plans).
    - If completed_index >= len(current_steps) - 1, the cancelled
      tail is empty and cancelled_step_numbers is [].
    - If new_steps is empty, returns (current_steps_with_cancelled_marks,
      cancelled_step_numbers) — caller decides whether to escalate.
    """
    cancelled_tail = current_steps[completed_index + 1:]
    cancelled_numbers: list[int] = []
    for step in cancelled_tail:
        if step.status not in ("done", "cancelled"):
            step.status = "cancelled"
            cancelled_numbers.append(step.number)

    head = current_steps[:completed_index + 1]
    keep = head + cancelled_tail
    next_num = max((s.number for s in keep), default=0) + 1
    for j, ns in enumerate(new_steps):
        ns.number = next_num + j
        if not ns.status:
            ns.status = "pending"
    return (keep + new_steps, cancelled_numbers)


# ---------------------------------------------------------------------------
# Workflow result
# ---------------------------------------------------------------------------

@dataclass
class WorkflowResult:
    """Outcome of a full workflow execution."""

    status: str  # "completed", "failed", "aborted"
    reason: str
    steps_completed: int
    steps_total: int
    step_results: list[dict[str, Any]] = field(default_factory=list)
    total_duration_s: float = 0.0


# ---------------------------------------------------------------------------
# Workflow runner
# ---------------------------------------------------------------------------

class WorkflowRunner:
    """Executes a planned sequence of steps through the vision agent.

    Args:
        agent: VisionAgent instance for executing individual steps.
        planner: TaskPlanner for initial planning and re-planning.
        max_retries_per_step: Max retry attempts per failed step.
        max_replans: Max re-plan attempts before aborting.
    """

    def __init__(
        self,
        agent: VisionAgent,
        planner: TaskPlanner,
        max_retries_per_step: int = 2,
        max_replans: int = 2,
        max_steps_per_step: int = 15,
        auto_approve: bool = False,
        auto_replan: bool = False,
    ) -> None:
        self._agent = agent
        self._planner = planner
        self._max_retries = max_retries_per_step
        self._max_replans = max_replans
        self._max_steps_per_step = max_steps_per_step
        self._auto_approve = auto_approve

        # Plan approval gate — set by approve_plan() from the UI
        self._plan_approved = threading.Event()
        self._plan_rejected = False

        # Phase 5 (DISCUSS-03/04): Integration Strategy A — AppController
        # pushes the (possibly modified) plan here via set_current_plan()
        # before calling approve_plan(). run() consumes the override
        # after the approval gate unblocks.
        self._override_steps: list[PlanStep] | None = None
        self._override_plan_version: int = 0

        # Escalation gate — set by resolve_escalation() from the UI
        self._escalation_resolved = threading.Event()

        # Pause/resume gate — set by resume() from the UI
        self._resume_event = threading.Event()

        # Phase 3 — replan decision gate (REPLAN-05). Mirrors the plan-approval
        # and escalation-resolved gate patterns above.
        self._replan_decision = threading.Event()
        self._replan_choice: str | None = None
        # H2 — single authoritative state for routing. Values:
        #   None              = no gate armed
        #   "replan_A"        = Path A dialog armed (verifier mismatch)
        #   "replan_B"        = Path B dialog armed (retry exhausted)
        #   "escalation_C"    = Path C dialog armed (escalation)
        # submit_replan_decision reads this to validate choice and route.
        self._active_gate: str | None = None
        # REPLAN-06: when True, paths A and B skip the dialog and synthesize
        # choice='replan'. Path C (escalation) NEVER bypasses regardless.
        self._auto_replan: bool = auto_replan
        # H5 — per-gate allowlist. Enforced in submit_replan_decision.
        self._GATE_ALLOWLIST: dict[str, frozenset[str]] = {
            "replan_A": frozenset({"continue", "replan", "abort"}),
            "replan_B": frozenset({"retry", "replan", "abort"}),
            "escalation_C": frozenset({"resume", "replan", "abort"}),
        }
        # I10 — _gate_lock guards _active_gate / _replan_choice / _auto_replan
        # against the HTTP-thread vs workflow-thread race. CPython's GIL makes
        # individual assignments atomic, but the read-validate-store sequence
        # in submit_replan_decision is not, and arming a gate while a stale
        # choice is still in _replan_choice could route the wrong choice to
        # the next gate. NEVER hold this lock while invoking on_progress
        # callbacks — that would create the lock-while-emitting deadlock the
        # AppController._lock pattern was designed to avoid.
        self._gate_lock = threading.Lock()
        # C3 — counter for consecutive verifier transport failures. After
        # this many in a row the workflow aborts to avoid silently advancing
        # with verification disabled (Path A regression-guard).
        self._max_consecutive_verifier_failures = 3
        self._consecutive_verifier_failures = 0

    def approve_plan(self) -> None:
        """Signal that the user approved the plan. Unblocks execution."""
        self._plan_rejected = False
        self._plan_approved.set()

    def reject_plan(self) -> None:
        """Signal that the user rejected the plan. Aborts workflow."""
        self._plan_rejected = True
        self._plan_approved.set()

    def set_current_plan(
        self, steps: list[PlanStep], plan_version: int = 0,
    ) -> None:
        """Override the plan that will execute when a gate unblocks.

        Called by AppController to push modified plans before approval
        or resume. The stored override is consumed once after the gate
        exits, then cleared.

        Thread safety: this method is called from the HTTP/UI thread while
        ``run()`` is blocked on ``_plan_approved.wait()`` or
        ``_resume_event.wait()`` in the workflow thread. The assignment
        to ``_override_steps`` is atomic for Python reference assignment.
        The override is consumed exactly once after the gate unblocks
        and before the execution loop — no interleaving is possible.

        Args:
            steps: The plan steps to use. Copied defensively.
            plan_version: Monotonic counter for traceability.
        """
        self._override_steps = list(steps)
        self._override_plan_version = plan_version

    def resolve_escalation(self) -> None:
        """Signal that the user resolved the escalation. Unblocks workflow."""
        self._escalation_resolved.set()

    def resume(self) -> None:
        """Unblock the pause gate. Call after pushing any plan modifications
        via set_current_plan()."""
        self._resume_event.set()

    def submit_replan_decision(self, choice: str) -> None:
        """H5 — single authoritative decision entry point for all three paths.

        Validates ``choice`` against the per-gate allowlist driven by
        ``self._active_gate``. Raises ValueError for any choice not in the
        allowlist for the current gate; raises RuntimeError if no gate is
        armed (i.e. submit arrived while the runner was not waiting).

        Valid combinations:
            gate "replan_A"      -> {"continue", "replan", "abort"}
            gate "replan_B"      -> {"retry", "replan", "abort"}
            gate "escalation_C"  -> {"resume", "replan", "abort"}
        """
        # I10 — atomic read/validate/store under _gate_lock so a concurrent
        # gate swap on the workflow thread cannot wedge a now-invalid choice
        # into _replan_choice. The Event.set() at the end is intentionally
        # outside the lock — Event has its own internal lock and we don't
        # want to nest.
        with self._gate_lock:
            gate = self._active_gate
            if gate is None:
                raise RuntimeError(
                    "no active replan gate — submit_replan_decision called "
                    "while the workflow was not waiting on a user decision"
                )
            allowed = self._GATE_ALLOWLIST.get(gate)
            if allowed is None or choice not in allowed:
                raise ValueError(
                    f"invalid choice {choice!r} for gate {gate!r}; "
                    f"allowed: {sorted(allowed) if allowed else []}"
                )
            self._replan_choice = choice
        self._replan_decision.set()

    def set_auto_replan(self, enabled: bool) -> None:
        """Toggle Auto Re-plan. Read next time the gate would block (REPLAN-06)."""
        with self._gate_lock:
            self._auto_replan = enabled

    def _await_replan_decision(
        self,
        path: str,
        payload: dict[str, Any],
        on_progress: Callable[[dict], None] | None,
    ) -> str:
        """Block until user submits a re-plan decision or the task is aborted.

        Path A and Path B respect the Auto Re-plan flag (REPLAN-06): when
        ``self._auto_replan`` is True, the dialog is bypassed and 'replan'
        is returned synthetically (after emitting a 'replan_auto' event AND
        a 'replan_dialog_resolved' event so any stale client dialog from
        a prior reconnect closes cleanly).

        Path C does NOT call this helper — escalation gating is adapted
        in Task 3 Edit D to also arm self._active_gate='escalation_C'
        and route through submit_replan_decision.

        H3 — the decision event and choice are cleared BEFORE emitting
        the dialog, so a fast synchronous submit between emit and wait
        is captured, not lost.

        Returns: choice string. Returns 'abort' on abort.
        """
        # I6 — auto_replan check BEFORE arming the gate so a concurrent
        # submit_replan_decision can't fire-then-be-discarded.
        if path not in ("A", "B"):
            raise ValueError(
                f"_await_replan_decision only handles paths A/B, got {path!r}"
            )

        if self._auto_replan:
            # Auto Re-plan bypass (REPLAN-06). Synthesize 'replan' without
            # arming any gate; gate stays None for the whole call.
            if on_progress:
                on_progress({"type": "replan_auto", "path": path})
                on_progress({"type": "replan_dialog_resolved", "choice": "replan"})
            return "replan"

        # try/finally guarantees _active_gate clears even if on_progress raises.
        try:
            # H3 ordering: clear events + choice BEFORE arming gate + emit.
            with self._gate_lock:
                self._replan_decision.clear()
                self._replan_choice = None
                self._active_gate = "replan_A" if path == "A" else "replan_B"

            # Emit dialog (Step 1c of decision_lifecycle).
            if on_progress:
                event = {"type": "replan_dialog", "path": path}
                event.update(payload)
                on_progress(event)

            # Wait for decision or abort.
            while not self._replan_decision.is_set():
                if self._agent._abort_event.is_set():
                    return "abort"
                self._replan_decision.wait(timeout=0.5)

            with self._gate_lock:
                choice = self._replan_choice or "abort"
            return choice
        finally:
            # Always clear the gate, even on exception.
            with self._gate_lock:
                self._active_gate = None

    def _attempt_step_once(
        self,
        step: PlanStep,
        step_goal: str,
        on_step: Callable[[dict], None] | None = None,
    ) -> tuple[bool, str, Any]:
        """Run a single agent.run() attempt for a step — no retry looping.

        I1 — on_step is now threaded through so Path B retry shows live
        per-action progress in the UI (was previously hidden).

        Returns (success, failure_reason, result).
        """
        saved_max_steps = self._agent._max_steps
        self._agent._max_steps = self._max_steps_per_step
        try:
            result = self._agent.run(
                step_goal,
                on_step=on_step,
                _preserve_transition_state=True,
            )
        finally:
            self._agent._max_steps = saved_max_steps

        if result.status == TaskStatus.COMPLETED:
            if result.completion_status in ("gave_up", "stuck"):
                return (
                    False,
                    (
                        f"Step reported {result.completion_status}: "
                        f"{result.reason}"
                    ),
                    result,
                )
            return (True, "", result)
        return (False, result.reason, result)

    def _retry_step_once(
        self,
        step: PlanStep,
        previous_failure: str,
        on_progress: Callable[[dict], None] | None,
        step_results: list[Any],
        completed_goals: list[str],
        current_steps: list[PlanStep],
        task_goal: str,
        start_time: float,
        step_index: int,
        budget_history: list[dict[str, Any]] | None = None,
    ) -> tuple[WorkflowResult | None, int]:
        """H4 — Path B "Retry Once" with FULL bookkeeping.

        Re-executes the step via the same capture→decide→act→append→verify
        path as a normal step. On success, the step is marked 'done',
        budget_history is appended, step_result is appended, step_done is
        emitted, and the step index is incremented so the outer loop
        advances.

        On retry failure: returns a WorkflowResult with status='failed'
        and message "Step {n} failed (after Retry Once): {reason}". This
        is an AUTO-ABORT — we do NOT re-emit the Path B dialog.

        On user abort during retry: returns a WorkflowResult with
        status='aborted', NOT 'failed' (review I1).

        I1 — on_progress is now threaded through to the underlying
        agent.run so the user sees per-action progress during the retry,
        and budget_history is appended so subsequent re-plans get
        complete calibration data.

        Returns: (result_or_none, new_step_index)
            - (None, step_index + 1)   → retry succeeded, caller continues loop
            - (WorkflowResult, step_index) → workflow terminates
        """
        step_goal = step.format_for_agent(
            total_steps=len(current_steps),
            completed=completed_goals if completed_goals else None,
        )
        success, failure_reason, result = self._attempt_step_once(
            step, step_goal, on_step=on_progress,
        )

        if success:
            # Bookkeeping — mirrors the success branch of the normal step
            # loop (lines ~998-1026): budget_history, step_results,
            # completed_goals, step_done event, status.
            step.status = "done"
            if budget_history is not None:
                budget_history.append({
                    "step_number": step.number,
                    "step_goal": step.goal,
                    "expected_actions": step.expected_actions,
                    "actions_used": result.total_steps,
                })
            step_results.append({
                "step_number": step.number,
                "step_goal": step.goal,
                "success": True,
                "failure_reason": "",
                "actions_used": result.total_steps,
                "expected_actions": step.expected_actions,
            })
            completed_goals.append(step.goal)
            if on_progress:
                on_progress({
                    "type": "step_done",
                    "step_number": step.number,
                    "step_goal": step.goal,
                    "steps_total": len(current_steps),
                    "steps_completed": len(completed_goals),
                    "actions_used": result.total_steps,
                    "expected_actions": step.expected_actions,
                })
            return (None, step_index + 1)

        # I1 — user abort during retry should report aborted, not failed.
        if result is not None and result.status == TaskStatus.ABORTED:
            return (
                WorkflowResult(
                    status="aborted",
                    reason=(
                        f"User aborted during Retry Once for step "
                        f"{step.number}: {result.reason or 'aborted'}"
                    ),
                    steps_completed=len(completed_goals),
                    steps_total=len(current_steps),
                    step_results=step_results,
                    total_duration_s=time.monotonic() - start_time,
                ),
                step_index,
            )

        # Retry failed — auto-abort per H4 decision (a).
        return (
            WorkflowResult(
                status="failed",
                reason=(
                    f"Step {step.number} failed (after Retry Once): "
                    f"{failure_reason or previous_failure}"
                ),
                steps_completed=len(completed_goals),
                steps_total=len(current_steps),
                step_results=step_results,
                total_duration_s=time.monotonic() - start_time,
            ),
            step_index,
        )

    def request_pause(self) -> None:
        """Request the current task to pause. Delegates to agent.pause().

        External callers (AppController) should use this instead of
        reaching into agent._pause_event directly.
        """
        self._agent.pause()

    # ------------------------------------------------------------------
    # Pause gate helper
    # ------------------------------------------------------------------

    def _handle_pause_gate(
        self,
        step: PlanStep,
        current_steps: list[PlanStep],
        completed_goals: list[str],
        step_results: list[dict[str, Any]],
        start_time: float,
        on_progress: Callable[[dict[str, Any]], None] | None,
        last_good_screenshot: str,
        *,
        is_partial: bool = False,
    ) -> WorkflowResult | None:
        """Block on pause gate until resume or abort.

        Returns WorkflowResult if aborted, None if resumed.

        Args:
            step: The current step when pause was triggered.
            current_steps: Full list of plan steps (mutable).
            completed_goals: Goals completed so far.
            step_results: Step result dicts accumulated so far.
            start_time: Workflow start time (monotonic).
            on_progress: Progress callback.
            last_good_screenshot: Fallback screenshot if capture fails.
            is_partial: True if the current step was interrupted mid-execution.
        """
        # CRUISE-02: capture fresh screenshot for paused view (D-05)
        # Addresses review HIGH-4: explicit fallback on capture failure
        try:
            pause_cap = self._agent._capture.capture()
            pause_screenshot = pause_cap.base64_jpeg
        except Exception as exc:
            logger.warning(
                "Capture failed during pause, falling back to last "
                "screenshot: %s", exc,
            )
            pause_screenshot = last_good_screenshot or ""

        # Compute step index for status assignment
        step_idx = next(
            (j for j, s in enumerate(current_steps) if s is step),
            -1,
        )

        # Emit task_paused event with step status + screenshot
        if on_progress:
            on_progress({
                "type": "task_paused",
                "step_number": step.number,
                "steps_completed": len(completed_goals),
                "steps": [
                    {
                        "number": s.number,
                        "goal": s.goal,
                        "status": (
                            "done" if j < len(completed_goals)
                            else "partial" if j == step_idx and is_partial
                            else "pending"
                        ),
                        "reboot_expected": s.reboot_expected,
                        "expected_actions": s.expected_actions,
                        "expected_outcome": s.expected_outcome,
                        "actions_used": (
                            step_results[j].get("actions_used")
                            if j < len(step_results) else None
                        ),
                    }
                    for j, s in enumerate(current_steps)
                ],
                "screenshot_base64": pause_screenshot,
            })

        # Block until resume or abort (same pattern as approval gate)
        self._resume_event.clear()
        while not self._resume_event.is_set():
            if self._agent._abort_event.is_set():
                return WorkflowResult(
                    status="aborted",
                    reason="User aborted while paused",
                    steps_completed=len(completed_goals),
                    steps_total=len(current_steps),
                    step_results=step_results,
                    total_duration_s=time.monotonic() - start_time,
                )
            self._resume_event.wait(timeout=0.5)

        # Consume plan override (same as approval gate pattern)
        if self._override_steps is not None:
            remaining_override = self._override_steps
            self._override_steps = None
            self._override_plan_version = 0
            # Rebuild: completed steps + new remaining steps
            current_steps[:] = (
                current_steps[:len(completed_goals)] + remaining_override
            )
            # Renumber remaining steps
            for j, ns in enumerate(remaining_override):
                ns.number = len(completed_goals) + 1 + j
            logger.info(
                "Workflow: consumed override plan on resume "
                "(%d remaining steps)",
                len(remaining_override),
            )

        # Clear pause state for next cycle
        self._agent._pause_event.clear()

        # Emit task_resumed event -- NOTE: this is NOT task_started.
        # Review HIGH-5 / Pitfall 2: resume must NOT emit task_started
        # because that would clear PlanDiscussionState in AppController.
        if on_progress:
            on_progress({"type": "task_resumed"})

        return None  # Resumed successfully, caller should continue

    # ------------------------------------------------------------------
    # Validation gating
    # ------------------------------------------------------------------

    def _should_skip_validation(
        self,
        step: PlanStep,
        result: TaskResult,
        pre_step_image: Any,
        post_step_image: Any,
        is_last_step: bool,
    ) -> bool:
        """Decide whether post-step validation can be skipped.

        Always-validate rules are checked **first** (D-05). If any is
        true, validation always runs (return False).  Only when none of
        them fire do we evaluate the skip conditions.

        Args:
            step: The plan step that just completed.
            result: TaskResult from the agent run.
            pre_step_image: PIL Image captured *before* the step.
            post_step_image: PIL Image captured *after* the step.
            is_last_step: True if this is the last step in the plan.

        Returns:
            True  -> validation can be skipped.
            False -> validation must run.
        """
        # ---- Always-validate rules (any True => return False) ----
        if step.reboot_expected:
            logger.debug("Validation gate: always-validate (reboot step)")
            return False

        if (step.expected_actions is not None
                and result.total_steps > step.expected_actions * 0.8):
            logger.debug("Validation gate: always-validate (>80%% budget)")
            return False

        if result.completion_status in ("gave_up", "stuck"):
            logger.debug(
                "Validation gate: always-validate (%s)",
                result.completion_status,
            )
            return False

        if is_last_step:
            logger.debug("Validation gate: always-validate (last step)")
            return False

        # ---- Skip conditions (all must be True => return True) ----
        if step.expected_actions is None:
            # D-03: never skip without budget data
            logger.debug("Validation gate: no budget data, cannot skip")
            return False

        if pre_step_image is None or post_step_image is None:
            logger.debug("Validation gate: missing images, cannot skip")
            return False

        diff_pct = compute_frame_diff(pre_step_image, post_step_image)

        if _FRAME_DIFF_DEBUG:
            _will_skip = (
                diff_pct > 5.0
                and result.total_steps < step.expected_actions * 0.5
            )
            logger.info(
                "FRAME_DIFF_DEBUG step=%d diff_score=%.2f%% "
                "expected_actions=%s actual_actions=%d "
                "decision=%s",
                step.number, diff_pct,
                step.expected_actions, result.total_steps,
                "skip" if _will_skip else "validate",
            )

        if diff_pct <= 5.0:
            logger.debug(
                "Validation gate: frame-diff %.1f%% <= 5%%, cannot skip",
                diff_pct,
            )
            return False

        if result.total_steps >= step.expected_actions * 0.5:
            logger.debug(
                "Validation gate: budget ratio %.1f >= 0.5, cannot skip",
                result.total_steps / step.expected_actions,
            )
            return False

        logger.debug(
            "Validation gate: skipping (diff=%.1f%%, budget=%.1f)",
            diff_pct,
            result.total_steps / step.expected_actions
            if step.expected_actions else 0,
        )
        return True

    def run(
        self,
        task_goal: str,
        screenshot_base64: str,
        skill_text: str | None = None,
        on_progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> WorkflowResult:
        """Plan and execute a task as a series of steps.

        Args:
            task_goal: The full task description.
            screenshot_base64: Current screen state for the planner.
            skill_text: Optional skill markdown for planning context.
            on_progress: Callback for step-level progress updates.

        Returns:
            WorkflowResult with status, completed steps, and details.
        """
        start_time = time.monotonic()

        # ---- Phase 1: Planning ----
        logger.info("Workflow: planning steps for task: %s", task_goal)
        steps = self._planner.plan(task_goal, screenshot_base64, skill_text)

        if steps is None:
            return WorkflowResult(
                status="failed",
                reason="Planning failed. The LLM could not decompose "
                       "the task into steps. Check API connectivity "
                       "and try again.",
                steps_completed=0,
                steps_total=0,
                total_duration_s=time.monotonic() - start_time,
            )

        if on_progress:
            # [REVIEWS: MEDIUM-2] Privacy tradeoff: screenshot_base64 is
            # broadcast to every connected WebSocket client via the on_progress
            # callback. Acceptable because (1) the UI runs on LAN only, (2) the
            # user authorized screen capture by submitting the task, and (3)
            # AppController caches the screenshot server-side for chat
            # grounding, so the frontend can discard its copy immediately
            # (ui/web/static/app.js plan_ready handler drops the field).
            on_progress({
                "type": "plan_ready",
                "steps": [
                    {
                        "number": s.number,
                        "goal": s.goal,
                        "reboot_expected": s.reboot_expected,
                        "expected_actions": s.expected_actions,
                        "expected_outcome": s.expected_outcome,
                    }
                    for s in steps
                ],
                "task_goal": task_goal,
                "screenshot_base64": screenshot_base64,
            })

        # ---- Wait for user approval ----
        if self._auto_approve:
            logger.info("Workflow: auto-approving plan (%d steps)", len(steps))
        else:
            logger.info(
                "Workflow: waiting for plan approval (%d steps)", len(steps),
            )
            self._plan_approved.clear()
            self._plan_rejected = False

            # Block until approve_plan() or reject_plan() is called,
            # or the abort event is set.
            while not self._plan_approved.is_set():
                if self._agent._abort_event.is_set():
                    return WorkflowResult(
                        status="aborted",
                        reason="User aborted during plan approval",
                        steps_completed=0,
                        steps_total=len(steps),
                        total_duration_s=time.monotonic() - start_time,
                    )
                self._plan_approved.wait(timeout=0.5)

            if self._plan_rejected:
                logger.info("Workflow: plan rejected by user")
                return WorkflowResult(
                    status="aborted",
                    reason="Plan rejected by user",
                    steps_completed=0,
                    steps_total=len(steps),
                    total_duration_s=time.monotonic() - start_time,
                )

            # Phase 5: consume any plan override pushed by AppController
            # via set_current_plan() before approve_plan() fired. This
            # closes Pitfall 1 (workflow executes the pre-edit plan).
            if self._override_steps is not None:
                logger.info(
                    "Workflow: using override plan (%d steps, version=%d)"
                    " from AppController (was %d steps)",
                    len(self._override_steps),
                    self._override_plan_version,
                    len(steps),
                )
                steps = self._override_steps
                self._override_steps = None
                self._override_plan_version = 0

        logger.info("Workflow: plan approved, starting execution")

        # Consume plan override if set_current_plan() was called during
        # the approval wait.
        if self._override_steps is not None:
            logger.info(
                "Workflow: consuming override plan (%d steps)",
                len(self._override_steps),
            )
            steps = self._override_steps
            self._override_steps = None
            self._override_plan_version = 0

        # ---- Phase 2: Execution ----
        completed_goals: list[str] = []
        step_results: list[dict[str, Any]] = []
        budget_history: list[dict[str, Any]] = []
        replans_used = 0
        current_steps = steps
        last_good_screenshot = screenshot_base64

        i = 0
        while i < len(current_steps):
            step = current_steps[i]

            # STEPS-03: skip cancelled steps (preserved in list for audit, not executed)
            if step.status == "cancelled":
                i += 1
                continue

            # Check abort between steps
            if self._agent._abort_event.is_set():
                logger.info("Workflow: abort detected between steps")
                return WorkflowResult(
                    status="aborted",
                    reason="User aborted",
                    steps_completed=len(completed_goals),
                    steps_total=len(current_steps),
                    step_results=step_results,
                    total_duration_s=time.monotonic() - start_time,
                )

            # Pitfall 1: check pause before starting a new step.
            # Addresses review MEDIUM-1 (pre-step race condition).
            if self._agent._pause_event.is_set():
                abort_result = self._handle_pause_gate(
                    step=step,
                    current_steps=current_steps,
                    completed_goals=completed_goals,
                    step_results=step_results,
                    start_time=start_time,
                    on_progress=on_progress,
                    last_good_screenshot=last_good_screenshot,
                    is_partial=False,  # Step has not started yet
                )
                if abort_result is not None:
                    return abort_result
                continue

            # Ensure capture device is alive before starting the step.
            # After a reboot (whether tagged [REBOOT EXPECTED] or not),
            # the HDMI signal drops and the V4L2/CSI device handle goes
            # stale. agent.run() fails at initial capture if we don't
            # recover here. This is cheap: one capture() call to check.
            try:
                self._agent._capture.capture()
            except Exception:
                logger.info(
                    "Workflow: capture device not ready before step %d, "
                    "attempting reboot transition recovery",
                    step.number,
                )
                if on_progress:
                    on_progress({
                        "type": "reboot_transition",
                        "step_number": step.number,
                    })
                try:
                    self._agent._wait_for_reboot_transition(
                        timeout_s=120.0,
                    )
                except Exception as e:
                    logger.warning(
                        "Workflow: transition recovery timed out before "
                        "step %d: %s — trying direct reconnect",
                        step.number, e,
                    )
                    try:
                        self._agent._capture.close()
                        time.sleep(2.0)
                        self._agent._capture.open()
                        self._agent._capture.capture()
                        logger.info(
                            "Workflow: capture recovered via direct "
                            "reconnect before step %d",
                            step.number,
                        )
                    except Exception as recovery_err:
                        logger.error(
                            "Workflow: capture recovery failed before "
                            "step %d: %s", step.number, recovery_err,
                        )
                        return WorkflowResult(
                            status="failed",
                            reason=(
                                f"Capture device recovery failed: {recovery_err}"
                            ),
                            steps_completed=len(completed_goals),
                            steps_total=len(current_steps),
                            step_results=step_results,
                            total_duration_s=(
                                time.monotonic() - start_time
                            ),
                        )

            # Report step start
            if on_progress:
                on_progress({
                    "type": "step_start",
                    "step_number": step.number,
                    "step_goal": step.goal,
                    "steps_total": len(current_steps),
                    "steps_completed": len(completed_goals),
                })

            # Capture pre-step screenshot for frame-diff comparison
            pre_step_image = None
            try:
                pre_step_cap = self._agent._capture.capture()
                pre_step_image = pre_step_cap.image
            except Exception as e:
                logger.debug(
                    "Pre-step capture failed (step %d): %s — "
                    "validation gate will fall back to always-validate",
                    step.number, e,
                )

            # Execute step with retries
            step_goal = step.format_for_agent(
                total_steps=len(current_steps),
                completed=completed_goals if completed_goals else None,
            )

            success = False
            failure_reason = ""
            paused = False

            for attempt in range(1 + self._max_retries):
                logger.info(
                    "Workflow: executing step %d/%d (attempt %d): %s",
                    step.number, len(current_steps), attempt + 1, step.goal,
                )

                # Limit per-step budget to prevent one step from
                # consuming the entire task timeout/step count.
                saved_max_steps = self._agent._max_steps
                self._agent._max_steps = self._max_steps_per_step
                try:
                    result = self._agent.run(
                        step_goal,
                        on_step=on_progress,
                        _preserve_transition_state=True,
                    )
                finally:
                    self._agent._max_steps = saved_max_steps

                if result.status == TaskStatus.COMPLETED:
                    # Check structured completion_status from LLM signal.
                    if result.completion_status in ("gave_up", "stuck"):
                        failure_reason = (
                            f"Step reported {result.completion_status}: "
                            f"{result.reason}"
                        )
                        logger.warning(
                            "Workflow: step %d completed with status %s: %s",
                            step.number, result.completion_status,
                            result.reason,
                        )
                    else:
                        success = True
                    break
                elif result.status == TaskStatus.ABORTED:
                    return WorkflowResult(
                        status="aborted",
                        reason="User aborted during step execution",
                        steps_completed=len(completed_goals),
                        steps_total=len(current_steps),
                        step_results=step_results,
                        total_duration_s=time.monotonic() - start_time,
                    )
                elif result.status == TaskStatus.PAUSED:
                    abort_result = self._handle_pause_gate(
                        step=step,
                        current_steps=current_steps,
                        completed_goals=completed_goals,
                        step_results=step_results,
                        start_time=start_time,
                        on_progress=on_progress,
                        last_good_screenshot=last_good_screenshot,
                        is_partial=True,  # Step was interrupted mid-execution
                    )
                    if abort_result is not None:
                        return abort_result
                    # Continue from current index WITHOUT advancing i --
                    # the interrupted step will be re-attempted with a
                    # fresh screenshot. D-03: interrupted step is partially
                    # completed. D-09: resume continues from next incomplete
                    # step. The LLM sees a fresh screenshot and adapts.
                    paused = True
                    break  # break out of retry loop
                else:
                    failure_reason = result.reason
                    logger.warning(
                        "Workflow: step %d attempt %d failed: %s",
                        step.number, attempt + 1, failure_reason,
                    )

            # If paused, skip step result recording and re-attempt step
            if paused:
                continue

            budget_history.append({
                "step_number": step.number,
                "step_goal": step.goal,
                "expected_actions": step.expected_actions,
                "actions_used": result.total_steps,
            })

            step_results.append({
                "step_number": step.number,
                "step_goal": step.goal,
                "success": success,
                "failure_reason": failure_reason if not success else "",
                "actions_used": result.total_steps,
                "expected_actions": step.expected_actions,
            })

            if success:
                completed_goals.append(step.goal)

                if on_progress:
                    on_progress({
                        "type": "step_done",
                        "step_number": step.number,
                        "step_goal": step.goal,
                        "steps_total": len(current_steps),
                        "steps_completed": len(completed_goals),
                        "actions_used": result.total_steps,
                        "expected_actions": step.expected_actions,
                    })

                # Handle reboot transition AFTER the step completes
                if step.reboot_expected:
                    logger.info(
                        "Workflow: step %d has [REBOOT EXPECTED], "
                        "waiting for transition",
                        step.number,
                    )
                    if on_progress:
                        on_progress({
                            "type": "reboot_transition",
                            "step_number": step.number,
                        })
                    try:
                        self._agent._wait_for_reboot_transition(
                            timeout_s=120.0,
                        )
                    except Exception as e:
                        logger.warning(
                            "Workflow: reboot transition timed out after "
                            "step %d: %s — attempting capture recovery",
                            step.number, e,
                        )
                        # Transition handler timed out, but the machine
                        # may have rebooted successfully with a capture
                        # device that lost sync.  Try one final reconnect
                        # before giving up.
                        try:
                            self._agent._capture.close()
                            time.sleep(2.0)
                            self._agent._capture.open()
                            self._agent._capture.capture()
                            logger.info(
                                "Workflow: capture recovered after reboot "
                                "transition timeout — continuing",
                            )
                        except Exception as recovery_err:
                            logger.error(
                                "Workflow: capture recovery failed after "
                                "reboot transition: %s", recovery_err,
                            )
                            return WorkflowResult(
                                status="failed",
                                reason=(
                                    f"Reboot transition failed: {recovery_err}"
                                ),
                                steps_completed=len(completed_goals),
                                steps_total=len(current_steps),
                                step_results=step_results,
                                total_duration_s=(
                                    time.monotonic() - start_time
                                ),
                            )

                # ---- Post-step validation ----
                # Ask the planner: does the plan still make sense
                # given what the screen looks like now?
                remaining = current_steps[i + 1:]
                is_last_step = not remaining

                # Capture post-step image for frame-diff comparison
                post_step_image = None
                try:
                    val_cap = self._agent._capture.capture()
                    val_screenshot = val_cap.base64_jpeg
                    post_step_image = val_cap.image
                    last_good_screenshot = val_screenshot
                except Exception as e:
                    logger.debug(
                        "Post-step capture failed (step %d): %s — "
                        "using last good screenshot",
                        step.number, e,
                    )
                    val_screenshot = last_good_screenshot

                # Check validation gate
                if self._should_skip_validation(
                    step=step,
                    result=result,
                    pre_step_image=pre_step_image,
                    post_step_image=post_step_image,
                    is_last_step=is_last_step,
                ):
                    logger.info(
                        "Workflow: skipping validation for step %d "
                        "(screen changed, low budget usage)",
                        step.number,
                    )
                    if on_progress:
                        on_progress({
                            "type": "validation_skipped",
                            "step_number": step.number,
                        })
                else:
                    # Phase 3 — Path A: run verify_step. On verified, advance.
                    # On mismatch, gate on user decision via _active_gate='replan_A'.
                    # C3 — verify_step now raises LLMTransportError /
                    # LLMAuthError on transport failures (rate limit,
                    # timeout, auth). Count consecutive failures and abort
                    # if the verifier is structurally unreachable so we
                    # don't silently advance with verification disabled.
                    try:
                        verification = self._planner.verify_step(
                            step=step,
                            observed_screenshot=val_screenshot,
                            skill_text=skill_text,
                        )
                        self._consecutive_verifier_failures = 0
                    except LLMAuthError as e:
                        # Hard fail — bad API key, user can't recover by retrying.
                        if on_progress:
                            on_progress({
                                "type": "verifier_unavailable",
                                "step_number": step.number,
                                "kind": "auth",
                                "error": str(e),
                            })
                        return WorkflowResult(
                            status="failed",
                            reason=f"Verifier auth failed: {e}",
                            steps_completed=len(completed_goals),
                            steps_total=len(current_steps),
                            step_results=step_results,
                            total_duration_s=time.monotonic() - start_time,
                        )
                    except LLMTransportError as e:
                        self._consecutive_verifier_failures += 1
                        if on_progress:
                            on_progress({
                                "type": "verifier_unavailable",
                                "step_number": step.number,
                                "kind": "transport",
                                "error": str(e),
                                "consecutive_failures": (
                                    self._consecutive_verifier_failures
                                ),
                            })
                        if (
                            self._consecutive_verifier_failures
                            >= self._max_consecutive_verifier_failures
                        ):
                            return WorkflowResult(
                                status="failed",
                                reason=(
                                    f"Verifier unreachable after "
                                    f"{self._consecutive_verifier_failures} "
                                    f"consecutive transport failures: {e}"
                                ),
                                steps_completed=len(completed_goals),
                                steps_total=len(current_steps),
                                step_results=step_results,
                                total_duration_s=(
                                    time.monotonic() - start_time
                                ),
                            )
                        # Below threshold — log and bias-to-true (VERIFY-03)
                        # for this single step. Workflow continues, but the
                        # next transport failure brings us closer to abort.
                        logger.warning(
                            "verify_step transport failure (%d/%d) for step %d; "
                            "biasing to verified=True for this step: %s",
                            self._consecutive_verifier_failures,
                            self._max_consecutive_verifier_failures,
                            step.number, e,
                        )
                        verification = StepVerification(
                            verified=True,
                            expected=step.expected_outcome,
                            observed="(verifier transport failure)",
                            mismatch_reason=None,
                            confidence=0.0,
                        )
                    if verification.verified:
                        logger.debug(
                            "verify_step: step %d verified (confidence=%.2f)",
                            step.number, verification.confidence,
                        )
                        # fall through to step.status='done' + i += 1 below
                    else:
                        # Verifier flagged a mismatch — gate on user decision (Path A).
                        choice = self._await_replan_decision(
                            path="A",
                            payload={
                                "step_number": step.number,
                                "step_goal": step.goal,
                                "expected": verification.expected,
                                "observed": verification.observed,
                                "mismatch_reason": verification.mismatch_reason,
                                "failure_reason": None,
                                "screenshot_base64": val_screenshot,
                            },
                            on_progress=on_progress,
                        )
                        # H8 — always emit replan_dialog_resolved so the frontend
                        # modal and server-side pending cache close unconditionally.
                        if on_progress:
                            on_progress({
                                "type": "replan_dialog_resolved",
                                "choice": choice,
                            })

                        if choice == "abort":
                            return WorkflowResult(
                                status="aborted",
                                reason="User aborted at re-plan dialog (Path A)",
                                steps_completed=len(completed_goals),
                                steps_total=len(current_steps),
                                step_results=step_results,
                                total_duration_s=time.monotonic() - start_time,
                            )

                        if choice == "continue":
                            step.status = "done"
                            i += 1
                            continue

                        if choice == "replan":
                            if replans_used >= self._max_replans:
                                return WorkflowResult(
                                    status="failed",
                                    reason="Max replans exhausted",
                                    steps_completed=len(completed_goals),
                                    steps_total=len(current_steps),
                                    step_results=step_results,
                                    total_duration_s=time.monotonic() - start_time,
                                )
                            new_steps = self._planner.replan(
                                task_goal=task_goal,
                                failed_step=step,
                                failure_reason=(
                                    verification.mismatch_reason
                                    or "Verifier flagged mismatch"
                                ),
                                screenshot_base64=val_screenshot,
                                remaining_steps=current_steps[i + 1:],
                                skill_text=skill_text,
                                calibration_data=budget_history,
                            )
                            if not new_steps:
                                if on_progress:
                                    on_progress({
                                        "type": "replan_failed",
                                        "step_number": step.number,
                                    })
                                return WorkflowResult(
                                    status="failed",
                                    reason="Re-plan returned no steps",
                                    steps_completed=len(completed_goals),
                                    steps_total=len(current_steps),
                                    step_results=step_results,
                                    total_duration_s=time.monotonic() - start_time,
                                )
                            replans_used += 1
                            current_steps, cancelled_numbers = _cancel_and_append(
                                current_steps, i, new_steps,
                            )
                            if on_progress:
                                on_progress({
                                    "type": "replanned",
                                    "old_step": step.number,
                                    "reason": (
                                        verification.mismatch_reason
                                        or "Verifier flagged mismatch"
                                    ),
                                    "new_steps": [
                                        {
                                            "number": s.number,
                                            "goal": s.goal,
                                            "reboot_expected": s.reboot_expected,
                                            "expected_actions": s.expected_actions,
                                            "expected_outcome": s.expected_outcome,
                                            "status": s.status,
                                        }
                                        for s in new_steps
                                    ],
                                    "cancelled_step_numbers": cancelled_numbers,
                                    "steps_completed": len(completed_goals),
                                    "screenshot_base64": val_screenshot,
                                })
                            step.status = "done"
                            i += 1
                            continue

                # Phase 3 — mark success and advance (verify_step passed or
                # validation was skipped).
                step.status = "done"
                i += 1

                # Path C — escalation trigger. The VisionAgent signals escalation
                # via completion_status='escalate'. The legacy validate_plan
                # escalation verdict is gone; this is the only live trigger.
                # (Path C lives OUTSIDE the if success: block so that even if
                # the verify_step path took `continue`, we still check. But
                # since `success` gates completion_status, escalation is
                # detected here only when success=True with completion_status='escalate'.)
                if getattr(result, "completion_status", "success") == "escalate":
                    # Path C — escalation gate. H3 ordering: clear all events
                    # and arm the gate BEFORE emitting, so a fast user submit
                    # cannot land between emit and clear and get discarded
                    # (mirrors _await_replan_decision for Paths A/B).
                    # The whole body is try/finally-guarded so _active_gate
                    # always returns to None even if on_progress raises.
                    try:
                        with self._gate_lock:
                            self._replan_choice = None
                            self._escalation_resolved.clear()
                            self._replan_decision.clear()
                            self._active_gate = "escalation_C"

                        if on_progress:
                            on_progress({
                                "type": "escalate",
                                "reason": result.reason,
                                "step_number": step.number,
                            })

                        while not (
                            self._replan_decision.is_set()
                            or self._escalation_resolved.is_set()
                        ):
                            if self._agent._abort_event.is_set():
                                return WorkflowResult(
                                    status="aborted",
                                    reason="User aborted during escalation",
                                    steps_completed=len(completed_goals),
                                    steps_total=len(current_steps),
                                    step_results=step_results,
                                    total_duration_s=(
                                        time.monotonic() - start_time
                                    ),
                                )
                            self._replan_decision.wait(timeout=0.5)

                        # Resolve choice. submit_replan_decision sets
                        # _replan_choice; legacy resolve_escalation() leaves it
                        # None — log INFO so the legacy path is debuggable.
                        with self._gate_lock:
                            choice = self._replan_choice
                            self._replan_choice = None
                        if choice is None:
                            logger.info(
                                "Path C resolved via legacy resolve_escalation "
                                "endpoint (no choice submitted) — defaulting to 'resume'"
                            )
                            choice = "resume"
                    finally:
                        # Guarantee gate clears on every exit path including
                        # exceptions raised inside on_progress (#2 of review).
                        with self._gate_lock:
                            self._active_gate = None

                    # H8 — emit resolved for every choice (outside try so an
                    # exception here can't re-arm the gate).
                    if on_progress:
                        on_progress({
                            "type": "replan_dialog_resolved",
                            "choice": choice,
                        })

                    if choice == "abort":
                        return WorkflowResult(
                            status="aborted",
                            reason="User aborted from escalation",
                            steps_completed=len(completed_goals),
                            steps_total=len(current_steps),
                            step_results=step_results,
                            total_duration_s=time.monotonic() - start_time,
                        )

                    if choice == "replan":
                        if replans_used >= self._max_replans:
                            return WorkflowResult(
                                status="failed",
                                reason="Max replans exhausted",
                                steps_completed=len(completed_goals),
                                steps_total=len(current_steps),
                                step_results=step_results,
                                total_duration_s=time.monotonic() - start_time,
                            )
                        try:
                            esc_cap = self._agent._capture.capture()
                            escalation_replan_screenshot = esc_cap.base64_jpeg
                            last_good_screenshot = escalation_replan_screenshot
                        except Exception as e:
                            # I8 — escalation capture failure was previously
                            # silent. Log it: escalations often fire when the
                            # screen is bad, so re-planning against a stale
                            # screenshot is a real risk worth surfacing.
                            logger.warning(
                                "Path C capture failed at re-plan; falling back "
                                "to last-good screenshot: %s", e,
                            )
                            escalation_replan_screenshot = last_good_screenshot or ""
                        replans_used += 1
                        new_steps = self._planner.replan(
                            task_goal=task_goal,
                            failed_step=step,
                            failure_reason=(
                                result.reason or "Escalation resolved with re-plan"
                            ),
                            screenshot_base64=escalation_replan_screenshot,
                            remaining_steps=current_steps[i:],
                            skill_text=skill_text,
                            calibration_data=budget_history,
                        )
                        if not new_steps:
                            return WorkflowResult(
                                status="failed",
                                reason="Re-plan returned no steps",
                                steps_completed=len(completed_goals),
                                steps_total=len(current_steps),
                                step_results=step_results,
                                total_duration_s=time.monotonic() - start_time,
                            )
                        # Path C: step has already been marked 'done' and i
                        # advanced; use i-1 so cancellation covers anything
                        # remaining after the escalating step.
                        current_steps, cancelled_numbers = _cancel_and_append(
                            current_steps, i - 1, new_steps,
                        )
                        if on_progress:
                            on_progress({
                                "type": "replanned",
                                "old_step": step.number,
                                "reason": (
                                    result.reason
                                    or "Escalation resolved with re-plan"
                                ),
                                "new_steps": [
                                    {
                                        "number": s.number,
                                        "goal": s.goal,
                                        "reboot_expected": s.reboot_expected,
                                        "expected_actions": s.expected_actions,
                                        "expected_outcome": s.expected_outcome,
                                        "status": s.status,
                                    }
                                    for s in new_steps
                                ],
                                "cancelled_step_numbers": cancelled_numbers,
                                "steps_completed": len(completed_goals),
                                "screenshot_base64": escalation_replan_screenshot,
                            })
                            on_progress({"type": "escalation_resolved"})
                        continue
                    # else: choice == "resume"
                    if on_progress:
                        on_progress({"type": "escalation_resolved"})
                    continue
            else:
                # Phase 3 — Path B: step failed after all retries — gate on user decision.
                try:
                    cap = self._agent._capture.capture()
                    replan_screenshot = cap.base64_jpeg
                except Exception as e:
                    logger.warning(
                        "Path B capture failed at retry-exhausted gate; "
                        "falling back to last-good screenshot: %s", e,
                    )
                    replan_screenshot = last_good_screenshot or ""

                # _await_replan_decision arms self._active_gate='replan_B'.
                choice = self._await_replan_decision(
                    path="B",
                    payload={
                        "step_number": step.number,
                        "step_goal": step.goal,
                        "expected": None,
                        "observed": None,
                        "mismatch_reason": None,
                        "failure_reason": failure_reason,
                        "screenshot_base64": replan_screenshot,
                    },
                    on_progress=on_progress,
                )

                # H8 — emit resolved for every choice.
                if on_progress:
                    on_progress({
                        "type": "replan_dialog_resolved",
                        "choice": choice,
                    })

                if choice == "abort":
                    return WorkflowResult(
                        status="aborted",
                        reason="User aborted at step-failure dialog (Path B)",
                        steps_completed=len(completed_goals),
                        steps_total=len(current_steps),
                        step_results=step_results,
                        total_duration_s=time.monotonic() - start_time,
                    )

                if choice == "retry":
                    # H4 — delegate to _retry_step_once with full bookkeeping.
                    retry_result, new_i = self._retry_step_once(
                        step=step,
                        previous_failure=failure_reason,
                        on_progress=on_progress,
                        step_results=step_results,
                        completed_goals=completed_goals,
                        current_steps=current_steps,
                        task_goal=task_goal,
                        start_time=start_time,
                        step_index=i,
                        budget_history=budget_history,
                    )
                    if retry_result is not None:
                        # Retry failed → auto-abort (H4 decision (a)).
                        return retry_result
                    # Retry succeeded — bookkeeping applied in helper. Reset
                    # failures and advance via helper's returned index.
                    i = new_i
                    continue

                if choice == "replan":
                    if replans_used >= self._max_replans:
                        return WorkflowResult(
                            status="failed",
                            reason="Max replans exhausted",
                            steps_completed=len(completed_goals),
                            steps_total=len(current_steps),
                            step_results=step_results,
                            total_duration_s=time.monotonic() - start_time,
                        )
                    replans_used += 1
                    new_steps = self._planner.replan(
                        task_goal=task_goal,
                        failed_step=step,
                        failure_reason=failure_reason,
                        screenshot_base64=replan_screenshot,
                        remaining_steps=current_steps[i:],
                        skill_text=skill_text,
                        calibration_data=budget_history,
                    )
                    if not new_steps:
                        return WorkflowResult(
                            status="failed",
                            reason="Re-plan returned no steps",
                            steps_completed=len(completed_goals),
                            steps_total=len(current_steps),
                            step_results=step_results,
                            total_duration_s=time.monotonic() - start_time,
                        )
                    # Path B: cancel the failing step itself plus remaining;
                    # use i-1 so _cancel_and_append treats `i` as part of the
                    # tail (the failed step should be cancelled).
                    current_steps, cancelled_numbers = _cancel_and_append(
                        current_steps, i - 1, new_steps,
                    )
                    if on_progress:
                        on_progress({
                            "type": "replanned",
                            "old_step": step.number,
                            "reason": failure_reason,
                            "new_steps": [
                                {
                                    "number": s.number,
                                    "goal": s.goal,
                                    "reboot_expected": s.reboot_expected,
                                    "expected_actions": s.expected_actions,
                                    "expected_outcome": s.expected_outcome,
                                    "status": s.status,
                                }
                                for s in new_steps
                            ],
                            "cancelled_step_numbers": cancelled_numbers,
                            "steps_completed": len(completed_goals),
                            "screenshot_base64": replan_screenshot,
                        })
                    continue

        # All steps completed
        duration = time.monotonic() - start_time
        logger.info(
            "Workflow completed: %d steps in %.1fs",
            len(completed_goals), duration,
        )

        if on_progress:
            on_progress({
                "type": "workflow_done",
                "steps_completed": len(completed_goals),
                "duration_s": round(duration, 1),
            })

        return WorkflowResult(
            status="completed",
            reason="All steps completed successfully",
            steps_completed=len(completed_goals),
            steps_total=len(current_steps),
            step_results=step_results,
            total_duration_s=round(duration, 1),
        )
