"""Workflow runner — executes a plan's steps one at a time through the agent.

Orchestrates the plan-then-execute loop:
  1. Receive steps from the planner
  2. Execute each step via VisionAgent.run() with a narrowly scoped goal
  3. Handle retries, re-planning, and transitions between steps

The runner is generic — it has no knowledge of specific applications (BIOS,
WeChat, etc.). All domain-specific behavior comes from the skill text and
the planner's step decomposition.

Usage::

    from agent.planner import TaskPlanner
    from agent.workflow_runner import WorkflowRunner

    planner = TaskPlanner(provider, model, api_key)
    runner = WorkflowRunner(agent, planner, capture)
    result = runner.run(task_goal, skill_text, on_progress=callback)
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from agent.planner import PlanStep, TaskPlanner
from agent.vision_agent import TaskResult, TaskStatus, VisionAgent
from capture.base import compute_frame_diff

logger = logging.getLogger("M2.workflow")

_FRAME_DIFF_DEBUG = os.environ.get(
    "CYBERRACCOON_FRAME_DIFF_DEBUG", ""
).lower() in ("1", "true")


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
    ) -> None:
        self._agent = agent
        self._planner = planner
        self._max_retries = max_retries_per_step
        self._max_replans = max_replans
        self._max_steps_per_step = max_steps_per_step
        self._auto_approve = auto_approve

        # Plan approval gate — set by approve_plan() from the UI
        import threading
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
                    # Run validation
                    verdict, new_steps, reason = self._planner.validate_plan(
                        task_goal=task_goal,
                        screenshot_base64=val_screenshot,
                        completed_step=step,
                        remaining_steps=remaining or [step],
                        skill_text=skill_text,
                    )

                    if is_last_step:
                        # Last step: handle both replan and escalate
                        if verdict == "replan":
                            if replans_used >= self._max_replans:
                                logger.warning(
                                    "Workflow: last-step validation wants "
                                    "replan but max replans reached"
                                )
                            elif new_steps:
                                logger.info(
                                    "Workflow: last-step validation "
                                    "triggered replan: %s", reason,
                                )
                                replans_used += 1
                                for j, ns in enumerate(new_steps):
                                    ns.number = len(completed_goals) + 1 + j
                                if on_progress:
                                    on_progress({
                                        "type": "replanned",
                                        "old_step": step.number,
                                        "reason": reason,
                                        "new_steps": [
                                            {
                                                "number": s.number,
                                                "goal": s.goal,
                                                "reboot_expected": s.reboot_expected,
                                                "expected_actions": s.expected_actions,
                                                "expected_outcome": s.expected_outcome,
                                            }
                                            for s in new_steps
                                        ],
                                        "steps_completed": len(
                                            completed_goals
                                        ),
                                        "screenshot_base64": val_screenshot,
                                    })
                                current_steps = (
                                    current_steps[:i + 1] + new_steps
                                )
                        elif verdict == "escalate":
                            logger.info(
                                "Workflow: escalation on last step: %s",
                                reason,
                            )
                            if on_progress:
                                on_progress({
                                    "type": "escalate",
                                    "reason": reason,
                                    "step_number": step.number,
                                })
                            self._escalation_resolved.clear()
                            while not self._escalation_resolved.is_set():
                                if self._agent._abort_event.is_set():
                                    return WorkflowResult(
                                        status="aborted",
                                        reason="User aborted during "
                                               "escalation",
                                        steps_completed=len(completed_goals),
                                        steps_total=len(current_steps),
                                        step_results=step_results,
                                        total_duration_s=(
                                            time.monotonic() - start_time
                                        ),
                                    )
                                self._escalation_resolved.wait(timeout=0.5)
                            logger.info(
                                "Workflow: escalation resolved by user",
                            )
                    else:
                        if verdict == "replan":
                            if replans_used >= self._max_replans:
                                logger.warning(
                                    "Workflow: validation wants replan but "
                                    "max replans reached, continuing"
                                )
                            elif new_steps:
                                logger.info(
                                    "Workflow: post-step validation "
                                    "triggered replan: %s", reason,
                                )
                                replans_used += 1
                                for j, ns in enumerate(new_steps):
                                    ns.number = len(completed_goals) + 1 + j
                                if on_progress:
                                    on_progress({
                                        "type": "replanned",
                                        "old_step": step.number,
                                        "reason": reason,
                                        "new_steps": [
                                            {
                                                "number": s.number,
                                                "goal": s.goal,
                                                "reboot_expected": s.reboot_expected,
                                                "expected_actions": s.expected_actions,
                                                "expected_outcome": s.expected_outcome,
                                            }
                                            for s in new_steps
                                        ],
                                        "steps_completed": len(
                                            completed_goals
                                        ),
                                        "screenshot_base64": val_screenshot,
                                    })
                                current_steps = (
                                    current_steps[:i + 1] + new_steps
                                )

                        elif verdict == "escalate":
                            logger.info(
                                "Workflow: escalation requested: %s",
                                reason,
                            )
                            if on_progress:
                                on_progress({
                                    "type": "escalate",
                                    "reason": reason,
                                    "step_number": step.number,
                                })

                            # Wait for user to resolve
                            self._escalation_resolved.clear()
                            while not self._escalation_resolved.is_set():
                                if self._agent._abort_event.is_set():
                                    return WorkflowResult(
                                        status="aborted",
                                        reason="User aborted during "
                                               "escalation",
                                        steps_completed=len(completed_goals),
                                        steps_total=len(current_steps),
                                        step_results=step_results,
                                        total_duration_s=(
                                            time.monotonic() - start_time
                                        ),
                                    )
                                self._escalation_resolved.wait(timeout=0.5)

                            logger.info(
                                "Workflow: escalation resolved by user",
                            )

                            # Fresh capture after user intervention
                            try:
                                esc_cap = self._agent._capture.capture()
                                esc_screenshot = esc_cap.base64_jpeg
                                last_good_screenshot = esc_screenshot
                            except Exception as e:
                                logger.debug(
                                    "Post-escalation capture failed (step %d): "
                                    "%s — using last good screenshot",
                                    step.number, e,
                                )
                                esc_screenshot = last_good_screenshot

                            if on_progress:
                                on_progress({
                                    "type": "escalation_resolved",
                                    "step_number": step.number,
                                })

                            # Re-validate plan after user intervention
                            esc_verdict, esc_new_steps, esc_reason = (
                                self._planner.validate_plan(
                                    task_goal=task_goal,
                                    screenshot_base64=esc_screenshot,
                                    completed_step=step,
                                    remaining_steps=remaining,
                                    skill_text=skill_text,
                                )
                            )
                            if esc_verdict == "replan" and esc_new_steps:
                                if replans_used < self._max_replans:
                                    logger.info(
                                        "Workflow: post-escalation replan: "
                                        "%s", esc_reason,
                                    )
                                    replans_used += 1
                                    for j, ns in enumerate(esc_new_steps):
                                        ns.number = (
                                            len(completed_goals) + 1 + j
                                        )
                                    if on_progress:
                                        on_progress({
                                            "type": "replanned",
                                            "old_step": step.number,
                                            "reason": esc_reason,
                                            "new_steps": [
                                                {
                                                    "number": s.number,
                                                    "goal": s.goal,
                                                    "reboot_expected": s.reboot_expected,
                                                    "expected_actions": s.expected_actions,
                                                    "expected_outcome": s.expected_outcome,
                                                }
                                                for s in esc_new_steps
                                            ],
                                            "steps_completed": len(
                                                completed_goals
                                            ),
                                            "screenshot_base64": esc_screenshot,
                                        })
                                    current_steps = (
                                        current_steps[:i + 1]
                                        + esc_new_steps
                                    )

                i += 1
            else:
                # Step failed after all retries — try re-planning
                if replans_used >= self._max_replans:
                    logger.error(
                        "Workflow: step %d failed and no re-plans left",
                        step.number,
                    )
                    return WorkflowResult(
                        status="failed",
                        reason=(
                            f"Step {step.number} failed after "
                            f"{self._max_retries} retries and "
                            f"{replans_used} re-plans: {failure_reason}"
                        ),
                        steps_completed=len(completed_goals),
                        steps_total=len(current_steps),
                        step_results=step_results,
                        total_duration_s=time.monotonic() - start_time,
                    )

                logger.info(
                    "Workflow: re-planning after step %d failure",
                    step.number,
                )
                replans_used += 1

                # Capture current screen for re-planner
                try:
                    cap = self._agent._capture.capture()
                    replan_screenshot = cap.base64_jpeg
                except Exception as e:
                    logger.debug(
                        "Replan capture failed (step %d): %s — "
                        "using initial screenshot",
                        step.number, e,
                    )
                    replan_screenshot = screenshot_base64  # fallback

                remaining = current_steps[i:]
                new_steps = self._planner.replan(
                    task_goal=task_goal,
                    failed_step=step,
                    failure_reason=failure_reason,
                    screenshot_base64=replan_screenshot,
                    remaining_steps=remaining,
                    skill_text=skill_text,
                    calibration_data=budget_history,
                )

                if new_steps is None or len(new_steps) == 0:
                    logger.error("Workflow: re-planning produced no steps")
                    return WorkflowResult(
                        status="failed",
                        reason=(
                            f"Step {step.number} failed and re-planning "
                            f"produced no usable steps: {failure_reason}"
                        ),
                        steps_completed=len(completed_goals),
                        steps_total=len(current_steps),
                        step_results=step_results,
                        total_duration_s=time.monotonic() - start_time,
                    )

                # Renumber new steps to continue after completed
                for j, ns in enumerate(new_steps):
                    ns.number = len(completed_goals) + 1 + j

                current_steps = current_steps[:i] + new_steps

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
                            }
                            for s in new_steps
                        ],
                        "steps_completed": len(completed_goals),
                        "screenshot_base64": replan_screenshot,
                    })

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
