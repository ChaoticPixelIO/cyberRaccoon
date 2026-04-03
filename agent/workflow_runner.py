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
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from agent.planner import PlanStep, TaskPlanner
from agent.vision_agent import TaskResult, TaskStatus, VisionAgent
from capture.base import compute_frame_diff

logger = logging.getLogger("M2.workflow")


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

        # Escalation gate — set by resolve_escalation() from the UI
        self._escalation_resolved = threading.Event()

    def approve_plan(self) -> None:
        """Signal that the user approved the plan. Unblocks execution."""
        self._plan_rejected = False
        self._plan_approved.set()

    def reject_plan(self) -> None:
        """Signal that the user rejected the plan. Aborts workflow."""
        self._plan_rejected = True
        self._plan_approved.set()

    def resolve_escalation(self) -> None:
        """Signal that the user resolved the escalation. Unblocks workflow."""
        self._escalation_resolved.set()

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
            on_progress({
                "type": "plan_ready",
                "steps": [
                    {
                        "number": s.number,
                        "goal": s.goal,
                        "reboot_expected": s.reboot_expected,
                        "expected_actions": s.expected_actions,
                    }
                    for s in steps
                ],
                "task_goal": task_goal,
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

        logger.info("Workflow: plan approved, starting execution")

        # ---- Phase 2: Execution ----
        completed_goals: list[str] = []
        step_results: list[dict[str, Any]] = []
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
            except Exception:
                pass  # validation gate will fall back to always-validate

            # Execute step with retries
            step_goal = step.format_for_agent(
                total_steps=len(current_steps),
                completed=completed_goals if completed_goals else None,
            )

            success = False
            failure_reason = ""

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
                else:
                    failure_reason = result.reason
                    logger.warning(
                        "Workflow: step %d attempt %d failed: %s",
                        step.number, attempt + 1, failure_reason,
                    )

            step_results.append({
                "step_number": step.number,
                "step_goal": step.goal,
                "success": success,
                "failure_reason": failure_reason if not success else "",
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
                except Exception:
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
                                            }
                                            for s in new_steps
                                        ],
                                        "steps_completed": len(
                                            completed_goals
                                        ),
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
                                            }
                                            for s in new_steps
                                        ],
                                        "steps_completed": len(
                                            completed_goals
                                        ),
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
                            except Exception:
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
                                                }
                                                for s in esc_new_steps
                                            ],
                                            "steps_completed": len(
                                                completed_goals
                                            ),
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
                except Exception:
                    replan_screenshot = screenshot_base64  # fallback

                remaining = current_steps[i:]
                new_steps = self._planner.replan(
                    task_goal=task_goal,
                    failed_step=step,
                    failure_reason=failure_reason,
                    screenshot_base64=replan_screenshot,
                    remaining_steps=remaining,
                    skill_text=skill_text,
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
                        "new_steps": [
                            {"number": s.number, "goal": s.goal}
                            for s in new_steps
                        ],
                        "steps_completed": len(completed_goals),
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
