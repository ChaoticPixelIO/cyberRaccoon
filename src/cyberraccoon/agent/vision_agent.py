"""M2 Vision Agent — core orchestrator for the capture-decide-act loop.

Synchronous blocking design. All errors are encapsulated in TaskResult;
this module never raises exceptions to callers.

Flow:
  0. Initial capture (no stability check — nothing has changed yet)
  Loop:
    1. Check termination conditions (abort, max steps, timeout, consecutive failures)
    2. Send screenshot + context to LLM via M3
    3. If action == "done", return COMPLETED
    4. Execute command via M4
    5. Update conversation history (screen_summary, not full screenshots)
    6. Notify via on_step callback
    7. Sleep post_action_delay, then capture with UI stability check
"""

from __future__ import annotations

import logging
import re
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from cyberraccoon.capture.base import CaptureError, CaptureResult, CaptureSource, compute_frame_diff
from cyberraccoon.agent.protocols.base import ComputerUseProtocol, StepResult
from cyberraccoon.executor.base_executor import BaseExecutor

logger = logging.getLogger("M2.agent")


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"
    PAUSED = "paused"


@dataclass
class TaskResult:
    """Outcome of a full task run."""

    status: TaskStatus
    reason: str                                     # Why the task ended
    total_steps: int
    total_input_tokens: int
    total_output_tokens: int
    total_duration_s: float
    total_cache_read_tokens: int = 0
    total_cache_creation_tokens: int = 0
    step_log: list[dict[str, Any]] = field(default_factory=list)
    completion_status: str = "success"  # "success", "gave_up", or "stuck"


class VisionAgent:
    """Core agent that orchestrates capture -> LLM -> execute loop.

    Usage::

        agent = VisionAgent(capture, llm, executor)
        result = agent.run("Open Notepad and type Hello", on_step=print)
    """

    def __init__(
        self,
        capture: CaptureSource,
        protocol: ComputerUseProtocol,
        executor: BaseExecutor,
        max_steps: int = 50,
        max_consecutive_failures: int = 3,
        post_action_delay_s: float = 1.0,
        task_timeout_s: float = 3000.0,
        stability_check: bool = True,
        stability_threshold: float = 2.0,
        stability_interval_s: float = 0.5,
        stability_max_wait_s: float = 5.0,
    ) -> None:
        self._capture = capture
        self._protocol = protocol
        self._executor = executor
        self._max_steps = max_steps
        self._max_consecutive_failures = max_consecutive_failures
        self._post_action_delay_s = post_action_delay_s
        self._task_timeout_s = task_timeout_s
        self._stability_check = stability_check
        self._stability_threshold = stability_threshold
        self._stability_interval_s = stability_interval_s
        self._stability_max_wait_s = stability_max_wait_s
        self._abort_event = threading.Event()
        self._pause_event = threading.Event()
        self._step_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="M2-step")

        # BIOS reboot transition state
        self._in_bios_mode = False
        self._expecting_reboot = False

    def run_workflow(
        self,
        task_goal: str,
        planner: Any,
        skill_text: str | None = None,
        on_step: Callable[[dict[str, Any]], None] | None = None,
    ) -> Any:
        """Execute a task using the plan-then-execute workflow.

        Uses the planner to decompose the task into steps, then executes
        each step individually via :meth:`run`. This gives better control
        than a single free-form ``run()`` call, especially for tasks that
        require following specific procedures from skills.

        Args:
            task_goal: Natural language task description.
            planner: TaskPlanner instance for step decomposition.
            skill_text: Optional skill markdown for planning context.
            on_step: Callback for progress updates.

        Returns:
            WorkflowResult with status and step details.
        """
        from cyberraccoon.agent.workflow_runner import WorkflowRunner

        # Capture initial screenshot for the planner
        try:
            cap = self._capture.capture()
            screenshot_b64 = cap.base64_jpeg
        except CaptureError as e:
            from cyberraccoon.agent.workflow_runner import WorkflowResult
            return WorkflowResult(
                status="failed",
                reason=f"Initial capture failed: {e}",
                steps_completed=0,
                steps_total=0,
            )

        runner = WorkflowRunner(agent=self, planner=planner)
        self._workflow_runner = runner  # expose for plan approval
        return runner.run(
            task_goal=task_goal,
            screenshot_base64=screenshot_b64,
            skill_text=skill_text,
            on_progress=on_step,
        )

    def run(
        self,
        task_goal: str,
        on_step: Callable[[dict[str, Any]], None] | None = None,
        *,
        _preserve_transition_state: bool = False,
    ) -> TaskResult:
        """Execute the task loop until completion or termination.

        Args:
            task_goal: Natural language description of the task.
            on_step: Optional callback invoked after each step with step info dict.
                     Useful for UI status updates (M5).
            _preserve_transition_state: Internal flag used by workflow runner.
                     When True, does not reset ``_in_bios_mode`` and
                     ``_expecting_reboot`` between steps, preserving
                     reboot transition continuity across workflow steps.

        Returns:
            TaskResult with status, reason, token usage, and step log.
        """
        logger.info("Task started: %s", task_goal)

        # Drain any orphaned task from a previous abort before reusing
        # the single-worker executor (prevents blocking on stale future).
        self._step_executor.shutdown(wait=True)
        self._step_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="M2-step",
        )

        self._abort_event.clear()
        self._protocol.reset()
        if not _preserve_transition_state:
            self._in_bios_mode = False
            self._expecting_reboot = False
        start_time = time.monotonic()
        step_count = 0
        consecutive_failures = 0
        step_log: list[dict[str, Any]] = []
        total_input_tokens = 0
        total_output_tokens = 0
        total_cache_read_tokens = 0
        total_cache_creation_tokens = 0

        try:
            # ---- Initial capture (with reconnect on failure) ----
            try:
                cap_result = self._capture.capture()
            except CaptureError as e:
                # Capture device may be stale (e.g. after a reboot in a
                # previous workflow step). Try reconnecting once.
                logger.warning(
                    "Initial capture failed (%s), attempting device "
                    "reconnect", e,
                )
                try:
                    self._capture.close()
                    time.sleep(1.0)
                    self._capture.open()
                    cap_result = self._capture.capture()
                    logger.info("Capture device reconnected successfully")
                except CaptureError as e2:
                    logger.error("Initial capture failed after reconnect: %s", e2)
                    return self._build_result(
                        TaskStatus.FAILED, f"Capture failed: {e2}",
                        step_count, total_input_tokens, total_output_tokens,
                        start_time, step_log,
                        total_cache_read_tokens, total_cache_creation_tokens,
                    )

            # Auto-detect target OS if not set
            if hasattr(self._executor, '_target_os') and self._executor._target_os is None:
                detected = self._protocol.detect_os(cap_result.base64_jpeg)
                if detected:
                    try:
                        from cyberraccoon.executor.clipboard_bridge import TargetOS
                        self._executor._target_os = TargetOS(detected)
                        logger.info("Auto-detected target OS: %s", detected)
                    except ValueError:
                        logger.warning("OS detection returned unknown value: %s", detected)

            # Emit initial screenshot so UI shows it immediately
            if on_step:
                try:
                    on_step({
                        "step": 0,
                        "command": None,
                        "llm_latency_ms": 0,
                        "execute_status": "screenshot",
                        "timestamp": time.monotonic(),
                        "screenshot_base64": cap_result.base64_jpeg,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_input_tokens": 0,
                        "total_output_tokens": 0,
                        "cache_read_tokens": 0,
                        "cache_creation_tokens": 0,
                        "total_cache_read_tokens": 0,
                        "total_cache_creation_tokens": 0,
                    })
                except Exception as cb_err:
                    logger.warning("on_step callback error: %s", cb_err)

            while True:
                # ---- Termination condition checks ----
                if self._abort_event.is_set():
                    return self._build_result(
                        TaskStatus.ABORTED, "User aborted",
                        step_count, total_input_tokens, total_output_tokens,
                        start_time, step_log,
                        total_cache_read_tokens, total_cache_creation_tokens,
                    )

                if self._pause_event.is_set():
                    return self._build_result(
                        TaskStatus.PAUSED, "User paused execution",
                        step_count, total_input_tokens, total_output_tokens,
                        start_time, step_log,
                        total_cache_read_tokens, total_cache_creation_tokens,
                    )

                if step_count >= self._max_steps:
                    return self._build_result(
                        TaskStatus.FAILED,
                        f"Reached max steps ({self._max_steps})",
                        step_count, total_input_tokens, total_output_tokens,
                        start_time, step_log,
                        total_cache_read_tokens, total_cache_creation_tokens,
                    )

                elapsed = time.monotonic() - start_time
                if elapsed > self._task_timeout_s:
                    return self._build_result(
                        TaskStatus.FAILED,
                        f"Task timeout ({self._task_timeout_s}s)",
                        step_count, total_input_tokens, total_output_tokens,
                        start_time, step_log,
                        total_cache_read_tokens, total_cache_creation_tokens,
                    )

                if consecutive_failures >= self._max_consecutive_failures:
                    return self._build_result(
                        TaskStatus.FAILED,
                        f"Too many consecutive failures ({consecutive_failures})",
                        step_count, total_input_tokens, total_output_tokens,
                        start_time, step_log,
                        total_cache_read_tokens, total_cache_creation_tokens,
                    )

                step_count += 1
                step_timestamp = time.monotonic()

                # ---- Step 1: Call protocol (using previously captured frame) ----
                # Run in sub-thread so abort can interrupt a blocking API call.
                future: Future[StepResult] = self._step_executor.submit(
                    self._protocol.step, cap_result.base64_jpeg, task_goal,
                )
                while not future.done():
                    if self._abort_event.wait(timeout=0.5):
                        if not future.cancel():
                            logger.warning(
                                "Abort: API call still in progress, "
                                "will complete in background (up to 30s)"
                            )
                        return self._build_result(
                            TaskStatus.ABORTED, "User aborted",
                            step_count, total_input_tokens, total_output_tokens,
                            start_time, step_log,
                            total_cache_read_tokens, total_cache_creation_tokens,
                        )
                    if self._pause_event.is_set():
                        # Wait for the LLM call to finish (don't waste it),
                        # then return PAUSED without executing the action.
                        future.result()  # blocks until LLM finishes
                        return self._build_result(
                            TaskStatus.PAUSED, "User paused execution",
                            step_count, total_input_tokens, total_output_tokens,
                            start_time, step_log,
                            total_cache_read_tokens, total_cache_creation_tokens,
                        )
                step_result = future.result()
                total_input_tokens += step_result.input_tokens
                total_output_tokens += step_result.output_tokens
                total_cache_read_tokens += step_result.cache_read_tokens
                total_cache_creation_tokens += step_result.cache_creation_tokens

                if not step_result.success:
                    logger.warning(
                        "Step %d: Protocol failed: %s",
                        step_count, step_result.error,
                    )
                    consecutive_failures += 1
                    self._protocol.report_result(False, step_result.error)
                    step_log.append({
                        "step": step_count,
                        "command": None,
                        "llm_latency_ms": step_result.latency_ms,
                        "llm_error": step_result.error,
                        "execute_status": "skipped",
                        "timestamp": step_timestamp,
                        "llm_response_text": step_result.raw_text,
                        "system_prompt": self._protocol.get_system_prompt(),
                        "prompt_messages": self._truncate_base64_in_messages(self._protocol.get_messages_snapshot()),
                        "input_tokens": step_result.input_tokens,
                        "output_tokens": step_result.output_tokens,
                        "total_input_tokens": total_input_tokens,
                        "total_output_tokens": total_output_tokens,
                        "cache_read_tokens": step_result.cache_read_tokens,
                        "cache_creation_tokens": step_result.cache_creation_tokens,
                        "total_cache_read_tokens": total_cache_read_tokens,
                        "total_cache_creation_tokens": total_cache_creation_tokens,
                    })
                    # Re-capture before retrying (screen may have changed)
                    try:
                        cap_result = self._capture.capture()
                    except CaptureError as e:
                        logger.error("Step %d: Capture failed: %s", step_count, e)
                        return self._build_result(
                            TaskStatus.FAILED, f"Capture failed: {e}",
                            step_count, total_input_tokens, total_output_tokens,
                            start_time, step_log,
                            total_cache_read_tokens, total_cache_creation_tokens,
                        )
                    continue

                # ---- Step 2: Check if task is done ----
                if step_result.is_done and not step_result.get_commands():
                    reason = step_result.done_reason or "Task completed"
                    logger.info("Task done at step %d: %s", step_count, reason)
                    done_info: dict[str, Any] = {
                        "step": step_count,
                        "command": {"action": "done", "reason": reason},
                        "llm_latency_ms": step_result.latency_ms,
                        "execute_status": "done",
                        "timestamp": step_timestamp,
                        "screenshot_base64": cap_result.base64_jpeg,
                        "llm_response_text": step_result.raw_text,
                        "system_prompt": self._protocol.get_system_prompt(),
                        "prompt_messages": self._truncate_base64_in_messages(self._protocol.get_messages_snapshot()),
                        "input_tokens": step_result.input_tokens,
                        "output_tokens": step_result.output_tokens,
                        "total_input_tokens": total_input_tokens,
                        "total_output_tokens": total_output_tokens,
                        "cache_read_tokens": step_result.cache_read_tokens,
                        "cache_creation_tokens": step_result.cache_creation_tokens,
                        "total_cache_read_tokens": total_cache_read_tokens,
                        "total_cache_creation_tokens": total_cache_creation_tokens,
                    }
                    step_log.append(done_info)

                    if on_step:
                        try:
                            on_step(done_info)
                        except Exception as cb_err:
                            logger.warning("on_step callback error: %s", cb_err)

                    return self._build_result(
                        TaskStatus.COMPLETED, reason,
                        step_count, total_input_tokens, total_output_tokens,
                        start_time, step_log,
                        total_cache_read_tokens, total_cache_creation_tokens,
                        completion_status=step_result.completion_status,
                    )

                # ---- Step 2b: Handle screenshot request ----
                if step_result.needs_screenshot:
                    logger.debug("Step %d: Model requested screenshot", step_count)
                    try:
                        cap_result = self._wait_for_stable_screen()
                    except CaptureError as e:
                        step_log.append({
                            "step": step_count,
                            "command": None,
                            "llm_latency_ms": step_result.latency_ms,
                            "execute_status": "screenshot",
                            "timestamp": step_timestamp,
                        })
                        logger.error("Step %d: Capture failed: %s", step_count, e)
                        return self._build_result(
                            TaskStatus.FAILED, f"Capture failed: {e}",
                            step_count, total_input_tokens, total_output_tokens,
                            start_time, step_log,
                            total_cache_read_tokens, total_cache_creation_tokens,
                        )
                    screenshot_step = {
                        "step": step_count,
                        "command": None,
                        "llm_latency_ms": step_result.latency_ms,
                        "execute_status": "screenshot",
                        "timestamp": step_timestamp,
                        "screenshot_base64": cap_result.base64_jpeg,
                        "input_tokens": step_result.input_tokens,
                        "output_tokens": step_result.output_tokens,
                        "total_input_tokens": total_input_tokens,
                        "total_output_tokens": total_output_tokens,
                        "cache_read_tokens": step_result.cache_read_tokens,
                        "cache_creation_tokens": step_result.cache_creation_tokens,
                        "total_cache_read_tokens": total_cache_read_tokens,
                        "total_cache_creation_tokens": total_cache_creation_tokens,
                    }
                    step_log.append(screenshot_step)
                    if on_step:
                        try:
                            on_step(screenshot_step)
                        except Exception as cb_err:
                            logger.warning("on_step callback error: %s", cb_err)
                    continue

                # ---- Step 3: Execute command batch ----
                commands = step_result.get_commands()
                exec_results: list[tuple[bool, str | None]] = []
                all_succeeded = True

                reboot_triggered = False
                for i, cmd in enumerate(commands):
                    # M2 generates the command ID, not the LLM
                    cmd.setdefault(
                        "id", f"step_{step_count}_{i}_{uuid.uuid4().hex[:6]}"
                    )
                    try:
                        exec_result = self._executor.execute(cmd)
                    except (KeyError, TypeError, ValueError) as e:
                        logger.warning(
                            "Step %d: Malformed command %s: %s",
                            step_count, cmd.get("action"), e,
                        )
                        exec_result = {"status": "error", "message": str(e)}

                    if exec_result["status"] in ("ok", "skipped"):
                        exec_results.append((True, None))
                    else:
                        error_msg = (
                            exec_result.get("error")
                            or exec_result.get("message", "Unknown executor error")
                        )
                        exec_results.append((False, error_msg))
                        all_succeeded = False
                        # Fill remaining as not-executed
                        for _ in range(i + 1, len(commands)):
                            exec_results.append((False, "Skipped due to earlier failure"))
                        break

                    # Interrupt batch if a reboot command was just executed
                    if self._is_reboot_command(cmd):
                        self._expecting_reboot = True
                        logger.info(
                            "Step %d: Reboot command detected in batch "
                            "item %d/%d, interrupting batch",
                            step_count, i + 1, len(commands),
                        )
                        reboot_triggered = True
                        # Mark remaining commands as skipped
                        for _ in range(i + 1, len(commands)):
                            exec_results.append(
                                (False, "Skipped: reboot command interrupted batch")
                            )
                        break

                # Determine overall status for step_info
                if all_succeeded:
                    consecutive_failures = 0
                    overall_status = "ok"
                else:
                    consecutive_failures += 1
                    overall_status = "error"

                # Report results to protocol
                self._protocol.report_results(exec_results)

                step_info: dict[str, Any] = {
                    "step": step_count,
                    "command": commands[0] if commands else None,
                    "commands": commands,
                    "batch_size": len(commands),
                    "exec_results": exec_results,
                    "llm_latency_ms": step_result.latency_ms,
                    "execute_status": overall_status,
                    "screenshot_base64": cap_result.base64_jpeg,
                    "timestamp": step_timestamp,
                    "llm_response_text": step_result.raw_text,
                    "system_prompt": self._protocol.get_system_prompt(),
                    "prompt_messages": self._truncate_base64_in_messages(self._protocol.get_messages_snapshot()),
                    "input_tokens": step_result.input_tokens,
                    "output_tokens": step_result.output_tokens,
                    "total_input_tokens": total_input_tokens,
                    "total_output_tokens": total_output_tokens,
                    "cache_read_tokens": step_result.cache_read_tokens,
                    "cache_creation_tokens": step_result.cache_creation_tokens,
                    "total_cache_read_tokens": total_cache_read_tokens,
                    "total_cache_creation_tokens": total_cache_creation_tokens,
                }
                step_log.append(step_info)

                if not all_succeeded:
                    logger.warning(
                        "Step %d: Batch execution failed at command %d/%d",
                        step_count,
                        next(i for i, (s, _) in enumerate(exec_results) if not s) + 1,
                        len(commands),
                    )

                # ---- Step 4: Notify callback ----
                if on_step:
                    try:
                        on_step(step_info)
                    except Exception as cb_err:
                        logger.warning("on_step callback error: %s", cb_err)

                # ---- Step 4b: Complete if batch included trailing done ----
                if step_result.is_done:
                    reason = step_result.done_reason or "Task completed"
                    logger.info("Task done at step %d: %s", step_count, reason)
                    return self._build_result(
                        TaskStatus.COMPLETED, reason,
                        step_count, total_input_tokens, total_output_tokens,
                        start_time, step_log,
                        total_cache_read_tokens, total_cache_creation_tokens,
                        completion_status=step_result.completion_status,
                    )

                # ---- Step 5: Wait for UI to settle, then capture ----
                # Enter reboot transition if a reboot command was detected.
                # Note: we do NOT check for black screen in BIOS mode here
                # because BIOS screens can flicker briefly during page
                # transitions. The WorkflowRunner handles reboots between
                # steps via [REBOOT EXPECTED] tags and pre-step capture
                # health checks.
                use_reboot_wait = reboot_triggered

                if use_reboot_wait:
                    timeout = 60.0 if not self._in_bios_mode else 120.0
                    try:
                        cap_result = self._wait_for_reboot_transition(
                            timeout_s=timeout,
                        )
                        # Update BIOS mode state
                        if self._expecting_reboot and not self._in_bios_mode:
                            self._in_bios_mode = True
                            logger.info("Entered BIOS mode")
                        elif self._in_bios_mode:
                            self._in_bios_mode = False
                            logger.info("Exited BIOS mode, back to OS")
                        self._expecting_reboot = False
                    except CaptureError as e:
                        logger.error(
                            "Step %d: Reboot transition failed: %s",
                            step_count, e,
                        )
                        self._expecting_reboot = False
                        self._in_bios_mode = False
                        return self._build_result(
                            TaskStatus.FAILED, f"Reboot transition failed: {e}",
                            step_count, total_input_tokens, total_output_tokens,
                            start_time, step_log,
                            total_cache_read_tokens, total_cache_creation_tokens,
                        )
                else:
                    if self._post_action_delay_s > 0:
                        time.sleep(self._post_action_delay_s)

                    try:
                        cap_result = self._wait_for_stable_screen()
                    except CaptureError as e:
                        # Capture failed. Try device reconnect first
                        # (handles stale V4L2/CSI handles after HDMI
                        # signal changes). If that fails, try full
                        # reboot transition recovery.
                        logger.warning(
                            "Step %d: Capture failed (%s), attempting "
                            "device reconnect",
                            step_count, e,
                        )
                        try:
                            self._capture.close()
                            time.sleep(1.0)
                            self._capture.open()
                            cap_result = self._wait_for_stable_screen()
                            logger.info(
                                "Step %d: Capture recovered after "
                                "device reconnect", step_count,
                            )
                        except CaptureError as e2:
                            logger.error(
                                "Step %d: Reconnect failed (%s), "
                                "capture unrecoverable",
                                step_count, e2,
                            )
                            return self._build_result(
                                TaskStatus.FAILED,
                                f"Capture failed: {e2}",
                                step_count, total_input_tokens,
                                total_output_tokens,
                                start_time, step_log,
                                total_cache_read_tokens,
                                total_cache_creation_tokens,
                            )

        except Exception as e:
            logger.error("Unexpected error in agent loop: %s", e, exc_info=True)
            return self._build_result(
                TaskStatus.FAILED, f"Unexpected error: {e}",
                step_count, total_input_tokens, total_output_tokens,
                start_time, step_log,
                total_cache_read_tokens, total_cache_creation_tokens,
            )

    def abort(self) -> None:
        """Request the running task to stop immediately."""
        self._abort_event.set()
        logger.info("Abort requested")

    def pause(self) -> None:
        """Request the running task to pause at the next opportunity.

        This is the public API for pausing. External callers (AppController)
        MUST use this method instead of setting _pause_event directly.
        Pause takes effect at the next cycle boundary -- between
        capture/decide/act phases or after a pending LLM call completes.
        No HID action fires after pause is set.
        """
        self._pause_event.set()
        logger.info("Pause requested")

    # ------------------------------------------------------------------
    # BIOS reboot transition handling
    # ------------------------------------------------------------------

    # Patterns that indicate a reboot-to-firmware command was typed.
    _REBOOT_PATTERNS = [
        re.compile(r"shutdown\s+/r\s+/fw", re.IGNORECASE),
        re.compile(r"systemctl\s+reboot\s+--firmware", re.IGNORECASE),
    ]

    @staticmethod
    def _is_reboot_command(cmd: dict[str, Any]) -> bool:
        """Check if a command is a reboot-to-firmware command.

        Looks for known reboot patterns in the ``text`` field of ``type``
        or ``key`` actions.
        """
        if cmd.get("action") != "type":
            return False
        text = cmd.get("text", "")
        if not text:
            return False
        return any(p.search(text) for p in VisionAgent._REBOOT_PATTERNS)

    @staticmethod
    def _is_screen_black(capture_result: CaptureResult, threshold: int = 20) -> bool:
        """Check if a captured frame is mostly black (mean pixel < threshold)."""
        if capture_result.image is None:
            return True
        import numpy as np
        gray = np.asarray(capture_result.image.convert("L"), dtype=np.uint8)
        return float(gray.mean()) < threshold

    def _wait_for_reboot_transition(
        self,
        timeout_s: float = 60.0,
        poll_interval_s: float = 0.5,
        stability_duration_s: float = 5.0,
    ) -> CaptureResult:
        """Wait for a reboot transition to complete.

        Polls the capture device through three stages:
        1. Wait for screen to go black (reboot started)
        2. Wait for non-black stable screen (BIOS or OS appeared)
        3. Return the stable frame

        On CaptureError (e.g. HDMI signal loss causing stale handle),
        attempts to reconnect the capture device via close() + open().

        Args:
            timeout_s: Maximum seconds to wait for a stable screen.
            poll_interval_s: Seconds between capture polls.
            stability_duration_s: Seconds of consecutive stable non-black
                frames required before declaring transition complete.

        Returns:
            CaptureResult of the first stable post-reboot frame.

        Raises:
            CaptureError: If timeout expires without a stable screen.
        """
        start = time.monotonic()
        saw_black = False
        stable_since: float | None = None
        last_frame: CaptureResult | None = None
        consecutive_errors = 0
        max_consecutive_errors = 5
        # Detect frozen capture device: if frames are perfectly identical
        # (diff < 0.1%) for too long, the device is returning stale buffers.
        frozen_since: float | None = None
        frozen_reconnect_threshold_s = 10.0

        logger.info(
            "Reboot transition: waiting up to %.0fs for stable screen",
            timeout_s,
        )

        while True:
            elapsed = time.monotonic() - start
            if elapsed > timeout_s:
                raise CaptureError(
                    f"Reboot transition timed out after {timeout_s:.0f}s. "
                    "Machine may have booted directly to OS, hung during "
                    "POST, or have a BIOS password."
                )

            if self._abort_event.is_set():
                raise CaptureError("Reboot transition aborted by user")

            # Attempt capture, reconnect on repeated failures
            try:
                frame = self._capture.capture()
                consecutive_errors = 0
            except CaptureError:
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    logger.warning(
                        "Reboot transition: %d consecutive capture errors, "
                        "attempting device reconnect",
                        consecutive_errors,
                    )
                    try:
                        self._capture.close()
                        time.sleep(1.0)
                        self._capture.open()
                        consecutive_errors = 0
                        logger.info("Capture device reconnected successfully")
                    except CaptureError as reopen_err:
                        logger.warning(
                            "Capture reconnect failed: %s", reopen_err,
                        )
                time.sleep(poll_interval_s)
                continue

            is_black = self._is_screen_black(frame)

            if is_black:
                if not saw_black:
                    logger.debug("Reboot transition: screen went black")
                saw_black = True
                stable_since = None
                last_frame = None
                frozen_since = None
            else:
                # Non-black frame
                if last_frame is not None:
                    try:
                        diff = compute_frame_diff(last_frame.image, frame.image)
                    except Exception:
                        diff = 100.0

                    # Detect frozen device: perfectly identical frames
                    # (diff < 0.1%) for too long means the CSI device is
                    # returning stale buffered frames, not live video.
                    if diff < 0.1:
                        if frozen_since is None:
                            frozen_since = time.monotonic()
                        elif time.monotonic() - frozen_since >= frozen_reconnect_threshold_s:
                            logger.warning(
                                "Reboot transition: frames frozen for "
                                "%.0fs, forcing device reconnect",
                                time.monotonic() - frozen_since,
                            )
                            try:
                                self._capture.close()
                                time.sleep(2.0)
                                self._capture.open()
                                logger.info("Capture device reconnected")
                            except CaptureError as re_err:
                                logger.warning(
                                    "Reconnect failed: %s", re_err,
                                )
                            frozen_since = None
                            stable_since = None
                            last_frame = None
                            time.sleep(poll_interval_s)
                            continue
                    else:
                        frozen_since = None

                    if diff < self._stability_threshold:
                        # Frame is stable (but not frozen)
                        if stable_since is None:
                            stable_since = time.monotonic()
                        elif time.monotonic() - stable_since >= stability_duration_s:
                            logger.info(
                                "Reboot transition complete: stable screen "
                                "detected after %.1fs",
                                time.monotonic() - start,
                            )
                            return frame
                    else:
                        # Screen still changing
                        stable_since = None
                else:
                    stable_since = None

                last_frame = frame

            time.sleep(poll_interval_s)

    def _wait_for_stable_screen(self) -> CaptureResult:
        """Capture frames until the screen stops changing, then return the last.

        If stability checking is disabled, captures and returns a single frame.

        Raises:
            CaptureError: If any capture call fails.
        """
        frame_a = self._capture.capture()

        if not self._stability_check:
            return frame_a

        wait_start = time.monotonic()
        while True:
            time.sleep(self._stability_interval_s)
            frame_b = self._capture.capture()

            try:
                diff_pct = compute_frame_diff(frame_a.image, frame_b.image)
            except Exception as e:
                logger.warning("Frame diff failed: %s, treating as unstable", e)
                diff_pct = 100.0
            if diff_pct < self._stability_threshold:
                logger.debug("Screen stable (%.1f%% diff)", diff_pct)
                return frame_b

            elapsed = time.monotonic() - wait_start
            if elapsed >= self._stability_max_wait_s:
                logger.warning(
                    "Stability timeout after %.1fs (%.1f%% diff), proceeding",
                    elapsed, diff_pct,
                )
                return frame_b

            frame_a = frame_b

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    # Regex for OpenAI data URI images
    _DATA_URI_RE = re.compile(r"^(data:image/[^;]+;base64,)")

    @staticmethod
    def _truncate_base64_in_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Replace base64 image data with short placeholders in a messages list.

        Performs shallow copies at each level, so it is safe to use on the
        output of get_messages_snapshot() (which already deep-copies).

        Handles four content formats used across protocols:
        - Anthropic image block: content[].source.data
        - Anthropic tool_result nested image: content[].content[].source.data
        - OpenAI image_url with data URI: content[].image_url.url
        - Plain string content: left as-is
        """

        def _truncate(data: str) -> str:
            return data[:20] + f"...({len(data)} bytes base64)"

        def _process_block(block: dict[str, Any]) -> dict[str, Any]:
            block = dict(block)  # shallow copy of this level
            btype = block.get("type")
            # Anthropic image block
            if btype == "image" and isinstance(block.get("source"), dict):
                src = dict(block["source"])
                if isinstance(src.get("data"), str) and len(src["data"]) > 40:
                    src["data"] = _truncate(src["data"])
                block["source"] = src
            # Anthropic tool_result with nested content list
            elif btype == "tool_result" and isinstance(block.get("content"), list):
                block["content"] = [_process_block(b) for b in block["content"]]
            # OpenAI image_url block
            elif btype == "image_url" and isinstance(block.get("image_url"), dict):
                iu = dict(block["image_url"])
                url = iu.get("url", "")
                if isinstance(url, str):
                    m = VisionAgent._DATA_URI_RE.match(url)
                    if m:
                        payload = url[m.end():]
                        iu["url"] = m.group(1) + _truncate(payload)
                block["image_url"] = iu
            return block

        result = []
        for msg in messages:
            msg = dict(msg)  # shallow copy
            content = msg.get("content")
            if isinstance(content, list):
                msg["content"] = [_process_block(b) if isinstance(b, dict) else b for b in content]
            result.append(msg)
        return result

    def _build_result(
        self,
        status: TaskStatus,
        reason: str,
        steps: int,
        in_tokens: int,
        out_tokens: int,
        start_time: float,
        step_log: list[dict[str, Any]],
        cache_read_tokens: int = 0,
        cache_creation_tokens: int = 0,
        completion_status: str = "success",
    ) -> TaskResult:
        """Construct a TaskResult with computed duration."""
        duration = time.monotonic() - start_time
        result = TaskResult(
            status=status,
            reason=reason,
            total_steps=steps,
            total_input_tokens=in_tokens,
            total_output_tokens=out_tokens,
            total_duration_s=round(duration, 1),
            total_cache_read_tokens=cache_read_tokens,
            total_cache_creation_tokens=cache_creation_tokens,
            step_log=step_log,
            completion_status=completion_status,
        )
        logger.info(
            "Task finished: status=%s, steps=%d, tokens=%d+%d, "
            "cache_read=%d, cache_write=%d, duration=%.1fs",
            status.value, steps, in_tokens, out_tokens,
            cache_read_tokens, cache_creation_tokens, duration,
        )
        return result
