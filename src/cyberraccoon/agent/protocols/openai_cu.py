"""OpenAI native computer-use protocol.

Uses OpenAI's Responses API with the ``computer`` tool to get structured
computer-use actions from GPT-5.5.
"""

from __future__ import annotations

import collections
import json
import logging
import time
from typing import Any

from cyberraccoon.agent.prompts import build_openai_cu_system_prompt
from cyberraccoon.agent.protocols.base import ComputerUseProtocol, StepResult
from cyberraccoon.agent.protocols.parsing import extract_completion_status

logger = logging.getLogger("M3.openai_cu")

# Key name mapping: OpenAI keypress names → executor key names.
# Process: lowercase the key, then look up; unmapped keys pass through as-is.
_KEY_MAP: dict[str, str] = {
    "control": "ctrl",
    "space": "space",
    "arrowdown": "down",
    "arrowup": "up",
    "arrowleft": "left",
    "arrowright": "right",
    "escape": "escape",
    "enter": "enter",
    "backspace": "backspace",
    "tab": "tab",
    "delete": "delete",
    "shift": "shift",
    "alt": "alt",
    "meta": "meta",
}


def _normalize_key(key: str) -> str:
    """Normalize an OpenAI keypress key name to executor format."""
    lowered = key.lower()
    return _KEY_MAP.get(lowered, lowered)


def _g(obj: Any, key: str, default: Any = None) -> Any:
    """Get a field from either a dict or an SDK object."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


class OpenAICUProtocol(ComputerUseProtocol):
    """OpenAI native computer-use protocol.

    Uses the Responses API with ``{"type": "computer"}`` tool to get
    structured ``computer_call`` responses from GPT-5.5.

    OpenAI may return multiple actions per ``computer_call``.  Since
    VisionAgent expects one command per ``step()``, actions are queued
    internally and popped one at a time.
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        *,
        display_width: int = 1280,
        display_height: int = 720,
        skill_text: str | None = None,
    ) -> None:
        import openai

        self._model = model
        self._display_width = display_width
        self._display_height = display_height

        self._openai = openai  # for exception handling
        self._client = openai.OpenAI(api_key=api_key, timeout=30.0)

        self._system_prompt = build_openai_cu_system_prompt()
        if skill_text:
            self._system_prompt += "\n\n## Application Skill\n\n" + skill_text

        # Conversation state (server-side stateful, with client-side mirror)
        self._last_response_id: str | None = None
        self._last_call_id: str | None = None
        self._pending_safety_checks: list[Any] = []

        # Action queue for batched responses
        self._action_queue: collections.deque[Any] = collections.deque()

        # Error from executor (cleared on next successful report)
        self._last_exec_error: str | None = None

        # Client-side message tracking for UI display.
        # The Responses API is server-side stateful, but we mirror
        # messages here so get_messages_snapshot() returns useful data.
        self._messages: list[dict[str, Any]] = []

        # Metrics
        self._step_count = 0
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_cache_read_tokens = 0

    # ------------------------------------------------------------------
    # ComputerUseProtocol interface
    # ------------------------------------------------------------------

    def step(self, screenshot_base64: str, task_goal: str) -> StepResult:
        # Queued actions from a previous batched response — no API call
        if self._action_queue:
            return self._pop_queued_action()

        start = time.monotonic()
        try:
            if self._step_count == 0:
                response = self._call_api_initial(screenshot_base64, task_goal)
            else:
                response = self._call_api_continuation(screenshot_base64)
        except self._openai.APIError as e:
            latency_ms = int((time.monotonic() - start) * 1000)
            logger.error("OpenAI CU API call failed: %s", e)
            return StepResult(
                command=None, is_done=False, done_reason="",
                screen_summary="", raw_text="",
                input_tokens=0, output_tokens=0,
                latency_ms=latency_ms, success=False, error=str(e),
            )

        latency_ms = int((time.monotonic() - start) * 1000)
        in_tok = response.usage.input_tokens if response.usage else 0
        out_tok = response.usage.output_tokens if response.usage else 0
        self._total_input_tokens += in_tok
        self._total_output_tokens += out_tok

        # OpenAI automatic caching — extract cached token count
        cache_read = 0
        if response.usage:
            details = getattr(response.usage, "input_tokens_details", None)
            if details:
                cache_read = getattr(details, "cached_tokens", 0) or 0
        self._total_cache_read_tokens += cache_read

        # Guard against None output (empty list is valid — means task done)
        if response.output is None:
            self._step_count += 1
            return StepResult(
                command=None, is_done=False, done_reason="",
                screen_summary="", raw_text="API returned no output",
                input_tokens=in_tok, output_tokens=out_tok,
                latency_ms=latency_ms, success=False,
                error="OpenAI API returned no output items",
                cache_read_tokens=cache_read,
            )

        # Collect text from response output for raw_text
        raw_text_parts: list[str] = []
        for item in response.output:
            if getattr(item, "type", None) == "text":
                raw_text_parts.append(getattr(item, "text", ""))

        # Find computer_call in response output
        computer_call = next(
            (item for item in response.output if item.type == "computer_call"),
            None,
        )

        # No computer_call → task complete
        if computer_call is None:
            done_text = (
                getattr(response, "output_text", "") or "Task completed"
            )
            completion_status = extract_completion_status(done_text)
            self._messages.append({
                "role": "assistant",
                "content": done_text[:500],
            })
            self._step_count += 1
            return StepResult(
                command=None, is_done=True, done_reason=done_text,
                screen_summary=done_text[:200], raw_text=done_text,
                input_tokens=in_tok, output_tokens=out_tok,
                latency_ms=latency_ms, success=True,
                cache_read_tokens=cache_read,
                completion_status=completion_status,
            )

        # Store state for continuation
        self._last_response_id = response.id
        self._last_call_id = computer_call.call_id
        if (
            hasattr(computer_call, "pending_safety_checks")
            and computer_call.pending_safety_checks
        ):
            self._pending_safety_checks = list(
                computer_call.pending_safety_checks,
            )

        # Extract actions: prefer batched list, fallback to single
        actions = self._extract_actions(computer_call)
        if not actions:
            self._step_count += 1
            # WR-03: stamp response_id — _last_response_id was updated
            # at line 193 so this failure row belongs to the same
            # response as the success path.
            return StepResult(
                command=None, is_done=False, done_reason="",
                screen_summary="", raw_text="No actions in computer_call",
                input_tokens=in_tok, output_tokens=out_tok,
                latency_ms=latency_ms, success=False,
                error="computer_call contained no actions",
                cache_read_tokens=cache_read,
                response_id=self._last_response_id,
            )

        # Include full action details in raw_text
        action_types = [_g(a, "type", "?") for a in actions]
        raw_text_parts.append(f"[computer_call] actions={action_types}")
        raw_text_parts.append(json.dumps(actions, indent=2, default=str))
        raw_text = "\n".join(raw_text_parts)
        screen_summary = raw_text[:200] if raw_text else ""

        # Track assistant response for get_messages_snapshot()
        self._messages.append({
            "role": "assistant",
            "content": (
                f"[response_id={response.id}] "
                + (raw_text[:500] if raw_text else "(no text)")
            ),
        })

        # Queue remaining actions, process first
        for action in actions[1:]:
            self._action_queue.append(action)

        first_action = actions[0]
        action_type = _g(first_action, "type", "unknown")
        logger.debug(
            "OpenAI CU raw action: type=%s, repr=%r",
            action_type, first_action,
        )
        command = self._normalize_action(first_action)
        self._step_count += 1

        needs_screenshot = (
            command is None and action_type == "screenshot"
        )

        # Failed normalization → fail so VisionAgent retries
        if command is None and not needs_screenshot:
            self._action_queue.clear()
            if action_type == "drag":
                error_msg = (
                    f"drag action requires at least 2 path points "
                    f"(raw: {first_action!r})"
                )
            else:
                error_msg = (
                    f"Unknown or unhandled action type: {action_type} "
                    f"(raw: {first_action!r})"
                )
            logger.warning("Step %d: %s", self._step_count, error_msg)
            # WR-03: stamp response_id so this failure row is grouped
            # with any sibling actions from the same LLM response.
            return StepResult(
                command=None, is_done=False, done_reason="",
                screen_summary=screen_summary, raw_text=raw_text,
                input_tokens=in_tok, output_tokens=out_tok,
                latency_ms=latency_ms, success=False, error=error_msg,
                cache_read_tokens=cache_read,
                response_id=self._last_response_id,
            )

        if command is not None:
            oob_error = self._validate_coordinates(command)
            if oob_error:
                self._action_queue.clear()
                logger.warning("Step %d: %s", self._step_count, oob_error)
                # WR-03: stamp response_id so the OOB failure row is
                # grouped with any sibling actions from the same LLM
                # response (consistent with the queued-pop OOB path).
                return StepResult(
                    command=None, is_done=False, done_reason="",
                    screen_summary=screen_summary, raw_text=raw_text,
                    input_tokens=in_tok, output_tokens=out_tok,
                    latency_ms=latency_ms, success=False, error=oob_error,
                    cache_read_tokens=cache_read,
                    response_id=self._last_response_id,
                )

        logger.info(
            "OpenAI CU: action=%s, latency=%dms, tokens=%d/%d, "
            "cache_read=%d",
            action_type, latency_ms, in_tok, out_tok, cache_read,
        )

        return StepResult(
            command=command, is_done=False, done_reason="",
            screen_summary=screen_summary, raw_text=raw_text,
            input_tokens=in_tok, output_tokens=out_tok,
            latency_ms=latency_ms, success=True,
            needs_screenshot=needs_screenshot,
            cache_read_tokens=cache_read,
            response_id=self._last_response_id,
        )

    def report_result(self, success: bool, error: str | None = None) -> None:
        if not success:
            self._action_queue.clear()
            self._last_exec_error = error
        else:
            self._last_exec_error = None

    def reset(self) -> None:
        self._action_queue.clear()
        self._last_response_id = None
        self._last_call_id = None
        self._pending_safety_checks.clear()
        self._last_exec_error = None
        self._step_count = 0
        self._messages.clear()

    def get_usage_summary(self) -> dict[str, int]:
        return {
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "total_cache_read_tokens": self._total_cache_read_tokens,
            "total_cache_creation_tokens": 0,
        }

    def get_system_prompt(self) -> str:
        return self._system_prompt

    def get_messages_snapshot(self) -> list[dict[str, Any]]:
        import copy
        return copy.deepcopy(self._messages)

    def detect_os(self, screenshot_base64: str) -> str | None:
        """Detect target OS from screenshot via a one-off Responses API call."""
        try:
            response = self._client.responses.create(
                model=self._model,
                input=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "input_image",
                            "image_url": (
                                f"data:image/jpeg;base64,{screenshot_base64}"
                            ),
                        },
                        {
                            "type": "input_text",
                            "text": (
                                "What operating system is shown in this "
                                "screenshot? Reply with exactly one word: "
                                "windows, macos, or linux."
                            ),
                        },
                    ],
                }],
            )
            raw = response.output_text.strip().lower()
            if raw in ("windows", "macos", "linux"):
                return raw
            logger.warning("OS detection returned unexpected value: %s", raw)
            return None
        except Exception as e:
            logger.warning("OS detection failed: %s", e)
            return None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _pop_queued_action(self) -> StepResult:
        """Pop and normalize the next queued action (no API call)."""
        action = self._action_queue.popleft()
        action_type = _g(action, "type", "unknown")
        command = self._normalize_action(action)
        self._step_count += 1

        raw_text = (
            f"[queued] {action_type}\n"
            + json.dumps(action, indent=2, default=str)
        )

        needs_screenshot = (
            command is None and action_type == "screenshot"
        )

        if command is None and not needs_screenshot:
            self._action_queue.clear()
            if action_type == "drag":
                error_msg = (
                    f"drag action requires at least 2 path points "
                    f"(raw: {action!r})"
                )
            else:
                error_msg = (
                    f"Unknown or unhandled action type: {action_type} "
                    f"(raw: {action!r})"
                )
            return StepResult(
                command=None, is_done=False, done_reason="",
                screen_summary="", raw_text=raw_text,
                input_tokens=0, output_tokens=0,
                latency_ms=0, success=False, error=error_msg,
                response_id=self._last_response_id,
            )

        if command is not None:
            oob_error = self._validate_coordinates(command)
            if oob_error:
                self._action_queue.clear()
                return StepResult(
                    command=None, is_done=False, done_reason="",
                    screen_summary="", raw_text=raw_text,
                    input_tokens=0, output_tokens=0,
                    latency_ms=0, success=False, error=oob_error,
                    response_id=self._last_response_id,
                )

        return StepResult(
            command=command, is_done=False, done_reason="",
            screen_summary="", raw_text=raw_text,
            input_tokens=0, output_tokens=0,
            latency_ms=0, success=True,
            needs_screenshot=needs_screenshot,
            response_id=self._last_response_id,
        )

    def _call_api_initial(
        self, screenshot_base64: str, task_goal: str,
    ) -> Any:
        """First API call: send screenshot + task goal with instructions."""
        # Track for get_messages_snapshot()
        self._messages.append({
            "role": "user",
            "content": f"[screenshot] Task: {task_goal}",
        })
        self._messages.append({
            "role": "system",
            "content": f"[API] model={self._model}, previous_response_id=None (initial)",
        })
        return self._client.responses.create(
            model=self._model,
            instructions=self._system_prompt,
            tools=[{"type": "computer"}],
            input=[{
                "role": "user",
                "content": [
                    {
                        "type": "input_image",
                        "image_url": (
                            f"data:image/jpeg;base64,{screenshot_base64}"
                        ),
                    },
                    {
                        "type": "input_text",
                        "text": f"Task: {task_goal}",
                    },
                ],
            }],
            truncation="auto",
        )

    def _call_api_continuation(self, screenshot_base64: str) -> Any:
        """Continuation API call: send screenshot as computer_call_output."""
        self._messages.append({
            "role": "user",
            "content": "[screenshot after action]",
        })
        self._messages.append({
            "role": "system",
            "content": (
                f"[API] model={self._model}, "
                f"previous_response_id={self._last_response_id}"
            ),
        })
        call_output: dict[str, Any] = {
            "type": "computer_call_output",
            "call_id": self._last_call_id,
            "output": {
                "type": "computer_screenshot",
                "image_url": (
                    f"data:image/jpeg;base64,{screenshot_base64}"
                ),
            },
        }

        # Auto-acknowledge any pending safety checks
        if self._pending_safety_checks:
            call_output["acknowledged_safety_checks"] = [
                {
                    "id": c.id,
                    "code": getattr(c, "code", None),
                    "message": getattr(c, "message", None),
                }
                for c in self._pending_safety_checks
            ]
            self._pending_safety_checks = []

        input_items: list[Any] = [call_output]

        # Include execution error so the model knows the action failed
        if self._last_exec_error:
            input_items.append({
                "role": "user",
                "content": f"ERROR: {self._last_exec_error}",
            })
            self._last_exec_error = None

        return self._client.responses.create(
            model=self._model,
            previous_response_id=self._last_response_id,
            tools=[{"type": "computer"}],
            input=input_items,
            truncation="auto",
        )

    @staticmethod
    def _extract_actions(computer_call: Any) -> list[Any]:
        """Extract action list from a computer_call.

        Prefers the batched ``actions`` list; falls back to wrapping a
        single ``action`` in a list for backward compatibility.
        """
        if hasattr(computer_call, "actions") and computer_call.actions:
            return list(computer_call.actions)
        if hasattr(computer_call, "action") and computer_call.action:
            return [computer_call.action]
        return []

    def _normalize_action(self, action: Any) -> dict[str, Any] | None:
        """Convert an OpenAI action (dict or SDK object) to executor command."""
        action_type = _g(action, "type")
        if action_type is None:
            return None

        if action_type == "click":
            x_val, y_val = _g(action, "x"), _g(action, "y")
            if x_val is None or y_val is None:
                logger.warning("click missing coordinates: %r", action)
                return None
            button = _g(action, "button", "left") or "left"
            if button == "wheel":
                button = "middle"
            return {
                "action": "click",
                "x": int(x_val), "y": int(y_val),
                "button": button,
            }

        if action_type == "double_click":
            x_val, y_val = _g(action, "x"), _g(action, "y")
            if x_val is None or y_val is None:
                logger.warning("double_click missing coordinates: %r", action)
                return None
            return {
                "action": "double_click",
                "x": int(x_val), "y": int(y_val),
            }

        if action_type == "scroll":
            x_val, y_val = _g(action, "x"), _g(action, "y")
            if x_val is None or y_val is None:
                logger.warning("scroll missing coordinates: %r", action)
                return None
            x, y = int(x_val), int(y_val)
            # API may return snake_case or camelCase — use explicit None
            # checks to avoid falsy 0 falling through the or-chain.
            scroll_x = _g(action, "scroll_x")
            if scroll_x is None:
                scroll_x = _g(action, "scrollX")
            if scroll_x is None:
                scroll_x = 0
            scroll_y = _g(action, "scroll_y")
            if scroll_y is None:
                scroll_y = _g(action, "scrollY")
            if scroll_y is None:
                scroll_y = 0
            logger.debug("scroll raw: %r → scroll_x=%s, scroll_y=%s",
                        action, scroll_x, scroll_y)
            # Executor only supports vertical scroll; map horizontal to
            # vertical down with a warning.
            if abs(scroll_y) >= abs(scroll_x):
                direction = "down" if scroll_y >= 0 else "up"
                # OpenAI returns pixel-based scroll values (e.g. 467);
                # executor expects HID notch count.  No universal
                # px-to-notch ratio exists — use display height / 6
                # so a full-screen scroll ≈ 6 notches, capped at 10.
                raw = abs(scroll_y)
                px_per_notch = self._display_height / 6
                amount = (
                    min(10, max(1, round(raw / px_per_notch)))
                    if raw != 0 else 3
                )
            else:
                logger.warning(
                    "Horizontal scroll requested (scroll_x=%s) but not "
                    "supported by executor; falling back to vertical "
                    "scroll down",
                    scroll_x,
                )
                direction = "down"
                amount = 3
            return {
                "action": "scroll", "x": x, "y": y,
                "direction": direction, "amount": amount,
            }

        if action_type == "type":
            return {"action": "type", "text": _g(action, "text", "")}

        if action_type == "keypress":
            keys = _g(action, "keys", []) or []
            normalized = [_normalize_key(k) for k in keys]
            return {"action": "key", "keys": normalized}

        if action_type == "wait":
            # GPT-5.5 sends duration in ms via "ms" or "duration_ms" field
            ms = _g(action, "ms")
            if ms is None:
                ms = _g(action, "duration_ms")
            if ms is None:
                ms = 1000
            try:
                duration_s = max(0.0, float(ms) / 1000.0)
            except (ValueError, TypeError):
                logger.warning("Invalid wait duration: %r, defaulting to 1s", ms)
                duration_s = 1.0
            return {"action": "wait", "duration_s": duration_s}

        if action_type == "screenshot":
            return None  # handled via needs_screenshot flag

        if action_type == "drag":
            path = _g(action, "path", []) or []
            if len(path) < 2:
                logger.warning("drag action has fewer than 2 path points: %r",
                               path)
                return None
            start, end = path[0], path[-1]
            sx, sy = _g(start, "x"), _g(start, "y")
            ex, ey = _g(end, "x"), _g(end, "y")
            if any(v is None for v in (sx, sy, ex, ey)):
                logger.warning("drag path missing coordinates: %r", action)
                return None
            return {
                "action": "drag",
                "from_x": int(sx), "from_y": int(sy),
                "to_x": int(ex), "to_y": int(ey),
            }

        if action_type == "move":
            x_val, y_val = _g(action, "x"), _g(action, "y")
            if x_val is None or y_val is None:
                logger.warning("move missing coordinates: %r", action)
                return None
            return {
                "action": "mouse_move",
                "x": int(x_val), "y": int(y_val),
            }

        logger.warning("Unknown OpenAI CU action type: %s", action_type)
        return None

    def _validate_coordinates(self, command: dict[str, Any]) -> str | None:
        """Check that coordinates are within display bounds."""
        w, h = self._display_width, self._display_height
        for xk, yk in [("x", "y"), ("from_x", "from_y"), ("to_x", "to_y")]:
            if xk in command and yk in command:
                x, y = command[xk], command[yk]
                if x < 0 or x >= w or y < 0 or y >= h:
                    return (
                        f"Coordinates ({x}, {y}) out of bounds "
                        f"for {w}x{h} display"
                    )
        return None
