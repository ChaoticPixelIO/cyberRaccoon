"""Anthropic native computer-use protocol.

Uses Anthropic's ``computer_20251124`` tool with the ``computer-use-2025-11-24``
beta to get structured computer-use actions from Claude models.
"""

from __future__ import annotations

import copy
import logging
import time
from typing import Any

from agent.prompts import build_anthropic_cu_system_prompt
from agent.protocols.base import ComputerUseProtocol, StepResult

logger = logging.getLogger("M3.anthropic_cu")

# Anthropic CU action -> executor action + field mapping
_ACTION_MAP: dict[str, str] = {
    "left_click": "click",
    "right_click": "click",
    "middle_click": "click",
    "double_click": "double_click",
    "triple_click": "triple_click",
    "mouse_move": "mouse_move",
    "left_click_drag": "drag",
    "left_mouse_down": "mouse_down",
    "left_mouse_up": "mouse_up",
    "type": "type",
    "key": "key",
    "hold_key": "hold_key",
    "scroll": "scroll",
    "wait": "wait",
    "screenshot": "_screenshot",  # internal: request re-capture
}

_BUTTON_MAP: dict[str, str] = {
    "left_click": "left",
    "right_click": "right",
    "middle_click": "middle",
}


class AnthropicCUProtocol(ComputerUseProtocol):
    """Anthropic native computer-use protocol.

    Uses the ``computer_20251124`` tool type with beta header to get
    structured tool_use responses from Claude.
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        history_max_turns: int = 10,
        display_width: int = 1280,
        display_height: int = 720,
        display_number: int = 1,
        enable_cache: bool = True,
        skill_text: str | None = None,
    ) -> None:
        import anthropic

        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._history_max_turns = history_max_turns
        self._display_width = display_width
        self._display_height = display_height
        self._enable_cache = enable_cache

        self._anthropic = anthropic  # for exception handling
        self._client = anthropic.Anthropic(api_key=api_key)

        # Tell the model the scaled dimensions so its coordinates match
        # what it sees in the (possibly downscaled) screenshot.
        scale = self._get_scale_factor()
        tool_w = int(display_width * scale) if scale < 1.0 else display_width
        tool_h = int(display_height * scale) if scale < 1.0 else display_height
        self._tool_def: dict[str, Any] = {
            "type": "computer_20251124",
            "name": "computer",
            "display_width_px": tool_w,
            "display_height_px": tool_h,
            "display_number": display_number,
        }
        self._system_prompt = build_anthropic_cu_system_prompt()
        if skill_text:
            self._system_prompt += "\n\n" + skill_text

        self._messages: list[dict[str, Any]] = []
        self._step_count = 0
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_cache_read_tokens = 0
        self._total_cache_creation_tokens = 0
        self._last_tool_use_id: str | None = None
        self._last_exec_error: str | None = None

    def step(self, screenshot_base64: str, task_goal: str) -> StepResult:
        start = time.monotonic()

        # Build message first — do NOT append to self._messages yet
        # to avoid corrupting conversation state if _call_api fails.
        user_message = self._build_next_message(screenshot_base64, task_goal)

        try:
            self._messages.append(user_message)
            self._trim_history()
            response = self._call_api()
        except self._anthropic.APIError as e:
            # Roll back the message we just appended
            self._messages.pop()
            latency_ms = int((time.monotonic() - start) * 1000)
            logger.error("Anthropic CU API call failed: %s", e)
            return StepResult(
                command=None,
                is_done=False,
                done_reason="",
                screen_summary="",
                raw_text="",
                input_tokens=0,
                output_tokens=0,
                latency_ms=latency_ms,
                success=False,
                error=str(e),
            )

        latency_ms = int((time.monotonic() - start) * 1000)
        in_tok = response.usage.input_tokens
        out_tok = response.usage.output_tokens
        cache_read = getattr(response.usage, "cache_read_input_tokens", 0) or 0
        cache_creation = getattr(response.usage, "cache_creation_input_tokens", 0) or 0
        self._total_input_tokens += in_tok
        self._total_output_tokens += out_tok
        self._total_cache_read_tokens += cache_read
        self._total_cache_creation_tokens += cache_creation

        # Parse response content blocks
        text_parts: list[str] = []
        tool_use_block = None

        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_use_block = block

        # Include tool_use action in raw_text so it's visible in the UI
        if tool_use_block is not None:
            import json
            text_parts.append(
                f"[tool_use] {json.dumps(tool_use_block.input, ensure_ascii=False)}"
            )

        raw_text = "\n".join(text_parts)
        screen_summary = raw_text[:200] if raw_text else ""

        # Convert SDK content blocks to plain dicts for JSON serialization
        content_dicts = [self._block_to_dict(b) for b in response.content]
        self._messages.append({"role": "assistant", "content": content_dicts})

        # No tool_use block -> model considers task done
        if tool_use_block is None:
            done_reason = raw_text or "Task completed"
            self._step_count += 1
            return StepResult(
                command=None,
                is_done=True,
                done_reason=done_reason,
                screen_summary=screen_summary,
                raw_text=raw_text,
                input_tokens=in_tok,
                output_tokens=out_tok,
                latency_ms=latency_ms,
                success=True,
                cache_read_tokens=cache_read,
                cache_creation_tokens=cache_creation,
            )

        # Normalize tool_use to executor command
        self._last_tool_use_id = tool_use_block.id
        self._step_count += 1
        action_input = tool_use_block.input
        cu_action = action_input.get("action", "")

        command = self._normalize_action(cu_action, action_input)

        # Scale coordinates back up, then validate against original display
        if command is not None:
            scale = self._get_scale_factor()
            if scale < 1.0:
                self._scale_coordinates_up(command, scale)

            oob_error = self._validate_coordinates(command)
            if oob_error:
                logger.warning("Step %d: %s", self._step_count, oob_error)
                return StepResult(
                    command=None,
                    is_done=False,
                    done_reason="",
                    screen_summary=screen_summary,
                    raw_text=raw_text,
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                    latency_ms=latency_ms,
                    success=False,
                    error=oob_error,
                    cache_read_tokens=cache_read,
                    cache_creation_tokens=cache_creation,
                )

        needs_screenshot = command is None and cu_action == "screenshot"

        if command is None and not needs_screenshot:
            logger.warning("Unknown Anthropic CU action: %s", cu_action)
            return StepResult(
                command=None,
                is_done=False,
                done_reason="",
                screen_summary=screen_summary,
                raw_text=raw_text,
                input_tokens=in_tok,
                output_tokens=out_tok,
                latency_ms=latency_ms,
                success=False,
                error=f"Unknown action: {cu_action}",
                cache_read_tokens=cache_read,
                cache_creation_tokens=cache_creation,
            )

        logger.info(
            "Anthropic CU: action=%s, latency=%dms, tokens=%d/%d, "
            "cache_read=%d, cache_write=%d",
            cu_action, latency_ms, in_tok, out_tok, cache_read, cache_creation,
        )

        return StepResult(
            command=command,
            is_done=False,
            done_reason="",
            screen_summary=screen_summary,
            raw_text=raw_text,
            input_tokens=in_tok,
            output_tokens=out_tok,
            latency_ms=latency_ms,
            success=True,
            needs_screenshot=needs_screenshot,
            cache_read_tokens=cache_read,
            cache_creation_tokens=cache_creation,
        )

    def report_result(self, success: bool, error: str | None = None) -> None:
        self._last_exec_error = error if not success else None

    def reset(self) -> None:
        self._messages.clear()
        self._step_count = 0
        self._last_tool_use_id = None
        self._last_exec_error = None

    def get_usage_summary(self) -> dict[str, int]:
        return {
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "total_cache_read_tokens": self._total_cache_read_tokens,
            "total_cache_creation_tokens": self._total_cache_creation_tokens,
        }

    def get_system_prompt(self) -> str:
        return self._system_prompt

    def get_messages_snapshot(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self._messages)

    def detect_os(self, screenshot_base64: str) -> str | None:
        """Detect target OS from screenshot via a one-off Anthropic API call."""
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=16,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": screenshot_base64,
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                "What operating system is shown in this screenshot? "
                                "Reply with exactly one word: windows, macos, or linux."
                            ),
                        },
                    ],
                }],
            )
            raw = response.content[0].text.strip().lower()
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

    def _build_next_message(
        self, screenshot_base64: str, task_goal: str,
    ) -> dict[str, Any]:
        """Build the next user message for the conversation.

        Returns the message dict without mutating ``self._messages``.
        The caller is responsible for appending after a successful API call.

        First call: user message with screenshot image + task goal.
        Subsequent calls: tool_result with screenshot.
        """
        scaled_screenshot = self._scale_screenshot(screenshot_base64)

        if self._step_count == 0:
            # First turn: user message with image + task
            return {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": scaled_screenshot,
                        },
                    },
                    {
                        "type": "text",
                        "text": f"Task: {task_goal}",
                    },
                ],
            }

        # Subsequent turns: tool_result with screenshot
        # Anthropic API requires is_error tool_results to contain only text
        # content blocks — no images allowed when is_error=True.
        if self._last_exec_error:
            result_content: list[dict[str, Any]] = [{
                "type": "text",
                "text": f"Action failed: {self._last_exec_error}",
            }]
            tool_result: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": self._last_tool_use_id,
                "content": result_content,
                "is_error": True,
            }
        else:
            result_content = [{
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": scaled_screenshot,
                },
            }]
            tool_result = {
                "type": "tool_result",
                "tool_use_id": self._last_tool_use_id,
                "content": result_content,
            }

        self._last_exec_error = None
        return {
            "role": "user",
            "content": [tool_result],
        }

    def _call_api(self) -> Any:
        """Call the Anthropic beta messages API with computer-use tool."""
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
            "system": self._system_prompt,
            "tools": [self._tool_def],
            "messages": self._messages,
            "betas": ["computer-use-2025-11-24"],
        }
        if self._enable_cache:
            kwargs["cache_control"] = {"type": "ephemeral"}
        return self._client.beta.messages.create(**kwargs)

    def _trim_history(self) -> None:
        """Trim conversation history to stay within budget.

        Each turn is a pair: user (tool_result) + assistant (tool_use).
        We keep the first message (initial task) and the most recent N pairs.
        """
        if len(self._messages) <= 1:
            return

        # First message is the initial task, rest are turn pairs
        # Each turn = user tool_result + assistant response = 2 messages
        max_messages = 1 + self._history_max_turns * 2
        if len(self._messages) > max_messages:
            # Keep first message + most recent turns
            self._messages[:] = (
                self._messages[:1] + self._messages[-(max_messages - 1):]
            )

    def _get_scale_factor(self) -> float:
        """Return the downscale factor for screenshots exceeding API limits.

        Returns 1.0 if no scaling is needed (resolution within limits).
        """
        max_dim = 1568
        max_pixels = 1_150_000
        w, h = self._display_width, self._display_height

        if w * h <= max_pixels and max(w, h) <= max_dim:
            return 1.0

        scale_dim = max_dim / max(w, h)
        scale_px = (max_pixels / (w * h)) ** 0.5
        return min(scale_dim, scale_px)

    def _scale_screenshot(self, screenshot_base64: str) -> str:
        """Downscale a screenshot if it exceeds Anthropic API resolution limits."""
        scale = self._get_scale_factor()
        if scale >= 1.0:
            return screenshot_base64

        import base64
        import io

        from PIL import Image

        raw = base64.b64decode(screenshot_base64)
        img = Image.open(io.BytesIO(raw))
        new_w = int(img.width * scale)
        new_h = int(img.height * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    @staticmethod
    def _scale_coordinates_up(
        command: dict[str, Any], scale: float,
    ) -> None:
        """Scale coordinates in a command back to the original resolution."""
        for key in ("x", "y", "from_x", "from_y", "to_x", "to_y"):
            if key in command:
                command[key] = int(command[key] / scale)

    def _validate_coordinates(self, command: dict[str, Any]) -> str | None:
        """Check that coordinates are within display bounds.

        Returns an error message string if invalid, None if ok.
        """
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

    @staticmethod
    def _normalize_action(
        cu_action: str, action_input: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Convert Anthropic CU tool input to executor command dict."""
        executor_action = _ACTION_MAP.get(cu_action)
        if executor_action is None:
            return None

        # screenshot is handled via needs_screenshot flag
        if executor_action == "_screenshot":
            return None

        cmd: dict[str, Any] = {"action": executor_action}

        # Coordinate extraction
        coord = action_input.get("coordinate")
        start_coord = action_input.get("start_coordinate")

        # Helper: extract optional modifier key from "text" field on
        # click/scroll actions (e.g. shift+click, ctrl+scroll)
        modifier = action_input.get("text") if cu_action not in (
            "type", "key", "hold_key",
        ) else None

        if cu_action in ("left_click", "right_click", "middle_click"):
            cmd["button"] = _BUTTON_MAP[cu_action]
            if not coord or len(coord) < 2:
                logger.warning("%s missing coordinate: %s", cu_action, action_input)
                return None
            cmd["x"], cmd["y"] = int(coord[0]), int(coord[1])
            if modifier:
                cmd["modifier"] = modifier

        elif cu_action in (
            "double_click", "triple_click", "mouse_move",
            "left_mouse_down", "left_mouse_up",
        ):
            if not coord or len(coord) < 2:
                logger.warning("%s missing coordinate: %s", cu_action, action_input)
                return None
            cmd["x"], cmd["y"] = int(coord[0]), int(coord[1])
            if modifier:
                cmd["modifier"] = modifier

        elif cu_action == "left_click_drag":
            if not start_coord or len(start_coord) < 2:
                logger.warning("left_click_drag missing start_coordinate")
                return None
            if not coord or len(coord) < 2:
                logger.warning("left_click_drag missing coordinate (end)")
                return None
            cmd["from_x"], cmd["from_y"] = (
                int(start_coord[0]), int(start_coord[1]),
            )
            cmd["to_x"], cmd["to_y"] = int(coord[0]), int(coord[1])

        elif cu_action == "type":
            cmd["text"] = action_input.get("text", "")

        elif cu_action == "key":
            # Anthropic sends key combos as "ctrl+c", "Return", etc.
            key_text = action_input.get("text", "")
            cmd["keys"] = [k.strip().lower() for k in key_text.split("+")]

        elif cu_action == "hold_key":
            key_text = action_input.get("text", "")
            cmd["keys"] = [k.strip().lower() for k in key_text.split("+")]
            cmd["duration_s"] = float(action_input.get("duration", 1.0))

        elif cu_action == "scroll":
            if coord:
                cmd["x"], cmd["y"] = int(coord[0]), int(coord[1])
            direction = action_input.get("scroll_direction", "down")
            cmd["direction"] = direction
            cmd["amount"] = int(action_input.get("scroll_amount", 3))
            if modifier:
                cmd["modifier"] = modifier

        elif cu_action == "wait":
            cmd["duration_s"] = float(action_input.get("duration", 1.0))

        return cmd

    @staticmethod
    def _block_to_dict(block: Any) -> dict[str, Any]:
        """Convert an Anthropic SDK content block to a plain dict.

        SDK objects (TextBlock, ToolUseBlock, etc.) aren't JSON-serializable.
        This ensures prompt_messages can be sent over WebSocket to the UI.
        """
        if isinstance(block, dict):
            return block
        if hasattr(block, "model_dump"):
            return block.model_dump()
        # Fallback: manual extraction for known block types
        if hasattr(block, "type"):
            d: dict[str, Any] = {"type": block.type}
            if block.type == "text":
                d["text"] = getattr(block, "text", "")
            elif block.type == "tool_use":
                d["id"] = getattr(block, "id", "")
                d["name"] = getattr(block, "name", "")
                d["input"] = getattr(block, "input", {})
            return d
        return {"type": "unknown", "repr": repr(block)}
