"""Prompt-based computer-use protocol (fallback).

For models without native computer-use support. Describes the action schema
in the system prompt and parses JSON from the LLM's text response.
"""

from __future__ import annotations

import copy
import logging
import time
from typing import Any

from agent.prompts import build_prompt_based_system_prompt
from agent.protocols.anthropic_cu import AnthropicCUProtocol
from agent.protocols.base import ComputerUseProtocol, StepResult
from agent.protocols.parsing import parse_json_actions

logger = logging.getLogger("M3.prompt_based")

# Actions the prompt-based protocol recognizes in LLM JSON output.
# Uses the same vocabulary as Anthropic CU for unified normalization.
VALID_ACTIONS: set[str] = {
    "left_click",
    "right_click",
    "middle_click",
    "double_click",
    "triple_click",
    "mouse_move",
    "left_click_drag",
    "left_mouse_down",
    "left_mouse_up",
    "type",
    "key",
    "hold_key",
    "scroll",
    "wait",
    "screenshot",
    "done",
}


class PromptBasedProtocol(ComputerUseProtocol):
    """Prompt-based fallback for models without native computer-use.

    Sends a system prompt describing the Anthropic CU action vocabulary.
    Parses JSON from the LLM's text response with 4-level fallback.
    Normalizes actions to executor format using the same mapping as
    :class:`AnthropicCUProtocol`.
    """

    def __init__(
        self,
        provider: str,
        model: str,
        api_key: str,
        base_url: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        history_max_turns: int = 10,
        display_width: int = 1280,
        display_height: int = 720,
        enable_cache: bool = True,
        skill_text: str | None = None,
    ) -> None:
        self._provider = provider.lower()
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._history_max_turns = history_max_turns
        self._enable_cache = enable_cache
        self._system_prompt = build_prompt_based_system_prompt(
            display_width, display_height,
        )
        if skill_text:
            self._system_prompt += "\n\n## Application Skill\n\n" + skill_text

        # Initialize the appropriate SDK client
        if self._provider == "anthropic":
            import anthropic
            self._anthropic_client = anthropic.Anthropic(api_key=api_key)
        else:
            import openai
            kwargs: dict[str, Any] = {"api_key": api_key}
            if base_url:
                kwargs["base_url"] = base_url
            self._openai_client = openai.OpenAI(**kwargs)

        self._messages: list[dict[str, Any]] = []
        self._step_count = 0
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_cache_read_tokens = 0
        self._total_cache_creation_tokens = 0
        self._last_exec_error: str | None = None

    def step(self, screenshot_base64: str, task_goal: str) -> StepResult:
        start = time.monotonic()

        try:
            if self._provider == "anthropic":
                raw_text, in_tok, out_tok, cache_read, cache_creation = (
                    self._call_anthropic(screenshot_base64, task_goal)
                )
            else:
                raw_text, in_tok, out_tok, cache_read, cache_creation = (
                    self._call_openai(screenshot_base64, task_goal)
                )
        except Exception as e:
            latency_ms = int((time.monotonic() - start) * 1000)
            logger.error(
                "Prompt-based API call failed (%s): %s",
                type(e).__name__, e,
            )
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
                error=f"{type(e).__name__}: {e}",
            )

        latency_ms = int((time.monotonic() - start) * 1000)
        self._total_input_tokens += in_tok
        self._total_output_tokens += out_tok
        self._total_cache_read_tokens += cache_read
        self._total_cache_creation_tokens += cache_creation

        # Parse commands from raw text (supports JSON array)
        parsed_list = self._parse_commands(raw_text)

        if parsed_list is None:
            logger.warning("Failed to parse command from: %.200s", raw_text)
            self._step_count += 1
            return StepResult(
                command=None,
                is_done=False,
                done_reason="",
                screen_summary="",
                raw_text=raw_text,
                input_tokens=in_tok,
                output_tokens=out_tok,
                latency_ms=latency_ms,
                success=False,
                error="Failed to parse valid command from LLM response",
                cache_read_tokens=cache_read,
                cache_creation_tokens=cache_creation,
            )

        # Use screen_summary from the last action in the array
        screen_summary = parsed_list[-1].get("screen_summary", "")

        # Update conversation history
        self._update_history(screen_summary, raw_text)
        self._step_count += 1

        # Handle done: if last action is done, set is_done and return
        # any preceding commands for execution first
        is_done = False
        done_reason = ""
        completion_status = "success"
        action_list = parsed_list
        if parsed_list[-1].get("action") == "done":
            is_done = True
            done_reason = parsed_list[-1].get("reason", "Task completed")
            completion_status = parsed_list[-1].get("status", "success")
            action_list = parsed_list[:-1]  # commands before done

        # If only action was done (no preceding commands)
        if is_done and not action_list:
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
                completion_status=completion_status,
            )

        # Normalize all actions to executor format
        commands: list[dict[str, Any]] = []
        needs_screenshot = False

        for parsed in action_list:
            cu_action = parsed.get("action", "")

            if cu_action == "screenshot":
                needs_screenshot = True
                continue

            command = AnthropicCUProtocol._normalize_action(cu_action, parsed)

            if command is None:
                logger.warning("Unknown prompt-based action: %s", cu_action)
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

            commands.append(command)

        logger.info(
            "Prompt-based: %d action(s), latency=%dms, tokens=%d/%d, "
            "cache_read=%d, cache_write=%d",
            len(commands), latency_ms, in_tok, out_tok, cache_read, cache_creation,
        )

        return StepResult(
            command=commands[0] if commands else None,
            is_done=is_done,
            done_reason=done_reason,
            screen_summary=screen_summary,
            raw_text=raw_text,
            input_tokens=in_tok,
            output_tokens=out_tok,
            latency_ms=latency_ms,
            success=True,
            needs_screenshot=needs_screenshot and not commands,
            commands=commands,
            cache_read_tokens=cache_read,
            cache_creation_tokens=cache_creation,
            completion_status=completion_status,
        )

    def report_result(self, success: bool, error: str | None = None) -> None:
        self._last_exec_error = error if not success else None

    def reset(self) -> None:
        self._messages.clear()
        self._step_count = 0
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
        """Detect target OS from screenshot via a one-off LLM call."""
        prompt_text = (
            "What operating system is shown in this screenshot? "
            "Reply with exactly one word: windows, macos, or linux."
        )
        try:
            if self._provider == "anthropic":
                response = self._anthropic_client.messages.create(
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
                            {"type": "text", "text": prompt_text},
                        ],
                    }],
                )
                raw = response.content[0].text.strip().lower()
            else:
                response = self._openai_client.chat.completions.create(
                    model=self._model,
                    max_completion_tokens=16,
                    messages=[{
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{screenshot_base64}",
                                },
                            },
                            {"type": "text", "text": prompt_text},
                        ],
                    }],
                )
                raw = response.choices[0].message.content.strip().lower()

            if raw in ("windows", "macos", "linux"):
                return raw
            logger.warning("OS detection returned unexpected value: %s", raw)
            return None
        except Exception as e:
            logger.warning("OS detection failed: %s", e)
            return None

    # ------------------------------------------------------------------
    # API calls
    # ------------------------------------------------------------------

    def _call_anthropic(
        self, screenshot_base64: str, task_goal: str,
    ) -> tuple[str, int, int, int, int]:
        current_user_content: list[dict[str, Any]] = []

        if self._last_exec_error:
            current_user_content.append({
                "type": "text",
                "text": f"Previous action failed: {self._last_exec_error}",
            })
            self._last_exec_error = None

        current_user_content.extend([
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
                "text": f"Task: {task_goal}",
            },
        ])

        messages: list[dict[str, Any]] = list(self._messages)
        messages.append({"role": "user", "content": current_user_content})

        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
            "system": self._system_prompt,
            "messages": messages,
        }
        if self._enable_cache:
            kwargs["cache_control"] = {"type": "ephemeral"}
        response = self._anthropic_client.messages.create(**kwargs)

        if not response.content:
            raise ValueError(
                f"Anthropic API returned empty content for model {self._model}"
            )
        raw_text = response.content[0].text
        cache_read = getattr(response.usage, "cache_read_input_tokens", 0) or 0
        cache_creation = getattr(response.usage, "cache_creation_input_tokens", 0) or 0
        return (
            raw_text, response.usage.input_tokens, response.usage.output_tokens,
            cache_read, cache_creation,
        )

    def _call_openai(
        self, screenshot_base64: str, task_goal: str,
    ) -> tuple[str, int, int, int, int]:
        current_user_content: list[dict[str, Any]] = []

        if self._last_exec_error:
            current_user_content.append({
                "type": "text",
                "text": f"Previous action failed: {self._last_exec_error}",
            })
            self._last_exec_error = None

        current_user_content.extend([
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{screenshot_base64}",
                },
            },
            {
                "type": "text",
                "text": f"Task: {task_goal}",
            },
        ])

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt},
        ]
        messages.extend(self._messages)
        messages.append({"role": "user", "content": current_user_content})

        response = self._openai_client.chat.completions.create(
            model=self._model,
            max_completion_tokens=self._max_tokens,
            temperature=self._temperature,
            messages=messages,
        )

        if not response.choices or not response.choices[0].message.content:
            raise ValueError(
                f"OpenAI API returned empty response for model {self._model}"
            )
        raw_text = response.choices[0].message.content
        usage = response.usage
        return (
            raw_text,
            usage.prompt_tokens if usage else 0,
            usage.completion_tokens if usage else 0,
            0, 0,
        )

    # ------------------------------------------------------------------
    # History management
    # ------------------------------------------------------------------

    def _update_history(self, screen_summary: str, raw_text: str) -> None:
        """Append turn to conversation history and trim."""
        if screen_summary:
            self._messages.append({
                "role": "user",
                "content": f"Screen state: {screen_summary}",
            })
        self._messages.append({
            "role": "assistant",
            "content": raw_text,
        })

        # Trim: each turn = user + assistant = 2 messages
        max_messages = self._history_max_turns * 2
        if len(self._messages) > max_messages:
            self._messages[:] = self._messages[-max_messages:]

    # ------------------------------------------------------------------
    # JSON parsing (4-level fallback via shared utility)
    # ------------------------------------------------------------------

    def _parse_commands(self, raw_text: str) -> list[dict[str, Any]] | None:
        """Extract valid command dicts from LLM response text.

        Supports both a single JSON object and a JSON array of objects.
        """
        return parse_json_actions(raw_text, VALID_ACTIONS)
