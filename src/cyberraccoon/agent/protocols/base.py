"""Computer-use protocol abstraction layer.

Defines the ABC that all computer-use protocols implement, plus a factory
function that selects the right protocol based on provider and model.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("M3.protocol")

# Models supporting the computer_20251124 tool (Anthropic native CU).
# Prefix matching: "claude-opus-4-7" matches "claude-opus-4-7-20260415", etc.
# Opus 4.7 (latest), Opus 4.6, Opus 4.5, and Sonnet 4.6 support
# computer_20251124. Older models (Sonnet 4.5, Sonnet 4, Opus 4, Opus 4.1,
# etc.) only support computer_20250124 and fall back to prompt-based protocol.
ANTHROPIC_CU_MODEL_PREFIXES: list[str] = [
    "claude-opus-4-7",
    "claude-opus-4-6",
    "claude-opus-4-5",
    "claude-sonnet-4-6",
]

# Models supporting OpenAI's native computer-use via the Responses API.
OPENAI_CU_MODEL_PREFIXES: list[str] = [
    "gpt-5.5",
    "gpt-5.4",
]


@dataclass
class StepResult:
    """Normalized output of a single protocol step."""

    command: dict[str, Any] | None  # Normalized command for executor
    is_done: bool                   # Model considers task complete
    done_reason: str                # Reason text if is_done
    screen_summary: str             # For logging / UI display
    raw_text: str                   # Model's text output
    input_tokens: int
    output_tokens: int
    latency_ms: int
    success: bool
    error: str | None = None
    needs_screenshot: bool = False  # Model requested a fresh screenshot
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    commands: list[dict[str, Any]] = field(default_factory=list)
    completion_status: str = "success"  # "success", "gave_up", "stuck", or "escalate"
    response_id: str | None = None  # Correlates queued-action steps from one LLM response (UAT gap 5)

    def get_commands(self) -> list[dict[str, Any]]:
        """Return the list of commands to execute.

        Returns ``commands`` if non-empty, otherwise wraps ``command``
        in a single-element list (backward compatibility).  Returns an
        empty list when there is nothing to execute.
        """
        if self.commands:
            return self.commands
        if self.command is not None:
            return [self.command]
        return []


class ComputerUseProtocol(ABC):
    """Abstract interface for computer-use LLM protocols.

    Each implementation owns its own conversation state and normalizes
    provider-specific responses to the executor's command dict format.
    """

    @abstractmethod
    def step(self, screenshot_base64: str, task_goal: str) -> StepResult:
        """Execute one decision step: send screenshot, get normalized action."""

    @abstractmethod
    def report_result(self, success: bool, error: str | None = None) -> None:
        """Report execution result so the next tool_result can include errors.

        Called by VisionAgent after executor.execute() completes.
        Protocols that use structured tool_result messages (e.g. Anthropic CU)
        should include the error in the next tool_result with is_error=True.
        """

    def report_results(
        self, results: list[tuple[bool, str | None]],
    ) -> None:
        """Report execution results for a batch of commands.

        Default implementation calls :meth:`report_result` with the first
        failure (so the LLM sees the actual error, not "skipped").  If all
        succeeded, reports the last entry.
        Protocols that need per-command feedback (e.g. Anthropic CU with
        multiple tool_use blocks) should override.
        """
        if results:
            for success, error in results:
                if not success:
                    self.report_result(success, error)
                    return
            self.report_result(*results[-1])

    @abstractmethod
    def reset(self) -> None:
        """Clear conversation state for a new task."""

    @abstractmethod
    def get_usage_summary(self) -> dict[str, int]:
        """Return cumulative token usage."""

    def get_system_prompt(self) -> str:
        """Return the system prompt currently in use.

        Used by VisionAgent to expose the full prompt context in the UI.
        Default returns empty string; implementations should override.
        """
        return ""

    def get_messages_snapshot(self) -> list[dict[str, Any]]:
        """Return a deep copy of the current conversation history.

        Used by VisionAgent to attach prompt context to step_info for
        the UI step detail viewer. Default returns empty list; protocol
        implementations that maintain ``self._messages`` should override.
        """
        return []

    def detect_os(self, screenshot_base64: str) -> str | None:
        """Detect the target OS from a screenshot.

        Makes a standalone, stateless LLM call — does NOT affect
        conversation history. Returns ``"windows"``, ``"macos"``, or
        ``"linux"`` on success, ``None`` if detection fails.

        Default implementation returns ``None``. Protocol implementations
        with API access should override.
        """
        return None


def _supports_anthropic_cu(model: str) -> bool:
    """Check if a model supports Anthropic's native computer-use tool."""
    model_lower = model.lower()
    return any(model_lower.startswith(p) for p in ANTHROPIC_CU_MODEL_PREFIXES)


def _supports_openai_cu(model: str) -> bool:
    """Check if a model supports OpenAI's native computer-use tool."""
    model_lower = model.lower()
    return any(model_lower.startswith(p) for p in OPENAI_CU_MODEL_PREFIXES)


def create_protocol(
    provider: str,
    model: str,
    api_key: str,
    *,
    base_url: str | None = None,
    max_tokens: int = 4096,
    temperature: float = 0.0,
    history_max_turns: int = 10,
    display_width: int = 1920,
    display_height: int = 1080,
    protocol_override: str = "auto",
    enable_cache: bool = True,
    skill_text: str | None = None,
) -> ComputerUseProtocol:
    """Create the appropriate protocol for the given provider/model.

    Args:
        provider: LLM provider name ("anthropic", "openai", etc.)
        model: Model identifier.
        api_key: API key.
        base_url: Optional custom API base URL.
        max_tokens: Max tokens for LLM response.
        temperature: Sampling temperature.
        history_max_turns: Max conversation turns to retain.
        display_width: Screen width in pixels.
        display_height: Screen height in pixels.
        protocol_override: "auto" (default), "native", or "prompt".
        enable_cache: Enable Anthropic prompt caching.
    """
    valid_overrides = {"auto", "native", "prompt"}
    if protocol_override not in valid_overrides:
        raise ValueError(
            f"Invalid protocol_override={protocol_override!r}. "
            f"Must be one of: {', '.join(sorted(valid_overrides))}"
        )

    if not api_key:
        raise ValueError(
            f"No API key configured for provider {provider!r}. "
            "Open the Config tab in the web UI and enter your API key, "
            "or pass --api-key on the CLI."
        )

    provider_lower = provider.lower()

    use_anthropic_native = False
    use_openai_native = False

    if protocol_override == "native":
        if provider_lower == "anthropic":
            use_anthropic_native = True
        elif provider_lower == "openai":
            use_openai_native = True
        else:
            raise ValueError(
                f"Native computer-use protocol requires provider='anthropic' "
                f"or 'openai', got provider={provider!r}. "
                f"Use --protocol auto or --protocol prompt."
            )
    elif protocol_override == "prompt":
        pass  # force prompt-based
    elif provider_lower == "anthropic" and _supports_anthropic_cu(model):
        use_anthropic_native = True
    elif provider_lower == "openai" and _supports_openai_cu(model):
        use_openai_native = True

    if use_anthropic_native:
        from cyberraccoon.agent.protocols.anthropic_cu import AnthropicCUProtocol

        logger.info("Using Anthropic native computer-use protocol for %s", model)
        return AnthropicCUProtocol(
            model=model,
            api_key=api_key,
            max_tokens=max_tokens,
            temperature=temperature,
            history_max_turns=history_max_turns,
            display_width=display_width,
            display_height=display_height,
            enable_cache=enable_cache,
            skill_text=skill_text,
        )

    if use_openai_native:
        from cyberraccoon.agent.protocols.openai_cu import OpenAICUProtocol

        logger.info("Using OpenAI native computer-use protocol for %s", model)
        return OpenAICUProtocol(
            model=model,
            api_key=api_key,
            display_width=display_width,
            display_height=display_height,
            skill_text=skill_text,
        )

    from cyberraccoon.agent.protocols.prompt_based import PromptBasedProtocol

    logger.info("Using prompt-based protocol for %s/%s", provider_lower, model)
    return PromptBasedProtocol(
        provider=provider_lower,
        model=model,
        api_key=api_key,
        base_url=base_url,
        max_tokens=max_tokens,
        temperature=temperature,
        history_max_turns=history_max_turns,
        display_width=display_width,
        display_height=display_height,
        enable_cache=enable_cache,
        skill_text=skill_text,
    )
