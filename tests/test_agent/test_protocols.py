"""Tests for the computer-use protocol abstraction layer.

Tests cover:
- AnthropicCUProtocol action normalization
- PromptBasedProtocol JSON parsing fallback
- create_protocol() factory auto-detection
- PromptBasedProtocol history management
"""

from __future__ import annotations

from typing import Any

import pytest

from cyberraccoon.agent.protocols.base import (
    ANTHROPIC_CU_MODEL_PREFIXES,
    OPENAI_CU_MODEL_PREFIXES,
    ComputerUseProtocol,
    StepResult,
    _supports_anthropic_cu,
    _supports_openai_cu,
    create_protocol,
)
from cyberraccoon.agent.protocols.anthropic_cu import AnthropicCUProtocol
from cyberraccoon.agent.protocols.prompt_based import PromptBasedProtocol, VALID_ACTIONS


# ===========================================================================
# Anthropic CU Action Normalization
# ===========================================================================

class TestAnthropicCUNormalization:
    """Tests for AnthropicCUProtocol._normalize_action()."""

    def test_left_click(self) -> None:
        cmd = AnthropicCUProtocol._normalize_action(
            "left_click", {"action": "left_click", "coordinate": [640, 360]},
        )
        assert cmd == {"action": "click", "button": "left", "x": 640, "y": 360}

    def test_right_click(self) -> None:
        cmd = AnthropicCUProtocol._normalize_action(
            "right_click", {"action": "right_click", "coordinate": [100, 200]},
        )
        assert cmd == {"action": "click", "button": "right", "x": 100, "y": 200}

    def test_middle_click(self) -> None:
        cmd = AnthropicCUProtocol._normalize_action(
            "middle_click", {"action": "middle_click", "coordinate": [50, 50]},
        )
        assert cmd == {"action": "click", "button": "middle", "x": 50, "y": 50}

    def test_double_click(self) -> None:
        cmd = AnthropicCUProtocol._normalize_action(
            "double_click", {"action": "double_click", "coordinate": [300, 400]},
        )
        assert cmd == {"action": "double_click", "x": 300, "y": 400}

    def test_triple_click(self) -> None:
        cmd = AnthropicCUProtocol._normalize_action(
            "triple_click", {"action": "triple_click", "coordinate": [10, 20]},
        )
        assert cmd == {"action": "triple_click", "x": 10, "y": 20}

    def test_mouse_move(self) -> None:
        cmd = AnthropicCUProtocol._normalize_action(
            "mouse_move", {"action": "mouse_move", "coordinate": [500, 300]},
        )
        assert cmd == {"action": "mouse_move", "x": 500, "y": 300}

    def test_left_click_drag(self) -> None:
        cmd = AnthropicCUProtocol._normalize_action(
            "left_click_drag",
            {
                "action": "left_click_drag",
                "start_coordinate": [100, 100],
                "coordinate": [500, 500],
            },
        )
        assert cmd == {
            "action": "drag",
            "from_x": 100, "from_y": 100,
            "to_x": 500, "to_y": 500,
        }

    def test_type(self) -> None:
        cmd = AnthropicCUProtocol._normalize_action(
            "type", {"action": "type", "text": "hello world"},
        )
        assert cmd == {"action": "type", "text": "hello world"}

    def test_key_single(self) -> None:
        cmd = AnthropicCUProtocol._normalize_action(
            "key", {"action": "key", "text": "Return"},
        )
        assert cmd == {"action": "key", "keys": ["return"]}

    def test_key_combo(self) -> None:
        cmd = AnthropicCUProtocol._normalize_action(
            "key", {"action": "key", "text": "ctrl+c"},
        )
        assert cmd == {"action": "key", "keys": ["ctrl", "c"]}

    def test_key_three_keys(self) -> None:
        cmd = AnthropicCUProtocol._normalize_action(
            "key", {"action": "key", "text": "ctrl+shift+s"},
        )
        assert cmd == {"action": "key", "keys": ["ctrl", "shift", "s"]}

    def test_scroll_down(self) -> None:
        cmd = AnthropicCUProtocol._normalize_action(
            "scroll",
            {
                "action": "scroll",
                "coordinate": [640, 360],
                "scroll_direction": "down",
                "scroll_amount": 5,
            },
        )
        assert cmd == {
            "action": "scroll",
            "x": 640, "y": 360,
            "direction": "down",
            "amount": 5,
        }

    def test_scroll_defaults(self) -> None:
        cmd = AnthropicCUProtocol._normalize_action(
            "scroll",
            {"action": "scroll", "coordinate": [0, 0]},
        )
        assert cmd["direction"] == "down"
        assert cmd["amount"] == 3

    def test_wait(self) -> None:
        cmd = AnthropicCUProtocol._normalize_action(
            "wait", {"action": "wait", "duration": 2.5},
        )
        assert cmd == {"action": "wait", "duration_s": 2.5}

    def test_wait_defaults(self) -> None:
        cmd = AnthropicCUProtocol._normalize_action(
            "wait", {"action": "wait"},
        )
        assert cmd == {"action": "wait", "duration_s": 1.0}

    def test_left_mouse_down(self) -> None:
        cmd = AnthropicCUProtocol._normalize_action(
            "left_mouse_down",
            {"action": "left_mouse_down", "coordinate": [200, 300]},
        )
        assert cmd == {"action": "mouse_down", "x": 200, "y": 300}

    def test_left_mouse_up(self) -> None:
        cmd = AnthropicCUProtocol._normalize_action(
            "left_mouse_up",
            {"action": "left_mouse_up", "coordinate": [200, 300]},
        )
        assert cmd == {"action": "mouse_up", "x": 200, "y": 300}

    def test_hold_key(self) -> None:
        cmd = AnthropicCUProtocol._normalize_action(
            "hold_key", {"action": "hold_key", "text": "shift", "duration": 2.0},
        )
        assert cmd == {"action": "hold_key", "keys": ["shift"], "duration_s": 2.0}

    def test_hold_key_defaults(self) -> None:
        cmd = AnthropicCUProtocol._normalize_action(
            "hold_key", {"action": "hold_key", "text": "ctrl"},
        )
        assert cmd == {"action": "hold_key", "keys": ["ctrl"], "duration_s": 1.0}

    def test_zoom_unknown_returns_none(self) -> None:
        """zoom is not a recognized action (removed — no benefit at 1280x720)."""
        cmd = AnthropicCUProtocol._normalize_action(
            "zoom", {"action": "zoom", "region": [100, 200, 400, 350]},
        )
        assert cmd is None

    def test_screenshot_returns_none(self) -> None:
        """screenshot action is handled via needs_screenshot flag, not a command."""
        cmd = AnthropicCUProtocol._normalize_action(
            "screenshot", {"action": "screenshot"},
        )
        assert cmd is None

    def test_unknown_action_returns_none(self) -> None:
        cmd = AnthropicCUProtocol._normalize_action(
            "unknown_action", {"action": "unknown_action"},
        )
        assert cmd is None

    def test_modifier_on_click(self) -> None:
        """Click actions with modifier key in text field."""
        cmd = AnthropicCUProtocol._normalize_action(
            "left_click",
            {"action": "left_click", "coordinate": [500, 300], "text": "shift"},
        )
        assert cmd == {
            "action": "click", "button": "left",
            "x": 500, "y": 300, "modifier": "shift",
        }

    def test_modifier_on_scroll(self) -> None:
        """Scroll actions with modifier key in text field."""
        cmd = AnthropicCUProtocol._normalize_action(
            "scroll",
            {
                "action": "scroll",
                "coordinate": [500, 400],
                "scroll_direction": "down",
                "scroll_amount": 3,
                "text": "ctrl",
            },
        )
        assert cmd["modifier"] == "ctrl"

    def test_type_text_not_treated_as_modifier(self) -> None:
        """type action's text field should NOT be treated as modifier."""
        cmd = AnthropicCUProtocol._normalize_action(
            "type", {"action": "type", "text": "shift"},
        )
        assert cmd == {"action": "type", "text": "shift"}
        assert "modifier" not in cmd

    def test_click_missing_coordinate_returns_none(self) -> None:
        """Click without coordinate should return None, not a partial command."""
        cmd = AnthropicCUProtocol._normalize_action(
            "left_click", {"action": "left_click"},
        )
        assert cmd is None

    def test_double_click_missing_coordinate_returns_none(self) -> None:
        cmd = AnthropicCUProtocol._normalize_action(
            "double_click", {"action": "double_click"},
        )
        assert cmd is None

    def test_drag_missing_start_coordinate_returns_none(self) -> None:
        cmd = AnthropicCUProtocol._normalize_action(
            "left_click_drag",
            {"action": "left_click_drag", "coordinate": [500, 500]},
        )
        assert cmd is None

    def test_drag_missing_end_coordinate_returns_none(self) -> None:
        cmd = AnthropicCUProtocol._normalize_action(
            "left_click_drag",
            {"action": "left_click_drag", "start_coordinate": [100, 100]},
        )
        assert cmd is None


# ===========================================================================
# Anthropic CU Error Feedback
# ===========================================================================

class TestAnthropicCUErrorFeedback:
    """Tests for report_result() affecting tool_result messages."""

    def _make_protocol(self) -> AnthropicCUProtocol:
        """Create a protocol instance with mocked client for testing."""
        from unittest.mock import MagicMock
        proto = AnthropicCUProtocol.__new__(AnthropicCUProtocol)
        proto._model = "claude-sonnet-4-6"
        proto._max_tokens = 4096
        proto._temperature = 0.0
        proto._history_max_turns = 10
        proto._display_width = 1280
        proto._display_height = 720
        mock_anthropic = MagicMock()
        proto._anthropic = mock_anthropic
        proto._client = MagicMock()
        proto._tool_def = {"type": "computer_20251124", "name": "computer"}
        proto._system_prompt = "test"
        proto._messages = []
        proto._step_count = 0
        proto._total_input_tokens = 0
        proto._total_output_tokens = 0
        proto._last_tool_use_id = None
        proto._last_exec_error = None
        return proto

    def test_report_result_success_no_error(self) -> None:
        proto = self._make_protocol()
        proto.report_result(True)
        assert proto._last_exec_error is None

    def test_report_result_failure_stores_error(self) -> None:
        proto = self._make_protocol()
        proto.report_result(False, "HID device not found")
        assert proto._last_exec_error == "HID device not found"

    def test_report_result_success_clears_error(self) -> None:
        proto = self._make_protocol()
        proto.report_result(False, "some error")
        proto.report_result(True)
        assert proto._last_exec_error is None

    def test_error_included_in_tool_result(self) -> None:
        """After report_result(False), _build_next_message should include is_error."""
        proto = self._make_protocol()
        # Simulate first turn already happened
        proto._step_count = 1
        proto._last_tool_use_id = "toolu_123"
        proto.report_result(False, "Coordinates out of bounds")

        msg = proto._build_next_message("fake_b64", "test task")

        assert msg["role"] == "user"
        tool_result = msg["content"][0]
        assert tool_result["type"] == "tool_result"
        assert tool_result["is_error"] is True
        # Error text should be in the content
        text_blocks = [b for b in tool_result["content"] if b["type"] == "text"]
        assert len(text_blocks) == 1
        assert "Coordinates out of bounds" in text_blocks[0]["text"]

    def test_no_error_no_is_error_flag(self) -> None:
        """Without report_result error, tool_result should not have is_error."""
        proto = self._make_protocol()
        proto._step_count = 1
        proto._last_tool_use_id = "toolu_456"

        msg = proto._build_next_message("fake_b64", "test task")

        tool_result = msg["content"][0]
        assert "is_error" not in tool_result
        # No text block for error
        text_blocks = [b for b in tool_result["content"] if b["type"] == "text"]
        assert len(text_blocks) == 0

    def test_error_cleared_after_build(self) -> None:
        """_build_next_message should clear the error after using it."""
        proto = self._make_protocol()
        proto._step_count = 1
        proto._last_tool_use_id = "toolu_789"
        proto.report_result(False, "some error")

        proto._build_next_message("fake_b64", "task")
        assert proto._last_exec_error is None

    def test_reset_clears_error(self) -> None:
        proto = self._make_protocol()
        proto.report_result(False, "error")
        proto.reset()
        assert proto._last_exec_error is None


# ===========================================================================
# Anthropic CU Coordinate Validation
# ===========================================================================

class TestAnthropicCUCoordinateValidation:
    """Tests for coordinate bounds checking."""

    def _make_protocol(self) -> AnthropicCUProtocol:
        proto = AnthropicCUProtocol.__new__(AnthropicCUProtocol)
        proto._display_width = 1280
        proto._display_height = 720
        return proto

    def test_valid_coordinates_pass(self) -> None:
        proto = self._make_protocol()
        cmd = {"action": "click", "x": 640, "y": 360}
        assert proto._validate_coordinates(cmd) is None

    def test_zero_coordinates_pass(self) -> None:
        proto = self._make_protocol()
        cmd = {"action": "click", "x": 0, "y": 0}
        assert proto._validate_coordinates(cmd) is None

    def test_max_edge_coordinates_pass(self) -> None:
        proto = self._make_protocol()
        cmd = {"action": "click", "x": 1279, "y": 719}
        assert proto._validate_coordinates(cmd) is None

    def test_x_out_of_bounds(self) -> None:
        proto = self._make_protocol()
        cmd = {"action": "click", "x": 1280, "y": 360}
        error = proto._validate_coordinates(cmd)
        assert error is not None
        assert "out of bounds" in error

    def test_y_out_of_bounds(self) -> None:
        proto = self._make_protocol()
        cmd = {"action": "click", "x": 640, "y": 720}
        error = proto._validate_coordinates(cmd)
        assert error is not None
        assert "out of bounds" in error

    def test_negative_coordinate(self) -> None:
        proto = self._make_protocol()
        cmd = {"action": "click", "x": -1, "y": 360}
        error = proto._validate_coordinates(cmd)
        assert error is not None
        assert "out of bounds" in error

    def test_drag_validates_both_endpoints(self) -> None:
        proto = self._make_protocol()
        # from coords ok, to coords out of bounds
        cmd = {"action": "drag", "from_x": 100, "from_y": 100, "to_x": 2000, "to_y": 100}
        error = proto._validate_coordinates(cmd)
        assert error is not None

    def test_no_coordinates_passes(self) -> None:
        """Commands without coordinates (e.g. type) should pass validation."""
        proto = self._make_protocol()
        cmd = {"action": "type", "text": "hello"}
        assert proto._validate_coordinates(cmd) is None


# ===========================================================================
# Anthropic CU Coordinate Scaling
# ===========================================================================

class TestAnthropicCUScaling:
    """Tests for screenshot scaling and coordinate upscaling."""

    def test_no_scaling_at_1280x720(self) -> None:
        proto = AnthropicCUProtocol.__new__(AnthropicCUProtocol)
        proto._display_width = 1280
        proto._display_height = 720
        assert proto._get_scale_factor() == 1.0

    def test_scaling_needed_for_high_res(self) -> None:
        proto = AnthropicCUProtocol.__new__(AnthropicCUProtocol)
        proto._display_width = 1920
        proto._display_height = 1080
        scale = proto._get_scale_factor()
        assert scale < 1.0
        # Scaled dimensions should be within limits
        assert int(1920 * scale) <= 1568
        assert int(1080 * scale) <= 1568
        assert int(1920 * scale) * int(1080 * scale) <= 1_150_000

    def test_scale_coordinates_up(self) -> None:
        cmd = {"action": "click", "x": 640, "y": 360}
        AnthropicCUProtocol._scale_coordinates_up(cmd, 0.5)
        assert cmd["x"] == 1280
        assert cmd["y"] == 720

    def test_scale_coordinates_up_drag(self) -> None:
        cmd = {"action": "drag", "from_x": 50, "from_y": 50, "to_x": 200, "to_y": 200}
        AnthropicCUProtocol._scale_coordinates_up(cmd, 0.5)
        assert cmd["from_x"] == 100
        assert cmd["from_y"] == 100
        assert cmd["to_x"] == 400
        assert cmd["to_y"] == 400


# ===========================================================================
# Prompt-Based Protocol JSON Parsing
# ===========================================================================

class TestPromptBasedParsing:
    """Tests for PromptBasedProtocol._parse_commands() via shared parsing."""

    def _parse(self, text: str) -> dict[str, Any] | None:
        """Convenience: call the shared try_parse_json utility."""
        from cyberraccoon.agent.protocols.parsing import try_parse_json
        return try_parse_json(text, VALID_ACTIONS)

    def test_valid_json_direct(self) -> None:
        result = self._parse('{"action": "left_click", "coordinate": [1, 2]}')
        assert result is not None
        assert result["action"] == "left_click"

    def test_invalid_action_rejected(self) -> None:
        result = self._parse('{"action": "fly_to_moon"}')
        assert result is None

    def test_non_dict_rejected(self) -> None:
        result = self._parse('[1, 2, 3]')
        assert result is None

    def test_invalid_json_rejected(self) -> None:
        result = self._parse('not json at all')
        assert result is None

    def test_all_valid_actions_accepted(self) -> None:
        for action in VALID_ACTIONS:
            result = self._parse(f'{{"action": "{action}"}}')
            assert result is not None, f"Action {action} should be valid"


# ===========================================================================
# Factory Function
# ===========================================================================

class TestCreateProtocol:
    """Tests for create_protocol() factory and model detection."""

    def test_supports_anthropic_cu_sonnet_4_6(self) -> None:
        assert _supports_anthropic_cu("claude-sonnet-4-6") is True
        assert _supports_anthropic_cu("claude-sonnet-4-6-20260101") is True

    def test_supports_anthropic_cu_opus_4_6(self) -> None:
        assert _supports_anthropic_cu("claude-opus-4-6") is True

    def test_supports_anthropic_cu_opus_4_5(self) -> None:
        assert _supports_anthropic_cu("claude-opus-4-5-20251009") is True

    def test_does_not_support_sonnet_4_5(self) -> None:
        """Sonnet 4.5 only supports computer_20250124, not 20251124."""
        assert _supports_anthropic_cu("claude-sonnet-4-5-20250514") is False

    def test_does_not_support_sonnet_4(self) -> None:
        """Sonnet 4 only supports computer_20250124."""
        assert _supports_anthropic_cu("claude-sonnet-4-20250514") is False

    def test_does_not_support_opus_4(self) -> None:
        """Opus 4 only supports computer_20250124."""
        assert _supports_anthropic_cu("claude-opus-4-20250514") is False

    def test_does_not_support_opus_4_1(self) -> None:
        """Opus 4.1 only supports computer_20250124."""
        assert _supports_anthropic_cu("claude-opus-4-1-20250527") is False

    def test_does_not_support_older_models(self) -> None:
        assert _supports_anthropic_cu("claude-3-sonnet-20240229") is False
        assert _supports_anthropic_cu("gpt-4o") is False

    def test_case_insensitive(self) -> None:
        assert _supports_anthropic_cu("Claude-Sonnet-4-6") is True

    def test_protocol_override_prompt_forces_prompt_based(self) -> None:
        """protocol_override='prompt' should always return PromptBasedProtocol."""
        from unittest.mock import patch, MagicMock

        mock_anthropic = MagicMock()
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            protocol = create_protocol(
                provider="anthropic",
                model="claude-sonnet-4-6",
                api_key="test-key",
                protocol_override="prompt",
            )
            assert isinstance(protocol, PromptBasedProtocol)

    def test_non_anthropic_provider_returns_prompt_based(self) -> None:
        """Non-anthropic providers should always get prompt-based protocol."""
        from unittest.mock import patch, MagicMock

        mock_openai = MagicMock()
        with patch.dict("sys.modules", {"openai": mock_openai}):
            protocol = create_protocol(
                provider="openai",
                model="gpt-4o",
                api_key="test-key",
            )
            assert isinstance(protocol, PromptBasedProtocol)

    def test_invalid_protocol_override_raises(self) -> None:
        """Invalid protocol_override should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid protocol_override"):
            create_protocol(
                provider="anthropic",
                model="claude-sonnet-4-6",
                api_key="test-key",
                protocol_override="natiive",
            )

    def test_native_override_with_unsupported_provider_raises(self) -> None:
        """protocol_override='native' with unsupported provider should raise."""
        with pytest.raises(ValueError, match="requires provider"):
            create_protocol(
                provider="google",
                model="gemini-2.5-flash",
                api_key="test-key",
                protocol_override="native",
            )

    def test_native_override_openai_creates_openai_cu(self) -> None:
        """protocol_override='native' + openai → OpenAICUProtocol."""
        from unittest.mock import patch, MagicMock

        mock_openai = MagicMock()
        with patch.dict("sys.modules", {"openai": mock_openai}):
            protocol = create_protocol(
                provider="openai",
                model="gpt-5.4",
                api_key="test-key",
                protocol_override="native",
            )
            from cyberraccoon.agent.protocols.openai_cu import OpenAICUProtocol
            assert isinstance(protocol, OpenAICUProtocol)

    def test_auto_openai_gpt54_creates_openai_cu(self) -> None:
        """auto + openai + gpt-5.4 → OpenAICUProtocol."""
        from unittest.mock import patch, MagicMock

        mock_openai = MagicMock()
        with patch.dict("sys.modules", {"openai": mock_openai}):
            protocol = create_protocol(
                provider="openai",
                model="gpt-5.4",
                api_key="test-key",
            )
            from cyberraccoon.agent.protocols.openai_cu import OpenAICUProtocol
            assert isinstance(protocol, OpenAICUProtocol)

    def test_supports_openai_cu_prefixes(self) -> None:
        assert _supports_openai_cu("gpt-5.4") is True
        assert _supports_openai_cu("gpt-5.4-2026-03-01") is True
        assert _supports_openai_cu("GPT-5.4") is True
        assert _supports_openai_cu("gpt-4o") is False
        assert _supports_openai_cu("gpt-5") is False
        assert _supports_openai_cu("claude-sonnet-4-6") is False


# ===========================================================================
# PromptBasedProtocol History Management
# ===========================================================================

class TestPromptBasedHistory:
    """Tests for PromptBasedProtocol internal history management."""

    def test_update_history_with_summary(self) -> None:
        """History should include screen_summary as user message."""
        proto = PromptBasedProtocol.__new__(PromptBasedProtocol)
        proto._messages = []
        proto._history_max_turns = 10

        proto._update_history("desktop with icons", '{"action": "left_click"}')

        assert len(proto._messages) == 2
        assert proto._messages[0]["role"] == "user"
        assert "desktop with icons" in proto._messages[0]["content"]
        assert proto._messages[1]["role"] == "assistant"

    def test_update_history_without_summary(self) -> None:
        """Without screen_summary, only assistant message is stored."""
        proto = PromptBasedProtocol.__new__(PromptBasedProtocol)
        proto._messages = []
        proto._history_max_turns = 10

        proto._update_history("", '{"action": "left_click"}')

        assert len(proto._messages) == 1
        assert proto._messages[0]["role"] == "assistant"

    def test_history_trimming(self) -> None:
        """History should be trimmed to history_max_turns."""
        proto = PromptBasedProtocol.__new__(PromptBasedProtocol)
        proto._messages = []
        proto._history_max_turns = 2

        for i in range(5):
            proto._update_history(f"step {i}", f"response {i}")

        # max_turns=2 -> max 4 messages (2 pairs)
        assert len(proto._messages) == 4

    def test_reset_clears_history(self) -> None:
        proto = PromptBasedProtocol.__new__(PromptBasedProtocol)
        proto._messages = [{"role": "user", "content": "test"}]
        proto._step_count = 5

        proto.reset()

        assert proto._messages == []
        assert proto._step_count == 0


# ===========================================================================
# StepResult Dataclass
# ===========================================================================

class TestStepResult:
    """Tests for StepResult defaults and fields."""

    def test_defaults(self) -> None:
        result = StepResult(
            command=None,
            is_done=False,
            done_reason="",
            screen_summary="",
            raw_text="",
            input_tokens=0,
            output_tokens=0,
            latency_ms=0,
            success=False,
        )
        assert result.error is None
        assert result.needs_screenshot is False

    def test_needs_screenshot_flag(self) -> None:
        result = StepResult(
            command=None,
            is_done=False,
            done_reason="",
            screen_summary="",
            raw_text="",
            input_tokens=0,
            output_tokens=0,
            latency_ms=0,
            success=True,
            needs_screenshot=True,
        )
        assert result.needs_screenshot is True


# ===========================================================================
# Executor mouse_move action
# ===========================================================================

class TestExecutorNewActions:
    """Test executor dispatch for new actions."""

    def _make_executor(self):
        from unittest.mock import MagicMock
        from cyberraccoon.executor.base_executor import BaseExecutor

        class TestExecutor(BaseExecutor):
            def open(self) -> None:
                self._keyboard = MagicMock()
                self._mouse = MagicMock()

            def close(self) -> None:
                pass

        executor = TestExecutor()
        executor.open()
        return executor

    def test_mouse_move_dispatched(self) -> None:
        executor = self._make_executor()
        result = executor.execute(
            {"id": "mm1", "action": "mouse_move", "x": 640, "y": 360},
        )
        assert result["status"] == "ok"
        executor._mouse.move.assert_called_once_with(x=640, y=360)

    def test_mouse_down_dispatched(self) -> None:
        executor = self._make_executor()
        result = executor.execute(
            {"id": "md1", "action": "mouse_down", "x": 100, "y": 200},
        )
        assert result["status"] == "ok"
        executor._mouse.mouse_down.assert_called_once_with(x=100, y=200)

    def test_mouse_up_dispatched(self) -> None:
        executor = self._make_executor()
        result = executor.execute(
            {"id": "mu1", "action": "mouse_up", "x": 100, "y": 200},
        )
        assert result["status"] == "ok"
        executor._mouse.mouse_up.assert_called_once_with(x=100, y=200)

    def test_hold_key_dispatched(self) -> None:
        executor = self._make_executor()
        result = executor.execute(
            {"id": "hk1", "action": "hold_key", "keys": ["shift"], "duration_s": 0.01},
        )
        assert result["status"] == "ok"
        executor._keyboard.press_keys.assert_called_once_with(["shift"])


# ===========================================================================
# detect_os() method
# ===========================================================================

class TestDetectOS:
    """Tests for the detect_os() method on protocol classes."""

    def test_base_class_returns_none(self) -> None:
        """Default detect_os on ComputerUseProtocol ABC returns None."""
        from tests.test_agent.conftest import MockProtocol
        proto = MockProtocol([])
        assert proto.detect_os("fake_b64") is None

    def test_mock_protocol_returns_configured_value(self) -> None:
        """MockProtocol with detect_os_result returns the configured value."""
        from tests.test_agent.conftest import MockProtocol
        proto = MockProtocol([], detect_os_result="windows")
        assert proto.detect_os("fake_b64") == "windows"
        assert len(proto.detect_os_calls) == 1

    def test_anthropic_cu_detect_os_parses_response(self) -> None:
        """AnthropicCUProtocol.detect_os should parse 'macos' from API response."""
        from unittest.mock import MagicMock
        proto = AnthropicCUProtocol.__new__(AnthropicCUProtocol)
        proto._model = "claude-sonnet-4-6"
        mock_client = MagicMock()
        proto._client = mock_client

        # Mock API response
        mock_block = MagicMock()
        mock_block.text = "macos"
        mock_response = MagicMock()
        mock_response.content = [mock_block]
        mock_client.messages.create.return_value = mock_response

        result = proto.detect_os("fake_screenshot_b64")
        assert result == "macos"
        mock_client.messages.create.assert_called_once()

    def test_anthropic_cu_detect_os_invalid_returns_none(self) -> None:
        """AnthropicCUProtocol.detect_os returns None for unrecognized OS."""
        from unittest.mock import MagicMock
        proto = AnthropicCUProtocol.__new__(AnthropicCUProtocol)
        proto._model = "claude-sonnet-4-6"
        mock_client = MagicMock()
        proto._client = mock_client

        mock_block = MagicMock()
        mock_block.text = "ChromeOS"
        mock_response = MagicMock()
        mock_response.content = [mock_block]
        mock_client.messages.create.return_value = mock_response

        result = proto.detect_os("fake_b64")
        assert result is None

    def test_anthropic_cu_detect_os_api_error_returns_none(self) -> None:
        """AnthropicCUProtocol.detect_os returns None on API error."""
        from unittest.mock import MagicMock
        proto = AnthropicCUProtocol.__new__(AnthropicCUProtocol)
        proto._model = "claude-sonnet-4-6"
        mock_client = MagicMock()
        proto._client = mock_client
        mock_client.messages.create.side_effect = Exception("API down")

        result = proto.detect_os("fake_b64")
        assert result is None


# ===========================================================================
# Prompt Caching
# ===========================================================================

class TestCacheMetrics:
    """Tests for Anthropic prompt caching integration."""

    def test_step_result_cache_defaults(self) -> None:
        """StepResult cache fields default to 0."""
        result = StepResult(
            command=None, is_done=False, done_reason="",
            screen_summary="", raw_text="",
            input_tokens=100, output_tokens=50,
            latency_ms=500, success=True,
        )
        assert result.cache_read_tokens == 0
        assert result.cache_creation_tokens == 0

    def test_step_result_cache_fields_set(self) -> None:
        result = StepResult(
            command=None, is_done=False, done_reason="",
            screen_summary="", raw_text="",
            input_tokens=100, output_tokens=50,
            latency_ms=500, success=True,
            cache_read_tokens=200, cache_creation_tokens=300,
        )
        assert result.cache_read_tokens == 200
        assert result.cache_creation_tokens == 300

    def _make_cu_protocol(
        self, *, enable_cache: bool = True,
    ) -> AnthropicCUProtocol:
        """Create an AnthropicCUProtocol with mocked client."""
        from unittest.mock import MagicMock
        proto = AnthropicCUProtocol.__new__(AnthropicCUProtocol)
        proto._model = "claude-sonnet-4-6"
        proto._max_tokens = 4096
        proto._temperature = 0.0
        proto._history_max_turns = 10
        proto._display_width = 1280
        proto._display_height = 720
        proto._enable_cache = enable_cache
        mock_anthropic = MagicMock()
        proto._anthropic = mock_anthropic
        proto._client = MagicMock()
        proto._tool_def = {
            "type": "computer_20251124", "name": "computer",
            "display_width_px": 1280, "display_height_px": 720,
            "display_number": 1,
        }
        proto._system_prompt = "test system prompt"
        proto._messages = []
        proto._step_count = 0
        proto._total_input_tokens = 0
        proto._total_output_tokens = 0
        proto._total_cache_read_tokens = 0
        proto._total_cache_creation_tokens = 0
        proto._last_tool_use_id = None
        proto._last_exec_error = None
        return proto

    def test_anthropic_cu_cache_enabled(self) -> None:
        """With enable_cache=True, cache_control is passed to the API."""
        proto = self._make_cu_protocol(enable_cache=True)
        proto._call_api()

        call_kwargs = proto._client.beta.messages.create.call_args
        assert "cache_control" in call_kwargs.kwargs
        assert call_kwargs.kwargs["cache_control"] == {"type": "ephemeral"}

    def test_anthropic_cu_cache_disabled(self) -> None:
        """With enable_cache=False, cache_control is NOT passed."""
        proto = self._make_cu_protocol(enable_cache=False)
        proto._call_api()

        call_kwargs = proto._client.beta.messages.create.call_args
        assert "cache_control" not in call_kwargs.kwargs


# ===========================================================================
# Completion Status Propagation (COMP-03)
# ===========================================================================

class TestCompletionStatusPropagation:
    """Tests for completion_status propagation through protocol implementations.

    Verifies that all three protocol implementations (PromptBased,
    AnthropicCU, OpenAICU) correctly extract and set completion_status
    on StepResult when the LLM signals task completion.
    """

    # -----------------------------------------------------------------------
    # PromptBasedProtocol tests
    # -----------------------------------------------------------------------

    def test_prompt_based_gave_up_status(self) -> None:
        """PromptBased: done action with status='gave_up' sets completion_status."""
        from unittest.mock import MagicMock, patch

        proto = PromptBasedProtocol.__new__(PromptBasedProtocol)
        proto._provider = "anthropic"
        proto._model = "test-model"
        proto._max_tokens = 4096
        proto._temperature = 0.0
        proto._history_max_turns = 10
        proto._enable_cache = False
        proto._system_prompt = "test"
        proto._messages = []
        proto._step_count = 0
        proto._total_input_tokens = 0
        proto._total_output_tokens = 0
        proto._total_cache_read_tokens = 0
        proto._total_cache_creation_tokens = 0
        proto._last_exec_error = None
        proto._anthropic_client = MagicMock()

        # Mock LLM returning done with gave_up status
        done_response = '{"action": "done", "status": "gave_up", "reason": "Cannot find button"}'
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=done_response)]
        mock_response.usage = MagicMock(
            input_tokens=100, output_tokens=10,
            cache_read_input_tokens=0, cache_creation_input_tokens=0,
        )
        proto._anthropic_client.messages.create.return_value = mock_response

        result = proto.step("fake_b64", "test task")
        assert result.is_done is True
        assert result.completion_status == "gave_up"

    def test_prompt_based_default_success_status(self) -> None:
        """PromptBased: done action without status field defaults to 'success'."""
        from unittest.mock import MagicMock

        proto = PromptBasedProtocol.__new__(PromptBasedProtocol)
        proto._provider = "anthropic"
        proto._model = "test-model"
        proto._max_tokens = 4096
        proto._temperature = 0.0
        proto._history_max_turns = 10
        proto._enable_cache = False
        proto._system_prompt = "test"
        proto._messages = []
        proto._step_count = 0
        proto._total_input_tokens = 0
        proto._total_output_tokens = 0
        proto._total_cache_read_tokens = 0
        proto._total_cache_creation_tokens = 0
        proto._last_exec_error = None
        proto._anthropic_client = MagicMock()

        done_response = '{"action": "done", "reason": "Done"}'
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=done_response)]
        mock_response.usage = MagicMock(
            input_tokens=100, output_tokens=10,
            cache_read_input_tokens=0, cache_creation_input_tokens=0,
        )
        proto._anthropic_client.messages.create.return_value = mock_response

        result = proto.step("fake_b64", "test task")
        assert result.is_done is True
        assert result.completion_status == "success"

    def test_prompt_based_stuck_status(self) -> None:
        """PromptBased: done action with status='stuck' sets completion_status."""
        from unittest.mock import MagicMock

        proto = PromptBasedProtocol.__new__(PromptBasedProtocol)
        proto._provider = "anthropic"
        proto._model = "test-model"
        proto._max_tokens = 4096
        proto._temperature = 0.0
        proto._history_max_turns = 10
        proto._enable_cache = False
        proto._system_prompt = "test"
        proto._messages = []
        proto._step_count = 0
        proto._total_input_tokens = 0
        proto._total_output_tokens = 0
        proto._total_cache_read_tokens = 0
        proto._total_cache_creation_tokens = 0
        proto._last_exec_error = None
        proto._anthropic_client = MagicMock()

        done_response = '{"action": "done", "status": "stuck", "reason": "Wrong screen"}'
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=done_response)]
        mock_response.usage = MagicMock(
            input_tokens=100, output_tokens=10,
            cache_read_input_tokens=0, cache_creation_input_tokens=0,
        )
        proto._anthropic_client.messages.create.return_value = mock_response

        result = proto.step("fake_b64", "test task")
        assert result.is_done is True
        assert result.completion_status == "stuck"

    # -----------------------------------------------------------------------
    # AnthropicCUProtocol tests
    # -----------------------------------------------------------------------

    def _make_anthropic_cu(self) -> "AnthropicCUProtocol":
        """Create an AnthropicCUProtocol with mocked client for testing."""
        from unittest.mock import MagicMock
        proto = AnthropicCUProtocol.__new__(AnthropicCUProtocol)
        proto._model = "claude-sonnet-4-6"
        proto._max_tokens = 4096
        proto._temperature = 0.0
        proto._history_max_turns = 10
        proto._display_width = 1280
        proto._display_height = 720
        proto._enable_cache = False
        mock_anthropic = MagicMock()
        proto._anthropic = mock_anthropic
        proto._client = MagicMock()
        proto._tool_def = {
            "type": "computer_20251124", "name": "computer",
            "display_width_px": 1280, "display_height_px": 720,
            "display_number": 1,
        }
        proto._system_prompt = "test"
        proto._messages = []
        proto._step_count = 0
        proto._total_input_tokens = 0
        proto._total_output_tokens = 0
        proto._total_cache_read_tokens = 0
        proto._total_cache_creation_tokens = 0
        proto._last_tool_use_id = None
        proto._last_exec_error = None
        return proto

    def test_anthropic_cu_gave_up_status(self) -> None:
        """AnthropicCU: done text with status JSON sets completion_status='gave_up'."""
        from unittest.mock import MagicMock
        proto = self._make_anthropic_cu()

        # Mock response with text block only (no tool_use = done path)
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = '{"status": "gave_up"} I could not find the button.'
        mock_response = MagicMock()
        mock_response.content = [text_block]
        mock_response.usage = MagicMock(
            input_tokens=100, output_tokens=10,
            cache_read_input_tokens=0, cache_creation_input_tokens=0,
        )
        proto._client.beta.messages.create.return_value = mock_response

        result = proto.step("fake_b64", "test task")
        assert result.is_done is True
        assert result.completion_status == "gave_up"

    def test_anthropic_cu_default_success(self) -> None:
        """AnthropicCU: done text without status JSON defaults to 'success'."""
        from unittest.mock import MagicMock
        proto = self._make_anthropic_cu()

        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "Task completed successfully."
        mock_response = MagicMock()
        mock_response.content = [text_block]
        mock_response.usage = MagicMock(
            input_tokens=100, output_tokens=10,
            cache_read_input_tokens=0, cache_creation_input_tokens=0,
        )
        proto._client.beta.messages.create.return_value = mock_response

        result = proto.step("fake_b64", "test task")
        assert result.is_done is True
        assert result.completion_status == "success"

    # -----------------------------------------------------------------------
    # OpenAICUProtocol tests
    # -----------------------------------------------------------------------

    def _make_openai_cu(self) -> "OpenAICUProtocol":
        """Create an OpenAICUProtocol with mocked client for testing."""
        from unittest.mock import MagicMock
        from cyberraccoon.agent.protocols.openai_cu import OpenAICUProtocol
        proto = OpenAICUProtocol.__new__(OpenAICUProtocol)
        proto._model = "gpt-5.4"
        proto._display_width = 1280
        proto._display_height = 720
        mock_openai = MagicMock()
        proto._openai = mock_openai
        proto._client = MagicMock()
        proto._system_prompt = "test"
        proto._last_response_id = None
        proto._last_call_id = None
        proto._pending_safety_checks = []
        import collections
        proto._action_queue = collections.deque()
        proto._last_exec_error = None
        proto._messages = []
        proto._step_count = 0
        proto._total_input_tokens = 0
        proto._total_output_tokens = 0
        proto._total_cache_read_tokens = 0
        return proto

    def test_openai_cu_stuck_status(self) -> None:
        """OpenAICU: done text with status JSON sets completion_status='stuck'."""
        from unittest.mock import MagicMock
        proto = self._make_openai_cu()

        # Mock response: no computer_call (done path)
        mock_response = MagicMock()
        mock_response.output = []  # no computer_call
        mock_response.output_text = '{"status": "stuck"} Screen does not match.'
        mock_response.usage = MagicMock(
            input_tokens=100, output_tokens=10,
        )
        mock_response.usage.input_tokens_details = None
        proto._client.responses.create.return_value = mock_response

        result = proto.step("fake_b64", "test task")
        assert result.is_done is True
        assert result.completion_status == "stuck"

    def test_openai_cu_default_success(self) -> None:
        """OpenAICU: done text without status JSON defaults to 'success'."""
        from unittest.mock import MagicMock
        proto = self._make_openai_cu()

        mock_response = MagicMock()
        mock_response.output = []  # no computer_call
        mock_response.output_text = "Task completed as requested."
        mock_response.usage = MagicMock(
            input_tokens=100, output_tokens=10,
        )
        mock_response.usage.input_tokens_details = None
        proto._client.responses.create.return_value = mock_response

        result = proto.step("fake_b64", "test task")
        assert result.is_done is True
        assert result.completion_status == "success"


# ===========================================================================
# Prompt Caching (continued)
# ===========================================================================

class TestCacheMetricsContinued(TestCacheMetrics):
    """Continuation of cache metric tests (split by TestCompletionStatusPropagation)."""

    def test_anthropic_cu_cache_metrics_in_step(self) -> None:
        """Cache metrics from response.usage are included in StepResult."""
        from unittest.mock import MagicMock
        proto = self._make_cu_protocol()

        # Mock tool_use response
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "Clicking button"
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.id = "toolu_123"
        tool_block.input = {
            "action": "left_click", "coordinate": [640, 360],
        }

        mock_response = MagicMock()
        mock_response.content = [text_block, tool_block]
        mock_response.usage.input_tokens = 1000
        mock_response.usage.output_tokens = 50
        mock_response.usage.cache_read_input_tokens = 800
        mock_response.usage.cache_creation_input_tokens = 200
        proto._client.beta.messages.create.return_value = mock_response

        result = proto.step("fake_b64", "Click the button")

        assert result.success
        assert result.input_tokens == 1000  # uncached only
        assert result.cache_read_tokens == 800
        assert result.cache_creation_tokens == 200
        assert proto._total_cache_read_tokens == 800
        assert proto._total_cache_creation_tokens == 200

    def test_anthropic_cu_cache_accumulates(self) -> None:
        """Cache metrics accumulate across steps."""
        from unittest.mock import MagicMock
        proto = self._make_cu_protocol()

        def make_response(cache_read: int, cache_create: int) -> MagicMock:
            text_block = MagicMock()
            text_block.type = "text"
            text_block.text = "action"
            tool_block = MagicMock()
            tool_block.type = "tool_use"
            tool_block.id = f"toolu_{cache_read}"
            tool_block.input = {
                "action": "left_click", "coordinate": [640, 360],
            }
            resp = MagicMock()
            resp.content = [text_block, tool_block]
            resp.usage.input_tokens = 1000
            resp.usage.output_tokens = 50
            resp.usage.cache_read_input_tokens = cache_read
            resp.usage.cache_creation_input_tokens = cache_create
            return resp

        proto._client.beta.messages.create.side_effect = [
            make_response(0, 500),    # first call: cache write
            make_response(500, 0),    # second call: cache hit
        ]

        proto.step("b64_1", "Task")
        proto.report_result(True)
        proto.step("b64_2", "Task")

        usage = proto.get_usage_summary()
        assert usage["total_cache_read_tokens"] == 500
        assert usage["total_cache_creation_tokens"] == 500

    def test_detect_os_no_cache(self) -> None:
        """detect_os() should NOT pass cache_control (one-off call)."""
        from unittest.mock import MagicMock
        proto = self._make_cu_protocol(enable_cache=True)

        mock_block = MagicMock()
        mock_block.text = "windows"
        mock_response = MagicMock()
        mock_response.content = [mock_block]
        proto._client.messages.create.return_value = mock_response

        proto.detect_os("fake_b64")

        call_kwargs = proto._client.messages.create.call_args
        assert "cache_control" not in call_kwargs.kwargs

    def test_prompt_based_anthropic_cache_enabled(self) -> None:
        """PromptBasedProtocol passes cache_control for Anthropic calls."""
        from unittest.mock import MagicMock, patch

        mock_anthropic = MagicMock()
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            proto = PromptBasedProtocol(
                provider="anthropic",
                model="claude-sonnet-4",
                api_key="test-key",
                enable_cache=True,
            )

        mock_block = MagicMock()
        mock_block.text = '{"action": "done", "reason": "done"}'
        mock_response = MagicMock()
        mock_response.content = [mock_block]
        mock_response.usage.input_tokens = 500
        mock_response.usage.output_tokens = 20
        mock_response.usage.cache_read_input_tokens = 300
        mock_response.usage.cache_creation_input_tokens = 100
        proto._anthropic_client.messages.create.return_value = mock_response

        result = proto.step("fake_b64", "Do something")

        call_kwargs = proto._anthropic_client.messages.create.call_args
        assert "cache_control" in call_kwargs.kwargs
        assert result.cache_read_tokens == 300
        assert result.cache_creation_tokens == 100

    def test_prompt_based_openai_no_cache(self) -> None:
        """OpenAI calls should not have cache_control, cache fields are 0."""
        from unittest.mock import MagicMock, patch

        mock_openai = MagicMock()
        with patch.dict("sys.modules", {"openai": mock_openai}):
            proto = PromptBasedProtocol(
                provider="openai",
                model="gpt-4o",
                api_key="test-key",
                enable_cache=True,
            )

        mock_msg = MagicMock()
        mock_msg.content = '{"action": "done", "reason": "done"}'
        mock_choice = MagicMock()
        mock_choice.message = mock_msg
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage.prompt_tokens = 500
        mock_response.usage.completion_tokens = 20
        proto._openai_client.chat.completions.create.return_value = mock_response

        result = proto.step("fake_b64", "Do something")

        call_kwargs = proto._openai_client.chat.completions.create.call_args
        assert "cache_control" not in call_kwargs.kwargs
        assert result.cache_read_tokens == 0
        assert result.cache_creation_tokens == 0


# ===========================================================================
# StepResult.get_commands()
# ===========================================================================

class TestGetCommands:
    """Tests for StepResult.get_commands() helper."""

    def test_returns_commands_when_set(self) -> None:
        cmds = [{"action": "click"}, {"action": "type"}]
        result = StepResult(
            command={"action": "click"},
            is_done=False, done_reason="", screen_summary="",
            raw_text="", input_tokens=0, output_tokens=0,
            latency_ms=0, success=True, commands=cmds,
        )
        assert result.get_commands() == cmds

    def test_falls_back_to_command(self) -> None:
        cmd = {"action": "click"}
        result = StepResult(
            command=cmd,
            is_done=False, done_reason="", screen_summary="",
            raw_text="", input_tokens=0, output_tokens=0,
            latency_ms=0, success=True,
        )
        assert result.get_commands() == [cmd]

    def test_returns_empty_when_none(self) -> None:
        result = StepResult(
            command=None,
            is_done=True, done_reason="done", screen_summary="",
            raw_text="", input_tokens=0, output_tokens=0,
            latency_ms=0, success=True,
        )
        assert result.get_commands() == []
