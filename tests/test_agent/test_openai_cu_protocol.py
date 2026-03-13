"""Tests for OpenAI native computer-use protocol.

Tests cover:
- Action normalization (OpenAI action types → executor commands)
- Action queue behavior (batched multi-action responses)
- Completion detection (done when no computer_call in output)
- Conversation state management (response_id chaining, reset)
- Safety check auto-acknowledgment
- Error handling (API errors, state preservation)
- OS detection
"""

from __future__ import annotations

import collections
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from agent.protocols.openai_cu import OpenAICUProtocol, _normalize_key


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

class _MockAPIError(Exception):
    """Stand-in for openai.APIError in tests."""


def _make_protocol() -> OpenAICUProtocol:
    """Create a protocol instance with mocked OpenAI client."""
    proto = OpenAICUProtocol.__new__(OpenAICUProtocol)
    proto._model = "gpt-5.4"
    proto._display_width = 1280
    proto._display_height = 720
    mock_openai = MagicMock()
    mock_openai.APIError = _MockAPIError
    proto._openai = mock_openai
    proto._client = MagicMock()
    proto._system_prompt = "test system prompt"
    proto._last_response_id = None
    proto._last_call_id = None
    proto._pending_safety_checks = []
    proto._action_queue = collections.deque()
    proto._last_exec_error = None
    proto._step_count = 0
    proto._total_input_tokens = 0
    proto._total_output_tokens = 0
    proto._total_cache_read_tokens = 0
    return proto


def _make_response(
    output: list[Any],
    response_id: str = "resp_123",
    input_tokens: int = 100,
    output_tokens: int = 50,
    output_text: str = "",
    cached_tokens: int = 0,
) -> SimpleNamespace:
    """Create a mock OpenAI API response."""
    usage = SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        input_tokens_details=SimpleNamespace(cached_tokens=cached_tokens),
    )
    return SimpleNamespace(
        id=response_id,
        output=output,
        usage=usage,
        output_text=output_text,
    )


def _make_computer_call(
    actions: list[Any],
    call_id: str = "call_123",
    pending_safety_checks: list[Any] | None = None,
) -> SimpleNamespace:
    """Create a mock computer_call output item."""
    return SimpleNamespace(
        type="computer_call",
        call_id=call_id,
        actions=actions,
        action=None,
        pending_safety_checks=pending_safety_checks or [],
    )


# ===========================================================================
# Action Normalization
# ===========================================================================

class TestOpenAICUNormalization:
    """Tests for _make_protocol()._normalize_action()."""

    def test_click_left(self) -> None:
        action = SimpleNamespace(type="click", x=500, y=300, button="left")
        cmd = _make_protocol()._normalize_action(action)
        assert cmd == {"action": "click", "x": 500, "y": 300, "button": "left"}

    def test_click_right(self) -> None:
        action = SimpleNamespace(type="click", x=100, y=200, button="right")
        cmd = _make_protocol()._normalize_action(action)
        assert cmd == {"action": "click", "x": 100, "y": 200, "button": "right"}

    def test_click_wheel_mapped_to_middle(self) -> None:
        action = SimpleNamespace(type="click", x=300, y=400, button="wheel")
        cmd = _make_protocol()._normalize_action(action)
        assert cmd == {"action": "click", "x": 300, "y": 400, "button": "middle"}

    def test_click_default_button(self) -> None:
        """Click with no button attribute defaults to left."""
        action = SimpleNamespace(type="click", x=500, y=300)
        cmd = _make_protocol()._normalize_action(action)
        assert cmd is not None
        assert cmd["button"] == "left"

    def test_click_none_button_defaults_to_left(self) -> None:
        action = SimpleNamespace(type="click", x=500, y=300, button=None)
        cmd = _make_protocol()._normalize_action(action)
        assert cmd is not None
        assert cmd["button"] == "left"

    def test_double_click(self) -> None:
        action = SimpleNamespace(type="double_click", x=640, y=360)
        cmd = _make_protocol()._normalize_action(action)
        assert cmd == {"action": "double_click", "x": 640, "y": 360}

    def test_scroll_down(self) -> None:
        action = SimpleNamespace(
            type="scroll", x=640, y=360, scroll_x=0, scroll_y=360,
        )
        cmd = _make_protocol()._normalize_action(action)
        assert cmd == {
            "action": "scroll", "x": 640, "y": 360,
            "direction": "down", "amount": 3,
        }

    def test_scroll_up(self) -> None:
        action = SimpleNamespace(
            type="scroll", x=640, y=360, scroll_x=0, scroll_y=-600,
        )
        cmd = _make_protocol()._normalize_action(action)
        assert cmd == {
            "action": "scroll", "x": 640, "y": 360,
            "direction": "up", "amount": 5,
        }

    def test_scroll_large_pixel_value(self) -> None:
        """Large pixel scroll_y (e.g. 467) converts to reasonable notches."""
        action = SimpleNamespace(
            type="scroll", x=1014, y=685, scroll_x=0, scroll_y=467,
        )
        cmd = _make_protocol()._normalize_action(action)
        assert cmd is not None
        assert cmd["direction"] == "down"
        # 720/6=120 px/notch → round(467/120) = 4
        assert cmd["amount"] == 4

    def test_scroll_small_pixel_value_clamps_to_1(self) -> None:
        """Very small scroll_y still produces at least 1 notch."""
        action = SimpleNamespace(
            type="scroll", x=640, y=360, scroll_x=0, scroll_y=10,
        )
        cmd = _make_protocol()._normalize_action(action)
        assert cmd is not None
        assert cmd["direction"] == "down"
        assert cmd["amount"] == 1

    def test_scroll_caps_at_10(self) -> None:
        """Extremely large scroll_y is capped at 10 notches."""
        action = SimpleNamespace(
            type="scroll", x=640, y=360, scroll_x=0, scroll_y=5000,
        )
        cmd = _make_protocol()._normalize_action(action)
        assert cmd is not None
        assert cmd["amount"] == 10

    def test_scroll_horizontal_falls_back_to_vertical(self) -> None:
        """Horizontal scroll unsupported by executor; falls back to down/3."""
        action = SimpleNamespace(
            type="scroll", x=640, y=360, scroll_x=4, scroll_y=0,
        )
        cmd = _make_protocol()._normalize_action(action)
        assert cmd == {
            "action": "scroll", "x": 640, "y": 360,
            "direction": "down", "amount": 3,
        }

    def test_scroll_horizontal_left_falls_back(self) -> None:
        """Horizontal scroll left also falls back to down/3."""
        action = SimpleNamespace(
            type="scroll", x=640, y=360, scroll_x=-2, scroll_y=0,
        )
        cmd = _make_protocol()._normalize_action(action)
        assert cmd == {
            "action": "scroll", "x": 640, "y": 360,
            "direction": "down", "amount": 3,
        }

    def test_scroll_defaults_to_down_3(self) -> None:
        """Both scroll axes zero → defaults to down with amount 3."""
        action = SimpleNamespace(
            type="scroll", x=640, y=360, scroll_x=0, scroll_y=0,
        )
        cmd = _make_protocol()._normalize_action(action)
        assert cmd is not None
        assert cmd["direction"] == "down"
        assert cmd["amount"] == 3

    def test_type(self) -> None:
        action = SimpleNamespace(type="type", text="hello world")
        cmd = _make_protocol()._normalize_action(action)
        assert cmd == {"action": "type", "text": "hello world"}

    def test_keypress_single(self) -> None:
        action = SimpleNamespace(type="keypress", keys=["Enter"])
        cmd = _make_protocol()._normalize_action(action)
        assert cmd == {"action": "key", "keys": ["enter"]}

    def test_keypress_combo(self) -> None:
        action = SimpleNamespace(type="keypress", keys=["Control", "c"])
        cmd = _make_protocol()._normalize_action(action)
        assert cmd == {"action": "key", "keys": ["ctrl", "c"]}

    def test_keypress_arrow_key_mapping(self) -> None:
        action = SimpleNamespace(type="keypress", keys=["ArrowDown"])
        cmd = _make_protocol()._normalize_action(action)
        assert cmd == {"action": "key", "keys": ["down"]}

    def test_keypress_space(self) -> None:
        action = SimpleNamespace(type="keypress", keys=["SPACE"])
        cmd = _make_protocol()._normalize_action(action)
        assert cmd == {"action": "key", "keys": ["space"]}

    def test_keypress_unmapped_passthrough(self) -> None:
        """Unmapped keys should pass through lowercased."""
        action = SimpleNamespace(type="keypress", keys=["a"])
        cmd = _make_protocol()._normalize_action(action)
        assert cmd == {"action": "key", "keys": ["a"]}

    def test_wait(self) -> None:
        action = SimpleNamespace(type="wait")
        cmd = _make_protocol()._normalize_action(action)
        assert cmd == {"action": "wait", "duration_s": 1.0}

    def test_wait_with_ms(self) -> None:
        action = SimpleNamespace(type="wait", ms=3000)
        cmd = _make_protocol()._normalize_action(action)
        assert cmd == {"action": "wait", "duration_s": 3.0}

    def test_wait_with_duration_ms(self) -> None:
        action = SimpleNamespace(type="wait", duration_ms=500)
        cmd = _make_protocol()._normalize_action(action)
        assert cmd == {"action": "wait", "duration_s": 0.5}

    def test_screenshot_returns_none(self) -> None:
        action = SimpleNamespace(type="screenshot")
        cmd = _make_protocol()._normalize_action(action)
        assert cmd is None

    def test_drag(self) -> None:
        path = [
            SimpleNamespace(x=100, y=100),
            SimpleNamespace(x=300, y=200),
            SimpleNamespace(x=500, y=500),
        ]
        action = SimpleNamespace(type="drag", path=path)
        cmd = _make_protocol()._normalize_action(action)
        assert cmd == {
            "action": "drag",
            "from_x": 100, "from_y": 100,
            "to_x": 500, "to_y": 500,
        }

    def test_drag_insufficient_path_returns_none(self) -> None:
        action = SimpleNamespace(type="drag", path=[SimpleNamespace(x=1, y=2)])
        cmd = _make_protocol()._normalize_action(action)
        assert cmd is None

    def test_move(self) -> None:
        action = SimpleNamespace(type="move", x=300, y=400)
        cmd = _make_protocol()._normalize_action(action)
        assert cmd == {"action": "mouse_move", "x": 300, "y": 400}

    def test_unknown_action_returns_none(self) -> None:
        action = SimpleNamespace(type="fly_to_moon")
        cmd = _make_protocol()._normalize_action(action)
        assert cmd is None

    def test_no_type_attribute_returns_none(self) -> None:
        action = SimpleNamespace(x=100, y=200)
        cmd = _make_protocol()._normalize_action(action)
        assert cmd is None

    def test_click_missing_coords_returns_none(self) -> None:
        action = SimpleNamespace(type="click", button="left")
        cmd = _make_protocol()._normalize_action(action)
        assert cmd is None

    def test_double_click_missing_coords_returns_none(self) -> None:
        action = SimpleNamespace(type="double_click")
        cmd = _make_protocol()._normalize_action(action)
        assert cmd is None

    def test_scroll_missing_coords_returns_none(self) -> None:
        action = SimpleNamespace(type="scroll", scroll_x=0, scroll_y=3)
        cmd = _make_protocol()._normalize_action(action)
        assert cmd is None

    def test_move_missing_coords_returns_none(self) -> None:
        action = SimpleNamespace(type="move")
        cmd = _make_protocol()._normalize_action(action)
        assert cmd is None

    def test_drag_missing_path_coords_returns_none(self) -> None:
        path = [SimpleNamespace(x=100), SimpleNamespace(x=500, y=500)]
        action = SimpleNamespace(type="drag", path=path)
        cmd = _make_protocol()._normalize_action(action)
        assert cmd is None


# ===========================================================================
# Dict-format Actions (OpenAI SDK returns dicts, not objects)
# ===========================================================================

class TestOpenAICUDictNormalization:
    """Verify _normalize_action works when actions are plain dicts."""

    def test_click_dict(self) -> None:
        cmd = _make_protocol()._normalize_action(
            {"type": "click", "x": 500, "y": 300, "button": "left"},
        )
        assert cmd == {"action": "click", "x": 500, "y": 300, "button": "left"}

    def test_double_click_dict(self) -> None:
        cmd = _make_protocol()._normalize_action(
            {"type": "double_click", "x": 640, "y": 360},
        )
        assert cmd == {"action": "double_click", "x": 640, "y": 360}

    def test_scroll_dict(self) -> None:
        cmd = _make_protocol()._normalize_action(
            {"type": "scroll", "x": 640, "y": 360, "scroll_x": 0, "scroll_y": -360},
        )
        assert cmd == {
            "action": "scroll", "x": 640, "y": 360,
            "direction": "up", "amount": 3,
        }

    def test_scroll_dict_camel_case(self) -> None:
        """API may return camelCase scrollX/scrollY instead of snake_case."""
        cmd = _make_protocol()._normalize_action(
            {"type": "scroll", "x": 640, "y": 360, "scrollX": 0, "scrollY": 600},
        )
        assert cmd == {
            "action": "scroll", "x": 640, "y": 360,
            "direction": "down", "amount": 5,
        }

    def test_scroll_dict_zero_snake_case_not_overridden_by_camel(self) -> None:
        """scroll_y=0 (present, falsy) should NOT fall through to scrollY."""
        cmd = _make_protocol()._normalize_action(
            {"type": "scroll", "x": 100, "y": 200,
             "scroll_x": 0, "scroll_y": 0, "scrollX": 5, "scrollY": 10},
        )
        # snake_case 0 is present → use it (both zero → default down/3)
        assert cmd == {
            "action": "scroll", "x": 100, "y": 200,
            "direction": "down", "amount": 3,
        }

    def test_scroll_dict_camel_case_horizontal_falls_back(self) -> None:
        """Horizontal scroll unsupported; falls back to down/3."""
        cmd = _make_protocol()._normalize_action(
            {"type": "scroll", "x": 100, "y": 200, "scrollX": -4, "scrollY": 0},
        )
        assert cmd == {
            "action": "scroll", "x": 100, "y": 200,
            "direction": "down", "amount": 3,
        }

    def test_type_dict(self) -> None:
        cmd = _make_protocol()._normalize_action(
            {"type": "type", "text": "hello"},
        )
        assert cmd == {"action": "type", "text": "hello"}

    def test_keypress_dict(self) -> None:
        cmd = _make_protocol()._normalize_action(
            {"type": "keypress", "keys": ["Control", "c"]},
        )
        assert cmd == {"action": "key", "keys": ["ctrl", "c"]}

    def test_screenshot_dict(self) -> None:
        cmd = _make_protocol()._normalize_action({"type": "screenshot"})
        assert cmd is None

    def test_drag_dict(self) -> None:
        cmd = _make_protocol()._normalize_action(
            {"type": "drag", "path": [{"x": 100, "y": 100}, {"x": 500, "y": 500}]},
        )
        assert cmd == {
            "action": "drag",
            "from_x": 100, "from_y": 100,
            "to_x": 500, "to_y": 500,
        }

    def test_move_dict(self) -> None:
        cmd = _make_protocol()._normalize_action(
            {"type": "move", "x": 300, "y": 400},
        )
        assert cmd == {"action": "mouse_move", "x": 300, "y": 400}

    def test_wait_dict(self) -> None:
        cmd = _make_protocol()._normalize_action({"type": "wait"})
        assert cmd == {"action": "wait", "duration_s": 1.0}


# ===========================================================================
# Key Name Normalization
# ===========================================================================

class TestKeyNormalization:
    """Tests for _normalize_key() helper."""

    def test_control_to_ctrl(self) -> None:
        assert _normalize_key("Control") == "ctrl"

    def test_space_variants(self) -> None:
        assert _normalize_key("SPACE") == "space"
        assert _normalize_key("Space") == "space"

    def test_arrow_keys(self) -> None:
        assert _normalize_key("ArrowUp") == "up"
        assert _normalize_key("ArrowDown") == "down"
        assert _normalize_key("ArrowLeft") == "left"
        assert _normalize_key("ArrowRight") == "right"

    def test_passthrough(self) -> None:
        assert _normalize_key("a") == "a"
        assert _normalize_key("F1") == "f1"


# ===========================================================================
# Queue Behavior
# ===========================================================================

class TestOpenAICUQueue:
    """Tests for action queue behavior with batched responses."""

    def test_batched_actions_queued(self) -> None:
        proto = _make_protocol()
        actions = [
            SimpleNamespace(type="click", x=100, y=200, button="left"),
            SimpleNamespace(type="type", text="hello"),
            SimpleNamespace(type="keypress", keys=["Enter"]),
        ]
        computer_call = _make_computer_call(actions, call_id="call_1")
        response = _make_response(
            [computer_call], response_id="resp_1",
        )
        proto._client.responses.create.return_value = response

        result = proto.step("fake_b64", "test task")

        assert result.success
        assert result.command is not None
        assert result.command["action"] == "click"
        assert len(proto._action_queue) == 2
        assert result.input_tokens == 100

    def test_queued_steps_return_zero_tokens(self) -> None:
        proto = _make_protocol()
        proto._action_queue.append(
            SimpleNamespace(type="type", text="hello"),
        )
        proto._step_count = 1

        result = proto.step("fake_b64", "test task")

        assert result.success
        assert result.command == {"action": "type", "text": "hello"}
        assert result.input_tokens == 0
        assert result.output_tokens == 0
        assert result.latency_ms == 0

    def test_queue_drains_then_triggers_api(self) -> None:
        proto = _make_protocol()
        proto._action_queue.append(
            SimpleNamespace(type="type", text="hello"),
        )
        proto._step_count = 1
        proto._last_response_id = "resp_1"
        proto._last_call_id = "call_1"

        # First step: pops from queue, no API call
        result1 = proto.step("fake_b64", "test task")
        assert result1.command is not None
        assert result1.command["action"] == "type"
        proto._client.responses.create.assert_not_called()

        # Second step: queue empty, needs API call
        computer_call = _make_computer_call(
            [SimpleNamespace(type="click", x=500, y=300, button="left")],
            call_id="call_2",
        )
        response = _make_response(
            [computer_call], response_id="resp_2",
            input_tokens=200, output_tokens=60,
        )
        proto._client.responses.create.return_value = response

        result2 = proto.step("fake_b64", "test task")
        assert result2.command is not None
        assert result2.command["action"] == "click"
        proto._client.responses.create.assert_called_once()

    def test_error_clears_queue(self) -> None:
        proto = _make_protocol()
        proto._action_queue.extend([
            SimpleNamespace(type="type", text="hello"),
            SimpleNamespace(type="keypress", keys=["Enter"]),
        ])

        proto.report_result(False, "HID device error")

        assert len(proto._action_queue) == 0
        assert proto._last_exec_error == "HID device error"

    def test_success_does_not_clear_queue(self) -> None:
        proto = _make_protocol()
        proto._action_queue.extend([
            SimpleNamespace(type="type", text="hello"),
        ])

        proto.report_result(True)

        assert len(proto._action_queue) == 1
        assert proto._last_exec_error is None


# ===========================================================================
# Completion Detection
# ===========================================================================

class TestOpenAICUCompletion:
    """Tests for done detection."""

    def test_no_computer_call_means_done(self) -> None:
        proto = _make_protocol()
        text_item = SimpleNamespace(type="text", text="Task completed!")
        response = _make_response(
            [text_item], output_text="Task completed!",
        )
        proto._client.responses.create.return_value = response

        result = proto.step("fake_b64", "test task")

        assert result.is_done
        assert "completed" in result.done_reason.lower()
        assert result.success

    def test_done_text_captured_as_reason(self) -> None:
        proto = _make_protocol()
        response = _make_response(
            [], output_text="Opened settings and enabled dark mode.",
        )
        proto._client.responses.create.return_value = response

        result = proto.step("fake_b64", "test task")

        assert result.is_done
        assert "dark mode" in result.done_reason.lower()


# ===========================================================================
# Conversation State
# ===========================================================================

class TestOpenAICUState:
    """Tests for conversation state management."""

    def test_response_id_chained(self) -> None:
        proto = _make_protocol()
        computer_call = _make_computer_call(
            [SimpleNamespace(type="click", x=100, y=200, button="left")],
            call_id="call_1",
        )
        response = _make_response(
            [computer_call], response_id="resp_1",
        )
        proto._client.responses.create.return_value = response

        proto.step("fake_b64", "test task")

        assert proto._last_response_id == "resp_1"
        assert proto._last_call_id == "call_1"

    def test_continuation_uses_previous_response_id(self) -> None:
        proto = _make_protocol()
        proto._step_count = 1
        proto._last_response_id = "resp_1"
        proto._last_call_id = "call_1"

        computer_call = _make_computer_call(
            [SimpleNamespace(type="type", text="hi")],
            call_id="call_2",
        )
        response = _make_response(
            [computer_call], response_id="resp_2",
        )
        proto._client.responses.create.return_value = response

        proto.step("fake_b64", "test task")

        call_kwargs = proto._client.responses.create.call_args.kwargs
        assert call_kwargs["previous_response_id"] == "resp_1"

    def test_reset_clears_all(self) -> None:
        proto = _make_protocol()
        proto._last_response_id = "resp_1"
        proto._last_call_id = "call_1"
        proto._step_count = 5
        proto._action_queue.append(SimpleNamespace(type="type", text="x"))
        proto._pending_safety_checks = [SimpleNamespace(id="sc_1")]
        proto._last_exec_error = "error"

        proto.reset()

        assert proto._last_response_id is None
        assert proto._last_call_id is None
        assert proto._step_count == 0
        assert len(proto._action_queue) == 0
        assert len(proto._pending_safety_checks) == 0
        assert proto._last_exec_error is None

    def test_usage_summary(self) -> None:
        proto = _make_protocol()
        proto._total_input_tokens = 500
        proto._total_output_tokens = 150

        summary = proto.get_usage_summary()
        assert summary == {
            "total_input_tokens": 500,
            "total_output_tokens": 150,
            "total_cache_read_tokens": 0,
            "total_cache_creation_tokens": 0,
        }

    def test_messages_snapshot_returns_server_managed(self) -> None:
        proto = _make_protocol()
        proto._last_response_id = "resp_42"

        snapshot = proto.get_messages_snapshot()
        assert snapshot == [
            {"note": "server-managed", "response_id": "resp_42"},
        ]


# ===========================================================================
# Safety Checks
# ===========================================================================

class TestOpenAICUSafetyChecks:
    """Tests for auto-acknowledgment of pending safety checks."""

    def test_safety_checks_acknowledged_on_continuation(self) -> None:
        proto = _make_protocol()

        # First call returns safety checks
        safety = [
            SimpleNamespace(id="sc_1", code="some_code", message="review"),
        ]
        computer_call = _make_computer_call(
            [SimpleNamespace(type="click", x=100, y=200, button="left")],
            call_id="call_1",
            pending_safety_checks=safety,
        )
        response1 = _make_response(
            [computer_call], response_id="resp_1",
        )
        proto._client.responses.create.return_value = response1
        proto.step("fake_b64", "test task")

        assert len(proto._pending_safety_checks) == 1

        # Continuation call should acknowledge them
        computer_call2 = _make_computer_call(
            [SimpleNamespace(type="type", text="hello")],
            call_id="call_2",
        )
        response2 = _make_response(
            [computer_call2], response_id="resp_2",
        )
        proto._client.responses.create.return_value = response2
        proto.step("fake_b64", "test task")

        # Safety checks should be cleared
        assert proto._pending_safety_checks == []

        # Verify the continuation call included acknowledged_safety_checks
        call_kwargs = proto._client.responses.create.call_args.kwargs
        input_items = call_kwargs["input"]
        call_output = input_items[0]
        assert "acknowledged_safety_checks" in call_output
        assert call_output["acknowledged_safety_checks"][0]["id"] == "sc_1"


# ===========================================================================
# Error Handling
# ===========================================================================

class TestOpenAICUErrors:
    """Tests for API error handling."""

    def test_api_error_returns_failure(self) -> None:
        proto = _make_protocol()
        proto._client.responses.create.side_effect = _MockAPIError(
            "rate limit exceeded",
        )

        result = proto.step("fake_b64", "test task")

        assert not result.success
        assert "rate limit" in result.error

    def test_api_error_preserves_state(self) -> None:
        proto = _make_protocol()
        proto._step_count = 3
        proto._last_response_id = "resp_old"
        proto._client.responses.create.side_effect = _MockAPIError("fail")

        proto.step("fake_b64", "test task")

        # State should not be corrupted
        assert proto._step_count == 3
        assert proto._last_response_id == "resp_old"

    def test_report_result_success_clears_error(self) -> None:
        proto = _make_protocol()
        proto.report_result(False, "some error")
        assert proto._last_exec_error == "some error"

        proto.report_result(True)
        assert proto._last_exec_error is None

    def test_none_output_returns_error(self) -> None:
        """response.output being None should not crash."""
        proto = _make_protocol()
        response = SimpleNamespace(
            id="resp_1",
            output=None,
            usage=SimpleNamespace(
                input_tokens=100, output_tokens=10, total_tokens=110,
                input_tokens_details=SimpleNamespace(cached_tokens=0),
            ),
            output_text="",
        )
        proto._client.responses.create.return_value = response

        result = proto.step("fake_b64", "test task")

        assert not result.success
        assert "no output" in result.error.lower()

    def test_empty_actions_returns_error(self) -> None:
        proto = _make_protocol()
        computer_call = SimpleNamespace(
            type="computer_call",
            call_id="call_1",
            actions=[],
            action=None,
            pending_safety_checks=[],
        )
        response = _make_response([computer_call])
        proto._client.responses.create.return_value = response

        result = proto.step("fake_b64", "test task")

        assert not result.success
        assert "no actions" in result.error.lower()

    def test_exec_error_sent_to_model_on_continuation(self) -> None:
        """Execution errors should be included in the next API call."""
        proto = _make_protocol()
        proto._step_count = 1
        proto._last_response_id = "resp_1"
        proto._last_call_id = "call_1"
        proto.report_result(False, "Cannot type non-ASCII text via HID")

        computer_call = _make_computer_call(
            [SimpleNamespace(type="type", text="hello")],
            call_id="call_2",
        )
        response = _make_response([computer_call], response_id="resp_2")
        proto._client.responses.create.return_value = response

        proto.step("fake_b64", "test task")

        call_kwargs = proto._client.responses.create.call_args.kwargs
        input_items = call_kwargs["input"]
        assert len(input_items) == 2
        error_msg = input_items[1]
        assert error_msg["role"] == "user"
        assert "non-ASCII" in error_msg["content"]
        # Error should be cleared after being sent
        assert proto._last_exec_error is None

    def test_no_error_sends_only_screenshot(self) -> None:
        """Without an error, continuation should only send the screenshot."""
        proto = _make_protocol()
        proto._step_count = 1
        proto._last_response_id = "resp_1"
        proto._last_call_id = "call_1"

        computer_call = _make_computer_call(
            [SimpleNamespace(type="click", x=100, y=200, button="left")],
            call_id="call_2",
        )
        response = _make_response([computer_call], response_id="resp_2")
        proto._client.responses.create.return_value = response

        proto.step("fake_b64", "test task")

        call_kwargs = proto._client.responses.create.call_args.kwargs
        input_items = call_kwargs["input"]
        assert len(input_items) == 1  # only computer_call_output


# ===========================================================================
# Coordinate Validation
# ===========================================================================

class TestOpenAICUCoordinates:
    """Tests for coordinate bounds checking."""

    def test_valid_coordinates(self) -> None:
        proto = _make_protocol()
        cmd = {"action": "click", "x": 640, "y": 360}
        assert proto._validate_coordinates(cmd) is None

    def test_out_of_bounds_x(self) -> None:
        proto = _make_protocol()
        cmd = {"action": "click", "x": 1280, "y": 360}
        error = proto._validate_coordinates(cmd)
        assert error is not None
        assert "out of bounds" in error

    def test_negative_coordinate(self) -> None:
        proto = _make_protocol()
        cmd = {"action": "click", "x": -1, "y": 360}
        error = proto._validate_coordinates(cmd)
        assert error is not None

    def test_oob_clears_queue_from_api_response(self) -> None:
        """Out-of-bounds on first action should clear queued actions."""
        proto = _make_protocol()
        actions = [
            SimpleNamespace(type="click", x=9999, y=9999, button="left"),
            SimpleNamespace(type="type", text="should not run"),
        ]
        computer_call = _make_computer_call(actions, call_id="call_1")
        response = _make_response([computer_call])
        proto._client.responses.create.return_value = response

        result = proto.step("fake_b64", "test task")

        assert not result.success
        assert "out of bounds" in result.error
        assert len(proto._action_queue) == 0


# ===========================================================================
# Extract Actions (single vs batched)
# ===========================================================================

class TestExtractActions:
    """Tests for _extract_actions() handling both action formats."""

    def test_prefers_actions_list(self) -> None:
        actions = [SimpleNamespace(type="click", x=1, y=2, button="left")]
        call = SimpleNamespace(
            actions=actions,
            action=SimpleNamespace(type="type", text="ignored"),
        )
        result = OpenAICUProtocol._extract_actions(call)
        assert len(result) == 1
        assert result[0].type == "click"

    def test_falls_back_to_single_action(self) -> None:
        call = SimpleNamespace(
            actions=None,
            action=SimpleNamespace(type="type", text="hello"),
        )
        result = OpenAICUProtocol._extract_actions(call)
        assert len(result) == 1
        assert result[0].type == "type"

    def test_returns_empty_for_no_actions(self) -> None:
        call = SimpleNamespace(actions=None, action=None)
        result = OpenAICUProtocol._extract_actions(call)
        assert result == []


# ===========================================================================
# detect_os()
# ===========================================================================

class TestOpenAICUDetectOS:
    """Tests for OS detection via one-off API call."""

    def test_detects_macos(self) -> None:
        proto = _make_protocol()
        response = SimpleNamespace(output_text="macos")
        proto._client.responses.create.return_value = response

        assert proto.detect_os("fake_b64") == "macos"

    def test_detects_windows(self) -> None:
        proto = _make_protocol()
        response = SimpleNamespace(output_text="  Windows  ")
        proto._client.responses.create.return_value = response

        assert proto.detect_os("fake_b64") == "windows"

    def test_unexpected_os_returns_none(self) -> None:
        proto = _make_protocol()
        response = SimpleNamespace(output_text="ChromeOS")
        proto._client.responses.create.return_value = response

        assert proto.detect_os("fake_b64") is None

    def test_api_error_returns_none(self) -> None:
        proto = _make_protocol()
        proto._client.responses.create.side_effect = Exception("API down")

        assert proto.detect_os("fake_b64") is None


# ===========================================================================
# Screenshot needs_screenshot flag
# ===========================================================================

class TestScreenshotAction:
    """Tests for the screenshot action → needs_screenshot flag."""

    def test_screenshot_from_api_sets_flag(self) -> None:
        proto = _make_protocol()
        computer_call = _make_computer_call(
            [SimpleNamespace(type="screenshot")],
            call_id="call_1",
        )
        response = _make_response([computer_call])
        proto._client.responses.create.return_value = response

        result = proto.step("fake_b64", "test task")

        assert result.success
        assert result.command is None
        assert result.needs_screenshot is True

    def test_screenshot_from_queue_sets_flag(self) -> None:
        proto = _make_protocol()
        proto._action_queue.append(SimpleNamespace(type="screenshot"))
        proto._step_count = 1

        result = proto.step("fake_b64", "test task")

        assert result.success
        assert result.command is None
        assert result.needs_screenshot is True


# ===========================================================================
# Cache Metrics
# ===========================================================================

class TestOpenAICUCacheMetrics:
    """Tests for OpenAI automatic caching metrics."""

    def test_cache_metrics_extracted_from_response(self) -> None:
        """Cached tokens from input_tokens_details are in StepResult."""
        proto = _make_protocol()
        computer_call = _make_computer_call(
            [SimpleNamespace(type="click", x=100, y=200, button="left")],
            call_id="call_1",
        )
        response = _make_response(
            [computer_call], response_id="resp_1",
            input_tokens=1000, output_tokens=50, cached_tokens=600,
        )
        proto._client.responses.create.return_value = response

        result = proto.step("fake_b64", "test task")

        assert result.success
        assert result.input_tokens == 1000
        assert result.cache_read_tokens == 600
        assert result.cache_creation_tokens == 0
        assert proto._total_cache_read_tokens == 600

    def test_cache_metrics_accumulate(self) -> None:
        """Cache read tokens accumulate across multiple steps."""
        proto = _make_protocol()

        for i, cached in enumerate([0, 400, 800]):
            computer_call = _make_computer_call(
                [SimpleNamespace(type="click", x=100, y=200, button="left")],
                call_id=f"call_{i}",
            )
            response = _make_response(
                [computer_call], response_id=f"resp_{i}",
                input_tokens=1000, output_tokens=50, cached_tokens=cached,
            )
            proto._client.responses.create.return_value = response
            proto.step("fake_b64", "test task")
            proto.report_result(True)

        assert proto._total_cache_read_tokens == 1200

    def test_cache_metrics_zero_when_no_details(self) -> None:
        """Missing input_tokens_details gracefully defaults to 0."""
        proto = _make_protocol()
        computer_call = _make_computer_call(
            [SimpleNamespace(type="click", x=100, y=200, button="left")],
            call_id="call_1",
        )
        # Build response with no input_tokens_details
        response = SimpleNamespace(
            id="resp_1",
            output=[computer_call],
            usage=SimpleNamespace(
                input_tokens=500, output_tokens=30,
                total_tokens=530,
            ),
            output_text="",
        )
        proto._client.responses.create.return_value = response

        result = proto.step("fake_b64", "test task")

        assert result.success
        assert result.cache_read_tokens == 0
        assert proto._total_cache_read_tokens == 0

    def test_get_usage_summary_includes_cache_keys(self) -> None:
        """get_usage_summary() includes all 4 expected keys."""
        proto = _make_protocol()
        proto._total_input_tokens = 2000
        proto._total_output_tokens = 300
        proto._total_cache_read_tokens = 1500

        summary = proto.get_usage_summary()

        assert summary == {
            "total_input_tokens": 2000,
            "total_output_tokens": 300,
            "total_cache_read_tokens": 1500,
            "total_cache_creation_tokens": 0,
        }
