"""Tests for the shared JSON parsing utility."""

from __future__ import annotations

import pytest

from agent.protocols.parsing import parse_json_action, parse_json_actions, try_parse_json

VALID = {"left_click", "type", "key", "done", "scroll", "screenshot"}


class TestTryParseJson:
    """Tests for try_parse_json()."""

    def test_valid_action(self) -> None:
        result = try_parse_json('{"action": "left_click", "coordinate": [1, 2]}', VALID)
        assert result == {"action": "left_click", "coordinate": [1, 2]}

    def test_invalid_action(self) -> None:
        result = try_parse_json('{"action": "fly"}', VALID)
        assert result is None

    def test_not_json(self) -> None:
        result = try_parse_json("hello world", VALID)
        assert result is None

    def test_not_dict(self) -> None:
        result = try_parse_json("[1, 2, 3]", VALID)
        assert result is None

    def test_missing_action(self) -> None:
        result = try_parse_json('{"x": 100}', VALID)
        assert result is None


class TestParseJsonAction:
    """Tests for parse_json_action() 4-level fallback."""

    def test_level1_direct_parse(self) -> None:
        raw = '{"action": "done", "reason": "complete"}'
        result = parse_json_action(raw, VALID)
        assert result is not None
        assert result["action"] == "done"

    def test_level2_markdown_block(self) -> None:
        raw = 'Here is my action:\n```json\n{"action": "type", "text": "hi"}\n```'
        result = parse_json_action(raw, VALID)
        assert result is not None
        assert result["action"] == "type"

    def test_level3_shallow_brace(self) -> None:
        raw = 'I will click. {"action": "left_click", "coordinate": [5, 5]} done.'
        result = parse_json_action(raw, VALID)
        assert result is not None
        assert result["action"] == "left_click"

    def test_level4_greedy(self) -> None:
        raw = 'Analysis:\n{"action": "scroll",\n"scroll_direction": "down"}'
        result = parse_json_action(raw, VALID)
        assert result is not None
        assert result["action"] == "scroll"

    def test_no_valid_json(self) -> None:
        raw = "I cannot find any action to perform."
        result = parse_json_action(raw, VALID)
        assert result is None

    def test_whitespace_stripped(self) -> None:
        raw = '  \n  {"action": "screenshot"}  \n  '
        result = parse_json_action(raw, VALID)
        assert result is not None
        assert result["action"] == "screenshot"


class TestParseJsonActions:
    """Tests for parse_json_actions() array support."""

    def test_json_array(self) -> None:
        raw = '[{"action": "left_click", "coordinate": [500, 300]}, {"action": "type", "text": "hi"}]'
        result = parse_json_actions(raw, VALID)
        assert result is not None
        assert len(result) == 2
        assert result[0]["action"] == "left_click"
        assert result[1]["action"] == "type"

    def test_array_in_markdown_code_block(self) -> None:
        raw = 'Here are actions:\n```json\n[{"action": "left_click", "coordinate": [1, 2]}, {"action": "done", "reason": "ok"}]\n```'
        result = parse_json_actions(raw, VALID)
        assert result is not None
        assert len(result) == 2

    def test_single_object_wrapped_in_list(self) -> None:
        raw = '{"action": "done", "reason": "complete"}'
        result = parse_json_actions(raw, VALID)
        assert result is not None
        assert len(result) == 1
        assert result[0]["action"] == "done"

    def test_empty_array_returns_none(self) -> None:
        raw = "[]"
        result = parse_json_actions(raw, VALID)
        assert result is None

    def test_array_with_invalid_action_falls_back(self) -> None:
        """Array with invalid action fails array parse, falls back to single-object."""
        raw = '[{"action": "left_click", "coordinate": [1, 2]}, {"action": "fly"}]'
        result = parse_json_actions(raw, VALID)
        # Array parse fails, but Level 3 extracts the first valid {..} object
        assert result is not None
        assert len(result) == 1
        assert result[0]["action"] == "left_click"

    def test_array_with_non_dict_falls_back(self) -> None:
        """Array with non-dict element fails array parse, falls back to single-object."""
        raw = '[{"action": "left_click"}, 42]'
        result = parse_json_actions(raw, VALID)
        # Array parse fails, Level 3 extracts the first valid {..}
        assert result is not None
        assert len(result) == 1
        assert result[0]["action"] == "left_click"

    def test_no_valid_json_returns_none(self) -> None:
        raw = "I cannot find any action to perform."
        result = parse_json_actions(raw, VALID)
        assert result is None

    def test_single_object_in_markdown(self) -> None:
        raw = '```json\n{"action": "type", "text": "hi"}\n```'
        result = parse_json_actions(raw, VALID)
        assert result is not None
        assert len(result) == 1
        assert result[0]["action"] == "type"

    def test_fallback_to_level3_brace(self) -> None:
        """Array parser falls back to single-object parser for embedded JSON."""
        raw = 'I will click. {"action": "left_click", "coordinate": [5, 5]} done.'
        result = parse_json_actions(raw, VALID)
        assert result is not None
        assert len(result) == 1
        assert result[0]["action"] == "left_click"
