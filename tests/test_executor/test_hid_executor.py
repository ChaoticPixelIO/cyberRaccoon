"""Tests for M4 HID executor — command routing, deduplication, error handling."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from cyberraccoon.executor.hid_executor import ActionExecutor
from tests.test_executor.conftest import MockHIDDevice


class MockActionExecutor(ActionExecutor):
    """ActionExecutor with mock HID device for testing.

    Overrides open() to inject a MockHIDDevice instead of real /dev/hidg0.
    Keyboard and mouse share the same device (using Report IDs).
    """

    def __init__(self, target_os: str | None = None) -> None:
        super().__init__(
            target_os=target_os, screen_width=1920, screen_height=1080,
        )
        self._mock_dev = MockHIDDevice("/dev/hidg0")

    def open(self) -> None:
        from cyberraccoon.executor.mouse import MouseController
        from cyberraccoon.executor.keyboard import KeyboardController

        self._mock_dev.open()

        self._hid_dev = self._mock_dev  # type: ignore[assignment]
        self._keyboard = KeyboardController(self._mock_dev)  # type: ignore[arg-type]
        self._mouse = MouseController(  # type: ignore[arg-type]
            self._mock_dev,
            screen_width=self._screen_width,
            screen_height=self._screen_height,
        )


@pytest.fixture
def executor() -> MockActionExecutor:
    ex = MockActionExecutor()
    ex.open()
    return ex


class TestCommandRouting:
    """Tests for action-to-controller routing."""

    def test_click_routes_to_mouse(self, executor: MockActionExecutor) -> None:
        from cyberraccoon.executor.mouse import REPORT_ID as MOUSE_ID
        result = executor.execute(
            {"id": "t1", "action": "click", "x": 640, "y": 360}
        )
        assert result["status"] == "ok"
        assert len(executor._mock_dev.reports) > 0
        assert all(r[0] == MOUSE_ID for r in executor._mock_dev.reports)

    def test_double_click_routes_to_mouse(self, executor: MockActionExecutor) -> None:
        result = executor.execute(
            {"id": "t2", "action": "double_click", "x": 100, "y": 200}
        )
        assert result["status"] == "ok"
        assert len(executor._mock_dev.reports) > 0

    def test_type_routes_to_keyboard(self, executor: MockActionExecutor) -> None:
        from cyberraccoon.executor.keyboard import REPORT_ID as KB_ID
        result = executor.execute(
            {"id": "t3", "action": "type", "text": "hi"}
        )
        assert result["status"] == "ok"
        assert len(executor._mock_dev.reports) > 0
        assert all(r[0] == KB_ID for r in executor._mock_dev.reports)

    def test_key_routes_to_keyboard(self, executor: MockActionExecutor) -> None:
        result = executor.execute(
            {"id": "t4", "action": "key", "keys": ["ctrl", "c"]}
        )
        assert result["status"] == "ok"
        assert len(executor._mock_dev.reports) > 0

    def test_scroll_routes_to_mouse(self, executor: MockActionExecutor) -> None:
        result = executor.execute(
            {"id": "t5", "action": "scroll", "x": 640, "y": 360,
             "direction": "down", "amount": 3}
        )
        assert result["status"] == "ok"
        assert len(executor._mock_dev.reports) > 0

    def test_drag_routes_to_mouse(self, executor: MockActionExecutor) -> None:
        result = executor.execute(
            {"id": "t6", "action": "drag",
             "from_x": 100, "from_y": 200, "to_x": 500, "to_y": 300}
        )
        assert result["status"] == "ok"
        assert len(executor._mock_dev.reports) > 0

    def test_unknown_action_returns_error(self, executor: MockActionExecutor) -> None:
        result = executor.execute(
            {"id": "t7", "action": "fly", "x": 0, "y": 0}
        )
        assert result["status"] == "error"
        assert "Unknown action" in result["error"]


class TestDoneAction:
    """Tests for the special 'done' action."""

    def test_done_returns_ok(self, executor: MockActionExecutor) -> None:
        result = executor.execute({"id": "d1", "action": "done"})
        assert result["status"] == "ok"
        assert result["action"] == "done"
        assert result["duration_ms"] == 0

    def test_done_not_tracked(self, executor: MockActionExecutor) -> None:
        """done should not be added to _executed_ids, so same ID can be reused."""
        executor.execute({"id": "d2", "action": "done"})
        # Use same ID with a real action — should NOT be skipped
        result = executor.execute(
            {"id": "d2", "action": "click", "x": 640, "y": 360}
        )
        assert result["status"] == "ok"


class TestDeduplication:
    """Tests for command ID deduplication."""

    def test_duplicate_id_returns_skipped(self, executor: MockActionExecutor) -> None:
        executor.execute({"id": "dup1", "action": "click", "x": 100, "y": 100})
        result = executor.execute(
            {"id": "dup1", "action": "click", "x": 100, "y": 100}
        )
        assert result["status"] == "skipped"
        assert "duplicate" in result["reason"]

    def test_different_ids_both_execute(self, executor: MockActionExecutor) -> None:
        r1 = executor.execute({"id": "a1", "action": "click", "x": 100, "y": 100})
        r2 = executor.execute({"id": "a2", "action": "click", "x": 200, "y": 200})
        assert r1["status"] == "ok"
        assert r2["status"] == "ok"

    def test_empty_id_never_skipped(self, executor: MockActionExecutor) -> None:
        """Commands with no ID should always execute."""
        r1 = executor.execute({"action": "click", "x": 100, "y": 100})
        r2 = executor.execute({"action": "click", "x": 100, "y": 100})
        assert r1["status"] == "ok"
        assert r2["status"] == "ok"

    def test_cache_overflow_evicts_oldest(self, executor: MockActionExecutor) -> None:
        from collections import deque
        # Replace with a tiny deque to trigger eviction quickly
        executor._executed_ids = deque(maxlen=5)
        # Fill exactly to capacity
        for i in range(5):
            executor.execute(
                {"id": f"fill_{i}", "action": "click", "x": 100, "y": 100}
            )
        # Add one more to evict fill_0
        executor.execute({"id": "fill_5", "action": "click", "x": 100, "y": 100})
        # fill_0 was evicted — it should be allowed to execute again
        result = executor.execute(
            {"id": "fill_0", "action": "click", "x": 100, "y": 100}
        )
        assert result["status"] == "ok"


class TestResultFormat:
    """Tests for result dict structure."""

    def test_ok_result_has_duration(self, executor: MockActionExecutor) -> None:
        result = executor.execute(
            {"id": "r1", "action": "click", "x": 100, "y": 100}
        )
        assert "duration_ms" in result
        assert isinstance(result["duration_ms"], int)
        assert result["duration_ms"] >= 0

    def test_error_result_has_error_field(self, executor: MockActionExecutor) -> None:
        result = executor.execute({"id": "r2", "action": "unknown"})
        assert result["status"] == "error"
        assert "error" in result


class TestWaitAction:
    """Tests for the wait action dispatch."""

    @patch("cyberraccoon.executor.base_executor.time.sleep")
    def test_wait_sleeps_for_duration(
        self, mock_sleep: object, executor: MockActionExecutor
    ) -> None:
        result = executor.execute(
            {"id": "w1", "action": "wait", "duration_s": 3.0}
        )
        assert result["status"] == "ok"
        assert result["action"] == "wait"
        mock_sleep.assert_called_once_with(3.0)

    @patch("cyberraccoon.executor.base_executor.time.sleep")
    def test_wait_duration_capped_at_10(
        self, mock_sleep: object, executor: MockActionExecutor
    ) -> None:
        executor.execute(
            {"id": "w2", "action": "wait", "duration_s": 30.0}
        )
        mock_sleep.assert_called_once_with(10.0)

    @patch("cyberraccoon.executor.base_executor.time.sleep")
    def test_wait_default_duration(
        self, mock_sleep: object, executor: MockActionExecutor
    ) -> None:
        """No duration_s field should default to 1.0."""
        executor.execute({"id": "w3", "action": "wait"})
        mock_sleep.assert_called_once_with(1.0)

    def test_wait_no_hid_reports(self, executor: MockActionExecutor) -> None:
        """Wait action should not send any HID reports."""
        executor.execute(
            {"id": "w4", "action": "wait", "duration_s": 1.0}
        )
        assert len(executor._mock_dev.reports) == 0


class TestNonAsciiRejection:
    """Tests for non-ASCII text rejection with helpful error messages."""

    @pytest.fixture
    def win_executor(self) -> MockActionExecutor:
        ex = MockActionExecutor(target_os="windows")
        ex.open()
        return ex

    @pytest.fixture
    def mac_executor(self) -> MockActionExecutor:
        ex = MockActionExecutor(target_os="macos")
        ex.open()
        return ex

    @pytest.fixture
    def linux_executor(self) -> MockActionExecutor:
        ex = MockActionExecutor(target_os="linux")
        ex.open()
        return ex

    def test_ascii_with_target_os_types_normally(self, win_executor: MockActionExecutor) -> None:
        """ASCII text should use normal HID typing regardless of target_os."""
        result = win_executor.execute(
            {"id": "cb1", "action": "type", "text": "hello"}
        )
        assert result["status"] == "ok"
        reports = win_executor._mock_dev.reports
        assert len(reports) == 10  # 5 chars × (press + release)

    def test_non_ascii_returns_error(self, win_executor: MockActionExecutor) -> None:
        """Non-ASCII text should return an error with base64 and OS hint."""
        result = win_executor.execute(
            {"id": "cb2", "action": "type", "text": "你好"}
        )
        assert result["status"] == "error"
        assert "Cannot type non-ASCII" in result["error"]
        assert "base64 command" in result["error"].lower() or "base64" in result["error"]
        # No HID reports sent
        assert len(win_executor._mock_dev.reports) == 0

    def test_mixed_text_returns_error(self, win_executor: MockActionExecutor) -> None:
        """Mixed ASCII + non-ASCII text should also return error."""
        result = win_executor.execute(
            {"id": "cb3", "action": "type", "text": "Hi 你好"}
        )
        assert result["status"] == "error"
        assert "Cannot type non-ASCII" in result["error"]

    def test_error_includes_windows_hint(self, win_executor: MockActionExecutor) -> None:
        """Windows target should include PowerShell clipboard command."""
        result = win_executor.execute(
            {"id": "cb4", "action": "type", "text": "你好"}
        )
        assert "powershell" in result["error"]
        assert "Set-Clipboard" in result["error"]
        assert "Ctrl+V" in result["error"]

    def test_error_includes_macos_hint(self, mac_executor: MockActionExecutor) -> None:
        """macOS target should include pbcopy command."""
        result = mac_executor.execute(
            {"id": "cb5", "action": "type", "text": "你好"}
        )
        assert "pbcopy" in result["error"]
        assert "base64 -D" in result["error"]
        assert "Cmd+V" in result["error"]

    def test_error_includes_linux_hint(self, linux_executor: MockActionExecutor) -> None:
        """Linux target should include xclip command."""
        result = linux_executor.execute(
            {"id": "cb6", "action": "type", "text": "你好"}
        )
        assert "xclip" in result["error"]
        assert "base64 -d" in result["error"]
        assert "Ctrl+V" in result["error"]

    def test_error_without_target_os(self) -> None:
        """Without target_os, error should still include base64 but no OS hint."""
        ex = MockActionExecutor(target_os=None)
        ex.open()
        result = ex.execute(
            {"id": "cb7", "action": "type", "text": "你好"}
        )
        assert result["status"] == "error"
        assert "Cannot type non-ASCII" in result["error"]
        assert "base64 command" in result["error"].lower() or "base64" in result["error"]
        # No OS-specific hint
        assert "pbcopy" not in result["error"]
        assert "powershell" not in result["error"]
        assert "xclip" not in result["error"]

    def test_error_base64_is_decodable(self, win_executor: MockActionExecutor) -> None:
        """The base64 in the error message should decode back to the original text."""
        import base64 as b64mod
        import re
        text = "你好世界"
        result = win_executor.execute(
            {"id": "cb8", "action": "type", "text": text}
        )
        # Extract base64 from the OS hint command (e.g., FromBase64String('XXXX'))
        error = result["error"]
        match = re.search(r"FromBase64String\('([A-Za-z0-9+/=]+)'\)", error)
        assert match is not None, f"No base64 found in error: {error}"
        decoded = b64mod.b64decode(match.group(1)).decode("utf-8")
        assert decoded == text
