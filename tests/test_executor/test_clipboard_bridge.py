"""Tests for clipboard bridge — non-ASCII text input via OS clipboard."""

from __future__ import annotations

import base64

import pytest

from executor.clipboard_bridge import (
    TargetOS,
    build_clipboard_sequence,
    has_non_typeable,
    split_text,
)


class TestHasNonTypeable:
    """Tests for has_non_typeable()."""

    def test_ascii_only(self) -> None:
        assert has_non_typeable("hello") is False

    def test_cjk(self) -> None:
        assert has_non_typeable("你好") is True

    def test_empty(self) -> None:
        assert has_non_typeable("") is False

    def test_symbols(self) -> None:
        assert has_non_typeable("!@#") is False

    def test_newline_tab(self) -> None:
        assert has_non_typeable("hello\nworld\t!") is False

    def test_mixed(self) -> None:
        assert has_non_typeable("Hi 你好") is True

    def test_emoji(self) -> None:
        assert has_non_typeable("Hello 🌍") is True

    def test_accented(self) -> None:
        assert has_non_typeable("café") is True


class TestSplitText:
    """Tests for split_text()."""

    def test_pure_ascii(self) -> None:
        assert split_text("hello") == [("hello", False)]

    def test_pure_cjk(self) -> None:
        assert split_text("你好") == [("你好", True)]

    def test_mixed(self) -> None:
        result = split_text("Hello 你好 World")
        assert result == [
            ("Hello ", False),
            ("你好", True),
            (" World", False),
        ]

    def test_emoji(self) -> None:
        result = split_text("Hi🌍!")
        assert result == [
            ("Hi", False),
            ("🌍", True),
            ("!", False),
        ]

    def test_alternating(self) -> None:
        result = split_text("A你B好C")
        assert len(result) == 5
        assert result == [
            ("A", False),
            ("你", True),
            ("B", False),
            ("好", True),
            ("C", False),
        ]

    def test_empty(self) -> None:
        assert split_text("") == []

    def test_newline_is_typeable(self) -> None:
        result = split_text("a\nb")
        assert result == [("a\nb", False)]


class TestBuildClipboardSequenceWindows:
    """Tests for build_clipboard_sequence() with Windows target."""

    def test_short_text_sequence(self) -> None:
        steps = build_clipboard_sequence("你好", TargetOS.WINDOWS)
        # Short path: Win+R -> type ps -> Enter -> sleep -> Ctrl+V
        kinds = [s["kind"] for s in steps]
        assert kinds == ["keys", "sleep", "type", "keys", "sleep", "keys"]
        assert steps[0]["keys"] == ["win", "r"]
        assert steps[3]["keys"] == ["enter"]
        assert steps[5]["keys"] == ["ctrl", "v"]

    def test_short_text_b64_roundtrip(self) -> None:
        text = "你好世界"
        steps = build_clipboard_sequence(text, TargetOS.WINDOWS)
        # Extract base64 from the typed powershell command
        typed = steps[2]["text"]
        # Find the Base64 string between the quotes
        start = typed.index("('") + 2
        end = typed.index("')", start)
        b64_str = typed[start:end]
        decoded = base64.b64decode(b64_str).decode("utf-8")
        assert decoded == text

    def test_long_text_uses_powershell_window(self) -> None:
        # 40 CJK chars => ~160 bytes Base64 => exceeds 155-char limit
        text = "你" * 40
        steps = build_clipboard_sequence(text, TargetOS.WINDOWS)
        kinds = [s["kind"] for s in steps]
        # Long path should have "exit" typed
        type_steps = [s for s in steps if s["kind"] == "type"]
        type_texts = [s["text"] for s in type_steps]
        assert any("powershell" in t for t in type_texts)
        assert any("exit" in t for t in type_texts)
        assert steps[-1] == {"kind": "keys", "keys": ["ctrl", "v"]}

    def test_long_text_b64_roundtrip(self) -> None:
        text = "这是一段很长的中文文本，用于测试长路径的Base64编解码正确性。"
        steps = build_clipboard_sequence(text, TargetOS.WINDOWS)
        # Find the Set-Clipboard command step
        set_clipboard_step = None
        for s in steps:
            if s["kind"] == "type" and "Set-Clipboard" in s.get("text", ""):
                set_clipboard_step = s
                break
        assert set_clipboard_step is not None
        typed = set_clipboard_step["text"]
        start = typed.index("('") + 2
        end = typed.index("')", start)
        b64_str = typed[start:end]
        decoded = base64.b64decode(b64_str).decode("utf-8")
        assert decoded == text


class TestBuildClipboardSequenceMacOS:
    """Tests for build_clipboard_sequence() with macOS target."""

    def test_macos_sequence(self) -> None:
        steps = build_clipboard_sequence("你好", TargetOS.MACOS)
        kinds = [s["kind"] for s in steps]
        # Spotlight -> Terminal -> pbcopy -> Cmd+Tab -> Cmd+V
        assert kinds[0] == "keys"   # Cmd+Space
        assert steps[0]["keys"] == ["cmd", "space"]
        assert kinds[2] == "type"   # "Terminal"
        assert steps[2]["text"] == "Terminal"
        assert steps[-1] == {"kind": "keys", "keys": ["cmd", "v"]}

    def test_macos_b64_roundtrip(self) -> None:
        text = "日本語テスト"
        steps = build_clipboard_sequence(text, TargetOS.MACOS)
        # Find the echo command
        echo_step = [s for s in steps if s["kind"] == "type" and "echo" in s.get("text", "")]
        assert len(echo_step) == 1
        typed = echo_step[0]["text"]
        start = typed.index("'") + 1
        end = typed.index("'", start)
        b64_str = typed[start:end]
        decoded = base64.b64decode(b64_str).decode("utf-8")
        assert decoded == text


class TestBuildClipboardSequenceLinux:
    """Tests for build_clipboard_sequence() with Linux target."""

    def test_short_text_gtk_unicode(self) -> None:
        steps = build_clipboard_sequence("你好", TargetOS.LINUX)
        # 2 chars -> 2 sets of (Ctrl+Shift+U, type hex, Enter)
        assert len(steps) == 6
        assert steps[0] == {"kind": "keys", "keys": ["ctrl", "shift", "u"]}
        assert steps[1]["kind"] == "type"
        assert steps[1]["text"] == f"{ord('你'):04x}"
        assert steps[2] == {"kind": "keys", "keys": ["enter"]}

    def test_long_text_xclip(self) -> None:
        # > 5 chars triggers xclip path
        text = "这是一段较长文本"
        steps = build_clipboard_sequence(text, TargetOS.LINUX)
        kinds = [s["kind"] for s in steps]
        # Ctrl+Alt+T -> sleep -> type cmd -> Enter -> sleep -> Ctrl+V
        assert steps[0] == {"kind": "keys", "keys": ["ctrl", "alt", "t"]}
        assert steps[-1] == {"kind": "keys", "keys": ["ctrl", "v"]}

    def test_long_text_b64_roundtrip(self) -> None:
        text = "한국어 테스트 긴 문장"
        steps = build_clipboard_sequence(text, TargetOS.LINUX)
        echo_step = [s for s in steps if s["kind"] == "type" and "echo" in s.get("text", "")]
        assert len(echo_step) == 1
        typed = echo_step[0]["text"]
        start = typed.index("'") + 1
        end = typed.index("'", start)
        b64_str = typed[start:end]
        decoded = base64.b64decode(b64_str).decode("utf-8")
        assert decoded == text

    def test_single_char_gtk(self) -> None:
        steps = build_clipboard_sequence("é", TargetOS.LINUX)
        assert len(steps) == 3  # Ctrl+Shift+U, type hex, Enter
        assert steps[1]["text"] == f"{ord('é'):04x}"
