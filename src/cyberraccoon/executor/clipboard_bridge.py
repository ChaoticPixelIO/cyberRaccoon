"""Clipboard bridge for non-ASCII text input.

When the target OS is known, intercepts type commands containing non-ASCII
characters (CJK, emoji, accented chars) and uses OS-specific clipboard
commands to paste them, since USB HID only supports US keyboard scancodes.
"""

from __future__ import annotations

import base64
import logging
from enum import Enum

from cyberraccoon.executor.keyboard import CHAR_MAP

logger = logging.getLogger("M4.clipboard_bridge")


class TargetOS(Enum):
    WINDOWS = "windows"
    MACOS = "macos"
    LINUX = "linux"


# Characters that type_text() can handle natively (CHAR_MAP + \n + \t)
_TYPEABLE_EXTRAS = frozenset("\n\t")


def has_non_typeable(text: str) -> bool:
    """Check if text contains characters outside CHAR_MAP + newline/tab."""
    for ch in text:
        if ch not in CHAR_MAP and ch not in _TYPEABLE_EXTRAS:
            return True
    return False


def _is_typeable(ch: str) -> bool:
    return ch in CHAR_MAP or ch in _TYPEABLE_EXTRAS


def split_text(text: str) -> list[tuple[str, bool]]:
    """Split text into (segment, needs_bridge) tuples.

    Groups consecutive typeable / non-typeable characters together.
    Returns an empty list for empty input.
    """
    if not text:
        return []

    segments: list[tuple[str, bool]] = []
    current: list[str] = []
    current_needs_bridge = not _is_typeable(text[0])

    for ch in text:
        ch_needs_bridge = not _is_typeable(ch)
        if ch_needs_bridge != current_needs_bridge:
            segments.append(("".join(current), current_needs_bridge))
            current = [ch]
            current_needs_bridge = ch_needs_bridge
        else:
            current.append(ch)

    if current:
        segments.append(("".join(current), current_needs_bridge))

    return segments


def _b64(text: str) -> str:
    """Base64-encode a UTF-8 string."""
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


# Maximum Base64 length that fits in Win+R's 260-char limit with the
# short PowerShell command (~105 chars overhead).
_WIN_SHORT_B64_LIMIT = 155


def build_clipboard_sequence(text: str, target_os: TargetOS) -> list[dict]:
    """Return a list of step dicts to set the clipboard and paste.

    Each step is one of:
      {"kind": "type",  "text": "..."}
      {"kind": "keys",  "keys": [...]}
      {"kind": "sleep", "seconds": float}
    """
    if target_os == TargetOS.WINDOWS:
        return _build_windows_sequence(text)
    elif target_os == TargetOS.MACOS:
        return _build_macos_sequence(text)
    else:
        return _build_linux_sequence(text)


def _build_windows_sequence(text: str) -> list[dict]:
    b64 = _b64(text)

    if len(b64) <= _WIN_SHORT_B64_LIMIT:
        # Short path: single Win+R command
        ps_cmd = (
            f"powershell -nop -w hidden -c "
            f"\"Set-Clipboard([Text.Encoding]::UTF8.GetString("
            f"[Convert]::FromBase64String('{b64}')))\""
        )
        return [
            {"kind": "keys", "keys": ["win", "r"]},
            {"kind": "sleep", "seconds": 0.5},
            {"kind": "type", "text": ps_cmd},
            {"kind": "keys", "keys": ["enter"]},
            {"kind": "sleep", "seconds": 0.8},
            {"kind": "keys", "keys": ["ctrl", "v"]},
        ]
    else:
        # Long path: open PowerShell window, type full command, exit
        set_cmd = (
            f"Set-Clipboard([Text.Encoding]::UTF8.GetString("
            f"[Convert]::FromBase64String('{b64}')))"
        )
        return [
            {"kind": "keys", "keys": ["win", "r"]},
            {"kind": "sleep", "seconds": 0.5},
            {"kind": "type", "text": "powershell -nop"},
            {"kind": "keys", "keys": ["enter"]},
            {"kind": "sleep", "seconds": 1.5},
            {"kind": "type", "text": set_cmd},
            {"kind": "keys", "keys": ["enter"]},
            {"kind": "sleep", "seconds": 0.5},
            {"kind": "type", "text": "exit"},
            {"kind": "keys", "keys": ["enter"]},
            {"kind": "sleep", "seconds": 0.5},
            {"kind": "keys", "keys": ["ctrl", "v"]},
        ]


def _build_macos_sequence(text: str) -> list[dict]:
    b64 = _b64(text)
    terminal_cmd = f"echo '{b64}' | base64 -D | pbcopy; exit"
    return [
        {"kind": "keys", "keys": ["cmd", "space"]},
        {"kind": "sleep", "seconds": 0.5},
        {"kind": "type", "text": "Terminal"},
        {"kind": "keys", "keys": ["enter"]},
        {"kind": "sleep", "seconds": 1.5},
        {"kind": "type", "text": terminal_cmd},
        {"kind": "keys", "keys": ["enter"]},
        {"kind": "sleep", "seconds": 0.8},
        {"kind": "keys", "keys": ["cmd", "tab"]},
        {"kind": "sleep", "seconds": 0.3},
        {"kind": "keys", "keys": ["cmd", "v"]},
    ]


# Threshold: use GTK Unicode input for <= this many chars, xclip for more
_LINUX_SHORT_CHAR_LIMIT = 5


def _build_linux_sequence(text: str) -> list[dict]:
    if len(text) <= _LINUX_SHORT_CHAR_LIMIT:
        return _build_linux_gtk_sequence(text)
    else:
        return _build_linux_xclip_sequence(text)


def _build_linux_gtk_sequence(text: str) -> list[dict]:
    """Per-character Ctrl+Shift+U Unicode input (GTK/IBus)."""
    steps: list[dict] = []
    for ch in text:
        hex_cp = f"{ord(ch):04x}"
        steps.append({"kind": "keys", "keys": ["ctrl", "shift", "u"]})
        steps.append({"kind": "type", "text": hex_cp})
        steps.append({"kind": "keys", "keys": ["enter"]})
    return steps


def _build_linux_xclip_sequence(text: str) -> list[dict]:
    """Terminal + xclip for longer text."""
    b64 = _b64(text)
    terminal_cmd = f"echo '{b64}' | base64 -d | xclip -selection clipboard; exit"
    return [
        {"kind": "keys", "keys": ["ctrl", "alt", "t"]},
        {"kind": "sleep", "seconds": 1.0},
        {"kind": "type", "text": terminal_cmd},
        {"kind": "keys", "keys": ["enter"]},
        {"kind": "sleep", "seconds": 0.8},
        {"kind": "keys", "keys": ["ctrl", "v"]},
    ]
