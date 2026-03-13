"""Keyboard HID report builder.

Builds 9-byte HID reports with Report ID prefix:
  report_id(1B=0x01) | modifier(1B) | reserved(1B) | keycodes(6B)

Supports US keyboard layout character mapping and modifier key combos.
"""

from __future__ import annotations

import logging
import time

from executor.hid_device import HIDDevice

logger = logging.getLogger("M4.keyboard")

# ---------------------------------------------------------------------------
# Modifier keys (byte 1 of keyboard report; byte 0 is Report ID)
# ---------------------------------------------------------------------------

MODIFIER_MAP: dict[str, int] = {
    "ctrl": 0x01,          # Left Ctrl
    "control": 0x01,       # alias
    "shift": 0x02,         # Left Shift
    "alt": 0x04,           # Left Alt
    "option": 0x04,        # alias (macOS)
    "meta": 0x08,          # Left GUI (Win / Cmd)
    "cmd": 0x08,           # alias (macOS)
    "command": 0x08,       # alias (macOS)
    "win": 0x08,           # alias (Windows)
    "super": 0x08,         # alias (Linux)
    "gui": 0x08,           # alias (HID spec name)
    "right_ctrl": 0x10,
    "right_shift": 0x20,
    "right_alt": 0x40,
    "right_meta": 0x80,
}

# ---------------------------------------------------------------------------
# Special keys (HID Usage IDs)
# ---------------------------------------------------------------------------

SPECIAL_KEY_MAP: dict[str, int] = {
    "enter": 0x28,
    "return": 0x28,        # alias (Anthropic CU sends "Return")
    "escape": 0x29,
    "esc": 0x29,              # alias
    "backspace": 0x2A,
    "back_space": 0x2A,       # xdotool alias
    "tab": 0x2B,
    "space": 0x2C,
    "delete": 0x4C,
    "home": 0x4A,
    "end": 0x4D,
    "pageup": 0x4B,
    "page_up": 0x4B,          # xdotool alias
    "pagedown": 0x4E,
    "page_down": 0x4E,        # xdotool alias
    "up": 0x52,
    "down": 0x51,
    "left": 0x50,
    "right": 0x4F,
    "f1": 0x3A, "f2": 0x3B, "f3": 0x3C, "f4": 0x3D,
    "f5": 0x3E, "f6": 0x3F, "f7": 0x40, "f8": 0x41,
    "f9": 0x42, "f10": 0x43, "f11": 0x44, "f12": 0x45,
    "capslock": 0x39,
    "printscreen": 0x46,
    "insert": 0x49,
}

# ---------------------------------------------------------------------------
# Character map: char → (hid_usage_id, needs_shift)
# ---------------------------------------------------------------------------

CHAR_MAP: dict[str, tuple[int, bool]] = {}


def _init_char_map() -> None:
    """Populate CHAR_MAP with US keyboard layout mappings."""
    # Lowercase a-z
    for i, c in enumerate("abcdefghijklmnopqrstuvwxyz"):
        CHAR_MAP[c] = (0x04 + i, False)

    # Uppercase A-Z (same HID codes, shift required)
    for i, c in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
        CHAR_MAP[c] = (0x04 + i, True)

    # Digits 1-9, 0 (HID codes 0x1E–0x27)
    for i, c in enumerate("1234567890"):
        CHAR_MAP[c] = (0x1E + i, False)

    # Shift + digits → symbols
    for i, c in enumerate("!@#$%^&*()"):
        CHAR_MAP[c] = (0x1E + i, True)

    # Punctuation and symbols (US layout)
    _symbols: dict[str, tuple[int, bool]] = {
        " ":  (0x2C, False),
        "-":  (0x2D, False), "_":  (0x2D, True),
        "=":  (0x2E, False), "+":  (0x2E, True),
        "[":  (0x2F, False), "{":  (0x2F, True),
        "]":  (0x30, False), "}":  (0x30, True),
        "\\": (0x31, False), "|":  (0x31, True),
        ";":  (0x33, False), ":":  (0x33, True),
        "'":  (0x34, False), '"':  (0x34, True),
        "`":  (0x35, False), "~":  (0x35, True),
        ",":  (0x36, False), "<":  (0x36, True),
        ".":  (0x37, False), ">":  (0x37, True),
        "/":  (0x38, False), "?":  (0x38, True),
    }
    CHAR_MAP.update(_symbols)


# Initialize on module load
_init_char_map()


# ---------------------------------------------------------------------------
# Report building
# ---------------------------------------------------------------------------

REPORT_ID = 0x01  # Keyboard Report ID in combined HID descriptor


def _build_report(modifier: int, keycodes: list[int]) -> bytes:
    """Build a 9-byte Keyboard HID report with Report ID prefix.

    Format: report_id(1B) | modifier(1B) | reserved(1B) | keycodes(up to 6B)
    """
    report = bytearray(9)
    report[0] = REPORT_ID
    report[1] = modifier
    # report[2] = 0  (reserved, already zero)
    for i, code in enumerate(keycodes[:6]):
        report[3 + i] = code
    return bytes(report)


_RELEASE_REPORT = bytes([REPORT_ID]) + bytes(8)  # Report ID + all zeros


class KeyboardController:
    """Generates and sends keyboard HID reports.

    Usage::

        kb = KeyboardController(hid_device)
        kb.type_text("hello world")
        kb.press_keys(["ctrl", "c"])
    """

    def __init__(self, hid_device: HIDDevice) -> None:
        self._device = hid_device

    def send_raw(self, report: bytes) -> None:
        """Send a pre-built HID report bytes directly to the device."""
        self._device.write(report)

    def release_all(self) -> None:
        """Send an all-zeros report to release all keys."""
        self._device.write(_RELEASE_REPORT)

    def press_keys(self, keys: list[str]) -> None:
        """Press a key combination (e.g. ["ctrl", "c"]) then release.

        Separates modifiers from regular keys, builds a single report
        with all pressed simultaneously, then releases.
        """
        modifier = 0
        keycodes: list[int] = []

        for key in keys:
            key_lower = key.lower()
            if key_lower in MODIFIER_MAP:
                modifier |= MODIFIER_MAP[key_lower]
            elif key_lower in SPECIAL_KEY_MAP:
                keycodes.append(SPECIAL_KEY_MAP[key_lower])
            elif key_lower in CHAR_MAP:
                usage_id, needs_shift = CHAR_MAP[key_lower]
                if needs_shift:
                    modifier |= MODIFIER_MAP["shift"]
                keycodes.append(usage_id)
            else:
                raise ValueError(f"Unknown key: {key}")

        # Send key press
        report = _build_report(modifier, keycodes)
        self._device.write(report)
        time.sleep(0.05)

        # Release all
        self.release_all()

    def type_text(self, text: str) -> None:
        """Type a string character by character with appropriate delays."""
        for char in text:
            if char == "\n":
                self.press_keys(["enter"])
                time.sleep(0.03)
                continue

            if char == "\t":
                self.press_keys(["tab"])
                time.sleep(0.03)
                continue

            if char not in CHAR_MAP:
                raise ValueError(f"Unsupported character: {char!r}")

            usage_id, needs_shift = CHAR_MAP[char]
            modifier = MODIFIER_MAP["shift"] if needs_shift else 0

            report = _build_report(modifier, [usage_id])
            self._device.write(report)
            time.sleep(0.02)

            self.release_all()
            time.sleep(0.02)
