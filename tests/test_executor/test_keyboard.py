"""Tests for M4 keyboard controller — char map, HID reports, key combos."""

from __future__ import annotations

import pytest

from executor.keyboard import (
    CHAR_MAP,
    MODIFIER_MAP,
    REPORT_ID,
    SPECIAL_KEY_MAP,
    KeyboardController,
    _build_report,
    _RELEASE_REPORT,
)
from tests.test_executor.conftest import MockHIDDevice


class TestCharMap:
    """Tests for US keyboard layout character mapping."""

    def test_all_lowercase_present(self) -> None:
        for c in "abcdefghijklmnopqrstuvwxyz":
            assert c in CHAR_MAP, f"Missing: {c}"

    def test_lowercase_no_shift(self) -> None:
        for c in "abcdefghijklmnopqrstuvwxyz":
            _, needs_shift = CHAR_MAP[c]
            assert not needs_shift, f"{c} should not need shift"

    def test_all_uppercase_present(self) -> None:
        for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            assert c in CHAR_MAP, f"Missing: {c}"

    def test_uppercase_needs_shift(self) -> None:
        for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            _, needs_shift = CHAR_MAP[c]
            assert needs_shift, f"{c} should need shift"

    def test_all_digits_present(self) -> None:
        for c in "0123456789":
            assert c in CHAR_MAP, f"Missing: {c}"

    def test_digits_no_shift(self) -> None:
        for c in "0123456789":
            _, needs_shift = CHAR_MAP[c]
            assert not needs_shift, f"{c} should not need shift"

    def test_shift_digit_symbols(self) -> None:
        for c in "!@#$%^&*()":
            assert c in CHAR_MAP, f"Missing: {c}"
            _, needs_shift = CHAR_MAP[c]
            assert needs_shift, f"{c} should need shift"

    def test_common_punctuation_present(self) -> None:
        for c in " -=[]\\;',./":
            assert c in CHAR_MAP, f"Missing: {repr(c)}"

    def test_shift_punctuation_present(self) -> None:
        for c in '_+{}|:"~<>?':
            assert c in CHAR_MAP, f"Missing: {repr(c)}"

    def test_same_hid_code_for_case_pair(self) -> None:
        """a/A should share the same HID usage code."""
        for lower, upper in zip("abcdefghijklmnopqrstuvwxyz",
                                 "ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
            assert CHAR_MAP[lower][0] == CHAR_MAP[upper][0]


class TestKeyboardReport:
    """Tests for keyboard HID report format."""

    def test_report_is_9_bytes(self) -> None:
        report = _build_report(0, [])
        assert len(report) == 9

    def test_report_id_in_first_byte(self) -> None:
        report = _build_report(0, [])
        assert report[0] == REPORT_ID

    def test_modifier_in_second_byte(self) -> None:
        report = _build_report(0x01, [0x04])  # Ctrl + 'a'
        assert report[1] == 0x01

    def test_reserved_byte_is_zero(self) -> None:
        report = _build_report(0xFF, [0x04, 0x05])
        assert report[2] == 0x00

    def test_keycode_in_correct_position(self) -> None:
        report = _build_report(0, [0x04])
        assert report[3] == 0x04

    def test_multiple_keycodes(self) -> None:
        report = _build_report(0, [0x04, 0x05, 0x06])
        assert report[3] == 0x04
        assert report[4] == 0x05
        assert report[5] == 0x06

    def test_max_6_keycodes(self) -> None:
        codes = [0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A]  # 7 codes
        report = _build_report(0, codes)
        # Should only include first 6
        assert report[3:9] == bytes([0x04, 0x05, 0x06, 0x07, 0x08, 0x09])

    def test_release_report_has_report_id(self) -> None:
        assert _RELEASE_REPORT[0] == REPORT_ID
        assert _RELEASE_REPORT[1:] == bytes(8)
        assert len(_RELEASE_REPORT) == 9


class TestKeyboardOperations:
    """Tests for keyboard press_keys and type_text."""

    def test_press_keys_sends_press_and_release(
        self, mock_keyboard: KeyboardController, mock_keyboard_device: MockHIDDevice
    ) -> None:
        mock_keyboard.press_keys(["a"])
        # Should send: key press + release
        assert len(mock_keyboard_device.reports) == 2

    def test_ctrl_c_modifier_byte(
        self, mock_keyboard: KeyboardController, mock_keyboard_device: MockHIDDevice
    ) -> None:
        mock_keyboard.press_keys(["ctrl", "c"])
        press_report = mock_keyboard_device.reports[0]
        # Report ID
        assert press_report[0] == REPORT_ID
        # Ctrl modifier
        assert press_report[1] == MODIFIER_MAP["ctrl"]
        # 'c' keycode
        assert press_report[3] == CHAR_MAP["c"][0]

    def test_type_text_sends_press_release_per_char(
        self, mock_keyboard: KeyboardController, mock_keyboard_device: MockHIDDevice
    ) -> None:
        mock_keyboard.type_text("ab")
        # 2 chars × (press + release) = 4 reports
        assert len(mock_keyboard_device.reports) == 4

    def test_type_text_uppercase_uses_shift(
        self, mock_keyboard: KeyboardController, mock_keyboard_device: MockHIDDevice
    ) -> None:
        mock_keyboard.type_text("A")
        press_report = mock_keyboard_device.reports[0]
        assert press_report[1] == MODIFIER_MAP["shift"]

    def test_type_text_newline_sends_enter(
        self, mock_keyboard: KeyboardController, mock_keyboard_device: MockHIDDevice
    ) -> None:
        mock_keyboard.type_text("\n")
        # press_keys(["enter"]) → 2 reports
        assert len(mock_keyboard_device.reports) == 2
        press_report = mock_keyboard_device.reports[0]
        assert press_report[3] == SPECIAL_KEY_MAP["enter"]

    def test_type_text_tab_sends_tab(
        self, mock_keyboard: KeyboardController, mock_keyboard_device: MockHIDDevice
    ) -> None:
        mock_keyboard.type_text("\t")
        assert len(mock_keyboard_device.reports) == 2
        press_report = mock_keyboard_device.reports[0]
        assert press_report[3] == SPECIAL_KEY_MAP["tab"]

    def test_special_key_enter(
        self, mock_keyboard: KeyboardController, mock_keyboard_device: MockHIDDevice
    ) -> None:
        mock_keyboard.press_keys(["enter"])
        press_report = mock_keyboard_device.reports[0]
        assert press_report[3] == SPECIAL_KEY_MAP["enter"]

    def test_special_key_f5(
        self, mock_keyboard: KeyboardController, mock_keyboard_device: MockHIDDevice
    ) -> None:
        mock_keyboard.press_keys(["f5"])
        press_report = mock_keyboard_device.reports[0]
        assert press_report[3] == SPECIAL_KEY_MAP["f5"]


class TestKeyboardErrors:
    """Tests for error raising on invalid input."""

    def test_unknown_key_raises(
        self, mock_keyboard: KeyboardController, mock_keyboard_device: MockHIDDevice
    ) -> None:
        with pytest.raises(ValueError, match="Unknown key: badkey"):
            mock_keyboard.press_keys(["badkey"])

    def test_unsupported_char_raises(
        self, mock_keyboard: KeyboardController, mock_keyboard_device: MockHIDDevice
    ) -> None:
        with pytest.raises(ValueError, match="Unsupported character"):
            mock_keyboard.type_text("\u20ac")
