"""Tests for M4 mouse controller — coordinate conversion, HID reports, operations."""

from __future__ import annotations

import struct

import pytest

from executor.mouse import (
    HID_MAX,
    REPORT_ID,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    _build_report,
    _screen_to_hid,
    MouseController,
)
from tests.test_executor.conftest import MockHIDDevice


class TestMouseCoordinates:
    """Tests for screen→HID coordinate conversion."""

    def test_origin(self) -> None:
        hx, hy = _screen_to_hid(0, 0)
        assert hx == 0
        assert hy == 0

    def test_center(self) -> None:
        hx, hy = _screen_to_hid(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
        # 640 * 32767 / 1280 = 16383.5 → 16383
        assert abs(hx - HID_MAX // 2) <= 1
        assert abs(hy - HID_MAX // 2) <= 1

    def test_max_corner(self) -> None:
        hx, hy = _screen_to_hid(SCREEN_WIDTH, SCREEN_HEIGHT)
        assert hx == HID_MAX
        assert hy == HID_MAX

    def test_negative_clamps_to_zero(self) -> None:
        hx, hy = _screen_to_hid(-10, -5)
        assert hx == 0
        assert hy == 0

    def test_overflow_clamps_to_max(self) -> None:
        hx, hy = _screen_to_hid(9999, 9999)
        assert hx == HID_MAX
        assert hy == HID_MAX


class TestMouseReport:
    """Tests for mouse HID report format."""

    def test_report_is_8_bytes(self) -> None:
        report = _build_report(0, 0, 0, 0)
        assert len(report) == 8

    def test_report_id_in_first_byte(self) -> None:
        report = _build_report(0, 0, 0, 0)
        assert report[0] == REPORT_ID

    def test_report_format(self) -> None:
        report = _build_report(0x01, 1000, 2000, -1)
        rid, buttons, x, y, wheel, pad = struct.unpack("<BBHHbB", report)
        assert rid == REPORT_ID
        assert buttons == 0x01
        assert x == 1000
        assert y == 2000
        assert wheel == -1
        assert pad == 0

    def test_zero_report(self) -> None:
        report = _build_report(0, 0, 0, 0)
        assert report == bytes([REPORT_ID]) + b"\x00" * 7


class TestMouseOperations:
    """Tests for mouse operation sequences."""

    def test_click_sends_3_reports(self, mock_mouse: MouseController,
                                    mock_mouse_device: MockHIDDevice) -> None:
        mock_mouse.click(640, 360)
        assert len(mock_mouse_device.reports) == 3

    def test_click_all_reports_8_bytes(self, mock_mouse: MouseController,
                                       mock_mouse_device: MockHIDDevice) -> None:
        mock_mouse.click(100, 200)
        for report in mock_mouse_device.reports:
            assert len(report) == 8

    def test_click_press_and_release(self, mock_mouse: MouseController,
                                      mock_mouse_device: MockHIDDevice) -> None:
        mock_mouse.click(640, 360)
        reports = mock_mouse_device.reports

        # Report 0: move (buttons=0), byte 1 is buttons (byte 0 is report ID)
        assert reports[0][1] == 0x00

        # Report 1: press (buttons=left=0x01)
        assert reports[1][1] == 0x01

        # Report 2: release (buttons=0)
        assert reports[2][1] == 0x00

    def test_right_click(self, mock_mouse: MouseController,
                          mock_mouse_device: MockHIDDevice) -> None:
        mock_mouse.click(640, 360, button="right")
        # Press report should have right button (0x02), byte 1 is buttons
        assert mock_mouse_device.reports[1][1] == 0x02

    def test_double_click_sends_6_reports(self, mock_mouse: MouseController,
                                           mock_mouse_device: MockHIDDevice) -> None:
        mock_mouse.double_click(640, 360)
        assert len(mock_mouse_device.reports) == 6

    def test_scroll_report_count(self, mock_mouse: MouseController,
                                  mock_mouse_device: MockHIDDevice) -> None:
        mock_mouse.scroll(640, 360, direction="down", amount=3)
        # 1 (move) + 3 (scroll ticks) + 1 (stop) = 5
        assert len(mock_mouse_device.reports) == 5

    def test_scroll_down_wheel_value(self, mock_mouse: MouseController,
                                      mock_mouse_device: MockHIDDevice) -> None:
        mock_mouse.scroll(640, 360, direction="down", amount=1)
        # reports: [move, scroll_tick, stop]
        scroll_report = mock_mouse_device.reports[1]
        _, _, _, _, wheel, _ = struct.unpack("<BBHHbB", scroll_report)
        assert wheel == -1

    def test_scroll_up_wheel_value(self, mock_mouse: MouseController,
                                    mock_mouse_device: MockHIDDevice) -> None:
        mock_mouse.scroll(640, 360, direction="up", amount=1)
        scroll_report = mock_mouse_device.reports[1]
        _, _, _, _, wheel, _ = struct.unpack("<BBHHbB", scroll_report)
        assert wheel == 1

    def test_drag_report_count(self, mock_mouse: MouseController,
                                mock_mouse_device: MockHIDDevice) -> None:
        mock_mouse.drag(100, 200, 500, 300)
        # 1 (move to start) + 1 (press) + 10 (interpolation) + 1 (release) = 13
        assert len(mock_mouse_device.reports) == 13

    def test_drag_holds_button_during_move(self, mock_mouse: MouseController,
                                            mock_mouse_device: MockHIDDevice) -> None:
        mock_mouse.drag(100, 200, 500, 300)
        reports = mock_mouse_device.reports
        # Reports 1 (press) through 11 (last interp step) should have button pressed
        # Byte 1 is buttons (byte 0 is report ID)
        for i in range(1, 12):
            assert reports[i][1] == 0x01, f"Report {i} should have left button"
        # Final release
        assert reports[12][1] == 0x00

    def test_move_sends_1_report(self, mock_mouse: MouseController,
                                  mock_mouse_device: MockHIDDevice) -> None:
        mock_mouse.move(640, 360)
        assert len(mock_mouse_device.reports) == 1


class TestMouseErrors:
    """Tests for error raising on invalid input."""

    def test_unknown_scroll_direction_raises(
        self, mock_mouse: MouseController, mock_mouse_device: MockHIDDevice
    ) -> None:
        with pytest.raises(ValueError, match="Unknown scroll direction"):
            mock_mouse.scroll(640, 360, direction="sideways")

    def test_unknown_button_raises(
        self, mock_mouse: MouseController, mock_mouse_device: MockHIDDevice
    ) -> None:
        with pytest.raises(ValueError, match="Unknown mouse button"):
            mock_mouse.click(640, 360, button="centre")
