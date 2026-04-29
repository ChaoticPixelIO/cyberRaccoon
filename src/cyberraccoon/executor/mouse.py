"""Mouse HID report builder — absolute coordinate protocol.

Builds 8-byte HID reports with Report ID prefix:
  report_id(1B=0x02) | buttons(1B) | X(2B LE) | Y(2B LE) | wheel(1B signed) | padding(1B)

Coordinates are absolute (0–32767) mapped from the LLM coordinate space
configured per ``MouseController`` instance via ``screen_width``/``screen_height``.
The dims must match the capture's ``target_width``/``target_height`` so the
HID layer interprets clicks in the same space the LLM is reasoning in.
"""

from __future__ import annotations

import logging
import struct
import time

from cyberraccoon.executor.hid_device import HIDDevice

logger = logging.getLogger("M4.mouse")

HID_MAX = 32767

# Mouse button bit masks
BUTTON_LEFT = 0x01
BUTTON_RIGHT = 0x02
BUTTON_MIDDLE = 0x04

BUTTON_MAP: dict[str, int] = {
    "left": BUTTON_LEFT,
    "right": BUTTON_RIGHT,
    "middle": BUTTON_MIDDLE,
}


REPORT_ID = 0x02  # Mouse Report ID in combined HID descriptor


def _build_report(buttons: int, hid_x: int, hid_y: int, wheel: int = 0) -> bytes:
    """Build an 8-byte absolute mouse HID report with Report ID prefix.

    Format: report_id(1B) | buttons(1B) | X(2B LE) | Y(2B LE) | wheel(1B signed) | pad(1B)
    """
    return struct.pack("<BBHHbB", REPORT_ID, buttons, hid_x, hid_y, wheel, 0)


class MouseController:
    """Generates and sends absolute-coordinate mouse HID reports.

    Usage::

        mouse = MouseController(hid_device, screen_width=1280, screen_height=720)
        mouse.click(640, 360)
        mouse.double_click(100, 200)
        mouse.scroll(640, 400, direction="down", amount=3)
        mouse.drag(100, 200, 500, 300)
    """

    def __init__(
        self,
        hid_device: HIDDevice,
        screen_width: int,
        screen_height: int,
    ) -> None:
        if screen_width <= 0:
            raise ValueError(f"screen_width must be > 0, got {screen_width}")
        if screen_height <= 0:
            raise ValueError(f"screen_height must be > 0, got {screen_height}")
        self._device = hid_device
        self._screen_width = screen_width
        self._screen_height = screen_height

    def _screen_to_hid(self, x: int, y: int) -> tuple[int, int]:
        """Convert screen coordinates to HID absolute (0–32767)."""
        hid_x = int(x * HID_MAX / self._screen_width)
        hid_y = int(y * HID_MAX / self._screen_height)
        hid_x = max(0, min(HID_MAX, hid_x))
        hid_y = max(0, min(HID_MAX, hid_y))
        return hid_x, hid_y

    def send_report(
        self, buttons: int, hid_x: int, hid_y: int, wheel: int = 0
    ) -> None:
        """Build and send a single mouse report."""
        report = _build_report(buttons, hid_x, hid_y, wheel)
        self._device.write(report)

    def move(self, x: int, y: int) -> None:
        """Move cursor to absolute screen position."""
        hid_x, hid_y = self._screen_to_hid(x, y)
        self.send_report(0, hid_x, hid_y)

    def click(self, x: int, y: int, button: str = "left") -> None:
        """Click at position: move → press → release (3 reports)."""
        hid_x, hid_y = self._screen_to_hid(x, y)
        btn_mask = BUTTON_MAP.get(button)
        if btn_mask is None:
            raise ValueError(f"Unknown mouse button: {button!r}")

        # Move to position
        self.send_report(0, hid_x, hid_y)
        time.sleep(0.02)

        # Press button
        self.send_report(btn_mask, hid_x, hid_y)
        time.sleep(0.05)

        # Release button
        self.send_report(0, hid_x, hid_y)

    def double_click(self, x: int, y: int) -> None:
        """Double-click at position: two clicks with 80ms gap."""
        self.click(x, y)
        time.sleep(0.08)
        self.click(x, y)

    def triple_click(self, x: int, y: int) -> None:
        """Triple-click at position: three clicks with 80ms gaps."""
        self.click(x, y)
        time.sleep(0.08)
        self.click(x, y)
        time.sleep(0.08)
        self.click(x, y)

    def scroll(
        self, x: int, y: int, direction: str = "down", amount: int = 3
    ) -> None:
        """Scroll at position: move → per-tick wheel reports → stop."""
        hid_x, hid_y = self._screen_to_hid(x, y)
        if direction not in ("up", "down"):
            raise ValueError(f"Unknown scroll direction: {direction!r}")
        single = -1 if direction == "down" else 1

        # Move to scroll position
        self.send_report(0, hid_x, hid_y)
        time.sleep(0.02)

        # Send one report per tick
        for _ in range(abs(amount)):
            self.send_report(0, hid_x, hid_y, single)
            time.sleep(0.05)

        # Stop scrolling
        self.send_report(0, hid_x, hid_y, 0)

    def mouse_down(self, x: int, y: int, button: str = "left") -> None:
        """Move to position and press button without releasing."""
        hid_x, hid_y = self._screen_to_hid(x, y)
        btn_mask = BUTTON_MAP.get(button)
        if btn_mask is None:
            raise ValueError(f"Unknown mouse button: {button!r}")
        self.send_report(0, hid_x, hid_y)
        time.sleep(0.02)
        self.send_report(btn_mask, hid_x, hid_y)

    def mouse_up(self, x: int, y: int) -> None:
        """Move to position and release all buttons."""
        hid_x, hid_y = self._screen_to_hid(x, y)
        self.send_report(0, hid_x, hid_y)

    def drag(self, from_x: int, from_y: int, to_x: int, to_y: int) -> None:
        """Drag from one position to another with 10-step linear interpolation."""
        from_hx, from_hy = self._screen_to_hid(from_x, from_y)
        to_hx, to_hy = self._screen_to_hid(to_x, to_y)
        steps = 10

        # Move to start position
        self.send_report(0, from_hx, from_hy)
        time.sleep(0.02)

        # Press left button
        self.send_report(BUTTON_LEFT, from_hx, from_hy)
        time.sleep(0.05)

        # Interpolate to destination
        for i in range(1, steps + 1):
            t = i / steps
            cx = int(from_hx + (to_hx - from_hx) * t)
            cy = int(from_hy + (to_hy - from_hy) * t)
            self.send_report(BUTTON_LEFT, cx, cy)
            time.sleep(0.02)

        # Release button
        self.send_report(0, to_hx, to_hy)
