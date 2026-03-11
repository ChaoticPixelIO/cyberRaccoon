"""M4 Action Executor — dispatches commands via USB HID Gadget.

Routes JSON commands to mouse/keyboard controllers using /dev/hidg*
device files. Inherits command dispatch, deduplication, and timing
from BaseExecutor.
"""

from __future__ import annotations

import logging

from config import HumanizeConfig
from executor.base_executor import BaseExecutor
from executor.hid_device import HIDDevice
from executor.keyboard import KeyboardController
from executor.mouse import MouseController

logger = logging.getLogger("M4.executor")


class ActionExecutor(BaseExecutor):
    """USB HID Gadget executor — sends commands via /dev/hidg* device files.

    Usage::

        executor = ActionExecutor()
        executor.open()
        result = executor.execute({"id": "step_1_abc", "action": "click", "x": 640, "y": 360})
        executor.close()

    Result format::

        {"id": "step_1_abc", "status": "ok", "action": "click", "duration_ms": 45}
        {"id": "step_1_abc", "status": "skipped", "reason": "duplicate command id"}
        {"id": "step_1_abc", "status": "error", "action": "click", "error": "..."}
    """

    def __init__(
        self,
        keyboard_device: str = "/dev/hidg0",
        mouse_device: str = "/dev/hidg1",
        humanize_config: HumanizeConfig | None = None,
        target_os: str | None = None,
    ) -> None:
        super().__init__(humanize_config=humanize_config, target_os=target_os)
        self._keyboard_device_path = keyboard_device
        self._mouse_device_path = mouse_device
        self._keyboard_dev: HIDDevice | None = None
        self._mouse_dev: HIDDevice | None = None

    def open(self) -> None:
        """Open both HID device files and initialize controllers."""
        self._keyboard_dev = HIDDevice(self._keyboard_device_path)
        self._keyboard_dev.open()

        self._mouse_dev = HIDDevice(self._mouse_device_path)
        try:
            self._mouse_dev.open()
        except Exception:
            self._keyboard_dev.close()
            self._keyboard_dev = None
            raise

        self._keyboard = self._wrap_keyboard_if_humanized(
            KeyboardController(self._keyboard_dev)
        )
        self._mouse = self._wrap_mouse_if_humanized(
            MouseController(self._mouse_dev)
        )

        logger.info("ActionExecutor opened (keyboard=%s, mouse=%s)",
                     self._keyboard_device_path, self._mouse_device_path)

    def close(self) -> None:
        """Close both HID device files."""
        if self._keyboard_dev:
            self._keyboard_dev.close()
            self._keyboard_dev = None
        if self._mouse_dev:
            self._mouse_dev.close()
            self._mouse_dev = None
        self._keyboard = None
        self._mouse = None
        logger.info("ActionExecutor closed")
