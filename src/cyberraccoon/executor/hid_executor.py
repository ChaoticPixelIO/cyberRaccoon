"""M4 Action Executor — dispatches commands via USB HID Gadget.

Routes JSON commands to mouse/keyboard controllers using a single
/dev/hidg0 device file with Report IDs. Inherits command dispatch,
deduplication, and timing from BaseExecutor.
"""

from __future__ import annotations

import logging

from cyberraccoon.config import HumanizeConfig
from cyberraccoon.executor.base_executor import BaseExecutor
from cyberraccoon.executor.hid_device import HIDDevice
from cyberraccoon.executor.keyboard import KeyboardController
from cyberraccoon.executor.mouse import MouseController

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
        device: str = "/dev/hidg0",
        humanize_config: HumanizeConfig | None = None,
        target_os: str | None = None,
        # Legacy kwarg accepted for backwards compatibility
        keyboard_device: str | None = None,
        *,
        screen_width: int,
        screen_height: int,
    ) -> None:
        super().__init__(
            humanize_config=humanize_config,
            target_os=target_os,
            screen_width=screen_width,
            screen_height=screen_height,
        )
        self._device_path = keyboard_device or device
        self._hid_dev: HIDDevice | None = None

    def open(self) -> None:
        """Open the HID device file and initialize controllers.

        Keyboard and mouse share a single /dev/hidg0 device using
        Report IDs (1=keyboard, 2=mouse) in a combined HID descriptor.
        """
        self._hid_dev = HIDDevice(self._device_path)
        self._hid_dev.open()

        self._keyboard = self._wrap_keyboard_if_humanized(
            KeyboardController(self._hid_dev)
        )
        self._mouse = self._wrap_mouse_if_humanized(
            MouseController(
                self._hid_dev,
                screen_width=self._screen_width,
                screen_height=self._screen_height,
            )
        )

        logger.info("ActionExecutor opened (device=%s)", self._device_path)

    def close(self) -> None:
        """Close the HID device file."""
        if self._hid_dev:
            self._hid_dev.close()
            self._hid_dev = None
        self._keyboard = None
        self._mouse = None
        logger.info("ActionExecutor closed")
