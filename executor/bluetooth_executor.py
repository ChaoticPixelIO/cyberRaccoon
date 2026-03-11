"""Bluetooth HID Action Executor — sends commands via Bluetooth.

Same interface as ActionExecutor (open/execute/close), but uses
Bluetooth L2CAP instead of USB Gadget device files. Inherits command
dispatch, deduplication, and timing from BaseExecutor.

Usage::

    executor = BluetoothExecutor()
    executor.open()          # sets up BT profile, waits for connection
    result = executor.execute({"id": "s1", "action": "click", "x": 640, "y": 360})
    executor.close()
"""

from __future__ import annotations

import logging

from config import HumanizeConfig
from executor.base_executor import BaseExecutor
from executor.bluetooth_device import (
    BluetoothHIDConnection,
    BluetoothHIDDevice,
)
from executor.keyboard import KeyboardController
from executor.mouse import MouseController

logger = logging.getLogger("M4.bt_executor")

# Report IDs matching the SDP record descriptor
_KEYBOARD_REPORT_ID = 0x01
_MOUSE_REPORT_ID = 0x02


class BluetoothExecutor(BaseExecutor):
    """Bluetooth HID executor — same interface as ActionExecutor.

    Sends keyboard and mouse HID commands to a target computer
    over a Bluetooth HID connection.

    Usage::

        executor = BluetoothExecutor()
        executor.open()   # configures BT, waits for host connection
        result = executor.execute({"action": "type", "text": "hello"})
        executor.close()

    Result format (identical to ActionExecutor)::

        {"id": "cmd_1", "status": "ok", "action": "click", "duration_ms": 45}
        {"id": "cmd_1", "status": "skipped", "reason": "duplicate command id"}
        {"id": "cmd_1", "status": "error", "action": "click", "error": "..."}
    """

    def __init__(
        self,
        device_name: str = "CyberRaccoon",
        connection_timeout: float = 60.0,
        humanize_config: HumanizeConfig | None = None,
        target_os: str | None = None,
    ) -> None:
        super().__init__(humanize_config=humanize_config, target_os=target_os)
        self._device_name = device_name
        self._connection_timeout = connection_timeout
        self._connection: BluetoothHIDConnection | None = None

    def open(self) -> None:
        """Set up Bluetooth HID profile and wait for host connection.

        This method blocks until a host computer connects via Bluetooth,
        or the connection timeout expires.

        Raises:
            HIDDeviceError: If Bluetooth setup or connection fails.
        """
        self._connection = BluetoothHIDConnection(
            device_name=self._device_name,
        )
        self._connection.setup()
        self._connection.wait_for_connection(timeout=self._connection_timeout)

        # Create device adapters (same interface as HIDDevice)
        kb_dev = BluetoothHIDDevice(self._connection, _KEYBOARD_REPORT_ID)
        ms_dev = BluetoothHIDDevice(self._connection, _MOUSE_REPORT_ID)

        # Create controllers with optional humanization
        self._keyboard = self._wrap_keyboard_if_humanized(
            KeyboardController(kb_dev)
        )
        self._mouse = self._wrap_mouse_if_humanized(
            MouseController(ms_dev)
        )

        logger.info(
            "BluetoothExecutor opened (device=%s)", self._device_name
        )

    @property
    def is_connected(self) -> bool:
        """Check if the Bluetooth HID connection is still active."""
        return self._connection is not None and self._connection.is_connected()

    @property
    def connected_host(self) -> str:
        """Friendly name (or MAC address) of the connected remote host."""
        if self._connection:
            return self._connection.remote_name
        return ""

    def close(self) -> None:
        """Disconnect Bluetooth and clean up."""
        if self._connection:
            self._connection.disconnect()
            self._connection = None
        self._keyboard = None
        self._mouse = None
        logger.info("BluetoothExecutor closed")
