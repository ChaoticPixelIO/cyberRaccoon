"""Low-level HID device file wrapper.

Handles opening, writing binary HID reports to, and closing
/dev/hidg0 (keyboard) and /dev/hidg1 (mouse) device files.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("M4.hid_device")


class HIDDeviceError(Exception):
    """Raised when HID device operations fail (not found, permission denied, write error)."""


class HIDDevice:
    """Wrapper around a Linux HID gadget device file.

    Usage::

        dev = HIDDevice("/dev/hidg0")
        dev.open()
        dev.write(report_bytes)
        dev.close()
    """

    def __init__(self, device_path: str) -> None:
        self._device_path = device_path
        self._file = None

    def open(self) -> None:
        """Open the device file for binary unbuffered writing."""
        try:
            self._file = open(self._device_path, "wb", buffering=0)
            logger.info("Opened HID device: %s", self._device_path)
        except FileNotFoundError:
            raise HIDDeviceError(
                f"HID device not found: {self._device_path}. "
                "Ensure USB Gadget is configured (run scripts/setup_gadget.sh)."
            )
        except PermissionError:
            raise HIDDeviceError(
                f"Permission denied: {self._device_path}. "
                "Try running with sudo or add user to the appropriate group."
            )

    def write(self, report: bytes) -> None:
        """Write a single HID report to the device."""
        if not self._file:
            raise HIDDeviceError(
                f"Device not opened: {self._device_path}. Call open() first."
            )
        try:
            self._file.write(report)
            self._file.flush()
        except OSError as e:
            raise HIDDeviceError(
                f"Failed to write to {self._device_path}: {e}"
            )

    def close(self) -> None:
        """Close the device file."""
        if self._file:
            try:
                self._file.close()
                logger.info("Closed HID device: %s", self._device_path)
            except OSError as e:
                logger.warning("Error closing %s: %s", self._device_path, e)
            finally:
                self._file = None

    def is_open(self) -> bool:
        """Check if the device file is currently open."""
        return self._file is not None
