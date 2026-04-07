"""Low-level HID device file wrapper.

Handles opening, writing binary HID reports to, and closing
the /dev/hidg0 device file (shared by keyboard and mouse via Report IDs).
"""

from __future__ import annotations

import logging
import os
import select

logger = logging.getLogger("M4.hid_device")

# Timeout for HID writes (seconds). If the USB host is not reading
# reports (e.g. cable disconnected), writes to /dev/hidgN block forever
# in the default blocking mode. We open in non-blocking mode and use
# select() to wait up to this timeout before raising an error.
_WRITE_TIMEOUT_S = 5.0


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
        """Open the device file for binary non-blocking writing."""
        try:
            fd = os.open(self._device_path, os.O_WRONLY | os.O_NONBLOCK)
            self._file = os.fdopen(fd, "wb", buffering=0)
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
        except OSError as e:
            raise HIDDeviceError(
                f"Cannot open HID device {self._device_path}: {e}. "
                "Ensure USB Gadget is configured and USB cable is connected."
            )

    def write(self, report: bytes) -> None:
        """Write a single HID report to the device.

        Uses select() to wait up to ``_WRITE_TIMEOUT_S`` for the device
        to become writable, preventing indefinite hangs when no USB host
        is connected.
        """
        if not self._file:
            raise HIDDeviceError(
                f"Device not opened: {self._device_path}. Call open() first."
            )
        try:
            fd = self._file.fileno()
            _, ready, _ = select.select([], [fd], [], _WRITE_TIMEOUT_S)
            if not ready:
                raise HIDDeviceError(
                    f"Write timeout ({_WRITE_TIMEOUT_S}s) on {self._device_path}. "
                    "The USB host is not reading HID reports. Check: "
                    "(1) USB cable is connected, "
                    "(2) on macOS, complete Keyboard Setup Assistant or grant "
                    "Input Monitoring permission."
                )
            self._file.write(report)
            self._file.flush()
        except HIDDeviceError:
            raise
        except (OSError, ValueError) as e:
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
