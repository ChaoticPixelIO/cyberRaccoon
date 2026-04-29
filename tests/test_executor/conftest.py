"""Shared fixtures for M4 executor tests.

Provides MockHIDDevice that records all HID reports written to it,
plus pre-built mouse and keyboard controller fixtures.
"""

from __future__ import annotations

import pytest

from cyberraccoon.executor.mouse import MouseController
from cyberraccoon.executor.keyboard import KeyboardController


class MockHIDDevice:
    """Mock HID device that records all written reports.

    Usage::

        dev = MockHIDDevice()
        dev.open()
        dev.write(b"\\x00" * 8)
        assert len(dev.reports) == 1
    """

    def __init__(self, device_path: str = "/dev/mock") -> None:
        self.device_path = device_path
        self.reports: list[bytes] = []
        self._open = False

    def open(self) -> None:
        self._open = True

    def write(self, report: bytes) -> None:
        if not self._open:
            raise RuntimeError("MockHIDDevice not opened")
        self.reports.append(report)

    def close(self) -> None:
        self._open = False

    def is_open(self) -> bool:
        return self._open


@pytest.fixture
def mock_mouse_device() -> MockHIDDevice:
    """A MockHIDDevice configured for mouse testing."""
    dev = MockHIDDevice("/dev/hidg0")
    dev.open()
    return dev


@pytest.fixture
def mock_keyboard_device() -> MockHIDDevice:
    """A MockHIDDevice configured for keyboard testing."""
    dev = MockHIDDevice("/dev/hidg0")
    dev.open()
    return dev


@pytest.fixture
def mock_mouse(mock_mouse_device: MockHIDDevice) -> MouseController:
    """MouseController wired to a MockHIDDevice (1920x1080 LLM coordinate space)."""
    return MouseController(
        mock_mouse_device, screen_width=1920, screen_height=1080,
    )


@pytest.fixture
def mock_mouse_720(mock_mouse_device: MockHIDDevice) -> MouseController:
    """MouseController wired to a MockHIDDevice (1280x720 LLM coordinate space)."""
    return MouseController(
        mock_mouse_device, screen_width=1280, screen_height=720,
    )


@pytest.fixture
def mock_keyboard(mock_keyboard_device: MockHIDDevice) -> KeyboardController:
    """KeyboardController wired to a MockHIDDevice."""
    return KeyboardController(mock_keyboard_device)
