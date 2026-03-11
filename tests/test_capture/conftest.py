"""Shared fixtures for M1 Screen Capture tests."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from capture.screen_capture import ScreenCapture


@pytest.fixture
def fake_frame_1080p() -> np.ndarray:
    """Simulated 1920x1080 BGR frame from capture card."""
    return np.zeros((1080, 1920, 3), dtype=np.uint8)


@pytest.fixture
def fake_frame_with_content() -> np.ndarray:
    """Frame with visible content (white rectangle) for non-trivial testing."""
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    # Draw a white rectangle so JPEG quality comparisons are meaningful
    frame[100:200, 100:300] = 255
    return frame


@pytest.fixture
def fake_frame_720p() -> np.ndarray:
    """Simulated 1280x720 BGR frame (no resize needed)."""
    return np.zeros((720, 1280, 3), dtype=np.uint8)


def make_mock_capture(
    frame: np.ndarray,
    width: int = 1280,
    height: int = 720,
    quality: int = 80,
) -> ScreenCapture:
    """Create a ScreenCapture with a mocked cv2.VideoCapture backend.

    Injects a mock VideoCapture that returns the given frame on read().
    """
    cap = ScreenCapture(
        target_width=width,
        target_height=height,
        jpeg_quality=quality,
    )

    mock_vc = MagicMock()
    mock_vc.isOpened.return_value = True
    mock_vc.read.return_value = (True, frame)
    mock_vc.grab.return_value = True
    mock_vc.get.return_value = 1920  # Simulated actual resolution query
    cap._cap = mock_vc

    return cap
