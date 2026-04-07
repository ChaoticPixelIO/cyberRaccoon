"""Tests for M1 CameraCapture — CSI camera capture via picamera2.

All tests use mock Picamera2 (no hardware needed).
"""

from __future__ import annotations

import base64
import io
import sys
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

from cyberraccoon.capture.camera_capture import CameraCapture
from cyberraccoon.capture.screen_capture import CaptureError


# ---------------------------------------------------------------------------
# Helper: create a CameraCapture with mocked picamera2
# ---------------------------------------------------------------------------

def make_mock_camera(
    frame: np.ndarray,
    width: int = 1280,
    height: int = 720,
    quality: int = 80,
) -> CameraCapture:
    """Create a CameraCapture with a mocked Picamera2 backend.

    Injects a mock Picamera2 that returns the given RGB frame on capture_array().
    """
    cam = CameraCapture(
        target_width=width,
        target_height=height,
        jpeg_quality=quality,
    )

    mock_picam = MagicMock()
    mock_picam.capture_array.return_value = frame
    cam._picam = mock_picam

    return cam


# ---------------------------------------------------------------------------
# Fixtures — RGB frames (picamera2 returns RGB, not BGR)
# ---------------------------------------------------------------------------

def _rgb_frame_1080p() -> np.ndarray:
    """Simulated 1920x1080 RGB frame from CSI camera."""
    return np.zeros((1080, 1920, 3), dtype=np.uint8)


def _rgb_frame_with_content() -> np.ndarray:
    """RGB frame with visible content (white rectangle)."""
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    frame[100:200, 100:300] = 255
    return frame


def _rgb_frame_720p() -> np.ndarray:
    """Simulated 1280x720 RGB frame (no resize needed)."""
    return np.zeros((720, 1280, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCaptureResult:
    """Tests for camera capture output correctness."""

    def test_output_dimensions(self) -> None:
        """Output image size should match target resolution."""
        cam = make_mock_camera(_rgb_frame_1080p())
        result = cam.capture()
        assert result.width == 1280
        assert result.height == 720
        assert result.image.size == (1280, 720)

    def test_base64_is_valid_jpeg(self) -> None:
        """Base64 decoded content should be valid JPEG at target resolution."""
        cam = make_mock_camera(_rgb_frame_1080p())
        result = cam.capture()
        jpeg_bytes = base64.b64decode(result.base64_jpeg)
        image = Image.open(io.BytesIO(jpeg_bytes))
        assert image.format == "JPEG"
        assert image.size == (1280, 720)

    def test_size_bytes_matches_jpeg(self) -> None:
        """size_bytes should equal actual JPEG byte count."""
        cam = make_mock_camera(_rgb_frame_1080p())
        result = cam.capture()
        jpeg_bytes = base64.b64decode(result.base64_jpeg)
        assert result.size_bytes == len(jpeg_bytes)

    def test_timestamp_is_positive(self) -> None:
        """Timestamp should be a positive monotonic value."""
        cam = make_mock_camera(_rgb_frame_1080p())
        result = cam.capture()
        assert result.timestamp > 0

    def test_higher_quality_larger_size(self) -> None:
        """Higher JPEG quality should produce larger files."""
        frame = _rgb_frame_with_content()
        cam_low = make_mock_camera(frame, quality=30)
        cam_high = make_mock_camera(frame, quality=95)
        r_low = cam_low.capture()
        r_high = cam_high.capture()
        assert r_high.size_bytes > r_low.size_bytes

    def test_custom_resolution(self) -> None:
        """Custom target resolution should be applied."""
        cam = make_mock_camera(_rgb_frame_1080p(), width=640, height=480)
        result = cam.capture()
        assert result.width == 640
        assert result.height == 480
        assert result.image.size == (640, 480)

    def test_no_resize_when_matching(self) -> None:
        """Should skip resize when frame already matches target resolution."""
        cam = make_mock_camera(_rgb_frame_720p(), width=1280, height=720)
        result = cam.capture()
        assert result.width == 1280
        assert result.height == 720

    def test_image_is_rgb(self) -> None:
        """Output PIL Image should be in RGB mode."""
        cam = make_mock_camera(_rgb_frame_1080p())
        result = cam.capture()
        assert result.image.mode == "RGB"

    def test_base64_round_trip(self) -> None:
        """Base64 encode/decode round-trip preserves image."""
        cam = make_mock_camera(_rgb_frame_with_content())
        result = cam.capture()
        jpeg_bytes = base64.b64decode(result.base64_jpeg)
        img = Image.open(io.BytesIO(jpeg_bytes))
        assert img.size == (1280, 720)
        assert img.mode == "RGB"


class TestCaptureErrors:
    """Tests for error handling."""

    def test_capture_before_open(self) -> None:
        """Calling capture() without open() should raise CaptureError."""
        cam = CameraCapture()
        try:
            cam.capture()
            assert False, "Should have raised CaptureError"
        except CaptureError as e:
            assert "not opened" in str(e)

    def test_capture_array_failure(self) -> None:
        """capture_array() exception should be wrapped in CaptureError."""
        cam = CameraCapture()
        mock_picam = MagicMock()
        mock_picam.capture_array.side_effect = RuntimeError("Camera disconnected")
        cam._picam = mock_picam

        try:
            cam.capture()
            assert False, "Should have raised CaptureError"
        except CaptureError as e:
            assert "Failed to capture" in str(e)

    def test_open_without_picamera2(self) -> None:
        """open() should raise CaptureError if picamera2 is not installed."""
        cam = CameraCapture()

        # Temporarily make picamera2 un-importable
        with patch.dict(sys.modules, {"picamera2": None}):
            try:
                cam.open()
                assert False, "Should have raised CaptureError"
            except CaptureError as e:
                assert "picamera2" in str(e)

    def test_open_camera_failure(self) -> None:
        """open() should raise CaptureError if camera init fails."""
        mock_picamera2_module = MagicMock()
        mock_picamera2_module.Picamera2.side_effect = RuntimeError("No camera found")

        with patch.dict(sys.modules, {"picamera2": mock_picamera2_module}):
            cam = CameraCapture()
            try:
                cam.open()
                assert False, "Should have raised CaptureError"
            except CaptureError as e:
                assert "Cannot open CSI camera" in str(e)


class TestDeviceLifecycle:
    """Tests for open/close/is_open lifecycle."""

    def test_is_open_initially_false(self) -> None:
        """New CameraCapture should not be open."""
        cam = CameraCapture()
        assert cam.is_open() is False

    def test_is_open_after_mock_setup(self) -> None:
        """Should report open after mock setup."""
        cam = make_mock_camera(_rgb_frame_1080p())
        assert cam.is_open() is True

    def test_close_stops_camera(self) -> None:
        """close() should stop and close the camera, set is_open to False."""
        cam = make_mock_camera(_rgb_frame_1080p())
        mock_picam = cam._picam
        cam.close()
        assert cam.is_open() is False
        mock_picam.stop.assert_called_once()
        mock_picam.close.assert_called_once()

    def test_double_close_safe(self) -> None:
        """Calling close() twice should not raise."""
        cam = make_mock_camera(_rgb_frame_1080p())
        cam.close()
        cam.close()  # Should not raise

    def test_close_without_open_safe(self) -> None:
        """Calling close() on un-opened camera should not raise."""
        cam = CameraCapture()
        cam.close()  # Should not raise


class TestInit:
    """Tests for constructor defaults."""

    def test_default_values(self) -> None:
        """Default constructor values should match design spec."""
        cam = CameraCapture()
        assert cam._camera_index == 0
        assert cam._target_width == 1280
        assert cam._target_height == 720
        assert cam._jpeg_quality == 80
        assert cam._picam is None

    def test_custom_values(self) -> None:
        """Custom constructor values should be stored."""
        cam = CameraCapture(
            camera_index=1,
            target_width=1920,
            target_height=1080,
            jpeg_quality=95,
        )
        assert cam._camera_index == 1
        assert cam._target_width == 1920
        assert cam._target_height == 1080
        assert cam._jpeg_quality == 95
