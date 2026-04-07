"""Tests for M1 Screen Capture — image processing, encoding, error handling.

All tests use mock VideoCapture (no hardware needed).
"""

from __future__ import annotations

import base64
import io
from unittest.mock import MagicMock

import numpy as np
from PIL import Image

from cyberraccoon.capture.screen_capture import CaptureError, CaptureResult, ScreenCapture
from tests.test_capture.conftest import make_mock_capture


class TestCaptureResult:
    """Tests for capture output correctness."""

    def test_output_dimensions(self, fake_frame_1080p: np.ndarray) -> None:
        """Output image size should match target resolution."""
        cap = make_mock_capture(fake_frame_1080p)
        result = cap.capture()
        assert result.width == 1280
        assert result.height == 720
        assert result.image.size == (1280, 720)

    def test_base64_is_valid_jpeg(self, fake_frame_1080p: np.ndarray) -> None:
        """Base64 decoded content should be valid JPEG at target resolution."""
        cap = make_mock_capture(fake_frame_1080p)
        result = cap.capture()
        jpeg_bytes = base64.b64decode(result.base64_jpeg)
        image = Image.open(io.BytesIO(jpeg_bytes))
        assert image.format == "JPEG"
        assert image.size == (1280, 720)

    def test_size_bytes_matches_jpeg(self, fake_frame_1080p: np.ndarray) -> None:
        """size_bytes should equal actual JPEG byte count."""
        cap = make_mock_capture(fake_frame_1080p)
        result = cap.capture()
        jpeg_bytes = base64.b64decode(result.base64_jpeg)
        assert result.size_bytes == len(jpeg_bytes)

    def test_timestamp_is_positive(self, fake_frame_1080p: np.ndarray) -> None:
        """Timestamp should be a positive monotonic value."""
        cap = make_mock_capture(fake_frame_1080p)
        result = cap.capture()
        assert result.timestamp > 0

    def test_higher_quality_larger_size(
        self, fake_frame_with_content: np.ndarray
    ) -> None:
        """Higher JPEG quality should produce larger files."""
        cap_low = make_mock_capture(fake_frame_with_content, quality=30)
        cap_high = make_mock_capture(fake_frame_with_content, quality=95)
        r_low = cap_low.capture()
        r_high = cap_high.capture()
        assert r_high.size_bytes > r_low.size_bytes

    def test_custom_resolution(self, fake_frame_1080p: np.ndarray) -> None:
        """Custom target resolution should be applied."""
        cap = make_mock_capture(fake_frame_1080p, width=640, height=480)
        result = cap.capture()
        assert result.width == 640
        assert result.height == 480
        assert result.image.size == (640, 480)

    def test_no_resize_when_matching(self, fake_frame_720p: np.ndarray) -> None:
        """Should skip resize when frame already matches target resolution."""
        cap = make_mock_capture(fake_frame_720p, width=1280, height=720)
        result = cap.capture()
        assert result.width == 1280
        assert result.height == 720

    def test_image_is_rgb(self, fake_frame_1080p: np.ndarray) -> None:
        """Output PIL Image should be in RGB mode."""
        cap = make_mock_capture(fake_frame_1080p)
        result = cap.capture()
        assert result.image.mode == "RGB"

    def test_base64_round_trip(self, fake_frame_with_content: np.ndarray) -> None:
        """Base64 encode/decode round-trip preserves image content."""
        cap = make_mock_capture(fake_frame_with_content)
        result = cap.capture()
        jpeg_bytes = base64.b64decode(result.base64_jpeg)
        img = Image.open(io.BytesIO(jpeg_bytes))
        # JPEG is lossy, so just check size and mode
        assert img.size == (1280, 720)
        assert img.mode == "RGB"


class TestBufferFlush:
    """Tests for buffer flushing behavior."""

    def test_grab_called_3_times(self, fake_frame_1080p: np.ndarray) -> None:
        """Should discard 3 buffered frames before reading."""
        cap = make_mock_capture(fake_frame_1080p)
        cap.capture()
        assert cap._cap.grab.call_count == 3

    def test_read_called_once(self, fake_frame_1080p: np.ndarray) -> None:
        """Should read exactly one frame."""
        cap = make_mock_capture(fake_frame_1080p)
        cap.capture()
        assert cap._cap.read.call_count == 1


class TestCaptureErrors:
    """Tests for error handling."""

    def test_capture_before_open(self) -> None:
        """Calling capture() without open() should raise CaptureError."""
        cap = ScreenCapture()
        try:
            cap.capture()
            assert False, "Should have raised CaptureError"
        except CaptureError as e:
            assert "not opened" in str(e)

    def test_read_failure(self) -> None:
        """Frame read failure should raise CaptureError."""
        cap = ScreenCapture()
        mock_vc = MagicMock()
        mock_vc.isOpened.return_value = True
        mock_vc.read.return_value = (False, None)
        mock_vc.grab.return_value = True
        cap._cap = mock_vc

        try:
            cap.capture()
            assert False, "Should have raised CaptureError"
        except CaptureError as e:
            assert "Failed to read" in str(e)

    def test_read_returns_none_frame(self) -> None:
        """Read returning None frame should raise CaptureError."""
        cap = ScreenCapture()
        mock_vc = MagicMock()
        mock_vc.isOpened.return_value = True
        mock_vc.read.return_value = (True, None)
        mock_vc.grab.return_value = True
        cap._cap = mock_vc

        try:
            cap.capture()
            assert False, "Should have raised CaptureError"
        except CaptureError as e:
            assert "Failed to read" in str(e)


class TestDeviceLifecycle:
    """Tests for open/close/is_open lifecycle."""

    def test_is_open_initially_false(self) -> None:
        """New ScreenCapture should not be open."""
        cap = ScreenCapture()
        assert cap.is_open() is False

    def test_is_open_after_mock_open(self, fake_frame_1080p: np.ndarray) -> None:
        """Should report open after mock setup."""
        cap = make_mock_capture(fake_frame_1080p)
        assert cap.is_open() is True

    def test_close_releases_device(self, fake_frame_1080p: np.ndarray) -> None:
        """close() should release the device and set is_open to False."""
        cap = make_mock_capture(fake_frame_1080p)
        cap.close()
        assert cap.is_open() is False

    def test_close_calls_release(self, fake_frame_1080p: np.ndarray) -> None:
        """close() should call release() on the VideoCapture."""
        cap = make_mock_capture(fake_frame_1080p)
        mock_vc = cap._cap
        cap.close()
        mock_vc.release.assert_called_once()

    def test_double_close_safe(self, fake_frame_1080p: np.ndarray) -> None:
        """Calling close() twice should not raise."""
        cap = make_mock_capture(fake_frame_1080p)
        cap.close()
        cap.close()  # Should not raise

    def test_close_without_open_safe(self) -> None:
        """Calling close() on un-opened device should not raise."""
        cap = ScreenCapture()
        cap.close()  # Should not raise


class TestInit:
    """Tests for constructor defaults."""

    def test_default_values(self) -> None:
        """Default constructor values should match design spec."""
        cap = ScreenCapture()
        assert cap._device_index == 0
        assert cap._target_width == 1280
        assert cap._target_height == 720
        assert cap._jpeg_quality == 80
        assert cap._cap is None

    def test_custom_values(self) -> None:
        """Custom constructor values should be stored."""
        cap = ScreenCapture(
            device_index=2,
            target_width=1920,
            target_height=1080,
            jpeg_quality=95,
        )
        assert cap._device_index == 2
        assert cap._target_width == 1920
        assert cap._target_height == 1080
        assert cap._jpeg_quality == 95
