"""Tests for capture.base — CaptureSource Protocol, factory, and helpers."""

from __future__ import annotations

import base64
import io

import numpy as np
import pytest
from PIL import Image

from cyberraccoon.capture.base import (
    CaptureError,
    CaptureResult,
    CaptureSource,
    compute_frame_diff,
    frame_to_capture_result,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rgb_image(width: int = 1920, height: int = 1080) -> Image.Image:
    """Create a dummy RGB PIL Image."""
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    arr[100:200, 100:300] = 255  # white rectangle for non-trivial JPEG
    return Image.fromarray(arr, mode="RGB")


# ---------------------------------------------------------------------------
# TestFrameToCaptureResult
# ---------------------------------------------------------------------------

class TestFrameToCaptureResult:
    """Tests for frame_to_capture_result() helper function."""

    def test_output_dimensions_match_target(self) -> None:
        image = _make_rgb_image(1920, 1080)
        result = frame_to_capture_result(image, 1280, 720)
        assert result.width == 1280
        assert result.height == 720
        assert result.image.size == (1280, 720)

    def test_no_resize_when_matching(self) -> None:
        image = _make_rgb_image(1280, 720)
        result = frame_to_capture_result(image, 1280, 720)
        assert result.width == 1280
        assert result.height == 720

    def test_custom_resolution(self) -> None:
        image = _make_rgb_image(1920, 1080)
        result = frame_to_capture_result(image, 640, 480)
        assert result.width == 640
        assert result.height == 480
        assert result.image.size == (640, 480)

    def test_base64_is_valid_jpeg(self) -> None:
        image = _make_rgb_image()
        result = frame_to_capture_result(image)
        jpeg_bytes = base64.b64decode(result.base64_jpeg)
        # JPEG magic bytes: FF D8 FF
        assert jpeg_bytes[:2] == b"\xff\xd8"

    def test_size_bytes_matches_jpeg(self) -> None:
        image = _make_rgb_image()
        result = frame_to_capture_result(image)
        jpeg_bytes = base64.b64decode(result.base64_jpeg)
        assert result.size_bytes == len(jpeg_bytes)

    def test_timestamp_is_positive(self) -> None:
        image = _make_rgb_image()
        result = frame_to_capture_result(image)
        assert result.timestamp > 0

    def test_image_is_rgb(self) -> None:
        image = _make_rgb_image()
        result = frame_to_capture_result(image)
        assert result.image.mode == "RGB"

    def test_higher_quality_larger_size(self) -> None:
        image = _make_rgb_image()
        result_low = frame_to_capture_result(image, jpeg_quality=30)
        result_high = frame_to_capture_result(image, jpeg_quality=95)
        assert result_high.size_bytes > result_low.size_bytes

    def test_base64_round_trip(self) -> None:
        image = _make_rgb_image()
        result = frame_to_capture_result(image)
        jpeg_bytes = base64.b64decode(result.base64_jpeg)
        round_trip = Image.open(io.BytesIO(jpeg_bytes))
        assert round_trip.size == (result.width, result.height)


# ---------------------------------------------------------------------------
# TestCaptureSourceProtocol
# ---------------------------------------------------------------------------

class TestCaptureSourceProtocol:
    """Tests for the CaptureSource Protocol (runtime_checkable)."""

    def test_screen_capture_satisfies_protocol(self) -> None:
        from cyberraccoon.capture.screen_capture import ScreenCapture
        cap = ScreenCapture()
        assert isinstance(cap, CaptureSource)

    def test_camera_capture_satisfies_protocol(self) -> None:
        from cyberraccoon.capture.camera_capture import CameraCapture
        cam = CameraCapture()
        assert isinstance(cam, CaptureSource)

    def test_arbitrary_class_with_methods_satisfies(self) -> None:
        """Any class with open/capture/close/is_open satisfies the Protocol."""
        class FakeSource:
            def open(self) -> None: ...
            def capture(self) -> CaptureResult: ...
            def close(self) -> None: ...
            def is_open(self) -> bool: ...

        assert isinstance(FakeSource(), CaptureSource)

    def test_incomplete_class_does_not_satisfy(self) -> None:
        """Missing methods should not satisfy the Protocol."""
        class Incomplete:
            def open(self) -> None: ...
            # missing capture, close, is_open

        assert not isinstance(Incomplete(), CaptureSource)


# ---------------------------------------------------------------------------
# TestRegistry
# ---------------------------------------------------------------------------

class TestRegistry:
    """Tests for the source registry and create_capture factory."""

    def test_hdmi_registered(self) -> None:
        from cyberraccoon.capture import available_sources
        assert "hdmi" in available_sources()

    def test_csi_registered(self) -> None:
        from cyberraccoon.capture import available_sources
        assert "csi" in available_sources()

    def test_create_hdmi(self) -> None:
        from cyberraccoon.capture import create_capture
        from cyberraccoon.capture.screen_capture import ScreenCapture
        cap = create_capture("hdmi", device_index=0)
        assert isinstance(cap, ScreenCapture)

    def test_create_csi(self) -> None:
        from cyberraccoon.capture import create_capture
        from cyberraccoon.capture.csi_capture import CsiHdmiCapture
        cap = create_capture("csi")
        assert isinstance(cap, CsiHdmiCapture)

    def test_create_picamera(self) -> None:
        from cyberraccoon.capture import create_capture
        from cyberraccoon.capture.camera_capture import CameraCapture
        cap = create_capture("picamera", camera_index=0)
        assert isinstance(cap, CameraCapture)

    def test_create_unknown_raises(self) -> None:
        from cyberraccoon.capture import create_capture
        with pytest.raises(ValueError, match="Unknown capture source"):
            create_capture("nonexistent")

    def test_register_custom_source(self) -> None:
        from cyberraccoon.capture import register_source, create_capture, available_sources

        class DummyCapture:
            def __init__(self, **kwargs: object) -> None:
                self.kwargs = kwargs
            def open(self) -> None: ...
            def capture(self) -> CaptureResult: ...
            def close(self) -> None: ...
            def is_open(self) -> bool: ...

        register_source("dummy_test", DummyCapture)
        try:
            assert "dummy_test" in available_sources()
            cap = create_capture("dummy_test", foo="bar")
            assert isinstance(cap, DummyCapture)
            assert cap.kwargs == {"foo": "bar"}
        finally:
            # Clean up: remove from registry
            from cyberraccoon.capture import _REGISTRY
            _REGISTRY.pop("dummy_test", None)


# ---------------------------------------------------------------------------
# TestCaptureError
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# TestComputeFrameDiff
# ---------------------------------------------------------------------------

class TestComputeFrameDiff:
    """Tests for compute_frame_diff() utility."""

    def test_identical_frames_zero_diff(self) -> None:
        """Two identical frames should have 0% difference."""
        img = _make_rgb_image(640, 480)
        assert compute_frame_diff(img, img) == 0.0

    def test_completely_different_frames(self) -> None:
        """Black vs white frames should have ~100% difference."""
        black = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8), "RGB")
        white = Image.fromarray(np.full((100, 100, 3), 255, dtype=np.uint8), "RGB")
        diff = compute_frame_diff(black, white)
        assert diff > 99.0

    def test_small_change_gives_small_diff(self) -> None:
        """Changing a small region should give a small percentage."""
        arr = np.zeros((100, 100, 3), dtype=np.uint8)
        img_a = Image.fromarray(arr, "RGB")
        arr_b = arr.copy()
        arr_b[0:10, 0:10] = 255  # 10×10 = 100 pixels out of 10000 = 1%
        img_b = Image.fromarray(arr_b, "RGB")
        diff = compute_frame_diff(img_a, img_b)
        assert 0.5 < diff < 2.0

    def test_below_intensity_threshold_ignored(self) -> None:
        """Tiny intensity changes below threshold should be ignored."""
        arr_a = np.full((100, 100, 3), 100, dtype=np.uint8)
        arr_b = np.full((100, 100, 3), 105, dtype=np.uint8)  # diff=5, below default 10
        img_a = Image.fromarray(arr_a, "RGB")
        img_b = Image.fromarray(arr_b, "RGB")
        assert compute_frame_diff(img_a, img_b) == 0.0

    def test_different_sizes_handled(self) -> None:
        """Frames of different sizes should not crash."""
        img_a = _make_rgb_image(640, 480)
        img_b = _make_rgb_image(1280, 720)
        # Should not raise
        diff = compute_frame_diff(img_a, img_b)
        assert 0.0 <= diff <= 100.0


# ---------------------------------------------------------------------------
# TestCaptureError
# ---------------------------------------------------------------------------

class TestCaptureError:
    """Basic tests for CaptureError exception."""

    def test_is_exception(self) -> None:
        assert issubclass(CaptureError, Exception)

    def test_message(self) -> None:
        err = CaptureError("test failure")
        assert str(err) == "test failure"

    def test_can_be_raised_and_caught(self) -> None:
        with pytest.raises(CaptureError):
            raise CaptureError("boom")
