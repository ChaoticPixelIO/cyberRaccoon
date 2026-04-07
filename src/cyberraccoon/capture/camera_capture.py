"""M1 CSI Camera Capture — captures target screen via Raspberry Pi camera.

Uses picamera2 to capture frames from the Pi CSI camera pointed at the target
computer's screen. This is an alternative to HDMI capture (ScreenCapture) for
situations where HDCP blocks the HDMI signal or no capture card is available.

The interface is identical to ScreenCapture: open/capture/close/is_open,
returning the same CaptureResult dataclass.

Note: picamera2 is only available on Raspberry Pi. On macOS/other platforms,
this module can still be imported (the import is deferred to open()), but
open() will raise CaptureError.
"""

from __future__ import annotations

import logging
from typing import Any

from PIL import Image

from capture.base import CaptureError, CaptureResult, frame_to_capture_result

logger = logging.getLogger("M1.camera")


class CameraCapture:
    """Captures frames from a Pi CSI camera via picamera2.

    Provides the same interface as ScreenCapture so they can be used
    interchangeably by M2 VisionAgent.

    Usage::

        cam = CameraCapture(camera_index=0)
        cam.open()
        result = cam.capture()   # -> CaptureResult (same as ScreenCapture)
        cam.close()
    """

    def __init__(
        self,
        camera_index: int = 0,
        target_width: int = 1280,
        target_height: int = 720,
        jpeg_quality: int = 80,
    ) -> None:
        self._camera_index = camera_index
        self._target_width = target_width
        self._target_height = target_height
        self._jpeg_quality = jpeg_quality
        self._picam: Any = None  # Picamera2 instance (Any to avoid import)

    def open(self) -> None:
        """Initialize picamera2, configure 1920×1080 RGB, and start capture.

        Raises:
            CaptureError: If picamera2 is not available or camera fails to open.
        """
        try:
            from picamera2 import Picamera2
        except ImportError as e:
            raise CaptureError(
                "picamera2 is not available. This module requires a Raspberry Pi "
                "with picamera2 installed. Install via: sudo apt install python3-picamera2"
            ) from e

        try:
            self._picam = Picamera2(self._camera_index)
        except Exception as e:
            self._picam = None
            raise CaptureError(
                f"Cannot open CSI camera {self._camera_index}: {e}"
            ) from e

        config = self._picam.create_still_configuration(
            main={"size": (1920, 1080), "format": "RGB888"}
        )
        self._picam.configure(config)
        self._picam.start()

        logger.info(
            "CSI camera %d opened (1920x1080 RGB), target: %dx%d",
            self._camera_index, self._target_width, self._target_height,
        )

    def capture(self) -> CaptureResult:
        """Capture a single frame from the CSI camera.

        picamera2 returns RGB numpy arrays directly (no BGR conversion needed).

        Returns:
            CaptureResult with the same fields as ScreenCapture output.

        Raises:
            CaptureError: If camera is not opened or frame capture fails.
        """
        if self._picam is None:
            raise CaptureError("Camera not opened. Call open() first.")

        try:
            frame = self._picam.capture_array("main")
        except Exception as e:
            raise CaptureError(f"Failed to capture frame: {e}") from e

        # picamera2 returns RGB directly — convert to PIL Image
        image = Image.fromarray(frame)

        result = frame_to_capture_result(
            image, self._target_width, self._target_height, self._jpeg_quality,
        )

        logger.debug(
            "Camera frame captured: %dx%d, JPEG %.1fKB",
            self._target_width, self._target_height,
            result.size_bytes / 1024,
        )

        return result

    def close(self) -> None:
        """Stop and release the camera. Safe to call multiple times."""
        if self._picam is not None:
            try:
                self._picam.stop()
                self._picam.close()
            except Exception as e:
                logger.warning("Error closing camera: %s", e)
            self._picam = None
            logger.info("CSI camera closed")

    def is_open(self) -> bool:
        """Check if the camera is currently open and streaming."""
        return self._picam is not None
