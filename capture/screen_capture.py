"""M1 Screen Capture — captures target computer screen via HDMI capture card.

Uses OpenCV to read from V4L2 device, resizes to 1280x720, outputs JPEG Base64.
"""

from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path

import cv2
from PIL import Image

# Canonical definitions live in capture.base.
# Re-export here for backward compatibility:
#   from capture.screen_capture import CaptureResult, CaptureError
from capture.base import (  # noqa: F401 — re-exported
    CaptureError,
    CaptureResult,
    frame_to_capture_result,
)

logger = logging.getLogger("M1.capture")


def find_capture_device(name_keyword: str = "USB") -> int:
    """Find a V4L2 capture device index by name keyword.

    Parses ``v4l2-ctl --list-devices`` output to find the first video device
    whose name contains *name_keyword* (case-insensitive).

    Useful because device indices (e.g. /dev/video0) can change after reboot
    when USB devices are re-enumerated.

    Args:
        name_keyword: Substring to match in device name (default: "USB").

    Returns:
        Integer device index of the first matching capture node.

    Raises:
        CaptureError: If no matching device is found or v4l2-ctl unavailable.

    Example::

        idx = find_capture_device("USB3 Video")  # finds "C1-1 USB3 Video"
        cap = ScreenCapture(device_index=idx)
    """
    try:
        result = subprocess.run(
            ["v4l2-ctl", "--list-devices"],
            capture_output=True, text=True, timeout=5,
        )
        output = result.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        raise CaptureError(f"v4l2-ctl not available: {e}") from e

    # Format: device-name lines are unindented, device-path lines are tab-indented.
    current_name = ""
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if not line.startswith(("\t", " ")):
            current_name = stripped          # e.g. "C1-1 USB3 Video: ... (usb-...)"
        elif stripped.startswith("/dev/video"):
            if name_keyword.lower() in current_name.lower():
                try:
                    idx = int(stripped.replace("/dev/video", ""))
                    logger.info(
                        "Found capture device '%s' at /dev/video%d",
                        current_name, idx,
                    )
                    return idx
                except ValueError:
                    continue

    raise CaptureError(
        f"No capture device matching '{name_keyword}' found.\n"
        f"Available devices:\n{output}"
    )


class ScreenCapture:
    """Captures frames from an HDMI capture card via OpenCV/V4L2.

    Uses MJPEG pixel format for better compatibility with USB capture cards.

    Usage::

        cap = ScreenCapture(device_index=0)
        cap.open()
        result = cap.capture()   # -> CaptureResult
        cap.close()

    Or auto-detect device by name::

        idx = find_capture_device("USB3 Video")
        cap = ScreenCapture(device_index=idx)
    """

    def __init__(
        self,
        device_index: int = 0,
        target_width: int = 1280,
        target_height: int = 720,
        jpeg_quality: int = 80,
    ) -> None:
        self._device_index = device_index
        self._target_width = target_width
        self._target_height = target_height
        self._jpeg_quality = jpeg_quality
        self._cap: cv2.VideoCapture | None = None

    def open(self) -> None:
        """Open the capture device via V4L2, requesting MJPEG format."""
        self._cap = cv2.VideoCapture(self._device_index, cv2.CAP_V4L2)

        if not self._cap.isOpened():
            raise CaptureError(
                f"Cannot open /dev/video{self._device_index}. "
                "Check: 1) HDMI cable connected  2) Capture card plugged in  "
                "3) Run 'ls /dev/video*' to verify device exists"
            )

        # Prefer MJPEG: better USB bandwidth efficiency and wider card compatibility
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        self._cap.set(cv2.CAP_PROP_FOURCC, fourcc)

        # Request highest resolution from capture card (actual depends on card)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fourcc_int = int(self._cap.get(cv2.CAP_PROP_FOURCC))
        fourcc_str = "".join(
            chr((actual_fourcc_int >> i) & 0xFF) for i in [0, 8, 16, 24]
        ).strip("\x00")
        logger.info(
            "Capture device opened: /dev/video%d (%dx%d %s), target: %dx%d",
            self._device_index, actual_w, actual_h, fourcc_str,
            self._target_width, self._target_height,
        )

    def capture(self) -> CaptureResult:
        """Capture a single frame.

        Discards 3 buffered frames first to ensure we get the latest image
        (V4L2 typically buffers 2-4 frames internally).

        Logs a WARNING if the frame is nearly black, which often indicates
        HDCP content protection blocking the signal. In that case, use an
        HDCP stripper device between the source and the capture card.
        """
        if self._cap is None or not self._cap.isOpened():
            raise CaptureError("Capture device not opened. Call open() first.")

        timestamp = time.monotonic()

        # Discard buffered frames to get the latest
        for _ in range(3):
            self._cap.grab()

        # Read the actual frame
        ret, frame = self._cap.read()
        if not ret or frame is None:
            raise CaptureError(
                "Failed to read frame. Capture card may be disconnected."
            )

        # Warn if nearly black — common symptom of HDCP blocking
        frame_max = int(frame.max())
        if frame_max < 10:
            logger.warning(
                "Frame nearly black (max_pixel=%d). Likely HDCP content protection. "
                "Fix: insert an HDCP stripper between source and capture card.",
                frame_max,
            )

        # OpenCV BGR -> RGB -> PIL Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)

        result = frame_to_capture_result(
            image, self._target_width, self._target_height, self._jpeg_quality,
        )
        # Override timestamp with the one taken before buffer flush
        result.timestamp = timestamp

        logger.debug(
            "Frame captured: %dx%d, JPEG %.1fKB, max_pixel=%d",
            self._target_width, self._target_height,
            result.size_bytes / 1024, frame_max,
        )

        return result

    def close(self) -> None:
        """Release the capture device."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info("Capture device closed")

    @property
    def v4l2_device_name(self) -> str:
        """Read the V4L2 device name from sysfs (e.g. 'USB3 Video')."""
        sysfs = Path(f"/sys/class/video4linux/video{self._device_index}/name")
        try:
            return sysfs.read_text().strip()
        except OSError:
            return f"/dev/video{self._device_index}"

    def is_open(self) -> bool:
        """Check if the capture device is currently open."""
        return self._cap is not None and self._cap.isOpened()
