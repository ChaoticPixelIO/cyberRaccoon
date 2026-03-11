"""M1 Capture — base types, interface definition, and shared utilities.

Defines the public types used across all capture sources:
- ``CaptureResult``: output of a single frame capture
- ``CaptureError``: exception for capture failures
- ``CaptureSource``: Protocol that all capture backends must satisfy

Also provides ``frame_to_capture_result()`` — a shared helper that converts
a PIL RGB image into a ``CaptureResult`` (resize → JPEG compress → Base64).
Each capture backend only needs to obtain an RGB frame; the final processing
is delegated to this helper to avoid code duplication.
"""

from __future__ import annotations

import base64
import io
import logging
import time
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from PIL import Image

logger = logging.getLogger("M1.base")


# ---------------------------------------------------------------------------
# Public exceptions
# ---------------------------------------------------------------------------

class CaptureError(Exception):
    """Raised when screen capture fails (device disconnected, read error, etc.)."""


# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------

@dataclass
class CaptureResult:
    """Output of a single screen capture.

    Attributes:
        image:       PIL Image in RGB mode at target resolution.
        base64_jpeg: JPEG-compressed, Base64-encoded string (for LLM API).
        width:       Image width in pixels (default 1280).
        height:      Image height in pixels (default 720).
        timestamp:   Capture time (``time.monotonic``).
        size_bytes:  JPEG byte count (for cost / bandwidth monitoring).
    """

    image: Image.Image
    base64_jpeg: str
    width: int
    height: int
    timestamp: float
    size_bytes: int


# ---------------------------------------------------------------------------
# Capture source interface
# ---------------------------------------------------------------------------

@runtime_checkable
class CaptureSource(Protocol):
    """Interface that all capture backends must satisfy.

    Lifecycle::

        source = SomeCapture(...)   # configure
        source.open()               # acquire resources
        result = source.capture()   # grab frame  (repeat)
        source.close()              # release resources

    Implementations must provide all four methods. Constructor signatures
    vary by backend (device index, RTP port, etc.) so they are *not* part
    of the Protocol — only the operational methods are.
    """

    def open(self) -> None:
        """Initialize hardware / software and prepare for capture.

        Raises:
            CaptureError: If initialization fails.
        """
        ...

    def capture(self) -> CaptureResult:
        """Capture a single frame and return it as a ``CaptureResult``.

        Raises:
            CaptureError: If not opened or frame capture fails.
        """
        ...

    def close(self) -> None:
        """Release all resources. Safe to call multiple times."""
        ...

    def is_open(self) -> bool:
        """Return ``True`` if the source is ready for ``capture()``."""
        ...


# ---------------------------------------------------------------------------
# Shared frame processing
# ---------------------------------------------------------------------------

def frame_to_capture_result(
    frame_rgb: Image.Image,
    target_width: int = 1280,
    target_height: int = 720,
    jpeg_quality: int = 80,
) -> CaptureResult:
    """Convert a PIL RGB image into a :class:`CaptureResult`.

    Handles resize (if dimensions differ), JPEG compression, and Base64
    encoding.  All capture backends share this final processing step;
    each backend only needs to produce an RGB ``Image``.

    Args:
        frame_rgb:    PIL Image (must be RGB mode).
        target_width: Desired output width (default 1280).
        target_height: Desired output height (default 720).
        jpeg_quality: JPEG compression quality 1-100 (default 80).

    Returns:
        A fully populated ``CaptureResult``.
    """
    timestamp = time.monotonic()

    # Resize to target resolution if needed
    if frame_rgb.size != (target_width, target_height):
        frame_rgb = frame_rgb.resize(
            (target_width, target_height),
            Image.LANCZOS,
        )

    # JPEG compress + Base64 encode
    buffer = io.BytesIO()
    frame_rgb.save(buffer, format="JPEG", quality=jpeg_quality)
    jpeg_bytes = buffer.getvalue()
    b64_str = base64.b64encode(jpeg_bytes).decode("ascii")

    return CaptureResult(
        image=frame_rgb,
        base64_jpeg=b64_str,
        width=target_width,
        height=target_height,
        timestamp=timestamp,
        size_bytes=len(jpeg_bytes),
    )


def compute_frame_diff(
    frame_a: Image.Image,
    frame_b: Image.Image,
    intensity_threshold: int = 10,
) -> float:
    """Return percentage of pixels that differ significantly between two frames.

    Converts both images to grayscale, computes the absolute per-pixel
    difference, and counts how many exceed *intensity_threshold*.

    Args:
        frame_a: First PIL Image (any mode; converted to grayscale).
        frame_b: Second PIL Image (any mode; converted to grayscale).
        intensity_threshold: Minimum absolute difference (0-255) to count
            a pixel as "changed".  Default 10 filters out JPEG artefacts
            and minor sensor noise.

    Returns:
        Percentage (0.0 – 100.0) of pixels that changed.
    """
    gray_a = np.asarray(frame_a.convert("L"), dtype=np.int16)
    gray_b = np.asarray(frame_b.convert("L"), dtype=np.int16)

    # Resize if dimensions don't match (shouldn't happen, but be safe)
    if gray_a.shape != gray_b.shape:
        logger.warning(
            "Frame size mismatch: %s vs %s, resizing second frame",
            gray_a.shape, gray_b.shape,
        )
        frame_b_resized = frame_b.resize(frame_a.size, Image.LANCZOS)
        gray_b = np.asarray(frame_b_resized.convert("L"), dtype=np.int16)

    diff = np.abs(gray_a - gray_b)
    changed = np.count_nonzero(diff > intensity_threshold)
    total = gray_a.size  # total number of pixels
    if total == 0:
        return 100.0
    return (changed / total) * 100.0
