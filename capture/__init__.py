"""M1 Capture — screen capture with pluggable backends.

Provides a unified interface (:class:`CaptureSource`) for all capture
backends and a registry / factory for instantiating them by name::

    from capture import create_capture

    cap = create_capture("hdmi", device_index=0)
    cap.open()
    result = cap.capture()
    cap.close()

Built-in sources:

======== ======================== ========================================
Name     Class                    Hardware
======== ======================== ========================================
hdmi     ScreenCapture            HDMI USB capture card (V4L2/MJPEG)
csi      CsiHdmiCapture           TC358743 HDMI-to-CSI bridge (V4L2/BGR)
airplay  AirPlayCapture           AirPlay mirroring via uxplay
picamera CameraCapture            Raspberry Pi CSI camera (picamera2)
======== ======================== ========================================

To add a new backend, implement the :class:`CaptureSource` protocol and
call ``register_source("name", YourClass)``.
"""

from capture.base import (
    CaptureError,
    CaptureResult,
    CaptureSource,
    frame_to_capture_result,
)
from capture.camera_capture import CameraCapture
from capture.csi_capture import CsiHdmiCapture
from capture.screen_capture import ScreenCapture, find_capture_device

# ---------------------------------------------------------------------------
# Source registry + factory
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, type] = {}


def register_source(name: str, cls: type) -> None:
    """Register a capture source class under *name*.

    Args:
        name: Short identifier used in CLI ``--source`` (e.g. ``"hdmi"``).
        cls:  Class that satisfies the :class:`CaptureSource` protocol.
    """
    _REGISTRY[name] = cls


def available_sources() -> list[str]:
    """Return the names of all registered capture sources."""
    return sorted(_REGISTRY)


def create_capture(source: str, **kwargs: object) -> CaptureSource:
    """Instantiate a capture source by *name*.

    Args:
        source:  Registered name (``"hdmi"``, ``"csi"``, ``"airplay"``, …).
        **kwargs: Forwarded to the source class constructor.

    Returns:
        An uninitialised capture source (call ``.open()`` before use).

    Raises:
        ValueError: If *source* is not registered.
    """
    if source not in _REGISTRY:
        available = ", ".join(available_sources())
        raise ValueError(
            f"Unknown capture source: '{source}'. "
            f"Available: {available}"
        )
    return _REGISTRY[source](**kwargs)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Built-in source registration
# ---------------------------------------------------------------------------

register_source("hdmi", ScreenCapture)
register_source("csi", CsiHdmiCapture)
register_source("picamera", CameraCapture)

# AirPlay is registered lazily to avoid import errors on systems without
# GStreamer / uxplay.  The import is wrapped in try/except so that the
# capture package remains importable everywhere.
try:
    from capture.airplay_capture import AirPlayCapture

    register_source("airplay", AirPlayCapture)
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Protocol + types
    "CaptureSource",
    "CaptureResult",
    "CaptureError",
    "frame_to_capture_result",
    # Concrete sources
    "ScreenCapture",
    "CsiHdmiCapture",
    "CameraCapture",
    # Factory
    "register_source",
    "available_sources",
    "create_capture",
    # Helpers
    "find_capture_device",
]
