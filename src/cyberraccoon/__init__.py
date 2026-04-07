"""CyberRaccoon -- AI computer control via HDMI capture + USB HID."""
from __future__ import annotations

try:
    from importlib.metadata import version

    __version__ = version("cyberraccoon")
except Exception:
    __version__ = "0.1.0"  # fallback when not installed
