"""CyberRaccoon -- AI computer control via HDMI capture + USB HID."""
from __future__ import annotations

try:
    from importlib.metadata import version

    __version__ = version("cyberraccoon")
except Exception:
    __version__ = "0.0.0+unknown"  # fallback when running from source without install
