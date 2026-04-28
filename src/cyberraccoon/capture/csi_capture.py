"""M1 CSI HDMI Capture — captures target screen via TC358743 HDMI-to-CSI bridge.

Uses V4L2 media controller API (via media-ctl and v4l2-ctl subprocesses) to
configure the TC358743 hardware pipeline, then OpenCV with V4L2 backend to
capture BGR frames.

Unlike USB HDMI capture cards (ScreenCapture) which present as standard UVC
devices, the CSI bridge requires explicit media pipeline configuration:
EDID load → DV timings → media-ctl link/format setup → V4L2 capture.

Unlike picamera2-based CameraCapture, this works with the TC358743 bridge
chip which libcamera does not support (no tuning file).

Requires:
    - dtoverlay=tc358743-pi5 in /boot/firmware/config.txt (Pi 5)
      Use cam0 for CAM0 (2-lane) or 4lane=1 for CAM1 (4-lane, 1080p capable)
    - v4l2-ctl and media-ctl installed (sudo apt install v4l-utils)
    - TC358743 HDMI-to-CSI board connected to the matching CAM port
    - HDMI source connected and outputting video

Note: importable on any platform (macOS, etc.) — hardware checks happen in
open(), not at import time. open() will fail with CaptureError on non-Pi.

Usage::

    from cyberraccoon.capture import create_capture

    cap = create_capture("csi")
    cap.open()                    # pipeline setup + OpenCV open
    result = cap.capture()        # -> CaptureResult (BGR frame)
    cap.close()
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import struct
import subprocess
import tempfile
import time

import cv2
import numpy as np
from PIL import Image

from cyberraccoon.capture.base import CaptureError, CaptureResult, frame_to_capture_result

logger = logging.getLogger("M1.csi_hdmi")


# ---------------------------------------------------------------------------
# 720p EDID — forces HDMI source to output 1280×720@60Hz
# ---------------------------------------------------------------------------
# Pi 5 CSI has 2 lanes (max ~1.94 Gbps).  1080p60 BGR needs ~3 Gbps and
# crashes the kernel.  This EDID advertises ONLY 720p60 so the source cannot
# choose a higher resolution.
#
# Structure (128-byte base EDID, no extensions):
#   Bytes  0-7:   Fixed header
#   Bytes  8-17:  Manufacturer / product / serial / date / EDID version
#   Bytes 18-24:  Basic display parameters (digital input)
#   Bytes 25-34:  Chromaticity (sRGB)
#   Bytes 35-53:  Established + standard timings (none)
#   Bytes 54-71:  Detailed timing: 1280×720@60Hz (74.25 MHz pixel clock)
#   Bytes 72-89:  Monitor name: "CyberRaccoon"
#   Bytes 90-107: Monitor range limits (59-61 Hz V, 15-50 kHz H)
#   Bytes 108-125: Dummy descriptor
#   Byte  126:    Extension count = 0
#   Byte  127:    Checksum
_EDID_720P_BYTES = bytes([
    # Header
    0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00,
    # Manufacturer "CYR" (CyberRaccoon), product 0x0001
    0x0E, 0xD9, 0x01, 0x00, 0x01, 0x00, 0x00, 0x00,
    # Week 1, Year 2025 (1990+35), EDID 1.3
    0x01, 0x23, 0x01, 0x03,
    # Digital input (DFP 1.x), no size, gamma 2.2, RGB+preferred
    0x80, 0x00, 0x00, 0x78, 0x0A,
    # Chromaticity (sRGB standard)
    0xEE, 0x91, 0xA3, 0x54, 0x4C, 0x99, 0x26, 0x0F, 0x50, 0x54,
    # Established timings: none
    0x00, 0x00, 0x00,
    # Standard timings: all unused (0x0101)
    0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
    0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
    # Detailed Timing Descriptor: 1280×720 @ 60 Hz
    # Pixel clock: 74.25 MHz = 7425 × 10 kHz → 0x1D09 (little-endian)
    0x09, 0x1D,
    # HActive=1280 low8=0x00, HBlanking=370 low8=0x72
    0x00, 0x72,
    # HActive[11:8]=5 (1280=0x500), HBlanking[11:8]=1 (370=0x172)
    0x51,
    # VActive=720 low8=0xD0 (720=0x2D0), VBlanking=30 low8=0x1E
    0xD0, 0x1E,
    # VActive[11:8]=2, VBlanking[11:8]=0
    0x20,
    # HSyncOffset=110 low8=0x6E, HSyncWidth=40 low8=0x28
    0x6E, 0x28,
    # VSyncOffset=5 [3:0], VSyncWidth=5 [3:0]
    0x55,
    # High bits of sync: all 0
    0x00,
    # Image size (mm): 0 (no physical size)
    0x00, 0x00, 0x00,
    # Border: 0, 0
    0x00, 0x00,
    # Non-interlaced, no stereo, digital separate sync, +H +V
    0x1E,
    # Monitor Name Descriptor
    0x00, 0x00, 0x00, 0xFC, 0x00,
    # "CyberRaccoon" + newline + padding
    0x43, 0x79, 0x62, 0x65, 0x72, 0x52, 0x61, 0x63,
    0x63, 0x6F, 0x6F, 0x6E, 0x0A,
    # Monitor Range Limits Descriptor
    0x00, 0x00, 0x00, 0xFD, 0x00,
    # Min V=59, Max V=61, Min H=15 kHz, Max H=50 kHz, MaxPixelClock/10=8 (80 MHz)
    0x3B, 0x3D, 0x0F, 0x32, 0x08,
    # GTF not supported + padding
    0x00, 0x0A, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
    # Dummy Descriptor
    0x00, 0x00, 0x00, 0x10, 0x00,
    0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
    0x20, 0x20, 0x20, 0x20, 0x20,
    # Extension count
    0x00,
    # Checksum placeholder (computed below)
    0x00,
])

# Compute correct checksum: all 128 bytes must sum to 0 mod 256
_edid = bytearray(_EDID_720P_BYTES)
_edid[127] = (256 - (sum(_edid[:127]) % 256)) % 256
EDID_720P = bytes(_edid)
del _edid


# ---------------------------------------------------------------------------
# 1080p EDID — forces HDMI source to output 1920×1080@60Hz
# ---------------------------------------------------------------------------
# Requires 4-lane CSI (CAM1 on Pi 5).  1080p60 BGR at 24bpp needs ~3 Gbps;
# 4 lanes at ~1 Gbps each provide ~4 Gbps.
#
# Structure matches EDID_720P above but with 1080p60 detailed timing
# (148.5 MHz pixel clock) and updated range limits.
_EDID_1080P_BYTES = bytes([
    # Header
    0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00,
    # Manufacturer "CYR" (CyberRaccoon), product 0x0001
    0x0E, 0xD9, 0x01, 0x00, 0x01, 0x00, 0x00, 0x00,
    # Week 1, Year 2025 (1990+35), EDID 1.3
    0x01, 0x23, 0x01, 0x03,
    # Digital input (DFP 1.x), no size, gamma 2.2, RGB+preferred
    0x80, 0x00, 0x00, 0x78, 0x0A,
    # Chromaticity (sRGB standard)
    0xEE, 0x91, 0xA3, 0x54, 0x4C, 0x99, 0x26, 0x0F, 0x50, 0x54,
    # Established timings: none
    0x00, 0x00, 0x00,
    # Standard timings: all unused (0x0101)
    0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
    0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
    # Detailed Timing Descriptor: 1920×1080 @ 60 Hz
    # Pixel clock: 148.5 MHz = 14850 × 10 kHz → 0x3A02 (little-endian)
    0x02, 0x3A,
    # HActive=1920 low8=0x80, HBlanking=280 low8=0x18
    0x80, 0x18,
    # HActive[11:8]=7 (1920=0x780), HBlanking[11:8]=1 (280=0x118)
    0x71,
    # VActive=1080 low8=0x38 (1080=0x438), VBlanking=45 low8=0x2D
    0x38, 0x2D,
    # VActive[11:8]=4, VBlanking[11:8]=0
    0x40,
    # HSyncOffset=88 low8=0x58, HSyncWidth=44 low8=0x2C
    0x58, 0x2C,
    # VSyncOffset=4 [3:0], VSyncWidth=5 [3:0]
    0x45,
    # High bits of sync: all 0
    0x00,
    # Image size (mm): 0 (no physical size)
    0x00, 0x00, 0x00,
    # Border: 0, 0
    0x00, 0x00,
    # Non-interlaced, no stereo, digital separate sync, +H +V
    0x1E,
    # Monitor Name Descriptor
    0x00, 0x00, 0x00, 0xFC, 0x00,
    # "CyberRaccoon" + newline + padding
    0x43, 0x79, 0x62, 0x65, 0x72, 0x52, 0x61, 0x63,
    0x63, 0x6F, 0x6F, 0x6E, 0x0A,
    # Monitor Range Limits Descriptor
    0x00, 0x00, 0x00, 0xFD, 0x00,
    # Min V=59, Max V=61, Min H=15 kHz, Max H=70 kHz, MaxPixelClock/10=15 (150 MHz)
    0x3B, 0x3D, 0x0F, 0x46, 0x0F,
    # GTF not supported + padding
    0x00, 0x0A, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
    # Dummy Descriptor
    0x00, 0x00, 0x00, 0x10, 0x00,
    0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
    0x20, 0x20, 0x20, 0x20, 0x20,
    # Extension count
    0x00,
    # Checksum placeholder (computed below)
    0x00,
])

_edid = bytearray(_EDID_1080P_BYTES)
_edid[127] = (256 - (sum(_edid[:127]) % 256)) % 256
EDID_1080P = bytes(_edid)
del _edid


def _edid_to_hex_file_content(edid: bytes) -> str:
    """Convert EDID bytes to the hex text format expected by v4l2-ctl."""
    lines = []
    for i in range(0, len(edid), 16):
        lines.append("".join(f"{b:02x}" for b in edid[i : i + 16]))
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CsiHdmiCapture
# ---------------------------------------------------------------------------

class CsiHdmiCapture:
    """Captures frames from a TC358743 HDMI-to-CSI bridge via V4L2.

    All pipeline setup (media device discovery, EDID, DV timings, media-ctl
    links/formats) happens automatically in :meth:`open`.  The user only
    needs to ensure the hardware and dtoverlay are configured.

    Usage::

        cap = CsiHdmiCapture()
        cap.open()
        result = cap.capture()   # -> CaptureResult
        cap.close()
    """

    # NOTE: target_width/target_height default to 1280x720 — NOT 1920x1080 like
    # the other capture sources. The TC358743 on 2-lane CAM0 is bandwidth-limited
    # to 720p (see _detect_lane_count / _max_capture_height below). 4-lane CAM1
    # users can pass target_width=1920, target_height=1080 to the constructor.
    def __init__(
        self,
        target_width: int = 1280,
        target_height: int = 720,
        jpeg_quality: int = 80,
        signal_timeout: float = 15.0,
    ) -> None:
        self._target_width = target_width
        self._target_height = target_height
        self._jpeg_quality = jpeg_quality
        self._signal_timeout = signal_timeout

        # Discovered during open()
        self._media_device: str | None = None
        self._subdev_path: str | None = None
        self._video_device: str | None = None
        self._signal_width: int = 0
        self._signal_height: int = 0
        self._lane_count: int = 0
        self._max_capture_height: int = 720
        self._bytesperline: int = 0
        self._opened: bool = False

    # ------------------------------------------------------------------
    # CaptureSource protocol
    # ------------------------------------------------------------------

    def open(self) -> None:
        """Discover hardware, configure media pipeline, and open V4L2 capture.

        Raises:
            CaptureError: If TC358743 not detected, no HDMI signal, pipeline
                setup fails, or V4L2 device cannot be opened.
        """
        self._check_tools()

        # Discover media topology
        topology = self._discover_media_device()
        self._subdev_path = self._parse_subdev(topology)
        self._video_device = self._parse_video_device(topology)

        logger.info(
            "TC358743 found: media=%s subdev=%s video=%s",
            self._media_device, self._subdev_path, self._video_device,
        )

        # Detect CSI lane count to determine max capture resolution.
        self._lane_count = self._detect_lane_count(topology)
        self._max_capture_height = 1080 if self._lane_count >= 4 else 720
        logger.info(
            "CSI lane count: %d, max capture height: %dp",
            self._lane_count, self._max_capture_height,
        )

        # Configure HDMI input.  Only load EDID if the source isn't already
        # at a safe resolution — EDID re-negotiation breaks the CSI data path
        # on the TC358743 (DV timings update but video frames stay black).
        #
        # EDID selection: 4-lane CSI loads a 1080p EDID (which also allows
        # 720p — the source picks its preferred mode).  2-lane CSI loads a
        # 720p-only EDID to prevent the source from choosing 1080p.
        current = self._query_current_signal()
        edid_loaded = False
        if current and current[1] <= self._max_capture_height:
            logger.info(
                "Source already at %dx%d — skipping EDID load", *current,
            )
            self._signal_width, self._signal_height = current
        else:
            edid_label = "1080p" if self._max_capture_height >= 1080 else "720p"
            if current:
                logger.info(
                    "Source at %dx%d (too high) — loading %s EDID",
                    *current, edid_label,
                )
            else:
                logger.info("No signal — loading %s EDID", edid_label)
            self._load_edid()
            self._wait_for_signal()
            edid_loaded = True
        self._set_dv_timings()

        # Configure CSI media pipeline
        self._configure_pipeline()

        # Set BGR3 pixel format and resolution on the V4L2 video device.
        # We use v4l2-ctl rather than OpenCV because OpenCV's CAP_V4L2
        # backend assumes a tightly packed `width * 3` row stride and
        # ignores the V4L2-reported ``bytesperline`` — which the RP1-CFE
        # CSI receiver pads to 16 bytes.  Non-aligned source widths (e.g.
        # 1720, when a Windows GPU picks a half-ultrawide fallback) then
        # produce diagonally sheared frames.
        self._run_cmd(
            [
                "v4l2-ctl", "-d", self._video_device,  # type: ignore[list-item]
                f"--set-fmt-video=width={self._signal_width},"
                f"height={self._signal_height},pixelformat=BGR3",
            ],
            check=True, timeout=5.0,
        )

        fmt = self._query_video_format()
        actual_w, actual_h = fmt["width"], fmt["height"]
        if actual_w != self._signal_width or actual_h != self._signal_height:
            raise CaptureError(
                f"V4L2 format mismatch: requested {self._signal_width}x"
                f"{self._signal_height} but device reports {actual_w}x"
                f"{actual_h}. The media pipeline may not have applied "
                "correctly. Try: reboot the Pi, or check "
                "'media-ctl -p' to verify the pipeline state."
            )
        self._bytesperline = fmt["bytesperline"]
        padding = self._bytesperline - actual_w * 3
        if padding < 0:
            raise CaptureError(
                f"V4L2 reports bytesperline={self._bytesperline} smaller "
                f"than width*3={actual_w * 3} for BGR3 — driver bug?"
            )
        if padding > 0:
            logger.info(
                "V4L2 row padding: %d bytes per row (%dx3=%d + %d pad "
                "= %d bytesperline). Will strip per-row on capture.",
                padding, actual_w, actual_w * 3, padding,
                self._bytesperline,
            )

        self._opened = True

        # Warmup: discard frames until we get a non-black one.  After a
        # resolution switch the CSI receiver may output blank frames for a
        # short period while the DMA buffers fill and the signal stabilises.
        # After an EDID load the HPD renegotiation needs longer to settle.
        warmup_timeout = 10.0 if edid_loaded else 5.0
        warmup_deadline = time.monotonic() + warmup_timeout
        while time.monotonic() < warmup_deadline:
            try:
                frame = self._read_frame(skip=2)
            except CaptureError:
                time.sleep(0.1)
                continue
            if int(frame.max()) > 10:
                logger.info("Pipeline warmed up (max_pixel=%d)", int(frame.max()))
                break
            time.sleep(0.05)
        else:
            self._opened = False
            raise CaptureError(
                "CSI pipeline opened but only producing black frames after "
                f"{warmup_timeout:.0f}s warmup. The HDMI signal may have dropped during setup, "
                "or the CSI data path is broken. Try: disconnect and "
                "reconnect the HDMI cable, then retry."
            )

        logger.info(
            "CSI HDMI capture opened on %s (%dx%d BGR), target: %dx%d",
            self._video_device, actual_w, actual_h,
            self._target_width, self._target_height,
        )

    def capture(self) -> CaptureResult:
        """Capture a single frame from the TC358743 HDMI-CSI bridge.

        Discards 3 buffered frames first to ensure the latest image
        (V4L2 typically buffers 2-4 frames internally).
        """
        if not self._opened:
            raise CaptureError(
                "CSI HDMI capture not opened. Call open() first."
            )

        timestamp = time.monotonic()
        frame = self._read_frame(skip=3)

        # Warn on nearly-black frames
        frame_max = int(frame.max())
        if frame_max < 10:
            logger.warning(
                "Frame nearly black (max_pixel=%d). Possible causes: "
                "1) HDCP content protection  2) HDMI cable disconnected  "
                "3) Source not outputting video",
                frame_max,
            )

        # BGR (V4L2) -> RGB -> PIL Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)

        result = frame_to_capture_result(
            image, self._target_width, self._target_height, self._jpeg_quality,
        )
        result.timestamp = timestamp

        logger.debug(
            "Frame captured: %dx%d, JPEG %.1fKB, max_pixel=%d",
            self._target_width, self._target_height,
            result.size_bytes / 1024, frame_max,
        )

        return result

    def close(self) -> None:
        """Release capture state. Safe to call multiple times."""
        if self._opened:
            self._opened = False
            logger.info("CSI HDMI capture closed")

    def is_open(self) -> bool:
        """Check if the capture device is currently open."""
        return self._opened

    # ------------------------------------------------------------------
    # V4L2 streaming helpers
    # ------------------------------------------------------------------

    def _query_video_format(self) -> dict[str, int]:
        """Query the current V4L2 format and return width/height/bytesperline.

        Runs ``v4l2-ctl --get-fmt-video`` and parses the human-readable output.
        """
        result = self._run_cmd(
            ["v4l2-ctl", "-d", self._video_device, "--get-fmt-video"],  # type: ignore[list-item]
            check=True, timeout=5.0,
        )
        fmt: dict[str, int] = {}
        for line in result.stdout.splitlines():
            line = line.strip()
            if line.startswith("Width/Height"):
                m = re.search(r":\s*(\d+)/(\d+)", line)
                if m:
                    fmt["width"] = int(m.group(1))
                    fmt["height"] = int(m.group(2))
            elif line.startswith("Bytes per Line"):
                m = re.search(r":\s*(\d+)", line)
                if m:
                    fmt["bytesperline"] = int(m.group(1))
        missing = {"width", "height", "bytesperline"} - fmt.keys()
        if missing:
            raise CaptureError(
                f"Could not parse v4l2-ctl --get-fmt-video output "
                f"(missing: {sorted(missing)}):\n{result.stdout}"
            )
        return fmt

    def _read_frame(self, skip: int = 3) -> np.ndarray:
        """Capture a single frame via ``v4l2-ctl --stream-mmap``.

        Grabs ``skip + 1`` frames and returns only the last one, which
        drains stale kernel-side buffers so the returned image reflects
        the current screen (V4L2 typically buffers 2-4 frames internally).

        The raw buffer is reshaped honoring the V4L2-reported
        ``bytesperline`` and sliced to the real pixel width, so any
        per-row padding added by the RP1-CFE CSI receiver (16-byte
        alignment) is stripped before conversion.

        Returns:
            An ``H × W × 3`` BGR ``np.uint8`` array.
        """
        frame_size = self._bytesperline * self._signal_height
        total = (skip + 1) * frame_size
        proc = subprocess.run(
            [
                "v4l2-ctl", "-d", self._video_device,  # type: ignore[list-item]
                "--stream-mmap",
                f"--stream-count={skip + 1}",
                f"--stream-skip={skip}",
                "--stream-to=-",
            ],
            capture_output=True, timeout=10.0,
        )
        if proc.returncode != 0:
            raise CaptureError(
                f"v4l2-ctl streaming failed (rc={proc.returncode}): "
                f"{proc.stderr.decode('utf-8', errors='replace').strip()}"
            )
        raw = proc.stdout
        if len(raw) < frame_size:
            raise CaptureError(
                f"v4l2-ctl returned {len(raw)} bytes, expected at least "
                f"{frame_size} ({self._signal_height} rows × "
                f"{self._bytesperline} bytes). HDMI signal may have dropped."
            )
        # Take the last full frame; --stream-skip should already drain
        # stale buffers but being defensive costs nothing.
        frame_raw = raw[-frame_size:] if len(raw) >= total else raw[-frame_size:]
        arr = np.frombuffer(frame_raw, dtype=np.uint8).reshape(
            self._signal_height, self._bytesperline,
        )
        # Strip per-row padding, reshape to (H, W, 3) BGR.
        arr = arr[:, : self._signal_width * 3].reshape(
            self._signal_height, self._signal_width, 3,
        )
        return arr

    # ------------------------------------------------------------------
    # Pipeline setup helpers
    # ------------------------------------------------------------------

    def _check_tools(self) -> None:
        """Verify v4l2-ctl and media-ctl are installed."""
        for tool in ("v4l2-ctl", "media-ctl"):
            if shutil.which(tool) is None:
                raise CaptureError(
                    f"Required tool not found: {tool}. "
                    "Install v4l-utils: sudo apt install v4l-utils"
                )

    def _discover_media_device(self) -> str:
        """Find the rp1-cfe media device containing the TC358743.

        Iterates /dev/media0 through /dev/media9 looking for the rp1-cfe
        driver with a tc358743 entity in its topology.

        Returns:
            The full ``media-ctl -p`` topology output for the matched device.

        Raises:
            CaptureError: If no matching media device is found.
        """
        for i in range(10):
            dev = f"/dev/media{i}"
            if not os.path.exists(dev):
                continue
            result = self._run_cmd(
                ["media-ctl", "-d", dev, "-p"],
                check=False, timeout=5.0,
            )
            if result.returncode != 0:
                continue
            output = result.stdout
            if "rp1-cfe" in output and "tc358743" in output:
                self._media_device = dev
                logger.info("Found rp1-cfe with TC358743 at %s", dev)
                return output

        raise CaptureError(
            "TC358743 HDMI-CSI bridge not detected. Check:\n"
            "  1) dtoverlay=tc358743-pi5 in /boot/firmware/config.txt "
            "(use cam0 for CAM0 or 4lane=1 for CAM1)\n"
            "  2) TC358743 board connected to the matching CAM port\n"
            "  3) Reboot after config.txt changes"
        )

    def _detect_lane_count(self, topology: str) -> int:
        """Detect the number of CSI data lanes from the device tree.

        Parses the ``bus info`` field from the media topology to find the
        CSI controller address, then reads the ``data-lanes`` property
        from the device tree.  Falls back to 2 (safe default) on any error.

        Returns:
            2 or 4 (the number of active CSI data lanes).
        """
        # Extract platform address from topology, e.g. "platform:1f00128000.csi"
        match = re.search(r"bus info\s+platform:([0-9a-fA-F]+)\.csi", topology)
        if not match:
            logger.warning("Could not parse CSI bus info — defaulting to 2 lanes")
            return 2

        # Map platform address to RP1 CSI register offset:
        # 1f00110000 → 110000 (CAM0), 1f00128000 → 128000 (CAM1)
        platform_addr = match.group(1)
        csi_addr = platform_addr[-6:]

        dt_path = (
            f"/sys/firmware/devicetree/base/axi/pcie@1000120000"
            f"/rp1/csi@{csi_addr}/port/endpoint/data-lanes"
        )
        try:
            with open(dt_path, "rb") as f:
                data = f.read()
        except OSError:
            logger.warning(
                "Cannot read %s — defaulting to 2 lanes", dt_path,
            )
            return 2

        if len(data) % 4 != 0 or len(data) == 0:
            logger.warning(
                "Unexpected data-lanes size (%d bytes) — defaulting to 2 lanes",
                len(data),
            )
            return 2

        lane_count = len(data) // 4
        if lane_count not in (2, 4):
            logger.warning(
                "Unexpected lane count %d — defaulting to 2 lanes", lane_count,
            )
            return 2

        return lane_count

    def _parse_subdev(self, topology: str) -> str:
        """Extract the TC358743 V4L2 subdevice path from media topology.

        Parses lines like::

            - entity 16: tc358743 10-000f (1 pad, 1 link, 0 routes)
                         ...
                         device node name /dev/v4l-subdev2
        """
        # Find tc358743 entity block, then its device node
        match = re.search(
            r"entity\s+\d+:\s+tc358743\s+.*?"
            r"device node name\s+(/dev/v4l-subdev\d+)",
            topology, re.DOTALL,
        )
        if not match:
            raise CaptureError(
                "TC358743 entity found in media topology but no subdevice node. "
                "The bridge chip may not be responding on I2C. "
                "Check: ribbon cable connection, board power LED."
            )
        return match.group(1)

    def _parse_video_device(self, topology: str) -> str:
        """Extract the rp1-cfe-csi2_ch0 video device path from media topology.

        Parses lines like::

            - entity 18: rp1-cfe-csi2_ch0 (1 pad, 1 link)
                         ...
                         device node name /dev/video0
        """
        match = re.search(
            r"entity\s+\d+:\s+rp1-cfe-csi2_ch0\s+.*?"
            r"device node name\s+(/dev/video\d+)",
            topology, re.DOTALL,
        )
        if not match:
            raise CaptureError(
                "rp1-cfe-csi2_ch0 video node not found in media topology."
            )
        return match.group(1)

    def _query_current_signal(self) -> tuple[int, int] | None:
        """Check if there is already an HDMI signal and return its resolution.

        Returns:
            (width, height) if a signal is present, or None if no signal.

        Raises:
            CaptureError: If the command fails for reasons other than
                "no signal" (e.g. permission denied, stale device path).
        """
        result = self._run_cmd(
            [
                "v4l2-ctl", "-d", self._subdev_path,  # type: ignore[list-item]
                "--query-dv-timings",
            ],
            check=False, timeout=5.0,
        )
        output = result.stdout + result.stderr

        if result.returncode != 0:
            # "No locks available" / "Link has been severed" = genuinely no signal
            if "No lock" in output or "Link has been" in output:
                return None
            # Permission denied is a common user mistake — surface it early
            if "Permission denied" in output or "EACCES" in output:
                raise CaptureError(
                    f"Permission denied accessing {self._subdev_path}. "
                    "Run with sudo or add your user to the 'video' group."
                )
            # Other unexpected errors — log and return None
            logger.warning(
                "v4l2-ctl --query-dv-timings failed (rc=%d): %s",
                result.returncode, result.stderr.strip(),
            )
            return None

        width_match = re.search(r"Active width:\s+(\d+)", output)
        height_match = re.search(r"Active height:\s+(\d+)", output)
        if width_match and height_match:
            w = int(width_match.group(1))
            h = int(height_match.group(1))
            if w > 0 and h > 0:
                return (w, h)
        return None

    def _load_edid(self) -> None:
        """Load the appropriate EDID onto the TC358743.

        Selects 1080p EDID on 4-lane CSI (allows source to choose up to
        1080p), 720p-only EDID on 2-lane CSI.  Overwrites any existing
        EDID without clearing first.  The TC358743 driver's ``set_edid``
        toggles HPD (hot-plug detect) internally to trigger source
        re-negotiation.  Clearing first (``clear_edid``) deasserts HPD
        in a way that breaks subsequent re-negotiation on the Pi 5
        rp1-cfe driver.
        """
        if self._max_capture_height >= 1080:
            edid, label = EDID_1080P, "1080p"
        else:
            edid, label = EDID_720P, "720p"
        hex_content = _edid_to_hex_file_content(edid)

        fd, path = tempfile.mkstemp(suffix=".txt", prefix=f"edid_{label}_")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(hex_content)
            self._run_cmd(
                [
                    "v4l2-ctl", "-d", self._subdev_path,  # type: ignore[list-item]
                    "--set-edid", f"pad=0,file={path}",
                ],
                error_msg="Failed to load EDID onto TC358743",
            )
            logger.info("%s EDID loaded on %s", label, self._subdev_path)
        finally:
            try:
                os.unlink(path)
            except OSError:
                logger.debug("Could not remove temp EDID file: %s", path)

        # Brief pause for HPD toggle to propagate; the _wait_for_signal
        # loop handles the full wait for the source to switch resolution.
        time.sleep(2.0)

    def _wait_for_signal(self) -> None:
        """Poll DV timings until HDMI signal locks at a safe resolution.

        The max safe height depends on the CSI lane count:
        - 2-lane (CAM0): max 720p (BGR needs ~1.33 Gbps, 2 lanes provide ~2 Gbps)
        - 4-lane (CAM1): max 1080p (BGR needs ~2.99 Gbps, 4 lanes provide ~4 Gbps)
        """
        max_safe_height = self._max_capture_height
        deadline = time.monotonic() + self._signal_timeout
        attempt = 0
        while True:
            attempt += 1
            result = self._run_cmd(
                [
                    "v4l2-ctl", "-d", self._subdev_path,  # type: ignore[list-item]
                    "--query-dv-timings",
                ],
                check=False, timeout=5.0,
            )
            output = result.stdout + result.stderr
            width_match = re.search(r"Active width:\s+(\d+)", output)
            height_match = re.search(r"Active height:\s+(\d+)", output)
            if width_match and height_match:
                w = int(width_match.group(1))
                h = int(height_match.group(1))
                if w > 0 and h > 0:
                    if h <= max_safe_height:
                        self._signal_width = w
                        self._signal_height = h
                        logger.info(
                            "HDMI signal locked: %dx%d (attempt %d)",
                            w, h, attempt,
                        )
                        return
                    # Source still at high resolution — keep waiting
                    logger.debug(
                        "Signal at %dx%d (too high for %d-lane CSI), "
                        "waiting for source to switch to %dp (attempt %d)...",
                        w, h, self._lane_count, max_safe_height, attempt,
                    )
                    if time.monotonic() >= deadline:
                        raise CaptureError(
                            f"HDMI source locked at {w}x{h} which exceeds "
                            f"{self._lane_count}-lane CSI bandwidth "
                            f"(max {max_safe_height}p). The EDID was "
                            "loaded but the source did not switch. Try:\n"
                            f"  1) Manually set source output to {max_safe_height}p\n"
                            "  2) Disconnect and reconnect the HDMI cable\n"
                            "  3) Restart the source computer"
                        )
                    time.sleep(1.0)
                    continue

            if time.monotonic() >= deadline:
                raise CaptureError(
                    f"No HDMI signal detected within {self._signal_timeout}s. Check:\n"
                    "  1) HDMI cable connected to TC358743 input\n"
                    "  2) Source computer is powered on\n"
                    "  3) Source is outputting video (not sleeping)"
                )
            logger.debug("Waiting for HDMI signal (attempt %d)...", attempt)
            time.sleep(1.0)

    def _set_dv_timings(self) -> None:
        """Lock the detected DV timings on the TC358743."""
        self._run_cmd(
            [
                "v4l2-ctl", "-d", self._subdev_path,  # type: ignore[list-item]
                "--set-dv-bt-timings", "query",
            ],
            error_msg="Failed to set DV timings on TC358743",
        )
        logger.info("DV timings set from detected signal")

    def _configure_pipeline(self) -> None:
        """Reset media links and configure the CSI2 pipeline for BGR888.

        Uses the resolution detected by :meth:`_wait_for_signal` (stored in
        ``_signal_width`` / ``_signal_height``).

        Runs four media-ctl commands:
        1. Reset all links
        2. Enable csi2:4 -> rp1-cfe-csi2_ch0:0 link
        3. Set csi2 pad 0 format (sink from TC358743)
        4. Set csi2 pad 4 format (source to video node)
        """
        dev = self._media_device
        w, h = self._signal_width, self._signal_height
        fmt = f"BGR888_1X24/{w}x{h} field:none colorspace:srgb"

        commands = [
            (["-r"], "reset links"),
            (["-l", '"csi2":4 -> "rp1-cfe-csi2_ch0":0 [1]'], "enable link"),
            (["-V", f'"csi2":0 [fmt:{fmt}]'], "set csi2 pad 0 format"),
            (["-V", f'"csi2":4 [fmt:{fmt}]'], "set csi2 pad 4 format"),
        ]

        for args, description in commands:
            self._run_cmd(
                ["media-ctl", "-d", dev, *args],  # type: ignore[list-item]
                error_msg=f"Pipeline setup failed ({description})",
            )
        logger.info("Media pipeline configured for BGR888 %dx%d", w, h)

    # ------------------------------------------------------------------
    # Subprocess helper
    # ------------------------------------------------------------------

    def _run_cmd(
        self,
        cmd: list[str],
        *,
        timeout: float = 10.0,
        check: bool = True,
        error_msg: str = "",
    ) -> subprocess.CompletedProcess[str]:
        """Run a subprocess command.

        Args:
            cmd:       Command and arguments.
            timeout:   Seconds before timeout.
            check:     If True, raise CaptureError on non-zero exit.
            error_msg: Human-readable prefix for error messages.

        Returns:
            The completed process result.

        Raises:
            CaptureError: If the command is not found, times out, or fails
                (when *check* is True).
        """
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout,
            )
        except FileNotFoundError as e:
            raise CaptureError(
                f"Command not found: {cmd[0]}. "
                "Install v4l-utils: sudo apt install v4l-utils"
            ) from e
        except subprocess.TimeoutExpired as e:
            raise CaptureError(
                f"Command timed out after {timeout}s: {' '.join(cmd)}"
            ) from e

        if check and result.returncode != 0:
            detail = error_msg or f"Command failed: {' '.join(cmd)}"
            stderr = result.stderr.strip()
            raise CaptureError(f"{detail}\n  stderr: {stderr}")

        return result
