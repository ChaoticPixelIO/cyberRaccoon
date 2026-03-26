"""Tests for M1 CsiHdmiCapture — TC358743 HDMI-to-CSI bridge capture.

All tests use mocked subprocess calls and cv2.VideoCapture (no hardware needed).
"""

from __future__ import annotations

import base64
import io
import subprocess
from unittest.mock import MagicMock, patch, call

import cv2
import numpy as np
import pytest
from PIL import Image

from capture.base import CaptureError
from capture.csi_capture import (
    CsiHdmiCapture,
    EDID_720P,
    EDID_1080P,
    _edid_to_hex_file_content,
)


# ---------------------------------------------------------------------------
# Realistic media-ctl topology output for mocking
# ---------------------------------------------------------------------------

SAMPLE_TOPOLOGY = """\
Media controller API version 6.12.62

Media device information
------------------------
driver          rp1-cfe
model           rp1-cfe
serial
bus info        platform:1f00110000.csi
hw revision     0x114666
driver version  6.12.62

Device topology
- entity 1: csi2 (8 pads, 8 links, 0 routes)
            type V4L2 subdev subtype Unknown flags 0
            device node name /dev/v4l-subdev0

- entity 10: pisp-fe (5 pads, 7 links, 0 routes)
             type V4L2 subdev subtype Unknown flags 0
             device node name /dev/v4l-subdev1

- entity 16: tc358743 10-000f (1 pad, 1 link, 0 routes)
             type V4L2 subdev subtype Unknown flags 0
             device node name /dev/v4l-subdev2
	pad0: SOURCE
		[stream:0 fmt:BGR888_1X24/1280x720 field:none colorspace:srgb]
		-> "csi2":0 [ENABLED,IMMUTABLE]

- entity 18: rp1-cfe-csi2_ch0 (1 pad, 1 link)
             type Node subtype V4L flags 0
             device node name /dev/video0
	pad0: SINK,MUST_CONNECT
		<- "csi2":4 [ENABLED]

- entity 22: rp1-cfe-embedded (1 pad, 1 link)
             type Node subtype V4L flags 0
             device node name /dev/video1
"""

# Topology for a non-CSI media device (e.g., pispbe)
SAMPLE_PISPBE_TOPOLOGY = """\
Media device information
------------------------
driver          pispbe
model           pispbe
"""

SAMPLE_DV_TIMINGS_LOCKED = """\
\tActive width: 1280
\tActive height: 720
\tTotal width: 1650
\tTotal height: 750
\tFrame format: progressive
\tPixelclock: 74250000 Hz (60.00 frames per second)
"""

SAMPLE_TOPOLOGY_4LANE = SAMPLE_TOPOLOGY.replace(
    "platform:1f00110000.csi", "platform:1f00128000.csi",
)

SAMPLE_DV_TIMINGS_1080P = """\
\tActive width: 1920
\tActive height: 1080
\tTotal width: 2200
\tTotal height: 1125
\tFrame format: progressive
\tPixelclock: 148500000 Hz (60.00 frames per second)
"""

SAMPLE_DV_TIMINGS_NO_SIGNAL = """\
VIDIOC_QUERY_DV_TIMINGS: failed: No locks available
\tActive width: 0
\tActive height: 0
"""


# ---------------------------------------------------------------------------
# Helper: create a CsiHdmiCapture with mocked V4L2 backend
# ---------------------------------------------------------------------------

def make_mock_csi_hdmi(
    frame: np.ndarray,
    width: int = 1280,
    height: int = 720,
    quality: int = 80,
) -> CsiHdmiCapture:
    """Create a CsiHdmiCapture with a mocked cv2.VideoCapture backend."""
    cap = CsiHdmiCapture(
        target_width=width,
        target_height=height,
        jpeg_quality=quality,
    )
    mock_vc = MagicMock()
    mock_vc.isOpened.return_value = True
    mock_vc.read.return_value = (True, frame)
    mock_vc.grab.return_value = True
    cap._cap = mock_vc
    return cap


# ---------------------------------------------------------------------------
# Fixtures — BGR frames (V4L2/OpenCV returns BGR)
# ---------------------------------------------------------------------------

def _bgr_frame_720p() -> np.ndarray:
    """Simulated 1280x720 BGR frame (no resize needed)."""
    return np.zeros((720, 1280, 3), dtype=np.uint8)


def _bgr_frame_with_content() -> np.ndarray:
    """BGR frame with visible content (white rectangle)."""
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[100:200, 100:300] = 255
    return frame


def _bgr_frame_black() -> np.ndarray:
    """Nearly-black BGR frame (simulates HDCP or no signal)."""
    frame = np.full((720, 1280, 3), 5, dtype=np.uint8)
    return frame


# ---------------------------------------------------------------------------
# Tests: EDID
# ---------------------------------------------------------------------------

class TestEdid:
    """Tests for the 720p EDID constant."""

    def test_edid_length(self) -> None:
        """EDID should be exactly 128 bytes (base block, no extensions)."""
        assert len(EDID_720P) == 128

    def test_edid_header(self) -> None:
        """EDID must start with the fixed header pattern."""
        assert EDID_720P[:8] == bytes([0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00])

    def test_edid_checksum_valid(self) -> None:
        """Sum of all 128 bytes must be 0 mod 256."""
        assert sum(EDID_720P) % 256 == 0

    def test_edid_version_1_3(self) -> None:
        """EDID version should be 1.3."""
        assert EDID_720P[18] == 1  # version
        assert EDID_720P[19] == 3  # revision

    def test_edid_no_extensions(self) -> None:
        """Extension count should be 0."""
        assert EDID_720P[126] == 0

    def test_edid_to_hex_format(self) -> None:
        """Hex output should have 8 lines of 32 hex chars each."""
        hex_str = _edid_to_hex_file_content(EDID_720P)
        lines = hex_str.strip().split("\n")
        assert len(lines) == 8
        for line in lines:
            assert len(line) == 32  # 16 bytes × 2 hex chars each


class TestEdid1080p:
    """Tests for the 1080p EDID constant."""

    def test_edid_length(self) -> None:
        assert len(EDID_1080P) == 128

    def test_edid_header(self) -> None:
        assert EDID_1080P[:8] == bytes([0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00])

    def test_edid_checksum_valid(self) -> None:
        assert sum(EDID_1080P) % 256 == 0

    def test_edid_version_1_3(self) -> None:
        assert EDID_1080P[18] == 1
        assert EDID_1080P[19] == 3

    def test_edid_no_extensions(self) -> None:
        assert EDID_1080P[126] == 0


# ---------------------------------------------------------------------------
# Tests: Lane detection
# ---------------------------------------------------------------------------

class TestLaneDetection:
    """Tests for _detect_lane_count()."""

    def test_2_lane_cam0(self) -> None:
        """Should detect 2 lanes from CAM0 device tree."""
        import struct
        data = struct.pack(">II", 1, 2)  # 2 big-endian uint32s
        cap = CsiHdmiCapture()
        with patch("builtins.open", MagicMock(
            return_value=MagicMock(
                __enter__=MagicMock(return_value=MagicMock(read=MagicMock(return_value=data))),
                __exit__=MagicMock(return_value=False),
            ),
        )):
            assert cap._detect_lane_count(SAMPLE_TOPOLOGY) == 2

    def test_4_lane_cam1(self) -> None:
        """Should detect 4 lanes from CAM1 device tree."""
        import struct
        data = struct.pack(">IIII", 1, 2, 3, 4)  # 4 big-endian uint32s
        cap = CsiHdmiCapture()
        with patch("builtins.open", MagicMock(
            return_value=MagicMock(
                __enter__=MagicMock(return_value=MagicMock(read=MagicMock(return_value=data))),
                __exit__=MagicMock(return_value=False),
            ),
        )):
            assert cap._detect_lane_count(SAMPLE_TOPOLOGY_4LANE) == 4

    def test_file_not_found_defaults_to_2(self) -> None:
        """Should fall back to 2 lanes if device tree file missing."""
        cap = CsiHdmiCapture()
        with patch("builtins.open", side_effect=FileNotFoundError):
            assert cap._detect_lane_count(SAMPLE_TOPOLOGY) == 2

    def test_no_bus_info_defaults_to_2(self) -> None:
        """Should fall back to 2 lanes if topology has no bus info."""
        cap = CsiHdmiCapture()
        assert cap._detect_lane_count(SAMPLE_PISPBE_TOPOLOGY) == 2

    def test_address_mapping_cam0(self) -> None:
        """CAM0 platform address should map to csi@110000."""
        import struct
        cap = CsiHdmiCapture()
        data = struct.pack(">II", 1, 2)
        with patch("builtins.open", MagicMock(
            return_value=MagicMock(
                __enter__=MagicMock(return_value=MagicMock(read=MagicMock(return_value=data))),
                __exit__=MagicMock(return_value=False),
            ),
        )) as mock_open:
            cap._detect_lane_count(SAMPLE_TOPOLOGY)
            path_arg = mock_open.call_args[0][0]
            assert "csi@110000" in path_arg

    def test_address_mapping_cam1(self) -> None:
        """CAM1 platform address should map to csi@128000."""
        import struct
        cap = CsiHdmiCapture()
        data = struct.pack(">IIII", 1, 2, 3, 4)
        with patch("builtins.open", MagicMock(
            return_value=MagicMock(
                __enter__=MagicMock(return_value=MagicMock(read=MagicMock(return_value=data))),
                __exit__=MagicMock(return_value=False),
            ),
        )) as mock_open:
            cap._detect_lane_count(SAMPLE_TOPOLOGY_4LANE)
            path_arg = mock_open.call_args[0][0]
            assert "csi@128000" in path_arg


# ---------------------------------------------------------------------------
# Tests: CaptureResult
# ---------------------------------------------------------------------------

class TestCaptureResult:
    """Tests for CSI HDMI capture output correctness."""

    def test_output_dimensions(self) -> None:
        cap = make_mock_csi_hdmi(_bgr_frame_720p())
        result = cap.capture()
        assert result.width == 1280
        assert result.height == 720
        assert result.image.size == (1280, 720)

    def test_base64_is_valid_jpeg(self) -> None:
        cap = make_mock_csi_hdmi(_bgr_frame_720p())
        result = cap.capture()
        jpeg_bytes = base64.b64decode(result.base64_jpeg)
        image = Image.open(io.BytesIO(jpeg_bytes))
        assert image.format == "JPEG"
        assert image.size == (1280, 720)

    def test_size_bytes_matches_jpeg(self) -> None:
        cap = make_mock_csi_hdmi(_bgr_frame_720p())
        result = cap.capture()
        jpeg_bytes = base64.b64decode(result.base64_jpeg)
        assert result.size_bytes == len(jpeg_bytes)

    def test_timestamp_is_positive(self) -> None:
        cap = make_mock_csi_hdmi(_bgr_frame_720p())
        result = cap.capture()
        assert result.timestamp > 0

    def test_image_is_rgb(self) -> None:
        """Output PIL Image should be RGB (converted from BGR)."""
        cap = make_mock_csi_hdmi(_bgr_frame_720p())
        result = cap.capture()
        assert result.image.mode == "RGB"

    def test_custom_resolution(self) -> None:
        cap = make_mock_csi_hdmi(_bgr_frame_720p(), width=640, height=480)
        result = cap.capture()
        assert result.width == 640
        assert result.height == 480
        assert result.image.size == (640, 480)


# ---------------------------------------------------------------------------
# Tests: Buffer flush
# ---------------------------------------------------------------------------

class TestBufferFlush:
    """Verify buffered frames are discarded before actual read."""

    def test_three_grabs_before_read(self) -> None:
        cap = make_mock_csi_hdmi(_bgr_frame_720p())
        cap.capture()
        assert cap._cap.grab.call_count == 3  # type: ignore[union-attr]
        assert cap._cap.read.call_count == 1  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Tests: Black frame warning
# ---------------------------------------------------------------------------

class TestBlackFrameWarning:
    """Verify warning logged for nearly-black frames."""

    def test_warns_on_black_frame(self, caplog: pytest.LogCaptureFixture) -> None:
        cap = make_mock_csi_hdmi(_bgr_frame_black())
        with caplog.at_level("WARNING", logger="M1.csi_hdmi"):
            cap.capture()
        assert "nearly black" in caplog.text.lower()

    def test_no_warning_on_normal_frame(self, caplog: pytest.LogCaptureFixture) -> None:
        cap = make_mock_csi_hdmi(_bgr_frame_with_content())
        with caplog.at_level("WARNING", logger="M1.csi_hdmi"):
            cap.capture()
        assert "nearly black" not in caplog.text.lower()


# ---------------------------------------------------------------------------
# Tests: Capture errors
# ---------------------------------------------------------------------------

class TestCaptureErrors:
    """Tests for error handling in capture()."""

    def test_capture_before_open(self) -> None:
        cap = CsiHdmiCapture()
        with pytest.raises(CaptureError, match="not opened"):
            cap.capture()

    def test_read_failure(self) -> None:
        cap = CsiHdmiCapture()
        mock_vc = MagicMock()
        mock_vc.isOpened.return_value = True
        mock_vc.read.return_value = (False, None)
        cap._cap = mock_vc

        with pytest.raises(CaptureError, match="Failed to read frame"):
            cap.capture()


# ---------------------------------------------------------------------------
# Tests: Device lifecycle
# ---------------------------------------------------------------------------

class TestDeviceLifecycle:
    """Tests for open/close/is_open lifecycle."""

    def test_is_open_initially_false(self) -> None:
        cap = CsiHdmiCapture()
        assert cap.is_open() is False

    def test_is_open_after_mock_setup(self) -> None:
        cap = make_mock_csi_hdmi(_bgr_frame_720p())
        assert cap.is_open() is True

    def test_close_releases_device(self) -> None:
        cap = make_mock_csi_hdmi(_bgr_frame_720p())
        mock_vc = cap._cap
        cap.close()
        assert cap.is_open() is False
        mock_vc.release.assert_called_once()  # type: ignore[union-attr]

    def test_double_close_safe(self) -> None:
        cap = make_mock_csi_hdmi(_bgr_frame_720p())
        cap.close()
        cap.close()  # Should not raise

    def test_close_without_open_safe(self) -> None:
        cap = CsiHdmiCapture()
        cap.close()  # Should not raise


# ---------------------------------------------------------------------------
# Tests: Constructor
# ---------------------------------------------------------------------------

class TestInit:
    """Tests for constructor defaults and custom values."""

    def test_default_values(self) -> None:
        cap = CsiHdmiCapture()
        assert cap._target_width == 1280
        assert cap._target_height == 720
        assert cap._jpeg_quality == 80
        assert cap._signal_timeout == 15.0
        assert cap._lane_count == 0
        assert cap._max_capture_height == 720
        assert cap._cap is None
        assert cap._media_device is None
        assert cap._subdev_path is None
        assert cap._video_device is None

    def test_custom_values(self) -> None:
        cap = CsiHdmiCapture(
            target_width=1920,
            target_height=1080,
            jpeg_quality=95,
            signal_timeout=30.0,
        )
        assert cap._target_width == 1920
        assert cap._target_height == 1080
        assert cap._jpeg_quality == 95
        assert cap._signal_timeout == 30.0


# ---------------------------------------------------------------------------
# Tests: Media device discovery
# ---------------------------------------------------------------------------

class TestMediaDiscovery:
    """Tests for media device and subdevice parsing."""

    def test_parse_subdev(self) -> None:
        """Should extract TC358743 subdev path from topology."""
        cap = CsiHdmiCapture()
        subdev = cap._parse_subdev(SAMPLE_TOPOLOGY)
        assert subdev == "/dev/v4l-subdev2"

    def test_parse_video_device(self) -> None:
        """Should extract rp1-cfe-csi2_ch0 video device from topology."""
        cap = CsiHdmiCapture()
        video = cap._parse_video_device(SAMPLE_TOPOLOGY)
        assert video == "/dev/video0"

    def test_parse_subdev_missing_raises(self) -> None:
        """Should raise CaptureError if tc358743 not found in topology."""
        cap = CsiHdmiCapture()
        with pytest.raises(CaptureError, match="entity found"):
            cap._parse_subdev(SAMPLE_PISPBE_TOPOLOGY)

    def test_parse_video_device_missing_raises(self) -> None:
        """Should raise CaptureError if csi2_ch0 not found in topology."""
        cap = CsiHdmiCapture()
        with pytest.raises(CaptureError, match="csi2_ch0.*not found"):
            cap._parse_video_device(SAMPLE_PISPBE_TOPOLOGY)


# ---------------------------------------------------------------------------
# Tests: open() pipeline (mocked subprocesses)
# ---------------------------------------------------------------------------

def _ok(stdout: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")


def _fail(stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr=stderr)


def _make_run_side_effects(
    *,
    media_device_index: int = 0,
    signal_already_720p: bool = True,
    signal_attempts: int = 1,
) -> list:
    """Build a list of subprocess.run side effects for a full open() sequence.

    Args:
        media_device_index: How many non-matching /dev/mediaN to probe first.
        signal_already_720p: If True, _query_current_signal returns 720p
            and EDID load is skipped. If False, returns no signal → EDID loaded.
        signal_attempts: When signal_already_720p=False, how many no-signal
            attempts before signal locks (1 = immediate lock).
    """
    effects: list[subprocess.CompletedProcess] = []

    # _discover_media_device: probes /dev/media0..N
    for i in range(media_device_index):
        effects.append(_fail(SAMPLE_PISPBE_TOPOLOGY))
    effects.append(_ok(SAMPLE_TOPOLOGY))

    # _query_current_signal
    if signal_already_720p:
        effects.append(_ok(SAMPLE_DV_TIMINGS_LOCKED))
        # EDID load skipped, _wait_for_signal skipped
    else:
        no_signal = subprocess.CompletedProcess(
            args=[], returncode=255, stdout="", stderr=SAMPLE_DV_TIMINGS_NO_SIGNAL,
        )
        effects.append(no_signal)  # _query_current_signal: no signal

        # _load_edid: set-edid (no clear)
        effects.append(_ok())  # set-edid

        # _wait_for_signal: may take multiple attempts
        for _ in range(signal_attempts - 1):
            effects.append(no_signal)
        effects.append(_ok(SAMPLE_DV_TIMINGS_LOCKED))

    # _set_dv_timings
    effects.append(_ok("BT timings set"))

    # _configure_pipeline: 4 media-ctl commands
    for _ in range(4):
        effects.append(_ok())

    return effects


class TestOpen:
    """Tests for the full open() pipeline with mocked subprocesses."""

    @patch("capture.csi_capture.CsiHdmiCapture._detect_lane_count", return_value=2)
    @patch("capture.csi_capture.time.sleep")
    @patch("capture.csi_capture.cv2.VideoCapture")
    @patch("capture.csi_capture.subprocess.run")
    @patch("capture.csi_capture.shutil.which", return_value="/usr/bin/v4l2-ctl")
    @patch("capture.csi_capture.os.path.exists")
    def test_open_success(
        self, mock_exists, mock_which, mock_run, mock_cv2_cap, mock_sleep,
        mock_lanes,
    ) -> None:
        """Full open() should succeed with proper mocking."""
        # /dev/media0-2 don't exist, /dev/media3 does
        mock_exists.side_effect = lambda p: p == "/dev/media3"
        mock_run.side_effect = _make_run_side_effects(media_device_index=0)

        mock_vc = MagicMock()
        mock_vc.isOpened.return_value = True
        # Return correct width/height for the respective property queries
        def _get_prop(prop_id):
            if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
                return 1280
            if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
                return 720
            return 0
        mock_vc.get.side_effect = _get_prop
        # Warmup: return a non-black frame
        warmup_frame = np.full((720, 1280, 3), 128, dtype=np.uint8)
        mock_vc.read.return_value = (True, warmup_frame)
        mock_vc.grab.return_value = True
        mock_cv2_cap.return_value = mock_vc

        cap = CsiHdmiCapture()
        cap.open()

        assert cap.is_open()
        assert cap._media_device == "/dev/media3"
        assert cap._subdev_path == "/dev/v4l-subdev2"
        assert cap._video_device == "/dev/video0"
        assert cap._signal_width == 1280
        assert cap._signal_height == 720

    @patch("capture.csi_capture.shutil.which", return_value="/usr/bin/v4l2-ctl")
    @patch("capture.csi_capture.os.path.exists", return_value=False)
    def test_open_no_media_device(self, mock_exists, mock_which) -> None:
        """Should raise CaptureError if no rp1-cfe media device found."""
        cap = CsiHdmiCapture()
        with pytest.raises(CaptureError, match="TC358743.*not detected"):
            cap.open()

    @patch("capture.csi_capture.shutil.which", return_value=None)
    def test_open_missing_tools(self, mock_which) -> None:
        """Should raise CaptureError if v4l2-ctl is not installed."""
        cap = CsiHdmiCapture()
        with pytest.raises(CaptureError, match="not found"):
            cap.open()

    @patch("capture.csi_capture.CsiHdmiCapture._detect_lane_count", return_value=2)
    @patch("capture.csi_capture.time.sleep")
    @patch("capture.csi_capture.cv2.VideoCapture")
    @patch("capture.csi_capture.subprocess.run")
    @patch("capture.csi_capture.shutil.which", return_value="/usr/bin/v4l2-ctl")
    @patch("capture.csi_capture.os.path.exists", return_value=True)
    def test_open_signal_retry(
        self, mock_exists, mock_which, mock_run, mock_cv2_cap, mock_sleep,
        mock_lanes,
    ) -> None:
        """Should retry signal detection before success."""
        mock_run.side_effect = _make_run_side_effects(
            media_device_index=0, signal_already_720p=False, signal_attempts=3,
        )
        mock_vc = MagicMock()
        mock_vc.isOpened.return_value = True
        def _get_prop(prop_id):
            if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
                return 1280
            if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
                return 720
            return 0
        mock_vc.get.side_effect = _get_prop
        warmup_frame = np.full((720, 1280, 3), 128, dtype=np.uint8)
        mock_vc.read.return_value = (True, warmup_frame)
        mock_vc.grab.return_value = True
        mock_cv2_cap.return_value = mock_vc

        cap = CsiHdmiCapture(signal_timeout=30.0)
        cap.open()
        assert cap.is_open()

    @patch("capture.csi_capture.CsiHdmiCapture._detect_lane_count", return_value=2)
    @patch("capture.csi_capture.subprocess.run")
    @patch("capture.csi_capture.shutil.which", return_value="/usr/bin/v4l2-ctl")
    @patch("capture.csi_capture.os.path.exists", return_value=True)
    @patch("capture.csi_capture.time.monotonic")
    @patch("capture.csi_capture.time.sleep")
    def test_open_signal_timeout(
        self, mock_sleep, mock_monotonic, mock_exists, mock_which, mock_run,
        mock_lanes,
    ) -> None:
        """Should raise CaptureError if signal never locks."""
        no_signal = subprocess.CompletedProcess(
            args=[], returncode=255, stdout="", stderr=SAMPLE_DV_TIMINGS_NO_SIGNAL,
        )
        mock_run.side_effect = [
            # _discover_media_device
            _ok(SAMPLE_TOPOLOGY),
            # _query_current_signal: no signal
            no_signal,
            # _load_edid: set (no clear)
            _ok(),
            # _wait_for_signal: always no signal (3 attempts before timeout)
            no_signal, no_signal, no_signal,
        ]
        # time.monotonic() call sequence:
        #   1) deadline = start + 10 → start=0.0
        #   2) after attempt 1: 0.0 < 10 → retry
        #   3) after attempt 2: 5.0 < 10 → retry
        #   4) after attempt 3: 15.0 >= 10 → timeout
        mock_monotonic.side_effect = [0.0, 0.0, 5.0, 15.0]

        cap = CsiHdmiCapture(signal_timeout=10.0)
        with pytest.raises(CaptureError, match="No HDMI signal"):
            cap.open()

    @patch("capture.csi_capture.CsiHdmiCapture._detect_lane_count", return_value=2)
    @patch("capture.csi_capture.time.sleep")
    @patch("capture.csi_capture.subprocess.run")
    @patch("capture.csi_capture.shutil.which", return_value="/usr/bin/v4l2-ctl")
    @patch("capture.csi_capture.os.path.exists", return_value=True)
    def test_open_pipeline_failure(
        self, mock_exists, mock_which, mock_run, mock_sleep, mock_lanes,
    ) -> None:
        """Should raise CaptureError if media-ctl pipeline setup fails."""
        effects = _make_run_side_effects(media_device_index=0)
        # Make the first pipeline command (reset) fail.
        # With signal_already_720p=True:
        #   Discovery(1) + query_signal(1) + DV(1) = 3, pipeline[0] at index 3
        effects[3] = _fail("Device busy")
        mock_run.side_effect = effects

        cap = CsiHdmiCapture()
        with pytest.raises(CaptureError, match="Pipeline setup failed"):
            cap.open()

    @patch("capture.csi_capture.CsiHdmiCapture._detect_lane_count", return_value=2)
    @patch("capture.csi_capture.time.sleep")
    @patch("capture.csi_capture.cv2.VideoCapture")
    @patch("capture.csi_capture.subprocess.run")
    @patch("capture.csi_capture.shutil.which", return_value="/usr/bin/v4l2-ctl")
    @patch("capture.csi_capture.os.path.exists", return_value=True)
    def test_open_opencv_fails(
        self, mock_exists, mock_which, mock_run, mock_cv2_cap, mock_sleep,
        mock_lanes,
    ) -> None:
        """Should raise CaptureError if OpenCV can't open the device."""
        mock_run.side_effect = _make_run_side_effects(media_device_index=0)
        mock_vc = MagicMock()
        mock_vc.isOpened.return_value = False
        mock_cv2_cap.return_value = mock_vc

        cap = CsiHdmiCapture()
        with pytest.raises(CaptureError, match="Cannot open"):
            cap.open()

    @patch("capture.csi_capture.CsiHdmiCapture._detect_lane_count", return_value=2)
    @patch("capture.csi_capture.time.sleep")
    @patch("capture.csi_capture.time.monotonic")
    @patch("capture.csi_capture.cv2.VideoCapture")
    @patch("capture.csi_capture.subprocess.run")
    @patch("capture.csi_capture.shutil.which", return_value="/usr/bin/v4l2-ctl")
    @patch("capture.csi_capture.os.path.exists", return_value=True)
    def test_open_warmup_timeout(
        self, mock_exists, mock_which, mock_run, mock_cv2_cap,
        mock_monotonic, mock_sleep, mock_lanes,
    ) -> None:
        """Should raise CaptureError if warmup produces only black frames."""
        mock_run.side_effect = _make_run_side_effects(media_device_index=0)

        mock_vc = MagicMock()
        mock_vc.isOpened.return_value = True
        def _get_prop(prop_id):
            if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
                return 1280
            if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
                return 720
            return 0
        mock_vc.get.side_effect = _get_prop
        # Always return black frames
        black_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        mock_vc.read.return_value = (True, black_frame)
        mock_vc.grab.return_value = True
        mock_cv2_cap.return_value = mock_vc

        # Simulate warmup timeout: monotonic returns past deadline immediately
        mock_monotonic.side_effect = [0.0, 6.0]

        cap = CsiHdmiCapture()
        with pytest.raises(CaptureError, match="only producing black frames"):
            cap.open()

    @patch("capture.csi_capture.CsiHdmiCapture._detect_lane_count", return_value=2)
    @patch("capture.csi_capture.subprocess.run")
    @patch("capture.csi_capture.shutil.which", return_value="/usr/bin/v4l2-ctl")
    @patch("capture.csi_capture.os.path.exists", return_value=True)
    @patch("capture.csi_capture.time.monotonic")
    @patch("capture.csi_capture.time.sleep")
    def test_open_source_stuck_at_1080p(
        self, mock_sleep, mock_monotonic, mock_exists, mock_which, mock_run,
        mock_lanes,
    ) -> None:
        """Should raise CaptureError if source stays at 1080p after EDID load (2-lane)."""
        dv_1080p = subprocess.CompletedProcess(
            args=[], returncode=0,
            stdout="\tActive width: 1920\n\tActive height: 1080\n"
                   "\tPixelclock: 148500000 Hz (60.00 frames per second)\n",
            stderr="",
        )
        no_signal = subprocess.CompletedProcess(
            args=[], returncode=255, stdout="", stderr=SAMPLE_DV_TIMINGS_NO_SIGNAL,
        )
        mock_run.side_effect = [
            # _discover_media_device
            _ok(SAMPLE_TOPOLOGY),
            # _query_current_signal: sees 1080p
            dv_1080p,
            # _load_edid: set
            _ok(),
            # _wait_for_signal: source stuck at 1080p
            dv_1080p, dv_1080p, dv_1080p,
        ]
        mock_monotonic.side_effect = [0.0, 0.0, 5.0, 16.0]

        cap = CsiHdmiCapture(signal_timeout=15.0)
        with pytest.raises(CaptureError, match="2-lane CSI bandwidth"):
            cap.open()

    @patch("capture.csi_capture.CsiHdmiCapture._detect_lane_count", return_value=4)
    @patch("capture.csi_capture.time.sleep")
    @patch("capture.csi_capture.cv2.VideoCapture")
    @patch("capture.csi_capture.subprocess.run")
    @patch("capture.csi_capture.shutil.which", return_value="/usr/bin/v4l2-ctl")
    @patch("capture.csi_capture.os.path.exists", return_value=True)
    def test_open_4lane_accepts_1080p(
        self, mock_exists, mock_which, mock_run, mock_cv2_cap, mock_sleep,
        mock_lanes,
    ) -> None:
        """On 4-lane CSI, 1080p signal should be accepted without EDID reload."""
        dv_1080p = _ok(SAMPLE_DV_TIMINGS_1080P)
        mock_run.side_effect = [
            _ok(SAMPLE_TOPOLOGY_4LANE),   # _discover_media_device
            dv_1080p,                      # _query_current_signal: 1080p (accepted)
            _ok("BT timings set"),         # _set_dv_timings
            _ok(), _ok(), _ok(), _ok(),    # _configure_pipeline (4 commands)
        ]

        mock_vc = MagicMock()
        mock_vc.isOpened.return_value = True
        def _get_prop(prop_id):
            if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
                return 1920
            if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
                return 1080
            return 0
        mock_vc.get.side_effect = _get_prop
        warmup_frame = np.full((1080, 1920, 3), 128, dtype=np.uint8)
        mock_vc.read.return_value = (True, warmup_frame)
        mock_vc.grab.return_value = True
        mock_cv2_cap.return_value = mock_vc

        cap = CsiHdmiCapture()
        cap.open()
        assert cap._signal_width == 1920
        assert cap._signal_height == 1080
        assert cap._max_capture_height == 1080
        assert cap._lane_count == 4

    @patch("capture.csi_capture.CsiHdmiCapture._detect_lane_count", return_value=4)
    @patch("capture.csi_capture.time.sleep")
    @patch("capture.csi_capture.cv2.VideoCapture")
    @patch("capture.csi_capture.subprocess.run")
    @patch("capture.csi_capture.shutil.which", return_value="/usr/bin/v4l2-ctl")
    @patch("capture.csi_capture.os.path.exists", return_value=True)
    def test_open_4lane_loads_1080p_edid(
        self, mock_exists, mock_which, mock_run, mock_cv2_cap, mock_sleep,
        mock_lanes,
    ) -> None:
        """On 4-lane CSI with no signal, should load 1080p EDID."""
        no_signal = subprocess.CompletedProcess(
            args=[], returncode=255, stdout="", stderr=SAMPLE_DV_TIMINGS_NO_SIGNAL,
        )
        dv_1080p = _ok(SAMPLE_DV_TIMINGS_1080P)
        mock_run.side_effect = [
            _ok(SAMPLE_TOPOLOGY_4LANE),   # _discover_media_device
            no_signal,                     # _query_current_signal: no signal
            _ok(),                         # _load_edid: set-edid
            dv_1080p,                      # _wait_for_signal: locks at 1080p
            _ok("BT timings set"),         # _set_dv_timings
            _ok(), _ok(), _ok(), _ok(),    # _configure_pipeline
        ]

        mock_vc = MagicMock()
        mock_vc.isOpened.return_value = True
        def _get_prop(prop_id):
            if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
                return 1920
            if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
                return 1080
            return 0
        mock_vc.get.side_effect = _get_prop
        warmup_frame = np.full((1080, 1920, 3), 128, dtype=np.uint8)
        mock_vc.read.return_value = (True, warmup_frame)
        mock_vc.grab.return_value = True
        mock_cv2_cap.return_value = mock_vc

        cap = CsiHdmiCapture()
        cap.open()
        assert cap._signal_width == 1920
        assert cap._signal_height == 1080


# ---------------------------------------------------------------------------
# Tests: _query_current_signal error handling
# ---------------------------------------------------------------------------

class TestQuerySignal:
    """Tests for signal query error discrimination."""

    @patch("capture.csi_capture.subprocess.run")
    def test_permission_denied_raises(self, mock_run) -> None:
        """Should raise CaptureError on permission denied."""
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="",
            stderr="Permission denied: /dev/v4l-subdev2",
        )
        cap = CsiHdmiCapture()
        cap._subdev_path = "/dev/v4l-subdev2"
        with pytest.raises(CaptureError, match="Permission denied"):
            cap._query_current_signal()

    @patch("capture.csi_capture.subprocess.run")
    def test_no_signal_returns_none(self, mock_run) -> None:
        """Should return None when no HDMI signal."""
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=255, stdout="",
            stderr=SAMPLE_DV_TIMINGS_NO_SIGNAL,
        )
        cap = CsiHdmiCapture()
        cap._subdev_path = "/dev/v4l-subdev2"
        assert cap._query_current_signal() is None

    @patch("capture.csi_capture.subprocess.run")
    def test_locked_signal_returns_resolution(self, mock_run) -> None:
        """Should return (w, h) when signal is locked."""
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=SAMPLE_DV_TIMINGS_LOCKED, stderr="",
        )
        cap = CsiHdmiCapture()
        cap._subdev_path = "/dev/v4l-subdev2"
        assert cap._query_current_signal() == (1280, 720)


# ---------------------------------------------------------------------------
# Tests: _run_cmd helper
# ---------------------------------------------------------------------------

class TestRunCmd:
    """Tests for the subprocess helper."""

    @patch("capture.csi_capture.subprocess.run")
    def test_success(self, mock_run) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=["echo"], returncode=0, stdout="ok", stderr="",
        )
        cap = CsiHdmiCapture()
        result = cap._run_cmd(["echo", "hello"])
        assert result.stdout == "ok"

    @patch("capture.csi_capture.subprocess.run", side_effect=FileNotFoundError)
    def test_command_not_found(self, mock_run) -> None:
        cap = CsiHdmiCapture()
        with pytest.raises(CaptureError, match="Command not found"):
            cap._run_cmd(["nonexistent"])

    @patch(
        "capture.csi_capture.subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd=["slow"], timeout=5),
    )
    def test_timeout(self, mock_run) -> None:
        cap = CsiHdmiCapture()
        with pytest.raises(CaptureError, match="timed out"):
            cap._run_cmd(["slow"], timeout=5.0)

    @patch("capture.csi_capture.subprocess.run")
    def test_nonzero_exit_checked(self, mock_run) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=["fail"], returncode=1, stdout="", stderr="bad thing happened",
        )
        cap = CsiHdmiCapture()
        with pytest.raises(CaptureError, match="bad thing happened"):
            cap._run_cmd(["fail"], error_msg="Test failed")

    @patch("capture.csi_capture.subprocess.run")
    def test_nonzero_exit_unchecked(self, mock_run) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=["fail"], returncode=1, stdout="", stderr="",
        )
        cap = CsiHdmiCapture()
        result = cap._run_cmd(["fail"], check=False)
        assert result.returncode == 1
