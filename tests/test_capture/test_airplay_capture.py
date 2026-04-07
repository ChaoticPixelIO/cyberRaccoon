"""Tests for M1 AirPlayCapture — AirPlay screen mirroring via uxplay.

Tests cover both capture modes:
- **RTP mode** (uxplay >= 1.73): mocked subprocess.Popen + cv2.VideoCapture
- **File mode** (uxplay < 1.73): mocked subprocess.Popen + real temp JPEG files

No hardware or network needed.

Note: AirPlayCapture uses deferred ``import cv2`` inside methods, so we mock
cv2 via ``patch.dict("sys.modules", ...)`` rather than ``@patch("module.cv2")``.
"""

from __future__ import annotations

import base64
import io
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from cyberraccoon.capture.airplay_capture import AirPlayCapture
from cyberraccoon.capture.base import CaptureError, CaptureSource


# ---------------------------------------------------------------------------
# Helper: create mock cv2 module for deferred imports
# ---------------------------------------------------------------------------

def _make_mock_cv2() -> MagicMock:
    """Create a mock cv2 module with cvtColor as passthrough (BGR→RGB is identity
    for zero frames) and COLOR_BGR2RGB constant."""
    mock_cv2 = MagicMock()
    mock_cv2.cvtColor.side_effect = lambda f, _: f  # passthrough
    mock_cv2.COLOR_BGR2RGB = 4
    mock_cv2.CAP_GSTREAMER = 1800
    mock_cv2.getBuildInformation.return_value = "GStreamer:                   YES"
    return mock_cv2


def _setup_subprocess_for_open(
    mock_subprocess: MagicMock,
    version: str = "UxPlay 1.73\n",
) -> None:
    """Configure a mocked subprocess module for open() tests.

    Sets up:
    - subprocess.DEVNULL (used for uxplay stdout)
    - subprocess.PIPE (used for uxplay stderr)
    - subprocess.run() return value for _detect_uxplay_version()
    """
    mock_subprocess.DEVNULL = subprocess.DEVNULL
    mock_subprocess.PIPE = subprocess.PIPE
    # Version check: subprocess.run(["uxplay", "-h"], ...)
    mock_version_result = MagicMock()
    mock_version_result.stdout = version
    mock_version_result.stderr = ""
    mock_subprocess.run.return_value = mock_version_result


# ---------------------------------------------------------------------------
# Helper: frames and mock setup
# ---------------------------------------------------------------------------

def _bgr_frame_1080p() -> np.ndarray:
    """Simulated 1920x1080 BGR frame from GStreamer pipeline."""
    return np.zeros((1080, 1920, 3), dtype=np.uint8)


def _bgr_frame_with_content() -> np.ndarray:
    """BGR frame with visible content (white rectangle)."""
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    frame[100:200, 100:300] = 255
    return frame


def _bgr_frame_720p() -> np.ndarray:
    """Simulated 1280x720 BGR frame (no resize needed)."""
    return np.zeros((720, 1280, 3), dtype=np.uint8)


def make_mock_airplay(
    frame: np.ndarray | None = None,
    width: int = 1280,
    height: int = 720,
    quality: int = 80,
    rtp_port: int = 5004,
) -> AirPlayCapture:
    """Create an AirPlayCapture with mocked uxplay subprocess + GStreamer capture.

    Injects:
    - A mock Popen for the uxplay process (alive, poll() returns None)
    - A mock cv2.VideoCapture that returns the given BGR frame on read()
    """
    if frame is None:
        frame = _bgr_frame_1080p()

    cap = AirPlayCapture(
        target_width=width,
        target_height=height,
        jpeg_quality=quality,
        rtp_port=rtp_port,
    )

    # Mock uxplay subprocess (alive)
    mock_proc = MagicMock(spec=subprocess.Popen)
    mock_proc.poll.return_value = None  # process alive
    mock_proc.returncode = None
    mock_proc.stderr = MagicMock()
    cap._uxplay_proc = mock_proc

    # Mock cv2.VideoCapture (GStreamer pipeline)
    mock_cv_cap = MagicMock()
    mock_cv_cap.isOpened.return_value = True
    mock_cv_cap.read.return_value = (True, frame)
    cap._cap = mock_cv_cap

    # RTP mode (default for make_mock_airplay)
    cap._use_rtp = True

    return cap


def _make_test_jpeg(path: str | Path, width: int = 1920, height: int = 1080,
                    color: tuple[int, int, int] = (128, 64, 32)) -> None:
    """Write a valid JPEG file for file-mode testing."""
    img = Image.new("RGB", (width, height), color=color)
    img.save(str(path), "JPEG")


def make_mock_airplay_file_mode(
    frame_dir: str | None = None,
    width: int = 1280,
    height: int = 720,
    quality: int = 80,
) -> AirPlayCapture:
    """Create an AirPlayCapture pre-configured for file mode testing.

    Injects:
    - A mock Popen for the uxplay process (alive, poll() returns None)
    - ``_use_rtp = False`` and ``_frame_dir`` set
    - No cv2 capture (``_cap = None``)
    """
    cap = AirPlayCapture(
        target_width=width,
        target_height=height,
        jpeg_quality=quality,
    )

    mock_proc = MagicMock(spec=subprocess.Popen)
    mock_proc.poll.return_value = None
    mock_proc.returncode = None
    mock_proc.stderr = MagicMock()
    cap._uxplay_proc = mock_proc

    cap._use_rtp = False
    cap._frame_dir = frame_dir or "/tmp/test_frames"
    cap._owns_frame_dir = True
    cap._cap = None

    return cap


# ---------------------------------------------------------------------------
# TestCaptureResult (RTP mode)
# ---------------------------------------------------------------------------

class TestCaptureResult:
    """Tests for AirPlay capture output correctness (RTP mode)."""

    def test_output_dimensions(self) -> None:
        """Output image size should match target resolution."""
        mock_cv2 = _make_mock_cv2()
        cap = make_mock_airplay(_bgr_frame_1080p())
        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            result = cap.capture()
        assert result.width == 1280
        assert result.height == 720
        assert result.image.size == (1280, 720)

    def test_base64_is_valid_jpeg(self) -> None:
        """Base64 decoded content should be valid JPEG."""
        mock_cv2 = _make_mock_cv2()
        cap = make_mock_airplay(_bgr_frame_1080p())
        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            result = cap.capture()
        jpeg_bytes = base64.b64decode(result.base64_jpeg)
        image = Image.open(io.BytesIO(jpeg_bytes))
        assert image.format == "JPEG"
        assert image.size == (1280, 720)

    def test_size_bytes_matches_jpeg(self) -> None:
        """size_bytes should equal actual JPEG byte count."""
        mock_cv2 = _make_mock_cv2()
        cap = make_mock_airplay(_bgr_frame_1080p())
        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            result = cap.capture()
        jpeg_bytes = base64.b64decode(result.base64_jpeg)
        assert result.size_bytes == len(jpeg_bytes)

    def test_timestamp_is_positive(self) -> None:
        """Timestamp should be a positive monotonic value."""
        mock_cv2 = _make_mock_cv2()
        cap = make_mock_airplay(_bgr_frame_1080p())
        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            result = cap.capture()
        assert result.timestamp > 0

    def test_higher_quality_larger_size(self) -> None:
        """Higher JPEG quality should produce larger files."""
        mock_cv2 = _make_mock_cv2()
        frame = _bgr_frame_with_content()
        cap_low = make_mock_airplay(frame, quality=30)
        cap_high = make_mock_airplay(frame, quality=95)
        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            r_low = cap_low.capture()
            r_high = cap_high.capture()
        assert r_high.size_bytes > r_low.size_bytes

    def test_custom_resolution(self) -> None:
        """Custom target resolution should be applied."""
        mock_cv2 = _make_mock_cv2()
        cap = make_mock_airplay(_bgr_frame_1080p(), width=640, height=480)
        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            result = cap.capture()
        assert result.width == 640
        assert result.height == 480
        assert result.image.size == (640, 480)

    def test_no_resize_when_matching(self) -> None:
        """Should skip resize when frame already matches target resolution."""
        mock_cv2 = _make_mock_cv2()
        cap = make_mock_airplay(_bgr_frame_720p(), width=1280, height=720)
        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            result = cap.capture()
        assert result.width == 1280
        assert result.height == 720

    def test_image_is_rgb(self) -> None:
        """Output PIL Image should be in RGB mode."""
        mock_cv2 = _make_mock_cv2()
        cap = make_mock_airplay(_bgr_frame_1080p())
        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            result = cap.capture()
        assert result.image.mode == "RGB"

    def test_base64_round_trip(self) -> None:
        """Base64 encode/decode round-trip preserves image."""
        mock_cv2 = _make_mock_cv2()
        cap = make_mock_airplay(_bgr_frame_with_content())
        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            result = cap.capture()
        jpeg_bytes = base64.b64decode(result.base64_jpeg)
        img = Image.open(io.BytesIO(jpeg_bytes))
        assert img.size == (1280, 720)
        assert img.mode == "RGB"


# ---------------------------------------------------------------------------
# TestCaptureErrors
# ---------------------------------------------------------------------------

class TestCaptureErrors:
    """Tests for error handling."""

    def test_capture_before_open(self) -> None:
        """Calling capture() without open() should raise CaptureError."""
        cap = AirPlayCapture()
        with pytest.raises(CaptureError, match="not opened"):
            cap.capture()

    def test_capture_before_open_file_mode(self) -> None:
        """Calling capture() in file mode without open() should raise CaptureError."""
        cap = AirPlayCapture()
        cap._use_rtp = False
        with pytest.raises(CaptureError, match="not opened"):
            cap.capture()

    def test_read_failure(self) -> None:
        """cap.read() returning False should raise CaptureError."""
        mock_cv2 = _make_mock_cv2()
        cap = make_mock_airplay()
        cap._cap.read.return_value = (False, None)
        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            with pytest.raises(CaptureError, match="Failed to read AirPlay frame"):
                cap.capture()

    def test_uxplay_exit_during_capture(self) -> None:
        """If uxplay exits unexpectedly, capture() should raise CaptureError."""
        mock_cv2 = _make_mock_cv2()
        cap = make_mock_airplay()
        cap._uxplay_proc.poll.return_value = 1  # process exited
        cap._uxplay_proc.returncode = 1
        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            with pytest.raises(CaptureError, match="uxplay process exited"):
                cap.capture()

    def test_uxplay_exit_during_capture_file_mode(self, tmp_path: Path) -> None:
        """If uxplay exits in file mode, capture() should raise CaptureError."""
        frame_dir = str(tmp_path / "frames")
        os.makedirs(frame_dir)
        cap = make_mock_airplay_file_mode(frame_dir=frame_dir)
        cap._uxplay_proc.poll.return_value = 1
        cap._uxplay_proc.returncode = 1
        with pytest.raises(CaptureError, match="uxplay process exited"):
            cap.capture()

    @patch("cyberraccoon.capture.airplay_capture.shutil")
    def test_open_uxplay_not_installed(self, mock_shutil: MagicMock) -> None:
        """open() should raise CaptureError if uxplay is not found."""
        mock_shutil.which.return_value = None
        cap = AirPlayCapture()
        with pytest.raises(CaptureError, match="uxplay is not installed"):
            cap.open()

    @patch("cyberraccoon.capture.airplay_capture.time")
    @patch("cyberraccoon.capture.airplay_capture.subprocess")
    @patch("cyberraccoon.capture.airplay_capture.shutil")
    def test_open_no_gstreamer(
        self,
        mock_shutil: MagicMock,
        mock_subprocess: MagicMock,
        mock_time: MagicMock,
    ) -> None:
        """open() should raise CaptureError if OpenCV lacks GStreamer (RTP mode)."""
        mock_shutil.which.return_value = "/usr/bin/uxplay"
        _setup_subprocess_for_open(mock_subprocess)
        cap = AirPlayCapture()

        mock_cv2 = MagicMock()
        mock_cv2.getBuildInformation.return_value = "GStreamer:                   NO"
        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            with pytest.raises(CaptureError, match="GStreamer"):
                cap.open()

    @patch("cyberraccoon.capture.airplay_capture.time")
    @patch("cyberraccoon.capture.airplay_capture.subprocess")
    @patch("cyberraccoon.capture.airplay_capture.shutil")
    def test_open_uxplay_exits_immediately(
        self,
        mock_shutil: MagicMock,
        mock_subprocess: MagicMock,
        mock_time: MagicMock,
    ) -> None:
        """open() should raise CaptureError if uxplay exits right after start."""
        mock_shutil.which.return_value = "/usr/bin/uxplay"
        _setup_subprocess_for_open(mock_subprocess)

        # Mock uxplay that exits immediately
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 1
        mock_proc.returncode = 1
        mock_proc.stderr = MagicMock()
        mock_proc.stderr.read.return_value = b"mDNS error"
        mock_subprocess.Popen.return_value = mock_proc

        cap = AirPlayCapture()

        mock_cv2 = MagicMock()
        mock_cv2.getBuildInformation.return_value = "GStreamer:                   YES"
        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            with pytest.raises(CaptureError, match="uxplay exited immediately"):
                cap.open()

    @patch("cyberraccoon.capture.airplay_capture.time")
    @patch("cyberraccoon.capture.airplay_capture.subprocess")
    @patch("cyberraccoon.capture.airplay_capture.shutil")
    def test_open_gstreamer_pipeline_fails(
        self,
        mock_shutil: MagicMock,
        mock_subprocess: MagicMock,
        mock_time: MagicMock,
    ) -> None:
        """open() should raise CaptureError if GStreamer pipeline fails to open."""
        mock_shutil.which.return_value = "/usr/bin/uxplay"
        _setup_subprocess_for_open(mock_subprocess)

        # Mock uxplay that stays alive
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_subprocess.Popen.return_value = mock_proc

        cap = AirPlayCapture()

        mock_cv2 = MagicMock()
        mock_cv2.getBuildInformation.return_value = "GStreamer:                   YES"
        mock_cv_cap = MagicMock()
        mock_cv_cap.isOpened.return_value = False
        mock_cv2.VideoCapture.return_value = mock_cv_cap
        mock_cv2.CAP_GSTREAMER = 1800

        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            with pytest.raises(CaptureError, match="Failed to open GStreamer"):
                cap.open()

        # uxplay should be terminated on pipeline failure
        mock_proc.terminate.assert_called_once()


# ---------------------------------------------------------------------------
# TestDeviceLifecycle
# ---------------------------------------------------------------------------

class TestDeviceLifecycle:
    """Tests for open/close/is_open lifecycle."""

    def test_is_open_initially_false(self) -> None:
        """New AirPlayCapture should not be open."""
        cap = AirPlayCapture()
        assert cap.is_open() is False

    def test_is_open_after_mock_setup(self) -> None:
        """Should report open after mock setup."""
        cap = make_mock_airplay()
        assert cap.is_open() is True

    def test_is_open_after_mock_setup_file_mode(self, tmp_path: Path) -> None:
        """Should report open in file mode after mock setup."""
        cap = make_mock_airplay_file_mode(frame_dir=str(tmp_path))
        assert cap.is_open() is True

    def test_is_open_false_when_uxplay_exits(self) -> None:
        """is_open() should return False if uxplay has exited."""
        cap = make_mock_airplay()
        cap._uxplay_proc.poll.return_value = 0  # exited
        assert cap.is_open() is False

    def test_is_open_false_when_uxplay_exits_file_mode(self, tmp_path: Path) -> None:
        """is_open() should return False if uxplay exits in file mode."""
        cap = make_mock_airplay_file_mode(frame_dir=str(tmp_path))
        cap._uxplay_proc.poll.return_value = 0
        assert cap.is_open() is False

    def test_close_releases_capture(self) -> None:
        """close() should release the GStreamer capture."""
        cap = make_mock_airplay()
        mock_cv_cap = cap._cap
        cap.close()
        mock_cv_cap.release.assert_called_once()

    def test_close_terminates_uxplay(self) -> None:
        """close() should terminate the uxplay subprocess."""
        cap = make_mock_airplay()
        mock_proc = cap._uxplay_proc
        cap.close()
        mock_proc.terminate.assert_called_once()

    def test_close_sets_is_open_false(self) -> None:
        """close() should make is_open() return False."""
        cap = make_mock_airplay()
        cap.close()
        assert cap.is_open() is False

    def test_double_close_safe(self) -> None:
        """Calling close() twice should not raise."""
        cap = make_mock_airplay()
        cap.close()
        cap.close()  # Should not raise

    def test_close_without_open_safe(self) -> None:
        """Calling close() on un-opened capture should not raise."""
        cap = AirPlayCapture()
        cap.close()  # Should not raise

    def test_close_resets_stream_connected(self) -> None:
        """close() should reset _stream_connected so re-opened capture can wait."""
        mock_cv2 = _make_mock_cv2()
        cap = make_mock_airplay()

        # First capture succeeds → _stream_connected = True
        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            cap.capture()
        assert cap._stream_connected is True

        # close() should reset it
        cap.close()
        assert cap._stream_connected is False

    @patch("cyberraccoon.capture.airplay_capture.time")
    @patch("cyberraccoon.capture.airplay_capture.subprocess")
    @patch("cyberraccoon.capture.airplay_capture.shutil")
    def test_full_open_lifecycle(
        self,
        mock_shutil: MagicMock,
        mock_subprocess: MagicMock,
        mock_time: MagicMock,
    ) -> None:
        """Full open → is_open → close cycle with mocked dependencies (RTP mode)."""
        mock_shutil.which.return_value = "/usr/bin/uxplay"
        _setup_subprocess_for_open(mock_subprocess)

        # Mock uxplay that stays alive
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_subprocess.Popen.return_value = mock_proc

        cap = AirPlayCapture()

        mock_cv2 = MagicMock()
        mock_cv2.getBuildInformation.return_value = "GStreamer:                   YES"
        mock_cv_cap = MagicMock()
        mock_cv_cap.isOpened.return_value = True
        mock_cv2.VideoCapture.return_value = mock_cv_cap
        mock_cv2.CAP_GSTREAMER = 1800

        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            cap.open()
            assert cap.is_open() is True
            assert cap._use_rtp is True

        cap.close()
        assert cap.is_open() is False
        mock_cv_cap.release.assert_called_once()
        mock_proc.terminate.assert_called_once()

    @patch("cyberraccoon.capture.airplay_capture.tempfile")
    @patch("cyberraccoon.capture.airplay_capture.time")
    @patch("cyberraccoon.capture.airplay_capture.subprocess")
    @patch("cyberraccoon.capture.airplay_capture.shutil")
    def test_full_open_lifecycle_file_mode(
        self,
        mock_shutil: MagicMock,
        mock_subprocess: MagicMock,
        mock_time: MagicMock,
        mock_tempfile: MagicMock,
    ) -> None:
        """Full open → is_open → close cycle in file mode."""
        mock_shutil.which.return_value = "/usr/bin/uxplay"
        mock_shutil.rmtree = MagicMock()
        _setup_subprocess_for_open(mock_subprocess, version="UxPlay 1.71\n")

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_subprocess.Popen.return_value = mock_proc

        mock_tempfile.mkdtemp.return_value = "/tmp/cyberraccoon_airplay_test"

        cap = AirPlayCapture()
        cap.open()

        assert cap.is_open() is True
        assert cap._use_rtp is False
        assert cap._frame_dir == "/tmp/cyberraccoon_airplay_test"

        cap.close()
        assert cap.is_open() is False
        mock_proc.terminate.assert_called_once()
        mock_shutil.rmtree.assert_called_once_with("/tmp/cyberraccoon_airplay_test")


# ---------------------------------------------------------------------------
# TestInit
# ---------------------------------------------------------------------------

class TestInit:
    """Tests for constructor defaults."""

    def test_default_fields(self) -> None:
        cap = AirPlayCapture()
        assert cap._target_width == 1280
        assert cap._target_height == 720
        assert cap._jpeg_quality == 80
        assert cap._rtp_port == 5004
        assert cap._uxplay_name == "CyberRaccoon"
        assert cap._uxplay_proc is None
        assert cap._cap is None
        assert cap._stream_connected is False
        assert cap._use_rtp is True
        assert cap._frame_dir is None

    def test_custom_fields(self) -> None:
        cap = AirPlayCapture(
            target_width=640,
            target_height=480,
            jpeg_quality=50,
            rtp_port=9000,
            uxplay_name="TestReceiver",
        )
        assert cap._target_width == 640
        assert cap._target_height == 480
        assert cap._jpeg_quality == 50
        assert cap._rtp_port == 9000
        assert cap._uxplay_name == "TestReceiver"


# ---------------------------------------------------------------------------
# TestVersionCheck
# ---------------------------------------------------------------------------

class TestVersionCheck:
    """Tests for _detect_uxplay_version()."""

    @patch("cyberraccoon.capture.airplay_capture.subprocess")
    def test_parses_version_173(self, mock_subprocess: MagicMock) -> None:
        result = MagicMock()
        result.stdout = "UxPlay 1.73 : An open-source AirPlay mirroring server\n"
        result.stderr = ""
        mock_subprocess.run.return_value = result
        assert AirPlayCapture._detect_uxplay_version() == 1.73

    @patch("cyberraccoon.capture.airplay_capture.subprocess")
    def test_parses_version_171(self, mock_subprocess: MagicMock) -> None:
        result = MagicMock()
        result.stdout = "UxPlay 1.71\n"
        result.stderr = ""
        mock_subprocess.run.return_value = result
        assert AirPlayCapture._detect_uxplay_version() == 1.71

    @patch("cyberraccoon.capture.airplay_capture.subprocess")
    def test_version_from_stderr(self, mock_subprocess: MagicMock) -> None:
        """Some uxplay versions print version to stderr."""
        result = MagicMock()
        result.stdout = ""
        result.stderr = "UxPlay 1.68\n"
        mock_subprocess.run.return_value = result
        assert AirPlayCapture._detect_uxplay_version() == 1.68

    @patch("cyberraccoon.capture.airplay_capture.subprocess")
    def test_unparseable_returns_none(self, mock_subprocess: MagicMock) -> None:
        result = MagicMock()
        result.stdout = "Usage: uxplay [options]\n"
        result.stderr = ""
        mock_subprocess.run.return_value = result
        assert AirPlayCapture._detect_uxplay_version() is None

    @patch("cyberraccoon.capture.airplay_capture.subprocess.run")
    def test_timeout_returns_none(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = subprocess.TimeoutExpired("uxplay", 5)
        assert AirPlayCapture._detect_uxplay_version() is None

    @patch("cyberraccoon.capture.airplay_capture.subprocess.run")
    def test_file_not_found_returns_none(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = FileNotFoundError("uxplay not found")
        assert AirPlayCapture._detect_uxplay_version() is None


# ---------------------------------------------------------------------------
# TestModeAutoDetection
# ---------------------------------------------------------------------------

class TestModeAutoDetection:
    """Tests for mode selection based on uxplay version."""

    @patch("cyberraccoon.capture.airplay_capture.time")
    @patch("cyberraccoon.capture.airplay_capture.subprocess")
    @patch("cyberraccoon.capture.airplay_capture.shutil")
    def test_version_173_selects_rtp(
        self,
        mock_shutil: MagicMock,
        mock_subprocess: MagicMock,
        mock_time: MagicMock,
    ) -> None:
        mock_shutil.which.return_value = "/usr/bin/uxplay"
        _setup_subprocess_for_open(mock_subprocess, version="UxPlay 1.73\n")

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_subprocess.Popen.return_value = mock_proc

        cap = AirPlayCapture()
        mock_cv2 = MagicMock()
        mock_cv2.getBuildInformation.return_value = "GStreamer:                   YES"
        mock_cv_cap = MagicMock()
        mock_cv_cap.isOpened.return_value = True
        mock_cv2.VideoCapture.return_value = mock_cv_cap
        mock_cv2.CAP_GSTREAMER = 1800

        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            cap.open()
        assert cap._use_rtp is True
        cap.close()

    @patch("cyberraccoon.capture.airplay_capture.tempfile")
    @patch("cyberraccoon.capture.airplay_capture.time")
    @patch("cyberraccoon.capture.airplay_capture.subprocess")
    @patch("cyberraccoon.capture.airplay_capture.shutil")
    def test_version_171_selects_file(
        self,
        mock_shutil: MagicMock,
        mock_subprocess: MagicMock,
        mock_time: MagicMock,
        mock_tempfile: MagicMock,
    ) -> None:
        mock_shutil.which.return_value = "/usr/bin/uxplay"
        mock_shutil.rmtree = MagicMock()
        _setup_subprocess_for_open(mock_subprocess, version="UxPlay 1.71\n")

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_subprocess.Popen.return_value = mock_proc
        mock_tempfile.mkdtemp.return_value = "/tmp/test_frames"

        cap = AirPlayCapture()
        cap.open()
        assert cap._use_rtp is False
        assert cap._frame_dir == "/tmp/test_frames"
        cap.close()

    @patch("cyberraccoon.capture.airplay_capture.time")
    @patch("cyberraccoon.capture.airplay_capture.subprocess")
    @patch("cyberraccoon.capture.airplay_capture.shutil")
    def test_unknown_version_defaults_to_rtp(
        self,
        mock_shutil: MagicMock,
        mock_subprocess: MagicMock,
        mock_time: MagicMock,
    ) -> None:
        """When version is unparseable, default to RTP (assume recent build)."""
        mock_shutil.which.return_value = "/usr/bin/uxplay"
        _setup_subprocess_for_open(mock_subprocess, version="Usage: uxplay\n")

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_subprocess.Popen.return_value = mock_proc

        cap = AirPlayCapture()
        mock_cv2 = MagicMock()
        mock_cv2.getBuildInformation.return_value = "GStreamer:                   YES"
        mock_cv_cap = MagicMock()
        mock_cv_cap.isOpened.return_value = True
        mock_cv2.VideoCapture.return_value = mock_cv_cap
        mock_cv2.CAP_GSTREAMER = 1800

        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            cap.open()
        assert cap._use_rtp is True
        cap.close()


# ---------------------------------------------------------------------------
# TestSubprocessManagement
# ---------------------------------------------------------------------------

class TestSubprocessManagement:
    """Tests for uxplay subprocess lifecycle management."""

    def test_stop_uxplay_terminates(self) -> None:
        cap = make_mock_airplay()
        proc = cap._uxplay_proc
        cap._stop_uxplay()
        proc.terminate.assert_called_once()
        assert cap._uxplay_proc is None

    def test_stop_uxplay_kills_on_timeout(self) -> None:
        cap = make_mock_airplay()
        proc = cap._uxplay_proc
        proc.wait.side_effect = subprocess.TimeoutExpired("uxplay", 5)
        cap._stop_uxplay()
        proc.terminate.assert_called_once()
        proc.kill.assert_called_once()

    def test_stop_uxplay_noop_when_none(self) -> None:
        cap = AirPlayCapture()
        cap._stop_uxplay()  # should not raise

    def test_detach_stderr_closes_pipe(self) -> None:
        cap = make_mock_airplay()
        stderr = cap._uxplay_proc.stderr
        cap._detach_stderr()
        stderr.close.assert_called_once()

    def test_detach_stderr_noop_when_none(self) -> None:
        cap = AirPlayCapture()
        cap._detach_stderr()  # should not raise

    def test_detach_stderr_noop_when_no_stderr(self) -> None:
        cap = make_mock_airplay()
        cap._uxplay_proc.stderr = None
        cap._detach_stderr()  # should not raise

    def test_check_uxplay_alive_raises_on_exit(self) -> None:
        cap = make_mock_airplay()
        cap._uxplay_proc.poll.return_value = 1
        cap._uxplay_proc.returncode = 1
        cap._uxplay_proc.stderr = MagicMock()
        cap._uxplay_proc.stderr.read.return_value = b"mDNS error"
        with pytest.raises(CaptureError, match="uxplay exited immediately"):
            cap._check_uxplay_alive()

    def test_check_uxplay_alive_ok_when_running(self) -> None:
        cap = make_mock_airplay()
        cap._uxplay_proc.poll.return_value = None
        cap._check_uxplay_alive()  # should not raise


# ---------------------------------------------------------------------------
# TestSystemReadyCheck
# ---------------------------------------------------------------------------

class TestSystemReadyCheck:
    """Tests for _wait_for_system_ready() — pre-start network/mDNS checks."""

    @patch("cyberraccoon.capture.airplay_capture.subprocess")
    def test_skips_on_macos(self, mock_subprocess: MagicMock) -> None:
        """Should silently return if systemctl is not available (macOS dev)."""
        mock_subprocess.run.side_effect = FileNotFoundError("systemctl")
        cap = AirPlayCapture()
        cap._wait_for_system_ready()  # should not raise

    @patch("cyberraccoon.capture.airplay_capture.subprocess")
    def test_warns_if_avahi_not_active(self, mock_subprocess: MagicMock) -> None:
        """Should return early with warning if avahi-daemon is not active."""
        result = MagicMock()
        result.returncode = 3  # systemd "inactive"
        mock_subprocess.run.return_value = result
        cap = AirPlayCapture()
        cap._wait_for_system_ready()  # should not raise

    @patch("cyberraccoon.capture.airplay_capture.time")
    @patch("cyberraccoon.capture.airplay_capture.subprocess")
    def test_waits_for_routable_ip(
        self, mock_subprocess: MagicMock, mock_time: MagicMock,
    ) -> None:
        """Should poll until a routable IPv4 address appears."""
        systemctl_result = MagicMock()
        systemctl_result.returncode = 0

        # First hostname -I: only link-local, second: routable
        hostname_i_linklocal = MagicMock()
        hostname_i_linklocal.stdout = "169.254.1.1 fe80::1"
        hostname_i_routable = MagicMock()
        hostname_i_routable.stdout = "192.168.1.100 fe80::1"

        hostname_s = MagicMock()
        hostname_s.stdout = "raspberrypi"

        avahi_resolve = MagicMock()
        avahi_resolve.returncode = 0
        avahi_resolve.stdout = "raspberrypi.local\t192.168.1.100"

        mock_subprocess.run.side_effect = [
            systemctl_result,       # systemctl is-active
            hostname_i_linklocal,   # hostname -I (link-local only)
            hostname_i_routable,    # hostname -I (routable)
            hostname_s,             # hostname -s
            avahi_resolve,          # avahi-resolve
        ]

        # time.monotonic: returns increasing values, always within deadline
        mock_time.monotonic.side_effect = [0, 0, 1, 2, 2, 3]
        mock_time.sleep = MagicMock()

        cap = AirPlayCapture()
        cap._wait_for_system_ready()

        # Should have slept once waiting for IP
        mock_time.sleep.assert_called_with(1.0)

    @patch("cyberraccoon.capture.airplay_capture.time")
    @patch("cyberraccoon.capture.airplay_capture.subprocess")
    def test_waits_for_avahi_resolve(
        self, mock_subprocess: MagicMock, mock_time: MagicMock,
    ) -> None:
        """Should poll avahi-resolve until hostname resolves."""
        systemctl_result = MagicMock()
        systemctl_result.returncode = 0

        hostname_i = MagicMock()
        hostname_i.stdout = "192.168.1.100"

        hostname_s = MagicMock()
        hostname_s.stdout = "raspberrypi"

        avahi_fail = MagicMock()
        avahi_fail.returncode = 2
        avahi_fail.stdout = ""

        avahi_ok = MagicMock()
        avahi_ok.returncode = 0
        avahi_ok.stdout = "raspberrypi.local\t192.168.1.100"

        mock_subprocess.run.side_effect = [
            systemctl_result,  # systemctl
            hostname_i,        # hostname -I
            hostname_s,        # hostname -s
            avahi_fail,        # avahi-resolve (not ready yet)
            avahi_ok,          # avahi-resolve (ready)
        ]

        mock_time.monotonic.side_effect = [0, 0, 1, 1, 2, 3]
        mock_time.sleep = MagicMock()

        cap = AirPlayCapture()
        cap._wait_for_system_ready()

        # Should have slept once during avahi polling
        assert mock_time.sleep.call_count >= 1

    @patch("cyberraccoon.capture.airplay_capture.time")
    @patch("cyberraccoon.capture.airplay_capture.subprocess")
    def test_all_ready_immediately(
        self, mock_subprocess: MagicMock, mock_time: MagicMock,
    ) -> None:
        """Should return quickly when everything is ready on first check."""
        systemctl_result = MagicMock()
        systemctl_result.returncode = 0

        hostname_i = MagicMock()
        hostname_i.stdout = "192.168.1.100"

        hostname_s = MagicMock()
        hostname_s.stdout = "raspberrypi"

        avahi_ok = MagicMock()
        avahi_ok.returncode = 0
        avahi_ok.stdout = "raspberrypi.local\t192.168.1.100"

        mock_subprocess.run.side_effect = [
            systemctl_result,  # systemctl
            hostname_i,        # hostname -I
            hostname_s,        # hostname -s
            avahi_ok,          # avahi-resolve
        ]

        mock_time.monotonic.side_effect = [0, 0, 1, 1]
        mock_time.sleep = MagicMock()

        cap = AirPlayCapture()
        cap._wait_for_system_ready()

        # No sleep needed — everything was ready
        mock_time.sleep.assert_not_called()

    @patch("cyberraccoon.capture.airplay_capture.subprocess")
    def test_skips_avahi_resolve_if_not_installed(
        self, mock_subprocess: MagicMock,
    ) -> None:
        """Should skip mDNS check if avahi-resolve is not installed."""
        systemctl_result = MagicMock()
        systemctl_result.returncode = 0

        hostname_i = MagicMock()
        hostname_i.stdout = "192.168.1.100"

        hostname_s = MagicMock()
        hostname_s.stdout = "raspberrypi"

        mock_subprocess.run.side_effect = [
            systemctl_result,     # systemctl
            hostname_i,           # hostname -I
            hostname_s,           # hostname -s
            FileNotFoundError(),  # avahi-resolve not installed
        ]

        cap = AirPlayCapture()
        cap._wait_for_system_ready()  # should not raise


# ---------------------------------------------------------------------------
# TestStreamWait (RTP mode)
# ---------------------------------------------------------------------------

class TestStreamWait:
    """Tests for _wait_for_stream() in RTP mode."""

    def test_immediate_return_when_timeout_zero(self) -> None:
        cap = make_mock_airplay()
        cap._stream_wait_timeout = 0.0
        ret, frame = cap._wait_for_stream()
        assert ret is False
        assert frame is None

    def test_immediate_return_when_no_cap(self) -> None:
        cap = AirPlayCapture()
        cap._stream_wait_timeout = 10.0
        ret, frame = cap._wait_for_stream()
        assert ret is False
        assert frame is None

    @patch("cyberraccoon.capture.airplay_capture.time")
    def test_returns_frame_when_stream_connects(self, mock_time: MagicMock) -> None:
        cap = make_mock_airplay()
        cap._stream_wait_timeout = 5.0

        frame = _bgr_frame_1080p()
        cap._cap.read.side_effect = [(False, None), (True, frame)]

        mock_time.monotonic.side_effect = [0.0, 0.5, 1.0]

        ret, result_frame = cap._wait_for_stream()
        assert ret is True
        assert result_frame is frame

    @patch("cyberraccoon.capture.airplay_capture.time")
    def test_returns_none_on_timeout(self, mock_time: MagicMock) -> None:
        cap = make_mock_airplay()
        cap._stream_wait_timeout = 1.0
        cap._cap.read.return_value = (False, None)

        mock_time.monotonic.side_effect = [0.0, 0.5, 2.0]

        ret, frame = cap._wait_for_stream()
        assert ret is False
        assert frame is None

    @patch("cyberraccoon.capture.airplay_capture.time")
    def test_returns_none_when_uxplay_dies(self, mock_time: MagicMock) -> None:
        cap = make_mock_airplay()
        cap._stream_wait_timeout = 10.0
        cap._uxplay_proc.poll.return_value = 1

        mock_time.monotonic.side_effect = [0.0, 0.5]

        ret, frame = cap._wait_for_stream()
        assert ret is False
        assert frame is None


# ---------------------------------------------------------------------------
# TestFileModeCapture
# ---------------------------------------------------------------------------

class TestFileModeCapture:
    """Tests for file-mode frame reading."""

    def test_read_latest_frame_basic(self, tmp_path: Path) -> None:
        """Should read the newest valid JPEG."""
        frame_dir = str(tmp_path / "frames")
        os.makedirs(frame_dir)

        _make_test_jpeg(Path(frame_dir) / "frame_00001.jpg", color=(255, 0, 0))
        _make_test_jpeg(Path(frame_dir) / "frame_00002.jpg", color=(0, 255, 0))

        cap = make_mock_airplay_file_mode(frame_dir=frame_dir)
        img = cap._read_latest_frame()

        assert img is not None
        assert img.mode == "RGB"
        pixel = img.getpixel((0, 0))
        assert pixel[1] > pixel[0]  # green channel dominant

    def test_read_latest_frame_empty_dir(self, tmp_path: Path) -> None:
        """No JPEG files returns None."""
        frame_dir = str(tmp_path / "frames")
        os.makedirs(frame_dir)

        cap = make_mock_airplay_file_mode(frame_dir=frame_dir)
        assert cap._read_latest_frame() is None

    def test_read_latest_frame_skips_truncated(self, tmp_path: Path) -> None:
        """Truncated JPEG should be skipped in favor of older valid file."""
        frame_dir = str(tmp_path / "frames")
        os.makedirs(frame_dir)

        _make_test_jpeg(Path(frame_dir) / "frame_00001.jpg", color=(0, 0, 255))

        truncated = Path(frame_dir) / "frame_00002.jpg"
        truncated.write_bytes(b"\xff\xd8\xff\xe0truncated")

        cap = make_mock_airplay_file_mode(frame_dir=frame_dir)
        img = cap._read_latest_frame()

        assert img is not None
        pixel = img.getpixel((0, 0))
        assert pixel[2] > pixel[0]  # blue channel dominant

    def test_read_latest_frame_none_when_frame_dir_none(self) -> None:
        cap = AirPlayCapture()
        cap._frame_dir = None
        assert cap._read_latest_frame() is None

    def test_capture_file_mode_full(self, tmp_path: Path) -> None:
        """Full capture() cycle in file mode produces valid CaptureResult."""
        frame_dir = str(tmp_path / "frames")
        os.makedirs(frame_dir)
        _make_test_jpeg(Path(frame_dir) / "frame_00001.jpg")

        cap = make_mock_airplay_file_mode(frame_dir=frame_dir)
        result = cap.capture()

        assert result.width == 1280
        assert result.height == 720
        assert result.image.size == (1280, 720)
        assert result.image.mode == "RGB"
        assert result.size_bytes > 0

        jpeg_bytes = base64.b64decode(result.base64_jpeg)
        img = Image.open(io.BytesIO(jpeg_bytes))
        assert img.format == "JPEG"


# ---------------------------------------------------------------------------
# TestFileModeStreamWait
# ---------------------------------------------------------------------------

class TestFileModeStreamWait:
    """Tests for _wait_for_stream_file()."""

    def test_immediate_return_when_timeout_zero(self, tmp_path: Path) -> None:
        cap = make_mock_airplay_file_mode(frame_dir=str(tmp_path))
        cap._stream_wait_timeout = 0.0
        assert cap._wait_for_stream_file() is None

    @patch("cyberraccoon.capture.airplay_capture.time")
    def test_returns_frame_when_file_appears(
        self, mock_time: MagicMock, tmp_path: Path,
    ) -> None:
        frame_dir = str(tmp_path / "frames")
        os.makedirs(frame_dir)

        cap = make_mock_airplay_file_mode(frame_dir=frame_dir)
        cap._stream_wait_timeout = 5.0

        _make_test_jpeg(Path(frame_dir) / "frame_00001.jpg")

        mock_time.monotonic.side_effect = [0.0, 0.5, 1.0]

        img = cap._wait_for_stream_file()
        assert img is not None
        assert img.mode == "RGB"

    @patch("cyberraccoon.capture.airplay_capture.time")
    def test_returns_none_on_timeout(
        self, mock_time: MagicMock, tmp_path: Path,
    ) -> None:
        frame_dir = str(tmp_path / "frames")
        os.makedirs(frame_dir)

        cap = make_mock_airplay_file_mode(frame_dir=frame_dir)
        cap._stream_wait_timeout = 1.0

        mock_time.monotonic.side_effect = [0.0, 0.5, 2.0]

        assert cap._wait_for_stream_file() is None

    @patch("cyberraccoon.capture.airplay_capture.time")
    def test_returns_none_when_uxplay_dies(
        self, mock_time: MagicMock, tmp_path: Path,
    ) -> None:
        frame_dir = str(tmp_path / "frames")
        os.makedirs(frame_dir)

        cap = make_mock_airplay_file_mode(frame_dir=frame_dir)
        cap._stream_wait_timeout = 10.0
        cap._uxplay_proc.poll.return_value = 1

        mock_time.monotonic.side_effect = [0.0, 0.5]

        assert cap._wait_for_stream_file() is None


# ---------------------------------------------------------------------------
# TestFileModeClosure
# ---------------------------------------------------------------------------

class TestFileModeClosure:
    """Tests for close() in file mode."""

    def test_close_removes_frame_dir(self, tmp_path: Path) -> None:
        frame_dir = str(tmp_path / "frames")
        os.makedirs(frame_dir)
        _make_test_jpeg(Path(frame_dir) / "frame_00001.jpg")

        cap = make_mock_airplay_file_mode(frame_dir=frame_dir)
        cap.close()

        assert not os.path.exists(frame_dir)
        assert cap._frame_dir is None

    def test_double_close_safe(self, tmp_path: Path) -> None:
        frame_dir = str(tmp_path / "frames")
        os.makedirs(frame_dir)

        cap = make_mock_airplay_file_mode(frame_dir=frame_dir)
        cap.close()
        cap.close()  # should not raise

    def test_close_resets_stream_connected(self, tmp_path: Path) -> None:
        frame_dir = str(tmp_path / "frames")
        os.makedirs(frame_dir)

        cap = make_mock_airplay_file_mode(frame_dir=frame_dir)
        cap._stream_connected = True
        cap.close()
        assert cap._stream_connected is False


# ---------------------------------------------------------------------------
# TestGStreamerPipeline
# ---------------------------------------------------------------------------

class TestGStreamerPipeline:
    """Tests for the GStreamer pipeline string used in RTP mode."""

    @patch("cyberraccoon.capture.airplay_capture.time")
    @patch("cyberraccoon.capture.airplay_capture.subprocess")
    @patch("cyberraccoon.capture.airplay_capture.shutil")
    def test_pipeline_uses_correct_port(
        self,
        mock_shutil: MagicMock,
        mock_subprocess: MagicMock,
        mock_time: MagicMock,
    ) -> None:
        mock_shutil.which.return_value = "/usr/bin/uxplay"
        _setup_subprocess_for_open(mock_subprocess)

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_subprocess.Popen.return_value = mock_proc

        cap = AirPlayCapture(rtp_port=7777)

        mock_cv2 = MagicMock()
        mock_cv2.getBuildInformation.return_value = "GStreamer:                   YES"
        mock_cv_cap = MagicMock()
        mock_cv_cap.isOpened.return_value = True
        mock_cv2.VideoCapture.return_value = mock_cv_cap
        mock_cv2.CAP_GSTREAMER = 1800

        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            cap.open()

        popen_call = mock_subprocess.Popen.call_args[0][0]
        assert "host=127.0.0.1 port=7777" in " ".join(popen_call)

        pipeline_arg = mock_cv2.VideoCapture.call_args[0][0]
        assert "port=7777" in pipeline_arg

        cap.close()


# ---------------------------------------------------------------------------
# TestProtocolCompliance
# ---------------------------------------------------------------------------

class TestProtocolCompliance:
    """Tests that AirPlayCapture satisfies the CaptureSource protocol."""

    def test_implements_capture_source(self) -> None:
        cap = AirPlayCapture()
        assert isinstance(cap, CaptureSource)

    def test_has_required_methods(self) -> None:
        cap = AirPlayCapture()
        assert callable(cap.open)
        assert callable(cap.capture)
        assert callable(cap.close)
        assert callable(cap.is_open)