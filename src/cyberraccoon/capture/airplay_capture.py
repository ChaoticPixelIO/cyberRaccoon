"""M1 AirPlay Capture — captures target screen via AirPlay screen mirroring.

Runs ``uxplay`` as a managed subprocess to receive AirPlay mirroring from a Mac.
Supports two capture modes, auto-selected based on the installed uxplay version:

**RTP mode** (uxplay >= 1.73):
    uxplay ``-vrtp`` forwards decrypted H.264 as RTP to localhost.  An OpenCV
    GStreamer pipeline decodes frames on demand.  Requires OpenCV with GStreamer.

**File mode** (uxplay < 1.73):
    uxplay decodes H.264 internally via ``jpegenc ! multifilesink``.  The latest
    JPEG frame is read from a temporary directory.  Only requires PIL — no OpenCV
    GStreamer support needed.

Requires:
- ``uxplay`` (any version; >= 1.73 enables RTP mode, < 1.73 falls back to file mode)
- GStreamer plugins: base, good, bad, libav
- OpenCV with GStreamer support (RTP mode only)

Note: On systems without these dependencies, the module can still be *imported*
(deferred checks happen in ``open()``), but ``open()`` will raise ``CaptureError``.

Usage::

    cap = AirPlayCapture(rtp_port=5004, uxplay_name="CyberRaccoon")
    cap.open()      # starts uxplay, opens capture pipeline
    # → Mac user selects "CyberRaccoon" in AirPlay menu
    result = cap.capture()   # -> CaptureResult
    cap.close()     # terminates uxplay
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from subprocess import TimeoutExpired
import tempfile
import time
from pathlib import Path
from typing import Any

from PIL import Image

from cyberraccoon.capture.base import CaptureError, CaptureResult, frame_to_capture_result

logger = logging.getLogger("M1.airplay")


class AirPlayCapture:
    """Captures frames from an AirPlay mirroring stream via uxplay.

    Two capture pipelines are supported, auto-selected in ``open()``
    based on the detected uxplay version:

    **RTP mode** (uxplay >= 1.73):

    1. ``uxplay -vrtp`` forwards decrypted H.264 as RTP to ``localhost:<port>``.
    2. OpenCV ``VideoCapture`` with a GStreamer pipeline decodes frames.

    **File mode** (uxplay < 1.73):

    1. ``uxplay -vs 'jpegenc ! multifilesink ...'`` decodes H.264 internally
       and writes JPEG frames to a temporary directory.
    2. ``capture()`` reads the latest JPEG file from disk.

    Constructor args are source-specific; the four operational methods
    (``open/capture/close/is_open``) satisfy the :class:`CaptureSource` protocol.
    """

    def __init__(
        self,
        target_width: int = 1920,
        target_height: int = 1080,
        jpeg_quality: int = 80,
        rtp_port: int = 5004,
        uxplay_name: str = "CyberRaccoon",
        uxplay_startup_timeout: float = 3.0,
        stream_wait_timeout: float = 0.0,
    ) -> None:
        self._target_width = target_width
        self._target_height = target_height
        self._jpeg_quality = jpeg_quality
        self._rtp_port = rtp_port
        self._uxplay_name = uxplay_name
        self._uxplay_startup_timeout = uxplay_startup_timeout
        self._stream_wait_timeout = stream_wait_timeout
        self._uxplay_proc: subprocess.Popen[bytes] | None = None
        self._cap: Any = None  # cv2.VideoCapture (Any to defer cv2 import)
        self._stream_connected: bool = False
        self._use_rtp: bool = True  # determined in open()
        self._frame_dir: str | None = None  # temp dir for file mode
        self._owns_frame_dir: bool = False  # False when reusing existing uxplay

    # ------------------------------------------------------------------
    # CaptureSource protocol
    # ------------------------------------------------------------------

    def open(self) -> None:
        """Start uxplay and open the capture pipeline.

        Auto-detects uxplay version:
        - >= 1.73: RTP mode (``-vrtp`` + OpenCV GStreamer pipeline)
        - < 1.73: file mode (``jpegenc ! multifilesink`` + JPEG file read)

        Raises:
            CaptureError: If uxplay is not installed, required dependencies
                are missing, or uxplay fails to start.
        """
        if shutil.which("uxplay") is None:
            raise CaptureError(
                "uxplay is not installed. "
                "Install via: sudo apt install uxplay  "
                "or run: sudo scripts/setup.sh --airplay"
            )

        version = self._detect_uxplay_version()
        # Default to RTP if version unknown (assume recent build)
        self._use_rtp = version is None or version >= 1.73

        if self._use_rtp:
            self._open_rtp_mode()
        else:
            self._open_file_mode()

    def capture(self) -> CaptureResult:
        """Read the latest frame from the AirPlay stream.

        On the first call, if ``stream_wait_timeout > 0``, retries with a
        short sleep until a frame arrives or the timeout expires.  This gives
        the Mac user time to select the AirPlay receiver after ``open()``.

        Raises:
            CaptureError: If not opened, uxplay has exited, or frame read fails.
        """
        if self._use_rtp:
            if self._cap is None:
                raise CaptureError(
                    "AirPlay capture not opened. Call open() first."
                )
        else:
            if self._frame_dir is None:
                raise CaptureError(
                    "AirPlay capture not opened. Call open() first."
                )

        # Check uxplay is still alive (common to both modes)
        if self._uxplay_proc is not None and self._uxplay_proc.poll() is not None:
            raise CaptureError(
                f"uxplay process exited unexpectedly "
                f"(exit code {self._uxplay_proc.returncode}). "
                "AirPlay stream may have been disconnected."
            )

        if self._use_rtp:
            image = self._capture_rtp_frame()
        else:
            image = self._capture_file_frame()

        result = frame_to_capture_result(
            image,
            self._target_width,
            self._target_height,
            self._jpeg_quality,
        )

        logger.debug(
            "AirPlay frame captured: %dx%d, JPEG %.1fKB",
            self._target_width, self._target_height,
            result.size_bytes / 1024,
        )

        return result

    def close(self) -> None:
        """Release capture resources and terminate uxplay.

        Safe to call multiple times.
        """
        # Release OpenCV capture (RTP mode)
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception as e:
                logger.warning("Error releasing GStreamer capture: %s", e)
            self._cap = None

        # Terminate uxplay subprocess
        self._stop_uxplay()

        # Clean up temp frame directory (file mode)
        self._cleanup_frame_dir()

        self._stream_connected = False

        logger.info("AirPlay capture closed")

    @property
    def has_client(self) -> bool:
        """Return ``True`` if uxplay has at least one established TCP connection.

        A lightweight check (no hostname resolution) suitable for polling.
        """
        return bool(self._get_client_ip())

    @property
    def connected_client(self) -> str:
        """Try to identify the AirPlay client device name.

        Checks uxplay's TCP connections to find the remote IP, then
        resolves the hostname via mDNS (avahi-resolve).  The ``.local``
        suffix is stripped for cleaner display.

        Returns:
            Client hostname (e.g. 'MacBook-Pro'), or empty string.
        """
        remote_ip = self._get_client_ip()
        if not remote_ip:
            return ""
        name = self._resolve_hostname(remote_ip)
        # Strip '.local' suffix for cleaner display
        if name.endswith(".local"):
            name = name[:-6]
        return name

    def _get_client_ip(self) -> str:
        """Extract the remote IP from uxplay's TCP connections via ``ss``."""
        try:
            result = subprocess.run(
                ["ss", "-tnp"],
                capture_output=True, text=True, timeout=5,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return ""

        for line in result.stdout.splitlines():
            if '"uxplay"' not in line or "ESTAB" not in line:
                continue
            # Parse remote address column (5th field)
            # Format: "ESTAB 0 0 [local]:port [remote]:port users:..."
            parts = line.split()
            if len(parts) < 5:
                continue
            remote = parts[4]
            # Strip port: "[fe80::...]:50319" or "192.168.1.1:50319"
            if remote.startswith("["):
                ip = remote[1:remote.rindex("]")]
            else:
                ip = remote.rsplit(":", 1)[0]
            # Skip link-local IPv6 — resolve the IPv4 from neighbor table
            if ip.startswith("fe80::"):
                ipv4 = self._resolve_ipv6_to_ipv4(ip)
                if ipv4:
                    return ipv4
                continue
            return ip
        return ""

    @staticmethod
    def _resolve_ipv6_to_ipv4(ipv6_addr: str) -> str:
        """Find IPv4 address sharing the same MAC as a link-local IPv6 address."""
        try:
            result = subprocess.run(
                ["ip", "neigh", "show"],
                capture_output=True, text=True, timeout=5,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return ""

        # Find the MAC for the IPv6 address
        target_mac = ""
        ipv4_by_mac: dict[str, str] = {}
        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            addr, mac = parts[0], parts[4]
            if addr == ipv6_addr:
                target_mac = mac
            elif "." in addr and ":" in mac:
                ipv4_by_mac[mac] = addr

        if target_mac and target_mac in ipv4_by_mac:
            return ipv4_by_mac[target_mac]
        return ""

    @staticmethod
    def _resolve_hostname(ip: str) -> str:
        """Resolve an IP address to a hostname via avahi-resolve (mDNS)."""
        try:
            result = subprocess.run(
                ["avahi-resolve", "-a", ip],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                # Format: "192.168.1.100\tHostname.local"
                parts = result.stdout.strip().split("\t")
                if len(parts) >= 2:
                    return parts[1]
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        return ""

    def is_open(self) -> bool:
        """Check if uxplay and the capture mechanism are active."""
        if self._use_rtp:
            if self._uxplay_proc is None or self._uxplay_proc.poll() is not None:
                return False
            return self._cap is not None
        # File mode: may be reusing an external uxplay (self._uxplay_proc is None)
        if self._frame_dir is None:
            return False
        if self._uxplay_proc is not None and self._uxplay_proc.poll() is not None:
            return False
        return True

    # ------------------------------------------------------------------
    # Private: mode-specific open
    # ------------------------------------------------------------------

    def _open_rtp_mode(self) -> None:
        """Start uxplay with ``-vrtp`` and open GStreamer capture pipeline."""
        import cv2

        # Check OpenCV GStreamer support
        build_info = cv2.getBuildInformation()
        if "GStreamer:                   YES" not in build_info:
            raise CaptureError(
                "OpenCV was not built with GStreamer support. "
                "On Raspberry Pi OS, use the system package: "
                "sudo apt install python3-opencv  "
                "(pip opencv-python does NOT include GStreamer)"
            )

        # Launch uxplay subprocess
        cmd = [
            "uxplay",
            "-n", self._uxplay_name,
            "-vrtp", f"host=127.0.0.1 port={self._rtp_port}",
            "-vs", "0",   # disable video display (headless)
            "-as", "0",   # disable audio sink
        ]
        self._start_uxplay(cmd)

        # Open GStreamer capture pipeline
        #   - drop=true + max-buffers=1: always deliver the latest frame,
        #     discard older buffered frames (critical for our capture→decide→act
        #     loop where each iteration may take seconds).
        #   - sync=false: don't block on playback clock — we read on-demand.
        gst_pipeline = (
            f"udpsrc port={self._rtp_port} "
            "! application/x-rtp,encoding-name=H264,payload=96 "
            "! rtph264depay ! h264parse ! avdec_h264 "
            "! videoconvert "
            "! appsink drop=true max-buffers=1 sync=false"
        )
        self._cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

        if not self._cap.isOpened():
            self._stop_uxplay()
            raise CaptureError(
                "Failed to open GStreamer capture pipeline. "
                "Check: gstreamer1.0-libav and gstreamer1.0-plugins-good installed"
            )

        logger.info(
            "AirPlay receiver started (RTP mode): %s (port %d), "
            "target: %dx%d",
            self._uxplay_name, self._rtp_port,
            self._target_width, self._target_height,
        )

    def _open_file_mode(self) -> None:
        """Start uxplay with ``jpegenc ! multifilesink`` for file-based capture.

        If a uxplay process is already running with a ``multifilesink`` pipeline,
        reuses its frame directory instead of launching a second instance (which
        would fail with a DNS name conflict).
        """
        existing_dir = self._find_existing_uxplay_frame_dir()
        if existing_dir is not None:
            self._frame_dir = existing_dir
            self._owns_frame_dir = False
            self._uxplay_proc = None  # not managed by us

            # Remove stale frame files from a previous session so that
            # _read_latest_frame() won't return yesterday's screenshot.
            for old_frame in Path(existing_dir).glob("frame_*.jpg"):
                try:
                    old_frame.unlink()
                except OSError:
                    pass

            # Liveness check: wait up to 10s for a frame.  If none
            # appears the existing uxplay is stale — kill it and fall
            # through to launch a fresh instance.
            logger.info(
                "Found existing uxplay, probing liveness (frame dir: %s)...",
                existing_dir,
            )
            if self._probe_existing_uxplay(timeout=10.0):
                logger.info(
                    "Reusing existing uxplay (file mode), frame dir: %s, "
                    "target: %dx%d",
                    self._frame_dir,
                    self._target_width, self._target_height,
                )
                return

            # Stale — kill it
            logger.warning("Existing uxplay is stale (no frames), killing it")
            self._kill_existing_uxplay()
            self._frame_dir = None
            self._owns_frame_dir = True

        self._frame_dir = tempfile.mkdtemp(prefix="cyberraccoon_airplay_")
        self._owns_frame_dir = True

        frame_pattern = os.path.join(self._frame_dir, "frame_%05d.jpg")
        vs_pipeline = (
            f"jpegenc ! multifilesink location={frame_pattern} max-files=3"
        )

        cmd = [
            "uxplay",
            "-n", self._uxplay_name,
            "-vs", vs_pipeline,
            "-as", "0",   # disable audio sink
        ]
        try:
            self._start_uxplay(cmd)
        except CaptureError:
            self._cleanup_frame_dir()
            raise

        logger.info(
            "AirPlay receiver started (file mode): %s, frame dir: %s, "
            "target: %dx%d",
            self._uxplay_name, self._frame_dir,
            self._target_width, self._target_height,
        )

    # ------------------------------------------------------------------
    # Private: mode-specific capture
    # ------------------------------------------------------------------

    def _capture_rtp_frame(self) -> Image.Image:
        """Read a frame via the GStreamer RTP pipeline."""
        import cv2

        ret, frame = self._cap.read()

        # First capture: optionally wait for the AirPlay stream to connect
        if (not ret or frame is None) and not self._stream_connected:
            ret, frame = self._wait_for_stream()

        if not ret or frame is None:
            raise CaptureError(
                "Failed to read AirPlay frame. "
                "Check: 1) Mac is connected via AirPlay  "
                "2) uxplay is running  3) GStreamer plugins installed"
            )

        self._stream_connected = True

        # OpenCV GStreamer outputs BGR
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame_rgb)

    def _capture_file_frame(self) -> Image.Image:
        """Read the latest JPEG frame from the file-based capture directory."""
        image = self._read_latest_frame()

        if image is None and not self._stream_connected:
            image = self._wait_for_stream_file()

        if image is None:
            raise CaptureError(
                "Failed to read AirPlay frame. "
                "Check: 1) Mac is connected via AirPlay  "
                "2) uxplay is running  "
                "3) JPEG frames in " + str(self._frame_dir)
            )

        self._stream_connected = True
        return image

    # ------------------------------------------------------------------
    # Private: shared helpers
    # ------------------------------------------------------------------

    def _wait_for_system_ready(self) -> None:
        """Ensure network and mDNS services are settled before starting uxplay.

        After a fresh reboot, avahi-daemon may still be probing for
        hostname conflicts and the network address may not yet be stable.
        Starting uxplay before these settle causes the first AirPlay
        connection to drop within seconds — the Mac connects, then avahi
        re-announces the service with the finalised address, and the Mac
        interprets this as a service change and disconnects.

        This method blocks (up to ~10 s) until:
        1. avahi-daemon is active
        2. A routable (non-link-local) IPv4 address is present (DHCP done)
        3. avahi can resolve the local hostname (probe complete)

        Silently returns on non-Linux systems or if tools are missing, so
        development on macOS is unaffected.
        """
        # 1) avahi-daemon must be active
        try:
            result = subprocess.run(
                ["systemctl", "is-active", "--quiet", "avahi-daemon"],
                capture_output=True, timeout=5,
            )
            if result.returncode != 0:
                logger.warning(
                    "avahi-daemon is not active — AirPlay discovery may fail"
                )
                return
        except (FileNotFoundError, TimeoutExpired):
            # systemctl not available (macOS dev) — skip all checks
            return

        deadline = time.monotonic() + 10.0

        # 2) Wait for a routable IPv4 address (DHCP / static assignment done)
        ip_settled = False
        while time.monotonic() < deadline:
            try:
                result = subprocess.run(
                    ["hostname", "-I"],
                    capture_output=True, text=True, timeout=5,
                )
                for addr in result.stdout.split():
                    if "." in addr and not addr.startswith("169.254."):
                        logger.debug("Routable IP ready: %s", addr)
                        ip_settled = True
                        break
            except (FileNotFoundError, TimeoutExpired):
                break
            if ip_settled:
                break
            time.sleep(1.0)

        if not ip_settled:
            logger.warning(
                "No routable IPv4 address found within timeout — "
                "proceeding anyway"
            )

        # 3) Resolve own hostname via mDNS (confirms avahi probing is done)
        try:
            hostname = subprocess.run(
                ["hostname", "-s"],
                capture_output=True, text=True, timeout=5,
            ).stdout.strip()
        except (FileNotFoundError, TimeoutExpired):
            return

        if not hostname:
            return

        fqdn = f"{hostname}.local"
        while time.monotonic() < deadline:
            try:
                result = subprocess.run(
                    ["avahi-resolve", "-n", fqdn],
                    capture_output=True, text=True, timeout=5,
                )
                if result.returncode == 0 and result.stdout.strip():
                    resolved = result.stdout.strip()
                    logger.debug("mDNS ready: %s → %s", fqdn, resolved)
                    return
            except (FileNotFoundError, TimeoutExpired):
                # avahi-resolve not installed — skip
                return
            time.sleep(1.0)

        logger.warning(
            "mDNS hostname %s did not resolve within timeout — "
            "first AirPlay connection may be unstable",
            fqdn,
        )

    def _start_uxplay(self, cmd: list[str]) -> None:
        """Launch the uxplay subprocess, wait for mDNS, and detach stderr.

        Raises:
            CaptureError: If uxplay fails to start or exits immediately.
        """
        self._wait_for_system_ready()

        try:
            self._uxplay_proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError as e:
            raise CaptureError(f"Failed to start uxplay: {e}") from e

        # Wait for uxplay to register mDNS service and confirm it stays alive
        time.sleep(self._uxplay_startup_timeout)
        self._check_uxplay_alive()

        # uxplay is alive — close stderr pipe to prevent 64 KB buffer deadlock
        self._detach_stderr()

    def _check_uxplay_alive(self) -> None:
        """Verify uxplay is still running after startup delay.

        Raises:
            CaptureError: If uxplay has already exited.
        """
        if self._uxplay_proc is not None and self._uxplay_proc.poll() is not None:
            stderr_output = ""
            if self._uxplay_proc.stderr:
                stderr_output = self._uxplay_proc.stderr.read().decode(
                    errors="replace"
                )
            raise CaptureError(
                f"uxplay exited immediately (code {self._uxplay_proc.returncode}). "
                f"stderr: {stderr_output}"
            )

    @staticmethod
    def _detect_uxplay_version() -> float | None:
        """Parse uxplay version from ``uxplay -h`` output.

        Returns:
            Version as a float (e.g. ``1.73``), or ``None`` if unparseable.
        """
        try:
            result = subprocess.run(
                ["uxplay", "-h"],
                capture_output=True, text=True, timeout=5,
            )
            # uxplay -h prints version on the first line, e.g. "UxPlay 1.73 ..."
            first_line = (result.stdout or result.stderr).split("\n", 1)[0]
            match = re.search(r"(\d+\.\d+)", first_line)
            if match:
                version = float(match.group(1))
                logger.debug("uxplay version: %s", match.group(1))
                return version
            else:
                logger.warning(
                    "Could not parse uxplay version from: %r. "
                    "Proceeding — will attempt RTP mode.",
                    first_line,
                )
                return None
        except (TimeoutExpired, FileNotFoundError) as e:
            logger.warning("Could not check uxplay version: %s", e)
            return None

    # ------------------------------------------------------------------
    # Private: file-mode helpers
    # ------------------------------------------------------------------

    def _read_latest_frame(self) -> Image.Image | None:
        """Read the newest complete JPEG from the frame directory.

        Handles the race condition where the newest file may be mid-write
        by ``multifilesink``: tries files newest-first, falling back to
        older (guaranteed-complete) files on read error.

        Returns:
            PIL Image in RGB mode, or ``None`` if no valid frame is available.
        """
        if self._frame_dir is None:
            return None

        frame_dir = Path(self._frame_dir)
        try:
            files = sorted(
                frame_dir.glob("frame_*.jpg"),
                key=lambda f: f.stat().st_mtime,
                reverse=True,
            )
        except OSError:
            return None

        for filepath in files[:3]:
            try:
                img = Image.open(filepath)
                img.load()  # force full read — detects truncation
                if img.mode != "RGB":
                    img = img.convert("RGB")
                return img
            except (OSError, SyntaxError):
                # OSError: file I/O or truncated JPEG
                # SyntaxError: PIL raises this for corrupt image headers
                continue

        return None

    def _wait_for_stream_file(self) -> Image.Image | None:
        """Poll for the first JPEG frame file up to ``_stream_wait_timeout`` seconds.

        Returns:
            PIL Image in RGB mode, or ``None`` if timeout expires.
        """
        if self._stream_wait_timeout <= 0:
            return None

        logger.info(
            "Waiting up to %.0fs for AirPlay stream (file mode)...",
            self._stream_wait_timeout,
        )

        deadline = time.monotonic() + self._stream_wait_timeout
        poll_interval = 0.5  # seconds between retries

        while time.monotonic() < deadline:
            time.sleep(poll_interval)

            # Bail out if uxplay died while we're waiting
            if self._uxplay_proc is not None and self._uxplay_proc.poll() is not None:
                return None

            image = self._read_latest_frame()
            if image is not None:
                logger.info("AirPlay stream connected (file mode)")
                return image

        logger.warning(
            "Timed out waiting for AirPlay stream (%.0fs)",
            self._stream_wait_timeout,
        )
        return None

    def _cleanup_frame_dir(self) -> None:
        """Remove the temporary frame directory and all its contents.

        Only removes the directory itself for directories we created
        (``_owns_frame_dir=True``).  For reused external directories,
        the directory is left intact — stale *files* are cleaned
        separately in ``_open_file_mode()`` to avoid returning
        screenshots from a previous session.
        """
        if self._frame_dir is not None and self._owns_frame_dir:
            try:
                shutil.rmtree(self._frame_dir)
            except Exception as e:
                logger.warning(
                    "Error cleaning up frame dir %s: %s", self._frame_dir, e,
                )
        self._frame_dir = None

    @staticmethod
    def _find_existing_uxplay_frame_dir() -> str | None:
        """Detect an already-running uxplay with a multifilesink pipeline.

        Parses ``/proc/<pid>/cmdline`` for uxplay processes and extracts the
        frame output directory from the ``location=`` argument.

        Returns:
            Path to the frame directory, or ``None`` if no match found.
        """
        try:
            result = subprocess.run(
                ["pgrep", "-a", "uxplay"],
                capture_output=True, text=True, timeout=5,
            )
        except (FileNotFoundError, TimeoutExpired):
            return None

        if result.returncode != 0 or not result.stdout.strip():
            return None

        for line in result.stdout.strip().splitlines():
            # Look for multifilesink location=<path> in the command line
            match = re.search(r"location=(\S+)", line)
            if match:
                frame_path = match.group(1)
                frame_dir = os.path.dirname(frame_path)
                if os.path.isdir(frame_dir):
                    logger.debug(
                        "Found existing uxplay (pid %s) writing to %s",
                        line.split()[0], frame_dir,
                    )
                    return frame_dir

        return None

    def _probe_existing_uxplay(self, timeout: float = 10.0) -> bool:
        """Check if an existing uxplay is alive by waiting for a frame file."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            time.sleep(0.5)
            if self._read_latest_frame() is not None:
                return True
        return False

    @staticmethod
    def _kill_existing_uxplay() -> None:
        """Kill all running uxplay processes."""
        try:
            subprocess.run(
                ["pkill", "-9", "uxplay"],
                capture_output=True, timeout=5,
            )
        except (FileNotFoundError, TimeoutExpired) as e:
            logger.warning("Failed to kill existing uxplay: %s", e)

    # ------------------------------------------------------------------
    # Private: RTP-mode stream wait
    # ------------------------------------------------------------------

    def _wait_for_stream(self) -> tuple[bool, Any]:
        """Poll for the first AirPlay frame up to ``_stream_wait_timeout`` seconds.

        Returns:
            ``(ret, frame)`` tuple — same contract as ``cv2.VideoCapture.read()``.
        """
        if self._stream_wait_timeout <= 0 or self._cap is None:
            return False, None

        logger.info(
            "Waiting up to %.0fs for AirPlay stream from Mac...",
            self._stream_wait_timeout,
        )

        deadline = time.monotonic() + self._stream_wait_timeout
        poll_interval = 0.5  # seconds between retries

        while time.monotonic() < deadline:
            time.sleep(poll_interval)

            # Bail out if uxplay died while we're waiting
            if self._uxplay_proc is not None and self._uxplay_proc.poll() is not None:
                return False, None

            ret, frame = self._cap.read()
            if ret and frame is not None:
                logger.info("AirPlay stream connected")
                return ret, frame

        logger.warning(
            "Timed out waiting for AirPlay stream (%.0fs)",
            self._stream_wait_timeout,
        )
        return False, None

    # ------------------------------------------------------------------
    # Private: subprocess management
    # ------------------------------------------------------------------

    def _detach_stderr(self) -> None:
        """Close the stderr pipe of the uxplay subprocess to prevent deadlock.

        After startup diagnostics, we no longer need stderr.  Closing it
        prevents the 64 KB pipe buffer from filling up and blocking uxplay
        during long sessions.
        """
        if self._uxplay_proc is not None and self._uxplay_proc.stderr:
            try:
                self._uxplay_proc.stderr.close()
            except Exception as e:
                logger.debug("Error closing uxplay stderr pipe: %s", e)

    def _stop_uxplay(self) -> None:
        """Terminate uxplay — both tracked subprocess and any orphaned process.

        When reusing an existing uxplay (not launched by us), ``_uxplay_proc``
        is ``None``. In that case, fall back to ``pkill`` to ensure the process
        is stopped so the mDNS name is freed for the next ``open()``.
        """
        if self._uxplay_proc is not None:
            try:
                self._uxplay_proc.terminate()
                try:
                    self._uxplay_proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.warning("uxplay did not exit after SIGTERM, sending SIGKILL")
                    self._uxplay_proc.kill()
                    self._uxplay_proc.wait(timeout=3)
            except Exception as e:
                logger.warning("Error stopping uxplay: %s", e)
            self._uxplay_proc = None
        else:
            # Kill any orphaned uxplay (e.g. reused from a previous session)
            self._kill_existing_uxplay()
