"""M1 Screen Capture CLI — debug tool for testing capture devices.

Usage::

    python -m capture.cli --device 0 --output screenshot.jpg
    python -m capture.cli --source csi --output camera.jpg
    python -m capture.cli --source airplay --output airplay.jpg
    python -m capture.cli --count 5
    python -m capture.cli --width 1920 --height 1080 --quality 95
    python -m capture.cli --info
    python -m capture.cli --diag          # hardware diagnostics (no file saved)
    python -m capture.cli --find USB      # find device index by name keyword
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

from capture import available_sources, create_capture
from capture.base import CaptureError
from capture.screen_capture import find_capture_device

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("M1.cli")


def _run_diag(device: int) -> None:
    """Run hardware diagnostics: signal, format, pixel stats, HDCP check."""
    import subprocess
    import cv2

    print("=" * 55)
    print("  CyberRaccoon M1 — Capture Card Diagnostics")
    print("=" * 55)

    # --- List all video devices ---
    print("\n[1] Detected V4L2 devices:")
    try:
        out = subprocess.run(
            ["v4l2-ctl", "--list-devices"],
            capture_output=True, text=True, timeout=5,
        ).stdout
        for line in out.splitlines():
            print("   ", line)
    except Exception as e:
        print(f"    (v4l2-ctl unavailable: {e})")

    # --- Open device ---
    print(f"\n[2] Opening /dev/video{device} ...")
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"    FAIL: Cannot open /dev/video{device}")
        print("    → Check USB connection and cable")
        return

    # --- Try MJPEG ---
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_str = "".join(
        chr((actual_fourcc_int >> i) & 0xFF) for i in [0, 8, 16, 24]
    ).strip("\x00")

    print(f"    Resolution : {actual_w}x{actual_h}")
    print(f"    Pixel fmt  : {fourcc_str}")

    # --- Flush and read frame ---
    print("\n[3] Reading frame (flushing 10 buffers) ...")
    for _ in range(10):
        cap.grab()
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        print("    FAIL: Could not read frame")
        print("    → Check HDMI cable between source and capture card")
        return

    frame_min = int(frame.min())
    frame_max = int(frame.max())
    frame_mean = float(frame.mean())
    print(f"    Frame shape: {frame.shape}")
    print(f"    Pixel stats: min={frame_min}, max={frame_max}, mean={frame_mean:.1f}")

    # --- HDCP diagnosis ---
    print("\n[4] Signal analysis:")
    if frame_max < 10:
        print("    ⚠️  NEARLY BLACK FRAME DETECTED")
        print("    Likely cause: HDCP content protection")
        print("    Solutions (pick one):")
        print("      A) Insert an HDCP stripper between source and capture card")
        print("      B) Use a cheap HDMI splitter (many strip HDCP as side effect)")
        print("      C) Disable HDCP on source GPU (if supported)")
    elif frame_max < 50:
        print("    ⚠️  Very dark frame — weak or no signal")
        print("    → Check HDMI cable, source display output is active")
    else:
        print(f"    ✓  Live signal detected (max_pixel={frame_max})")
        print("    Capture card appears to be working correctly")

    print("\n" + "=" * 55)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CyberRaccoon M1 Screen Capture CLI"
    )
    parser.add_argument(
        "--source", choices=available_sources(), default="hdmi",
        help=f"Capture source: {', '.join(available_sources())} (default: hdmi)",
    )
    parser.add_argument(
        "--device", type=int, default=0,
        help="Device index for hdmi/csi mode (default: 0)",
    )
    parser.add_argument(
        "--rtp-port", type=int, default=5004,
        help="RTP port for AirPlay video stream (default: 5004)",
    )
    parser.add_argument(
        "--output", type=str, default="capture.jpg",
        help="Output file name (default: capture.jpg)",
    )
    parser.add_argument(
        "--count", type=int, default=1,
        help="Number of frames to capture (default: 1)",
    )
    parser.add_argument(
        "--width", type=int, default=1280,
        help="Output image width (default: 1280)",
    )
    parser.add_argument(
        "--height", type=int, default=720,
        help="Output image height (default: 720)",
    )
    parser.add_argument(
        "--quality", type=int, default=80,
        help="JPEG quality 1-100 (default: 80)",
    )
    parser.add_argument(
        "--info", action="store_true",
        help="Print device info only, don't capture",
    )
    parser.add_argument(
        "--diag", action="store_true",
        help="Run hardware diagnostics: signal check, HDCP detection, pixel stats",
    )
    parser.add_argument(
        "--find", type=str, metavar="KEYWORD",
        help="Find device index by name keyword and print it (e.g. --find 'USB3 Video')",
    )

    args = parser.parse_args()

    # --find: just print the device index and exit
    if args.find:
        try:
            idx = find_capture_device(args.find)
            print(idx)
        except CaptureError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        return

    # --diag: run diagnostics and exit
    if args.diag:
        _run_diag(args.device)
        return

    # Normal capture — build source-specific kwargs and use factory
    source_kwargs: dict[str, object] = {
        "target_width": args.width,
        "target_height": args.height,
        "jpeg_quality": args.quality,
    }
    if args.source == "hdmi":
        source_kwargs["device_index"] = args.device
    elif args.source == "csi":
        pass  # CsiHdmiCapture discovers devices dynamically
    elif args.source == "picamera":
        source_kwargs["camera_index"] = args.device
    elif args.source == "airplay":
        source_kwargs["rtp_port"] = args.rtp_port

    cap = create_capture(args.source, **source_kwargs)

    try:
        cap.open()

        if args.info:
            print("Device opened successfully. Exiting.")
            return

        for i in range(args.count):
            result = cap.capture()

            if args.count == 1:
                filename = args.output
            else:
                base, ext = args.output.rsplit(".", 1)
                filename = f"{base}_{i:03d}.{ext}"

            result.image.save(filename)
            print(
                f"[{i + 1}/{args.count}] Saved: {filename} "
                f"({result.width}x{result.height}, "
                f"JPEG {result.size_bytes / 1024:.1f}KB, "
                f"Base64 {len(result.base64_jpeg)} chars)"
            )

            if args.count > 1 and i < args.count - 1:
                time.sleep(0.5)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        cap.close()


if __name__ == "__main__":
    main()
