"""CyberRaccoon — AI computer control via HDMI capture + USB HID.

Usage::

    # One-shot task execution
    python -m cyberraccoon --task "Open Notepad and type Hello"
    python -m cyberraccoon --task "Click the Start menu" --provider openai --model gpt-4o
    python -m cyberraccoon --task "Open Chrome" --device 0 --max-steps 20
    python -m cyberraccoon --task "Open Notepad" --source csi    # use Pi camera
    python -m cyberraccoon --task "Open Chrome" --transport bt   # use Bluetooth HID
    python -m cyberraccoon --task "Open Safari" --source airplay --transport bt

    # Web UI (FastAPI + Alpine.js)
    python -m cyberraccoon --web
    python -m cyberraccoon --web --host 0.0.0.0 --port 8080

    # Interactive CLI REPL
    python -m cyberraccoon --cli
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

from cyberraccoon.capture import available_sources, create_capture
from cyberraccoon.config import (
    HUMANIZE_PRESETS,
    LLM_PROVIDER_DEFAULTS,
    HumanizeConfig,
    resolve_api_key,
    resolve_provider_base_url,
    resolve_provider_model,
)
from cyberraccoon.agent.protocols import create_protocol
from cyberraccoon.agent.skills import SkillNotFoundError, load_skills
from cyberraccoon.agent.vision_agent import TaskStatus, VisionAgent
from cyberraccoon.executor.bluetooth_executor import BluetoothExecutor
from cyberraccoon.executor.hid_executor import ActionExecutor


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for all modules."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def on_step_print(step_info: dict) -> None:
    """Default on_step callback — print step summary to stdout."""
    step = step_info["step"]
    cmd = step_info.get("command", {})
    action = cmd.get("action", "?") if cmd else "?"
    status = step_info.get("execute_status", "?")
    latency = step_info.get("llm_latency_ms", 0)

    # Build a short description
    if action == "click":
        detail = f"({cmd.get('x')}, {cmd.get('y')})"
    elif action == "type":
        text = cmd.get("text", "")
        detail = f'"{text[:30]}..."' if len(text) > 30 else f'"{text}"'
    elif action == "key":
        keys = cmd.get("keys", [])
        detail = "+".join(keys)
    elif action == "scroll":
        detail = f"{cmd.get('direction', '?')} x{cmd.get('amount', 3)}"
    elif action == "drag":
        detail = (
            f"({cmd.get('from_x')},{cmd.get('from_y')}) -> "
            f"({cmd.get('to_x')},{cmd.get('to_y')})"
        )
    else:
        detail = ""

    print(
        f"  Step {step:>2}: {action:<12} {detail:<30} "
        f"[{status}] ({latency}ms)"
    )


def _start_web(args: argparse.Namespace, ctrl: "AppController",
               *, background: bool = False) -> None:
    """Start the Web UI server (FastAPI + uvicorn).

    Args:
        args:       Parsed CLI arguments.
        ctrl:       Shared AppController instance.
        background: If True, run uvicorn in a daemon thread (non-blocking).
    """
    import threading
    import uvicorn
    from cyberraccoon.ui.web.server import create_app

    config = ctrl.get_config()
    host = args.host or config.network.web_host
    port = args.port or config.network.web_port

    app = create_app(ctrl)

    print(f"  Web UI:   http://{host}:{port}")

    if background:
        thread = threading.Thread(
            target=uvicorn.run,
            kwargs={"app": app, "host": host, "port": port, "log_level": "warning"},
            name="M5-web-server",
            daemon=True,
        )
        thread.start()
    else:
        uvicorn.run(app, host=host, port=port, log_level="info")


def _start_cli(ctrl: "AppController") -> None:
    """Start the interactive CLI REPL (blocking)."""
    from cyberraccoon.ui.cli.repl import CLIRepl

    repl = CLIRepl(ctrl, auto_init=False)
    repl.run()


def _run_task(args: argparse.Namespace, logger: logging.Logger) -> None:
    """Run a single task (original one-shot mode)."""
    # ---- Initialize modules ----
    print("=" * 60)
    print("  CyberRaccoon — AI Computer Control")
    print("=" * 60)
    print(f"  Task:     {args.task}")
    print(f"  Provider: {args.provider} / {args.model}")
    _source_labels = {
        "uvc": f"UVC /dev/video{args.device}",
        "csi": "HDMI-CSI (TC358743)",
        "airplay": "AirPlay (waiting for connection)",
    }
    print(f"  Source:   {_source_labels.get(args.source, args.source)}")
    print(f"  Transport:{' Bluetooth HID' if args.transport == 'bt' else ' USB HID'}")
    if args.humanize:
        print(f"  Humanize: ON ({args.humanize_preset})")
    if args.skills:
        print(f"  Skills:   {', '.join(args.skills)}")
    print(f"  Limits:   {args.max_steps} steps, {args.timeout}s timeout")
    print("=" * 60)

    # M1: Screen Capture — use factory
    source_kwargs: dict[str, object] = {}
    if args.source == "uvc":
        source_kwargs["device_index"] = args.device
    elif args.source == "csi":
        pass  # CsiHdmiCapture discovers devices dynamically
    elif args.source == "airplay":
        source_kwargs["rtp_port"] = args.rtp_port

    try:
        capture = create_capture(args.source, **source_kwargs)
        capture.open()
        logger.info("M1 %s capture initialized", args.source)
    except Exception as e:
        print(f"\nError: Failed to open capture device: {e}", file=sys.stderr)
        _source_hints = {
            "uvc": "Check: HDMI cable, UVC capture card, /dev/video*",
            "csi": "Check: TC358743 on CAM0, dtoverlay in config.txt, HDMI cable, v4l-utils installed",
            "airplay": "Check: uxplay installed, GStreamer plugins, run scripts/setup.sh --airplay",
        }
        print(f"  {_source_hints.get(args.source, 'Check device')}", file=sys.stderr)
        sys.exit(1)

    # Load skills (if specified)
    skill_text: str | None = None
    if args.skills:
        try:
            skill_text = load_skills(args.skills)
        except (SkillNotFoundError, ValueError) as e:
            capture.close()
            print(f"\nError: {e}", file=sys.stderr)
            sys.exit(1)

    # M3: Protocol
    try:
        protocol = create_protocol(
            provider=args.provider,
            model=args.model,
            api_key=args.api_key or resolve_api_key(args.provider),
            base_url=args.base_url,
            protocol_override=args.protocol,
            enable_cache=not args.no_cache,
            skill_text=skill_text,
        )
        logger.info("M3 Protocol initialized")
    except (ValueError, ImportError) as e:
        capture.close()
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)

    # M4: Action Executor
    humanize_config = (
        HUMANIZE_PRESETS[args.humanize_preset] if args.humanize else None
    )
    target_os = args.target_os if args.target_os != "auto" else None
    try:
        if args.transport == "bt":
            executor = BluetoothExecutor(
                humanize_config=humanize_config,
                target_os=target_os,
            )
        else:
            executor = ActionExecutor(
                device=args.hid_device,
                humanize_config=humanize_config,
                target_os=target_os,
            )
        executor.open()
        logger.info(
            "M4 Action Executor initialized (%s)",
            "Bluetooth" if args.transport == "bt" else "USB",
        )
    except Exception as e:
        capture.close()
        print(f"\nError: Failed to open executor: {e}", file=sys.stderr)
        if args.transport == "bt":
            print("  Check: Bluetooth enabled, run scripts/setup.sh --bt", file=sys.stderr)
        else:
            print("  Check: USB Gadget configured, run scripts/setup.sh --gadget", file=sys.stderr)
        sys.exit(1)

    # M2: Vision Agent
    agent = VisionAgent(
        capture=capture,
        protocol=protocol,
        executor=executor,
        max_steps=args.max_steps,
        task_timeout_s=args.timeout,
        post_action_delay_s=args.delay,
    )

    # ---- Run task ----
    print("\nStarting task...\n")
    start = time.monotonic()

    try:
        result = agent.run(args.task, on_step=on_step_print)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        agent.abort()
        result = None
    finally:
        capture.close()
        executor.close()
        logger.info("All devices closed")

    # ---- Print results ----
    elapsed = time.monotonic() - start

    print("\n" + "=" * 60)
    if result is None:
        print("  Result: INTERRUPTED")
    else:
        status_icon = {
            TaskStatus.COMPLETED: "OK",
            TaskStatus.FAILED: "FAIL",
            TaskStatus.ABORTED: "ABORT",
        }.get(result.status, "?")

        print(f"  Result:   {status_icon} {result.status.value.upper()}")
        print(f"  Reason:   {result.reason}")
        print(f"  Steps:    {result.total_steps}")
        print(f"  Tokens:   {result.total_input_tokens} in / "
              f"{result.total_output_tokens} out")
        if result.total_cache_read_tokens or result.total_cache_creation_tokens:
            print(f"  Cache:    {result.total_cache_read_tokens} read / "
                  f"{result.total_cache_creation_tokens} created")
        print(f"  Duration: {result.total_duration_s}s")

        # Usage summary from protocol
        usage = protocol.get_usage_summary()
        logger.debug("Protocol usage: %s", usage)

    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CyberRaccoon — AI Computer Control",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            '  python -m cyberraccoon --task "Open Notepad and type Hello"\n'
            '  python -m cyberraccoon --task "Click Start menu" --provider openai\n'
            '  python -m cyberraccoon --web                      # Web UI at :8000\n'
            '  python -m cyberraccoon --web --port 8080           # custom port\n'
            '  python -m cyberraccoon --cli                       # interactive REPL\n'
            '  python -m cyberraccoon --web --cli                 # both at once\n'
        ),
    )

    # Mode (--web and --cli can be combined; --task is standalone)
    parser.add_argument(
        "--task",
        help="Run a single task (one-shot mode)",
    )
    parser.add_argument(
        "--web", action="store_true",
        help="Start the Web UI server (FastAPI + Alpine.js)",
    )
    parser.add_argument(
        "--cli", action="store_true",
        help="Start the interactive CLI REPL",
    )

    # Capture
    parser.add_argument(
        "--source",
        choices=available_sources(),
        default=os.environ.get("CYBERRACCOON_SOURCE", "uvc"),
        help=f"Capture source: {', '.join(available_sources())} (default: uvc)",
    )
    parser.add_argument(
        "--device", type=int,
        default=int(os.environ.get("CYBERRACCOON_DEVICE", "0")),
        help="Device index for uvc/csi mode (default: 0)",
    )
    parser.add_argument(
        "--rtp-port", type=int, default=5004,
        help="RTP port for AirPlay video stream (default: 5004)",
    )

    # LLM
    parser.add_argument(
        "--provider",
        default=os.environ.get("CYBERRACCOON_PROVIDER", "openai"),
        help="LLM provider: openai or anthropic (default: openai)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name (default: from ~/.cyberraccoon/config.yaml, or provider's built-in default)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key (default: from ~/.cyberraccoon/config.yaml — set it in the web UI's Config tab)",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Custom API base URL (default: from ~/.cyberraccoon/config.yaml)",
    )
    parser.add_argument(
        "--protocol",
        choices=["auto", "native", "prompt"],
        default="auto",
        help="Protocol mode: auto (default), native (force native CU), prompt (force prompt-based)",
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Disable Anthropic prompt caching",
    )

    # Agent
    parser.add_argument(
        "--max-steps", type=int, default=50,
        help="Maximum steps per task (default: 50)",
    )
    parser.add_argument(
        "--timeout", type=float, default=600.0,
        help="Task timeout in seconds (default: 600)",
    )
    parser.add_argument(
        "--delay", type=float, default=1.0,
        help="Post-action delay in seconds (default: 1.0)",
    )

    # Executor
    parser.add_argument(
        "--transport",
        choices=["usb", "bt"],
        default=os.environ.get("CYBERRACCOON_TRANSPORT", "usb"),
        help="Transport: usb (USB HID Gadget) or bt (Bluetooth HID) (default: usb)",
    )
    parser.add_argument(
        "--hid-device", default="/dev/hidg0",
        help="HID device path for USB mode (default: /dev/hidg0)",
    )

    parser.add_argument(
        "--target-os",
        choices=["auto", "windows", "macos", "linux"],
        default=os.environ.get("CYBERRACCOON_TARGET_OS", "auto"),
        help="Target OS for non-ASCII text input via clipboard bridge (default: auto-detect)",
    )

    # Web server
    parser.add_argument(
        "--host", default=None,
        help="Web server bind address (default: from config or 0.0.0.0)",
    )
    parser.add_argument(
        "--port", type=int, default=None,
        help="Web server port (default: from config or 8000)",
    )

    # Skills
    parser.add_argument(
        "--skill", dest="skills", action="append", default=[],
        help="Load application skill(s) (repeatable, e.g. --skill wechat --skill blender)",
    )

    # Humanization
    parser.add_argument(
        "--humanize", action="store_true",
        help="Enable input humanization (anti anti-bot)",
    )
    parser.add_argument(
        "--humanize-preset",
        choices=list(HUMANIZE_PRESETS.keys()),
        default="normal",
        help="Humanization preset: subtle, normal, aggressive (default: normal)",
    )

    # Misc
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    # Resolve provider-scoped LLM defaults now that --provider is known.
    # CLI flag wins; otherwise fall back to {PROVIDER}_MODEL / _BASE_URL env
    # vars, then the provider's built-in default model.
    provider_defaults = LLM_PROVIDER_DEFAULTS.get(args.provider, {})
    if args.model is None:
        args.model = (
            resolve_provider_model(args.provider)
            or provider_defaults.get("model", "")
        )
    if args.base_url is None:
        args.base_url = (
            resolve_provider_base_url(args.provider)
            or provider_defaults.get("base_url")
        )

    # Require at least one mode
    if not args.task and not args.web and not args.cli:
        parser.print_help()
        sys.exit(1)

    # --task is incompatible with --web/--cli
    if args.task and (args.web or args.cli):
        parser.error("--task cannot be combined with --web or --cli")

    setup_logging(args.verbose)
    logger = logging.getLogger("main")

    if args.task:
        _run_task(args, logger)
        return

    # --web and/or --cli: share a single AppController
    from cyberraccoon.ui.app_controller import AppController

    ctrl = AppController()
    ctrl.load_config()
    ctrl.install_log_handler()

    # Apply CLI overrides to config
    config = ctrl.get_config()
    config.agent.protocol_override = args.protocol
    config.agent.enable_cache = not args.no_cache
    if args.skills:
        config.agent.skills = args.skills

    print("=" * 60)
    print("  CyberRaccoon")
    print("=" * 60)
    print(f"  Provider: {config.llm.provider} / {config.llm.model}")
    print(f"  Source:   {config.capture_source}")
    print(f"  Transport:{' Bluetooth HID' if config.executor_transport == 'bt' else ' USB HID'}")
    if args.protocol != "auto":
        print(f"  Protocol: {args.protocol}")
    if args.skills:
        print(f"  Skills:   {', '.join(args.skills)}")

    if args.web and args.cli:
        # Web server in background thread, CLI REPL in foreground
        _start_web(args, ctrl, background=True)
        _start_cli(ctrl)
    elif args.web:
        _start_web(args, ctrl)
    else:
        _start_cli(ctrl)


if __name__ == "__main__":
    main()
