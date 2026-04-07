"""M4 CLI — manual command-line tool for testing HID executor.

Usage::

    python -m cyberraccoon.executor.cli click 640 360
    python -m cyberraccoon.executor.cli click 640 360 right
    python -m cyberraccoon.executor.cli double_click 100 200
    python -m cyberraccoon.executor.cli type "hello world"
    python -m cyberraccoon.executor.cli key ctrl c
    python -m cyberraccoon.executor.cli scroll 640 400 down 5
    python -m cyberraccoon.executor.cli drag 100 200 500 300
    python -m cyberraccoon.executor.cli json '{"action":"click","x":640,"y":360}'
    python -m cyberraccoon.executor.cli --transport bt type "hello bluetooth"
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid

from cyberraccoon.config import HumanizeConfig, HUMANIZE_PRESETS
from cyberraccoon.executor.bluetooth_executor import BluetoothExecutor
from cyberraccoon.executor.hid_executor import ActionExecutor


def build_command(action: str, params: list[str], cmd_id: str) -> dict:
    """Parse CLI arguments into a command dict."""
    if action == "click":
        cmd = {"action": "click", "x": int(params[0]), "y": int(params[1])}
        if len(params) > 2:
            cmd["button"] = params[2]
        return {**cmd, "id": cmd_id}

    if action == "double_click":
        return {
            "action": "double_click",
            "x": int(params[0]),
            "y": int(params[1]),
            "id": cmd_id,
        }

    if action == "type":
        return {
            "action": "type",
            "text": " ".join(params),
            "id": cmd_id,
        }

    if action == "key":
        return {
            "action": "key",
            "keys": params,
            "id": cmd_id,
        }

    if action == "scroll":
        cmd: dict = {
            "action": "scroll",
            "x": int(params[0]),
            "y": int(params[1]),
            "id": cmd_id,
        }
        if len(params) > 2:
            cmd["direction"] = params[2]
        if len(params) > 3:
            cmd["amount"] = int(params[3])
        return cmd

    if action == "drag":
        return {
            "action": "drag",
            "from_x": int(params[0]),
            "from_y": int(params[1]),
            "to_x": int(params[2]),
            "to_y": int(params[3]),
            "id": cmd_id,
        }

    if action == "json":
        raw = " ".join(params)
        cmd = json.loads(raw)
        cmd.setdefault("id", cmd_id)
        return cmd

    raise ValueError(f"Unknown action: {action}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CyberRaccoon M4 HID Executor CLI"
    )
    parser.add_argument(
        "action",
        choices=["click", "double_click", "type", "key", "scroll", "drag", "json"],
        help="Action to execute",
    )
    parser.add_argument(
        "params",
        nargs="*",
        help="Action-specific parameters",
    )
    parser.add_argument(
        "--transport",
        choices=["usb", "bt"],
        default="usb",
        help="Transport: usb (USB HID Gadget) or bt (Bluetooth HID) (default: usb)",
    )
    parser.add_argument(
        "--device",
        default="/dev/hidg0",
        help="HID device path for USB mode (default: /dev/hidg0)",
    )
    parser.add_argument(
        "--target-os",
        choices=["auto", "windows", "macos", "linux"],
        default=None,
        help="Target OS for non-ASCII text input via clipboard bridge (default: auto-detect)",
    )
    parser.add_argument(
        "--humanize",
        action="store_true",
        help="Enable input humanization (anti anti-bot)",
    )
    parser.add_argument(
        "--humanize-preset",
        choices=list(HUMANIZE_PRESETS.keys()),
        default="normal",
        help="Humanization preset: subtle, normal, aggressive (default: normal)",
    )

    args = parser.parse_args()
    cmd_id = f"cli_{uuid.uuid4().hex[:8]}"

    try:
        command = build_command(args.action, args.params, cmd_id)
    except (ValueError, IndexError, json.JSONDecodeError) as e:
        print(f"Error parsing command: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Command: {json.dumps(command, ensure_ascii=False)}")

    humanize_config = (
        HUMANIZE_PRESETS[args.humanize_preset] if args.humanize else None
    )
    target_os = args.target_os if args.target_os not in (None, "auto") else None

    if args.transport == "bt":
        executor = BluetoothExecutor(
            humanize_config=humanize_config,
            target_os=target_os,
        )
    else:
        executor = ActionExecutor(
            device=args.device,
            humanize_config=humanize_config,
            target_os=target_os,
        )

    try:
        executor.open()
        result = executor.execute(command)
        print(f"Result:  {json.dumps(result, ensure_ascii=False)}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        executor.close()


if __name__ == "__main__":
    main()
