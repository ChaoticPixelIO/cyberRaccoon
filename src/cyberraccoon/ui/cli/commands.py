"""M5 CLI command handlers.

Each command is a method on :class:`CommandHandler` that takes an argument
string and returns output text (or empty string for no output).

Command list::

    config show            — display current configuration
    config set KEY VALUE   — update a config field
    config reset           — reset to defaults
    task run GOAL          — start a task
    task abort             — abort the running task
    task status            — show task status
    capture test           — take a test screenshot
    wifi scan              — scan for Wi-Fi networks
    wifi connect SSID PWD  — connect to a Wi-Fi network
    wifi status            — show Wi-Fi status
    logs                   — show recent logs
    logs tail [N]          — show last N log lines
    logs clear             — clear log buffer
    status                 — show system status
    help                   — show help
"""

from __future__ import annotations

import logging
import textwrap
from dataclasses import fields
from typing import Any, Callable

from cyberraccoon.ui.app_controller import AppController

logger = logging.getLogger("M5.cli.cmd")


class CommandHandler:
    """Routes CLI commands to handler methods.

    Args:
        controller: The shared AppController instance.
    """

    def __init__(self, controller: AppController) -> None:
        self._ctrl = controller

        # Command table: name → (handler, help_text)
        self._commands: dict[str, tuple[Callable[[str], str], str]] = {
            "config": (self._cmd_config, "config show|set|reset — manage configuration"),
            "task": (self._cmd_task, "task run|abort|status — control tasks"),
            "capture": (self._cmd_capture, "capture test — test screenshot"),
            "wifi": (self._cmd_wifi, "wifi scan|connect|status — Wi-Fi management"),
            "logs": (self._cmd_logs, "logs [tail N|clear] — view/manage logs"),
            "status": (self._cmd_status, "status — show system status"),
            "help": (self._cmd_help, "help — show this help"),
        }

    def command_names(self) -> list[str]:
        """Return all top-level command names (for completion)."""
        extras = [
            "config show", "config set", "config reset",
            "task run", "task abort", "task status",
            "capture test",
            "wifi scan", "wifi connect", "wifi status",
            "logs tail", "logs clear",
            "quit", "exit", "help", "status",
        ]
        return sorted(set(list(self._commands.keys()) + extras))

    def execute(self, cmd: str, args: str) -> str:
        """Dispatch a command and return output text."""
        handler_entry = self._commands.get(cmd)
        if handler_entry is None:
            return f"Unknown command: {cmd}. Type 'help' for available commands."
        handler, _ = handler_entry
        try:
            return handler(args)
        except Exception as e:
            logger.error("Command error [%s %s]: %s", cmd, args, e)
            return f"Error: {e}"

    # ------------------------------------------------------------------
    # config
    # ------------------------------------------------------------------

    def _cmd_config(self, args: str) -> str:
        parts = args.split(None, 2)
        sub = parts[0].lower() if parts else "show"

        if sub == "show":
            return self._config_show()
        elif sub == "set":
            if len(parts) < 3:
                return "Usage: config set <key> <value>\nExample: config set llm.model gpt-4o"
            return self._config_set(parts[1], parts[2])
        elif sub == "reset":
            self._ctrl.reset_config()
            return "Config reset to defaults."
        else:
            return "Usage: config show|set|reset"

    def _config_show(self) -> str:
        config = self._ctrl.get_config()
        lines: list[str] = ["Configuration:"]

        lines.append(f"  capture_source:    {config.capture_source}")
        lines.append(f"  executor_transport:{config.executor_transport}")
        lines.append("")

        # Show each section
        sections = [
            ("capture", config.capture),
            ("llm", config.llm),
            ("agent", config.agent),
            ("executor", config.executor),
            ("network", config.network),
            ("ble", config.ble),
        ]
        for name, sub in sections:
            lines.append(f"  [{name}]")
            for f in fields(type(sub)):
                val = getattr(sub, f.name)
                # Mask secrets
                if f.name in ("api_key", "wifi_password") and val:
                    display = val[:4] + "..." if len(str(val)) > 4 else "***"
                else:
                    display = val
                lines.append(f"    {f.name}: {display}")
            lines.append("")

        return "\n".join(lines)

    def _config_set(self, key: str, value: str) -> str:
        # Try to coerce common types
        if value.lower() in ("true", "false"):
            typed_value: Any = value.lower() == "true"
        else:
            try:
                typed_value = int(value)
            except ValueError:
                try:
                    typed_value = float(value)
                except ValueError:
                    typed_value = value

        self._ctrl.update_config(**{key: typed_value})
        return f"Set {key} = {typed_value}"

    # ------------------------------------------------------------------
    # task
    # ------------------------------------------------------------------

    def _cmd_task(self, args: str) -> str:
        parts = args.split(None, 1)
        sub = parts[0].lower() if parts else ""

        if sub == "run":
            if len(parts) < 2 or not parts[1].strip():
                return 'Usage: task run <goal>\nExample: task run "Open Notepad"'
            goal = parts[1].strip().strip('"').strip("'")
            try:
                self._ctrl.start_task(goal)
                return ""  # Output comes via events
            except Exception as e:
                return f"Cannot start task: {e}"

        elif sub == "abort":
            self._ctrl.abort_task()
            return "Abort requested."

        elif sub == "status":
            status = self._ctrl.get_task_status()
            st = status.get("status", "unknown")
            goal = status.get("goal", "")
            lines = [f"Task status: {st}"]
            if goal:
                lines.append(f"  Goal: {goal}")
            if "reason" in status:
                lines.append(f"  Reason: {status['reason']}")
            if "total_steps" in status:
                lines.append(f"  Steps: {status['total_steps']}")
            if "total_duration_s" in status:
                lines.append(f"  Duration: {status['total_duration_s']}s")
            return "\n".join(lines)

        else:
            return "Usage: task run|abort|status"

    # ------------------------------------------------------------------
    # capture
    # ------------------------------------------------------------------

    def _cmd_capture(self, args: str) -> str:
        sub = args.strip().lower() if args else ""

        if sub == "test":
            result = self._ctrl.capture_preview()
            if result is None:
                return "Capture not available. Are modules initialised?"
            return (
                f"Capture OK: {result.width}x{result.height}, "
                f"{result.size_bytes} bytes"
            )
        else:
            return "Usage: capture test"

    # ------------------------------------------------------------------
    # wifi
    # ------------------------------------------------------------------

    def _cmd_wifi(self, args: str) -> str:
        parts = args.split(None, 2)
        sub = parts[0].lower() if parts else ""

        wm = self._ctrl.get_wifi_manager()
        if wm is None:
            return "Wi-Fi manager not available (no nmcli or wpa_supplicant)."

        if sub == "scan":
            networks = wm.scan()
            if not networks:
                return "No networks found."
            lines = ["Wi-Fi networks:"]
            for n in networks:
                conn = " *" if n.connected else "  "
                lines.append(
                    f"  {conn} {n.ssid:<28} {n.signal_strength:>4}dBm  {n.security}"
                )
            return "\n".join(lines)

        elif sub == "connect":
            if len(parts) < 2:
                return "Usage: wifi connect <ssid> [password]"
            ssid = parts[1]
            password = parts[2] if len(parts) > 2 else ""
            try:
                wm.connect(ssid, password)
                ip = wm.get_ip_address()
                return f"Connected to {ssid}" + (f" (IP: {ip})" if ip else "")
            except Exception as e:
                return f"Connection failed: {e}"

        elif sub == "status":
            lines = [f"Wi-Fi backend: {wm.backend}"]
            if wm.is_connected():
                ssid = wm.get_current_network() or "unknown"
                ip = wm.get_ip_address() or "unknown"
                lines.append(f"  Connected: {ssid}")
                lines.append(f"  IP: {ip}")
            else:
                lines.append("  Not connected")
            return "\n".join(lines)

        else:
            return "Usage: wifi scan|connect|status"

    # ------------------------------------------------------------------
    # logs
    # ------------------------------------------------------------------

    def _cmd_logs(self, args: str) -> str:
        parts = args.split() if args else []
        sub = parts[0].lower() if parts else ""

        if sub == "clear":
            self._ctrl.clear_logs()
            return "Logs cleared."

        elif sub == "tail":
            count = 20
            if len(parts) > 1:
                try:
                    count = int(parts[1])
                except ValueError:
                    return "Usage: logs tail [N]"
            entries = self._ctrl.get_logs(count)
            if not entries:
                return "(no logs)"
            return "\n".join(entries)

        else:
            # Default: show all logs
            entries = self._ctrl.get_logs()
            if not entries:
                return "(no logs)"
            return "\n".join(entries)

    # ------------------------------------------------------------------
    # status
    # ------------------------------------------------------------------

    def _cmd_status(self, args: str) -> str:
        status = self._ctrl.get_status()
        lines = ["System Status:"]
        lines.append(f"  Modules ready: {status['modules_ready']}")
        lines.append(f"  Capture:       {status['capture_source']}")
        lines.append(f"  Transport:     {status['executor_transport']}")
        lines.append(f"  LLM:           {status['llm_provider']} / {status['llm_model']}")

        task = status.get("task", {})
        lines.append(f"  Task:          {task.get('status', 'idle')}")
        if task.get("goal"):
            lines.append(f"    Goal: {task['goal']}")

        wifi = status.get("wifi", {})
        if wifi.get("available"):
            if wifi.get("connected"):
                lines.append(f"  Wi-Fi:         {wifi.get('ssid', '?')} ({wifi.get('ip', '?')})")
            else:
                lines.append("  Wi-Fi:         not connected")
        else:
            lines.append("  Wi-Fi:         unavailable")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # help
    # ------------------------------------------------------------------

    def _cmd_help(self, args: str) -> str:
        lines = ["Available commands:", ""]
        for name, (_, help_text) in sorted(self._commands.items()):
            lines.append(f"  {help_text}")
        lines.append("  quit / exit — exit the REPL")
        lines.append("")
        lines.append("Examples:")
        lines.append('  config set llm.model gpt-4o')
        lines.append('  task run "Open Notepad and type Hello"')
        lines.append('  wifi connect MyNetwork password123')
        lines.append('  logs tail 50')
        return "\n".join(lines)
