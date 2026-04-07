"""M5 CLI REPL — interactive command loop for CyberRaccoon.

Provides a ``prompt_toolkit``-based REPL with command completion,
real-time step output, and graceful shutdown. Falls back to plain
``input()`` if ``prompt_toolkit`` is not installed.

Usage::

    from cyberraccoon.ui.app_controller import AppController
    from cyberraccoon.ui.cli.repl import CLIRepl

    ctrl = AppController()
    repl = CLIRepl(ctrl)
    repl.run()   # blocking

Or from the command line::

    python3 -m ui.cli.repl
"""

from __future__ import annotations

import logging
import sys
from typing import Any

from cyberraccoon.ui.app_controller import AppController, AppEvent, AppEventType
from cyberraccoon.ui.cli.commands import CommandHandler

logger = logging.getLogger("M5.cli")

# Try prompt_toolkit for fancy REPL; fall back to basic input()
try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.completion import WordCompleter
    from prompt_toolkit.history import InMemoryHistory

    _HAS_PROMPT_TOOLKIT = True
except ImportError:
    _HAS_PROMPT_TOOLKIT = False


class CLIRepl:
    """Interactive CLI REPL backed by :class:`AppController`.

    Args:
        controller: The shared AppController instance.
        auto_init: If ``True``, automatically loads config and installs
            the log handler on :meth:`run`.
    """

    PROMPT = "raccoon> "

    def __init__(
        self,
        controller: AppController,
        *,
        auto_init: bool = True,
    ) -> None:
        self._ctrl = controller
        self._auto_init = auto_init
        self._handler = CommandHandler(controller)
        self._running = False

    def run(self) -> None:
        """Start the blocking REPL loop.

        Loads config, installs the log handler, prints a welcome banner,
        and enters the read-eval-print loop until the user types ``quit``
        or presses Ctrl-D.
        """
        if self._auto_init:
            self._ctrl.load_config()
            self._ctrl.install_log_handler()

        # Subscribe to events for real-time step output
        self._ctrl.add_listener(self._on_event)

        self._print_banner()
        self._running = True

        try:
            if _HAS_PROMPT_TOOLKIT:
                self._loop_prompt_toolkit()
            else:
                self._loop_basic()
        except KeyboardInterrupt:
            print("\nInterrupted.")
        finally:
            self._running = False
            self._ctrl.remove_listener(self._on_event)
            self._ctrl.remove_log_handler()

    # ------------------------------------------------------------------
    # Input loops
    # ------------------------------------------------------------------

    def _loop_prompt_toolkit(self) -> None:
        """REPL loop using prompt_toolkit (with completion + history)."""
        completer = WordCompleter(
            self._handler.command_names(),
            ignore_case=True,
        )
        session: PromptSession[str] = PromptSession(
            history=InMemoryHistory(),
            completer=completer,
        )

        while self._running:
            try:
                line = session.prompt(self.PROMPT).strip()
            except EOFError:
                break
            except KeyboardInterrupt:
                print()  # blank line after ^C
                continue

            if not line:
                continue
            self._dispatch(line)

    def _loop_basic(self) -> None:
        """Fallback REPL loop using plain input()."""
        while self._running:
            try:
                line = input(self.PROMPT).strip()
            except EOFError:
                break
            except KeyboardInterrupt:
                print()
                continue

            if not line:
                continue
            self._dispatch(line)

    # ------------------------------------------------------------------
    # Command dispatch
    # ------------------------------------------------------------------

    def _dispatch(self, line: str) -> None:
        """Parse and execute a single command line."""
        parts = line.split(None, 1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        if cmd in ("quit", "exit"):
            self._running = False
            return

        output = self._handler.execute(cmd, args)
        if output:
            print(output)

    # ------------------------------------------------------------------
    # Event handler (real-time output)
    # ------------------------------------------------------------------

    def _on_event(self, event: AppEvent) -> None:
        """Handle AppController events in the REPL context."""
        if event.type == AppEventType.TASK_STEP:
            self._print_step(event.data)
        elif event.type == AppEventType.TASK_STARTED:
            print(f"\n  Task started: {event.data.get('goal', '')}")
        elif event.type == AppEventType.TASK_FINISHED:
            self._print_task_result(event.data)

    def _print_step(self, step_info: dict[str, Any]) -> None:
        """Format and print a single step (same style as main.py)."""
        step = step_info.get("step", "?")
        cmd = step_info.get("command", {})
        action = cmd.get("action", "?") if cmd else "?"
        status = step_info.get("execute_status", "?")
        latency = step_info.get("llm_latency_ms", 0)

        if action == "click":
            detail = f"({cmd.get('x')}, {cmd.get('y')})"
        elif action == "type":
            text = cmd.get("text", "")
            detail = f'"{text[:30]}..."' if len(text) > 30 else f'"{text}"'
        elif action == "key":
            keys = cmd.get("keys", [])
            detail = "+".join(keys) if isinstance(keys, list) else str(keys)
        elif action == "scroll":
            detail = f"{cmd.get('direction', '?')} x{cmd.get('amount', 3)}"
        elif action == "drag":
            detail = (
                f"({cmd.get('from_x')},{cmd.get('from_y')}) -> "
                f"({cmd.get('to_x')},{cmd.get('to_y')})"
            )
        elif action == "done":
            detail = cmd.get("reason", "")
        else:
            detail = ""

        print(
            f"  Step {step:>2}: {action:<12} {detail:<30} "
            f"[{status}] ({latency}ms)"
        )

    def _print_task_result(self, data: dict[str, Any]) -> None:
        """Print task completion summary."""
        status = data.get("status", "?")
        reason = data.get("reason", "")
        steps = data.get("total_steps", 0)
        duration = data.get("total_duration_s", 0)
        in_tok = data.get("total_input_tokens", 0)
        out_tok = data.get("total_output_tokens", 0)

        icon = {"completed": "\u2713", "failed": "\u2717", "aborted": "\u2298"}.get(status, "?")

        print(f"\n  {icon} Task {status}: {reason}")
        print(f"    Steps: {steps}  Duration: {duration}s  Tokens: {in_tok}+{out_tok}")

    # ------------------------------------------------------------------
    # Banner
    # ------------------------------------------------------------------

    def _print_banner(self) -> None:
        """Print welcome banner with current status."""
        config = self._ctrl.get_config()
        print()
        print("=" * 55)
        print("  CyberRaccoon CLI")
        print("=" * 55)
        print(f"  Provider:  {config.llm.provider} / {config.llm.model}")
        print(f"  Source:    {config.capture_source}")
        print(f"  Transport: {config.executor_transport}")
        if _HAS_PROMPT_TOOLKIT:
            print("  (Tab completion enabled)")
        else:
            print("  (Install prompt_toolkit for tab completion)")
        print()
        print('  Type "help" for commands, "quit" to exit.')
        print("=" * 55)
        print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point for ``python3 -m ui.cli.repl``."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    ctrl = AppController()
    repl = CLIRepl(ctrl)
    repl.run()


if __name__ == "__main__":
    main()
