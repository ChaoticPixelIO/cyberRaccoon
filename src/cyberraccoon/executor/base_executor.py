"""Base executor — shared command dispatch, deduplication, and timing logic.

Subclasses only need to implement open() and close() for their specific
transport (USB HID Gadget or Bluetooth L2CAP).
"""

from __future__ import annotations

import base64
import logging
import time
from abc import ABC, abstractmethod
from collections import deque
from typing import TYPE_CHECKING, Any

from config import HumanizeConfig
from executor.clipboard_bridge import TargetOS, has_non_typeable
from executor.hid_device import HIDDeviceError
from executor.keyboard import KeyboardController
from executor.mouse import MouseController

if TYPE_CHECKING:
    from executor.humanize import HumanizedKeyboardController, HumanizedMouseController

logger = logging.getLogger("M4.executor")


class BaseExecutor(ABC):
    """Abstract base for HID command executors.

    Provides:
    - Command deduplication via ``_executed_ids``
    - Action dispatch routing (click, type, key, scroll, drag)
    - Timing and result-dict formatting
    - Humanized controller wrapping helpers

    Subclasses must implement :meth:`open` and :meth:`close` for their
    transport layer.

    Usage (via subclass)::

        executor = SomeExecutor(...)
        executor.open()
        result = executor.execute({"id": "s1", "action": "click", "x": 640, "y": 360})
        executor.close()

    Result format::

        {"id": "s1", "status": "ok", "action": "click", "duration_ms": 45}
        {"id": "s1", "status": "skipped", "reason": "duplicate command id"}
        {"id": "s1", "status": "error", "action": "click", "error": "..."}
    """

    def __init__(
        self,
        humanize_config: HumanizeConfig | None = None,
        target_os: str | None = None,
    ) -> None:
        self._humanize_config = humanize_config
        self._keyboard: KeyboardController | HumanizedKeyboardController | None = None
        self._mouse: MouseController | HumanizedMouseController | None = None
        self._executed_ids: deque[str] = deque(maxlen=1000)

        # Target OS for non-ASCII error hints
        self._target_os: TargetOS | None = TargetOS(target_os) if target_os is not None else None

    # -- Abstract transport methods ----------------------------------------

    @abstractmethod
    def open(self) -> None:
        """Initialize transport and create keyboard/mouse controllers.

        Subclasses must set ``self._keyboard`` and ``self._mouse``.
        Use :meth:`_wrap_mouse_if_humanized` and
        :meth:`_wrap_keyboard_if_humanized` to optionally apply humanization.
        """

    @abstractmethod
    def close(self) -> None:
        """Release transport resources."""

    # -- Shared execution logic --------------------------------------------

    def execute(self, command: dict[str, Any]) -> dict[str, Any]:
        """Execute a single command and return a status dict.

        Handles deduplication, routing, timing, and error wrapping.
        """
        cmd_id = command.get("id", "")
        action = command.get("action", "")

        # Deduplication check
        if cmd_id and cmd_id in self._executed_ids:
            return {
                "id": cmd_id,
                "status": "skipped",
                "reason": "duplicate command id",
            }

        # "done" action: return immediately, don't track ID
        if action == "done":
            return {
                "id": cmd_id,
                "status": "ok",
                "action": "done",
                "duration_ms": 0,
            }

        # Dispatch and execute
        start = time.monotonic()
        try:
            self._dispatch(command)
            duration_ms = int((time.monotonic() - start) * 1000)

            # Track executed ID
            if cmd_id:
                self._executed_ids.append(cmd_id)

            return {
                "id": cmd_id,
                "status": "ok",
                "action": action,
                "duration_ms": duration_ms,
            }

        except (HIDDeviceError, ValueError, KeyError, AttributeError, OverflowError) as e:
            duration_ms = int((time.monotonic() - start) * 1000)
            logger.error("Execute failed [%s] %s: %s", cmd_id, action, e)
            return {
                "id": cmd_id,
                "status": "error",
                "action": action,
                "error": str(e),
                "duration_ms": duration_ms,
            }

    def _dispatch(self, command: dict[str, Any]) -> None:
        """Route command to the appropriate controller method."""
        if self._mouse is None or self._keyboard is None:
            raise HIDDeviceError("Executor not opened — call open() before execute()")
        action = command.get("action", "")

        if action == "click":
            self._mouse.click(
                x=int(command["x"]), y=int(command["y"]),
                button=command.get("button", "left"),
            )
        elif action == "double_click":
            self._mouse.double_click(x=int(command["x"]), y=int(command["y"]))
        elif action == "triple_click":
            self._mouse.triple_click(x=int(command["x"]), y=int(command["y"]))
        elif action == "type":
            text = command["text"]
            if has_non_typeable(text):
                b64 = base64.b64encode(text.encode("utf-8")).decode("ascii")
                os_hint = ""
                if self._target_os == TargetOS.MACOS:
                    os_hint = (f"\nOn macOS, run in Terminal: "
                               f"echo '{b64}' | base64 -D | pbcopy; exit\n"
                               f"Then Cmd+V to paste.")
                elif self._target_os == TargetOS.WINDOWS:
                    os_hint = (f"\nOn Windows, press Win+R and run: "
                               f"powershell -nop -w hidden -c "
                               f"\"Set-Clipboard([Text.Encoding]::UTF8.GetString("
                               f"[Convert]::FromBase64String('{b64}')))\"\n"
                               f"Then Ctrl+V to paste.")
                elif self._target_os == TargetOS.LINUX:
                    os_hint = (f"\nOn Linux, run in terminal: "
                               f"echo '{b64}' | base64 -d | xclip -selection clipboard; exit\n"
                               f"Then Ctrl+V to paste.")
                raise ValueError(
                    f"Cannot type non-ASCII text via HID keyboard. "
                    f"You MUST use the exact base64 command below — do NOT "
                    f"type the raw non-ASCII characters in any command.{os_hint}"
                )
            self._keyboard.type_text(text)
        elif action == "scroll":
            self._mouse.scroll(
                x=int(command["x"]), y=int(command["y"]),
                direction=command.get("direction", "down"),
                amount=int(command.get("amount", 3)),
            )
        elif action == "key":
            self._keyboard.press_keys(command["keys"])
        elif action == "drag":
            self._mouse.drag(
                from_x=int(command["from_x"]), from_y=int(command["from_y"]),
                to_x=int(command["to_x"]), to_y=int(command["to_y"]),
            )
        elif action == "mouse_move":
            self._mouse.move(x=int(command["x"]), y=int(command["y"]))
        elif action == "mouse_down":
            self._mouse.mouse_down(x=int(command["x"]), y=int(command["y"]))
        elif action == "mouse_up":
            self._mouse.mouse_up(x=int(command["x"]), y=int(command["y"]))
        elif action == "hold_key":
            # WARNING: press_keys does press + release (~50ms), so the key
            # is NOT actually held during the sleep. True hold requires
            # separate press_down/release_all methods on KeyboardController
            # (not yet implemented). This is a known-incorrect approximation.
            duration = max(0.0, min(float(command.get("duration_s", 1.0)), 10.0))
            self._keyboard.press_keys(command["keys"])
            time.sleep(duration)
        elif action == "wait":
            try:
                duration = max(0.0, min(float(command.get("duration_s", 1.0)), 10.0))
            except (ValueError, TypeError):
                raise ValueError(f"Invalid duration_s: {command.get('duration_s')!r}")
            time.sleep(duration)
        else:
            raise ValueError(f"Unknown action: {action}")

    # -- Humanization helpers ----------------------------------------------

    def _wrap_mouse_if_humanized(
        self, mouse: MouseController,
    ) -> MouseController | HumanizedMouseController:
        """Optionally wrap a MouseController with humanization proxy."""
        if self._humanize_config and self._humanize_config.enabled:
            from executor.humanize import HumanizedMouseController
            logger.info("Mouse humanization enabled")
            return HumanizedMouseController(mouse, self._humanize_config)
        return mouse

    def _wrap_keyboard_if_humanized(
        self, keyboard: KeyboardController,
    ) -> KeyboardController | HumanizedKeyboardController:
        """Optionally wrap a KeyboardController with humanization proxy."""
        if self._humanize_config and self._humanize_config.enabled:
            from executor.humanize import HumanizedKeyboardController
            logger.info("Keyboard humanization enabled")
            return HumanizedKeyboardController(keyboard, self._humanize_config)
        return keyboard
