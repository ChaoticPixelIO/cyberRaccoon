"""M5 AppController — central facade for all UI frontends.

All three frontends (CLI REPL, Web UI, Companion App) interact with the
system exclusively through this class. It owns:

- **Configuration** — CRUD via :class:`ConfigStore`.
- **Module lifecycle** — init / close M1 (Capture), M3 (LLM), M4 (Executor).
- **Task control** — start / abort tasks in a background thread via M2.
- **Event dispatch** — listeners receive typed :class:`AppEvent` objects.
- **Log capture** — ring-buffer handler for real-time debug logs.
- **Capture preview** — single-shot screenshot for the Web UI.

Thread safety: a single ``threading.Lock`` protects mutable state.
The VisionAgent's ``run()`` method blocks in a worker thread; UI threads
subscribe to events via callbacks.

Usage::

    ctrl = AppController()
    ctrl.add_listener(my_callback)
    ctrl.load_config()
    ctrl.init_modules()
    ctrl.start_task("Open Notepad")
    # … events fire asynchronously …
    ctrl.close_modules()
"""

from __future__ import annotations

import logging
import os
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from agent.vision_agent import TaskResult, TaskStatus, VisionAgent
from agent.protocols import create_protocol
from capture import create_capture
from capture.base import CaptureError, CaptureResult, CaptureSource
from config import AppConfig
from executor.hid_executor import ActionExecutor
from executor.bluetooth_executor import BluetoothExecutor
from ui.config_store import ConfigStore
from ui.exceptions import TaskError
from ui.wifi_manager import WiFiManager

logger = logging.getLogger("M5.controller")


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

class AppEventType(Enum):
    """Events emitted by AppController for UI consumption."""

    # Task lifecycle
    TASK_STARTED = "task_started"
    TASK_STEP = "task_step"
    WORKFLOW_EVENT = "workflow_event"
    TASK_FINISHED = "task_finished"

    # Config
    CONFIG_CHANGED = "config_changed"

    # Module lifecycle (aggregate — kept for backward compat)
    MODULES_READY = "modules_ready"
    MODULES_CLOSED = "modules_closed"

    # Individual connection lifecycle
    CAPTURE_READY = "capture_ready"
    CAPTURE_CLOSED = "capture_closed"
    EXECUTOR_READY = "executor_ready"
    EXECUTOR_CLOSED = "executor_closed"

    # Logging
    LOG_MESSAGE = "log_message"


@dataclass
class AppEvent:
    """Typed event payload broadcast to listeners.

    Attributes:
        type:      Which event occurred.
        data:      Event-specific payload (dict contents vary by type).
        timestamp: ``time.monotonic()`` when the event was created.
    """

    type: AppEventType
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.monotonic)


AppEventListener = Callable[[AppEvent], None]


# ---------------------------------------------------------------------------
# Log capture handler (ring buffer)
# ---------------------------------------------------------------------------

class LogCaptureHandler(logging.Handler):
    """Logging handler that stores records in a ring buffer.

    Used by AppController to provide ``get_logs()`` and emit
    ``LOG_MESSAGE`` events for real-time debug streaming.

    Args:
        max_records: Maximum records to keep (oldest are dropped).
        on_record:   Optional callback invoked for each record (for event dispatch).
    """

    def __init__(
        self,
        max_records: int = 1000,
        on_record: Callable[[logging.LogRecord], None] | None = None,
    ) -> None:
        super().__init__()
        self._buffer: deque[logging.LogRecord] = deque(maxlen=max_records)
        self._on_record = on_record

    def emit(self, record: logging.LogRecord) -> None:
        self._buffer.append(record)
        if self._on_record:
            try:
                self._on_record(record)
            except Exception:
                pass  # Never let log handler crash the app

    def get_records(self, count: int | None = None) -> list[logging.LogRecord]:
        """Return the most recent *count* records (all if *count* is ``None``)."""
        if count is None:
            return list(self._buffer)
        return list(self._buffer)[-count:]

    def clear(self) -> None:
        """Discard all buffered records."""
        self._buffer.clear()


# ---------------------------------------------------------------------------
# AppController
# ---------------------------------------------------------------------------

class AppController:
    """Central orchestrator bridging UI frontends to core modules.

    Args:
        config_path: Override for ConfigStore file path.
    """

    def __init__(self, config_path: str | None = None) -> None:
        self._lock = threading.RLock()

        # Config
        self._config_store = ConfigStore(config_path)
        self._config: AppConfig | None = None

        # Modules (initialised via init_capture / init_executor)
        self._capture: CaptureSource | None = None
        self._protocol: Any = None
        self._executor: ActionExecutor | BluetoothExecutor | None = None
        self._agent: VisionAgent | None = None
        self._capture_ready = False
        self._executor_ready = False
        self._capture_device_name: str = ""
        self._executor_device_name: str = ""

        # Disconnect monitors
        self._airplay_monitor_stop = threading.Event()
        self._bt_monitor_stop = threading.Event()

        # Wi-Fi (lazy init)
        self._wifi: WiFiManager | None = None

        # Task state
        self._task_thread: threading.Thread | None = None
        self._task_result: TaskResult | None = None
        self._task_goal: str = ""

        # Events
        self._listeners: list[AppEventListener] = []

        # Logging
        self._log_handler = LogCaptureHandler(
            max_records=1000,
            on_record=self._on_log_record,
        )
        self._log_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s")
        )

    # ------------------------------------------------------------------
    # Event system
    # ------------------------------------------------------------------

    def add_listener(self, listener: AppEventListener) -> None:
        """Register a callback to receive :class:`AppEvent` objects."""
        with self._lock:
            self._listeners.append(listener)

    def remove_listener(self, listener: AppEventListener) -> None:
        """Unregister a previously registered listener."""
        with self._lock:
            try:
                self._listeners.remove(listener)
            except ValueError:
                pass

    def _emit(self, event: AppEvent) -> None:
        """Broadcast an event to all registered listeners."""
        with self._lock:
            listeners = list(self._listeners)
        for fn in listeners:
            try:
                fn(event)
            except Exception as e:
                logger.warning("Event listener error: %s", e)

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def load_config(self) -> AppConfig:
        """Load configuration (3-tier merge) and emit CONFIG_CHANGED.

        Returns:
            The loaded :class:`AppConfig`.
        """
        config = self._config_store.load()
        with self._lock:
            self._config = config
        self._emit(AppEvent(
            type=AppEventType.CONFIG_CHANGED,
            data={"source": "load"},
        ))
        logger.info("Config loaded from %s", self._config_store.path)
        return config

    def get_config(self) -> AppConfig:
        """Return the current in-memory config (load first if needed)."""
        with self._lock:
            if self._config is None:
                pass  # Fall through to load
            else:
                return self._config
        return self.load_config()

    def update_config(self, **kwargs: Any) -> AppConfig:
        """Update config fields and persist to YAML.

        Accepts dotted keys like ``llm.model`` or top-level like ``capture_source``.

        Args:
            **kwargs: Mapping of dotted-path → value.

        Returns:
            The updated :class:`AppConfig`.
        """
        config = self.get_config()

        for key, value in kwargs.items():
            if "." in key:
                section, _, field_name = key.partition(".")
                sub = getattr(config, section, None)
                if sub is not None and hasattr(sub, field_name):
                    setattr(sub, field_name, value)
                else:
                    logger.warning("Unknown config key: %s", key)
            else:
                if hasattr(config, key):
                    setattr(config, key, value)
                else:
                    logger.warning("Unknown config key: %s", key)

        # When provider changes, re-resolve API key from env vars
        if "llm.provider" in kwargs:
            from config import resolve_api_key
            env_key = resolve_api_key(config.llm.provider)
            if env_key:
                config.llm.api_key = env_key

        self._config_store.save(config)
        with self._lock:
            self._config = config
        self._emit(AppEvent(
            type=AppEventType.CONFIG_CHANGED,
            data={"source": "update", "keys": list(kwargs.keys())},
        ))
        return config

    def reset_config(self) -> AppConfig:
        """Reset config to defaults (delete YAML) and reload.

        Returns:
            The default :class:`AppConfig`.
        """
        self._config_store.reset()
        return self.load_config()

    @property
    def config_store(self) -> ConfigStore:
        """Access to the underlying ConfigStore for path inspection."""
        return self._config_store

    # ------------------------------------------------------------------
    # Module lifecycle — individual connections
    # ------------------------------------------------------------------

    def init_capture(self) -> CaptureResult | None:
        """Initialise M1 (Capture), grab a test frame, emit ``CAPTURE_READY``.

        For AirPlay, this may block indefinitely waiting for a Mac to mirror.
        The capture object is stored early (before the blocking call) so that
        ``close_capture()`` can kill it from another thread if the user cancels.

        Returns:
            A test :class:`CaptureResult` on success, or ``None`` if the
            capture was cancelled mid-connect or the test frame failed.

        Raises:
            TaskError: If capture cannot be opened.
        """
        config = self.get_config()

        with self._lock:
            if self._capture_ready:
                logger.warning("Capture already initialised, closing first")
                self._close_capture_locked()

        source_kwargs: dict[str, object] = {}
        if config.capture_source == "hdmi":
            source_kwargs["device_index"] = config.capture.device_index
        elif config.capture_source == "csi":
            pass  # CsiHdmiCapture discovers devices dynamically
        elif config.capture_source == "airplay":
            # Wait indefinitely for a real AirPlay stream (user must mirror)
            source_kwargs["stream_wait_timeout"] = float("inf")

        try:
            capture = create_capture(config.capture_source, **source_kwargs)
            capture.open()
        except Exception as e:
            raise TaskError(f"Capture init failed ({config.capture_source}): {e}") from e

        # Store early so close_capture() can kill a blocking capture() call.
        with self._lock:
            self._capture = capture
            self._capture_ready = False  # not ready yet
            self._agent = None

        # Grab a test frame — may block for AirPlay waiting for stream
        test_frame: CaptureResult | None = None
        try:
            test_frame = capture.capture()
        except CaptureError as e:
            logger.warning("Test frame failed (capture still ready): %s", e)

        # Build a human-readable device name (describes the remote/hardware side)
        if config.capture_source == "hdmi":
            device_name = getattr(capture, "v4l2_device_name", f"/dev/video{config.capture.device_index}")
        elif config.capture_source == "csi":
            device_name = "TC358743 HDMI-CSI"
        elif config.capture_source == "airplay":
            device_name = getattr(capture, "connected_client", "")
        else:
            device_name = config.capture_source

        with self._lock:
            if self._capture is not capture:
                # Cancelled by close_capture() while we were waiting
                capture.close()
                return None
            self._capture_ready = True
            self._capture_device_name = device_name

        self._emit(AppEvent(
            type=AppEventType.CAPTURE_READY,
            data={"device": device_name},
        ))
        if self.modules_ready:
            self._emit(AppEvent(type=AppEventType.MODULES_READY))
        logger.info("Capture initialised (source=%s, device=%s)", config.capture_source, device_name)

        # Start AirPlay disconnect monitor
        if config.capture_source == "airplay":
            self._airplay_monitor_stop.clear()
            t = threading.Thread(
                target=self._monitor_airplay, daemon=True,
                name="airplay-monitor",
            )
            t.start()

        return test_frame

    def close_capture(self) -> None:
        """Close capture and emit ``CAPTURE_CLOSED``."""
        self._airplay_monitor_stop.set()
        was_ready = self.modules_ready
        with self._lock:
            self._close_capture_locked()
        self._emit(AppEvent(type=AppEventType.CAPTURE_CLOSED))
        if was_ready:
            self._emit(AppEvent(type=AppEventType.MODULES_CLOSED))

    def _close_capture_locked(self) -> None:
        """Internal capture close (caller must hold ``_lock``)."""
        if self._capture:
            try:
                self._capture.close()
            except Exception as e:
                logger.warning("Capture close error: %s", e)
            self._capture = None
        self._capture_ready = False
        self._capture_device_name = ""
        self._agent = None  # agent holds a capture reference
        logger.info("Capture closed")

    def _monitor_airplay(self) -> None:
        """Background thread: detect AirPlay client disconnect and auto-close."""
        while not self._airplay_monitor_stop.wait(timeout=3.0):
            with self._lock:
                capture = self._capture
                ready = self._capture_ready
            if not ready or capture is None:
                return
            if not getattr(capture, "has_client", True):
                logger.info("AirPlay client disconnected")
                self.close_capture()
                return

    def _monitor_bluetooth(self) -> None:
        """Background thread: detect Bluetooth HID disconnect and auto-close."""
        while not self._bt_monitor_stop.wait(timeout=3.0):
            with self._lock:
                executor = self._executor
                ready = self._executor_ready
            if not ready or executor is None:
                return
            if not getattr(executor, "is_connected", True):
                logger.info("Bluetooth HID client disconnected")
                self.close_executor()
                return

    def _setup_usb_gadget(self) -> None:
        """Run ``setup_gadget.sh`` to create ``/dev/hidg0``.

        Called automatically when the USB HID device file does not exist.

        Raises:
            TaskError: If the setup script fails.
        """
        script = Path(__file__).resolve().parent.parent / "scripts" / "setup_gadget.sh"
        if not script.exists():
            raise TaskError(f"USB Gadget setup script not found: {script}")
        logger.info("USB Gadget device not found, running setup_gadget.sh ...")
        try:
            result = subprocess.run(
                [str(script)],
                capture_output=True, text=True, timeout=30,
            )
        except subprocess.TimeoutExpired as e:
            raise TaskError("USB Gadget setup timed out") from e
        if result.returncode != 0:
            stderr = result.stderr.strip() or result.stdout.strip()
            raise TaskError(f"USB Gadget setup failed: {stderr}")
        logger.info("USB Gadget setup completed successfully")

    def init_executor(self) -> None:
        """Initialise M4 (Executor), emit ``EXECUTOR_READY``.

        For Bluetooth transport this may block while waiting for pairing.
        The executor is stored early so ``close_executor()`` can kill a
        blocking ``open()`` from another thread if the user cancels.

        Raises:
            TaskError: If executor cannot be opened.
        """
        config = self.get_config()

        with self._lock:
            if self._executor_ready:
                logger.warning("Executor already initialised, closing first")
                self._close_executor_locked()

        target_os = config.target_os or None
        if config.executor_transport == "bt":
            executor: ActionExecutor | BluetoothExecutor = BluetoothExecutor(
                target_os=target_os,
            )
        else:
            device_path = config.executor.device
            if not os.path.exists(device_path):
                self._setup_usb_gadget()
            executor = ActionExecutor(
                device=device_path,
                target_os=target_os,
            )

        # Store early so close_executor() can kill a blocking open().
        with self._lock:
            self._executor = executor
            self._executor_ready = False  # not ready yet
            self._agent = None

        try:
            executor.open()
        except Exception as e:
            with self._lock:
                if self._executor is executor:
                    self._executor = None
            raise TaskError(f"Executor init failed: {e}") from e

        # Build a human-readable device name (describes the remote host)
        if config.executor_transport == "bt":
            device_name = getattr(executor, "connected_host", "")
        else:
            device_name = config.executor.device

        with self._lock:
            if self._executor is not executor:
                # Cancelled by close_executor() while we were waiting
                executor.close()
                return
            self._executor_ready = True
            self._executor_device_name = device_name

        self._emit(AppEvent(
            type=AppEventType.EXECUTOR_READY,
            data={"device": device_name},
        ))
        if self.modules_ready:
            self._emit(AppEvent(type=AppEventType.MODULES_READY))
        logger.info("Executor initialised (transport=%s, device=%s)", config.executor_transport, device_name)

        # Start Bluetooth disconnect monitor
        if config.executor_transport == "bt":
            self._bt_monitor_stop.clear()
            t = threading.Thread(
                target=self._monitor_bluetooth, daemon=True,
                name="bt-monitor",
            )
            t.start()

    def close_executor(self) -> None:
        """Close executor and emit ``EXECUTOR_CLOSED``."""
        self._bt_monitor_stop.set()
        was_ready = self.modules_ready
        with self._lock:
            self._close_executor_locked()
        self._emit(AppEvent(type=AppEventType.EXECUTOR_CLOSED))
        if was_ready:
            self._emit(AppEvent(type=AppEventType.MODULES_CLOSED))

    def _close_executor_locked(self) -> None:
        """Internal executor close (caller must hold ``_lock``)."""
        if self._executor:
            try:
                self._executor.close()
            except Exception as e:
                logger.warning("Executor close error: %s", e)
            self._executor = None
        self._executor_ready = False
        self._executor_device_name = ""
        self._agent = None  # agent holds an executor reference
        logger.info("Executor closed")

    # ------------------------------------------------------------------
    # Module lifecycle — convenience wrappers (backward compat)
    # ------------------------------------------------------------------

    def init_modules(self) -> None:
        """Initialise both capture and executor, then emit ``MODULES_READY``.

        Convenience wrapper that calls :meth:`init_capture` and
        :meth:`init_executor`. Kept for backward compatibility with
        CLI REPL and existing tests.

        Raises:
            TaskError: If any module init fails.
        """
        self.init_capture()
        self.init_executor()

    def close_modules(self) -> None:
        """Close all initialised modules and emit ``MODULES_CLOSED``."""
        with self._lock:
            had_capture = self._capture_ready
            had_executor = self._executor_ready
            self._close_capture_locked()
            self._close_executor_locked()
        if had_capture:
            self._emit(AppEvent(type=AppEventType.CAPTURE_CLOSED))
        if had_executor:
            self._emit(AppEvent(type=AppEventType.EXECUTOR_CLOSED))
        self._emit(AppEvent(type=AppEventType.MODULES_CLOSED))

    # ------------------------------------------------------------------
    # Module readiness properties
    # ------------------------------------------------------------------

    @property
    def capture_ready(self) -> bool:
        """Return ``True`` if capture source is initialised and ready."""
        with self._lock:
            return self._capture is not None and self._capture_ready

    @property
    def executor_ready(self) -> bool:
        """Return ``True`` if executor is initialised and ready."""
        with self._lock:
            return self._executor is not None and self._executor_ready

    @property
    def modules_ready(self) -> bool:
        """Return ``True`` if both capture and executor are ready."""
        with self._lock:
            return (self._capture is not None and self._capture_ready
                    and self._executor is not None and self._executor_ready)

    # ------------------------------------------------------------------
    # Task control
    # ------------------------------------------------------------------

    def _ensure_agent(self) -> None:
        """Create the VisionAgent if both capture and executor are ready.

        Called lazily from :meth:`start_task` so that capture and executor
        can be connected independently and in any order.

        Caller must NOT hold ``_lock``.
        """
        config = self.get_config()

        # Load skills (if configured).
        # When skills are loaded, the planner handles them (plan-then-execute
        # mode). The executor protocol does NOT get skill_text — putting
        # skills in the tool-enabled prompt is what causes the LLM to
        # ignore them (it goes into autonomous mode instead of following
        # the skill's procedure).
        skill_text: str | None = None
        if config.agent.skills:
            from agent.skills import load_skills
            try:
                skill_text = load_skills(config.agent.skills)
            except Exception as e:
                raise TaskError(f"Failed to load skills: {e}") from e

        # M3: Protocol — skill_text intentionally NOT passed here.
        # Skills go to the planner, not the executor.
        try:
            protocol = create_protocol(
                provider=config.llm.provider,
                model=config.llm.model,
                api_key=config.llm.api_key,
                base_url=config.llm.base_url,
                history_max_turns=config.agent.history_max_turns,
                protocol_override=config.agent.protocol_override,
                enable_cache=config.agent.enable_cache,
            )
        except Exception as e:
            raise TaskError(f"Protocol init failed: {e}") from e

        with self._lock:
            agent = VisionAgent(
                capture=self._capture,
                protocol=protocol,
                executor=self._executor,
                max_steps=config.agent.max_steps,
                max_consecutive_failures=config.agent.max_consecutive_failures,
                post_action_delay_s=config.agent.post_action_delay_s,
                task_timeout_s=config.agent.task_timeout_s,
                stability_check=config.agent.stability_check,
                stability_threshold=config.agent.stability_threshold,
                stability_interval_s=config.agent.stability_interval_s,
                stability_max_wait_s=config.agent.stability_max_wait_s,
            )
            self._protocol = protocol
            self._agent = agent

    def start_task(self, goal: str) -> None:
        """Start a task in a background thread.

        Creates the LLM client and VisionAgent on demand if both capture
        and executor are ready.

        Args:
            goal: Natural language task description.

        Raises:
            TaskError: If modules not ready or a task is already running.
        """
        with self._lock:
            if not (self._capture_ready and self._executor_ready):
                raise TaskError("Modules not initialised. Call init_modules() first.")
            if self._task_thread and self._task_thread.is_alive():
                raise TaskError("A task is already running. Abort it first.")

            self._task_goal = goal
            self._task_result = None

        # Create agent on demand (outside lock — may call get_config)
        self._ensure_agent()

        self._emit(AppEvent(
            type=AppEventType.TASK_STARTED,
            data={"goal": goal},
        ))

        thread = threading.Thread(
            target=self._run_task,
            args=(goal,),
            name="M5-task-worker",
            daemon=True,
        )
        with self._lock:
            self._task_thread = thread
        thread.start()
        logger.info("Task started in background: %s", goal)

    def abort_task(self) -> None:
        """Request the running task to abort."""
        with self._lock:
            agent = self._agent
        if agent:
            agent.abort()
            logger.info("Task abort requested")

    def approve_plan(self) -> None:
        """Approve the pending workflow plan. Unblocks execution."""
        with self._lock:
            agent = self._agent
        if agent and hasattr(agent, '_workflow_runner') and agent._workflow_runner:
            agent._workflow_runner.approve_plan()
            logger.info("Workflow plan approved")

    def reject_plan(self) -> None:
        """Reject the pending workflow plan. Aborts the workflow."""
        with self._lock:
            agent = self._agent
        if agent and hasattr(agent, '_workflow_runner') and agent._workflow_runner:
            agent._workflow_runner.reject_plan()
            logger.info("Workflow plan rejected")

    def resolve_escalation(self) -> None:
        """Signal that the user resolved an escalation. Unblocks workflow."""
        with self._lock:
            agent = self._agent
        if agent and hasattr(agent, '_workflow_runner') and agent._workflow_runner:
            agent._workflow_runner.resolve_escalation()
            logger.info("Workflow escalation resolved")

    def get_task_status(self) -> dict[str, Any]:
        """Return current task state as a dict.

        Returns:
            Dict with ``status``, ``goal``, ``result`` (if finished).
        """
        with self._lock:
            thread = self._task_thread
            goal = self._task_goal
            result = self._task_result

        if thread and thread.is_alive():
            return {"status": "running", "goal": goal}
        if result:
            return {
                "status": result.status.value,
                "goal": goal,
                "reason": result.reason,
                "total_steps": result.total_steps,
                "total_duration_s": result.total_duration_s,
                "total_input_tokens": result.total_input_tokens,
                "total_output_tokens": result.total_output_tokens,
            }
        return {"status": "idle", "goal": ""}

    @property
    def task_running(self) -> bool:
        """Return ``True`` if a task thread is currently alive."""
        with self._lock:
            return self._task_thread is not None and self._task_thread.is_alive()

    def _run_task(self, goal: str) -> None:
        """Worker thread: run VisionAgent or workflow and emit events."""
        try:
            with self._lock:
                agent = self._agent
            if not agent:
                logger.error("Agent not available in task thread")
                return

            config = self.get_config()
            use_workflow = bool(config.agent.skills)

            if use_workflow:
                # Plan-then-execute: planner decomposes task into steps,
                # agent executes each step individually.
                from agent.planner import TaskPlanner

                planner = TaskPlanner(
                    provider=config.llm.provider,
                    model=config.llm.model,
                    api_key=config.llm.api_key,
                    base_url=config.llm.base_url,
                )

                # Load skill text for the planner
                skill_text: str | None = None
                if config.agent.skills:
                    from agent.skills import load_skills
                    try:
                        skill_text = load_skills(config.agent.skills)
                    except Exception as e:
                        logger.warning("Failed to load skills for planner: %s", e)

                workflow_result = agent.run_workflow(
                    task_goal=goal,
                    planner=planner,
                    skill_text=skill_text,
                    on_step=self._on_step_bridge,
                )

                # Convert WorkflowResult to TaskResult for UI.
                # Token usage is aggregated from per-step results
                # reported via the on_step callback (forwarded to
                # agent.run() since fix 8).
                status_map = {
                    "completed": TaskStatus.COMPLETED,
                    "aborted": TaskStatus.ABORTED,
                }
                with self._lock:
                    self._task_result = TaskResult(
                        status=status_map.get(
                            workflow_result.status, TaskStatus.FAILED,
                        ),
                        reason=workflow_result.reason,
                        total_steps=workflow_result.steps_completed,
                        total_input_tokens=0,
                        total_output_tokens=0,
                        total_duration_s=workflow_result.total_duration_s,
                    )

                self._emit(AppEvent(
                    type=AppEventType.TASK_FINISHED,
                    data={
                        "status": workflow_result.status,
                        "reason": workflow_result.reason,
                        "total_steps": workflow_result.steps_completed,
                        "total_duration_s": workflow_result.total_duration_s,
                        "total_input_tokens": 0,
                        "total_output_tokens": 0,
                        "total_cache_read_tokens": 0,
                        "total_cache_creation_tokens": 0,
                        "workflow": True,
                        "steps_total": workflow_result.steps_total,
                    },
                ))
            else:
                result = agent.run(goal, on_step=self._on_step_bridge)

                with self._lock:
                    self._task_result = result

                self._emit(AppEvent(
                    type=AppEventType.TASK_FINISHED,
                    data={
                        "status": result.status.value,
                        "reason": result.reason,
                        "total_steps": result.total_steps,
                        "total_duration_s": result.total_duration_s,
                        "total_input_tokens": result.total_input_tokens,
                        "total_output_tokens": result.total_output_tokens,
                        "total_cache_read_tokens": result.total_cache_read_tokens,
                        "total_cache_creation_tokens": result.total_cache_creation_tokens,
                    },
                ))
        except Exception as e:
            logger.error("Task thread error: %s", e, exc_info=True)
            with self._lock:
                self._task_result = TaskResult(
                    status=TaskStatus.FAILED,
                    reason=f"Internal error: {e}",
                    total_steps=0,
                    total_input_tokens=0,
                    total_output_tokens=0,
                    total_duration_s=0.0,
                )
            self._emit(AppEvent(
                type=AppEventType.TASK_FINISHED,
                data={"status": "failed", "reason": str(e)},
            ))

    def _on_step_bridge(self, step_info: dict[str, Any]) -> None:
        """Bridge VisionAgent's on_step callback to the event system.

        Distinguishes between workflow-level events (plan_ready, step_start,
        step_done, etc.) and per-action agent events (screenshots, clicks).
        Workflow events go as 'workflow_event', agent events as 'task_step'.
        """
        if "type" in step_info:
            # Workflow-level event from WorkflowRunner
            self._emit(AppEvent(
                type=AppEventType.WORKFLOW_EVENT,
                data=step_info,
            ))
        else:
            # Per-action event from VisionAgent.run()
            self._emit(AppEvent(
                type=AppEventType.TASK_STEP,
                data=step_info,
            ))

    # ------------------------------------------------------------------
    # Capture preview
    # ------------------------------------------------------------------

    def capture_preview(self) -> CaptureResult | None:
        """Take a single screenshot (for Web UI preview).

        Returns:
            :class:`CaptureResult` or ``None`` if capture unavailable.
        """
        with self._lock:
            cap = self._capture
        if cap is None:
            return None
        try:
            return cap.capture()
        except CaptureError as e:
            logger.warning("Preview capture failed: %s", e)
            return None

    # ------------------------------------------------------------------
    # Wi-Fi
    # ------------------------------------------------------------------

    def get_wifi_manager(self) -> WiFiManager | None:
        """Return the WiFiManager (lazy-init), or None if unavailable."""
        with self._lock:
            if self._wifi is not None:
                return self._wifi
        try:
            wm = WiFiManager()
            with self._lock:
                self._wifi = wm
            return wm
        except Exception as e:
            logger.warning("WiFiManager init failed: %s", e)
            return None

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def install_log_handler(self, logger_name: str = "") -> None:
        """Attach the ring-buffer log handler to a logger.

        Args:
            logger_name: Logger name (empty string = root logger).
        """
        target = logging.getLogger(logger_name)
        target.addHandler(self._log_handler)

    def remove_log_handler(self, logger_name: str = "") -> None:
        """Remove the ring-buffer handler from a logger."""
        target = logging.getLogger(logger_name)
        target.removeHandler(self._log_handler)

    def get_logs(self, count: int | None = None) -> list[str]:
        """Return recent log messages as formatted strings.

        Args:
            count: Number of recent entries (``None`` = all buffered).
        """
        records = self._log_handler.get_records(count)
        return [self._log_handler.format(r) for r in records]

    def clear_logs(self) -> None:
        """Discard all buffered log records."""
        self._log_handler.clear()

    def _on_log_record(self, record: logging.LogRecord) -> None:
        """Emit a LOG_MESSAGE event for each captured log record."""
        self._emit(AppEvent(
            type=AppEventType.LOG_MESSAGE,
            data={
                "level": record.levelname,
                "name": record.name,
                "message": self._log_handler.format(record),
            },
        ))

    # ------------------------------------------------------------------
    # System status
    # ------------------------------------------------------------------

    def get_status(self) -> dict[str, Any]:
        """Return a summary of system state for display.

        Includes config, module readiness, task status, and Wi-Fi info.
        """
        config = self.get_config()
        task = self.get_task_status()

        wifi_info: dict[str, Any] = {"available": False}
        wm = self.get_wifi_manager()
        if wm:
            wifi_info["available"] = True
            wifi_info["backend"] = wm.backend
            try:
                wifi_info["connected"] = wm.is_connected()
                wifi_info["ip"] = wm.get_ip_address()
                wifi_info["ssid"] = wm.get_current_network()
            except Exception:
                pass

        with self._lock:
            capture = self._capture
            capture_device = self._capture_device_name
            executor_device = self._executor_device_name

        # AirPlay client can change mid-session — resolve live
        if config.capture_source == "airplay" and capture is not None:
            live_name = getattr(capture, "connected_client", "")
            if live_name:
                capture_device = live_name

        return {
            "modules_ready": self.modules_ready,
            "capture_ready": self.capture_ready,
            "executor_ready": self.executor_ready,
            "capture_source": config.capture_source,
            "executor_transport": config.executor_transport,
            "capture_device": capture_device,
            "executor_device": executor_device,
            "llm_provider": config.llm.provider,
            "llm_model": config.llm.model,
            "task": task,
            "wifi": wifi_info,
        }
