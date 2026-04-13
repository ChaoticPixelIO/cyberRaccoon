"""M5 Web Server — FastAPI REST API + WebSocket for real-time events.

Provides the HTTP backend for the Alpine.js single-page application.

REST endpoints:
    GET  /api/config              — current configuration
    PUT  /api/config/{section}    — update a config section
    POST /api/task                — start a task
    DELETE /api/task              — abort the running task
    GET  /api/status              — system status
    GET  /api/capture/preview     — single screenshot (base64)
    POST /api/wifi/connect        — connect to Wi-Fi
    GET  /api/wifi/scan           — scan networks
    GET  /api/wifi/status         — Wi-Fi status
    GET  /api/logs                — recent log entries
    DELETE /api/logs              — clear log buffer

WebSocket:
    /ws — real-time event stream (task steps, logs, status changes)

Static files:
    GET / — serves index.html from ui/web/static/

Usage::

    from cyberraccoon.ui.app_controller import AppController
    from cyberraccoon.ui.web.server import create_app

    ctrl = AppController()
    ctrl.load_config()
    app = create_app(ctrl)

    # Run with: uvicorn cyberraccoon.ui.web.server:app
"""

from __future__ import annotations

import asyncio
import json
import logging
import queue
from contextlib import asynccontextmanager
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, AsyncGenerator

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from cyberraccoon.agent.skills import (
    SkillFormatError,
    SkillIncompleteError,
    SkillNotFoundError,
    delete_user_skill,
    get_skill_info,
    get_skill_source,
    list_skills,
    save_user_skill,
)
from cyberraccoon.ui.app_controller import AppController, AppEvent, AppEventType

logger = logging.getLogger("M5.web")

STATIC_DIR = Path(__file__).parent / "static"


# ---------------------------------------------------------------------------
# Pydantic models for request validation
# ---------------------------------------------------------------------------

class TaskRequest(BaseModel):
    goal: str


class WiFiConnectRequest(BaseModel):
    ssid: str
    password: str = ""


class SkillContentRequest(BaseModel):
    content: str


class ChatRequest(BaseModel):
    question: str


class RewriteRequest(BaseModel):
    """Body for POST /api/task/request-plan-rewrite (Phase 5, DISCUSS-03)."""

    request: str


class EditStepRequest(BaseModel):
    """Body for POST /api/task/edit-plan-step (Phase 5, DISCUSS-04)."""

    step_number: int
    new_goal: str


class DeleteStepRequest(BaseModel):
    """Body for POST /api/task/delete-plan-step (Phase 5, DISCUSS-04)."""

    step_number: int


# ---------------------------------------------------------------------------
# WebSocket connection manager
# ---------------------------------------------------------------------------

class ConnectionManager:
    """Manages active WebSocket connections and broadcasts events."""

    def __init__(self) -> None:
        self._connections: list[WebSocket] = []
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            self._connections.append(ws)
        logger.info("WebSocket connected (%d total)", len(self._connections))

    async def disconnect(self, ws: WebSocket) -> None:
        async with self._lock:
            try:
                self._connections.remove(ws)
            except ValueError:
                pass
        logger.info("WebSocket disconnected (%d total)", len(self._connections))

    async def broadcast(self, message: dict[str, Any]) -> None:
        """Send a JSON message to all connected clients."""
        async with self._lock:
            connections = list(self._connections)
        dead: list[WebSocket] = []
        text = json.dumps(message, default=str)
        for ws in connections:
            try:
                await ws.send_text(text)
            except Exception:
                dead.append(ws)
        # Clean up dead connections
        if dead:
            async with self._lock:
                for ws in dead:
                    try:
                        self._connections.remove(ws)
                    except ValueError:
                        pass

    @property
    def connection_count(self) -> int:
        return len(self._connections)


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(controller: AppController) -> FastAPI:
    """Create a FastAPI application wired to the given AppController.

    Args:
        controller: The shared AppController instance.

    Returns:
        Configured FastAPI app ready to run with uvicorn.
    """
    manager = ConnectionManager()

    # Bridge AppController events → WebSocket broadcast.
    # Uses a thread-safe ``queue.Queue`` since _event_bridge is called
    # from sync threads (VisionAgent worker, CLI, etc.). A background
    # async task polls this queue and broadcasts to WebSocket clients.
    _event_queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=500)

    def _event_bridge(event: AppEvent) -> None:
        """Forward AppController events into the thread-safe queue."""
        msg = {
            "event": event.type.value,
            "data": event.data,
            "timestamp": event.timestamp,
        }
        try:
            _event_queue.put_nowait(msg)
        except queue.Full:
            pass  # Drop oldest-style: caller is too fast

    controller.add_listener(_event_bridge)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        """Manage the event-drain background task."""
        async def _drain_events() -> None:
            while True:
                # Poll the thread-safe queue from the async loop
                try:
                    msg = _event_queue.get_nowait()
                    await manager.broadcast(msg)
                except queue.Empty:
                    await asyncio.sleep(0.05)

        task = asyncio.create_task(_drain_events())
        logger.info("Web server started")
        try:
            yield
        finally:
            task.cancel()
            controller.remove_listener(_event_bridge)
            logger.info("Web server stopped")

    app = FastAPI(title="CyberRaccoon", version="0.1.0", lifespan=lifespan)

    # Store references for route handlers
    app.state.controller = controller
    app.state.manager = manager

    # ---- Static files + index ----
    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        index_path = STATIC_DIR / "index.html"
        if index_path.exists():
            return HTMLResponse(index_path.read_text(encoding="utf-8"))
        return HTMLResponse("<h1>CyberRaccoon — static files not found</h1>")

    # ---- Config API ----

    @app.get("/api/config")
    async def get_config() -> JSONResponse:
        config = controller.get_config()
        data: dict[str, Any] = {
            "capture_source": config.capture_source,
            "executor_transport": config.executor_transport,
            "target_os": config.target_os,
        }
        sections = ["capture", "llm", "agent", "executor", "network", "ble"]
        for name in sections:
            sub = getattr(config, name)
            section_dict: dict[str, Any] = {}
            for f in fields(type(sub)):
                # LLM per-provider snapshots are backend-only state; hide
                # them from the API to avoid round-tripping masked secrets
                # back into the authoritative config.
                if name == "llm" and f.name == "providers":
                    continue
                val = getattr(sub, f.name)
                # Mask secrets in API response
                if f.name == "api_key" and val:
                    section_dict[f.name] = val[:4] + "..." if len(val) > 4 else "***"
                elif f.name == "wifi_password":
                    section_dict[f.name] = "***" if val else ""
                else:
                    section_dict[f.name] = val
            data[name] = section_dict
        return JSONResponse(data)

    @app.put("/api/config/{section}")
    async def update_config(section: str, body: dict[str, Any]) -> JSONResponse:
        try:
            config = controller.get_config()
            attr = getattr(config, section, None)
            # Top-level scalar field (e.g. capture_source, executor_transport)
            if attr is not None and isinstance(attr, str) and section in body:
                controller.update_config(**{section: body[section]})
            else:
                # Sub-section (e.g. llm, agent, network)
                updates = {f"{section}.{k}": v for k, v in body.items()}
                controller.update_config(**updates)
            return JSONResponse({"status": "ok"})
        except Exception as e:
            return JSONResponse({"status": "error", "message": str(e)}, status_code=400)

    # ---- Modules API (legacy — kept for backward compat) ----

    @app.post("/api/modules/init")
    async def init_modules() -> JSONResponse:
        try:
            controller.init_modules()
            return JSONResponse({"status": "ok"})
        except Exception as e:
            return JSONResponse({"status": "error", "message": str(e)}, status_code=400)

    @app.post("/api/modules/close")
    async def close_modules() -> JSONResponse:
        controller.close_modules()
        return JSONResponse({"status": "ok"})

    # ---- Individual connection API ----

    @app.post("/api/capture/connect")
    async def capture_connect() -> JSONResponse:
        try:
            # May block for AirPlay — run in thread to avoid blocking event loop
            test_frame = await asyncio.to_thread(controller.init_capture)
            data: dict[str, Any] = {"status": "ok"}
            if test_frame is not None:
                data["image"] = test_frame.base64_jpeg
                data["width"] = test_frame.width
                data["height"] = test_frame.height
            status = controller.get_status()
            data["device"] = status.get("capture_device", "")
            return JSONResponse(data)
        except Exception as e:
            return JSONResponse({"status": "error", "message": str(e)}, status_code=400)

    @app.post("/api/capture/disconnect")
    async def capture_disconnect() -> JSONResponse:
        controller.close_capture()
        return JSONResponse({"status": "ok"})

    @app.post("/api/executor/connect")
    async def executor_connect() -> JSONResponse:
        try:
            # May block for BT pairing — run in thread to avoid blocking event loop
            await asyncio.to_thread(controller.init_executor)
            status = controller.get_status()
            device = status.get("executor_device", "")
            return JSONResponse({"status": "ok", "device": device})
        except Exception as e:
            return JSONResponse({"status": "error", "message": str(e)}, status_code=400)

    @app.post("/api/executor/disconnect")
    async def executor_disconnect() -> JSONResponse:
        controller.close_executor()
        return JSONResponse({"status": "ok"})

    # ---- Task API ----

    @app.post("/api/task")
    async def start_task(req: TaskRequest) -> JSONResponse:
        try:
            controller.start_task(req.goal)
            return JSONResponse({"status": "ok"})
        except Exception as e:
            return JSONResponse({"status": "error", "message": str(e)}, status_code=400)

    @app.delete("/api/task")
    async def abort_task() -> JSONResponse:
        controller.abort_task()
        return JSONResponse({"status": "ok"})

    @app.post("/api/task/reset")
    async def reset_task() -> JSONResponse:
        """Force-clear stale task state when no task is actually running."""
        result = controller.force_reset_task()
        return JSONResponse({"status": result})

    @app.post("/api/task/approve-plan")
    async def approve_plan() -> JSONResponse:
        controller.approve_plan()
        return JSONResponse({"status": "ok"})

    @app.post("/api/task/reject-plan")
    async def reject_plan() -> JSONResponse:
        controller.reject_plan()
        return JSONResponse({"status": "ok"})

    @app.post("/api/task/resolve-escalation")
    async def resolve_escalation() -> JSONResponse:
        controller.resolve_escalation()
        return JSONResponse({"status": "ok"})

    # Phase 7 — Pause / Resume / Cancel (CRUISE-03, CRUISE-05)
    # State validation returns 409 for invalid transitions (review MEDIUM-3).

    @app.post("/api/task/pause")
    async def pause_task() -> JSONResponse:
        accepted = controller.pause_task()
        if not accepted:
            return JSONResponse(
                {"status": "error", "message": "No task is currently executing."},
                status_code=409,
            )
        return JSONResponse({"status": "ok"})

    @app.post("/api/task/resume")
    async def resume_task() -> JSONResponse:
        accepted = controller.resume_task()
        if not accepted:
            return JSONResponse(
                {"status": "error", "message": "No task is currently paused."},
                status_code=409,
            )
        return JSONResponse({"status": "ok"})

    @app.post("/api/task/cancel")
    async def cancel_paused_task() -> JSONResponse:
        controller.cancel_paused_task()
        return JSONResponse({"status": "ok"})

    @app.post("/api/task/chat-about-plan")
    async def chat_about_plan_endpoint(req: ChatRequest) -> JSONResponse:
        """Answer a user question about the pending plan.

        Delegates to AppController.chat_about_plan which uses the cached
        plan discussion state to ground the answer. Runs the LLM call in
        a worker thread (asyncio.to_thread) so the FastAPI event loop is
        never blocked — same pattern as /api/capture/connect.
        """
        if not (req.question or "").strip():
            return JSONResponse(
                {"status": "error", "message": "Question cannot be blank"},
                status_code=422,
            )
        # The controller call is sync and may block for 1-5s during the LLM call
        answer = await asyncio.to_thread(
            controller.chat_about_plan, req.question,
        )
        if answer is None:
            return JSONResponse(
                {
                    "status": "error",
                    "message": "No plan pending or chat call failed",
                },
                status_code=503,
            )
        return JSONResponse({"status": "ok", "answer": answer})

    # ---- Plan modification API (Phase 5, DISCUSS-03/04) ----

    @app.post("/api/task/request-plan-rewrite")
    async def request_plan_rewrite_endpoint(
        req: RewriteRequest,
    ) -> JSONResponse:
        """Ask the LLM to rewrite the cached plan (DISCUSS-03).

        The LLM response is a typed RewriteResult. On success the
        controller broadcasts ``plan_modification_proposed`` (for an
        actual rewrite) or ``plan_rewrite_no_change`` (for the escape
        hatch) via WebSocket -- this REST call only acknowledges receipt
        with the discriminator value so the frontend can distinguish
        fast failures from the WebSocket branch.
        """
        # [REVIEWS MEDIUM] Validate non-blank request at API boundary
        if not (req.request or "").strip():
            return JSONResponse(
                {"status": "error", "message": "Modification request cannot be blank"},
                status_code=422,
            )
        result = await asyncio.to_thread(
            controller.request_plan_rewrite, req.request,
        )
        if result is None:
            return JSONResponse(
                {
                    "status": "error",
                    "message": "No plan pending or rewrite call failed",
                },
                status_code=503,
            )
        return JSONResponse(
            {"status": "ok", "action": result.action},
        )

    @app.post("/api/task/accept-plan-rewrite")
    async def accept_plan_rewrite_endpoint() -> JSONResponse:
        """Commit the pending LLM rewrite to the live plan (DISCUSS-03)."""
        ok = await asyncio.to_thread(controller.accept_plan_rewrite)
        if not ok:
            return JSONResponse(
                {
                    "status": "error",
                    "message": "No pending rewrite to accept",
                },
                status_code=409,
            )
        return JSONResponse({"status": "ok"})

    @app.post("/api/task/discard-plan-rewrite")
    async def discard_plan_rewrite_endpoint() -> JSONResponse:
        """Drop the pending LLM rewrite without mutating the plan.

        Idempotent -- discarding a non-existent preview returns 200 with
        status=noop rather than 409. This matches the D-13 semantics:
        Discard is a safe "never mind" action, not an error path.
        """
        ok = await asyncio.to_thread(controller.discard_plan_rewrite)
        return JSONResponse(
            {"status": "ok" if ok else "noop"},
        )

    @app.post("/api/task/edit-plan-step")
    async def edit_plan_step_endpoint(
        req: EditStepRequest,
    ) -> JSONResponse:
        """Save a manual inline edit to a step's goal text (DISCUSS-04)."""
        # [REVIEWS MEDIUM] Validate at API boundary before controller dispatch
        if not (req.new_goal or "").strip():
            return JSONResponse(
                {"status": "error", "message": "Step goal cannot be blank"},
                status_code=422,
            )
        ok = await asyncio.to_thread(
            controller.edit_plan_step, req.step_number, req.new_goal,
        )
        if not ok:
            return JSONResponse(
                {
                    "status": "error",
                    "message": "No plan pending or step not found",
                },
                status_code=404,
            )
        return JSONResponse({"status": "ok"})

    @app.post("/api/task/add-plan-step")
    async def add_plan_step_endpoint() -> JSONResponse:
        """Append a blank step to the plan (DISCUSS-04)."""
        ok = await asyncio.to_thread(controller.add_plan_step)
        if not ok:
            return JSONResponse(
                {
                    "status": "error",
                    "message": "No plan pending or preview active",
                },
                status_code=409,
            )
        return JSONResponse({"status": "ok"})

    @app.post("/api/task/delete-plan-step")
    async def delete_plan_step_endpoint(
        req: DeleteStepRequest,
    ) -> JSONResponse:
        """Delete a step from the plan and renumber the remainder (DISCUSS-04)."""
        ok = await asyncio.to_thread(
            controller.delete_plan_step, req.step_number,
        )
        if not ok:
            return JSONResponse(
                {
                    "status": "error",
                    "message": "No plan pending or step not found",
                },
                status_code=404,
            )
        return JSONResponse({"status": "ok"})

    # ---- Status API ----

    @app.get("/api/status")
    async def get_status() -> JSONResponse:
        return JSONResponse(controller.get_status())

    # ---- Capture preview ----

    @app.get("/api/capture/preview")
    async def capture_preview() -> JSONResponse:
        result = controller.capture_preview()
        if result is None:
            return JSONResponse(
                {"status": "error", "message": "Capture not available"},
                status_code=503,
            )
        return JSONResponse({
            "image": result.base64_jpeg,
            "width": result.width,
            "height": result.height,
            "size_bytes": result.size_bytes,
        })

    # ---- Wi-Fi API ----

    @app.get("/api/wifi/scan")
    async def wifi_scan() -> JSONResponse:
        wm = controller.get_wifi_manager()
        if wm is None:
            return JSONResponse(
                {"status": "error", "message": "Wi-Fi not available"},
                status_code=503,
            )
        networks = wm.scan()
        return JSONResponse([
            {
                "ssid": n.ssid,
                "signal_strength": n.signal_strength,
                "security": n.security,
                "connected": n.connected,
            }
            for n in networks
        ])

    @app.post("/api/wifi/connect")
    async def wifi_connect(req: WiFiConnectRequest) -> JSONResponse:
        wm = controller.get_wifi_manager()
        if wm is None:
            return JSONResponse(
                {"status": "error", "message": "Wi-Fi not available"},
                status_code=503,
            )
        try:
            wm.connect(req.ssid, req.password)
            ip = wm.get_ip_address()
            return JSONResponse({"status": "ok", "ip": ip})
        except Exception as e:
            return JSONResponse({"status": "error", "message": str(e)}, status_code=400)

    @app.get("/api/wifi/status")
    async def wifi_status() -> JSONResponse:
        wm = controller.get_wifi_manager()
        if wm is None:
            return JSONResponse(
                {"status": "error", "message": "Wi-Fi not available"},
                status_code=503,
            )
        return JSONResponse({
            "connected": wm.is_connected(),
            "ssid": wm.get_current_network(),
            "ip": wm.get_ip_address(),
            "backend": wm.backend,
        })

    # ---- Logs API ----

    @app.get("/api/logs")
    async def get_logs(limit: int = 200) -> JSONResponse:
        entries = controller.get_logs(limit)
        return JSONResponse(entries)

    @app.delete("/api/logs")
    async def clear_logs() -> JSONResponse:
        controller.clear_logs()
        return JSONResponse({"status": "ok"})

    # ---- Skills API ----

    @app.get("/api/skills")
    async def get_skills() -> JSONResponse:
        names = list_skills()
        skills = []
        for name in names:
            entry: dict[str, Any] = {"name": name, "source": "unknown", "description": None}
            try:
                entry["source"] = get_skill_source(name)
            except (SkillNotFoundError, ValueError) as e:
                logger.warning("Could not determine source for skill %r: %s", name, e)
            try:
                info = get_skill_info(name)
                entry["description"] = info["description"]
            except SkillIncompleteError as e:
                entry["error"] = "missing_skill_md"
                logger.warning("Skill %r is incomplete: %s", name, e)
            except SkillFormatError as e:
                entry["error"] = "invalid_frontmatter"
                logger.warning("Skill %r has invalid frontmatter: %s", name, e)
            except (SkillNotFoundError, ValueError) as e:
                entry["error"] = "load_failed"
                logger.warning("Could not load skill %r: %s", name, e)
            skills.append(entry)
        return JSONResponse({"skills": skills})

    @app.get("/api/skills/{name}")
    async def get_skill(name: str) -> JSONResponse:
        try:
            info = get_skill_info(name)
            return JSONResponse(info)
        except SkillIncompleteError as e:
            return JSONResponse(
                {"status": "error", "message": str(e)},
                status_code=422,
            )
        except SkillFormatError as e:
            return JSONResponse(
                {"status": "error", "message": str(e)},
                status_code=422,
            )
        except SkillNotFoundError:
            return JSONResponse(
                {"status": "error", "message": f"Skill {name!r} not found"},
                status_code=404,
            )
        except ValueError as e:
            return JSONResponse(
                {"status": "error", "message": str(e)},
                status_code=400,
            )

    @app.put("/api/skills/{name}")
    async def put_skill(name: str, req: SkillContentRequest) -> JSONResponse:
        try:
            save_user_skill(name, req.content)
            return JSONResponse({"status": "ok"})
        except SkillFormatError as e:
            return JSONResponse(
                {"status": "error", "message": str(e)},
                status_code=400,
            )
        except ValueError as e:
            return JSONResponse(
                {"status": "error", "message": str(e)},
                status_code=400,
            )
        except OSError as e:
            logger.error("Failed to save skill %r: %s", name, e)
            return JSONResponse(
                {"status": "error", "message": f"Failed to write skill file: {e}"},
                status_code=500,
            )

    @app.delete("/api/skills/{name}")
    async def delete_skill(name: str) -> JSONResponse:
        try:
            deleted = delete_user_skill(name)
            if not deleted:
                return JSONResponse(
                    {"status": "error", "message": f"No user skill {name!r} to delete"},
                    status_code=400,
                )
            return JSONResponse({"status": "ok"})
        except ValueError as e:
            return JSONResponse(
                {"status": "error", "message": str(e)},
                status_code=400,
            )
        except OSError as e:
            logger.error("Failed to delete skill %r: %s", name, e)
            return JSONResponse(
                {"status": "error", "message": f"Failed to delete skill directory: {e}"},
                status_code=500,
            )

    # ---- WebSocket ----

    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket) -> None:
        await manager.connect(ws)
        try:
            while True:
                # Keep connection alive; handle client messages
                data = await ws.receive_text()
                try:
                    msg = json.loads(data)
                    action = msg.get("action")
                    if action == "ping":
                        await ws.send_text(json.dumps({"event": "pong"}))
                except json.JSONDecodeError:
                    pass
        except WebSocketDisconnect:
            pass
        finally:
            await manager.disconnect(ws)

    return app


# ---------------------------------------------------------------------------
# Module-level app (for `uvicorn cyberraccoon.ui.web.server:app`)
# ---------------------------------------------------------------------------


def _lazy_app() -> FastAPI:
    """Create the module-level app on first access.

    Only used when running standalone via ``uvicorn cyberraccoon.ui.web.server:app``.
    When launched from ``python -m cyberraccoon --web``, this is never called
    because ``__main__.py`` imports ``create_app`` directly.
    """
    ctrl = AppController()
    ctrl.load_config()
    ctrl.install_log_handler()
    return create_app(ctrl)


class _LazyApp:
    """Descriptor that defers app creation until uvicorn accesses it."""

    def __init__(self) -> None:
        self._app: FastAPI | None = None

    def __getattr__(self, name: str) -> Any:
        if self._app is None:
            self._app = _lazy_app()
        return getattr(self._app, name)

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if self._app is None:
            self._app = _lazy_app()
        await self._app(scope, receive, send)


app = _LazyApp()
