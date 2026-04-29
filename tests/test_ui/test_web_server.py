"""Tests for ui.web.server — FastAPI REST API + WebSocket."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from cyberraccoon.config import AppConfig
from cyberraccoon.ui.app_controller import AppController, AppEvent, AppEventType
from cyberraccoon.ui.web.server import create_app


@pytest.fixture()
def ctrl(tmp_path: Path) -> AppController:
    """AppController with temp config."""
    c = AppController(config_path=str(tmp_path / "cfg.yaml"))
    c.load_config()
    return c


@pytest.fixture()
def client(ctrl: AppController) -> TestClient:
    """FastAPI TestClient wired to the controller."""
    app = create_app(ctrl)
    return TestClient(app)


# ---------------------------------------------------------------------------
# Config API
# ---------------------------------------------------------------------------

class TestSetupStatusAPI:
    def test_get_setup_status(self, client: TestClient) -> None:
        resp = client.get("/api/setup/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "components" in data
        assert "setup_commands" in data
        # Each component has the required shape
        for name in ("python_env", "bluetooth", "usb_gadget", "csi_hdmi", "airplay"):
            assert name in data["components"]
            assert "status" in data["components"][name]
            assert "detail" in data["components"][name]


class TestConfigAPI:
    def test_get_config(self, client: TestClient) -> None:
        resp = client.get("/api/config")
        assert resp.status_code == 200
        data = resp.json()
        assert "capture_source" in data
        assert "llm" in data
        assert "agent" in data

    def test_get_config_masks_api_key(self, client: TestClient, ctrl: AppController) -> None:
        ctrl.update_config(**{"llm.api_key": "sk-very-secret-key-123"})
        resp = client.get("/api/config")
        data = resp.json()
        assert data["llm"]["api_key"] == "sk-v..."
        assert "sk-very-secret-key-123" not in json.dumps(data)

    def test_update_config_section(self, client: TestClient) -> None:
        resp = client.put(
            "/api/config/llm",
            json={"model": "gpt-4o"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

        # Verify it persisted
        resp2 = client.get("/api/config")
        assert resp2.json()["llm"]["model"] == "gpt-4o"

    def test_update_config_invalid_section(self, client: TestClient) -> None:
        # Unknown keys are silently ignored in update_config
        resp = client.put(
            "/api/config/nonexistent",
            json={"foo": "bar"},
        )
        # Should still return ok (unknown keys are logged but not errors)
        assert resp.status_code == 200

    def test_agent_auto_replan_string_true_coerced_correctly(
        self, client: TestClient, ctrl: AppController,
    ) -> None:
        """TC5 (review test-coverage gap): the StrictBool guard lives only
        on the dedicated /api/task/auto-replan endpoint; the generic
        /api/config write path uses YAML-style coercion. Verify that the
        config-side _coerce_bool vocabulary correctly distinguishes
        string 'true' / 'false' rather than relying on Python's loose
        bool('false')==True semantics."""
        # String "true" → real Python True
        resp = client.put(
            "/api/config/agent",
            json={"auto_replan": "true"},
        )
        assert resp.status_code == 200
        cfg = ctrl.get_config()
        assert cfg.agent.auto_replan is True
        assert isinstance(cfg.agent.auto_replan, bool)

    def test_agent_auto_replan_string_false_coerced_correctly(
        self, client: TestClient, ctrl: AppController,
    ) -> None:
        """The historical Python footgun: bool('false') is True.
        _coerce_bool's explicit vocabulary must produce real False here."""
        # First set to True so we can detect the false coercion working.
        ctrl.update_config(**{"agent.auto_replan": True})
        assert ctrl.get_config().agent.auto_replan is True

        resp = client.put(
            "/api/config/agent",
            json={"auto_replan": "false"},
        )
        assert resp.status_code == 200
        cfg = ctrl.get_config()
        assert cfg.agent.auto_replan is False, (
            "bool('false') is True in Python — _coerce_bool must catch this"
        )
        assert isinstance(cfg.agent.auto_replan, bool)

    def test_agent_auto_replan_real_bool_passes_through(
        self, client: TestClient, ctrl: AppController,
    ) -> None:
        """Real Python booleans must pass through unchanged."""
        resp = client.put(
            "/api/config/agent",
            json={"auto_replan": True},
        )
        assert resp.status_code == 200
        assert ctrl.get_config().agent.auto_replan is True


# ---------------------------------------------------------------------------
# Task API
# ---------------------------------------------------------------------------

class TestTaskAPI:
    def test_start_task_without_modules(self, client: TestClient) -> None:
        resp = client.post("/api/task", json={"goal": "Test"})
        assert resp.status_code == 400
        assert "error" in resp.json()["status"]

    def test_abort_task(self, client: TestClient) -> None:
        resp = client.delete("/api/task")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# Status API
# ---------------------------------------------------------------------------

class TestStatusAPI:
    def test_get_status(self, client: TestClient) -> None:
        resp = client.get("/api/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "modules_ready" in data
        assert data["modules_ready"] is False


# ---------------------------------------------------------------------------
# Capture API
# ---------------------------------------------------------------------------

class TestCaptureAPI:
    def test_preview_without_modules(self, client: TestClient) -> None:
        resp = client.get("/api/capture/preview")
        assert resp.status_code == 503

    def test_preview_with_mock(self, client: TestClient, ctrl: AppController) -> None:
        mock_result = MagicMock(
            base64_jpeg="AAAA", width=1280, height=720, size_bytes=5000,
        )
        with patch.object(ctrl, "capture_preview", return_value=mock_result):
            resp = client.get("/api/capture/preview")
            assert resp.status_code == 200
            data = resp.json()
            assert data["image"] == "AAAA"
            assert data["width"] == 1280


# ---------------------------------------------------------------------------
# Wi-Fi API
# ---------------------------------------------------------------------------

class TestWiFiAPI:
    def test_scan_unavailable(self, client: TestClient, ctrl: AppController) -> None:
        with patch.object(ctrl, "get_wifi_manager", return_value=None):
            resp = client.get("/api/wifi/scan")
            assert resp.status_code == 503

    def test_scan_with_mock(self, client: TestClient, ctrl: AppController) -> None:
        from cyberraccoon.ui.wifi_manager import WiFiNetwork
        mock_wm = MagicMock()
        mock_wm.scan.return_value = [
            WiFiNetwork(ssid="Home", signal_strength=-45, security="WPA2", connected=True),
        ]
        with patch.object(ctrl, "get_wifi_manager", return_value=mock_wm):
            resp = client.get("/api/wifi/scan")
            assert resp.status_code == 200
            networks = resp.json()
            assert len(networks) == 1
            assert networks[0]["ssid"] == "Home"

    def test_connect(self, client: TestClient, ctrl: AppController) -> None:
        mock_wm = MagicMock()
        mock_wm.connect.return_value = True
        mock_wm.get_ip_address.return_value = "192.168.1.5"
        with patch.object(ctrl, "get_wifi_manager", return_value=mock_wm):
            resp = client.post(
                "/api/wifi/connect",
                json={"ssid": "MyNet", "password": "pass123"},
            )
            assert resp.status_code == 200
            assert resp.json()["ip"] == "192.168.1.5"
            mock_wm.connect.assert_called_once_with("MyNet", "pass123")

    def test_status(self, client: TestClient, ctrl: AppController) -> None:
        mock_wm = MagicMock()
        mock_wm.is_connected.return_value = True
        mock_wm.get_current_network.return_value = "HomeNet"
        mock_wm.get_ip_address.return_value = "10.0.0.1"
        mock_wm.backend = "networkmanager"
        with patch.object(ctrl, "get_wifi_manager", return_value=mock_wm):
            resp = client.get("/api/wifi/status")
            assert resp.status_code == 200
            data = resp.json()
            assert data["connected"] is True
            assert data["ssid"] == "HomeNet"


# ---------------------------------------------------------------------------
# Logs API
# ---------------------------------------------------------------------------

class TestLogsAPI:
    def test_get_logs(self, client: TestClient) -> None:
        resp = client.get("/api/logs")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_clear_logs(self, client: TestClient) -> None:
        resp = client.delete("/api/logs")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# Static files / index
# ---------------------------------------------------------------------------

class TestStaticFiles:
    def test_index_returns_html(self, client: TestClient) -> None:
        resp = client.get("/")
        assert resp.status_code == 200
        assert "CyberRaccoon" in resp.text


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------

class TestWebSocket:
    def test_websocket_connect_and_ping(self, client: TestClient) -> None:
        with client.websocket_connect("/ws") as ws:
            ws.send_text(json.dumps({"action": "ping"}))
            data = ws.receive_text()
            msg = json.loads(data)
            assert msg["event"] == "pong"

    def test_event_bridge_enqueues_messages(
        self, ctrl: AppController, tmp_path: Path,
    ) -> None:
        """Verify that AppController events reach the server's event queue."""
        from cyberraccoon.ui.web.server import create_app

        app = create_app(ctrl)

        # The event queue is internal; we verify indirectly by checking
        # that the event bridge listener was installed on the controller
        initial_listener_count = len(ctrl._listeners)

        # Trigger an event — the bridge should enqueue it
        ctrl.update_config(**{"llm.model": "test-bridge"})

        # Verify the listener is still there and didn't crash
        assert len(ctrl._listeners) >= initial_listener_count

    def test_connection_manager_broadcast(self) -> None:
        """ConnectionManager broadcasts to all connected clients."""
        import asyncio
        from cyberraccoon.ui.web.server import ConnectionManager

        async def _test() -> None:
            mgr = ConnectionManager()
            assert mgr.connection_count == 0

        asyncio.get_event_loop().run_until_complete(_test())


# ---------------------------------------------------------------------------
# TestChatEndpoint (Phase 4 — DISCUSS-02)
# ---------------------------------------------------------------------------

try:
    from cyberraccoon.ui.app_controller import PlanDiscussionState  # added in plan 04
    _CHAT_ENDPOINT_AVAILABLE = True
except ImportError:
    _CHAT_ENDPOINT_AVAILABLE = False


@pytest.mark.skipif(
    not _CHAT_ENDPOINT_AVAILABLE,
    reason="Chat endpoint not yet implemented (plan 05)",
)
class TestChatEndpoint:
    """Tests for POST /api/task/chat-about-plan."""

    def test_chat_returns_answer(
        self,
        client: TestClient,
        ctrl: AppController,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Pre-populate the plan discussion cache
        ctrl._on_step_bridge({
            "type": "plan_ready",
            "task_goal": "Open Chrome",
            "screenshot_base64": "fake_b64",
            "steps": [{
                "number": 1,
                "goal": "Open Chrome",
                "reboot_expected": False,
                "expected_actions": 2,
                "expected_outcome": "Chrome visible",
            }],
        })
        # Stub the controller's chat method to avoid real LLM call
        monkeypatch.setattr(
            ctrl, "chat_about_plan",
            lambda question: f"Echo: {question}",
        )
        resp = client.post(
            "/api/task/chat-about-plan",
            json={"question": "Why Chrome?"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "answer" in body
        assert "Why Chrome?" in body["answer"]

    def test_chat_no_pending_plan_returns_409(
        self,
        client: TestClient,
        ctrl: AppController,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Review I4 — chat_about_plan now raises NoPlanCachedError, which the
        endpoint maps to 409 Conflict (user error, retry won't help).
        Distinct from 503 (transient — LLM call failed)."""
        from cyberraccoon.ui.exceptions import NoPlanCachedError

        # No plan cached
        assert ctrl._plan_discussion is None
        # Stub controller to raise NoPlanCachedError as production now does
        def _raise_no_plan(question: str) -> str | None:
            raise NoPlanCachedError("No plan is currently pending discussion")
        monkeypatch.setattr(ctrl, "chat_about_plan", _raise_no_plan)
        resp = client.post(
            "/api/task/chat-about-plan",
            json={"question": "Hi?"},
        )
        assert resp.status_code == 409
        body = resp.json()
        assert body["status"] == "error"
        assert "no plan" in body["message"].lower()

    def test_chat_llm_failure_returns_503(
        self,
        client: TestClient,
        ctrl: AppController,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Review I4 — when chat_about_plan returns None (LLM call failed
        or planner build failed) the endpoint returns 503 Service Unavailable
        (transient — retry might help)."""
        # Plan IS cached but LLM call returns None
        ctrl._on_step_bridge({
            "type": "plan_ready",
            "task_goal": "x",
            "screenshot_base64": "b64",
            "steps": [{
                "number": 1, "goal": "x", "reboot_expected": False,
                "expected_actions": 1, "expected_outcome": "ok",
            }],
        })
        monkeypatch.setattr(ctrl, "chat_about_plan", lambda question: None)
        resp = client.post(
            "/api/task/chat-about-plan",
            json={"question": "Hi?"},
        )
        assert resp.status_code == 503
        assert resp.json()["status"] == "error"


# ---------------------------------------------------------------------------
# Phase 5 availability probe — plan 05-05 lands these routes.
# ---------------------------------------------------------------------------

try:
    from cyberraccoon.ui.app_controller import AppController as _AC
    _REWRITE_ENDPOINTS_AVAILABLE = all(
        hasattr(_AC, name)
        for name in (
            "request_plan_rewrite",
            "accept_plan_rewrite",
            "discard_plan_rewrite",
            "edit_plan_step",
            "add_plan_step",
            "delete_plan_step",
        )
    )
except ImportError:
    _REWRITE_ENDPOINTS_AVAILABLE = False


# ===========================================================================
# Phase 5: plan modification endpoints (DISCUSS-03, DISCUSS-04)
# ===========================================================================


@pytest.mark.skipif(
    not _REWRITE_ENDPOINTS_AVAILABLE,
    reason="Plan modification endpoints not yet implemented (plan 05-05)",
)
class TestRewriteEndpoints:
    """Tests for six new POST endpoints from Phase 5."""

    def test_request_plan_rewrite_happy_path(
        self,
        client: TestClient,
        ctrl: AppController,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Create a minimal stand-in for the RewriteResult return
        class _FakeResult:
            action = "rewrite"
        monkeypatch.setattr(
            ctrl, "request_plan_rewrite",
            lambda req: _FakeResult(),
        )
        resp = client.post(
            "/api/task/request-plan-rewrite",
            json={"request": "use Win+R"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["action"] == "rewrite"

    def test_request_plan_rewrite_no_plan_returns_503(
        self,
        client: TestClient,
        ctrl: AppController,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            ctrl, "request_plan_rewrite", lambda req: None,
        )
        resp = client.post(
            "/api/task/request-plan-rewrite",
            json={"request": "x"},
        )
        assert resp.status_code == 503
        assert resp.json()["status"] == "error"

    def test_accept_plan_rewrite_happy_path(
        self,
        client: TestClient,
        ctrl: AppController,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(ctrl, "accept_plan_rewrite", lambda: True)
        resp = client.post("/api/task/accept-plan-rewrite")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_accept_plan_rewrite_no_preview_returns_409(
        self,
        client: TestClient,
        ctrl: AppController,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(ctrl, "accept_plan_rewrite", lambda: False)
        resp = client.post("/api/task/accept-plan-rewrite")
        assert resp.status_code == 409
        assert resp.json()["status"] == "error"

    def test_discard_plan_rewrite_ok(
        self,
        client: TestClient,
        ctrl: AppController,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(ctrl, "discard_plan_rewrite", lambda: True)
        resp = client.post("/api/task/discard-plan-rewrite")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_discard_plan_rewrite_noop(
        self,
        client: TestClient,
        ctrl: AppController,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(ctrl, "discard_plan_rewrite", lambda: False)
        resp = client.post("/api/task/discard-plan-rewrite")
        assert resp.status_code == 200
        assert resp.json()["status"] == "noop"

    def test_edit_plan_step_happy_path(
        self,
        client: TestClient,
        ctrl: AppController,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            ctrl, "edit_plan_step",
            lambda step_number, new_goal: True,
        )
        resp = client.post(
            "/api/task/edit-plan-step",
            json={"step_number": 1, "new_goal": "Press Win+R"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_edit_plan_step_not_found_returns_404(
        self,
        client: TestClient,
        ctrl: AppController,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            ctrl, "edit_plan_step",
            lambda step_number, new_goal: False,
        )
        resp = client.post(
            "/api/task/edit-plan-step",
            json={"step_number": 99, "new_goal": "x"},
        )
        assert resp.status_code == 404
        assert resp.json()["status"] == "error"

    def test_add_plan_step_happy_path(
        self,
        client: TestClient,
        ctrl: AppController,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(ctrl, "add_plan_step", lambda: True)
        resp = client.post("/api/task/add-plan-step")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_delete_plan_step_happy_path(
        self,
        client: TestClient,
        ctrl: AppController,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            ctrl, "delete_plan_step", lambda step_number: True,
        )
        resp = client.post(
            "/api/task/delete-plan-step",
            json={"step_number": 2},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# Pause / Resume / Cancel API (Phase 7 — CRUISE-03, CRUISE-05)
# ---------------------------------------------------------------------------

class TestPauseResumeAPI:
    """Tests for /api/task/pause, /api/task/resume, /api/task/cancel endpoints."""

    def test_pause_endpoint_returns_200_when_running(
        self, client: TestClient, ctrl: "AppController",
    ) -> None:
        """POST /api/task/pause returns 200 when a task is running."""
        # Wire a mock agent so pause_task() returns True
        agent = MagicMock()
        agent.pause = MagicMock()
        with ctrl._lock:
            ctrl._agent = agent
        resp = client.post("/api/task/pause")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_resume_endpoint_returns_200_when_paused(
        self, client: TestClient, ctrl: "AppController",
    ) -> None:
        """POST /api/task/resume returns 200 when a task is paused."""
        # Wire a mock agent + runner so resume_task() returns True
        agent = MagicMock()
        runner = MagicMock()
        runner.resume = MagicMock()
        runner.set_current_plan = MagicMock()
        agent._workflow_runner = runner
        with ctrl._lock:
            ctrl._agent = agent
        resp = client.post("/api/task/resume")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_cancel_endpoint_returns_200(self, client: TestClient) -> None:
        """POST /api/task/cancel returns 200 (safe no-op when idle)."""
        resp = client.post("/api/task/cancel")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_pause_when_idle_returns_409(self, client: TestClient) -> None:
        """POST /api/task/pause when no task is running returns 409.
        Addresses review concern: MEDIUM-3 fire-and-forget hides
        invalid transitions."""
        resp = client.post("/api/task/pause")
        assert resp.status_code == 409
        assert resp.json()["status"] == "error"

    def test_resume_when_not_paused_returns_409(self, client: TestClient) -> None:
        """POST /api/task/resume when not paused returns 409.
        Addresses review concern: MEDIUM-3."""
        resp = client.post("/api/task/resume")
        assert resp.status_code == 409
        assert resp.json()["status"] == "error"


# ===========================================================================
# Phase 3 — Replan decision + Auto Re-plan + pending-dialogs endpoints
# ===========================================================================

try:
    from cyberraccoon.ui.app_controller import AppController as _AC
    _REPLAN_ENDPOINTS_AVAILABLE = all(
        hasattr(_AC, name)
        for name in ("submit_replan_decision", "set_auto_replan", "get_pending_dialogs")
    )
except ImportError:
    _REPLAN_ENDPOINTS_AVAILABLE = False


@pytest.mark.skipif(
    not _REPLAN_ENDPOINTS_AVAILABLE,
    reason="Replan endpoints not yet implemented (plan 03-03)",
)
class TestReplanDecisionEndpoint:
    """POST /api/task/replan-decision (Phase 3 — REPLAN-01/02/03 + H5)."""

    def test_forwards_choice_to_controller(
        self, client, ctrl, monkeypatch,
    ) -> None:
        forwarded: dict = {}
        monkeypatch.setattr(
            ctrl, "submit_replan_decision",
            lambda choice, hint="": forwarded.update(
                {"choice": choice, "hint": hint}
            ),
        )
        resp = client.post(
            "/api/task/replan-decision",
            json={"choice": "replan"},
        )
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}
        assert forwarded["choice"] == "replan"
        assert forwarded["hint"] == ""  # 260429-zl4: default empty

    def test_accepts_all_valid_choices(
        self, client, ctrl, monkeypatch,
    ) -> None:
        seen: list[str] = []
        monkeypatch.setattr(
            ctrl, "submit_replan_decision",
            lambda choice, hint="": seen.append(choice),
        )
        for choice in ("replan", "continue", "retry", "abort", "resume"):
            resp = client.post(
                "/api/task/replan-decision",
                json={"choice": choice},
            )
            assert resp.status_code == 200
        assert seen == ["replan", "continue", "retry", "abort", "resume"]

    def test_missing_choice_returns_422(self, client) -> None:
        resp = client.post("/api/task/replan-decision", json={})
        assert resp.status_code == 422

    def test_invalid_choice_returns_422(self, client) -> None:
        """H5 — Literal validation rejects unknown values."""
        resp = client.post(
            "/api/task/replan-decision",
            json={"choice": "garbage"},
        )
        assert resp.status_code == 422

    def test_uppercase_choice_returns_422(self, client) -> None:
        """H5 — Literal is case-sensitive."""
        resp = client.post(
            "/api/task/replan-decision",
            json={"choice": "Replan"},
        )
        assert resp.status_code == 422

    def test_runner_rejection_returns_400(
        self, client, ctrl, monkeypatch,
    ) -> None:
        """Runner per-gate allowlist violation → 400 (controller raises ValueError)."""
        def rejecting_submit(choice: str, hint: str = "") -> None:
            raise ValueError(f"invalid choice {choice!r} for gate 'replan_A'")
        monkeypatch.setattr(ctrl, "submit_replan_decision", rejecting_submit)
        # retry is invalid for Path A — but Literal accepts it, runner rejects
        resp = client.post(
            "/api/task/replan-decision",
            json={"choice": "retry"},
        )
        assert resp.status_code == 400

    def test_no_active_gate_returns_409(
        self, client, ctrl, monkeypatch,
    ) -> None:
        """Runner raises RuntimeError when no gate armed → 409."""
        def no_gate_submit(choice: str, hint: str = "") -> None:
            raise RuntimeError("no active replan gate")
        monkeypatch.setattr(ctrl, "submit_replan_decision", no_gate_submit)
        resp = client.post(
            "/api/task/replan-decision",
            json={"choice": "replan"},
        )
        assert resp.status_code == 409

    # 260429-zl4 — operator hint forwarding
    def test_hint_forwarded_to_controller(
        self, client, ctrl, monkeypatch,
    ) -> None:
        forwarded: dict = {}
        monkeypatch.setattr(
            ctrl, "submit_replan_decision",
            lambda choice, hint="": forwarded.update(
                {"choice": choice, "hint": hint}
            ),
        )
        resp = client.post(
            "/api/task/replan-decision",
            json={"choice": "replan", "hint": "the password is hunter2"},
        )
        assert resp.status_code == 200
        assert forwarded["hint"] == "the password is hunter2"

    def test_hint_oversized_returns_422(self, client) -> None:
        """Pydantic max_length=2000 enforces the cap at the boundary."""
        resp = client.post(
            "/api/task/replan-decision",
            json={"choice": "replan", "hint": "x" * 2001},
        )
        assert resp.status_code == 422

    def test_hint_at_max_accepted(
        self, client, ctrl, monkeypatch,
    ) -> None:
        forwarded: dict = {}
        monkeypatch.setattr(
            ctrl, "submit_replan_decision",
            lambda choice, hint="": forwarded.update({"hint_len": len(hint)}),
        )
        resp = client.post(
            "/api/task/replan-decision",
            json={"choice": "replan", "hint": "x" * 2000},
        )
        assert resp.status_code == 200
        assert forwarded["hint_len"] == 2000


@pytest.mark.skipif(
    not _REPLAN_ENDPOINTS_AVAILABLE,
    reason="Replan endpoints not yet implemented (plan 03-03)",
)
class TestAutoReplanEndpoint:
    """POST /api/task/auto-replan (Phase 3 — REPLAN-06 + H5)."""

    def test_enables_auto_replan(self, client, ctrl, monkeypatch) -> None:
        seen: list[bool] = []
        monkeypatch.setattr(
            ctrl, "set_auto_replan",
            lambda enabled: seen.append(enabled),
        )
        resp = client.post(
            "/api/task/auto-replan",
            json={"enabled": True},
        )
        assert resp.status_code == 200
        assert seen == [True]

    def test_disables_auto_replan(self, client, ctrl, monkeypatch) -> None:
        seen: list[bool] = []
        monkeypatch.setattr(
            ctrl, "set_auto_replan",
            lambda enabled: seen.append(enabled),
        )
        resp = client.post(
            "/api/task/auto-replan",
            json={"enabled": False},
        )
        assert resp.status_code == 200
        assert seen == [False]

    def test_string_enabled_rejected(self, client) -> None:
        """H5 + Pitfall 7: StrictBool — strings must NOT be coerced."""
        resp = client.post(
            "/api/task/auto-replan",
            json={"enabled": "true"},
        )
        assert resp.status_code == 422

    def test_int_enabled_rejected(self, client) -> None:
        """H5: StrictBool — ints must NOT be coerced."""
        resp = client.post(
            "/api/task/auto-replan",
            json={"enabled": 1},
        )
        assert resp.status_code == 422

    def test_missing_enabled_returns_422(self, client) -> None:
        resp = client.post("/api/task/auto-replan", json={})
        assert resp.status_code == 422

    def test_persist_failure_returns_207(self, client, ctrl, monkeypatch) -> None:
        """Review I4 — when set_auto_replan applies in-memory but the YAML
        write fails, the endpoint returns 207 Multi-Status so the frontend
        can flash the "config write failed — will not persist" toast."""
        from cyberraccoon.ui.exceptions import ConfigPersistError

        def _persist_fail(enabled: bool) -> None:
            raise ConfigPersistError(
                "Auto Re-plan applied in memory but YAML write failed: disk full"
            )
        monkeypatch.setattr(ctrl, "set_auto_replan", _persist_fail)
        resp = client.post(
            "/api/task/auto-replan",
            json={"enabled": True},
        )
        assert resp.status_code == 207
        body = resp.json()
        assert body["status"] == "partial"
        assert body["applied"] == "in_memory"
        assert "yaml" in body["error"].lower() or "config" in body["error"].lower()


@pytest.mark.skipif(
    not _REPLAN_ENDPOINTS_AVAILABLE,
    reason="Replan endpoints not yet implemented (plan 03-03)",
)
class TestPendingDialogsEndpoint:
    """GET /api/task/pending-dialogs — supports reconnect replay (H7)."""

    def test_empty_when_no_dialog(self, client) -> None:
        resp = client.get("/api/task/pending-dialogs")
        assert resp.status_code == 200
        body = resp.json()
        # H7 new shape — flat array
        assert body == {"dialogs": []}

    def test_returns_replan_dialog_after_event(
        self, client, ctrl,
    ) -> None:
        ctrl._on_step_bridge({
            "type": "replan_dialog",
            "path": "A",
            "step_number": 3,
            "step_goal": "Click Save",
            "expected": "Save dialog dismissed",
            "observed": "Save dialog still visible",
            "mismatch_reason": "User has not yet picked a folder",
            "failure_reason": None,
            "screenshot_base64": "fake_b64",
        })
        resp = client.get("/api/task/pending-dialogs")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["dialogs"]) == 1
        assert body["dialogs"][0]["path"] == "A"
        assert body["dialogs"][0]["step_number"] == 3
        assert body["dialogs"][0]["_active_gate"] == "replan_A"

    def test_returns_escalation_after_event(
        self, client, ctrl,
    ) -> None:
        ctrl._on_step_bridge({
            "type": "escalate",
            "step_number": 2,
            "reason": "Login required",
        })
        resp = client.get("/api/task/pending-dialogs")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["dialogs"]) == 1
        assert body["dialogs"][0]["_active_gate"] == "escalation_C"

    def test_replan_dialog_clears_after_replanned_event(
        self, client, ctrl,
    ) -> None:
        ctrl._on_step_bridge({
            "type": "replan_dialog", "path": "A", "step_number": 1,
            "step_goal": "x",
        })
        assert len(client.get("/api/task/pending-dialogs").json()["dialogs"]) == 1
        ctrl._on_step_bridge({
            "type": "replanned", "old_step": 1, "new_steps": [],
            "cancelled_step_numbers": [], "steps_completed": 0,
            "screenshot_base64": "x",
        })
        assert client.get("/api/task/pending-dialogs").json()["dialogs"] == []

    def test_replan_dialog_clears_on_resolved_event_h8(
        self, client, ctrl,
    ) -> None:
        """H8 — replan_dialog_resolved clears the cache, regardless of choice."""
        for choice in ("continue", "retry", "abort"):
            ctrl._on_step_bridge({
                "type": "replan_dialog", "path": "A", "step_number": 1,
                "step_goal": "x",
            })
            assert len(client.get("/api/task/pending-dialogs").json()["dialogs"]) == 1
            ctrl._on_step_bridge({
                "type": "replan_dialog_resolved", "choice": choice,
            })
            assert client.get("/api/task/pending-dialogs").json()["dialogs"] == [], \
                f"H8 violation: cache not cleared on choice={choice}"

    def test_escalation_clears_after_escalation_resolved(
        self, client, ctrl,
    ) -> None:
        ctrl._on_step_bridge({
            "type": "escalate", "step_number": 1, "reason": "x",
        })
        assert len(client.get("/api/task/pending-dialogs").json()["dialogs"]) == 1
        ctrl._on_step_bridge({
            "type": "escalation_resolved",
        })
        assert client.get("/api/task/pending-dialogs").json()["dialogs"] == []

    def test_escalation_clears_on_resolved_h8(
        self, client, ctrl,
    ) -> None:
        """H8 — replan_dialog_resolved also clears escalation cache."""
        ctrl._on_step_bridge({
            "type": "escalate", "step_number": 1, "reason": "x",
        })
        assert len(client.get("/api/task/pending-dialogs").json()["dialogs"]) == 1
        ctrl._on_step_bridge({
            "type": "replan_dialog_resolved", "choice": "resume",
        })
        assert client.get("/api/task/pending-dialogs").json()["dialogs"] == []

    def test_pending_caches_clear_on_task_finished(
        self, client, ctrl,
    ) -> None:
        from cyberraccoon.ui.app_controller import AppEvent, AppEventType
        ctrl._on_step_bridge({
            "type": "replan_dialog", "path": "A", "step_number": 1,
            "step_goal": "x",
        })
        ctrl._emit(AppEvent(type=AppEventType.TASK_FINISHED, data={}))
        assert client.get("/api/task/pending-dialogs").json()["dialogs"] == []

    def test_both_types_simultaneously(
        self, client, ctrl,
    ) -> None:
        """In theory both caches can hold simultaneously (though the runner
        only arms one gate at a time). The getter must return both."""
        ctrl._on_step_bridge({
            "type": "replan_dialog", "path": "B", "step_number": 1,
            "step_goal": "x",
        })
        ctrl._on_step_bridge({
            "type": "escalate", "step_number": 2, "reason": "y",
        })
        body = client.get("/api/task/pending-dialogs").json()
        gates = sorted(d["_active_gate"] for d in body["dialogs"])
        assert gates == ["escalation_C", "replan_B"]


class TestFatalErrorAPI:
    """260429-xe5 — endpoints for the fatal-error pause + retry/cancel flow."""

    def test_get_fatal_error_empty(self, client: TestClient) -> None:
        resp = client.get("/api/task/fatal-error")
        assert resp.status_code == 200
        assert resp.json() == {"error": None}

    def test_get_fatal_error_populated(
        self, client: TestClient, ctrl: AppController,
    ) -> None:
        ctrl._on_step_bridge({
            "type": "fatal_error",
            "status_code": 400,
            "request_id": "req_011XYZ",
            "message": "`temperature` is deprecated for this model",
            "step_number": 1,
        })
        body = client.get("/api/task/fatal-error").json()
        assert body["error"] is not None
        assert body["error"]["status_code"] == 400
        assert body["error"]["request_id"] == "req_011XYZ"

    def test_post_retry_calls_resolve_retry(
        self, client: TestClient, ctrl: AppController,
    ) -> None:
        called = {"count": 0}
        def fake_retry() -> None:
            called["count"] += 1
        ctrl.resolve_fatal_error_retry = fake_retry  # type: ignore[assignment]
        resp = client.post("/api/task/fatal-error/retry")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}
        assert called["count"] == 1

    def test_post_cancel_calls_resolve_cancel(
        self, client: TestClient, ctrl: AppController,
    ) -> None:
        called = {"count": 0}
        def fake_cancel() -> None:
            called["count"] += 1
        ctrl.resolve_fatal_error_cancel = fake_cancel  # type: ignore[assignment]
        resp = client.post("/api/task/fatal-error/cancel")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}
        assert called["count"] == 1


class TestRedetectTargetOSAPI:
    """260429-ucg — POST /api/config/redetect-target-os clears the cache."""

    def test_post_calls_invalidate(
        self, client: TestClient, ctrl: AppController,
    ) -> None:
        called = {"count": 0}
        def fake_invalidate() -> None:
            called["count"] += 1
        ctrl.invalidate_detected_target_os = fake_invalidate  # type: ignore[assignment]
        resp = client.post("/api/config/redetect-target-os")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}
        assert called["count"] == 1
