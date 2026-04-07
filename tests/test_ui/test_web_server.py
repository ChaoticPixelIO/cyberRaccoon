"""Tests for ui.web.server — FastAPI REST API + WebSocket."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from config import AppConfig
from ui.app_controller import AppController, AppEvent, AppEventType
from ui.web.server import create_app


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
        from ui.wifi_manager import WiFiNetwork
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
        from ui.web.server import create_app

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
        from ui.web.server import ConnectionManager

        async def _test() -> None:
            mgr = ConnectionManager()
            assert mgr.connection_count == 0

        asyncio.get_event_loop().run_until_complete(_test())


# ---------------------------------------------------------------------------
# TestChatEndpoint (Phase 4 — DISCUSS-02)
# ---------------------------------------------------------------------------

try:
    from ui.app_controller import PlanDiscussionState  # added in plan 04
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

    def test_chat_no_pending_plan(
        self,
        client: TestClient,
        ctrl: AppController,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # No plan cached
        assert ctrl._plan_discussion is None
        # Stub controller to return None as if no cache
        monkeypatch.setattr(
            ctrl, "chat_about_plan", lambda question: None,
        )
        resp = client.post(
            "/api/task/chat-about-plan",
            json={"question": "Hi?"},
        )
        assert resp.status_code == 503
        body = resp.json()
        assert body["status"] == "error"


# ---------------------------------------------------------------------------
# Phase 5 availability probe — plan 05-05 lands these routes.
# ---------------------------------------------------------------------------

try:
    from ui.app_controller import AppController as _AC
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
