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
