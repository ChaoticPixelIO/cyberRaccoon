"""Tests for the Skills REST API endpoints in ui.web.server."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ui.app_controller import AppController
from ui.web.server import create_app


@pytest.fixture()
def skill_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """Redirect bundled and user skill dirs to temp paths."""
    bundled = tmp_path / "bundled"
    user = tmp_path / "user"
    bundled.mkdir()
    user.mkdir()
    monkeypatch.setattr("agent.skills._bundled_skills_dir", lambda: bundled)
    monkeypatch.setattr("agent.skills._user_skills_dir", lambda: user)
    return bundled, user


@pytest.fixture()
def ctrl(tmp_path: Path) -> AppController:
    c = AppController(config_path=str(tmp_path / "cfg.yaml"))
    c.load_config()
    return c


@pytest.fixture()
def client(ctrl: AppController, skill_dirs: tuple[Path, Path]) -> TestClient:
    app = create_app(ctrl)
    return TestClient(app)


def _write_skill(directory: Path, name: str, content: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.md"
    path.write_text(content, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# GET /api/skills
# ---------------------------------------------------------------------------

class TestListSkills:
    def test_empty(self, client: TestClient) -> None:
        resp = client.get("/api/skills")
        assert resp.status_code == 200
        assert resp.json() == {"skills": []}

    def test_with_bundled_and_user(
        self, client: TestClient, skill_dirs: tuple[Path, Path],
    ) -> None:
        bundled, user = skill_dirs
        _write_skill(bundled, "blender", "# Blender")
        _write_skill(user, "my-erp", "# My ERP")

        resp = client.get("/api/skills")
        assert resp.status_code == 200
        skills = resp.json()["skills"]
        names = [s["name"] for s in skills]
        assert "blender" in names
        assert "my-erp" in names
        # Check source info
        by_name = {s["name"]: s for s in skills}
        assert by_name["blender"]["source"] == "bundled"
        assert by_name["my-erp"]["source"] == "user"

    def test_user_override_shows_user_source(
        self, client: TestClient, skill_dirs: tuple[Path, Path],
    ) -> None:
        bundled, user = skill_dirs
        _write_skill(bundled, "blender", "# Bundled")
        _write_skill(user, "blender", "# User Override")

        resp = client.get("/api/skills")
        skills = resp.json()["skills"]
        by_name = {s["name"]: s for s in skills}
        assert by_name["blender"]["source"] == "user"


# ---------------------------------------------------------------------------
# GET /api/skills/{name}
# ---------------------------------------------------------------------------

class TestGetSkill:
    def test_bundled(
        self, client: TestClient, skill_dirs: tuple[Path, Path],
    ) -> None:
        bundled, _ = skill_dirs
        _write_skill(bundled, "kicad", "# KiCad\nPCB tips.")

        resp = client.get("/api/skills/kicad")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "kicad"
        assert "KiCad" in data["content"]
        assert data["source"] == "bundled"

    def test_user(
        self, client: TestClient, skill_dirs: tuple[Path, Path],
    ) -> None:
        _, user = skill_dirs
        _write_skill(user, "custom", "# Custom Skill")

        resp = client.get("/api/skills/custom")
        assert resp.status_code == 200
        data = resp.json()
        assert data["source"] == "user"

    def test_not_found(self, client: TestClient) -> None:
        resp = client.get("/api/skills/nonexistent")
        assert resp.status_code == 404
        assert "not found" in resp.json()["message"]

    def test_path_traversal_rejected(self, client: TestClient) -> None:
        # ".." in name triggers validation even without slashes
        resp = client.get("/api/skills/..secret")
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# PUT /api/skills/{name}
# ---------------------------------------------------------------------------

class TestPutSkill:
    def test_create_user_skill(
        self, client: TestClient, skill_dirs: tuple[Path, Path],
    ) -> None:
        _, user = skill_dirs
        resp = client.put(
            "/api/skills/my-app",
            json={"content": "# My App\nInstructions."},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

        # Verify file written
        assert (user / "my-app.md").read_text() == "# My App\nInstructions."

        # Can load back via GET
        resp2 = client.get("/api/skills/my-app")
        assert resp2.status_code == 200
        assert "My App" in resp2.json()["content"]

    def test_override_bundled_creates_user_copy(
        self, client: TestClient, skill_dirs: tuple[Path, Path],
    ) -> None:
        bundled, user = skill_dirs
        _write_skill(bundled, "blender", "# Bundled Blender")

        resp = client.put(
            "/api/skills/blender",
            json={"content": "# Custom Blender"},
        )
        assert resp.status_code == 200

        # User file created, bundled untouched
        assert (user / "blender.md").read_text() == "# Custom Blender"
        assert (bundled / "blender.md").read_text() == "# Bundled Blender"

        # GET returns the user version
        resp2 = client.get("/api/skills/blender")
        assert "Custom Blender" in resp2.json()["content"]
        assert resp2.json()["source"] == "user"

    def test_empty_content_rejected(self, client: TestClient) -> None:
        resp = client.put(
            "/api/skills/test",
            json={"content": "  \n  "},
        )
        assert resp.status_code == 400
        assert "empty" in resp.json()["message"].lower()

    def test_path_traversal_name_rejected(self, client: TestClient) -> None:
        resp = client.put(
            "/api/skills/..evil",
            json={"content": "# Evil"},
        )
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# DELETE /api/skills/{name}
# ---------------------------------------------------------------------------

class TestDeleteSkill:
    def test_delete_user_skill(
        self, client: TestClient, skill_dirs: tuple[Path, Path],
    ) -> None:
        _, user = skill_dirs
        _write_skill(user, "temp", "# Temp")

        resp = client.delete("/api/skills/temp")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
        assert not (user / "temp.md").exists()

    def test_delete_bundled_only_fails(
        self, client: TestClient, skill_dirs: tuple[Path, Path],
    ) -> None:
        bundled, _ = skill_dirs
        _write_skill(bundled, "blender", "# Blender")

        resp = client.delete("/api/skills/blender")
        assert resp.status_code == 400
        assert "No user skill" in resp.json()["message"]

        # Bundled file untouched
        assert (bundled / "blender.md").exists()

    def test_delete_nonexistent(self, client: TestClient) -> None:
        resp = client.delete("/api/skills/nope")
        assert resp.status_code == 400

    def test_path_traversal_rejected(self, client: TestClient) -> None:
        resp = client.delete("/api/skills/..evil")
        assert resp.status_code == 400
