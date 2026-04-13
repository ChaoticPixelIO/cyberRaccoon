"""Tests for the Skills REST API endpoints in ui.web.server."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from cyberraccoon.agent.skills import SKILL_FILENAME
from cyberraccoon.ui.app_controller import AppController
from cyberraccoon.ui.web.server import create_app


@pytest.fixture()
def skill_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """Redirect bundled and user skill dirs to temp paths."""
    bundled = tmp_path / "bundled"
    user = tmp_path / "user"
    bundled.mkdir()
    user.mkdir()
    monkeypatch.setattr("cyberraccoon.agent.skills._bundled_skills_dir", lambda: bundled)
    monkeypatch.setattr("cyberraccoon.agent.skills._user_skills_dir", lambda: user)
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


def _make_skill_md(name: str, body: str, *, description: str | None = None) -> str:
    desc = description if description is not None else f"Test skill {name}."
    return f"---\nname: {name}\ndescription: {desc}\n---\n\n{body}"


def _write_skill(directory: Path, name: str, body: str, *, description: str | None = None) -> Path:
    """Write a directory-based skill with valid frontmatter under *directory*."""
    skill_dir = directory / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    path = skill_dir / SKILL_FILENAME
    path.write_text(_make_skill_md(name, body, description=description), encoding="utf-8")
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
        _write_skill(bundled, "blender", "# Blender", description="Blender 3D")
        _write_skill(user, "my-erp", "# My ERP", description="ERP system")

        resp = client.get("/api/skills")
        assert resp.status_code == 200
        skills = resp.json()["skills"]
        names = [s["name"] for s in skills]
        assert "blender" in names
        assert "my-erp" in names
        # Check source + description
        by_name = {s["name"]: s for s in skills}
        assert by_name["blender"]["source"] == "bundled"
        assert by_name["blender"]["description"] == "Blender 3D"
        assert by_name["my-erp"]["source"] == "user"
        assert by_name["my-erp"]["description"] == "ERP system"

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

    def test_incomplete_skill_listed_with_error(
        self, client: TestClient, skill_dirs: tuple[Path, Path],
    ) -> None:
        bundled, _ = skill_dirs
        # Directory without SKILL.md
        (bundled / "halfbaked").mkdir()
        (bundled / "halfbaked" / "resource.png").write_bytes(b"\x89PNG")

        resp = client.get("/api/skills")
        assert resp.status_code == 200
        skills = resp.json()["skills"]
        by_name = {s["name"]: s for s in skills}
        assert "halfbaked" in by_name
        assert by_name["halfbaked"]["description"] is None
        assert by_name["halfbaked"]["error"] == "missing_skill_md"


# ---------------------------------------------------------------------------
# GET /api/skills/{name}
# ---------------------------------------------------------------------------

class TestGetSkill:
    def test_bundled(
        self, client: TestClient, skill_dirs: tuple[Path, Path],
    ) -> None:
        bundled, _ = skill_dirs
        _write_skill(bundled, "kicad", "# KiCad\nPCB tips.", description="KiCad PCB editor")

        resp = client.get("/api/skills/kicad")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "kicad"
        assert "KiCad" in data["content"]
        assert data["source"] == "bundled"
        assert data["description"] == "KiCad PCB editor"

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

    def test_incomplete_returns_422(
        self, client: TestClient, skill_dirs: tuple[Path, Path],
    ) -> None:
        bundled, _ = skill_dirs
        (bundled / "halfbaked").mkdir()

        resp = client.get("/api/skills/halfbaked")
        assert resp.status_code == 422
        assert SKILL_FILENAME in resp.json()["message"]

    def test_invalid_frontmatter_returns_422(
        self, client: TestClient, skill_dirs: tuple[Path, Path],
    ) -> None:
        bundled, _ = skill_dirs
        skill_dir = bundled / "bad"
        skill_dir.mkdir()
        (skill_dir / SKILL_FILENAME).write_text("# No frontmatter at all", encoding="utf-8")

        resp = client.get("/api/skills/bad")
        assert resp.status_code == 422

    def test_path_traversal_rejected(self, client: TestClient) -> None:
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
        content = _make_skill_md("my-app", "# My App\nInstructions.")
        resp = client.put(
            "/api/skills/my-app",
            json={"content": content},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

        # Verify SKILL.md inside the skill directory
        assert (user / "my-app" / SKILL_FILENAME).read_text() == content

        # Can load back via GET
        resp2 = client.get("/api/skills/my-app")
        assert resp2.status_code == 200
        assert "My App" in resp2.json()["content"]

    def test_override_bundled_creates_user_copy(
        self, client: TestClient, skill_dirs: tuple[Path, Path],
    ) -> None:
        bundled, user = skill_dirs
        bundled_content = _make_skill_md("blender", "# Bundled Blender")
        _write_skill(bundled, "blender", "# Bundled Blender")

        user_content = _make_skill_md("blender", "# Custom Blender")
        resp = client.put(
            "/api/skills/blender",
            json={"content": user_content},
        )
        assert resp.status_code == 200

        # User dir created, bundled untouched
        assert (user / "blender" / SKILL_FILENAME).read_text() == user_content
        # Bundled SKILL.md is what _write_skill wrote (full frontmatter form)
        assert (bundled / "blender" / SKILL_FILENAME).read_text() == bundled_content

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

    def test_missing_frontmatter_rejected(
        self, client: TestClient, skill_dirs: tuple[Path, Path],
    ) -> None:
        _, user = skill_dirs
        resp = client.put(
            "/api/skills/test",
            json={"content": "# No frontmatter\nJust markdown."},
        )
        assert resp.status_code == 400
        # Nothing written to disk
        assert not (user / "test").exists()

    def test_name_mismatch_rejected(
        self, client: TestClient, skill_dirs: tuple[Path, Path],
    ) -> None:
        _, user = skill_dirs
        bad = "---\nname: differentname\ndescription: x\n---\n\n# Body"
        resp = client.put(
            "/api/skills/expected",
            json={"content": bad},
        )
        assert resp.status_code == 400
        assert not (user / "expected").exists()

    def test_path_traversal_name_rejected(self, client: TestClient) -> None:
        resp = client.put(
            "/api/skills/..evil",
            json={"content": _make_skill_md("evil", "# Evil")},
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
        assert not (user / "temp").exists()

    def test_delete_user_skill_with_resources(
        self, client: TestClient, skill_dirs: tuple[Path, Path],
    ) -> None:
        _, user = skill_dirs
        _write_skill(user, "withres", "# With Res")
        (user / "withres" / "image.png").write_bytes(b"\x89PNG")

        resp = client.delete("/api/skills/withres")
        assert resp.status_code == 200
        assert not (user / "withres").exists()

    def test_delete_bundled_only_fails(
        self, client: TestClient, skill_dirs: tuple[Path, Path],
    ) -> None:
        bundled, _ = skill_dirs
        _write_skill(bundled, "blender", "# Blender")

        resp = client.delete("/api/skills/blender")
        assert resp.status_code == 400
        assert "No user skill" in resp.json()["message"]

        # Bundled directory untouched
        assert (bundled / "blender" / SKILL_FILENAME).exists()

    def test_delete_nonexistent(self, client: TestClient) -> None:
        resp = client.delete("/api/skills/nope")
        assert resp.status_code == 400

    def test_path_traversal_rejected(self, client: TestClient) -> None:
        resp = client.delete("/api/skills/..evil")
        assert resp.status_code == 400
