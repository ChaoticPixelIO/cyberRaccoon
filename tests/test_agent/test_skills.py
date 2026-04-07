"""Tests for agent.skills — skill loader and list_skills."""

from __future__ import annotations

from pathlib import Path

import pytest

from cyberraccoon.agent.skills import (
    SkillNotFoundError,
    _bundled_skills_dir,
    _user_skills_dir,
    delete_user_skill,
    get_skill_info,
    get_skill_source,
    list_skills,
    load_skill,
    load_skills,
    save_user_skill,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_skill(directory: Path, name: str, content: str) -> Path:
    """Write a skill file into the given directory."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.md"
    path.write_text(content, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# load_skill tests
# ---------------------------------------------------------------------------

class TestLoadSkill:
    """Tests for load_skill()."""

    def test_bundled_skill(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        bundled = tmp_path / "bundled"
        user = tmp_path / "user"
        monkeypatch.setattr("cyberraccoon.agent.skills._bundled_skills_dir", lambda: bundled)
        monkeypatch.setattr("cyberraccoon.agent.skills._user_skills_dir", lambda: user)

        _write_skill(bundled, "kicad", "# KiCad Skill\nPCB layout tips.")

        result = load_skill("kicad")
        assert "KiCad Skill" in result
        assert "PCB layout tips." in result

    def test_user_skill(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        bundled = tmp_path / "bundled"
        user = tmp_path / "user"
        monkeypatch.setattr("cyberraccoon.agent.skills._bundled_skills_dir", lambda: bundled)
        monkeypatch.setattr("cyberraccoon.agent.skills._user_skills_dir", lambda: user)

        _write_skill(user, "myapp", "# My Custom App\nCustom instructions.")

        result = load_skill("myapp")
        assert "My Custom App" in result

    def test_user_overrides_bundled(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        bundled = tmp_path / "bundled"
        user = tmp_path / "user"
        monkeypatch.setattr("cyberraccoon.agent.skills._bundled_skills_dir", lambda: bundled)
        monkeypatch.setattr("cyberraccoon.agent.skills._user_skills_dir", lambda: user)

        _write_skill(bundled, "blender", "# Bundled Blender")
        _write_skill(user, "blender", "# User Blender Override")

        result = load_skill("blender")
        assert "User Blender Override" in result
        assert "Bundled Blender" not in result

    def test_not_found(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        bundled = tmp_path / "bundled"
        user = tmp_path / "user"
        monkeypatch.setattr("cyberraccoon.agent.skills._bundled_skills_dir", lambda: bundled)
        monkeypatch.setattr("cyberraccoon.agent.skills._user_skills_dir", lambda: user)

        with pytest.raises(SkillNotFoundError) as exc_info:
            load_skill("nonexistent")

        assert "nonexistent" in str(exc_info.value)
        assert exc_info.value.name == "nonexistent"
        assert len(exc_info.value.searched_paths) == 2

    def test_empty_file_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        bundled = tmp_path / "bundled"
        user = tmp_path / "user"
        monkeypatch.setattr("cyberraccoon.agent.skills._bundled_skills_dir", lambda: bundled)
        monkeypatch.setattr("cyberraccoon.agent.skills._user_skills_dir", lambda: user)

        _write_skill(bundled, "empty", "   \n  \n  ")

        with pytest.raises(ValueError, match="empty"):
            load_skill("empty")

    def test_whitespace_only_file_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        bundled = tmp_path / "bundled"
        user = tmp_path / "user"
        monkeypatch.setattr("cyberraccoon.agent.skills._bundled_skills_dir", lambda: bundled)
        monkeypatch.setattr("cyberraccoon.agent.skills._user_skills_dir", lambda: user)

        _write_skill(user, "blank", "")

        with pytest.raises(ValueError, match="empty"):
            load_skill("blank")


# ---------------------------------------------------------------------------
# load_skills tests (multi-skill)
# ---------------------------------------------------------------------------

class TestLoadSkills:
    """Tests for load_skills() — multi-skill loader."""

    def test_empty_list_returns_none(self) -> None:
        assert load_skills([]) is None

    def test_single_skill(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        bundled = tmp_path / "bundled"
        user = tmp_path / "user"
        monkeypatch.setattr("cyberraccoon.agent.skills._bundled_skills_dir", lambda: bundled)
        monkeypatch.setattr("cyberraccoon.agent.skills._user_skills_dir", lambda: user)

        _write_skill(bundled, "kicad", "# KiCad Skill")

        result = load_skills(["kicad"])
        assert result == "# KiCad Skill"

    def test_multiple_skills_concatenated(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        bundled = tmp_path / "bundled"
        user = tmp_path / "user"
        monkeypatch.setattr("cyberraccoon.agent.skills._bundled_skills_dir", lambda: bundled)
        monkeypatch.setattr("cyberraccoon.agent.skills._user_skills_dir", lambda: user)

        _write_skill(bundled, "alpha", "# Alpha")
        _write_skill(bundled, "beta", "# Beta")

        result = load_skills(["alpha", "beta"])
        assert result is not None
        assert "# Alpha" in result
        assert "# Beta" in result
        assert result == "# Alpha\n\n# Beta"

    def test_one_missing_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        bundled = tmp_path / "bundled"
        user = tmp_path / "user"
        monkeypatch.setattr("cyberraccoon.agent.skills._bundled_skills_dir", lambda: bundled)
        monkeypatch.setattr("cyberraccoon.agent.skills._user_skills_dir", lambda: user)

        _write_skill(bundled, "good", "# Good Skill")

        with pytest.raises(SkillNotFoundError):
            load_skills(["good", "missing"])


# ---------------------------------------------------------------------------
# Validation / path traversal
# ---------------------------------------------------------------------------

class TestValidation:
    """Tests for name validation in load_skill()."""

    def test_empty_name(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            load_skill("")

    def test_null_byte(self) -> None:
        with pytest.raises(ValueError, match="null"):
            load_skill("bad\0name")

    def test_forward_slash(self) -> None:
        with pytest.raises(ValueError, match="path separator"):
            load_skill("../etc/passwd")

    def test_backslash(self) -> None:
        with pytest.raises(ValueError, match="path separator"):
            load_skill("..\\etc\\passwd")

    def test_dot_dot(self) -> None:
        with pytest.raises(ValueError, match="\\.\\."):
            load_skill("..secret")


# ---------------------------------------------------------------------------
# list_skills tests
# ---------------------------------------------------------------------------

class TestListSkills:
    """Tests for list_skills()."""

    def test_bundled_only(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        bundled = tmp_path / "bundled"
        user = tmp_path / "user"
        monkeypatch.setattr("cyberraccoon.agent.skills._bundled_skills_dir", lambda: bundled)
        monkeypatch.setattr("cyberraccoon.agent.skills._user_skills_dir", lambda: user)

        _write_skill(bundled, "alpha", "# Alpha")
        _write_skill(bundled, "beta", "# Beta")

        result = list_skills()
        assert result == ["alpha", "beta"]

    def test_both_dirs_deduplicated(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        bundled = tmp_path / "bundled"
        user = tmp_path / "user"
        monkeypatch.setattr("cyberraccoon.agent.skills._bundled_skills_dir", lambda: bundled)
        monkeypatch.setattr("cyberraccoon.agent.skills._user_skills_dir", lambda: user)

        _write_skill(bundled, "blender", "# Bundled")
        _write_skill(bundled, "kicad", "# KiCad")
        _write_skill(user, "blender", "# User Override")
        _write_skill(user, "myapp", "# Custom")

        result = list_skills()
        assert result == ["blender", "kicad", "myapp"]

    def test_empty_dirs(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        bundled = tmp_path / "bundled"
        user = tmp_path / "user"
        monkeypatch.setattr("cyberraccoon.agent.skills._bundled_skills_dir", lambda: bundled)
        monkeypatch.setattr("cyberraccoon.agent.skills._user_skills_dir", lambda: user)

        result = list_skills()
        assert result == []

    def test_ignores_non_md_files(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        bundled = tmp_path / "bundled"
        user = tmp_path / "user"
        monkeypatch.setattr("cyberraccoon.agent.skills._bundled_skills_dir", lambda: bundled)
        monkeypatch.setattr("cyberraccoon.agent.skills._user_skills_dir", lambda: user)

        _write_skill(bundled, "valid", "# Valid")
        bundled.mkdir(parents=True, exist_ok=True)
        (bundled / "readme.txt").write_text("not a skill")

        result = list_skills()
        assert result == ["valid"]


# ---------------------------------------------------------------------------
# list_skills with real bundled dir
# ---------------------------------------------------------------------------

class TestRealBundledSkills:
    """Verify the actual bundled skills directory works."""

    def test_blender_in_bundled(self) -> None:
        """The bundled blender.md should be discoverable."""
        assert "blender" in list_skills()

    def test_load_bundled_blender(self) -> None:
        """The bundled blender skill should load successfully."""
        content = load_skill("blender")
        assert "Blender" in content
        assert len(content) > 100


# ---------------------------------------------------------------------------
# Protocol integration tests
# ---------------------------------------------------------------------------

class TestProtocolIntegration:
    """Test that skill_text flows through to protocol system prompts."""

    def test_prompt_based_includes_skill(self) -> None:
        """PromptBasedProtocol includes skill text in system prompt."""
        from unittest.mock import MagicMock, patch

        mock_openai = MagicMock()
        with patch.dict("sys.modules", {"openai": mock_openai}):
            from cyberraccoon.agent.protocols.prompt_based import PromptBasedProtocol

            proto = PromptBasedProtocol(
                provider="openai",
                model="gpt-4o",
                api_key="fake",
                skill_text="# Blender Shortcuts\nG = Grab",
            )
            prompt = proto.get_system_prompt()
            assert "# Blender Shortcuts" in prompt
            assert "G = Grab" in prompt

    def test_prompt_based_no_skill_unchanged(self) -> None:
        """Without skill_text, system prompt is unmodified."""
        from unittest.mock import MagicMock, patch

        mock_openai = MagicMock()
        with patch.dict("sys.modules", {"openai": mock_openai}):
            from cyberraccoon.agent.protocols.prompt_based import PromptBasedProtocol

            proto_no_skill = PromptBasedProtocol(
                provider="openai",
                model="gpt-4o",
                api_key="fake",
            )
            proto_none = PromptBasedProtocol(
                provider="openai",
                model="gpt-4o",
                api_key="fake",
                skill_text=None,
            )
            assert proto_no_skill.get_system_prompt() == proto_none.get_system_prompt()

    def test_anthropic_cu_includes_skill(self) -> None:
        """AnthropicCUProtocol includes skill text in system prompt."""
        from unittest.mock import MagicMock, patch

        mock_anthropic = MagicMock()
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            from cyberraccoon.agent.protocols.anthropic_cu import AnthropicCUProtocol

            proto = AnthropicCUProtocol(
                model="claude-opus-4-6",
                api_key="fake",
                skill_text="# KiCad Tips",
            )
            assert "# KiCad Tips" in proto.get_system_prompt()

    def test_anthropic_cu_no_skill_unchanged(self) -> None:
        """Without skill_text, AnthropicCU system prompt is unmodified."""
        from unittest.mock import MagicMock, patch

        mock_anthropic = MagicMock()
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            from cyberraccoon.agent.protocols.anthropic_cu import AnthropicCUProtocol

            proto = AnthropicCUProtocol(
                model="claude-opus-4-6",
                api_key="fake",
            )
            proto_none = AnthropicCUProtocol(
                model="claude-opus-4-6",
                api_key="fake",
                skill_text=None,
            )
            assert proto.get_system_prompt() == proto_none.get_system_prompt()

    def test_openai_cu_includes_skill(self) -> None:
        """OpenAICUProtocol includes skill text in system prompt."""
        from unittest.mock import MagicMock, patch

        mock_openai = MagicMock()
        with patch.dict("sys.modules", {"openai": mock_openai}):
            from cyberraccoon.agent.protocols.openai_cu import OpenAICUProtocol

            proto = OpenAICUProtocol(
                model="gpt-5.4",
                api_key="fake",
                skill_text="# WeChat Tips",
            )
            assert "# WeChat Tips" in proto.get_system_prompt()

    def test_openai_cu_no_skill_unchanged(self) -> None:
        """Without skill_text, OpenAICU system prompt is unmodified."""
        from unittest.mock import MagicMock, patch

        mock_openai = MagicMock()
        with patch.dict("sys.modules", {"openai": mock_openai}):
            from cyberraccoon.agent.protocols.openai_cu import OpenAICUProtocol

            proto = OpenAICUProtocol(
                model="gpt-5.4",
                api_key="fake",
            )
            proto_none = OpenAICUProtocol(
                model="gpt-5.4",
                api_key="fake",
                skill_text=None,
            )
            assert proto.get_system_prompt() == proto_none.get_system_prompt()

    def test_factory_passes_skill_to_prompt_based(self) -> None:
        """create_protocol passes skill_text through to PromptBasedProtocol."""
        from unittest.mock import MagicMock, patch

        mock_openai = MagicMock()
        with patch.dict("sys.modules", {"openai": mock_openai}):
            from cyberraccoon.agent.protocols.base import create_protocol

            proto = create_protocol(
                provider="openai",
                model="gpt-4o",
                api_key="fake",
                skill_text="# Factory Skill",
            )
            assert "# Factory Skill" in proto.get_system_prompt()


# ---------------------------------------------------------------------------
# get_skill_source tests
# ---------------------------------------------------------------------------

class TestGetSkillSource:
    """Tests for get_skill_source()."""

    def test_bundled_source(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        bundled = tmp_path / "bundled"
        user = tmp_path / "user"
        monkeypatch.setattr("cyberraccoon.agent.skills._bundled_skills_dir", lambda: bundled)
        monkeypatch.setattr("cyberraccoon.agent.skills._user_skills_dir", lambda: user)

        _write_skill(bundled, "kicad", "# KiCad")
        assert get_skill_source("kicad") == "bundled"

    def test_user_source(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        bundled = tmp_path / "bundled"
        user = tmp_path / "user"
        monkeypatch.setattr("cyberraccoon.agent.skills._bundled_skills_dir", lambda: bundled)
        monkeypatch.setattr("cyberraccoon.agent.skills._user_skills_dir", lambda: user)

        _write_skill(user, "myapp", "# My App")
        assert get_skill_source("myapp") == "user"

    def test_user_override_returns_user(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        bundled = tmp_path / "bundled"
        user = tmp_path / "user"
        monkeypatch.setattr("cyberraccoon.agent.skills._bundled_skills_dir", lambda: bundled)
        monkeypatch.setattr("cyberraccoon.agent.skills._user_skills_dir", lambda: user)

        _write_skill(bundled, "blender", "# Bundled")
        _write_skill(user, "blender", "# User Override")
        assert get_skill_source("blender") == "user"

    def test_not_found_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        bundled = tmp_path / "bundled"
        user = tmp_path / "user"
        monkeypatch.setattr("cyberraccoon.agent.skills._bundled_skills_dir", lambda: bundled)
        monkeypatch.setattr("cyberraccoon.agent.skills._user_skills_dir", lambda: user)

        with pytest.raises(SkillNotFoundError):
            get_skill_source("missing")


# ---------------------------------------------------------------------------
# get_skill_info tests
# ---------------------------------------------------------------------------

class TestGetSkillInfo:
    """Tests for get_skill_info()."""

    def test_returns_name_content_source(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        bundled = tmp_path / "bundled"
        user = tmp_path / "user"
        monkeypatch.setattr("cyberraccoon.agent.skills._bundled_skills_dir", lambda: bundled)
        monkeypatch.setattr("cyberraccoon.agent.skills._user_skills_dir", lambda: user)

        _write_skill(bundled, "kicad", "# KiCad\nTips.")
        info = get_skill_info("kicad")
        assert info["name"] == "kicad"
        assert "KiCad" in info["content"]
        assert info["source"] == "bundled"


# ---------------------------------------------------------------------------
# save_user_skill tests
# ---------------------------------------------------------------------------

class TestSaveUserSkill:
    """Tests for save_user_skill()."""

    def test_creates_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        user = tmp_path / "user_skills"
        monkeypatch.setattr("cyberraccoon.agent.skills._user_skills_dir", lambda: user)

        path = save_user_skill("myapp", "# My App\nContent.")
        assert path.exists()
        assert path.read_text(encoding="utf-8") == "# My App\nContent."

    def test_creates_directory(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        user = tmp_path / "deep" / "nested" / "skills"
        monkeypatch.setattr("cyberraccoon.agent.skills._user_skills_dir", lambda: user)

        save_user_skill("test", "# Test")
        assert (user / "test.md").exists()

    def test_empty_content_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        user = tmp_path / "user"
        monkeypatch.setattr("cyberraccoon.agent.skills._user_skills_dir", lambda: user)

        with pytest.raises(ValueError, match="empty"):
            save_user_skill("test", "  \n  ")

    def test_invalid_name_raises(self) -> None:
        with pytest.raises(ValueError, match="path separator"):
            save_user_skill("../evil", "# Evil")

    def test_overwrites_existing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        user = tmp_path / "user"
        monkeypatch.setattr("cyberraccoon.agent.skills._user_skills_dir", lambda: user)

        save_user_skill("app", "# Version 1")
        save_user_skill("app", "# Version 2")
        assert (user / "app.md").read_text(encoding="utf-8") == "# Version 2"


# ---------------------------------------------------------------------------
# delete_user_skill tests
# ---------------------------------------------------------------------------

class TestDeleteUserSkill:
    """Tests for delete_user_skill()."""

    def test_deletes_existing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        user = tmp_path / "user"
        monkeypatch.setattr("cyberraccoon.agent.skills._user_skills_dir", lambda: user)

        _write_skill(user, "temp", "# Temp")
        assert delete_user_skill("temp") is True
        assert not (user / "temp.md").exists()

    def test_returns_false_if_not_found(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        user = tmp_path / "user"
        user.mkdir()
        monkeypatch.setattr("cyberraccoon.agent.skills._user_skills_dir", lambda: user)

        assert delete_user_skill("nope") is False

    def test_invalid_name_raises(self) -> None:
        with pytest.raises(ValueError, match="path separator"):
            delete_user_skill("../evil")
