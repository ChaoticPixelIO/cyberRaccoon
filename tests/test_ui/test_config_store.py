"""Tests for ui.config_store — ConfigStore with YAML persistence + 3-tier merge."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest
import yaml

from cyberraccoon.config import AppConfig, LLMConfig, NetworkConfig
from cyberraccoon.ui.config_store import ConfigStore
from cyberraccoon.ui.exceptions import ConfigError


class TestSaveAndLoad:
    """Round-trip save → load tests."""

    def test_save_then_load_roundtrips(
        self, tmp_config_path: Path, sample_config: AppConfig
    ) -> None:
        store = ConfigStore(str(tmp_config_path))
        store.save(sample_config)

        loaded = store.load()

        assert loaded.llm.model == "gpt-4o"
        assert loaded.capture_source == "csi"
        assert loaded.network.wifi_ssid == "TestNetwork"
        assert loaded.network.web_port == 9999

    def test_wifi_password_never_persisted(
        self, tmp_config_path: Path, sample_config: AppConfig
    ) -> None:
        store = ConfigStore(str(tmp_config_path))
        store.save(sample_config)

        raw = yaml.safe_load(tmp_config_path.read_text())
        network = raw.get("network", {})
        assert "wifi_password" not in network

    def test_api_key_excluded_by_default(
        self, tmp_config_path: Path, sample_config: AppConfig
    ) -> None:
        store = ConfigStore(str(tmp_config_path))
        store.save(sample_config)

        raw = yaml.safe_load(tmp_config_path.read_text())
        llm = raw.get("llm", {})
        assert "api_key" not in llm

    def test_api_key_included_when_requested(
        self, tmp_config_path: Path, sample_config: AppConfig
    ) -> None:
        store = ConfigStore(str(tmp_config_path))
        store.save(sample_config, include_secrets=True)

        raw = yaml.safe_load(tmp_config_path.read_text())
        assert raw["llm"]["api_key"] == "sk-test-key"

    def test_file_permissions_0600(
        self, tmp_config_path: Path, sample_config: AppConfig
    ) -> None:
        store = ConfigStore(str(tmp_config_path))
        store.save(sample_config)

        mode = tmp_config_path.stat().st_mode & 0o777
        assert mode == 0o600, f"Expected 0600, got {oct(mode)}"


class TestLoadDefaults:
    """Loading when no YAML file exists."""

    def test_load_defaults_without_yaml(self, tmp_config_path: Path) -> None:
        store = ConfigStore(str(tmp_config_path))
        config = store.load()

        # Should be equivalent to a fresh AppConfig()
        defaults = AppConfig()
        assert config.llm.model == defaults.llm.model
        assert config.capture_source == defaults.capture_source

    def test_exists_false_when_no_file(self, tmp_config_path: Path) -> None:
        store = ConfigStore(str(tmp_config_path))
        assert store.exists() is False


class TestEnvironmentOverrides:
    """Environment variables override YAML and defaults."""

    def test_env_overrides_default(
        self, tmp_config_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CYBERRACCOON_MODEL", "claude-opus-4")
        store = ConfigStore(str(tmp_config_path))

        config = store.load()
        assert config.llm.model == "claude-opus-4"

    def test_env_overrides_yaml(
        self, tmp_config_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Save a YAML with model = "gpt-4o"
        store = ConfigStore(str(tmp_config_path))
        cfg = AppConfig()
        cfg.llm.model = "gpt-4o"
        store.save(cfg)

        # Env should win
        monkeypatch.setenv("CYBERRACCOON_MODEL", "claude-opus-4")
        loaded = store.load()
        assert loaded.llm.model == "claude-opus-4"

    def test_env_int_coercion(
        self, tmp_config_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CYBERRACCOON_WEB_PORT", "3000")
        store = ConfigStore(str(tmp_config_path))

        config = store.load()
        assert config.network.web_port == 3000
        assert isinstance(config.network.web_port, int)


class TestPartialYAML:
    """YAML files with only some fields set."""

    def test_partial_yaml_merges_with_defaults(
        self, tmp_config_path: Path
    ) -> None:
        # Write a minimal YAML with only llm.model
        tmp_config_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_config_path.write_text(
            yaml.dump({"llm": {"model": "custom-model"}}),
            encoding="utf-8",
        )

        store = ConfigStore(str(tmp_config_path))
        config = store.load()

        assert config.llm.model == "custom-model"
        # Other fields should still be defaults
        assert config.llm.temperature == 0.0
        assert config.capture_source == "hdmi"

    def test_unknown_keys_ignored(self, tmp_config_path: Path) -> None:
        tmp_config_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_config_path.write_text(
            yaml.dump({
                "llm": {"model": "x", "unknown_key": 42},
                "future_section": {"a": 1},
            }),
            encoding="utf-8",
        )

        store = ConfigStore(str(tmp_config_path))
        config = store.load()  # should not raise
        assert config.llm.model == "x"


class TestInvalidYAML:
    """Malformed YAML files should raise ConfigError."""

    def test_invalid_yaml_raises(self, tmp_config_path: Path) -> None:
        tmp_config_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_config_path.write_text(
            "not: valid: yaml: [broken",
            encoding="utf-8",
        )

        store = ConfigStore(str(tmp_config_path))
        with pytest.raises(ConfigError, match="Invalid YAML"):
            store.load()

    def test_non_dict_yaml_raises(self, tmp_config_path: Path) -> None:
        tmp_config_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_config_path.write_text(
            "- just\n- a\n- list\n",
            encoding="utf-8",
        )

        store = ConfigStore(str(tmp_config_path))
        with pytest.raises(ConfigError, match="Expected top-level mapping"):
            store.load()

    def test_empty_yaml_returns_defaults(self, tmp_config_path: Path) -> None:
        tmp_config_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_config_path.write_text("", encoding="utf-8")

        store = ConfigStore(str(tmp_config_path))
        config = store.load()
        assert config.llm.model == AppConfig().llm.model


class TestReset:
    """ConfigStore.reset() deletes the YAML file."""

    def test_reset_deletes_file(
        self, tmp_config_path: Path, sample_config: AppConfig
    ) -> None:
        store = ConfigStore(str(tmp_config_path))
        store.save(sample_config)
        assert store.exists()

        store.reset()
        assert not store.exists()

    def test_reset_noop_when_no_file(self, tmp_config_path: Path) -> None:
        store = ConfigStore(str(tmp_config_path))
        store.reset()  # should not raise
        assert not store.exists()


class TestSkillsBackwardCompat:
    """Backward compat: old `skill: "x"` → new `skills: ["x"]`."""

    def test_legacy_skill_string_migrated(self, tmp_config_path: Path) -> None:
        tmp_config_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_config_path.write_text(
            yaml.dump({"agent": {"skill": "blender"}}),
            encoding="utf-8",
        )

        store = ConfigStore(str(tmp_config_path))
        config = store.load()
        assert config.agent.skills == ["blender"]

    def test_legacy_empty_skill_not_migrated(self, tmp_config_path: Path) -> None:
        tmp_config_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_config_path.write_text(
            yaml.dump({"agent": {"skill": ""}}),
            encoding="utf-8",
        )

        store = ConfigStore(str(tmp_config_path))
        config = store.load()
        assert config.agent.skills == []

    def test_new_skills_list_not_overwritten_by_legacy(self, tmp_config_path: Path) -> None:
        """If both `skill` and `skills` exist, `skills` takes precedence."""
        tmp_config_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_config_path.write_text(
            yaml.dump({"agent": {"skill": "old", "skills": ["new1", "new2"]}}),
            encoding="utf-8",
        )

        store = ConfigStore(str(tmp_config_path))
        config = store.load()
        assert config.agent.skills == ["new1", "new2"]

    def test_skills_list_roundtrips(
        self, tmp_config_path: Path, sample_config: AppConfig
    ) -> None:
        sample_config.agent.skills = ["wechat", "blender"]
        store = ConfigStore(str(tmp_config_path))
        store.save(sample_config)

        loaded = store.load()
        assert loaded.agent.skills == ["wechat", "blender"]


class TestListCoercion:
    """_coerce_value handles list types from YAML."""

    def test_list_passthrough(self) -> None:
        from cyberraccoon.ui.config_store import _coerce_value
        assert _coerce_value(["a", "b"], "list[str]") == ["a", "b"]

    def test_csv_string_to_list(self) -> None:
        from cyberraccoon.ui.config_store import _coerce_value
        assert _coerce_value("a, b, c", "list[str]") == ["a", "b", "c"]

    def test_empty_string_to_empty_list(self) -> None:
        from cyberraccoon.ui.config_store import _coerce_value
        assert _coerce_value("", "list[str]") == []


class TestConfigPath:
    """Config path resolution."""

    def test_env_overrides_default_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        custom = tmp_path / "custom" / "config.yaml"
        monkeypatch.setenv("CYBERRACCOON_CONFIG_PATH", str(custom))

        store = ConfigStore()
        assert store.path == custom

    def test_explicit_path_overrides_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CYBERRACCOON_CONFIG_PATH", "/env/path")
        explicit = tmp_path / "explicit.yaml"

        store = ConfigStore(str(explicit))
        assert store.path == explicit
