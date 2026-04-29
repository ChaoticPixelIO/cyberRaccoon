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

    def test_api_key_persisted_by_default(
        self, tmp_config_path: Path, sample_config: AppConfig
    ) -> None:
        """The yaml is the canonical source of truth for API keys, so
        save() persists them by default. File permissions (0o600) keep
        the file owner-only readable."""
        store = ConfigStore(str(tmp_config_path))
        store.save(sample_config)

        raw = yaml.safe_load(tmp_config_path.read_text())
        assert raw["llm"]["api_key"] == "sk-test-key"

    def test_api_key_can_be_stripped_for_export(
        self, tmp_config_path: Path, sample_config: AppConfig
    ) -> None:
        """``include_secrets=False`` preserves the pre-yaml-only behavior
        for uses like exporting a redacted config for sharing/backup."""
        store = ConfigStore(str(tmp_config_path))
        store.save(sample_config, include_secrets=False)

        raw = yaml.safe_load(tmp_config_path.read_text())
        llm = raw.get("llm", {})
        assert "api_key" not in llm

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
    """Infrastructure env vars (paths, network, etc.) still override YAML.

    Provider-scoped env vars (``{PROVIDER}_API_KEY`` / ``_MODEL`` /
    ``_BASE_URL``) are intentionally NOT supported — API keys, models, and
    base URLs live exclusively in the YAML and are edited via the Config
    tab.
    """

    def test_provider_env_vars_are_ignored(
        self, tmp_config_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Setting OPENAI_API_KEY / OPENAI_MODEL has no effect — yaml wins."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-should-be-ignored")
        monkeypatch.setenv("OPENAI_MODEL", "gpt-ignored")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-should-be-ignored")
        store = ConfigStore(str(tmp_config_path))

        config = store.load()
        # No yaml exists → defaults; api_key stays empty, model = default.
        assert config.llm.api_key == ""
        assert config.llm.model != "gpt-ignored"
        assert "anthropic" not in config.llm.providers or \
            config.llm.providers["anthropic"].get("api_key", "") \
                != "sk-ant-should-be-ignored"

    def test_cyberraccoon_provider_env_still_selects_active(
        self, tmp_config_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``CYBERRACCOON_PROVIDER`` is infrastructure config (which provider
        to use) and is still honored; only the per-provider *secrets* aren't."""
        monkeypatch.setenv("CYBERRACCOON_PROVIDER", "anthropic")
        store = ConfigStore(str(tmp_config_path))

        config = store.load()
        assert config.llm.provider == "anthropic"

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
        assert config.capture_source == "uvc"

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


class TestLLMProviderSnapshots:
    """llm.providers per-provider snapshot persistence."""

    def test_yaml_without_providers_block_seeds_active_provider_snapshot(
        self, tmp_config_path: Path
    ) -> None:
        # YAML with only flat llm fields and no `providers` sub-dict —
        # the loader should seed `providers[<active>]` from those fields.
        tmp_config_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_config_path.write_text(
            yaml.dump({
                "llm": {
                    "provider": "anthropic",
                    "model": "claude-opus-4-7",
                    "base_url": "https://api.example.com",
                    "temperature": 0.2,
                },
            }),
            encoding="utf-8",
        )

        store = ConfigStore(str(tmp_config_path))
        config = store.load()

        assert "anthropic" in config.llm.providers
        snap = config.llm.providers["anthropic"]
        assert snap["model"] == "claude-opus-4-7"
        assert snap["base_url"] == "https://api.example.com"
        assert snap["temperature"] == 0.2

    def test_providers_dict_round_trips(self, tmp_config_path: Path) -> None:
        store = ConfigStore(str(tmp_config_path))
        cfg = AppConfig()
        cfg.llm.provider = "anthropic"
        cfg.llm.model = "claude-opus-4-7"
        cfg.llm.providers = {
            "anthropic": {
                "model": "claude-opus-4-7",
                "api_key": "sk-ant-xxx",
                "base_url": None,
                "max_tokens": 1024,
                "temperature": 0.0,
            },
            "openai": {
                "model": "gpt-4o",
                "api_key": "sk-openai-xxx",
                "base_url": None,
                "max_tokens": 2048,
                "temperature": 0.5,
            },
        }
        store.save(cfg)

        loaded = store.load()
        assert set(loaded.llm.providers) == {"anthropic", "openai"}
        assert loaded.llm.providers["openai"]["model"] == "gpt-4o"
        assert loaded.llm.providers["openai"]["temperature"] == 0.5

    def test_api_key_persisted_in_provider_snapshots(
        self, tmp_config_path: Path
    ) -> None:
        """Per-provider snapshots keep their api_key on save so the user's
        non-active provider keys don't vanish when they switch providers."""
        store = ConfigStore(str(tmp_config_path))
        cfg = AppConfig()
        cfg.llm.providers = {
            "anthropic": {"model": "claude-opus-4-7", "api_key": "sk-secret"},
        }
        store.save(cfg)

        raw = yaml.safe_load(tmp_config_path.read_text())
        providers = raw["llm"]["providers"]
        assert providers["anthropic"]["api_key"] == "sk-secret"

    def test_api_key_stripped_from_provider_snapshots_on_export(
        self, tmp_config_path: Path
    ) -> None:
        """``include_secrets=False`` strips api_keys from snapshots too."""
        store = ConfigStore(str(tmp_config_path))
        cfg = AppConfig()
        cfg.llm.providers = {
            "anthropic": {"model": "claude-opus-4-7", "api_key": "sk-secret"},
        }
        store.save(cfg, include_secrets=False)

        raw = yaml.safe_load(tmp_config_path.read_text())
        providers = raw["llm"]["providers"]
        assert "api_key" not in providers["anthropic"]


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


class TestSkillsList:
    """Skills list persistence."""

    def test_skills_list_roundtrips(
        self, tmp_config_path: Path, sample_config: AppConfig
    ) -> None:
        sample_config.agent.skills = ["notepad", "blender"]
        store = ConfigStore(str(tmp_config_path))
        store.save(sample_config)

        loaded = store.load()
        assert loaded.agent.skills == ["notepad", "blender"]


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
