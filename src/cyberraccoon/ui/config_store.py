"""M5 Config Store — YAML configuration persistence with 3-tier merge.

Priority (highest to lowest):
    1. Environment variables (``CYBERRACCOON_*``)
    2. YAML config file (``~/.cyberraccoon/config.yaml``)
    3. Dataclass defaults

Usage::

    store = ConfigStore()                      # default path
    config = store.load()                      # 3-tier merged AppConfig
    config.llm.model = "gpt-4o"
    store.save(config)                         # persist to YAML

Security:
    - ``wifi_password`` is **never** written to YAML.
    - ``api_key`` is excluded by default (override with ``save(..., include_secrets=True)``).
    - Config file is created with mode ``0o600`` (owner-only).
"""

from __future__ import annotations

import logging
import os
import stat
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import yaml

from cyberraccoon.config import (
    AgentConfig,
    AppConfig,
    BLEConfig,
    CaptureConfig,
    ExecutorConfig,
    LLMConfig,
    NetworkConfig,
    load_app_config,
)
from cyberraccoon.ui.exceptions import ConfigError

logger = logging.getLogger("M5.config")

# Fields that must never be persisted to YAML
_NEVER_PERSIST: set[str] = {"wifi_password"}

# Fields excluded from YAML by default (can be overridden)
_SECRET_FIELDS: set[str] = {"api_key"}

# Environment variable → config path mapping
_ENV_MAP: dict[str, tuple[str, str]] = {
    "CYBERRACCOON_SOURCE": ("", "capture_source"),
    "CYBERRACCOON_TRANSPORT": ("", "executor_transport"),
    "CYBERRACCOON_DEVICE": ("capture", "device_index"),
    "CYBERRACCOON_PROVIDER": ("llm", "provider"),
    # Per-provider LLM settings (api_key, model, base_url) live exclusively
    # in the YAML — configure them via the Config tab, not via env vars.
    "CYBERRACCOON_WEB_HOST": ("network", "web_host"),
    "CYBERRACCOON_WEB_PORT": ("network", "web_port"),
    "CYBERRACCOON_WIFI_SSID": ("network", "wifi_ssid"),
    "CYBERRACCOON_WIFI_PASSWORD": ("network", "wifi_password"),
    "CYBERRACCOON_BLE_NAME": ("ble", "advertise_name"),
    "CYBERRACCOON_TARGET_OS": ("", "target_os"),
}

# Section name → dataclass type (for type coercion)
_SECTION_TYPES: dict[str, type] = {
    "capture": CaptureConfig,
    "llm": LLMConfig,
    "agent": AgentConfig,
    "executor": ExecutorConfig,
    "network": NetworkConfig,
    "ble": BLEConfig,
}


class ConfigStore:
    """YAML config file persistence with 3-tier merge.

    Args:
        path: Config file path. Defaults to ``~/.cyberraccoon/config.yaml``.
    """

    DEFAULT_PATH: str = "~/.cyberraccoon/config.yaml"

    def __init__(self, path: str | None = None) -> None:
        env_path = os.environ.get("CYBERRACCOON_CONFIG_PATH")
        raw = path or env_path or self.DEFAULT_PATH
        self._path = Path(raw).expanduser()

    @property
    def path(self) -> Path:
        """Resolved config file path."""
        return self._path

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> AppConfig:
        """Load config with 3-tier merge: defaults → YAML → env vars.

        Returns:
            Fully-merged ``AppConfig``.

        Raises:
            ConfigError: If the YAML file exists but is malformed.
        """
        # Tier 1: dataclass defaults
        config = AppConfig()

        # Tier 2: YAML file overrides
        if self.exists():
            yaml_data = self._read_yaml()
            config = self._merge_yaml(config, yaml_data)

        # Tier 3: environment variable overrides (highest priority)
        config = self._apply_env_overrides(config)

        return config

    def save(self, config: AppConfig, *, include_secrets: bool = True) -> None:
        """Persist config to YAML file.

        Args:
            config: The configuration to save.
            include_secrets: If ``True`` (default), persist ``api_key`` values
                so they survive restarts. ``wifi_password`` is **never** saved
                regardless of this flag. The yaml is written with ``0o600``
                permissions; API keys are the canonical source of truth and
                belong in this file (there is no env-var fallback).

        Raises:
            ConfigError: If the directory cannot be created or file not writable.
        """
        data = self._config_to_dict(config, include_secrets=include_secrets)

        try:
            self._ensure_dir()
            self._write_yaml(data)
            self._set_file_permissions()
        except OSError as e:
            raise ConfigError(f"Failed to save config: {e}") from e

        logger.info("Config saved to %s", self._path)

    def exists(self) -> bool:
        """Return ``True`` if the YAML config file exists."""
        return self._path.is_file()

    def reset(self) -> None:
        """Delete the config file, reverting to defaults.

        Raises:
            ConfigError: If the file cannot be deleted.
        """
        if self._path.exists():
            try:
                self._path.unlink()
                logger.info("Config reset (deleted %s)", self._path)
            except OSError as e:
                raise ConfigError(f"Failed to delete config: {e}") from e

    # ------------------------------------------------------------------
    # YAML I/O
    # ------------------------------------------------------------------

    def _read_yaml(self) -> dict[str, Any]:
        """Read and parse the YAML file.

        Returns:
            Parsed dict (may be empty if file is blank).

        Raises:
            ConfigError: On parse or read errors.
        """
        try:
            text = self._path.read_text(encoding="utf-8")
        except OSError as e:
            raise ConfigError(f"Cannot read {self._path}: {e}") from e

        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as e:
            raise ConfigError(f"Invalid YAML in {self._path}: {e}") from e

        if data is None:
            return {}
        if not isinstance(data, dict):
            raise ConfigError(
                f"Expected top-level mapping in {self._path}, got {type(data).__name__}"
            )
        return data

    def _write_yaml(self, data: dict[str, Any]) -> None:
        """Write dict to YAML file."""
        text = yaml.dump(
            data,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )
        self._path.write_text(text, encoding="utf-8")

    def _ensure_dir(self) -> None:
        """Create the config directory if it doesn't exist."""
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def _set_file_permissions(self) -> None:
        """Set config file to owner-only read/write (0600)."""
        try:
            self._path.chmod(stat.S_IRUSR | stat.S_IWUSR)
        except OSError:
            # Best-effort; may fail on some filesystems
            pass

    # ------------------------------------------------------------------
    # Merge logic
    # ------------------------------------------------------------------

    def _merge_yaml(self, config: AppConfig, yaml_data: dict[str, Any]) -> AppConfig:
        """Apply YAML values on top of dataclass defaults.

        Unknown keys are silently ignored (forward compatibility).
        """
        # Top-level scalar fields
        if "capture_source" in yaml_data:
            config.capture_source = str(yaml_data["capture_source"])
        if "executor_transport" in yaml_data:
            config.executor_transport = str(yaml_data["executor_transport"])
        if "target_os" in yaml_data:
            config.target_os = str(yaml_data["target_os"])

        # Also accept capture.source and executor.transport as aliases
        capture_data = yaml_data.get("capture", {})
        if isinstance(capture_data, dict) and "source" in capture_data:
            config.capture_source = str(capture_data.pop("source"))

        executor_data = yaml_data.get("executor", {})
        if isinstance(executor_data, dict) and "transport" in executor_data:
            config.executor_transport = str(executor_data.pop("transport"))

        # Merge each section
        for section_name, dc_type in _SECTION_TYPES.items():
            section_data = yaml_data.get(section_name, {})
            if not isinstance(section_data, dict):
                continue
            current = getattr(config, section_name)
            updated = self._merge_section(current, section_data, dc_type)
            setattr(config, section_name, updated)

        # Backward compat: migrate legacy `skill: "name"` → `skills: ["name"]`
        agent_data = yaml_data.get("agent", {})
        if isinstance(agent_data, dict) and "skill" in agent_data and "skills" not in agent_data:
            old_skill = agent_data["skill"]
            if old_skill:
                config.agent.skills = [str(old_skill)]

        # Backward compat: seed `llm.providers[<active>]` from the flat llm
        # fields if the YAML predates per-provider snapshots. Keeps existing
        # configs working without a manual migration.
        if config.llm.provider not in config.llm.providers:
            config.llm.providers[config.llm.provider] = {
                "model": config.llm.model,
                "api_key": config.llm.api_key,
                "base_url": config.llm.base_url,
                "max_tokens": config.llm.max_tokens,
                "temperature": config.llm.temperature,
            }

        return config

    @staticmethod
    def _merge_section(current: Any, data: dict[str, Any], dc_type: type) -> Any:
        """Merge a dict into a dataclass instance, coercing types."""
        valid_fields = {f.name: f for f in fields(dc_type)}
        updates: dict[str, Any] = {}

        for key, value in data.items():
            if key not in valid_fields:
                continue  # Ignore unknown keys
            target_type = valid_fields[key].type
            # dict[str, dict[str, Any]] — used by LLMConfig.providers. The
            # generic _coerce_value branches don't cover nested dicts, so
            # pass through when we get a dict.
            if "dict" in str(target_type) and isinstance(value, dict):
                updates[key] = {
                    str(k): dict(v) if isinstance(v, dict) else v
                    for k, v in value.items()
                }
            else:
                updates[key] = _coerce_value(value, target_type)

        if not updates:
            return current

        # Create new instance with merged values
        current_dict = {f.name: getattr(current, f.name) for f in fields(dc_type)}
        current_dict.update(updates)
        return dc_type(**current_dict)

    @staticmethod
    def _apply_env_overrides(config: AppConfig) -> AppConfig:
        """Apply environment variable overrides (highest priority)."""
        for env_var, (section, key) in _ENV_MAP.items():
            value = os.environ.get(env_var)
            if value is None:
                continue

            if not section:
                # Top-level field
                setattr(config, key, value)
            else:
                sub_config = getattr(config, section)
                valid_fields = {f.name: f for f in fields(type(sub_config))}
                if key in valid_fields:
                    coerced = _coerce_value(value, valid_fields[key].type)
                    setattr(sub_config, key, coerced)

        # Sync the active provider's stored snapshot into the flat LLMConfig
        # fields so every consumer that reads ``config.llm.model`` etc. sees
        # the values the user last saved under this provider.
        active_snap = config.llm.providers.get(config.llm.provider)
        if active_snap:
            for fname in ("model", "api_key", "base_url", "max_tokens", "temperature"):
                if fname in active_snap:
                    setattr(config.llm, fname, active_snap[fname])

        return config

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    @staticmethod
    def _config_to_dict(
        config: AppConfig,
        *,
        include_secrets: bool = False,
    ) -> dict[str, Any]:
        """Convert AppConfig to a dict suitable for YAML output.

        Filters out sensitive fields and ``None`` values.
        """
        data: dict[str, Any] = {}

        # Top-level scalars
        data["capture_source"] = config.capture_source
        data["executor_transport"] = config.executor_transport
        data["target_os"] = config.target_os

        # Each section
        for section_name in _SECTION_TYPES:
            sub = getattr(config, section_name)
            section_dict = asdict(sub)

            # Remove sensitive fields
            for field_name in list(section_dict):
                if field_name in _NEVER_PERSIST:
                    del section_dict[field_name]
                elif field_name in _SECRET_FIELDS and not include_secrets:
                    del section_dict[field_name]

            # Apply the same secret-stripping rule inside llm.providers
            # snapshots so api_keys aren't persisted per-provider either.
            if section_name == "llm":
                providers = section_dict.get("providers")
                if isinstance(providers, dict):
                    cleaned: dict[str, dict[str, Any]] = {}
                    for name, snap in providers.items():
                        if not isinstance(snap, dict):
                            continue
                        pruned = {
                            k: v for k, v in snap.items()
                            if k not in _NEVER_PERSIST
                            and (k not in _SECRET_FIELDS or include_secrets)
                            and v is not None
                        }
                        if pruned:
                            cleaned[name] = pruned
                    if cleaned:
                        section_dict["providers"] = cleaned
                    else:
                        section_dict.pop("providers", None)

            # Remove None values
            section_dict = {
                k: v for k, v in section_dict.items() if v is not None
            }

            if section_dict:
                data[section_name] = section_dict

        return data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _coerce_value(value: Any, type_hint: str | type) -> Any:
    """Best-effort type coercion for YAML/env values.

    Handles common type annotations: ``str``, ``int``, ``float``, ``bool``,
    ``str | None``. Falls through to returning the original value if
    coercion is not possible.
    """
    # Normalise type hint to string for pattern matching
    hint = str(type_hint)

    if value is None:
        return None

    # Handle Optional / union with None
    if "None" in hint:
        if isinstance(value, str) and value.lower() in ("", "none", "null"):
            return None
        # Strip the Optional wrapper and recurse
        base_hint = hint.replace(" | None", "").replace("None | ", "")
        return _coerce_value(value, base_hint)

    if hint in ("str", "<class 'str'>"):
        return str(value)
    if hint in ("int", "<class 'int'>"):
        return int(value)
    if hint in ("float", "<class 'float'>"):
        return float(value)
    if hint in ("bool", "<class 'bool'>"):
        return _coerce_bool(value)

    if "list" in hint:
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            return [v.strip() for v in value.split(",") if v.strip()]
        return value

    return value


# Review type-design — strict-ish bool coercion for YAML values.
# Defense in depth for AgentConfig.auto_replan and other bool fields. The
# HTTP boundary uses Pydantic StrictBool, but the YAML write path was
# previously the unprotected door — a manually-edited config.yaml with
# `auto_replan: "asdf"` would silently coerce to True/False via Python's
# `bool("asdf") = True`. Now we recognise an explicit vocabulary and log
# (rather than raise — config errors should never break startup).
_TRUE_TOKENS = frozenset({"true", "1", "yes", "y", "on"})
_FALSE_TOKENS = frozenset({"false", "0", "no", "n", "off", ""})


def _coerce_bool(value: Any) -> bool:
    """Coerce a YAML-loaded value to bool.

    YAML `bool` and Python `bool` pass through unchanged. Strings are
    matched against an explicit true/false vocabulary; unknown strings
    log a warning and fall back to ``bool(value.strip())`` so the field
    still has a sensible value but the operator can see what happened.

    Other types (int, float, list, dict) fall back to ``bool(value)``
    with the same caveat as before.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalised = value.strip().lower()
        if normalised in _TRUE_TOKENS:
            return True
        if normalised in _FALSE_TOKENS:
            return False
        logger.warning(
            "config: unrecognised bool value %r; "
            "expected one of %s (true) or %s (false)",
            value,
            sorted(_TRUE_TOKENS),
            sorted(t for t in _FALSE_TOKENS if t),
        )
        return bool(normalised)
    return bool(value)
