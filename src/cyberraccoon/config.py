"""CyberRaccoon central configuration.

Reads settings from environment variables and provides typed config
dataclasses for each module.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# M1 Screen Capture
# ---------------------------------------------------------------------------

@dataclass
class CaptureConfig:
    device_index: int = 0
    target_width: int = 1280
    target_height: int = 720
    jpeg_quality: int = 80
    source_width: int = 1920
    source_height: int = 1080


# ---------------------------------------------------------------------------
# M3 LLM Client
# ---------------------------------------------------------------------------

# Built-in defaults for each LLM provider, used when switching to a provider
# that has no saved snapshot yet. Keyed by the value of ``LLMConfig.provider``.
# Fields mirror the flat ``LLMConfig`` fields populated when that provider is
# active (``model``, ``base_url``, ``max_tokens``, ``temperature``). ``api_key``
# is intentionally excluded — it is resolved from env vars at load time.
LLM_PROVIDER_DEFAULTS: dict[str, dict[str, Any]] = {
    "anthropic": {
        "model": "claude-opus-4-6",
        "base_url": None,
        "max_tokens": 1024,
        "temperature": 0.0,
    },
    "openai": {
        "model": "gpt-5.4",
        "base_url": None,
        "max_tokens": 1024,
        "temperature": 0.0,
    },
}

# Fields snapshotted per provider inside ``LLMConfig.providers``.
_LLM_PROVIDER_SNAPSHOT_FIELDS: tuple[str, ...] = (
    "model",
    "api_key",
    "base_url",
    "max_tokens",
    "temperature",
)


@dataclass
class LLMConfig:
    provider: str = "openai"
    model: str = "gpt-5.4"
    api_key: str = ""
    base_url: str | None = None
    max_tokens: int = 1024
    temperature: float = 0.0
    # Per-provider snapshots of the flat fields above. When the active provider
    # is switched, the current flat fields are saved here under the old provider
    # name, and the new provider's snapshot (or ``LLM_PROVIDER_DEFAULTS``) is
    # loaded into the flat fields.
    providers: dict[str, dict[str, Any]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# M2 Vision Agent
# ---------------------------------------------------------------------------

@dataclass
class AgentConfig:
    max_steps: int = 50
    max_consecutive_failures: int = 3
    post_action_delay_s: float = 1.0
    history_max_turns: int = 10
    task_timeout_s: float = 600.0
    stability_check: bool = True
    stability_threshold: float = 2.0
    stability_interval_s: float = 0.5
    stability_max_wait_s: float = 5.0
    protocol_override: str = "auto"
    enable_cache: bool = True
    skills: list[str] = field(default_factory=list)
    # Phase 3 — REPLAN-06: when True, paths A and B skip the dialog and re-plan automatically.
    auto_replan: bool = False


# ---------------------------------------------------------------------------
# M4 Action Executor
# ---------------------------------------------------------------------------

@dataclass
class ExecutorConfig:
    device: str = "/dev/hidg0"
    screen_width: int = 1280
    screen_height: int = 720


# ---------------------------------------------------------------------------
# M4 Input Humanization (anti anti-bot)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HumanizeConfig:
    """Configuration for input humanization (anti anti-bot).

    When enabled, mouse movements follow natural Bezier curves instead of
    instant teleports, and typing uses variable inter-key timing with
    punctuation pauses and speed ramp-up.
    """

    enabled: bool = False

    # Mouse movement
    movement_speed: float = 1.0          # multiplier: 0.5=fast, 2.0=slow
    curve_noise: float = 1.0             # Bezier noise: 0.0=straight, 1.0=natural
    click_jitter_px: int = 2             # max random offset on click target (px)
    timing_variance: float = 0.3         # per-delay random perturbation fraction
    overshoot_probability: float = 0.1   # chance of overshoot-then-correct
    overshoot_distance_px: int = 15      # max overshoot distance (screen px)
    micro_movement_enabled: bool = True  # hand tremor simulation before click
    micro_movement_amplitude_px: float = 0.8  # tremor amplitude (screen px)

    # Keyboard typing
    typing_speed: float = 1.0            # multiplier: <1 faster, >1 slower
    typing_variance: float = 0.3         # per-key delay variance fraction
    punctuation_pause_ms: float = 80.0   # extra ms after .,!?:;
    word_pause_ms: float = 40.0          # extra ms after space

    def __post_init__(self) -> None:
        if self.movement_speed <= 0:
            raise ValueError(f"movement_speed must be > 0, got {self.movement_speed}")
        if self.curve_noise < 0:
            raise ValueError(f"curve_noise must be >= 0, got {self.curve_noise}")
        if self.click_jitter_px < 0:
            raise ValueError(f"click_jitter_px must be >= 0, got {self.click_jitter_px}")
        if not 0.0 <= self.timing_variance <= 1.0:
            raise ValueError(f"timing_variance must be in [0, 1], got {self.timing_variance}")
        if not 0.0 <= self.overshoot_probability <= 1.0:
            raise ValueError(
                f"overshoot_probability must be in [0, 1], got {self.overshoot_probability}"
            )
        if self.overshoot_distance_px < 0:
            raise ValueError(
                f"overshoot_distance_px must be >= 0, got {self.overshoot_distance_px}"
            )
        if self.micro_movement_amplitude_px < 0:
            raise ValueError(
                f"micro_movement_amplitude_px must be >= 0, "
                f"got {self.micro_movement_amplitude_px}"
            )
        if self.typing_speed <= 0:
            raise ValueError(f"typing_speed must be > 0, got {self.typing_speed}")
        if not 0.0 <= self.typing_variance <= 1.0:
            raise ValueError(
                f"typing_variance must be in [0, 1], got {self.typing_variance}"
            )
        if self.punctuation_pause_ms < 0:
            raise ValueError(
                f"punctuation_pause_ms must be >= 0, got {self.punctuation_pause_ms}"
            )
        if self.word_pause_ms < 0:
            raise ValueError(f"word_pause_ms must be >= 0, got {self.word_pause_ms}")


# ---------------------------------------------------------------------------
# M5 Network
# ---------------------------------------------------------------------------

@dataclass
class NetworkConfig:
    """Network and Web UI configuration."""
    wifi_ssid: str = ""
    wifi_password: str = ""      # Security-sensitive: not persisted to YAML
    web_host: str = "0.0.0.0"
    web_port: int = 8000


# ---------------------------------------------------------------------------
# M5 BLE Provisioning
# ---------------------------------------------------------------------------

@dataclass
class BLEConfig:
    """BLE provisioning service configuration."""
    advertise_name: str = "CyberRaccoon"
    service_uuid: str = "12345678-1234-5678-1234-56789abcdef0"
    provisioning_timeout_s: float = 300.0
    auto_start: bool = True      # Auto-start provisioning when no Wi-Fi


# ---------------------------------------------------------------------------
# App-wide (aggregates all sub-configs)
# ---------------------------------------------------------------------------

@dataclass
class AppConfig:
    """Top-level configuration aggregating all module configs."""
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    executor: ExecutorConfig = field(default_factory=ExecutorConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    ble: BLEConfig = field(default_factory=BLEConfig)
    capture_source: str = "uvc"        # uvc | csi | airplay | picamera
    executor_transport: str = "bt"    # usb | bt
    target_os: str = ""               # "" (auto-detect) | windows | macos | linux


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_llm_config() -> LLMConfig:
    """Build LLMConfig by reading the active provider's values from the YAML
    config at ``~/.cyberraccoon/config.yaml``.

    If the YAML has no entry for the active provider, the built-in defaults
    for that provider (from :data:`LLM_PROVIDER_DEFAULTS`) are used instead,
    with an empty ``api_key``.  Callers should surface a clear error when
    ``api_key`` is empty — keys are managed exclusively via the Config tab
    of the web UI (or by hand-editing the YAML).
    """
    # Lazy import to avoid config ↔ config_store circular import.
    from cyberraccoon.ui.config_store import ConfigStore

    store = ConfigStore()
    config = store.load() if store.exists() else None
    if config is not None:
        return config.llm

    # No YAML yet → fall back to dataclass defaults for the requested provider.
    provider = "openai"
    defaults = LLM_PROVIDER_DEFAULTS.get(provider, {})
    return LLMConfig(
        provider=provider,
        model=defaults.get("model", ""),
        api_key="",
        base_url=defaults.get("base_url"),
        max_tokens=defaults.get("max_tokens", 1024),
        temperature=defaults.get("temperature", 0.0),
    )


def _load_llm_section() -> LLMConfig | None:
    """Internal: load ``AppConfig.llm`` from YAML, or ``None`` if no YAML yet."""
    from cyberraccoon.ui.config_store import ConfigStore
    store = ConfigStore()
    return store.load().llm if store.exists() else None


def resolve_api_key(provider: str) -> str:
    """Look up the API key for ``provider`` from the YAML config file.

    Returns ``""`` if no key is configured (neither the active provider's
    flat ``api_key`` nor the provider's snapshot under ``llm.providers``
    holds one).  API keys are configured exclusively via the Config tab
    of the web UI; there is no environment-variable fallback.
    """
    llm = _load_llm_section()
    if llm is None:
        return ""
    if provider == llm.provider and llm.api_key:
        return llm.api_key
    snap = llm.providers.get(provider) or {}
    return snap.get("api_key", "") or ""


def resolve_provider_model(provider: str) -> str:
    """Look up the model for ``provider`` from the YAML config file.

    Returns ``""`` if unset.
    """
    llm = _load_llm_section()
    if llm is None:
        return ""
    if provider == llm.provider and llm.model:
        return llm.model
    snap = llm.providers.get(provider) or {}
    return snap.get("model", "") or ""


def resolve_provider_base_url(provider: str) -> str | None:
    """Look up the base URL for ``provider`` from the YAML config file.

    Returns ``None`` if unset so callers can distinguish "not configured"
    from "explicitly empty".
    """
    llm = _load_llm_section()
    if llm is None:
        return None
    if provider == llm.provider:
        return llm.base_url or None
    snap = llm.providers.get(provider) or {}
    value = snap.get("base_url")
    return value if value else None


def load_capture_config() -> CaptureConfig:
    """Build CaptureConfig (all defaults for now)."""
    return CaptureConfig()


def load_agent_config() -> AgentConfig:
    """Build AgentConfig (all defaults for now)."""
    return AgentConfig()


def load_executor_config() -> ExecutorConfig:
    """Build ExecutorConfig (all defaults for now)."""
    return ExecutorConfig()


def _env_float(var: str, default: str) -> float:
    """Read an env var as float, raising ValueError with a clear message on failure."""
    raw = os.environ.get(var, default)
    try:
        return float(raw)
    except (ValueError, TypeError):
        raise ValueError(
            f"Invalid value {raw!r} for environment variable {var}: "
            f"expected a float (e.g. {default!r})"
        )


def _env_int(var: str, default: str) -> int:
    """Read an env var as int, raising ValueError with a clear message on failure."""
    raw = os.environ.get(var, default)
    try:
        return int(raw)
    except (ValueError, TypeError):
        raise ValueError(
            f"Invalid value {raw!r} for environment variable {var}: "
            f"expected an integer (e.g. {default!r})"
        )


def load_humanize_config() -> HumanizeConfig:
    """Build HumanizeConfig from environment variables."""
    enabled = os.environ.get("CYBERRACCOON_HUMANIZE", "0") == "1"
    micro_enabled_raw = os.environ.get("CYBERRACCOON_HUMANIZE_MICRO_ENABLED", "1")
    return HumanizeConfig(
        enabled=enabled,
        movement_speed=_env_float("CYBERRACCOON_HUMANIZE_SPEED", "1.0"),
        curve_noise=_env_float("CYBERRACCOON_HUMANIZE_NOISE", "1.0"),
        click_jitter_px=_env_int("CYBERRACCOON_HUMANIZE_JITTER", "2"),
        timing_variance=_env_float("CYBERRACCOON_HUMANIZE_VARIANCE", "0.3"),
        overshoot_probability=_env_float("CYBERRACCOON_HUMANIZE_OVERSHOOT_PROB", "0.1"),
        overshoot_distance_px=_env_int("CYBERRACCOON_HUMANIZE_OVERSHOOT_PX", "15"),
        micro_movement_enabled=micro_enabled_raw not in ("0", "false", "False"),
        micro_movement_amplitude_px=_env_float("CYBERRACCOON_HUMANIZE_MICRO_AMP", "0.8"),
        typing_speed=_env_float("CYBERRACCOON_HUMANIZE_TYPING_SPEED", "1.0"),
        typing_variance=_env_float("CYBERRACCOON_HUMANIZE_TYPING_VARIANCE", "0.3"),
        punctuation_pause_ms=_env_float("CYBERRACCOON_HUMANIZE_PUNCT_PAUSE", "80.0"),
        word_pause_ms=_env_float("CYBERRACCOON_HUMANIZE_WORD_PAUSE", "40.0"),
    )


# ---------------------------------------------------------------------------
# Humanization presets
# ---------------------------------------------------------------------------

HUMANIZE_PRESETS: dict[str, HumanizeConfig] = {
    "subtle": HumanizeConfig(
        enabled=True,
        curve_noise=0.3, click_jitter_px=1, timing_variance=0.15,
        overshoot_probability=0.05, micro_movement_enabled=False,
        typing_variance=0.15, punctuation_pause_ms=40.0, word_pause_ms=20.0,
    ),
    "normal": HumanizeConfig(enabled=True),
    "aggressive": HumanizeConfig(
        enabled=True,
        movement_speed=2.0, curve_noise=1.5, click_jitter_px=4,
        timing_variance=0.5, overshoot_probability=0.25,
        overshoot_distance_px=25, micro_movement_amplitude_px=1.5,
        typing_speed=1.5, typing_variance=0.5,
        punctuation_pause_ms=120.0, word_pause_ms=60.0,
    ),
}


def load_network_config() -> NetworkConfig:
    """Build NetworkConfig from environment variables."""
    return NetworkConfig(
        wifi_ssid=os.environ.get("CYBERRACCOON_WIFI_SSID", ""),
        wifi_password=os.environ.get("CYBERRACCOON_WIFI_PASSWORD", ""),
        web_host=os.environ.get("CYBERRACCOON_WEB_HOST", "0.0.0.0"),
        web_port=int(os.environ.get("CYBERRACCOON_WEB_PORT", "8000")),
    )


def load_ble_config() -> BLEConfig:
    """Build BLEConfig from environment variables."""
    return BLEConfig(
        advertise_name=os.environ.get("CYBERRACCOON_BLE_NAME", "CyberRaccoon"),
    )


def load_app_config() -> AppConfig:
    """Build complete AppConfig from environment variables.

    This is the environment-variable layer only. For full 3-tier loading
    (defaults → YAML → env), use ``ConfigStore.load()`` instead.
    """
    return AppConfig(
        capture=load_capture_config(),
        llm=load_llm_config(),
        agent=load_agent_config(),
        executor=load_executor_config(),
        network=load_network_config(),
        ble=load_ble_config(),
        capture_source=os.environ.get("CYBERRACCOON_SOURCE", "uvc"),
        executor_transport=os.environ.get("CYBERRACCOON_TRANSPORT", "bt"),
        target_os=os.environ.get("CYBERRACCOON_TARGET_OS", ""),
    )
