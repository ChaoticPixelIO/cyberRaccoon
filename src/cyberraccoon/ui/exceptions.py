"""M5 User Interface — custom exceptions.

Each exception class covers a distinct failure domain within M5.
All inherit from ``M5Error`` so callers can catch the entire module
with a single except clause when appropriate.
"""


class M5Error(Exception):
    """Base exception for M5 module."""


class ConfigError(M5Error):
    """Configuration errors: invalid YAML, validation failure, unwritable path."""


class ProvisioningError(M5Error):
    """BLE provisioning errors: Bluetooth unavailable, GATT failure, timeout."""


class WiFiError(M5Error):
    """Wi-Fi management errors: connection failed, backend unavailable, timeout."""


class TaskError(M5Error):
    """Task control errors: task already running, modules not initialised."""


class NoPlanCachedError(M5Error):
    """Raised when chat_about_plan / request_plan_rewrite is called but no
    plan is currently pending. Distinct from "LLM call failed" so the HTTP
    boundary can return 409 Conflict (user error — retry won't help) instead
    of 503 Service Unavailable (transient — retry might).
    """


class ConfigPersistError(M5Error):
    """Raised when set_auto_replan succeeds in updating in-memory state but
    fails to persist to disk. Endpoint can return 207 Multi-Status so the
    frontend's existing fallback toast ("config write failed — will not
    persist across restart") fires instead of being shown a green 200.
    """
