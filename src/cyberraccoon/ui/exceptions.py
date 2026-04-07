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
