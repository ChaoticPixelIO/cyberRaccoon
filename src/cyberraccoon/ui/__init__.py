"""M5 User Interface — configuration, Wi-Fi, AppController, CLI, Web UI.

Exports for Phases 1-3: exceptions, config store, Wi-Fi manager,
AppController with event system, CLI REPL, and Web server factory.
"""

from ui.exceptions import (
    ConfigError,
    M5Error,
    ProvisioningError,
    TaskError,
    WiFiError,
)
from ui.config_store import ConfigStore
from ui.wifi_manager import WiFiManager, WiFiNetwork
from ui.app_controller import (
    AppController,
    AppEvent,
    AppEventType,
    AppEventListener,
    LogCaptureHandler,
)

__all__ = [
    # Exceptions
    "M5Error",
    "ConfigError",
    "ProvisioningError",
    "WiFiError",
    "TaskError",
    # Config
    "ConfigStore",
    # Wi-Fi
    "WiFiManager",
    "WiFiNetwork",
    # AppController
    "AppController",
    "AppEvent",
    "AppEventType",
    "AppEventListener",
    "LogCaptureHandler",
]
