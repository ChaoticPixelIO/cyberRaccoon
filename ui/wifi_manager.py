"""M5 Wi-Fi Manager — system Wi-Fi management via nmcli or wpa_supplicant.

Provides a unified interface for Wi-Fi operations used by both BLE
provisioning (Sub1) and the CLI REPL (Sub5).

Usage::

    wm = WiFiManager()
    networks = wm.scan()
    wm.connect("MySSID", "password123")
    print(wm.get_ip_address())   # "192.168.1.42"
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass

from ui.exceptions import WiFiError

logger = logging.getLogger("M5.wifi")

_CMD_TIMEOUT = 30  # seconds


@dataclass
class WiFiNetwork:
    """A Wi-Fi network discovered during scanning."""

    ssid: str
    signal_strength: int     # dBm (negative, e.g. -45)
    security: str            # "WPA2", "WPA3", "Open", etc.
    connected: bool          # Whether this network is currently connected


class WiFiManager:
    """Wi-Fi connection manager.

    Detects the available backend on construction:
        - ``nmcli`` (NetworkManager) — preferred
        - ``wpa_supplicant`` — fallback

    All public methods raise ``WiFiError`` on failure.
    """

    def __init__(self) -> None:
        self._backend = self._detect_backend()
        logger.info("WiFiManager using backend: %s", self._backend)

    @property
    def backend(self) -> str:
        """Return the detected backend name."""
        return self._backend

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self, ssid: str, password: str) -> bool:
        """Connect to a Wi-Fi network.

        Args:
            ssid: Network SSID.
            password: Network password (empty string for open networks).

        Returns:
            ``True`` if connection succeeded.

        Raises:
            WiFiError: On connection failure or unsupported backend.
        """
        if not ssid:
            raise WiFiError("SSID must not be empty")

        logger.info("Connecting to Wi-Fi: %s", ssid)

        if self._backend == "networkmanager":
            return self._nmcli_connect(ssid, password)
        elif self._backend == "wpa_supplicant":
            return self._wpa_connect(ssid, password)
        else:
            raise WiFiError(f"Unsupported backend: {self._backend}")

    def disconnect(self) -> bool:
        """Disconnect from the current Wi-Fi network.

        Returns:
            ``True`` if disconnection succeeded.
        """
        if self._backend == "networkmanager":
            result = self._run_cmd(
                ["nmcli", "device", "disconnect", "wlan0"],
            )
            return result.returncode == 0
        return False

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def is_connected(self) -> bool:
        """Return ``True`` if currently connected to a Wi-Fi network."""
        if self._backend == "networkmanager":
            result = self._run_cmd(
                ["nmcli", "-t", "-f", "TYPE,STATE",
                 "device", "status"],
            )
            if result.returncode != 0:
                return False
            for line in result.stdout.strip().splitlines():
                parts = line.split(":")
                if len(parts) >= 2 and parts[0] == "wifi" and parts[1] == "connected":
                    return True
            return False

        # Fallback: check if wlan0 has an IP
        return self.get_ip_address() is not None

    def get_current_network(self) -> str | None:
        """Return the SSID of the currently connected network, or ``None``."""
        if self._backend == "networkmanager":
            result = self._run_cmd(
                ["nmcli", "-t", "-f", "active,ssid",
                 "device", "wifi", "list"],
            )
            if result.returncode != 0:
                return None
            for line in result.stdout.strip().splitlines():
                parts = line.split(":", 1)
                if len(parts) >= 2 and parts[0] == "yes":
                    return parts[1]
        return None

    def get_ip_address(self) -> str | None:
        """Return the IP address of the wlan0 interface, or ``None``."""
        if self._backend == "networkmanager":
            result = self._run_cmd(
                ["nmcli", "-t", "-f", "IP4.ADDRESS",
                 "device", "show", "wlan0"],
            )
            if result.returncode != 0:
                return None
            for line in result.stdout.strip().splitlines():
                if line.startswith("IP4.ADDRESS"):
                    # Format: "IP4.ADDRESS[1]:192.168.1.42/24"
                    _, _, addr = line.partition(":")
                    if "/" in addr:
                        return addr.split("/")[0]
                    return addr if addr else None
            return None

        # Fallback: use ip command
        result = self._run_cmd(
            ["ip", "-4", "-o", "addr", "show", "wlan0"],
        )
        if result.returncode != 0 or not result.stdout.strip():
            return None
        # Format: "3: wlan0  inet 192.168.1.42/24 ..."
        parts = result.stdout.strip().split()
        for i, part in enumerate(parts):
            if part == "inet" and i + 1 < len(parts):
                addr = parts[i + 1]
                return addr.split("/")[0]
        return None

    # ------------------------------------------------------------------
    # Scanning
    # ------------------------------------------------------------------

    def scan(self) -> list[WiFiNetwork]:
        """Scan for nearby Wi-Fi networks.

        Returns:
            List of discovered networks, sorted by signal strength.
        """
        if self._backend == "networkmanager":
            return self._nmcli_scan()
        return []

    def has_saved_network(self) -> bool:
        """Return ``True`` if there are saved Wi-Fi connections."""
        if self._backend == "networkmanager":
            result = self._run_cmd(
                ["nmcli", "-t", "-f", "TYPE,NAME",
                 "connection", "show"],
            )
            if result.returncode != 0:
                return False
            for line in result.stdout.strip().splitlines():
                parts = line.split(":", 1)
                if len(parts) >= 2 and "wireless" in parts[0]:
                    return True
            return False

        # Fallback: check wpa_supplicant config
        wpa_conf = "/etc/wpa_supplicant/wpa_supplicant.conf"
        try:
            text = open(wpa_conf).read()
            return "ssid=" in text
        except OSError:
            return False

    # ------------------------------------------------------------------
    # Backend detection
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_backend() -> str:
        """Detect the available Wi-Fi management backend.

        Returns:
            ``"networkmanager"`` or ``"wpa_supplicant"``.

        Raises:
            WiFiError: If no backend is available.
        """
        if shutil.which("nmcli"):
            return "networkmanager"
        if shutil.which("wpa_supplicant") or shutil.which("wpa_cli"):
            return "wpa_supplicant"
        raise WiFiError(
            "No Wi-Fi backend available. "
            "Install NetworkManager (apt install network-manager) "
            "or ensure wpa_supplicant is available."
        )

    # ------------------------------------------------------------------
    # nmcli implementation
    # ------------------------------------------------------------------

    def _nmcli_connect(self, ssid: str, password: str) -> bool:
        """Connect via nmcli."""
        cmd = ["nmcli", "device", "wifi", "connect", ssid]
        if password:
            cmd.extend(["password", password])

        result = self._run_cmd(cmd)
        if result.returncode != 0:
            error_msg = result.stderr.strip() or result.stdout.strip()
            raise WiFiError(f"nmcli connect failed: {error_msg}")
        logger.info("Connected to Wi-Fi: %s", ssid)
        return True

    def _nmcli_scan(self) -> list[WiFiNetwork]:
        """Scan via nmcli."""
        result = self._run_cmd(
            ["nmcli", "-t", "-f", "SSID,SIGNAL,SECURITY,ACTIVE",
             "device", "wifi", "list", "--rescan", "yes"],
        )
        if result.returncode != 0:
            logger.warning("Wi-Fi scan failed: %s", result.stderr.strip())
            return []

        networks: list[WiFiNetwork] = []
        seen_ssids: set[str] = set()

        for line in result.stdout.strip().splitlines():
            parts = line.split(":")
            if len(parts) < 4:
                continue

            ssid = parts[0].strip()
            if not ssid or ssid in seen_ssids:
                continue  # skip duplicates and hidden networks
            seen_ssids.add(ssid)

            try:
                signal = int(parts[1])
            except ValueError:
                signal = -100

            security = parts[2] if parts[2] else "Open"
            active = parts[3].strip().lower() == "yes"

            networks.append(WiFiNetwork(
                ssid=ssid,
                signal_strength=signal,
                security=security,
                connected=active,
            ))

        # Sort by signal strength (strongest first)
        networks.sort(key=lambda n: n.signal_strength, reverse=True)
        return networks

    # ------------------------------------------------------------------
    # wpa_supplicant implementation (basic)
    # ------------------------------------------------------------------

    def _wpa_connect(self, ssid: str, password: str) -> bool:
        """Connect via wpa_supplicant (basic implementation).

        This is a minimal fallback. For production use, NetworkManager
        is strongly recommended.
        """
        wpa_conf = "/etc/wpa_supplicant/wpa_supplicant.conf"

        # Generate network block
        if password:
            # Use wpa_passphrase to generate PSK
            result = self._run_cmd(
                ["wpa_passphrase", ssid, password],
            )
            if result.returncode != 0:
                raise WiFiError(f"wpa_passphrase failed: {result.stderr.strip()}")
            network_block = result.stdout
        else:
            network_block = f'network={{\n    ssid="{ssid}"\n    key_mgmt=NONE\n}}\n'

        # Append to wpa_supplicant.conf
        try:
            with open(wpa_conf, "a") as f:
                f.write("\n" + network_block)
        except OSError as e:
            raise WiFiError(f"Cannot write {wpa_conf}: {e}") from e

        # Reconfigure
        result = self._run_cmd(["wpa_cli", "-i", "wlan0", "reconfigure"])
        if result.returncode != 0:
            raise WiFiError(f"wpa_cli reconfigure failed: {result.stderr.strip()}")

        logger.info("Connected to Wi-Fi via wpa_supplicant: %s", ssid)
        return True

    # ------------------------------------------------------------------
    # Subprocess helper
    # ------------------------------------------------------------------

    @staticmethod
    def _run_cmd(
        cmd: list[str],
        timeout: int = _CMD_TIMEOUT,
    ) -> subprocess.CompletedProcess[str]:
        """Run a subprocess command with timeout and error handling."""
        try:
            return subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as e:
            raise WiFiError(
                f"Command timed out after {timeout}s: {' '.join(cmd)}"
            ) from e
        except FileNotFoundError as e:
            raise WiFiError(f"Command not found: {cmd[0]}") from e
