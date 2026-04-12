"""Bluetooth HID transport layer — L2CAP connection management.

Registers a Bluetooth HID Profile via BlueZ D-Bus API, listens for
incoming L2CAP connections on PSM 17 (Control) and PSM 19 (Interrupt),
and sends HID reports to the connected host.

BluetoothHIDDevice wraps the connection with the same open/write/close
interface as HIDDevice, so KeyboardController and MouseController
can be used without modification.

Usage::

    conn = BluetoothHIDConnection()
    conn.setup()
    conn.wait_for_connection()   # blocks until host connects

    kb_dev = BluetoothHIDDevice(conn, report_id=0x01)
    ms_dev = BluetoothHIDDevice(conn, report_id=0x02)

    keyboard = KeyboardController(kb_dev)
    mouse = MouseController(ms_dev)

    conn.disconnect()
"""

from __future__ import annotations

import logging
import os
import re
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from cyberraccoon.executor.hid_device import HIDDeviceError

logger = logging.getLogger("M4.bluetooth")

# L2CAP channel numbers for HID
_PSM_CTRL = 0x0011   # HID Control (PSM 17)
_PSM_INTR = 0x0013   # HID Interrupt (PSM 19)

# BDADDR_ANY — explicit form required by Python 3.13+
_BDADDR_ANY = "00:00:00:00:00:00"

# HID report header byte (DATA | Input)
_REPORT_HEADER = 0xA1

# SDP record file (relative to this module)
_SDP_RECORD_PATH = Path(__file__).parent / "sdp_record.xml"


class BluetoothHIDConnection:
    """Manages a Bluetooth HID connection lifecycle.

    Handles:
    - BlueZ D-Bus profile registration with SDP record
    - L2CAP socket setup (PSM 17 + PSM 19)
    - Connection accept and management
    - HID report transmission
    """

    def __init__(
        self,
        device_name: str = "CyberRaccoon",
        device_class: int = 0x002540,
    ) -> None:
        self._device_name = device_name
        self._device_class = device_class
        self._ctrl_sock: socket.socket | None = None
        self._intr_sock: socket.socket | None = None
        self._ctrl_client: socket.socket | None = None
        self._intr_client: socket.socket | None = None
        self._dbus_profile: Any = None
        self._dbus_agent: Any = None
        self._glib_loop: Any = None
        self._glib_thread: threading.Thread | None = None
        self._connected = False
        self._remote_addr: str = ""

    def setup(self) -> None:
        """Register HID profile with BlueZ and prepare L2CAP sockets.

        Raises HIDDeviceError if Bluetooth is unavailable or setup fails.
        """
        try:
            import dbus
            import dbus.mainloop.glib
            import dbus.service
        except ImportError as e:
            raise HIDDeviceError(
                "dbus-python is not available. "
                "Install with: sudo apt install python3-dbus"
            ) from e

        # Initialize D-Bus GLib main loop (required for async callbacks)
        dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)

        logger.info("Setting up Bluetooth HID profile...")

        # Register HID Profile via D-Bus (must come BEFORE adapter config
        # because BlueZ overrides device class during profile registration)
        self._register_profile(dbus)

        # Configure Bluetooth adapter AFTER profile registration
        # so our device class (0x002540) isn't overridden by BlueZ
        self._configure_adapter()

        # Register pairing agent (auto-accept pairing requests)
        self._register_agent(dbus)

        # Create L2CAP listening sockets
        self._create_sockets()

        # Start GLib main loop in background thread for D-Bus callbacks
        self._start_glib_loop()

        logger.info("Bluetooth HID setup complete, waiting for connection")

    def _start_glib_loop(self) -> None:
        """Start GLib main loop in a background thread.

        Required for D-Bus agent callbacks (pairing confirmation)
        to work while the main thread blocks on socket.accept().
        """
        try:
            from gi.repository import GLib
            self._glib_loop = GLib.MainLoop()

            def run_loop() -> None:
                try:
                    self._glib_loop.run()
                except Exception as e:
                    logger.debug("GLib loop exited: %s", e)

            self._glib_thread = threading.Thread(
                target=run_loop, daemon=True, name="glib-loop",
            )
            self._glib_thread.start()
            logger.info("GLib main loop started for D-Bus callbacks")
        except ImportError:
            logger.warning(
                "gi.repository not available; D-Bus callbacks may not work. "
                "Install with: sudo apt install python3-gi"
            )

    def _configure_adapter(self) -> None:
        """Set Bluetooth adapter name, class, and discoverable mode.

        BlueZ often overrides device class after profile registration,
        so we set it here (called after _register_profile) and verify.
        """
        try:
            class_hex = f"0x{self._device_class:06x}"

            # Set device class (keyboard + mouse combo) — retry up to 3 times
            for attempt in range(3):
                subprocess.run(
                    ["hciconfig", "hci0", "class", class_hex],
                    check=True, capture_output=True, timeout=5,
                )
                # Verify the class was actually set
                result = subprocess.run(
                    ["hciconfig", "hci0", "class"],
                    capture_output=True, text=True, timeout=5,
                )
                if class_hex.lower() in result.stdout.lower():
                    break
                time.sleep(0.2)

            # Set device name via hciconfig
            subprocess.run(
                ["hciconfig", "hci0", "name", self._device_name],
                check=True, capture_output=True, timeout=5,
            )

            # Also set alias via bluetoothctl (more reliable for name)
            subprocess.run(
                ["bluetoothctl", "system-alias", self._device_name],
                capture_output=True, timeout=5,
            )

            # Make discoverable and pairable
            subprocess.run(
                ["hciconfig", "hci0", "piscan"],
                check=True, capture_output=True, timeout=5,
            )

            # Final class check
            result = subprocess.run(
                ["hciconfig", "hci0", "class"],
                capture_output=True, text=True, timeout=5,
            )
            logger.info(
                "Adapter configured: name=%s class=%s (actual: %s)",
                self._device_name, class_hex,
                result.stdout.strip().split("Class: ")[-1].split()[0]
                if "Class:" in result.stdout else "unknown",
            )
        except FileNotFoundError:
            raise HIDDeviceError(
                "hciconfig not found. Install with: sudo apt install bluez"
            )
        except subprocess.CalledProcessError as e:
            stderr = e.stderr.decode()
            if "Operation not permitted" in stderr or "not permitted" in stderr.lower():
                logger.warning(
                    "Adapter config skipped (no CAP_NET_ADMIN): %s. "
                    "Run 'sudo scripts/setup.sh --bt' once to pre-configure.",
                    stderr.strip(),
                )
            else:
                raise HIDDeviceError(
                    f"Failed to configure Bluetooth adapter: {stderr}"
                )

    def _register_profile(self, dbus_module: Any) -> None:
        """Register HID profile with BlueZ ProfileManager1."""
        bus = dbus_module.SystemBus()

        # Read SDP record
        if not _SDP_RECORD_PATH.exists():
            raise HIDDeviceError(
                f"SDP record not found: {_SDP_RECORD_PATH}. "
                "Ensure executor/sdp_record.xml exists."
            )
        sdp_record = _SDP_RECORD_PATH.read_text()

        # Register profile
        manager = dbus_module.Interface(
            bus.get_object("org.bluez", "/org/bluez"),
            "org.bluez.ProfileManager1",
        )

        profile_path = "/cyberraccoon/hid/profile"
        hid_uuid = "00001124-0000-1000-8000-00805f9b34fb"

        opts = {
            "ServiceRecord": sdp_record,
            "Role": "server",
            "RequireAuthentication": dbus_module.Boolean(False),
            "RequireAuthorization": dbus_module.Boolean(False),
        }

        try:
            manager.RegisterProfile(profile_path, hid_uuid, opts)
            logger.info("HID profile registered with BlueZ")
        except Exception as e:
            error_msg = str(e)
            if "Already Exists" in error_msg or "already registered" in error_msg.lower():
                logger.info("HID profile already registered")
            else:
                raise HIDDeviceError(
                    f"Failed to register HID profile: {e}"
                ) from e

    def _register_agent(self, dbus_module: Any) -> None:
        """Register a NoInputNoOutput pairing agent with BlueZ.

        This agent auto-accepts all pairing requests so the target
        computer can pair without manual confirmation on the Pi side.
        """
        import dbus.service

        bus = dbus_module.SystemBus()

        agent_path = "/cyberraccoon/hid/agent"

        # Define agent class inline to keep D-Bus dependency contained
        class PairingAgent(dbus.service.Object):
            """BlueZ pairing agent that auto-accepts all requests."""

            AGENT_INTERFACE = "org.bluez.Agent1"

            @dbus.service.method(AGENT_INTERFACE, in_signature="", out_signature="")
            def Release(self):
                logger.debug("Agent released")

            @dbus.service.method(AGENT_INTERFACE, in_signature="os", out_signature="")
            def AuthorizeService(self, device, uuid):
                logger.info("Authorizing service %s for %s", uuid, device)

            @dbus.service.method(AGENT_INTERFACE, in_signature="o", out_signature="s")
            def RequestPinCode(self, device):
                logger.info("RequestPinCode from %s, returning '0000'", device)
                return "0000"

            @dbus.service.method(AGENT_INTERFACE, in_signature="o", out_signature="u")
            def RequestPasskey(self, device):
                logger.info("RequestPasskey from %s, returning 0", device)
                return dbus_module.UInt32(0)

            @dbus.service.method(AGENT_INTERFACE, in_signature="ouq", out_signature="")
            def DisplayPasskey(self, device, passkey, entered):
                logger.info("DisplayPasskey: %s %06d", device, passkey)

            @dbus.service.method(AGENT_INTERFACE, in_signature="os", out_signature="")
            def DisplayPinCode(self, device, pincode):
                logger.info("DisplayPinCode: %s %s", device, pincode)

            @dbus.service.method(AGENT_INTERFACE, in_signature="ou", out_signature="")
            def RequestConfirmation(self, device, passkey):
                logger.info(
                    "Auto-confirming pairing with %s (passkey %06d)",
                    device, passkey,
                )
                # Return without error = confirmation accepted

            @dbus.service.method(AGENT_INTERFACE, in_signature="o", out_signature="")
            def RequestAuthorization(self, device):
                logger.info("Auto-authorizing %s", device)

            @dbus.service.method(AGENT_INTERFACE, in_signature="", out_signature="")
            def Cancel(self):
                logger.debug("Agent pairing cancelled")

        try:
            self._dbus_agent = PairingAgent(bus, agent_path)

            agent_manager = dbus_module.Interface(
                bus.get_object("org.bluez", "/org/bluez"),
                "org.bluez.AgentManager1",
            )

            try:
                agent_manager.RegisterAgent(agent_path, "NoInputNoOutput")
            except Exception as e:
                if "Already Exists" in str(e):
                    agent_manager.UnregisterAgent(agent_path)
                    agent_manager.RegisterAgent(agent_path, "NoInputNoOutput")
                else:
                    raise

            agent_manager.RequestDefaultAgent(agent_path)
            logger.info("Pairing agent registered (auto-accept mode)")

        except Exception as e:
            logger.warning("Failed to register pairing agent: %s", e)
            logger.warning("Manual pairing confirmation may be required")

    def _create_sockets(self) -> None:
        """Create L2CAP listening sockets for HID Control and Interrupt."""
        try:
            # Control channel (PSM 17)
            self._ctrl_sock = socket.socket(
                socket.AF_BLUETOOTH,
                socket.SOCK_SEQPACKET,
                socket.BTPROTO_L2CAP,
            )
            self._ctrl_sock.setsockopt(
                socket.SOL_SOCKET, socket.SO_REUSEADDR, 1
            )
            self._ctrl_sock.bind((_BDADDR_ANY, _PSM_CTRL))
            self._ctrl_sock.listen(1)

            # Interrupt channel (PSM 19)
            self._intr_sock = socket.socket(
                socket.AF_BLUETOOTH,
                socket.SOCK_SEQPACKET,
                socket.BTPROTO_L2CAP,
            )
            self._intr_sock.setsockopt(
                socket.SOL_SOCKET, socket.SO_REUSEADDR, 1
            )
            self._intr_sock.bind((_BDADDR_ANY, _PSM_INTR))
            self._intr_sock.listen(1)

            logger.info(
                "L2CAP sockets listening: Control(PSM %d) Interrupt(PSM %d)",
                _PSM_CTRL, _PSM_INTR,
            )
        except AttributeError:
            raise HIDDeviceError(
                "Bluetooth sockets not supported on this platform. "
                "AF_BLUETOOTH requires Linux with BlueZ."
            )
        except OSError as e:
            if e.errno == 13:  # EACCES
                raise HIDDeviceError(
                    f"Failed to create L2CAP sockets: {e}. "
                    "Python needs CAP_NET_BIND_SERVICE to bind PSM 17/19. "
                    "Fix: sudo scripts/setup.sh --bt"
                )
            if e.errno == 98:  # EADDRINUSE
                self._kill_stale_l2cap_holder()
                # Retry once after killing
                try:
                    self._create_sockets()
                    return
                except OSError:
                    pass  # fall through to the generic error below
            raise HIDDeviceError(
                f"Failed to create L2CAP sockets: {e}. "
                "Ensure bluetoothd is running with input plugin disabled."
            )

    def _kill_stale_l2cap_holder(self) -> None:
        """Kill a stale CyberRaccoon process holding L2CAP PSM 17/19 sockets.

        Only kills Python processes whose command line contains ``cyberraccoon``
        to avoid accidentally terminating unrelated Bluetooth services.
        """
        try:
            result = subprocess.run(
                ["ss", "-lbp", "src", f":{_PSM_CTRL}"],
                capture_output=True, text=True, timeout=5,
            )
            for line in result.stdout.splitlines():
                match = re.search(r"pid=(\d+)", line)
                if not match:
                    continue
                pid = int(match.group(1))
                if pid == os.getpid():
                    continue
                # Only kill if it's our own stale process
                try:
                    cmdline = Path(f"/proc/{pid}/cmdline").read_text()
                except OSError:
                    continue
                if "cyberraccoon" not in cmdline:
                    logger.warning(
                        "PID %d holds L2CAP sockets but is not ours, skipping", pid,
                    )
                    continue
                logger.warning(
                    "Killing stale CyberRaccoon process %d holding L2CAP sockets", pid,
                )
                os.kill(pid, 9)
            time.sleep(1)  # wait for sockets to be released
        except Exception as e:
            logger.warning("Failed to kill stale L2CAP holder: %s", e)

    def wait_for_connection(self, timeout: float = 60.0) -> None:
        """Connect to a paired host or wait for a new host to pair.

        Uses a dual strategy:

        1. **Outbound** — tries to connect to already-paired devices by
           initiating L2CAP connections to their PSM 17/19.  This handles
           reconnections (macOS only initiates HID channels on first pair;
           subsequent reconnects must be device-initiated).

        2. **Inbound** — listens on L2CAP sockets for a new host to pair
           and connect.  This handles first-time pairing.

        Both strategies run concurrently: outbound is attempted every few
        seconds while the inbound accept runs with a short timeout.

        Args:
            timeout: Maximum seconds to wait for connection.

        Raises:
            HIDDeviceError: If timeout expires or connection fails.
        """
        if not self._ctrl_sock or not self._intr_sock:
            raise HIDDeviceError("Sockets not created. Call setup() first.")

        logger.info(
            "Waiting for Bluetooth HID connection (timeout=%ds)...", timeout
        )

        # Try outbound connection to paired devices first
        paired_addrs = self._get_paired_device_addresses()
        if paired_addrs:
            logger.info(
                "Found paired device(s): %s — trying outbound connection",
                ", ".join(paired_addrs),
            )
            for addr in paired_addrs:
                if self._try_outbound_connect(addr):
                    return
        else:
            logger.info(
                "No paired devices. Pair '%s' from the target computer.",
                self._device_name,
            )

        # Fall back to inbound accept with periodic outbound retries
        deadline = time.monotonic() + timeout
        poll_interval = 5.0  # seconds between outbound retries

        self._ctrl_sock.settimeout(poll_interval)
        self._intr_sock.settimeout(poll_interval)

        while time.monotonic() < deadline:
            # Try inbound accept
            try:
                self._ctrl_client, ctrl_info = self._ctrl_sock.accept()
                logger.info("Control channel connected from %s", ctrl_info)

                self._intr_sock.settimeout(10.0)
                self._intr_client, intr_info = self._intr_sock.accept()
                logger.info("Interrupt channel connected from %s", intr_info)

                self._connected = True
                self._remote_addr = ctrl_info[0] if ctrl_info else ""
                logger.info("Bluetooth HID connection established (inbound from %s)!", self._remote_addr)
                self._handle_hidp_handshake()
                return

            except socket.timeout:
                pass
            except OSError as e:
                logger.debug("Inbound accept error: %s", e)

            # Retry outbound to paired devices
            paired_addrs = self._get_paired_device_addresses()
            for addr in paired_addrs:
                if self._try_outbound_connect(addr):
                    return

        raise HIDDeviceError(
            f"Bluetooth connection timed out after {timeout}s. "
            f"Ensure the target computer is pairing with '{self._device_name}'."
        )

    def _try_outbound_connect(self, host_addr: str) -> bool:
        """Try to initiate L2CAP HID connections to a paired host.

        Args:
            host_addr: Bluetooth MAC address of the host (e.g. "AA:BB:CC:DD:EE:FF").

        Returns:
            True if both channels connected successfully.
        """
        try:
            ctrl = socket.socket(
                socket.AF_BLUETOOTH,
                socket.SOCK_SEQPACKET,
                socket.BTPROTO_L2CAP,
            )
            ctrl.settimeout(3.0)
            ctrl.connect((host_addr, _PSM_CTRL))
            logger.info("Outbound control channel connected to %s", host_addr)

            intr = socket.socket(
                socket.AF_BLUETOOTH,
                socket.SOCK_SEQPACKET,
                socket.BTPROTO_L2CAP,
            )
            intr.settimeout(3.0)
            intr.connect((host_addr, _PSM_INTR))
            logger.info("Outbound interrupt channel connected to %s", host_addr)

            self._ctrl_client = ctrl
            self._intr_client = intr
            self._connected = True
            self._remote_addr = host_addr
            logger.info("Bluetooth HID connection established (outbound to %s)!", host_addr)
            self._handle_hidp_handshake()
            return True

        except (socket.timeout, OSError) as e:
            logger.debug("Outbound connect to %s failed: %s", host_addr, e)
            return False

    @staticmethod
    def _get_paired_devices() -> list[tuple[str, str]]:
        """Get paired devices as (address, name) tuples via bluetoothctl."""
        try:
            result = subprocess.run(
                ["bluetoothctl", "devices"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode != 0:
                return []

            devices = []
            for line in result.stdout.strip().splitlines():
                # Format: "Device AA:BB:CC:DD:EE:FF Name With Spaces"
                parts = line.strip().split(None, 2)
                if len(parts) >= 3 and parts[0] == "Device":
                    devices.append((parts[1], parts[2]))
                elif len(parts) == 2 and parts[0] == "Device":
                    devices.append((parts[1], parts[1]))
            return devices

        except (FileNotFoundError, subprocess.TimeoutExpired):
            return []

    @staticmethod
    def _get_paired_device_addresses() -> list[str]:
        """Get Bluetooth addresses of paired devices via bluetoothctl."""
        return [addr for addr, _ in BluetoothHIDConnection._get_paired_devices()]

    def _handle_hidp_handshake(self) -> None:
        """Handle initial HIDP messages from the host after connection.

        macOS sends SET_PROTOCOL (0x71 = Report Mode) immediately after
        L2CAP connection. We must reply with HANDSHAKE successful (0x00)
        or macOS will disconnect after a few seconds.

        Also starts a background thread to handle subsequent control
        channel messages (GET_REPORT, SET_REPORT, etc.).
        """
        import select

        if not self._ctrl_client:
            return

        # HIDP message types (high nibble of header byte)
        _HANDSHAKE = 0x0
        _SET_PROTOCOL = 0x7
        _HANDSHAKE_OK = bytes([0x00])

        # Wait up to 1 second for initial control messages
        for _ in range(10):
            readable, _, _ = select.select([self._ctrl_client], [], [], 0.1)
            if not readable:
                continue

            try:
                data = self._ctrl_client.recv(1024)
                if not data:
                    break

                hdr = data[0]
                msg_type = (hdr >> 4) & 0x0F
                param = hdr & 0x0F
                logger.info(
                    "HIDP control message: type=0x%x param=0x%x data=%s",
                    msg_type, param, data.hex(),
                )

                if msg_type == _SET_PROTOCOL:
                    self._ctrl_client.send(_HANDSHAKE_OK)
                    logger.info(
                        "SET_PROTOCOL(%s) acknowledged",
                        "Report" if param == 1 else "Boot",
                    )
                else:
                    # Reply HANDSHAKE OK to any unhandled message
                    self._ctrl_client.send(_HANDSHAKE_OK)
                    logger.info("Replied HANDSHAKE OK to type=0x%x", msg_type)

            except OSError as e:
                logger.warning("Error reading control channel: %s", e)
                break

        # Start background thread for ongoing control channel messages
        self._start_ctrl_listener()

    def _start_ctrl_listener(self) -> None:
        """Listen for control channel messages in a background thread."""
        import select

        _HANDSHAKE_OK = bytes([0x00])

        def listen() -> None:
            while self._connected and self._ctrl_client:
                try:
                    sock = self._ctrl_client
                    if sock is None:
                        break

                    readable, _, _ = select.select([sock], [], [], 1.0)
                    if not readable:
                        continue

                    # Re-check after select() — close() may have set it to None
                    if self._ctrl_client is None:
                        break

                    data = sock.recv(1024)
                    if not data:
                        logger.info("Control channel closed by host")
                        self._connected = False
                        break

                    hdr = data[0]
                    msg_type = (hdr >> 4) & 0x0F
                    logger.debug(
                        "HIDP ctrl: type=0x%x data=%s", msg_type, data.hex(),
                    )
                    # Reply HANDSHAKE OK to all control messages
                    sock.send(_HANDSHAKE_OK)

                except OSError:
                    self._connected = False
                    break

        thread = threading.Thread(
            target=listen, daemon=True, name="hidp-ctrl",
        )
        thread.start()

    def send_report(self, report_id: int, data: bytes) -> None:
        """Send an HID report on the interrupt channel.

        Args:
            report_id: Report ID (0x01 for keyboard, 0x02 for mouse).
            data: Raw HID report bytes.

        Raises:
            HIDDeviceError: If not connected or send fails.
        """
        if not self._connected or not self._intr_client:
            raise HIDDeviceError(
                "Bluetooth HID not connected. Call wait_for_connection() first."
            )

        # Format: 0xA1 (DATA|Input) + Report ID + report data
        packet = bytes([_REPORT_HEADER, report_id]) + data

        try:
            self._intr_client.send(packet)
        except OSError as e:
            self._connected = False
            raise HIDDeviceError(
                f"Failed to send Bluetooth HID report: {e}"
            )

    def disconnect(self) -> None:
        """Close all sockets, stop GLib loop, and clean up."""
        self._connected = False

        # Stop GLib main loop (may not exist if setup() was never called)
        glib_loop = getattr(self, "_glib_loop", None)
        if glib_loop and glib_loop.is_running():
            glib_loop.quit()
        self._glib_loop = None
        self._glib_thread = None

        for name, sock in [
            ("intr_client", self._intr_client),
            ("ctrl_client", self._ctrl_client),
            ("intr_sock", self._intr_sock),
            ("ctrl_sock", self._ctrl_sock),
        ]:
            if sock:
                try:
                    sock.close()
                except OSError as e:
                    logger.warning("Error closing %s: %s", name, e)

        self._intr_client = None
        self._ctrl_client = None
        self._intr_sock = None
        self._ctrl_sock = None

        logger.info("Bluetooth HID disconnected")

    @property
    def remote_addr(self) -> str:
        """Bluetooth MAC address of the connected remote host, or empty."""
        return self._remote_addr

    @property
    def remote_name(self) -> str:
        """Friendly name of the connected remote host.

        Looks up the address in the paired device list. Falls back to the
        raw MAC address if the name cannot be resolved.
        """
        if not self._remote_addr:
            return ""
        for addr, name in self._get_paired_devices():
            if addr.upper() == self._remote_addr.upper():
                return name
        return self._remote_addr

    def is_connected(self) -> bool:
        """Check if both L2CAP channels are connected."""
        return self._connected


class BluetoothHIDDevice:
    """Adapter that wraps BluetoothHIDConnection with the HIDDevice interface.

    Provides open/write/close methods compatible with KeyboardController
    and MouseController, so they can send HID reports over Bluetooth
    without modification.

    Usage::

        conn = BluetoothHIDConnection(...)
        kb_dev = BluetoothHIDDevice(conn, report_id=0x01)
        mouse_ctrl = MouseController(kb_dev)  # works transparently
    """

    def __init__(
        self,
        connection: BluetoothHIDConnection,
        report_id: int,
    ) -> None:
        self._connection = connection
        self._report_id = report_id

    def open(self) -> None:
        """No-op — connection is managed by BluetoothHIDConnection."""
        pass

    def write(self, report: bytes) -> None:
        """Send an HID report over the Bluetooth connection.

        Reports include the Report ID at byte 0 (for USB HID). Strip it
        here since send_report() adds the 0xA1 header + report ID itself.
        """
        if len(report) < 2:
            raise HIDDeviceError(
                f"Report too short ({len(report)} bytes) — expected "
                f"Report ID + data"
            )
        self._connection.send_report(self._report_id, report[1:])

    def close(self) -> None:
        """No-op — connection is managed by BluetoothHIDConnection."""
        pass

    def is_open(self) -> bool:
        """Check if the Bluetooth connection is active."""
        return self._connection.is_connected()
