"""Tests for Bluetooth HID transport — BluetoothHIDDevice and BluetoothExecutor.

All tests use mocks (no real Bluetooth hardware needed).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch, ANY

from executor.bluetooth_device import BluetoothHIDConnection, BluetoothHIDDevice
from executor.bluetooth_executor import BluetoothExecutor
from executor.hid_device import HIDDeviceError


# ---------------------------------------------------------------------------
# Helper: create a connected BluetoothHIDConnection with mocked sockets
# ---------------------------------------------------------------------------

def make_mock_connection() -> BluetoothHIDConnection:
    """Create a BluetoothHIDConnection with mocked L2CAP sockets.

    Simulates a connected state without real Bluetooth hardware.
    """
    conn = BluetoothHIDConnection.__new__(BluetoothHIDConnection)
    conn._device_name = "TestRaccoon"
    conn._device_class = 0x002540
    conn._ctrl_sock = MagicMock()
    conn._intr_sock = MagicMock()
    conn._ctrl_client = MagicMock()
    conn._intr_client = MagicMock()
    conn._dbus_profile = None
    conn._connected = True
    return conn


def make_mock_bt_executor() -> BluetoothExecutor:
    """Create a BluetoothExecutor with mocked Bluetooth connection.

    Returns a ready-to-use executor without Bluetooth hardware.
    """
    conn = make_mock_connection()

    executor = BluetoothExecutor.__new__(BluetoothExecutor)
    executor._device_name = "TestRaccoon"
    executor._connection_timeout = 60.0
    executor._connection = conn
    executor._humanize_config = None
    executor._target_os = None
    from collections import deque
    executor._executed_ids = deque(maxlen=1000)

    # Create BluetoothHIDDevice adapters
    kb_dev = BluetoothHIDDevice(conn, report_id=0x01)
    ms_dev = BluetoothHIDDevice(conn, report_id=0x02)

    # Reuse real controllers
    from executor.keyboard import KeyboardController
    from executor.mouse import MouseController

    executor._keyboard = KeyboardController(kb_dev)
    executor._mouse = MouseController(ms_dev)

    return executor


# ---------------------------------------------------------------------------
# Tests: BluetoothHIDDevice (adapter interface)
# ---------------------------------------------------------------------------

class TestBluetoothHIDDevice:
    """Tests for the BluetoothHIDDevice adapter."""

    def test_write_sends_report_with_header(self) -> None:
        """write() should call connection.send_report with report ID."""
        conn = make_mock_connection()
        dev = BluetoothHIDDevice(conn, report_id=0x01)

        report = bytes([0x01]) + bytes(8)  # keyboard release (Report ID + 8 zeros)
        dev.write(report)

        conn._intr_client.send.assert_called_once()
        sent_data = conn._intr_client.send.call_args[0][0]
        # Should have 0xA1 header + report_id + report data (without embedded ID)
        assert sent_data[0] == 0xA1
        assert sent_data[1] == 0x01
        assert sent_data[2:] == bytes(8)

    def test_write_mouse_report_id(self) -> None:
        """Mouse device should use report ID 0x02."""
        conn = make_mock_connection()
        dev = BluetoothHIDDevice(conn, report_id=0x02)

        report = bytes(7)  # mouse report
        dev.write(report)

        sent_data = conn._intr_client.send.call_args[0][0]
        assert sent_data[0] == 0xA1
        assert sent_data[1] == 0x02

    def test_is_open_reflects_connection(self) -> None:
        """is_open() should reflect the connection state."""
        conn = make_mock_connection()
        dev = BluetoothHIDDevice(conn, report_id=0x01)

        assert dev.is_open() is True

        conn._connected = False
        assert dev.is_open() is False

    def test_open_is_noop(self) -> None:
        """open() should not raise."""
        conn = make_mock_connection()
        dev = BluetoothHIDDevice(conn, report_id=0x01)
        dev.open()  # Should not raise

    def test_close_is_noop(self) -> None:
        """close() should not raise."""
        conn = make_mock_connection()
        dev = BluetoothHIDDevice(conn, report_id=0x01)
        dev.close()  # Should not raise


# ---------------------------------------------------------------------------
# Tests: BluetoothHIDConnection
# ---------------------------------------------------------------------------

class TestBluetoothHIDConnection:
    """Tests for BluetoothHIDConnection lifecycle."""

    def test_initial_state(self) -> None:
        """New connection should not be connected."""
        conn = BluetoothHIDConnection()
        assert conn.is_connected() is False

    def test_send_report_when_not_connected(self) -> None:
        """send_report() should raise when not connected."""
        conn = BluetoothHIDConnection()
        try:
            conn.send_report(0x01, bytes(8))
            assert False, "Should have raised HIDDeviceError"
        except HIDDeviceError as e:
            assert "not connected" in str(e)

    def test_send_report_format(self) -> None:
        """send_report() should send 0xA1 + report_id + data."""
        conn = make_mock_connection()
        data = bytes([0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])  # mouse

        conn.send_report(0x02, data)

        sent = conn._intr_client.send.call_args[0][0]
        assert len(sent) == 2 + len(data)
        assert sent[0] == 0xA1
        assert sent[1] == 0x02
        assert sent[2:] == data

    def test_send_report_socket_error(self) -> None:
        """send_report() should raise HIDDeviceError on socket failure."""
        conn = make_mock_connection()
        conn._intr_client.send.side_effect = OSError("Connection reset")

        try:
            conn.send_report(0x01, bytes(8))
            assert False, "Should have raised HIDDeviceError"
        except HIDDeviceError as e:
            assert "Failed to send" in str(e)
        assert conn.is_connected() is False

    def test_disconnect(self) -> None:
        """disconnect() should close all sockets and set state."""
        conn = make_mock_connection()
        ctrl_client = conn._ctrl_client
        intr_client = conn._intr_client

        conn.disconnect()

        assert conn.is_connected() is False
        ctrl_client.close.assert_called_once()
        intr_client.close.assert_called_once()

    def test_disconnect_twice_safe(self) -> None:
        """Calling disconnect() twice should not raise."""
        conn = make_mock_connection()
        conn.disconnect()
        conn.disconnect()  # Should not raise


# ---------------------------------------------------------------------------
# Tests: BluetoothExecutor (command dispatch)
# ---------------------------------------------------------------------------

class TestBluetoothExecutorRouting:
    """Tests for BluetoothExecutor command routing."""

    def test_click_sends_mouse_reports(self) -> None:
        """click action should send reports via mouse device."""
        executor = make_mock_bt_executor()
        result = executor.execute(
            {"id": "t1", "action": "click", "x": 640, "y": 360}
        )
        assert result["status"] == "ok"
        # Verify reports were sent to interrupt socket
        conn = executor._connection
        assert conn._intr_client.send.call_count > 0

    def test_type_sends_keyboard_reports(self) -> None:
        """type action should send reports via keyboard device."""
        executor = make_mock_bt_executor()
        result = executor.execute(
            {"id": "t2", "action": "type", "text": "hi"}
        )
        assert result["status"] == "ok"
        conn = executor._connection
        assert conn._intr_client.send.call_count > 0

    def test_key_combo(self) -> None:
        """key action should work for modifier combos."""
        executor = make_mock_bt_executor()
        result = executor.execute(
            {"id": "t3", "action": "key", "keys": ["ctrl", "c"]}
        )
        assert result["status"] == "ok"

    def test_scroll(self) -> None:
        """scroll action should work."""
        executor = make_mock_bt_executor()
        result = executor.execute(
            {"id": "t4", "action": "scroll", "x": 640, "y": 360,
             "direction": "down", "amount": 3}
        )
        assert result["status"] == "ok"

    def test_drag(self) -> None:
        """drag action should work."""
        executor = make_mock_bt_executor()
        result = executor.execute(
            {"id": "t5", "action": "drag",
             "from_x": 100, "from_y": 200, "to_x": 500, "to_y": 300}
        )
        assert result["status"] == "ok"

    def test_unknown_action_error(self) -> None:
        """Unknown action should return error status."""
        executor = make_mock_bt_executor()
        result = executor.execute(
            {"id": "t6", "action": "fly"}
        )
        assert result["status"] == "error"
        assert "Unknown action" in result["error"]

    def test_done_action(self) -> None:
        """done action should return ok with 0 duration."""
        executor = make_mock_bt_executor()
        result = executor.execute(
            {"id": "d1", "action": "done"}
        )
        assert result["status"] == "ok"
        assert result["action"] == "done"
        assert result["duration_ms"] == 0


class TestBluetoothExecutorDedup:
    """Tests for BluetoothExecutor deduplication."""

    def test_duplicate_skipped(self) -> None:
        """Same command ID executed twice should be skipped."""
        executor = make_mock_bt_executor()
        cmd = {"id": "dup1", "action": "click", "x": 100, "y": 100}
        r1 = executor.execute(cmd)
        r2 = executor.execute(cmd)
        assert r1["status"] == "ok"
        assert r2["status"] == "skipped"

    def test_different_ids_both_execute(self) -> None:
        """Different command IDs should both execute."""
        executor = make_mock_bt_executor()
        r1 = executor.execute(
            {"id": "a", "action": "click", "x": 100, "y": 100}
        )
        r2 = executor.execute(
            {"id": "b", "action": "click", "x": 200, "y": 200}
        )
        assert r1["status"] == "ok"
        assert r2["status"] == "ok"

    def test_empty_id_always_executes(self) -> None:
        """Empty command ID should always execute (never skipped)."""
        executor = make_mock_bt_executor()
        r1 = executor.execute(
            {"id": "", "action": "click", "x": 100, "y": 100}
        )
        r2 = executor.execute(
            {"id": "", "action": "click", "x": 100, "y": 100}
        )
        assert r1["status"] == "ok"
        assert r2["status"] == "ok"

    def test_done_not_tracked(self) -> None:
        """done action should not be tracked in dedup cache."""
        executor = make_mock_bt_executor()
        executor.execute({"id": "d1", "action": "done"})
        # Same ID with different action should still execute
        result = executor.execute(
            {"id": "d1", "action": "click", "x": 100, "y": 100}
        )
        assert result["status"] == "ok"


class TestBluetoothExecutorReportFormat:
    """Tests for verifying correct report format over Bluetooth."""

    def test_keyboard_report_has_header_and_id(self) -> None:
        """Keyboard reports should have 0xA1 + 0x01 + 8 bytes."""
        executor = make_mock_bt_executor()
        executor.execute(
            {"id": "k1", "action": "key", "keys": ["a"]}
        )
        conn = executor._connection
        # Check that at least one send was a keyboard report
        calls = conn._intr_client.send.call_args_list
        # Find keyboard reports (report_id = 0x01)
        kb_reports = [
            c[0][0] for c in calls
            if len(c[0][0]) == 10 and c[0][0][1] == 0x01
        ]
        assert len(kb_reports) > 0
        for r in kb_reports:
            assert r[0] == 0xA1   # header
            assert r[1] == 0x01   # report ID
            assert len(r) == 10   # 2 header + 8 report

    def test_mouse_report_has_header_and_id(self) -> None:
        """Mouse reports should have 0xA1 + 0x02 + 7 bytes."""
        executor = make_mock_bt_executor()
        executor.execute(
            {"id": "m1", "action": "click", "x": 640, "y": 360}
        )
        conn = executor._connection
        calls = conn._intr_client.send.call_args_list
        # Find mouse reports (report_id = 0x02)
        ms_reports = [
            c[0][0] for c in calls
            if len(c[0][0]) == 9 and c[0][0][1] == 0x02
        ]
        assert len(ms_reports) > 0
        for r in ms_reports:
            assert r[0] == 0xA1   # header
            assert r[1] == 0x02   # report ID
            assert len(r) == 9    # 2 header + 7 report


# ---------------------------------------------------------------------------
# Tests: BluetoothExecutor wait action
# ---------------------------------------------------------------------------

class TestBluetoothWaitAction:
    """Tests for the wait action via BluetoothExecutor."""

    @patch("executor.base_executor.time.sleep")
    def test_wait_sleeps_for_duration(self, mock_sleep: MagicMock) -> None:
        executor = make_mock_bt_executor()
        result = executor.execute(
            {"id": "w1", "action": "wait", "duration_s": 3.0}
        )
        assert result["status"] == "ok"
        assert result["action"] == "wait"
        mock_sleep.assert_called_once_with(3.0)

    @patch("executor.base_executor.time.sleep")
    def test_wait_duration_capped_at_10(self, mock_sleep: MagicMock) -> None:
        executor = make_mock_bt_executor()
        executor.execute(
            {"id": "w2", "action": "wait", "duration_s": 30.0}
        )
        mock_sleep.assert_called_once_with(10.0)

    @patch("executor.base_executor.time.sleep")
    def test_wait_default_duration(self, mock_sleep: MagicMock) -> None:
        """No duration_s field should default to 1.0."""
        executor = make_mock_bt_executor()
        executor.execute({"id": "w3", "action": "wait"})
        mock_sleep.assert_called_once_with(1.0)

    @patch("executor.base_executor.time.sleep")
    def test_wait_no_bt_reports(self, mock_sleep: MagicMock) -> None:
        """Wait action should not send any Bluetooth reports."""
        executor = make_mock_bt_executor()
        conn = executor._connection
        conn._intr_client.send.reset_mock()
        executor.execute(
            {"id": "w4", "action": "wait", "duration_s": 1.0}
        )
        conn._intr_client.send.assert_not_called()
