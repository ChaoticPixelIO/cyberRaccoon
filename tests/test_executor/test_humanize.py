"""Tests for input humanization (anti anti-bot).

Covers:
  - Mouse pure function unit tests (Bezier paths, step counts, delays, jitter, etc.)
  - Keyboard pure function unit tests (typing delays, key hold durations)
  - Mouse integration tests with MockHIDDevice
  - Keyboard integration tests with MockHIDDevice
  - Config edge cases
"""

from __future__ import annotations

import math
import random

import pytest

from cyberraccoon.config import HumanizeConfig
from cyberraccoon.executor.humanize import (
    HumanizedKeyboardController,
    HumanizedMouseController,
    apply_jitter,
    compute_key_hold_duration,
    compute_shortcut_hold_duration,
    compute_step_count,
    compute_step_delays,
    compute_typing_delays,
    generate_bezier_path,
    generate_micro_movements,
    generate_overshoot_path,
)
from cyberraccoon.executor.keyboard import KeyboardController
from cyberraccoon.executor.mouse import MouseController

from tests.test_executor.conftest import MockHIDDevice

# LLM coordinate space used across these tests (the historical hardcoded values).
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080


# =========================================================================
# Pure function tests
# =========================================================================


class TestGenerateBezierPath:
    def test_starts_at_start(self) -> None:
        path = generate_bezier_path(
            (100, 200), (500, 400), 20, SCREEN_WIDTH, SCREEN_HEIGHT,
        )
        assert path[0] == (100, 200)

    def test_ends_at_end(self) -> None:
        path = generate_bezier_path(
            (100, 200), (500, 400), 20, SCREEN_WIDTH, SCREEN_HEIGHT,
        )
        assert path[-1] == (500, 400)

    def test_length_matches_num_points(self) -> None:
        path = generate_bezier_path(
            (0, 0), (640, 360), 30, SCREEN_WIDTH, SCREEN_HEIGHT,
        )
        assert len(path) == 30

    def test_stays_within_screen_bounds(self) -> None:
        random.seed(42)
        path = generate_bezier_path(
            (0, 0), (1279, 719), 50, SCREEN_WIDTH, SCREEN_HEIGHT, noise=2.0,
        )
        for x, y in path:
            assert 0 <= x <= SCREEN_WIDTH - 1
            assert 0 <= y <= SCREEN_HEIGHT - 1

    def test_zero_noise_straight_line(self) -> None:
        path = generate_bezier_path(
            (0, 0), (100, 0), 10, SCREEN_WIDTH, SCREEN_HEIGHT, noise=0.0,
        )
        for _, y in path:
            assert y == 0  # no perpendicular deviation

    def test_deterministic_with_seed(self) -> None:
        random.seed(42)
        path1 = generate_bezier_path(
            (0, 0), (200, 200), 20, SCREEN_WIDTH, SCREEN_HEIGHT,
        )
        random.seed(42)
        path2 = generate_bezier_path(
            (0, 0), (200, 200), 20, SCREEN_WIDTH, SCREEN_HEIGHT,
        )
        assert path1 == path2

    def test_short_distance_returns_at_least_endpoints(self) -> None:
        path = generate_bezier_path(
            (100, 100), (101, 100), 2, SCREEN_WIDTH, SCREEN_HEIGHT,
        )
        assert len(path) >= 2
        assert path[0] == (100, 100)
        assert path[-1] == (101, 100)

    def test_zero_distance(self) -> None:
        path = generate_bezier_path(
            (100, 100), (100, 100), 10, SCREEN_WIDTH, SCREEN_HEIGHT,
        )
        assert path[0] == (100, 100)
        assert path[-1] == (100, 100)


class TestComputeStepCount:
    def test_minimum_cap(self) -> None:
        assert compute_step_count(1) >= 5

    def test_monotonic_with_distance(self) -> None:
        short = compute_step_count(50)
        long = compute_step_count(500)
        assert long > short

    def test_speed_multiplier_increases_steps(self) -> None:
        normal = compute_step_count(200, speed=1.0)
        slow = compute_step_count(200, speed=2.0)
        assert slow > normal

    def test_maximum_cap(self) -> None:
        assert compute_step_count(99999) <= 80

    def test_zero_distance(self) -> None:
        assert compute_step_count(0) == 5


class TestComputeStepDelays:
    def test_length_matches_steps(self) -> None:
        delays = compute_step_delays(20)
        assert len(delays) == 20

    def test_all_positive(self) -> None:
        random.seed(42)
        delays = compute_step_delays(30, variance=0.3)
        assert all(d > 0 for d in delays)

    def test_ease_in_out_profile(self) -> None:
        delays = compute_step_delays(20, variance=0.0)
        mid = len(delays) // 2
        # Edges should be slower than middle
        assert delays[0] > delays[mid]
        assert delays[-1] > delays[mid]

    def test_zero_variance_deterministic(self) -> None:
        d1 = compute_step_delays(20, variance=0.0)
        d2 = compute_step_delays(20, variance=0.0)
        assert d1 == d2

    def test_single_step(self) -> None:
        delays = compute_step_delays(1)
        assert len(delays) == 1

    def test_empty(self) -> None:
        assert compute_step_delays(0) == []


class TestApplyJitter:
    def test_within_bounds(self) -> None:
        random.seed(42)
        for _ in range(100):
            x, y = apply_jitter(640, 360, SCREEN_WIDTH, SCREEN_HEIGHT, max_offset_px=3)
            assert abs(x - 640) <= 3
            assert abs(y - 360) <= 3

    def test_clamps_at_screen_edges(self) -> None:
        random.seed(42)
        for _ in range(50):
            x, y = apply_jitter(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT, max_offset_px=5)
            assert x >= 0
            assert y >= 0
            x, y = apply_jitter(
                SCREEN_WIDTH - 1, SCREEN_HEIGHT - 1,
                SCREEN_WIDTH, SCREEN_HEIGHT, max_offset_px=5,
            )
            assert x <= SCREEN_WIDTH - 1
            assert y <= SCREEN_HEIGHT - 1

    def test_zero_jitter_identity(self) -> None:
        x, y = apply_jitter(640, 360, SCREEN_WIDTH, SCREEN_HEIGHT, max_offset_px=0)
        assert (x, y) == (640, 360)


class TestGenerateOvershootPath:
    def test_returns_to_target(self) -> None:
        path = generate_overshoot_path(
            (640, 360), (100, 50), SCREEN_WIDTH, SCREEN_HEIGHT, overshoot_px=15,
        )
        assert path[-1] == (640, 360)

    def test_first_point_extends_past_target(self) -> None:
        random.seed(42)
        target = (640, 360)
        direction = (200, 0)  # moving right
        path = generate_overshoot_path(
            target, direction, SCREEN_WIDTH, SCREEN_HEIGHT, overshoot_px=20,
        )
        # First point should be to the right of target
        assert path[0][0] > target[0]

    def test_stays_within_screen(self) -> None:
        random.seed(42)
        path = generate_overshoot_path(
            (SCREEN_WIDTH - 1, SCREEN_HEIGHT - 1), (100, 100),
            SCREEN_WIDTH, SCREEN_HEIGHT, overshoot_px=50,
        )
        for x, y in path:
            assert 0 <= x <= SCREEN_WIDTH - 1
            assert 0 <= y <= SCREEN_HEIGHT - 1

    def test_zero_direction_returns_target(self) -> None:
        path = generate_overshoot_path(
            (640, 360), (0, 0), SCREEN_WIDTH, SCREEN_HEIGHT,
        )
        assert path == [(640, 360)]


class TestGenerateMicroMovements:
    def test_within_amplitude(self) -> None:
        random.seed(42)
        center = (640, 360)
        points = generate_micro_movements(
            center, SCREEN_WIDTH, SCREEN_HEIGHT,
            num_frames=10, amplitude_px=2.0,
        )
        for x, y in points:
            dist = math.hypot(x - center[0], y - center[1])
            assert dist <= 3.0  # amplitude + rounding tolerance

    def test_correct_count(self) -> None:
        points = generate_micro_movements(
            (640, 360), SCREEN_WIDTH, SCREEN_HEIGHT, num_frames=5,
        )
        assert len(points) == 5

    def test_zero_frames(self) -> None:
        assert generate_micro_movements(
            (640, 360), SCREEN_WIDTH, SCREEN_HEIGHT, num_frames=0,
        ) == []

    def test_zero_amplitude(self) -> None:
        assert generate_micro_movements(
            (640, 360), SCREEN_WIDTH, SCREEN_HEIGHT, amplitude_px=0,
        ) == []


# =========================================================================
# Integration tests with MockHIDDevice
# =========================================================================


def _make_humanized_mouse(
    **config_overrides: object,
) -> tuple[HumanizedMouseController, MockHIDDevice]:
    """Create a HumanizedMouseController with MockHIDDevice."""
    defaults = {
        "enabled": True,
        "overshoot_probability": 0.0,
        "micro_movement_enabled": False,
        "click_jitter_px": 0,
        "timing_variance": 0.0,
        "movement_speed": 1.0,
        "curve_noise": 0.0,  # straight line for predictable tests
    }
    defaults.update(config_overrides)
    config = HumanizeConfig(**defaults)  # type: ignore[arg-type]

    device = MockHIDDevice()
    device.open()
    inner = MouseController(
        device, screen_width=SCREEN_WIDTH, screen_height=SCREEN_HEIGHT,
    )
    mouse = HumanizedMouseController(inner, config)
    return mouse, device


class TestHumanizedMouseClick:
    def test_click_sends_more_reports_than_raw(self) -> None:
        """Humanized click from distance should generate movement + click reports."""
        mouse, device = _make_humanized_mouse()
        mouse._last_x, mouse._last_y = 0, 0
        mouse.click(640, 360)
        # Raw click = 3 reports; humanized adds movement path
        assert len(device.reports) > 3

    def test_click_at_current_position_sends_click_only(self) -> None:
        """Click at (nearly) current position skips movement."""
        mouse, device = _make_humanized_mouse()
        mouse._last_x, mouse._last_y = 640, 360
        mouse.click(640, 360)
        # No movement (distance < 3), just the 3 click reports
        assert len(device.reports) == 3

    def test_final_reports_are_click_pattern(self) -> None:
        """Last 3 reports should be move→press→release."""
        mouse, device = _make_humanized_mouse()
        mouse._last_x, mouse._last_y = 0, 0
        mouse.click(640, 360)
        reports = device.reports
        # Last 3: move(buttons=0), press(buttons=1), release(buttons=0)
        # Byte 1 is buttons (byte 0 is report ID)
        assert reports[-3][1] == 0x00  # move
        assert reports[-2][1] == 0x01  # press (left button)
        assert reports[-1][1] == 0x00  # release

    def test_updates_last_position(self) -> None:
        mouse, device = _make_humanized_mouse()
        mouse._last_x, mouse._last_y = 0, 0
        mouse.click(640, 360)
        assert mouse._last_x == 640
        assert mouse._last_y == 360


class TestHumanizedMouseDoubleClick:
    def test_sends_two_click_patterns(self) -> None:
        """Should have two press-release sequences."""
        mouse, device = _make_humanized_mouse()
        mouse._last_x, mouse._last_y = 640, 360  # already at target
        mouse.double_click(640, 360)
        # Two clicks: each is move+press+release = 6 reports
        assert len(device.reports) == 6

    def test_updates_last_position(self) -> None:
        mouse, device = _make_humanized_mouse()
        mouse._last_x, mouse._last_y = 0, 0
        mouse.double_click(640, 360)
        assert mouse._last_x == 640
        assert mouse._last_y == 360


class TestHumanizedMouseDrag:
    def test_drag_sends_many_reports(self) -> None:
        """Drag should produce movement path + button-held path."""
        mouse, device = _make_humanized_mouse()
        mouse._last_x, mouse._last_y = 100, 100
        mouse.drag(100, 100, 500, 400)
        # Movement to start (skipped, already there) + press + drag path + release
        assert len(device.reports) > 5

    def test_drag_has_button_held_reports(self) -> None:
        """Intermediate drag reports should have left button pressed."""
        mouse, device = _make_humanized_mouse()
        mouse._last_x, mouse._last_y = 100, 100
        mouse.drag(100, 100, 500, 400)
        reports = device.reports
        # Find reports with button held (0x01 in byte 1; byte 0 is report ID)
        held = [r for r in reports if r[1] == 0x01]
        assert len(held) > 3  # press + drag path points

    def test_drag_updates_last_position(self) -> None:
        mouse, device = _make_humanized_mouse()
        mouse._last_x, mouse._last_y = 100, 100
        mouse.drag(100, 100, 500, 400)
        assert mouse._last_x == 500
        assert mouse._last_y == 400


class TestHumanizedMouseScroll:
    def test_scroll_delegates_to_inner(self) -> None:
        """Scroll should produce movement + scroll tick reports."""
        mouse, device = _make_humanized_mouse()
        mouse._last_x, mouse._last_y = 640, 360  # already at target
        mouse.scroll(640, 360, "down", 3)
        # Inner scroll: move(1) + 3 ticks + stop(1) = 5 reports
        assert len(device.reports) == 5

    def test_scroll_updates_last_position(self) -> None:
        mouse, device = _make_humanized_mouse()
        mouse._last_x, mouse._last_y = 0, 0
        mouse.scroll(640, 360, "down", 3)
        assert mouse._last_x == 640
        assert mouse._last_y == 360

    def test_scroll_amount_zero(self) -> None:
        """Scroll with amount=0 should produce move + stop reports only (no ticks)."""
        mouse, device = _make_humanized_mouse()
        mouse._last_x, mouse._last_y = 640, 360  # already at target
        mouse.scroll(640, 360, "down", 0)
        # move(1) + 0 ticks + stop(1) = 2 reports
        assert len(device.reports) == 2


class TestHumanizedMouseWithFeatures:
    def test_jitter_offsets_target(self) -> None:
        """With jitter enabled, final click position may differ from requested."""
        random.seed(42)
        mouse, device = _make_humanized_mouse(click_jitter_px=5)
        mouse._last_x, mouse._last_y = 640, 360  # already near target
        mouse.click(640, 360)
        # The click happened but position tracking reflects jittered coords
        assert abs(mouse._last_x - 640) <= 5
        assert abs(mouse._last_y - 360) <= 5

    def test_micro_movements_add_reports(self) -> None:
        """With micro-movements enabled, more reports are generated."""
        random.seed(42)
        mouse_no_micro, dev_no = _make_humanized_mouse(
            micro_movement_enabled=False,
        )
        mouse_no_micro._last_x, mouse_no_micro._last_y = 640, 360
        mouse_no_micro.click(640, 360)
        count_without = len(dev_no.reports)

        random.seed(42)
        mouse_micro, dev_yes = _make_humanized_mouse(
            micro_movement_enabled=True,
            micro_movement_amplitude_px=1.0,
        )
        mouse_micro._last_x, mouse_micro._last_y = 640, 360
        mouse_micro.click(640, 360)
        count_with = len(dev_yes.reports)

        assert count_with > count_without

    def test_consecutive_clicks_track_position(self) -> None:
        """Second click should move from first click's position."""
        mouse, device = _make_humanized_mouse()
        mouse._last_x, mouse._last_y = 0, 0

        mouse.click(100, 100)
        reports_after_first = len(device.reports)

        mouse.click(110, 100)  # short distance from (100, 100)
        reports_after_second = len(device.reports)
        second_click_reports = reports_after_second - reports_after_first

        # Short movement (10px) should produce fewer movement reports
        # than the first click (distance ~141px)
        first_click_reports = reports_after_first
        assert second_click_reports < first_click_reports


# =========================================================================
# Additional drag tests
# =========================================================================


class TestHumanizedMouseDragExtended:
    def test_drag_releases_button_at_end(self) -> None:
        """The final drag report must release the button (buttons byte == 0x00)."""
        mouse, device = _make_humanized_mouse()
        mouse._last_x, mouse._last_y = 100, 100
        mouse.drag(100, 100, 500, 400)
        # Last report is the release: buttons byte (byte 1) must be 0x00
        assert device.reports[-1][1] == 0x00

    def test_drag_from_distant_position_has_approach(self) -> None:
        """Starting far from drag origin should produce approach movement reports."""
        mouse, device = _make_humanized_mouse()
        mouse._last_x, mouse._last_y = 0, 0  # far from drag origin
        mouse.drag(640, 360, 700, 400)

        # Collect reports before the first button-held report (0x01 in byte 1)
        approach_reports = []
        for r in device.reports:
            if r[1] == 0x01:
                break
            approach_reports.append(r)

        # Should have multiple approach movement reports (Bezier path)
        assert len(approach_reports) >= 3


# =========================================================================
# Additional scroll tests
# =========================================================================


class TestHumanizedMouseScrollExtended:
    def test_scroll_from_distant_position_has_approach(self) -> None:
        """Starting far from scroll target should produce approach reports."""
        mouse, device = _make_humanized_mouse()
        mouse._last_x, mouse._last_y = 0, 0  # far from scroll target
        mouse.scroll(640, 360, "down", 3)

        # Total reports should exceed the base scroll reports (move+3 ticks+stop = 5)
        # because approach movement adds extra movement reports
        assert len(device.reports) > 5

    def test_scroll_up_direction(self) -> None:
        """Scroll up should produce positive wheel values."""
        mouse, device = _make_humanized_mouse()
        mouse._last_x, mouse._last_y = 640, 360
        mouse.scroll(640, 360, "up", 2)
        # Check that at least one report has positive wheel value (byte 5 signed)
        wheel_values = []
        for r in device.reports:
            if len(r) >= 7:
                wheel = r[6]
                if wheel != 0:
                    # Convert unsigned to signed
                    if wheel > 127:
                        wheel -= 256
                    wheel_values.append(wheel)
        assert any(w > 0 for w in wheel_values), f"Expected positive wheel, got {wheel_values}"


# =========================================================================
# Keyboard pure function tests
# =========================================================================


class TestComputeTypingDelays:
    def test_length_matches_text(self) -> None:
        """Returns one delay per character."""
        text = "hello world"
        delays = compute_typing_delays(text, variance=0.0)
        assert len(delays) == len(text)

    def test_empty_text(self) -> None:
        assert compute_typing_delays("") == []

    def test_single_char(self) -> None:
        delays = compute_typing_delays("a", variance=0.0)
        assert len(delays) == 1
        assert delays[0] > 0

    def test_punctuation_slower_than_letter(self) -> None:
        """With zero variance, punctuation char delay > letter delay."""
        letter_delays = compute_typing_delays("abcde", variance=0.0)
        punct_delays = compute_typing_delays("a.b,c", variance=0.0)
        # Punctuation chars (index 1, 3) should be slower than corresponding letters
        assert punct_delays[1] > letter_delays[1]
        assert punct_delays[3] > letter_delays[3]

    def test_space_slower_than_letter(self) -> None:
        """Space (word boundary) adds extra pause."""
        letter_delays = compute_typing_delays("abcdefgh", variance=0.0)
        space_delays = compute_typing_delays("abcd fgh", variance=0.0)
        # The space char (index 4) should be slower than a plain letter
        assert space_delays[4] > letter_delays[4]

    def test_ramp_up_first_chars(self) -> None:
        """With zero variance, first chars should be slower (ramp-up)."""
        text = "abcdefghij"  # >= 4 chars to trigger ramp-up
        delays = compute_typing_delays(text, variance=0.0)
        # First char (index 0) should have 1.5x multiplier
        # Fourth char (index 3) should have ~1.0x multiplier
        # Middle char (index 5) should have base delay (no ramp)
        assert delays[0] > delays[5]
        assert delays[1] > delays[5]

    def test_all_delays_positive(self) -> None:
        """All delays should be positive even with high variance."""
        random.seed(42)
        delays = compute_typing_delays("Hello, world!", variance=0.5)
        assert all(d > 0 for d in delays)

    def test_speed_multiplier(self) -> None:
        """Higher speed multiplier should produce longer delays."""
        d_normal = compute_typing_delays("abc", variance=0.0, speed=1.0)
        d_slow = compute_typing_delays("abc", variance=0.0, speed=2.0)
        # Slow should be about 2x longer (ignoring ramp effects)
        for dn, ds in zip(d_normal, d_slow):
            assert ds > dn

    def test_minimum_delay_clamp(self) -> None:
        """No delay should go below 5ms minimum."""
        random.seed(42)
        delays = compute_typing_delays("test", variance=0.9, speed=0.1)
        assert all(d >= 0.005 for d in delays)


class TestComputeKeyHoldDuration:
    def test_within_range(self) -> None:
        """100 samples should all be within [10ms, 60ms]."""
        random.seed(42)
        for _ in range(100):
            hold = compute_key_hold_duration()
            assert 0.010 <= hold <= 0.060

    def test_zero_variance_returns_base(self) -> None:
        hold = compute_key_hold_duration(base_hold_s=0.025, variance=0.0)
        assert hold == 0.025

    def test_clamps_low(self) -> None:
        """Very low base should be clamped to 10ms."""
        hold = compute_key_hold_duration(base_hold_s=0.001, variance=0.0)
        assert hold == 0.010

    def test_clamps_high(self) -> None:
        """Very high base should be clamped to 60ms."""
        hold = compute_key_hold_duration(base_hold_s=0.200, variance=0.0)
        assert hold == 0.060


class TestComputeShortcutHoldDuration:
    def test_within_range(self) -> None:
        """100 samples should all be within [30ms, 100ms]."""
        random.seed(42)
        for _ in range(100):
            hold = compute_shortcut_hold_duration()
            assert 0.030 <= hold <= 0.100

    def test_zero_variance_returns_base(self) -> None:
        hold = compute_shortcut_hold_duration(base_hold_s=0.05, variance=0.0)
        assert hold == 0.05

    def test_clamps_low(self) -> None:
        hold = compute_shortcut_hold_duration(base_hold_s=0.001, variance=0.0)
        assert hold == 0.030

    def test_clamps_high(self) -> None:
        hold = compute_shortcut_hold_duration(base_hold_s=0.500, variance=0.0)
        assert hold == 0.100

    def test_longer_than_key_hold(self) -> None:
        """Shortcut hold should generally be longer than single key hold."""
        random.seed(42)
        key_holds = [compute_key_hold_duration() for _ in range(50)]
        shortcut_holds = [compute_shortcut_hold_duration() for _ in range(50)]
        # Average shortcut hold should exceed average key hold
        assert sum(shortcut_holds) / len(shortcut_holds) > sum(key_holds) / len(key_holds)


# =========================================================================
# Keyboard integration tests with MockHIDDevice
# =========================================================================


def _make_humanized_keyboard(
    **config_overrides: object,
) -> tuple[HumanizedKeyboardController, MockHIDDevice]:
    """Create a HumanizedKeyboardController with MockHIDDevice."""
    defaults = {
        "enabled": True,
        "typing_speed": 1.0,
        "typing_variance": 0.0,  # deterministic by default
        "punctuation_pause_ms": 80.0,
        "word_pause_ms": 40.0,
    }
    defaults.update(config_overrides)
    config = HumanizeConfig(**defaults)  # type: ignore[arg-type]

    device = MockHIDDevice()
    device.open()
    inner = KeyboardController(device)
    kb = HumanizedKeyboardController(inner, config)
    return kb, device


class TestHumanizedKeyboardTypeText:
    def test_sends_press_and_release_per_char(self) -> None:
        """Each character should produce 2 reports: press + release."""
        kb, device = _make_humanized_keyboard()
        kb.type_text("abc")
        # 3 chars × (press + release) = 6 reports
        assert len(device.reports) == 6

    def test_press_reports_have_keycode(self) -> None:
        """Press reports (even indices) should have a non-zero keycode."""
        kb, device = _make_humanized_keyboard()
        kb.type_text("a")
        # Report 0 is the press
        press_report = device.reports[0]
        assert press_report[3] != 0  # keycode byte is non-zero

    def test_release_reports_are_all_zeros(self) -> None:
        """Release reports (odd indices) should be all zeros."""
        kb, device = _make_humanized_keyboard()
        kb.type_text("a")
        release_report = device.reports[1]
        assert release_report == bytes([0x01]) + bytes(8)

    def test_shift_for_uppercase(self) -> None:
        """Uppercase characters should have shift modifier in press report."""
        kb, device = _make_humanized_keyboard()
        kb.type_text("A")
        press_report = device.reports[0]
        # Byte 1 should have shift bit (0x02); byte 0 is report ID
        assert press_report[1] & 0x02 != 0

    def test_newline_sends_enter_key(self) -> None:
        """Newline character should send enter key (shortcut-style)."""
        kb, device = _make_humanized_keyboard()
        kb.type_text("\n")
        # press_keys for "enter" sends press + release = 2 reports
        assert len(device.reports) == 2
        # Press report should have enter keycode (0x28)
        assert device.reports[0][3] == 0x28

    def test_tab_sends_tab_key(self) -> None:
        """Tab character should send tab key."""
        kb, device = _make_humanized_keyboard()
        kb.type_text("\t")
        assert len(device.reports) == 2
        assert device.reports[0][3] == 0x2B  # tab keycode

    def test_empty_text_sends_nothing(self) -> None:
        kb, device = _make_humanized_keyboard()
        kb.type_text("")
        assert len(device.reports) == 0

    def test_mixed_case_report_count(self) -> None:
        """Mixed case text should still produce 2 reports per char."""
        kb, device = _make_humanized_keyboard()
        kb.type_text("aB")
        assert len(device.reports) == 4


class TestHumanizedKeyboardPressKeys:
    def test_sends_two_reports(self) -> None:
        """press_keys should produce exactly 2 reports: press + release."""
        kb, device = _make_humanized_keyboard()
        kb.press_keys(["ctrl", "c"])
        assert len(device.reports) == 2

    def test_press_has_modifier_and_keycode(self) -> None:
        """Ctrl+C press report should have ctrl modifier and 'c' keycode."""
        kb, device = _make_humanized_keyboard()
        kb.press_keys(["ctrl", "c"])
        press = device.reports[0]
        assert press[1] & 0x01 != 0  # ctrl modifier bit (byte 1; byte 0 is report ID)
        assert press[3] == 0x06  # 'c' HID usage ID

    def test_release_is_all_zeros(self) -> None:
        kb, device = _make_humanized_keyboard()
        kb.press_keys(["enter"])
        release = device.reports[1]
        assert release == bytes([0x01]) + bytes(8)

    def test_multiple_modifiers(self) -> None:
        """Ctrl+Shift+key should combine modifiers."""
        kb, device = _make_humanized_keyboard()
        kb.press_keys(["ctrl", "shift", "a"])
        press = device.reports[0]
        # Ctrl = 0x01, Shift = 0x02 (byte 1; byte 0 is report ID)
        assert press[1] & 0x01 != 0
        assert press[1] & 0x02 != 0


# =========================================================================
# Config edge cases
# =========================================================================


class TestConfigEdgeCases:
    def test_zero_movement_speed(self) -> None:
        """speed=0.0 should return minimum step count (5)."""
        result = compute_step_count(200, speed=0.0)
        assert result == 5

    def test_high_timing_variance(self) -> None:
        """All delays should remain positive even with variance=0.9."""
        random.seed(42)
        delays = compute_step_delays(50, variance=0.9)
        assert all(d > 0 for d in delays)
        assert len(delays) == 50

    def test_high_typing_variance(self) -> None:
        """All typing delays should remain positive with variance=0.9."""
        random.seed(42)
        delays = compute_typing_delays("Hello, world!", variance=0.9)
        assert all(d > 0 for d in delays)

    def test_very_slow_movement_speed(self) -> None:
        """Very high speed multiplier should still be capped at 80 steps."""
        result = compute_step_count(1000, speed=10.0)
        assert result <= 80

    def test_disabled_config_skips_humanization(self) -> None:
        """When enabled=False, wrapping helpers return the original controller."""
        from cyberraccoon.executor.base_executor import BaseExecutor
        config = HumanizeConfig(enabled=False)
        device = MockHIDDevice()
        device.open()
        mouse = MouseController(
            device, screen_width=SCREEN_WIDTH, screen_height=SCREEN_HEIGHT,
        )
        keyboard = KeyboardController(device)

        # Use a concrete BaseExecutor subclass for testing
        class _TestExecutor(BaseExecutor):
            def open(self) -> None:
                pass
            def close(self) -> None:
                pass

        executor = _TestExecutor(
            humanize_config=config,
            screen_width=SCREEN_WIDTH, screen_height=SCREEN_HEIGHT,
        )
        wrapped_mouse = executor._wrap_mouse_if_humanized(mouse)
        wrapped_kb = executor._wrap_keyboard_if_humanized(keyboard)

        # Should return the originals since enabled=False
        assert wrapped_mouse is mouse
        assert wrapped_kb is keyboard

    def test_enabled_config_wraps_controllers(self) -> None:
        """When enabled=True, wrapping helpers return humanized proxies."""
        from cyberraccoon.executor.base_executor import BaseExecutor
        config = HumanizeConfig(enabled=True)
        device = MockHIDDevice()
        device.open()
        mouse = MouseController(
            device, screen_width=SCREEN_WIDTH, screen_height=SCREEN_HEIGHT,
        )
        keyboard = KeyboardController(device)

        class _TestExecutor(BaseExecutor):
            def open(self) -> None:
                pass
            def close(self) -> None:
                pass

        executor = _TestExecutor(
            humanize_config=config,
            screen_width=SCREEN_WIDTH, screen_height=SCREEN_HEIGHT,
        )
        wrapped_mouse = executor._wrap_mouse_if_humanized(mouse)
        wrapped_kb = executor._wrap_keyboard_if_humanized(keyboard)

        assert isinstance(wrapped_mouse, HumanizedMouseController)
        assert isinstance(wrapped_kb, HumanizedKeyboardController)


# =========================================================================
# Move at target
# =========================================================================


class TestHumanizedMouseMoveAtTarget:
    def test_move_at_target_sends_no_reports(self) -> None:
        """move() when already at target should send no HID reports."""
        mouse, device = _make_humanized_mouse()
        mouse._last_x, mouse._last_y = 640, 360
        mouse.move(640, 360)
        assert len(device.reports) == 0

    def test_move_at_target_preserves_position(self) -> None:
        """move() when already at target should keep last_x/last_y correct."""
        mouse, device = _make_humanized_mouse()
        mouse._last_x, mouse._last_y = 640, 360
        mouse.move(640, 360)
        assert mouse._last_x == 640
        assert mouse._last_y == 360

    def test_move_updates_position_after_travel(self) -> None:
        """move() to a new position should update last_x/last_y."""
        mouse, device = _make_humanized_mouse()
        mouse._last_x, mouse._last_y = 0, 0
        mouse.move(640, 360)
        assert mouse._last_x == 640
        assert mouse._last_y == 360


# =========================================================================
# Unsupported characters in type_text
# =========================================================================


class TestHumanizedKeyboardUnsupportedChars:
    def test_unsupported_char_raises(self) -> None:
        """Characters not in CHAR_MAP should raise ValueError."""
        kb, device = _make_humanized_keyboard()
        with pytest.raises(ValueError, match="Unsupported character"):
            kb.type_text("a€b")

    def test_only_unsupported_chars_raises(self) -> None:
        """A string of only unsupported chars should raise ValueError."""
        kb, device = _make_humanized_keyboard()
        with pytest.raises(ValueError, match="Unsupported character"):
            kb.type_text("€£¥")


# =========================================================================
# Cache eviction (deque-based dedup)
# =========================================================================


class TestHumanizedMouseTripleClick:
    def test_triple_click_sends_three_click_patterns(self) -> None:
        """Should have three press-release sequences."""
        mouse, device = _make_humanized_mouse()
        mouse._last_x, mouse._last_y = 640, 360  # already at target
        mouse.triple_click(640, 360)
        # Three clicks: each is move+press+release = 9 reports
        assert len(device.reports) == 9

    def test_triple_click_updates_last_position(self) -> None:
        mouse, device = _make_humanized_mouse()
        mouse._last_x, mouse._last_y = 0, 0
        mouse.triple_click(640, 360)
        assert mouse._last_x == 640
        assert mouse._last_y == 360


class TestPressKeysUnknownKey:
    def test_all_unknown_keys_raises(self) -> None:
        """Unknown keys should raise ValueError."""
        kb, device = _make_humanized_keyboard()
        with pytest.raises(ValueError, match="Unknown key: nokey1"):
            kb.press_keys(["nokey1", "nokey2"])

    def test_mixed_known_unknown_raises(self) -> None:
        """Even with some known keys, an unknown key should raise."""
        kb, device = _make_humanized_keyboard()
        with pytest.raises(ValueError, match="Unknown key: nokey"):
            kb.press_keys(["ctrl", "nokey"])


class TestScrollUnknownDirection:
    def test_unknown_direction_raises(self) -> None:
        """An unrecognized direction should raise ValueError."""
        mouse, device = _make_humanized_mouse()
        mouse._last_x, mouse._last_y = 640, 360
        with pytest.raises(ValueError, match="Unknown scroll direction"):
            mouse.scroll(640, 360, "sideways", 1)


class TestHumanizeConfigValidation:
    def test_invalid_movement_speed_raises(self) -> None:
        with pytest.raises(ValueError, match="movement_speed"):
            HumanizeConfig(enabled=True, movement_speed=0.0)

    def test_invalid_movement_speed_raises_when_disabled(self) -> None:
        """Validation is now unconditional — fires even with enabled=False."""
        with pytest.raises(ValueError, match="movement_speed"):
            HumanizeConfig(enabled=False, movement_speed=-1.0)

    def test_invalid_curve_noise_raises(self) -> None:
        with pytest.raises(ValueError, match="curve_noise"):
            HumanizeConfig(enabled=True, curve_noise=-0.1)

    def test_invalid_timing_variance_raises(self) -> None:
        with pytest.raises(ValueError, match="timing_variance"):
            HumanizeConfig(enabled=True, timing_variance=1.5)

    def test_invalid_overshoot_probability_raises(self) -> None:
        with pytest.raises(ValueError, match="overshoot_probability"):
            HumanizeConfig(enabled=True, overshoot_probability=2.0)

    def test_invalid_overshoot_distance_raises(self) -> None:
        with pytest.raises(ValueError, match="overshoot_distance_px"):
            HumanizeConfig(enabled=True, overshoot_distance_px=-1)

    def test_invalid_micro_movement_amplitude_raises(self) -> None:
        with pytest.raises(ValueError, match="micro_movement_amplitude_px"):
            HumanizeConfig(enabled=True, micro_movement_amplitude_px=-0.1)

    def test_invalid_typing_speed_raises(self) -> None:
        with pytest.raises(ValueError, match="typing_speed"):
            HumanizeConfig(enabled=True, typing_speed=0.0)

    def test_invalid_typing_variance_raises(self) -> None:
        with pytest.raises(ValueError, match="typing_variance"):
            HumanizeConfig(enabled=True, typing_variance=-0.1)

    def test_invalid_punctuation_pause_raises(self) -> None:
        with pytest.raises(ValueError, match="punctuation_pause_ms"):
            HumanizeConfig(enabled=True, punctuation_pause_ms=-1.0)

    def test_invalid_word_pause_raises(self) -> None:
        with pytest.raises(ValueError, match="word_pause_ms"):
            HumanizeConfig(enabled=True, word_pause_ms=-1.0)

    def test_frozen_config_is_immutable(self) -> None:
        """HumanizeConfig is frozen — fields cannot be mutated after creation."""
        config = HumanizeConfig(enabled=True)
        with pytest.raises((AttributeError, TypeError)):
            config.movement_speed = 2.0  # type: ignore[misc]


class TestExecuteBeforeOpen:
    def test_execute_before_open_returns_error(self) -> None:
        """Calling execute() before open() should return a structured error."""
        from cyberraccoon.executor.base_executor import BaseExecutor

        class _TestExecutor(BaseExecutor):
            def open(self) -> None:
                pass
            def close(self) -> None:
                pass

        executor = _TestExecutor(
            screen_width=SCREEN_WIDTH, screen_height=SCREEN_HEIGHT,
        )
        result = executor.execute({"id": "x1", "action": "click", "x": 100, "y": 100})
        assert result["status"] == "error"
        assert result["id"] == "x1"
        assert "open" in result["error"].lower()

    def test_execute_missing_field_returns_error(self) -> None:
        """execute() with a malformed command (missing 'x') returns error, not exception."""
        from cyberraccoon.executor.base_executor import BaseExecutor
        from cyberraccoon.executor.keyboard import KeyboardController
        from cyberraccoon.executor.mouse import MouseController

        class _TestExecutor(BaseExecutor):
            def open(self) -> None:
                device = MockHIDDevice()
                device.open()
                self._keyboard = KeyboardController(device)
                self._mouse = MouseController(
                    device,
                    screen_width=self._screen_width,
                    screen_height=self._screen_height,
                )
            def close(self) -> None:
                pass

        executor = _TestExecutor(
            screen_width=SCREEN_WIDTH, screen_height=SCREEN_HEIGHT,
        )
        executor.open()
        result = executor.execute({"id": "x2", "action": "click"})  # missing x, y
        assert result["status"] == "error"
        assert result["id"] == "x2"


class TestLoadHumanizeConfigEnvErrors:
    def test_invalid_float_env_var_raises_valueerror(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from cyberraccoon.config import load_humanize_config
        monkeypatch.setenv("CYBERRACCOON_HUMANIZE_SPEED", "fast")
        with pytest.raises(ValueError, match="CYBERRACCOON_HUMANIZE_SPEED"):
            load_humanize_config()

    def test_invalid_int_env_var_raises_valueerror(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from cyberraccoon.config import load_humanize_config
        monkeypatch.setenv("CYBERRACCOON_HUMANIZE_JITTER", "2px")
        with pytest.raises(ValueError, match="CYBERRACCOON_HUMANIZE_JITTER"):
            load_humanize_config()

    def test_valid_env_vars_load_correctly(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from cyberraccoon.config import load_humanize_config
        monkeypatch.setenv("CYBERRACCOON_HUMANIZE", "1")
        monkeypatch.setenv("CYBERRACCOON_HUMANIZE_SPEED", "1.5")
        monkeypatch.setenv("CYBERRACCOON_HUMANIZE_JITTER", "3")
        config = load_humanize_config()
        assert config.enabled is True
        assert config.movement_speed == 1.5
        assert config.click_jitter_px == 3

    def test_micro_movement_env_vars_loaded(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from cyberraccoon.config import load_humanize_config
        monkeypatch.setenv("CYBERRACCOON_HUMANIZE_MICRO_ENABLED", "0")
        monkeypatch.setenv("CYBERRACCOON_HUMANIZE_MICRO_AMP", "1.5")
        config = load_humanize_config()
        assert config.micro_movement_enabled is False
        assert config.micro_movement_amplitude_px == 1.5


class TestBaseExecutorCacheEviction:
    def _make_executor(self) -> object:
        from cyberraccoon.executor.base_executor import BaseExecutor

        class _TestExecutor(BaseExecutor):
            def open(self) -> None:
                pass
            def close(self) -> None:
                pass

        return _TestExecutor(
            screen_width=SCREEN_WIDTH, screen_height=SCREEN_HEIGHT,
        )

    def test_deque_evicts_oldest_on_overflow(self) -> None:
        """When the deque reaches maxlen, oldest IDs are evicted automatically."""
        executor = self._make_executor()
        for i in range(1001):
            executor._executed_ids.append(f"id_{i}")
        # Oldest entry should be gone
        assert "id_0" not in executor._executed_ids
        # Most recent entry should still be present
        assert "id_1000" in executor._executed_ids

    def test_recent_ids_still_deduplicated_after_eviction(self) -> None:
        """After eviction the deque still deduplicates recent entries."""
        executor = self._make_executor()
        for i in range(1001):
            executor._executed_ids.append(f"id_{i}")
        # A recently added ID should still be detected as a duplicate
        assert "id_999" in executor._executed_ids
        assert "id_1000" in executor._executed_ids

    def test_evicted_id_is_no_longer_duplicate(self) -> None:
        """An evicted ID should no longer be considered a duplicate."""
        executor = self._make_executor()
        for i in range(1001):
            executor._executed_ids.append(f"id_{i}")
        # id_0 was evicted — it should NOT be treated as a known duplicate
        assert "id_0" not in executor._executed_ids
