"""Input humanization — anti anti-bot.

Wraps MouseController and KeyboardController with human-like behavior:
- Mouse: natural Bezier curves, timing variation, jitter, overshoot, tremor
- Keyboard: variable inter-key delays, punctuation pauses, typing ramp-up
- Scroll: per-tick timing variance with ease-in-out

All pure helper functions are stateless and independently testable.
HumanizedMouseController and HumanizedKeyboardController are drop-in proxies.
"""

from __future__ import annotations

import logging
import math
import random
import time

from cyberraccoon.config import HumanizeConfig
from cyberraccoon.executor.keyboard import (
    CHAR_MAP,
    KeyboardController,
    MODIFIER_MAP,
    SPECIAL_KEY_MAP,
    _build_report as _kb_build_report,
)
from cyberraccoon.executor.mouse import (
    BUTTON_LEFT,
    MouseController,
    _screen_to_hid,
    SCREEN_WIDTH,
    SCREEN_HEIGHT,
)

logger = logging.getLogger("M4.humanize")

# Bezier curve control point offset as a fraction of the chord length.
# Higher values produce more pronounced curves.
_BEZIER_CURVE_FACTOR: float = 0.2

# Movements shorter than this (px) are skipped — cursor is considered on target.
_MIN_MOVEMENT_DISTANCE_PX: float = 3.0

# Overshoot correction segments are this many times slower than the
# preceding movement delay (gives a natural slow-down feel).
_OVERSHOOT_DELAY_FACTOR: float = 1.5

# Fallback delay (s) used for overshoot segments when no prior delay exists.
_OVERSHOOT_FALLBACK_DELAY_S: float = 0.008

# Per-frame sleep and jitter used during micro-movement (hand tremor) simulation.
_MICRO_MOVEMENT_SLEEP_S: float = 0.010
_MICRO_MOVEMENT_JITTER_S: float = 0.003


# ---------------------------------------------------------------------------
# Pure helper functions
# ---------------------------------------------------------------------------

def generate_bezier_path(
    start: tuple[int, int],
    end: tuple[int, int],
    num_points: int,
    noise: float = 1.0,
) -> list[tuple[int, int]]:
    """Generate a cubic Bezier curve path between two screen coordinates.

    Control points are offset perpendicular to the direct line, scaled by
    *noise*. Returns *num_points* evenly spaced samples clamped to screen.
    """
    sx, sy = start
    ex, ey = end
    dx, dy = ex - sx, ey - sy
    dist = math.hypot(dx, dy)

    if dist < 1 or num_points < 2:
        return [start, end]

    # Perpendicular unit vector
    perp_x, perp_y = -dy / dist, dx / dist

    # Two control points at ~1/3 and ~2/3 along the line
    offset_scale = dist * _BEZIER_CURVE_FACTOR * noise
    off1 = random.gauss(0, offset_scale)
    off2 = random.gauss(0, offset_scale)

    c1x = sx + dx / 3 + perp_x * off1
    c1y = sy + dy / 3 + perp_y * off1
    c2x = sx + 2 * dx / 3 + perp_x * off2
    c2y = sy + 2 * dy / 3 + perp_y * off2

    # Sample the cubic Bezier
    path: list[tuple[int, int]] = []
    for i in range(num_points):
        t = i / (num_points - 1)
        u = 1 - t
        bx = u**3 * sx + 3 * u**2 * t * c1x + 3 * u * t**2 * c2x + t**3 * ex
        by = u**3 * sy + 3 * u**2 * t * c1y + 3 * u * t**2 * c2y + t**3 * ey

        px = max(0, min(SCREEN_WIDTH - 1, round(bx)))
        py = max(0, min(SCREEN_HEIGHT - 1, round(by)))
        path.append((px, py))

    # Guarantee exact endpoints
    path[0] = start
    path[-1] = end
    return path


def compute_step_count(distance_px: float, speed: float = 1.0) -> int:
    """Compute interpolation steps based on distance (Fitts's law-inspired).

    More steps for longer distances, sublinear growth.
    Returns between 5 and 80.
    """
    if distance_px < 1:
        return 5
    raw = int(15 * math.log2(1 + distance_px / 50) * speed)
    return max(5, min(80, raw))


def compute_step_delays(
    num_steps: int,
    base_delay_s: float = 0.005,
    variance: float = 0.3,
) -> list[float]:
    """Generate per-step delays with sinusoidal ease-in-out profile.

    Slower at start and end, faster in the middle. Each delay is perturbed
    by +/- *variance* fraction.
    """
    if num_steps <= 0:
        return []
    if num_steps == 1:
        return [base_delay_s]

    delays: list[float] = []
    for i in range(num_steps):
        # Sine profile: 1.5x at edges, ~0.8x in the middle
        phase = math.sin(math.pi * i / (num_steps - 1))
        raw_delay = base_delay_s * (1.5 - 0.7 * phase)

        if variance > 0:
            factor = random.uniform(1 - variance, 1 + variance)
            raw_delay *= factor

        delays.append(max(0.001, raw_delay))

    return delays


def apply_jitter(
    x: int,
    y: int,
    max_offset_px: int = 2,
) -> tuple[int, int]:
    """Add small random offset to click target within a circle.

    Uses polar coordinates for uniform distribution. Clamps to screen.
    """
    if max_offset_px <= 0:
        return (x, y)

    angle = random.uniform(0, 2 * math.pi)
    radius = math.sqrt(random.uniform(0, 1)) * max_offset_px
    jx = round(x + radius * math.cos(angle))
    jy = round(y + radius * math.sin(angle))

    jx = max(0, min(SCREEN_WIDTH - 1, jx))
    jy = max(0, min(SCREEN_HEIGHT - 1, jy))
    return (jx, jy)


def generate_overshoot_path(
    target: tuple[int, int],
    approach_direction: tuple[float, float],
    overshoot_px: int = 15,
) -> list[tuple[int, int]]:
    """Generate a short path that overshoots the target then corrects back.

    *approach_direction* is the (dx, dy) vector of the incoming movement.
    Returns 3-5 points: overshoot point → correction → target.
    """
    tx, ty = target
    dx, dy = approach_direction
    dist = math.hypot(dx, dy)
    if dist < 1:
        return [target]

    # Normalize direction
    nx, ny = dx / dist, dy / dist

    # Overshoot point: extend past target
    overshoot_amount = random.uniform(0.4, 1.0) * overshoot_px
    ox = round(tx + nx * overshoot_amount)
    oy = round(ty + ny * overshoot_amount)
    ox = max(0, min(SCREEN_WIDTH - 1, ox))
    oy = max(0, min(SCREEN_HEIGHT - 1, oy))

    # Midpoint correction (between overshoot and target, slight offset)
    mx = round((ox + tx) / 2 + random.gauss(0, 1))
    my = round((oy + ty) / 2 + random.gauss(0, 1))
    mx = max(0, min(SCREEN_WIDTH - 1, mx))
    my = max(0, min(SCREEN_HEIGHT - 1, my))

    return [(ox, oy), (mx, my), target]


def generate_micro_movements(
    center: tuple[int, int],
    num_frames: int = 3,
    amplitude_px: float = 0.8,
) -> list[tuple[int, int]]:
    """Generate tiny tremor movements simulating hand settling.

    Small random offsets from *center*, each within *amplitude_px*.
    """
    if num_frames <= 0 or amplitude_px <= 0:
        return []

    cx, cy = center
    points: list[tuple[int, int]] = []
    for _ in range(num_frames):
        angle = random.uniform(0, 2 * math.pi)
        r = random.uniform(0, amplitude_px)
        px = max(0, min(SCREEN_WIDTH - 1, round(cx + r * math.cos(angle))))
        py = max(0, min(SCREEN_HEIGHT - 1, round(cy + r * math.sin(angle))))
        points.append((px, py))
    return points


# ---------------------------------------------------------------------------
# Keyboard helper functions
# ---------------------------------------------------------------------------

PUNCTUATION_CHARS: set[str] = set(".,!?:;")


def compute_typing_delays(
    text: str,
    base_delay_s: float = 0.04,
    variance: float = 0.3,
    punctuation_pause_s: float = 0.08,
    word_pause_s: float = 0.04,
    speed: float = 1.0,
) -> list[float]:
    """Compute per-character inter-key delays for a text string.

    Applies:
    - Base delay scaled by *speed* multiplier
    - Ramp-up for first 4 chars (1.5x → 1.0x linear)
    - Ramp-down for last 2 chars of long text (1.0x → 1.3x)
    - Extra pause after punctuation chars (.,!?:;)
    - Extra pause after space (word boundary)
    - Random variance per key

    Returns one delay per character (delay AFTER typing that char).
    """
    n = len(text)
    if n == 0:
        return []

    delays: list[float] = []
    for i, char in enumerate(text):
        delay = base_delay_s * speed

        # Ramp-up: first 4 chars are slower
        if i < 4 and n >= 4:
            delay *= 1.5 - 0.5 * (i / 3)

        # Ramp-down: last 2 chars of long text are slower
        if n >= 8 and i >= n - 2:
            progress = (i - (n - 2))  # 0 or 1
            delay *= 1.0 + 0.3 * progress

        # Punctuation pause
        if char in PUNCTUATION_CHARS:
            delay += punctuation_pause_s

        # Word boundary pause
        if char == " ":
            delay += word_pause_s

        # Random variance
        if variance > 0:
            delay *= random.uniform(1 - variance, 1 + variance)

        delays.append(max(0.005, delay))

    return delays


def _compute_hold_duration(
    base_hold_s: float,
    variance: float,
    min_s: float,
    max_s: float,
) -> float:
    """Apply variance to a hold duration and clamp to [min_s, max_s]."""
    hold = base_hold_s
    if variance > 0:
        hold *= random.uniform(1 - variance, 1 + variance)
    return max(min_s, min(max_s, hold))


def compute_key_hold_duration(
    base_hold_s: float = 0.025,
    variance: float = 0.3,
) -> float:
    """Compute a single key hold duration with variance.

    Returns seconds, clamped to [0.010, 0.060].
    """
    return _compute_hold_duration(base_hold_s, variance, 0.010, 0.060)


def compute_shortcut_hold_duration(
    base_hold_s: float = 0.05,
    variance: float = 0.2,
) -> float:
    """Compute hold duration for keyboard shortcuts (Ctrl+C, etc.).

    Longer and less variable than single-key holds.
    Clamped to [0.030, 0.100].
    """
    return _compute_hold_duration(base_hold_s, variance, 0.030, 0.100)


# ---------------------------------------------------------------------------
# HumanizedMouseController (proxy)
# ---------------------------------------------------------------------------

class HumanizedMouseController:
    """Proxy that adds human-like behavior to MouseController.

    Delegates all HID report building and device I/O to the inner
    MouseController. Intercepts high-level operations to generate
    natural movement paths and timing.

    Usage::

        inner = MouseController(hid_device)
        mouse = HumanizedMouseController(inner, config)
        mouse.click(640, 360)  # natural curved path + jitter + timing
    """

    def __init__(
        self,
        inner: MouseController,
        config: HumanizeConfig,
    ) -> None:
        self._inner = inner
        self._config = config
        self._last_x: int = 0
        self._last_y: int = 0

    # -- Private helpers ---------------------------------------------------

    def _move_naturally(self, target_x: int, target_y: int) -> None:
        """Move cursor from current position to target via Bezier curve."""
        distance = math.hypot(
            target_x - self._last_x, target_y - self._last_y
        )
        if distance < _MIN_MOVEMENT_DISTANCE_PX:
            return  # already at target

        step_count = compute_step_count(distance, self._config.movement_speed)
        path = generate_bezier_path(
            (self._last_x, self._last_y),
            (target_x, target_y),
            step_count,
            self._config.curve_noise,
        )
        delays = compute_step_delays(
            len(path),
            variance=self._config.timing_variance,
        )

        # Optional overshoot
        if random.random() < self._config.overshoot_probability:
            direction = (
                target_x - self._last_x,
                target_y - self._last_y,
            )
            overshoot = generate_overshoot_path(
                (target_x, target_y),
                direction,
                self._config.overshoot_distance_px,
            )
            path.extend(overshoot)
            # Overshoot segments are slower (correction movement)
            for _ in overshoot:
                delays.append(
                    delays[-1] * _OVERSHOOT_DELAY_FACTOR
                    if delays else _OVERSHOOT_FALLBACK_DELAY_S
                )

        # Execute movement
        for i, (px, py) in enumerate(path):
            self._inner.move(px, py)
            if i < len(delays):
                time.sleep(delays[i])

    def _do_micro_movements(self, x: int, y: int) -> None:
        """Execute micro-movements (hand tremor) at a position."""
        if not self._config.micro_movement_enabled:
            return
        tremors = generate_micro_movements(
            (x, y),
            num_frames=random.randint(2, 4),
            amplitude_px=self._config.micro_movement_amplitude_px,
        )
        for tx, ty in tremors:
            self._inner.move(tx, ty)
            time.sleep(
                _MICRO_MOVEMENT_SLEEP_S
                + random.uniform(-_MICRO_MOVEMENT_JITTER_S, _MICRO_MOVEMENT_JITTER_S)
            )
        # Return exactly to target
        self._inner.move(x, y)

    # -- Public interface (mirrors MouseController) ------------------------

    def move(self, x: int, y: int) -> None:
        """Move cursor to position via natural Bezier curve."""
        self._move_naturally(x, y)
        self._last_x = x
        self._last_y = y

    def click(self, x: int, y: int, button: str = "left") -> None:
        """Click with natural approach, jitter, and optional tremor."""
        jx, jy = apply_jitter(x, y, self._config.click_jitter_px)
        self._move_naturally(jx, jy)
        self._do_micro_movements(jx, jy)
        self._inner.click(jx, jy, button)
        self._last_x = jx
        self._last_y = jy

    def double_click(self, x: int, y: int) -> None:
        """Double-click with natural approach and varied inter-click gap."""
        jx, jy = apply_jitter(x, y, self._config.click_jitter_px)
        self._move_naturally(jx, jy)
        self._do_micro_movements(jx, jy)

        # First click
        self._inner.click(jx, jy)
        # Varied inter-click gap (human double-click is 60-120ms)
        gap = 0.08 * random.uniform(0.75, 1.25)
        time.sleep(gap)
        # Second click (no movement, same position)
        self._inner.click(jx, jy)

        self._last_x = jx
        self._last_y = jy

    def triple_click(self, x: int, y: int) -> None:
        """Triple-click with natural approach and varied inter-click gaps."""
        jx, jy = apply_jitter(x, y, self._config.click_jitter_px)
        self._move_naturally(jx, jy)
        self._do_micro_movements(jx, jy)

        self._inner.click(jx, jy)
        time.sleep(0.08 * random.uniform(0.75, 1.25))
        self._inner.click(jx, jy)
        time.sleep(0.08 * random.uniform(0.75, 1.25))
        self._inner.click(jx, jy)

        self._last_x = jx
        self._last_y = jy

    def scroll(
        self,
        x: int,
        y: int,
        direction: str = "down",
        amount: int = 3,
    ) -> None:
        """Scroll with natural approach and per-tick timing variance."""
        self._move_naturally(x, y)

        # Build scroll manually for per-tick timing control
        hid_x, hid_y = _screen_to_hid(x, y)
        if direction not in ("up", "down"):
            raise ValueError(f"Unknown scroll direction: {direction!r}")
        single = -1 if direction == "down" else 1

        # Move to scroll position
        self._inner.send_report(0, hid_x, hid_y)
        time.sleep(0.02 * random.uniform(0.8, 1.2))

        # Per-tick delays with ease-in-out
        tick_delays = compute_step_delays(
            abs(amount),
            base_delay_s=0.05,
            variance=self._config.timing_variance,
        )

        for i in range(abs(amount)):
            self._inner.send_report(0, hid_x, hid_y, single)
            if i < len(tick_delays):
                time.sleep(tick_delays[i])

        # Stop scrolling
        self._inner.send_report(0, hid_x, hid_y, 0)

        self._last_x = x
        self._last_y = y

    def drag(
        self,
        from_x: int,
        from_y: int,
        to_x: int,
        to_y: int,
    ) -> None:
        """Drag with Bezier curve path and eased timing.

        Moves naturally to start, then sends button-held reports along a
        curved path from start to end.
        """
        # Move naturally to drag start
        self._move_naturally(from_x, from_y)
        self._do_micro_movements(from_x, from_y)

        # Compute curved drag path
        distance = math.hypot(to_x - from_x, to_y - from_y)
        step_count = compute_step_count(distance, self._config.movement_speed)
        path = generate_bezier_path(
            (from_x, from_y),
            (to_x, to_y),
            step_count,
            self._config.curve_noise * 0.5,  # less noise during drag
        )
        delays = compute_step_delays(
            len(path),
            base_delay_s=0.008,
            variance=self._config.timing_variance,
        )

        # Move to start (no button)
        from_hx, from_hy = _screen_to_hid(from_x, from_y)
        self._inner.send_report(0, from_hx, from_hy)
        time.sleep(0.02 * random.uniform(0.8, 1.2))

        # Press button
        self._inner.send_report(BUTTON_LEFT, from_hx, from_hy)
        time.sleep(0.05 * random.uniform(0.8, 1.2))

        # Drag along curved path with button held
        for i, (px, py) in enumerate(path):
            hx, hy = _screen_to_hid(px, py)
            self._inner.send_report(BUTTON_LEFT, hx, hy)
            if i < len(delays):
                time.sleep(delays[i])

        # Release at destination
        to_hx, to_hy = _screen_to_hid(to_x, to_y)
        self._inner.send_report(0, to_hx, to_hy)

        self._last_x = to_x
        self._last_y = to_y


# ---------------------------------------------------------------------------
# HumanizedKeyboardController (proxy)
# ---------------------------------------------------------------------------

class HumanizedKeyboardController:
    """Proxy that adds human-like timing to KeyboardController.

    Delegates all HID report building and device I/O to the inner
    KeyboardController. Intercepts type_text() and press_keys() to
    introduce variable inter-key timing.

    Usage::

        inner = KeyboardController(hid_device)
        kb = HumanizedKeyboardController(inner, config)
        kb.type_text("hello world")   # natural typing rhythm
        kb.press_keys(["ctrl", "c"])  # varied hold duration
    """

    def __init__(
        self,
        inner: KeyboardController,
        config: HumanizeConfig,
    ) -> None:
        self._inner = inner
        self._config = config

    def type_text(self, text: str) -> None:
        """Type a string with human-like variable timing."""
        delays = compute_typing_delays(
            text,
            variance=self._config.typing_variance,
            punctuation_pause_s=self._config.punctuation_pause_ms / 1000,
            word_pause_s=self._config.word_pause_ms / 1000,
            speed=self._config.typing_speed,
        )

        delay_idx = 0
        for char in text:
            if char == "\n":
                self.press_keys(["enter"])
                if delay_idx < len(delays):
                    time.sleep(delays[delay_idx])
                delay_idx += 1
                continue

            if char == "\t":
                self.press_keys(["tab"])
                if delay_idx < len(delays):
                    time.sleep(delays[delay_idx])
                delay_idx += 1
                continue

            if char not in CHAR_MAP:
                raise ValueError(f"Unsupported character: {char!r}")

            usage_id, needs_shift = CHAR_MAP[char]
            modifier = MODIFIER_MAP["shift"] if needs_shift else 0

            # Press key
            report = _kb_build_report(modifier, [usage_id])
            self._inner.send_raw(report)
            time.sleep(compute_key_hold_duration(
                variance=self._config.typing_variance,
            ))

            # Release
            self._inner.release_all()

            # Inter-key delay
            if delay_idx < len(delays):
                time.sleep(delays[delay_idx])
            delay_idx += 1

    def press_keys(self, keys: list[str]) -> None:
        """Press a key combination with varied hold duration."""
        modifier = 0
        keycodes: list[int] = []

        for key in keys:
            key_lower = key.lower()
            if key_lower in MODIFIER_MAP:
                modifier |= MODIFIER_MAP[key_lower]
            elif key_lower in SPECIAL_KEY_MAP:
                keycodes.append(SPECIAL_KEY_MAP[key_lower])
            elif key_lower in CHAR_MAP:
                usage_id, needs_shift = CHAR_MAP[key_lower]
                if needs_shift:
                    modifier |= MODIFIER_MAP["shift"]
                keycodes.append(usage_id)
            else:
                raise ValueError(f"Unknown key: {key}")

        report = _kb_build_report(modifier, keycodes)
        self._inner.send_raw(report)
        time.sleep(compute_shortcut_hold_duration(
            variance=self._config.typing_variance,
        ))
        self._inner.release_all()
