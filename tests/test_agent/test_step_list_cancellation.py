"""Pure-function tests for _cancel_and_append.

Covers the step status state machine: when a step's verification fails,
the current step is cancelled and a re-plan is appended to the step list.
"""
from __future__ import annotations

import pytest

try:
    from cyberraccoon.agent.planner import PlanStep
    from cyberraccoon.agent.workflow_runner import _cancel_and_append
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _AVAILABLE,
    reason="_cancel_and_append helper not yet present",
)


def _step(n: int, status: str = "pending") -> PlanStep:
    return PlanStep(number=n, goal=f"step {n}", status=status)


class TestCancelAndAppend:
    """STEPS-01: cancelled steps are preserved (not deleted)."""

    def test_marks_remaining_cancelled(self) -> None:
        steps = [_step(1, "done"), _step(2, "done"), _step(3), _step(4), _step(5)]
        result, cancelled = _cancel_and_append(steps, completed_index=1, new_steps=[])
        assert len(result) == 5  # nothing deleted
        assert result[2].status == "cancelled"
        assert result[3].status == "cancelled"
        assert result[4].status == "cancelled"
        assert cancelled == [3, 4, 5]

    def test_does_not_delete_steps_when_appending(self) -> None:
        steps = [_step(1, "done"), _step(2), _step(3), _step(4), _step(5)]
        new = [_step(0), _step(0)]  # numbers will be reassigned
        result, _ = _cancel_and_append(steps, completed_index=0, new_steps=new)
        assert len(result) == 5 + 2

    def test_done_steps_preserved_as_done(self) -> None:
        steps = [_step(1, "done"), _step(2, "done"), _step(3), _step(4)]
        result, cancelled = _cancel_and_append(steps, completed_index=1, new_steps=[])
        assert result[0].status == "done"
        assert result[1].status == "done"
        assert 1 not in cancelled
        assert 2 not in cancelled

    def test_already_cancelled_not_double_marked(self) -> None:
        # First re-plan
        steps = [_step(1, "done"), _step(2), _step(3)]
        r1, c1 = _cancel_and_append(steps, completed_index=0, new_steps=[_step(0)])
        assert c1 == [2, 3]
        # Second re-plan over the same list — already-cancelled tail must NOT be re-emitted
        r2, c2 = _cancel_and_append(r1, completed_index=3, new_steps=[_step(0)])
        assert c2 == []  # nothing new to cancel
        assert 2 not in c2 and 3 not in c2


class TestNumbering:
    """STEPS-02: cancelled keep original numbers, new start at max+1."""

    def test_cancelled_steps_keep_original_numbers(self) -> None:
        steps = [_step(1, "done"), _step(2), _step(3), _step(4)]
        result, _ = _cancel_and_append(steps, completed_index=0, new_steps=[])
        assert [s.number for s in result] == [1, 2, 3, 4]
        assert result[1].status == "cancelled"
        assert result[1].number == 2  # unchanged

    def test_new_steps_get_max_plus_one(self) -> None:
        steps = [_step(1, "done"), _step(2), _step(3)]
        new = [_step(0), _step(0), _step(0)]
        result, _ = _cancel_and_append(steps, completed_index=0, new_steps=new)
        new_nums = [s.number for s in result if s.status == "pending"]
        assert new_nums == [4, 5, 6]

    def test_multiple_replans_growing_list(self) -> None:
        steps = [_step(1, "done"), _step(2), _step(3)]
        r1, _ = _cancel_and_append(steps, 0, [_step(0), _step(0)])
        # r1 = [1(done), 2(cancelled), 3(cancelled), 4(pending), 5(pending)]
        assert [s.number for s in r1] == [1, 2, 3, 4, 5]
        r2, _ = _cancel_and_append(r1, 3, [_step(0)])
        # r2 = [..., 5(cancelled), 6(pending)]
        assert [s.number for s in r2] == [1, 2, 3, 4, 5, 6]
        assert r2[5].number == 6
        assert r2[5].status == "pending"

    def test_empty_new_steps_returns_only_cancellation(self) -> None:
        steps = [_step(1, "done"), _step(2), _step(3)]
        result, cancelled = _cancel_and_append(steps, 0, [])
        assert [s.number for s in result] == [1, 2, 3]
        assert cancelled == [2, 3]

    def test_completed_index_at_end_yields_no_cancellation(self) -> None:
        steps = [_step(1, "done"), _step(2, "done")]
        result, cancelled = _cancel_and_append(steps, 1, [_step(0)])
        assert cancelled == []
        assert result[2].number == 3
        assert result[2].status == "pending"

    def test_new_step_status_defaults_to_pending(self) -> None:
        steps = [_step(1, "done")]
        new = [_step(0, status="")]  # explicitly empty
        result, _ = _cancel_and_append(steps, 0, new)
        assert result[1].status == "pending"

    def test_new_step_status_preserved_if_set(self) -> None:
        steps = [_step(1, "done")]
        new = [_step(0, status="running")]
        result, _ = _cancel_and_append(steps, 0, new)
        assert result[1].status == "running"


class TestEdgeCases:
    def test_empty_input_with_new_steps(self) -> None:
        result, cancelled = _cancel_and_append([], -1, [_step(0)])
        assert cancelled == []
        assert result[0].number == 1

    def test_single_completed_step_then_replan(self) -> None:
        steps = [_step(1, "done")]
        new = [_step(0), _step(0)]
        result, cancelled = _cancel_and_append(steps, 0, new)
        assert cancelled == []
        assert [s.number for s in result] == [1, 2, 3]
