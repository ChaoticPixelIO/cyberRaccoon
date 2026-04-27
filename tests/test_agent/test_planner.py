"""Tests for the task planner — step parsing and plan/replan logic."""

from __future__ import annotations

from typing import Any

import pytest

from cyberraccoon.agent.planner import PlanStep, TaskPlanner, parse_steps

# Feature-detect _format_calibration_lines (plan 02 implementation)
try:
    from cyberraccoon.agent.planner import _format_calibration_lines
    _CALIBRATION_AVAILABLE = True
except ImportError:
    _CALIBRATION_AVAILABLE = False


# ===========================================================================
# Step Parsing
# ===========================================================================

class TestParseSteps:
    """Tests for parse_steps() function."""

    def test_standard_numbered_list(self) -> None:
        text = "1. Open a terminal\n2. Type hello\n3. Press Enter"
        steps = parse_steps(text)
        assert len(steps) == 3
        assert steps[0].number == 1
        assert steps[0].goal == "Open a terminal"
        assert steps[2].goal == "Press Enter"

    def test_parenthesis_format(self) -> None:
        text = "1) Open a terminal\n2) Type hello"
        steps = parse_steps(text)
        assert len(steps) == 2
        assert steps[0].goal == "Open a terminal"

    def test_reboot_tag_detected(self) -> None:
        text = "1. Type shutdown command [REBOOT EXPECTED]\n2. Wait for BIOS"
        steps = parse_steps(text)
        assert steps[0].reboot_expected is True
        assert steps[1].reboot_expected is False

    def test_reboot_tag_removed_from_goal(self) -> None:
        text = "1. Type shutdown /r /fw /t 0 [REBOOT EXPECTED]"
        steps = parse_steps(text)
        assert "[REBOOT" not in steps[0].goal
        assert "shutdown /r /fw /t 0" in steps[0].goal

    def test_reboot_tag_case_insensitive(self) -> None:
        text = "1. Reboot [reboot expected]"
        steps = parse_steps(text)
        assert steps[0].reboot_expected is True

    def test_empty_input(self) -> None:
        assert parse_steps("") == []

    def test_no_numbered_lines(self) -> None:
        text = "Just some text without numbers"
        assert parse_steps(text) == []

    def test_mixed_format(self) -> None:
        text = "Here is the plan:\n1. Step one\n2. Step two\nDone."
        steps = parse_steps(text)
        assert len(steps) == 2

    def test_single_step(self) -> None:
        text = "1. Just do the thing"
        steps = parse_steps(text)
        assert len(steps) == 1
        assert steps[0].goal == "Just do the thing"


# ===========================================================================
# PlanStep.format_for_agent
# ===========================================================================

class TestPlanStepFormat:
    """Tests for PlanStep.format_for_agent()."""

    def test_includes_step_number_and_total(self) -> None:
        step = PlanStep(number=2, goal="Click the button")
        result = step.format_for_agent(total_steps=5)
        assert "2 of 5" in result
        assert "Click the button" in result

    def test_includes_completed_steps(self) -> None:
        step = PlanStep(number=3, goal="Do step 3")
        result = step.format_for_agent(
            total_steps=3, completed=["Open terminal", "Type command"],
        )
        assert "Open terminal" in result
        assert "Type command" in result

    def test_includes_stop_signal(self) -> None:
        step = PlanStep(number=1, goal="Open Chrome")
        result = step.format_for_agent(total_steps=2)
        assert "stop" in result.lower() or "Stop" in result

    def test_no_completed_steps(self) -> None:
        step = PlanStep(number=1, goal="First step")
        result = step.format_for_agent(total_steps=3)
        assert "Previous" not in result


# ===========================================================================
# TaskPlanner (with mock LLM)
# ===========================================================================

class MockTaskPlanner(TaskPlanner):
    """TaskPlanner that returns canned responses instead of calling an LLM."""

    def __init__(self, response: str | None = None) -> None:
        super().__init__(provider="anthropic", model="test", api_key="test")
        self._mock_response = response
        self._last_system_prompt: str = ""
        self._last_user_text: str = ""

    def _call_llm(
        self, system_prompt: str, user_text: str,
        screenshot_base64: str | None = None,
    ) -> str | None:
        self._last_system_prompt = system_prompt
        self._last_user_text = user_text
        return self._mock_response


class TestTaskPlanner:
    """Tests for TaskPlanner.plan() and replan()."""

    def test_plan_happy_path(self) -> None:
        planner = MockTaskPlanner(
            "1. Open terminal\n2. Type command\n3. Verify result"
        )
        steps = planner.plan("do something", "fake_screenshot")
        assert len(steps) == 3
        assert steps[0].goal == "Open terminal"

    def test_plan_with_skill_text(self) -> None:
        planner = MockTaskPlanner("1. Follow skill step")
        steps = planner.plan("do something", "fake_screenshot", skill_text="# My Skill")
        assert len(steps) == 1

    def test_plan_fallback_on_none_response(self) -> None:
        planner = MockTaskPlanner(None)
        steps = planner.plan("do something", "fake_screenshot")
        assert len(steps) == 1
        assert steps[0].goal == "do something"

    def test_plan_fallback_on_unparseable_response(self) -> None:
        planner = MockTaskPlanner("I don't know how to do that.")
        steps = planner.plan("do something", "fake_screenshot")
        assert len(steps) == 1
        assert steps[0].goal == "do something"

    def test_replan_happy_path(self) -> None:
        planner = MockTaskPlanner("1. Try different approach\n2. Verify")
        failed = PlanStep(number=3, goal="Original step 3")
        remaining = [failed, PlanStep(number=4, goal="Step 4")]
        result = planner.replan(
            task_goal="big task",
            failed_step=failed,
            failure_reason="button not found",
            screenshot_base64="fake",
            remaining_steps=remaining,
        )
        assert result is not None
        assert len(result) == 2

    def test_replan_returns_none_on_failure(self) -> None:
        planner = MockTaskPlanner(None)
        failed = PlanStep(number=1, goal="Step 1")
        result = planner.replan(
            task_goal="task",
            failed_step=failed,
            failure_reason="error",
            screenshot_base64="fake",
            remaining_steps=[failed],
        )
        assert result is None


# ===========================================================================
# Expected Outcome Parsing
# ===========================================================================

class TestExpectedOutcomeParsing:
    """Tests for expected_outcome extraction in parse_steps()."""

    def test_expected_outcome_parsed(self) -> None:
        text = (
            "1. Open Chrome browser\n"
            "Expected: Chrome window visible with address bar\n"
            "2. Go to example.com\n"
            "Expected: example.com homepage loaded\n"
        )
        steps = parse_steps(text)
        assert len(steps) == 2
        assert steps[0].expected_outcome == "Chrome window visible with address bar"
        assert steps[1].expected_outcome == "example.com homepage loaded"

    def test_missing_expected_outcome_auto_generated(self) -> None:
        text = "1. Open terminal\n2. Type hello"
        steps = parse_steps(text)
        assert steps[0].expected_outcome != ""
        assert "terminal" in steps[0].expected_outcome.lower()
        assert steps[1].expected_outcome != ""

    def test_mixed_expected_outcomes(self) -> None:
        text = (
            "1. Open Chrome\n"
            "Expected: Chrome is open\n"
            "2. Click login\n"
            "3. Type password\n"
            "Expected: Login form filled\n"
        )
        steps = parse_steps(text)
        assert steps[0].expected_outcome == "Chrome is open"
        assert steps[1].expected_outcome != ""  # auto-generated fallback
        assert steps[2].expected_outcome == "Login form filled"


# ===========================================================================
# Action Count Parsing
# ===========================================================================

class TestActionCountParsing:
    """Tests for [ACTIONS: N] tag extraction in parse_steps()."""

    def test_single_action_count(self) -> None:
        text = "1. Open Notepad [ACTIONS: 2]\nExpected: Notepad visible"
        steps = parse_steps(text)
        assert steps[0].expected_actions == 2

    def test_range_action_count_stores_upper_bound(self) -> None:
        text = "1. Navigate to settings [ACTIONS: 3-5]"
        steps = parse_steps(text)
        assert steps[0].expected_actions == 5  # upper bound per D-04

    def test_no_action_tag_returns_none(self) -> None:
        text = "1. Open Chrome"
        steps = parse_steps(text)
        assert steps[0].expected_actions is None

    def test_action_tag_removed_from_goal(self) -> None:
        text = "1. Open Chrome and search [ACTIONS: 3]"
        steps = parse_steps(text)
        assert "[ACTIONS" not in steps[0].goal
        assert steps[0].goal == "Open Chrome and search"

    def test_action_tag_case_insensitive(self) -> None:
        text = "1. Click button [actions: 1]"
        steps = parse_steps(text)
        assert steps[0].expected_actions == 1

    def test_action_tag_no_space_after_colon(self) -> None:
        text = "1. Type command [ACTIONS:4]"
        steps = parse_steps(text)
        assert steps[0].expected_actions == 4

    def test_combined_reboot_and_actions_tags(self) -> None:
        text = "1. Reboot into BIOS [REBOOT EXPECTED] [ACTIONS: 3-5]"
        steps = parse_steps(text)
        assert steps[0].reboot_expected is True
        assert steps[0].expected_actions == 5
        assert "[REBOOT" not in steps[0].goal
        assert "[ACTIONS" not in steps[0].goal

    def test_multiple_steps_with_mixed_actions(self) -> None:
        text = (
            "1. Open Chrome [ACTIONS: 2]\n"
            "Expected: Chrome visible\n"
            "2. Type URL\n"
            "Expected: URL typed\n"
            "3. Press Enter [ACTIONS: 1]\n"
        )
        steps = parse_steps(text)
        assert steps[0].expected_actions == 2
        assert steps[1].expected_actions is None
        assert steps[2].expected_actions == 1

    def test_format_for_agent_excludes_expected_actions(self) -> None:
        step = PlanStep(number=1, goal="Open Notepad", expected_actions=2)
        result = step.format_for_agent(total_steps=3)
        assert "expected_actions" not in result
        assert "ACTIONS" not in result


# ===========================================================================
# Granularity Guidelines
# ===========================================================================

class TestGranularityGuidelines:
    """Tests for prompt granularity guidelines (PLAN-01, PLAN-02)."""

    def test_planning_prompt_no_fixed_action_rule(self) -> None:
        """PLAN-02: The rigid '1-5 agent actions' rule must be removed."""
        from cyberraccoon.agent.planner import PLANNING_SYSTEM_PROMPT
        assert "1-5 agent actions" not in PLANNING_SYSTEM_PROMPT

    def test_replan_prompt_no_fixed_action_rule(self) -> None:
        """PLAN-02: REPLAN prompt also must not have the fixed rule."""
        from cyberraccoon.agent.planner import REPLAN_SYSTEM_PROMPT
        assert "1-5 agent actions" not in REPLAN_SYSTEM_PROMPT

    def test_planning_prompt_has_granularity_examples(self) -> None:
        """PLAN-01: Prompt contains calibration examples for different complexity levels."""
        from cyberraccoon.agent.planner import PLANNING_SYSTEM_PROMPT
        assert "Simple tasks" in PLANNING_SYSTEM_PROMPT or "simple tasks" in PLANNING_SYSTEM_PROMPT
        assert "Complex tasks" in PLANNING_SYSTEM_PROMPT or "complex tasks" in PLANNING_SYSTEM_PROMPT
        assert "Application Skill" in PLANNING_SYSTEM_PROMPT

    def test_planning_prompt_has_actions_tag_instruction(self) -> None:
        """PLAN-03/D-08: Prompt instructs LLM to add [ACTIONS: N] tags."""
        from cyberraccoon.agent.planner import PLANNING_SYSTEM_PROMPT
        assert "[ACTIONS:" in PLANNING_SYSTEM_PROMPT

    def test_replan_prompt_has_actions_tag_instruction(self) -> None:
        """Consistency: REPLAN prompt also instructs [ACTIONS: N] tags."""
        from cyberraccoon.agent.planner import REPLAN_SYSTEM_PROMPT
        assert "[ACTIONS:" in REPLAN_SYSTEM_PROMPT


# ===========================================================================
# Calibration Line Formatting (BUDGET-02)
# ===========================================================================

@pytest.mark.skipif(
    not _CALIBRATION_AVAILABLE,
    reason="_format_calibration_lines not yet implemented (plan 02)",
)
class TestCalibrationLines:
    """Tests for _format_calibration_lines() helper (D-01/D-02/D-03 format)."""

    def test_step_within_budget(self) -> None:
        """Step that used fewer actions than estimated shows OK (D-03)."""
        entry = {
            "step_number": 1,
            "step_goal": "Click OK",
            "expected_actions": 3,
            "actions_used": 2,
        }
        result = _format_calibration_lines([entry])
        assert result == "Step 1: Click OK [estimated 3, used 2] OK"

    def test_step_over_budget(self) -> None:
        """Step that exceeded budget shows +N over (D-03)."""
        entry = {
            "step_number": 2,
            "step_goal": "Navigate to Settings",
            "expected_actions": 3,
            "actions_used": 7,
        }
        result = _format_calibration_lines([entry])
        assert result == "Step 2: Navigate to Settings [estimated 3, used 7] +4 over"

    def test_step_no_estimate(self) -> None:
        """Step with no expected_actions shows [no estimate, used N] (D-02)."""
        entry = {
            "step_number": 3,
            "step_goal": "Open browser",
            "expected_actions": None,
            "actions_used": 5,
        }
        result = _format_calibration_lines([entry])
        assert result == "Step 3: Open browser [no estimate, used 5]"

    def test_step_exact_budget(self) -> None:
        """Step that used exactly the estimated actions shows OK (D-03)."""
        entry = {
            "step_number": 1,
            "step_goal": "Type hello",
            "expected_actions": 2,
            "actions_used": 2,
        }
        result = _format_calibration_lines([entry])
        assert result == "Step 1: Type hello [estimated 2, used 2] OK"

    def test_multiple_steps(self) -> None:
        """Multiple entries produce one line per step, joined by newlines."""
        entries = [
            {"step_number": 1, "step_goal": "Open app", "expected_actions": 2, "actions_used": 1},
            {"step_number": 2, "step_goal": "Click menu", "expected_actions": 3, "actions_used": 5},
            {"step_number": 3, "step_goal": "Save file", "expected_actions": None, "actions_used": 4},
        ]
        result = _format_calibration_lines(entries)
        lines = result.split("\n")
        assert len(lines) == 3
        assert "Step 1:" in lines[0]
        assert "Step 2:" in lines[1]
        assert "Step 3:" in lines[2]

    def test_missing_keys_tolerated(self) -> None:
        """Entry missing step_goal does NOT crash (graceful degradation)."""
        entry = {
            "step_number": 1,
            "expected_actions": 2,
            "actions_used": 3,
        }
        # Must not raise KeyError
        result = _format_calibration_lines([entry])
        assert isinstance(result, str)
        # Should still produce some output (with fallback or skip)
        assert "Step 1" in result or result == ""


# ===========================================================================
# Replan Calibration (BUDGET-02)
# ===========================================================================

class CaptureReplanPlanner(MockTaskPlanner):
    """MockTaskPlanner that captures user_text from replan's _call_llm call."""

    def __init__(self) -> None:
        super().__init__("1. Revised step [ACTIONS: 3]")
        self._last_user_text: str | None = None

    def _call_llm(
        self, system_prompt: str, user_text: str,
        screenshot_base64: str | None = None, **kwargs,
    ) -> str | None:
        self._last_user_text = user_text
        return super()._call_llm(system_prompt, user_text, screenshot_base64, **kwargs)


@pytest.mark.skipif(
    not _CALIBRATION_AVAILABLE,
    reason="calibration not yet implemented (plan 02)",
)
class TestReplanCalibration:
    """Tests for calibration data threading into replan() prompt."""

    def test_calibration_in_prompt(self) -> None:
        """replan() with calibration_data includes calibration section in prompt."""
        planner = CaptureReplanPlanner()
        calibration = [
            {
                "step_number": 1,
                "step_goal": "Open Chrome",
                "expected_actions": 2,
                "actions_used": 5,
            },
        ]
        planner.replan(
            task_goal="task",
            failed_step=PlanStep(1, "s1"),
            failure_reason="err",
            screenshot_base64="fake",
            remaining_steps=[PlanStep(2, "s2")],
            calibration_data=calibration,
        )
        assert planner._last_user_text is not None
        assert "Calibration data" in planner._last_user_text
        assert "estimated 2, used 5" in planner._last_user_text

    def test_no_calibration_backward_compat(self) -> None:
        """replan() without calibration_data still works (backward compat)."""
        planner = CaptureReplanPlanner()
        result = planner.replan(
            task_goal="task",
            failed_step=PlanStep(1, "s1"),
            failure_reason="err",
            screenshot_base64="fake",
            remaining_steps=[PlanStep(2, "s2")],
        )
        assert result is not None
        assert planner._last_user_text is not None
        assert "Calibration data" not in planner._last_user_text


# ===========================================================================
# Replan Prompt Calibration Rule (BUDGET-02, D-04)
# ===========================================================================

def test_calibration_rule_in_replan_prompt() -> None:
    """REPLAN_SYSTEM_PROMPT contains calibration adjustment rule (D-04).

    NOTE: This test intentionally has NO skipif guard. It tests an
    existing constant (REPLAN_SYSTEM_PROMPT) and should FAIL (not skip)
    until plan 02 adds the calibration rule.
    """
    from cyberraccoon.agent.planner import REPLAN_SYSTEM_PROMPT
    assert "[ACTIONS:" in REPLAN_SYSTEM_PROMPT
    # D-04: must instruct LLM to adjust estimates based on calibration data
    prompt_lower = REPLAN_SYSTEM_PROMPT.lower()
    assert "calibrat" in prompt_lower, (
        "REPLAN_SYSTEM_PROMPT must contain a calibration instruction "
        "(e.g., 'Calibration Data' section reference)"
    )


# ===========================================================================
# Paused-State Rewrite (Phase 8, 08-02)
# ===========================================================================

class MockRewritePlanner(TaskPlanner):
    """TaskPlanner that captures LLM call args for rewrite_plan tests."""

    def __init__(self, response: str | None = None) -> None:
        super().__init__(provider="anthropic", model="test", api_key="test")
        self._mock_response = response
        self._last_system_prompt: str = ""
        self._last_user_text: str = ""

    def _call_llm(
        self, system_prompt: str, user_text: str,
        screenshot_base64: str | None = None,
        **kwargs: Any,
    ) -> str | None:
        self._last_system_prompt = system_prompt
        self._last_user_text = user_text
        return self._mock_response


class TestPausedStateRewrite:
    """Tests for paused-state rewrite support in TaskPlanner."""

    _REWRITE_JSON = (
        '{"action": "rewrite", "steps": ['
        '{"number": 1, "goal": "Open Chrome", "expected_actions": 2, '
        '"expected_outcome": "Chrome visible", "reboot_expected": false},'
        '{"number": 2, "goal": "Type URL", "expected_actions": 1, '
        '"expected_outcome": "URL entered", "reboot_expected": false},'
        '{"number": 3, "goal": "Click Go", "expected_actions": 1, '
        '"expected_outcome": "Page loaded", "reboot_expected": false},'
        '{"number": 4, "goal": "Verify page", "expected_actions": 1, '
        '"expected_outcome": "Correct page shown", "reboot_expected": false}'
        '], "summary": "rewrote plan"}'
    )

    def _make_steps(self) -> list[PlanStep]:
        return [
            PlanStep(number=1, goal="Open Chrome", expected_actions=2),
            PlanStep(number=2, goal="Type URL", expected_actions=1),
            PlanStep(number=3, goal="Click Go", expected_actions=1),
            PlanStep(number=4, goal="Verify page", expected_actions=1),
        ]

    def test_completed_steps_marked_in_prompt(self) -> None:
        """rewrite_plan with completed_step_numbers marks steps in user_text."""
        planner = MockRewritePlanner(self._REWRITE_JSON)
        steps = self._make_steps()
        planner.rewrite_plan(
            task_goal="Browse web",
            current_steps=steps,
            screenshot_base64="fake",
            modification_request="change step 3",
            completed_step_numbers={1, 2},
        )
        user_text = planner._last_user_text
        # Steps 1 and 2 should have [COMPLETED] prefix
        assert "[COMPLETED] Open Chrome" in user_text
        assert "[COMPLETED] Type URL" in user_text
        # Steps 3 and 4 should NOT have [COMPLETED]
        lines = user_text.split("\n")
        for line in lines:
            if "Click Go" in line and line.strip().startswith("3"):
                assert "[COMPLETED]" not in line
            if "Verify page" in line and line.strip().startswith("4"):
                assert "[COMPLETED]" not in line

    def test_no_completed_markers_without_param(self) -> None:
        """rewrite_plan without completed_step_numbers has no [COMPLETED]."""
        planner = MockRewritePlanner(self._REWRITE_JSON)
        steps = self._make_steps()
        planner.rewrite_plan(
            task_goal="Browse web",
            current_steps=steps,
            screenshot_base64="fake",
            modification_request="change step 3",
        )
        assert "[COMPLETED]" not in planner._last_user_text

    def test_paused_addendum_in_system_prompt(self) -> None:
        """rewrite_plan with completed steps includes paused addendum."""
        planner = MockRewritePlanner(self._REWRITE_JSON)
        steps = self._make_steps()
        planner.rewrite_plan(
            task_goal="Browse web",
            current_steps=steps,
            screenshot_base64="fake",
            modification_request="change step 3",
            completed_step_numbers={1},
        )
        assert "Some steps are marked [COMPLETED]" in planner._last_system_prompt

    def test_no_paused_addendum_without_completed_steps(self) -> None:
        """rewrite_plan without completed steps has no paused addendum."""
        planner = MockRewritePlanner(self._REWRITE_JSON)
        steps = self._make_steps()
        planner.rewrite_plan(
            task_goal="Browse web",
            current_steps=steps,
            screenshot_base64="fake",
            modification_request="change step 3",
        )
        assert "Some steps are marked [COMPLETED]" not in planner._last_system_prompt

    def test_completed_marker_not_in_rewrite_result(self) -> None:
        """[COMPLETED] markers never leak to RewriteResult.steps."""
        # LLM echoes back [COMPLETED] in step goals
        response_with_markers = (
            '{"action": "rewrite", "steps": ['
            '{"number": 1, "goal": "[COMPLETED] Open Chrome", "expected_actions": 2, '
            '"expected_outcome": "Chrome visible", "reboot_expected": false},'
            '{"number": 2, "goal": "New step", "expected_actions": 1, '
            '"expected_outcome": "Done", "reboot_expected": false}'
            '], "summary": "rewrote"}'
        )
        planner = MockRewritePlanner(response_with_markers)
        result = planner.rewrite_plan(
            task_goal="Browse web",
            current_steps=self._make_steps(),
            screenshot_base64="fake",
            modification_request="change",
            completed_step_numbers={1},
        )
        assert result is not None
        assert result.action == "rewrite"
        for step in result.steps:
            assert "[COMPLETED]" not in step.goal

    def test_completed_marker_stripped_from_parsed_steps(self) -> None:
        """parse_steps strips [COMPLETED] prefix from step goals."""
        raw = "1. [COMPLETED] Open Chrome\n2. Type in search bar"
        steps = parse_steps(raw)
        assert len(steps) == 2
        assert steps[0].goal == "Open Chrome"
        assert "[COMPLETED]" not in steps[0].goal
        assert steps[1].goal == "Type in search bar"
