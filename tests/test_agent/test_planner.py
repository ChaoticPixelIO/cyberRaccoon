"""Tests for the task planner — step parsing and plan/replan logic."""

from __future__ import annotations

import pytest

from agent.planner import PlanStep, TaskPlanner, parse_steps, VALIDATE_PLAN_SYSTEM_PROMPT


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

    def test_missing_expected_outcome(self) -> None:
        text = "1. Open terminal\n2. Type hello"
        steps = parse_steps(text)
        assert steps[0].expected_outcome == ""
        assert steps[1].expected_outcome == ""

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
        assert steps[1].expected_outcome == ""
        assert steps[2].expected_outcome == "Login form filled"


# ===========================================================================
# Validate Plan
# ===========================================================================

class TestValidatePlan:
    """Tests for TaskPlanner.validate_plan()."""

    def test_continue_verdict(self) -> None:
        planner = MockTaskPlanner("CONTINUE")
        step = PlanStep(number=1, goal="Open Chrome", expected_outcome="Chrome visible")
        remaining = [PlanStep(number=2, goal="Go to site")]
        verdict, new_steps, reason = planner.validate_plan(
            task_goal="browse", screenshot_base64="fake",
            completed_step=step, remaining_steps=remaining,
        )
        assert verdict == "continue"
        assert new_steps is None
        assert reason == ""

    def test_replan_verdict(self) -> None:
        planner = MockTaskPlanner(
            "REPLAN: Wrong browser opened.\n"
            "1. Close Edge browser\n"
            "2. Open Chrome browser\n"
            "3. Go to example.com"
        )
        step = PlanStep(number=1, goal="Open browser")
        remaining = [PlanStep(number=2, goal="Go to site")]
        verdict, new_steps, reason = planner.validate_plan(
            task_goal="browse", screenshot_base64="fake",
            completed_step=step, remaining_steps=remaining,
        )
        assert verdict == "replan"
        assert new_steps is not None
        assert len(new_steps) == 3
        assert "Wrong browser" in reason

    def test_escalate_verdict(self) -> None:
        planner = MockTaskPlanner(
            "ESCALATE: Login page detected. Please log in manually."
        )
        step = PlanStep(number=2, goal="Navigate to taobao.com")
        remaining = [PlanStep(number=3, goal="Search")]
        verdict, new_steps, reason = planner.validate_plan(
            task_goal="buy raspberry pi", screenshot_base64="fake",
            completed_step=step, remaining_steps=remaining,
        )
        assert verdict == "escalate"
        assert new_steps is None
        assert "Login" in reason or "login" in reason

    def test_llm_failure_defaults_to_continue(self) -> None:
        planner = MockTaskPlanner(None)
        step = PlanStep(number=1, goal="Open Chrome")
        remaining = [PlanStep(number=2, goal="Go to site")]
        verdict, _, _ = planner.validate_plan(
            task_goal="browse", screenshot_base64="fake",
            completed_step=step, remaining_steps=remaining,
        )
        assert verdict == "continue"

    def test_unparseable_response_defaults_to_continue(self) -> None:
        planner = MockTaskPlanner("I don't understand the question.")
        step = PlanStep(number=1, goal="Open Chrome")
        remaining = [PlanStep(number=2, goal="Go to site")]
        verdict, _, _ = planner.validate_plan(
            task_goal="browse", screenshot_base64="fake",
            completed_step=step, remaining_steps=remaining,
        )


# ===========================================================================
# Validate Plan Prompt Structure (D-06, D-07)
# ===========================================================================

class TestValidatePlanPromptStructure:
    """Tests for system/user prompt split in validate_plan (D-06, D-07)."""

    def test_system_prompt_is_static(self) -> None:
        """VALIDATE_PLAN_SYSTEM_PROMPT has no format placeholders."""
        assert "{step_number}" not in VALIDATE_PLAN_SYSTEM_PROMPT
        assert "{step_goal}" not in VALIDATE_PLAN_SYSTEM_PROMPT
        assert "{expected_outcome}" not in VALIDATE_PLAN_SYSTEM_PROMPT
        assert "{task_goal}" not in VALIDATE_PLAN_SYSTEM_PROMPT
        assert "{remaining_steps}" not in VALIDATE_PLAN_SYSTEM_PROMPT

    def test_system_prompt_has_verdict_instructions(self) -> None:
        """System prompt contains the CONTINUE/REPLAN/ESCALATE instructions."""
        assert "CONTINUE" in VALIDATE_PLAN_SYSTEM_PROMPT
        assert "REPLAN" in VALIDATE_PLAN_SYSTEM_PROMPT
        assert "ESCALATE" in VALIDATE_PLAN_SYSTEM_PROMPT

    def test_user_text_contains_step_context(self) -> None:
        """validate_plan passes step-specific context as user_text."""
        planner = MockTaskPlanner("CONTINUE")
        step = PlanStep(
            number=3, goal="Click save button",
            expected_outcome="Save dialog dismissed",
        )
        remaining = [PlanStep(number=4, goal="Verify saved")]
        planner.validate_plan(
            task_goal="save the document", screenshot_base64="fake",
            completed_step=step, remaining_steps=remaining,
        )
        user_text = planner._last_user_text
        assert "3" in user_text  # step_number
        assert "Click save button" in user_text  # step_goal
        assert "Save dialog dismissed" in user_text  # expected_outcome
        assert "save the document" in user_text  # task_goal
        assert "Verify saved" in user_text  # remaining step

    def test_user_text_contains_skill(self) -> None:
        """When skill_text is provided, it appears in user_text."""
        planner = MockTaskPlanner("CONTINUE")
        step = PlanStep(number=1, goal="Open BIOS")
        remaining = [PlanStep(number=2, goal="Navigate")]
        planner.validate_plan(
            task_goal="configure BIOS", screenshot_base64="fake",
            completed_step=step, remaining_steps=remaining,
            skill_text="# BIOS Navigation Skill\n\nUse F2 to enter setup.",
        )
        user_text = planner._last_user_text
        assert "BIOS Navigation Skill" in user_text
        assert "F2 to enter setup" in user_text

    def test_skill_not_in_system_prompt(self) -> None:
        """Skill text must not be appended to the system prompt."""
        planner = MockTaskPlanner("CONTINUE")
        step = PlanStep(number=1, goal="Open BIOS")
        remaining = [PlanStep(number=2, goal="Navigate")]
        planner.validate_plan(
            task_goal="configure BIOS", screenshot_base64="fake",
            completed_step=step, remaining_steps=remaining,
            skill_text="# BIOS Navigation Skill",
        )
        assert "BIOS Navigation Skill" not in planner._last_system_prompt

    def test_system_prompt_not_empty_user_text_not_empty(self) -> None:
        """Neither system_prompt nor user_text should be empty."""
        planner = MockTaskPlanner("CONTINUE")
        step = PlanStep(number=1, goal="Open Chrome")
        remaining = [PlanStep(number=2, goal="Go to site")]
        planner.validate_plan(
            task_goal="browse", screenshot_base64="fake",
            completed_step=step, remaining_steps=remaining,
        )
        assert planner._last_system_prompt != ""
        assert planner._last_user_text != ""

    def test_validate_uses_system_prompt_constant(self) -> None:
        """validate_plan passes VALIDATE_PLAN_SYSTEM_PROMPT as system_prompt."""
        planner = MockTaskPlanner("CONTINUE")
        step = PlanStep(number=1, goal="Open Chrome")
        remaining = [PlanStep(number=2, goal="Go to site")]
        planner.validate_plan(
            task_goal="browse", screenshot_base64="fake",
            completed_step=step, remaining_steps=remaining,
        )
        assert planner._last_system_prompt == VALIDATE_PLAN_SYSTEM_PROMPT


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
        from agent.planner import PLANNING_SYSTEM_PROMPT
        assert "1-5 agent actions" not in PLANNING_SYSTEM_PROMPT

    def test_replan_prompt_no_fixed_action_rule(self) -> None:
        """PLAN-02: REPLAN prompt also must not have the fixed rule."""
        from agent.planner import REPLAN_SYSTEM_PROMPT
        assert "1-5 agent actions" not in REPLAN_SYSTEM_PROMPT

    def test_planning_prompt_has_granularity_examples(self) -> None:
        """PLAN-01: Prompt contains calibration examples for different complexity levels."""
        from agent.planner import PLANNING_SYSTEM_PROMPT
        assert "Simple tasks" in PLANNING_SYSTEM_PROMPT or "simple tasks" in PLANNING_SYSTEM_PROMPT
        assert "Complex tasks" in PLANNING_SYSTEM_PROMPT or "complex tasks" in PLANNING_SYSTEM_PROMPT
        assert "Application Skill" in PLANNING_SYSTEM_PROMPT

    def test_planning_prompt_has_actions_tag_instruction(self) -> None:
        """PLAN-03/D-08: Prompt instructs LLM to add [ACTIONS: N] tags."""
        from agent.planner import PLANNING_SYSTEM_PROMPT
        assert "[ACTIONS:" in PLANNING_SYSTEM_PROMPT

    def test_replan_prompt_has_actions_tag_instruction(self) -> None:
        """Consistency: REPLAN prompt also instructs [ACTIONS: N] tags."""
        from agent.planner import REPLAN_SYSTEM_PROMPT
        assert "[ACTIONS:" in REPLAN_SYSTEM_PROMPT
