"""Task planner — decomposes a goal into executable steps using LLM reasoning.

Makes a stateless LLM call WITHOUT the computer-use tool. This is
intentional: without tools, the LLM follows skill instructions reliably
instead of going into autonomous agent mode.

The planner sees:
  - A screenshot of the current screen state
  - The task goal
  - Skill text (if any)

It returns a numbered list of concrete steps the agent can execute
one at a time.

Usage::

    planner = TaskPlanner(provider="anthropic", model="claude-sonnet-4-6",
                          api_key="sk-...")
    steps = planner.plan(task_goal, screenshot_b64, skill_text)
    # steps = ["Open a terminal", "Type: shutdown /r /fw /t 0", ...]
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger("M2.planner")

# ---------------------------------------------------------------------------
# Planning system prompt (generic — no app-specific content)
# ---------------------------------------------------------------------------

PLANNING_SYSTEM_PROMPT = """\
You are a task planner for CyberRaccoon, an AI that controls a computer \
via screenshots and keyboard/mouse input.

Given a task and an Application Skill (if provided), break the task into \
small, numbered steps. Each step must be a SINGLE concrete action that \
can be verified from a screenshot.

Rules:
1. If an Application Skill section is provided below, follow its procedure \
EXACTLY. The skill contains domain-specific instructions that override your \
default approach.
2. Adjust step granularity based on task complexity:
   - Simple tasks (open an app, click a button): 1-2 coarse steps.
     Example: "Open Notepad" -> 1 step: "Open Notepad and verify it's visible [ACTIONS: 2]"
   - Moderate tasks (navigate to a page, fill a form): 2-4 steps.
     Example: "Search Google for weather" -> 2 steps
   - Complex tasks or tasks with an Application Skill: 5-10+ detailed steps \
following the skill procedure. Each step is a distinct verifiable checkpoint.
     Example: "Configure BIOS boot order" with skill -> 8-12 steps matching skill sections
   - When a skill is provided, it defines the step structure. Follow its sections \
as step boundaries.
3. After each step's goal text, add an action estimate tag: [ACTIONS: N] for exact \
estimates, or [ACTIONS: N-M] for ranges. This tells the executor how many \
clicks/keystrokes the step should take.
4. Include the EXACT command or text to type when applicable.
5. If a step will cause a system reboot or restart, mark it with the tag: \
[REBOOT EXPECTED]
6. Number steps sequentially: 1, 2, 3, etc.
7. The final step should verify the task is complete.
8. If the task is simple enough to complete in one step, return just one step.
9. After each step, add "Expected: " followed by a brief description of what \
the screen should look like after this step completes.

Output ONLY the numbered step list. No explanation before or after.

Example format:
1. Open Chrome browser [ACTIONS: 2]
Expected: Chrome browser window is visible with address bar
2. Navigate to example.com [ACTIONS: 2-3]
Expected: example.com homepage is loaded in Chrome
"""

VALIDATE_PLAN_SYSTEM_PROMPT = """\
You are a plan validator for CyberRaccoon, an AI that controls a computer \
via screenshots and keyboard/mouse input.

A task is being executed step by step. Look at the screenshot showing the \
current screen state and evaluate whether the plan is still valid.

Compare the screenshot to the expected outcome. Reply with exactly ONE of:

CONTINUE
(The screen matches expectations. Proceed with the remaining steps.)

REPLAN
(The screen does NOT match expectations. Explain what went wrong in one \
sentence, then produce a revised numbered step list from the current state. \
Follow the Application Skill if provided.)

ESCALATE
(The situation requires human intervention that the agent CANNOT perform: \
typing a password, solving a CAPTCHA, completing 2FA, or entering credentials \
on a login page. Explain what the user needs to do.)

Do NOT escalate for simple confirmation dialogs that the agent can click \
through: Windows UAC "Yes/No" prompts, "Are you sure?" dialogs, permission \
consent popups, or any dialog with a clearly correct button to click. \
Just click the appropriate button and CONTINUE.
"""

REPLAN_SYSTEM_PROMPT = """\
You are a task planner for CyberRaccoon, an AI that controls a computer \
via screenshots and keyboard/mouse input.

A previous plan failed at one of its steps. You are given:
- The original task goal
- The step that failed and why
- A screenshot of the CURRENT screen state
- The Application Skill (if any)
- The remaining steps from the original plan

Produce a REVISED numbered step list that picks up from the current state. \
Start numbering from 1. Follow the Application Skill if provided.

Rules:
1. Look at the screenshot to understand where things are NOW.
2. Do NOT repeat steps that already completed successfully.
3. Adjust step granularity to match task complexity. Add an action estimate \
tag [ACTIONS: N] or [ACTIONS: N-M] after each step's goal text.
4. If a step will cause a reboot, mark it: [REBOOT EXPECTED]
5. Output ONLY the numbered step list. No explanation.
"""


# ---------------------------------------------------------------------------
# Step dataclass
# ---------------------------------------------------------------------------

@dataclass
class PlanStep:
    """A single step in a task plan."""

    number: int
    goal: str
    reboot_expected: bool = False
    expected_outcome: str = ""
    expected_actions: int | None = None

    def format_for_agent(
        self,
        total_steps: int,
        completed: list[str] | None = None,
    ) -> str:
        """Format this step as a goal string for agent.run().

        Includes context about previous steps and a clear stop signal.
        """
        parts = []
        if completed:
            parts.append(
                "Previous steps completed: "
                + "; ".join(f"{i+1}) {s}" for i, s in enumerate(completed))
            )
        parts.append(f"Current step: {self.number} of {total_steps}.")
        parts.append(f"Goal: {self.goal}")
        parts.append(
            "When this step is done, stop and report success. "
            "Do NOT continue to the next step."
        )
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Step parser
# ---------------------------------------------------------------------------

# Matches "1. ...", "1) ...", "1: ..." at start of line
_STEP_RE = re.compile(r"^\s*(\d+)[.):\s]\s*(.+)", re.MULTILINE)

_REBOOT_TAG_RE = re.compile(r"\[REBOOT\s+EXPECTED\]", re.IGNORECASE)

_ACTIONS_TAG_RE = re.compile(
    r"\[ACTIONS:\s*(\d+)(?:\s*-\s*(\d+))?\]", re.IGNORECASE
)

_EXPECTED_RE = re.compile(r"^Expected:\s*(.+)", re.MULTILINE)


def parse_steps(text: str) -> list[PlanStep]:
    """Parse numbered steps from LLM output.

    Handles formats like "1. ...", "1) ...", "1: ...".
    Detects [REBOOT EXPECTED] tag and "Expected: ..." outcome lines.

    Returns:
        List of PlanStep. Empty list if no steps found.
    """
    steps: list[PlanStep] = []
    # Split text into chunks per step number
    step_matches = list(_STEP_RE.finditer(text))
    for i, match in enumerate(step_matches):
        number = int(match.group(1))
        goal_text = match.group(2).strip()
        reboot = bool(_REBOOT_TAG_RE.search(goal_text))
        clean_goal = _REBOOT_TAG_RE.sub("", goal_text).strip()

        # Extract [ACTIONS: N] or [ACTIONS: N-M] tag
        actions_match = _ACTIONS_TAG_RE.search(clean_goal)
        if actions_match:
            if actions_match.group(2):
                expected_actions: int | None = int(actions_match.group(2))  # upper bound for ranges
            else:
                expected_actions = int(actions_match.group(1))
            clean_goal = _ACTIONS_TAG_RE.sub("", clean_goal).strip()
        else:
            expected_actions = None

        # Extract expected outcome from text between this step and the next
        start = match.end()
        end = step_matches[i + 1].start() if i + 1 < len(step_matches) else len(text)
        between = text[start:end]
        expected_match = _EXPECTED_RE.search(between)
        expected = expected_match.group(1).strip() if expected_match else ""

        steps.append(PlanStep(
            number=number,
            goal=clean_goal,
            reboot_expected=reboot,
            expected_outcome=expected,
            expected_actions=expected_actions,
        ))
    return steps


# ---------------------------------------------------------------------------
# Task Planner
# ---------------------------------------------------------------------------

class TaskPlanner:
    """Decomposes tasks into steps via a tool-free LLM call.

    Args:
        provider: "anthropic" or "openai".
        model: Model identifier.
        api_key: API key for the provider.
    """

    def __init__(
        self,
        provider: str,
        model: str,
        api_key: str,
        base_url: str | None = None,
    ) -> None:
        self._provider = provider.lower()
        self._model = model
        self._api_key = api_key
        self._base_url = base_url

        # Lazy-init clients to avoid import errors on platforms
        # that don't have both SDKs installed.
        self._anthropic_client: object | None = None
        self._openai_client: object | None = None

    def _get_anthropic_client(self) -> object:
        if self._anthropic_client is None:
            import anthropic
            self._anthropic_client = anthropic.Anthropic(api_key=self._api_key)
        return self._anthropic_client

    def _get_openai_client(self) -> object:
        if self._openai_client is None:
            import openai
            kwargs: dict[str, str] = {"api_key": self._api_key}
            if self._base_url:
                kwargs["base_url"] = self._base_url
            self._openai_client = openai.OpenAI(**kwargs)
        return self._openai_client

    def plan(
        self,
        task_goal: str,
        screenshot_base64: str,
        skill_text: str | None = None,
    ) -> list[PlanStep] | None:
        """Decompose a task into numbered steps.

        Args:
            task_goal: The full task description from the user.
            screenshot_base64: Current screen state (JPEG base64).
            skill_text: Optional skill markdown to guide planning.

        Returns:
            List of PlanStep objects. Returns None if planning fails
            and a skill is loaded (cannot safely fall back to free-form).
            Falls back to a single step if no skill is loaded.
        """
        user_text = self._build_user_text(task_goal, skill_text)
        raw = self._call_llm(
            PLANNING_SYSTEM_PROMPT, user_text, screenshot_base64,
        )

        if raw is None:
            logger.error("Planning call failed")
            if skill_text:
                # Skills are loaded — cannot fall back to free-form mode
                # because the LLM would ignore the skill instructions.
                return None  # type: ignore[return-value]
            return [PlanStep(number=1, goal=task_goal)]

        steps = parse_steps(raw)
        if not steps:
            logger.warning(
                "Could not parse steps from planner output. Raw: %s",
                raw[:200],
            )
            if skill_text:
                return None  # type: ignore[return-value]
            return [PlanStep(number=1, goal=task_goal)]

        logger.info("Planner produced %d steps for task: %s", len(steps), task_goal)
        for s in steps:
            tag = " [REBOOT]" if s.reboot_expected else ""
            logger.debug("  Step %d%s: %s", s.number, tag, s.goal)

        return steps

    def validate_plan(
        self,
        task_goal: str,
        screenshot_base64: str,
        completed_step: PlanStep,
        remaining_steps: list[PlanStep],
        skill_text: str | None = None,
    ) -> tuple[str, list[PlanStep] | None, str]:
        """Validate the plan against the current screen state.

        Args:
            task_goal: The ultimate user goal.
            screenshot_base64: Current screen after the step completed.
            completed_step: The step that just finished.
            remaining_steps: Steps not yet executed.
            skill_text: Optional skill markdown for replan context.

        Returns:
            (verdict, new_steps, reason)
            verdict: "continue", "replan", or "escalate"
            new_steps: revised plan (only for "replan"), None otherwise
            reason: explanation (for "replan" and "escalate"), empty for "continue"
        """
        remaining_text = "\n".join(
            f"  {s.number}. {s.goal}" for s in remaining_steps
        )
        user_parts = [
            f"The step that just completed was:",
            f"  Step {completed_step.number}: {completed_step.goal}",
            f"  Expected outcome: {completed_step.expected_outcome or '(not specified)'}",
            f"",
            f"The user's ultimate goal is: {task_goal}",
            f"",
            f"Remaining steps:",
            remaining_text or "(none, this was the last step)",
        ]
        if skill_text:
            user_parts.append(f"\n## Application Skill\n\n{skill_text}")
        user_text = "\n".join(user_parts)

        raw = self._call_llm(VALIDATE_PLAN_SYSTEM_PROMPT, user_text, screenshot_base64)

        if raw is None:
            logger.warning("Validation call failed, defaulting to CONTINUE")
            return ("continue", None, "")

        raw_upper = raw.strip().upper()

        if raw_upper.startswith("CONTINUE"):
            logger.debug("Plan validation: CONTINUE")
            return ("continue", None, "")

        if raw_upper.startswith("ESCALATE"):
            reason = raw.strip()
            # Remove the "ESCALATE" prefix
            reason = reason[len("ESCALATE"):].strip().lstrip(":").strip()
            if not reason:
                reason = "Human intervention required"
            logger.info("Plan validation: ESCALATE — %s", reason)
            return ("escalate", None, reason)

        if raw_upper.startswith("REPLAN"):
            reason_and_steps = raw.strip()
            # Extract reason (first line after REPLAN) and new steps
            lines = reason_and_steps.split("\n", 1)
            reason = lines[0][len("REPLAN"):].strip().lstrip(":").strip()
            new_steps = parse_steps(reason_and_steps) if len(lines) > 1 else []
            if not new_steps:
                new_steps = parse_steps(reason_and_steps)
            if not new_steps:
                logger.warning(
                    "Validation said REPLAN but no steps parsed, "
                    "defaulting to CONTINUE. Raw: %s", raw[:200],
                )
                return ("continue", None, "")
            logger.info(
                "Plan validation: REPLAN — %s (%d new steps)",
                reason, len(new_steps),
            )
            return ("replan", new_steps, reason)

        # Unparseable response, default to continue
        logger.warning(
            "Validation response not recognized, defaulting to CONTINUE. "
            "Raw: %s", raw[:100],
        )
        return ("continue", None, "")

    def replan(
        self,
        task_goal: str,
        failed_step: PlanStep,
        failure_reason: str,
        screenshot_base64: str,
        remaining_steps: list[PlanStep],
        skill_text: str | None = None,
    ) -> list[PlanStep] | None:
        """Produce a revised plan after a step failure.

        Returns:
            Revised list of PlanStep, or None if re-planning fails.
        """
        remaining_text = "\n".join(
            f"{s.number}. {s.goal}" for s in remaining_steps
        )
        context = (
            f"Original task: {task_goal}\n\n"
            f"Failed step {failed_step.number}: {failed_step.goal}\n"
            f"Failure reason: {failure_reason}\n\n"
            f"Remaining steps from original plan:\n{remaining_text}"
        )
        if skill_text:
            context += f"\n\n## Application Skill\n\n{skill_text}"

        raw = self._call_llm(
            REPLAN_SYSTEM_PROMPT, context, screenshot_base64,
        )

        if raw is None:
            logger.warning("Re-planning call failed")
            return None

        steps = parse_steps(raw)
        if not steps:
            logger.warning("Could not parse re-plan output: %s", raw[:200])
            return None

        logger.info("Re-planner produced %d revised steps", len(steps))
        return steps

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_user_text(
        task_goal: str,
        skill_text: str | None = None,
    ) -> str:
        """Build the text portion of the user message."""
        parts = [f"Task: {task_goal}"]
        if skill_text:
            parts.append(f"\n## Application Skill\n\n{skill_text}")
        return "\n".join(parts)

    def _call_llm(
        self,
        system_prompt: str,
        user_text: str,
        screenshot_base64: str | None = None,
    ) -> str | None:
        """Make a stateless LLM call with optional screenshot. No tools.

        Args:
            system_prompt: System-level instructions.
            user_text: User message text.
            screenshot_base64: Optional JPEG screenshot for visual context.

        Returns the raw text response, or None on failure.
        """
        try:
            if self._provider == "anthropic":
                return self._call_anthropic(
                    system_prompt, user_text, screenshot_base64,
                )
            else:
                return self._call_openai(
                    system_prompt, user_text, screenshot_base64,
                )
        except Exception as e:
            logger.error("Planner LLM call failed: %s", e)
            return None

    def _call_anthropic(
        self,
        system_prompt: str,
        user_text: str,
        screenshot_base64: str | None = None,
    ) -> str:
        client = self._get_anthropic_client()
        content: list[dict] = []
        if screenshot_base64:
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": screenshot_base64,
                },
            })
        content.append({"type": "text", "text": user_text})
        response = client.messages.create(  # type: ignore[union-attr]
            model=self._model,
            max_tokens=2048,
            system=system_prompt,
            messages=[{"role": "user", "content": content}],
        )
        return response.content[0].text  # type: ignore[union-attr]

    def _call_openai(
        self,
        system_prompt: str,
        user_text: str,
        screenshot_base64: str | None = None,
    ) -> str:
        client = self._get_openai_client()
        content: list[dict] = []
        if screenshot_base64:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{screenshot_base64}",
                },
            })
        content.append({"type": "text", "text": user_text})
        response = client.chat.completions.create(  # type: ignore[union-attr]
            model=self._model,
            max_completion_tokens=2048,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
        )
        return response.choices[0].message.content  # type: ignore[union-attr]
