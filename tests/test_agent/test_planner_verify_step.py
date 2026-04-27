"""Tests for TaskPlanner.verify_step (Phase 3 — VERIFY-01/02/03).

Replaces TestValidatePlan / TestValidatePlanPromptStructure from
test_planner.py (those classes are deleted in plan 03-01 task 3).
"""
from __future__ import annotations

import json
import logging

import pytest

# Feature-detect import for gradual landing (per PATTERNS § Test imports)
try:
    from cyberraccoon.agent.planner import (
        PlanStep,
        StepVerification,
        TaskPlanner,
        VERIFY_STEP_SYSTEM_PROMPT,
    )
    _VERIFY_STEP_AVAILABLE = True
except ImportError:
    _VERIFY_STEP_AVAILABLE = False


pytestmark = pytest.mark.skipif(
    not _VERIFY_STEP_AVAILABLE,
    reason="StepVerification / verify_step not yet implemented",
)


# --------------------------------------------------------------------------
# Inline mock (duplicated from tests/test_agent/test_planner.py L112-127).
# Review M4: do NOT import across test modules — keeps this file independent
# of tests/ being an importable package.
# --------------------------------------------------------------------------
class LocalMockPlanner(TaskPlanner):
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


def _make_step(
    number: int = 1,
    goal: str = "Open Chrome",
    expected_outcome: str = "Chrome browser window is visible with address bar",
) -> PlanStep:
    return PlanStep(number=number, goal=goal, expected_outcome=expected_outcome)


def _verified_true_json(
    expected: str = "Chrome browser window is visible with address bar",
    observed: str = "Chrome window in focus with new-tab page loaded",
    confidence: float = 0.95,
) -> str:
    return json.dumps({
        "verified": True,
        "expected": expected,
        "observed": observed,
        "mismatch_reason": None,
        "confidence": confidence,
    })


def _verified_false_json(
    expected: str = "Search results page is visible",
    observed: str = "Search query still in address bar; no results page loaded",
    mismatch_reason: str = "Enter key apparently swallowed; no navigation occurred",
    confidence: float = 0.92,
) -> str:
    return json.dumps({
        "verified": False,
        "expected": expected,
        "observed": observed,
        "mismatch_reason": mismatch_reason,
        "confidence": confidence,
    })


class TestVerifyStep:
    """VERIFY-01: verify_step returns StepVerification dataclass."""

    def test_returns_step_verification_dataclass(self) -> None:
        planner = LocalMockPlanner(_verified_true_json())
        result = planner.verify_step(
            step=_make_step(),
            observed_screenshot="fake_b64",
        )
        assert isinstance(result, StepVerification)

    def test_clean_match_returns_verified_true(self) -> None:
        planner = LocalMockPlanner(_verified_true_json(
            expected="Chrome browser window is visible with address bar",
            observed="Chrome window in focus",
            confidence=0.98,
        ))
        result = planner.verify_step(
            step=_make_step(),
            observed_screenshot="fake_b64",
        )
        assert result.verified is True
        assert result.expected == "Chrome browser window is visible with address bar"
        assert result.observed == "Chrome window in focus"
        assert result.mismatch_reason is None
        assert result.confidence == 0.98

    def test_concrete_mismatch_returns_verified_false_with_reason(self) -> None:
        planner = LocalMockPlanner(_verified_false_json())
        result = planner.verify_step(
            step=_make_step(),
            observed_screenshot="fake_b64",
        )
        assert result.verified is False
        assert result.mismatch_reason is not None
        assert "Enter key apparently swallowed" in result.mismatch_reason
        assert result.confidence == 0.92

    def test_skill_text_included_in_user_text(self) -> None:
        planner = LocalMockPlanner(_verified_true_json())
        planner.verify_step(
            step=_make_step(),
            observed_screenshot="fake_b64",
            skill_text="When verifying Chrome, look for the omnibox.",
        )
        assert "When verifying Chrome" in planner._last_user_text

    def test_no_skill_text_omits_skill_section(self) -> None:
        planner = LocalMockPlanner(_verified_true_json())
        planner.verify_step(
            step=_make_step(),
            observed_screenshot="fake_b64",
        )
        assert "Application Skill" not in planner._last_user_text


class TestVerifyStepPromptStructure:
    """VERIFY-02: prompt biases toward verified=True with concrete-evidence rule."""

    def test_prompt_states_default_to_verified_true(self) -> None:
        assert "DEFAULT TO verified=true" in VERIFY_STEP_SYSTEM_PROMPT

    def test_prompt_lists_all_must_hold_criteria(self) -> None:
        assert "ALL of the following" in VERIFY_STEP_SYSTEM_PROMPT

    def test_prompt_states_when_in_doubt_rule(self) -> None:
        assert "When in doubt, set verified=true" in VERIFY_STEP_SYSTEM_PROMPT

    def test_prompt_excludes_cosmetic_and_partial_progress(self) -> None:
        assert "cosmetic" in VERIFY_STEP_SYSTEM_PROMPT.lower()
        assert "partial-progress" in VERIFY_STEP_SYSTEM_PROMPT.lower() \
            or "partial progress" in VERIFY_STEP_SYSTEM_PROMPT.lower()

    def test_prompt_requires_json_only_output(self) -> None:
        assert "Output ONLY the JSON object" in VERIFY_STEP_SYSTEM_PROMPT

    def test_user_text_contains_step_number_and_goal(self) -> None:
        planner = LocalMockPlanner(_verified_true_json())
        planner.verify_step(
            step=_make_step(number=7, goal="Click File menu"),
            observed_screenshot="fake_b64",
        )
        assert "Step 7" in planner._last_user_text
        assert "Click File menu" in planner._last_user_text

    def test_user_text_contains_expected_outcome(self) -> None:
        planner = LocalMockPlanner(_verified_true_json())
        planner.verify_step(
            step=_make_step(expected_outcome="The Save dialog is dismissed"),
            observed_screenshot="fake_b64",
        )
        assert "The Save dialog is dismissed" in planner._last_user_text

    def test_user_text_handles_empty_expected_outcome(self) -> None:
        planner = LocalMockPlanner(_verified_true_json())
        planner.verify_step(
            step=_make_step(expected_outcome=""),
            observed_screenshot="fake_b64",
        )
        assert "(not specified)" in planner._last_user_text

    def test_system_prompt_passed_to_call_llm(self) -> None:
        planner = LocalMockPlanner(_verified_true_json())
        planner.verify_step(
            step=_make_step(),
            observed_screenshot="fake_b64",
        )
        assert planner._last_system_prompt == VERIFY_STEP_SYSTEM_PROMPT


class TestVerifyStepFailureModes:
    """VERIFY-03: semantic failures bias to verified=True + logger.error.
    Transport failures are propagated as LLMTransportError so the workflow
    runner can count consecutive failures and abort if the verifier is
    structurally unreachable (review C3)."""

    def test_raw_none_treated_as_parse_failure(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        # raw=None from _call_llm is now treated as a semantic parse
        # failure rather than a separate "LLM call failed" branch.
        # Genuine transport failures raise LLMTransportError (see
        # test_transport_error_propagates_to_caller below).
        planner = LocalMockPlanner(None)
        with caplog.at_level(logging.ERROR, logger="M2.planner"):
            result = planner.verify_step(
                step=_make_step(number=3),
                observed_screenshot="fake_b64",
            )
        assert result.verified is True
        assert result.observed == "(LLM output unparseable)"
        assert result.confidence == 0.0
        assert any(
            "verify_step: parse failed for step 3" in r.message
            for r in caplog.records
        )

    def test_unparseable_response_defaults_to_verified_true(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        planner = LocalMockPlanner("not json at all, not even close")
        with caplog.at_level(logging.ERROR, logger="M2.planner"):
            result = planner.verify_step(
                step=_make_step(number=4),
                observed_screenshot="fake_b64",
            )
        assert result.verified is True
        assert result.observed == "(LLM output unparseable)"
        assert result.confidence == 0.0
        assert any(
            "verify_step: parse failed for step 4" in r.message
            for r in caplog.records
        )

    def test_transport_error_propagates_to_caller(self) -> None:
        """C3 — _call_llm raising LLMTransportError must propagate up so
        WorkflowRunner can count consecutive verifier failures."""
        from cyberraccoon.agent.planner import LLMTransportError

        class TransportFailPlanner(TaskPlanner):
            def __init__(self) -> None:
                super().__init__(provider="anthropic", model="test", api_key="test")

            def _call_llm(  # type: ignore[override]
                self, system_prompt: str, user_text: str,
                screenshot_base64: str | None = None,
            ) -> str | None:
                raise LLMTransportError("simulated rate limit")

        planner = TransportFailPlanner()
        try:
            planner.verify_step(
                step=_make_step(number=99),
                observed_screenshot="fake_b64",
            )
            assert False, "should have raised LLMTransportError"
        except LLMTransportError as e:
            assert "simulated rate limit" in str(e)

    def test_schema_violation_defaults_to_verified_true(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        # `verified` is wrong type — must NOT be coerced to truthy
        planner = LocalMockPlanner('{"verified": "yes", "expected": null}')
        with caplog.at_level(logging.WARNING, logger="M2.planner"):
            result = planner.verify_step(
                step=_make_step(number=5),
                observed_screenshot="fake_b64",
            )
        assert result.verified is True
        assert result.observed == "(LLM output schema invalid)"
        assert result.confidence == 0.0
        assert any(
            "verify_step: schema violation for step 5" in r.message
            for r in caplog.records
        )

    def test_missing_verified_field_defaults_to_verified_true(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        planner = LocalMockPlanner('{"expected": "x", "observed": "y"}')
        with caplog.at_level(logging.WARNING, logger="M2.planner"):
            result = planner.verify_step(
                step=_make_step(),
                observed_screenshot="fake_b64",
            )
        assert result.verified is True


class TestConfidenceClamp:
    """Review L1: StepVerification.__post_init__ clamps confidence to [0.0, 1.0]."""

    def test_above_one_clamped_to_one(self) -> None:
        sv = StepVerification(
            verified=True, expected="x", observed="y", confidence=1.7,
        )
        assert sv.confidence == 1.0

    def test_below_zero_clamped_to_zero(self) -> None:
        sv = StepVerification(
            verified=True, expected="x", observed="y", confidence=-0.3,
        )
        assert sv.confidence == 0.0

    def test_in_range_preserved(self) -> None:
        sv = StepVerification(
            verified=True, expected="x", observed="y", confidence=0.7,
        )
        assert sv.confidence == 0.7

    def test_nan_collapses_to_zero(self) -> None:
        sv = StepVerification(
            verified=True, expected="x", observed="y", confidence=float("nan"),
        )
        assert sv.confidence == 0.0

    def test_llm_confidence_above_one_clamped_via_verify_step(self) -> None:
        # LLM returns an out-of-range confidence — must be clamped, NOT rejected.
        planner = LocalMockPlanner(json.dumps({
            "verified": True,
            "expected": "x",
            "observed": "y",
            "mismatch_reason": None,
            "confidence": 2.5,
        }))
        result = planner.verify_step(
            step=_make_step(),
            observed_screenshot="fake_b64",
        )
        assert result.verified is True  # still verified=True (NOT a schema error)
        assert result.confidence == 1.0


# VERIFY-02 confidence sampling: 7 input variations from VALIDATION.md L1671-1700
@pytest.mark.parametrize(
    "scenario,canned_json,expected_verified",
    [
        (
            "clean_match",
            json.dumps({
                "verified": True,
                "expected": "Chrome window visible",
                "observed": "Chrome window with new-tab page in focus",
                "mismatch_reason": None,
                "confidence": 0.98,
            }),
            True,
        ),
        (
            "clean_mismatch",
            json.dumps({
                "verified": False,
                "expected": "Search results visible",
                "observed": "Search query still in address bar, no results loaded",
                "mismatch_reason": "Address bar still shows the unsubmitted query",
                "confidence": 0.9,
            }),
            False,
        ),
        (
            "partial_progress",
            json.dumps({
                "verified": True,
                "expected": "Login form filled",
                "observed": "Username field filled, password field empty",
                "mismatch_reason": None,
                "confidence": 0.7,
            }),
            True,
        ),
        (
            "cosmetic",
            json.dumps({
                "verified": True,
                "expected": "Notepad open",
                "observed": "Notepad open with default font slightly different",
                "mismatch_reason": None,
                "confidence": 0.95,
            }),
            True,
        ),
        (
            "ambiguous",
            json.dumps({
                "verified": True,
                "expected": "Settings dialog dismissed",
                "observed": "Unknown screen state, possibly transitional",
                "mismatch_reason": None,
                "confidence": 0.55,
            }),
            True,
        ),
        (
            "mid_animation",
            json.dumps({
                "verified": True,
                "expected": "Window minimized to tray",
                "observed": "Window animation in progress, position unclear",
                "mismatch_reason": None,
                "confidence": 0.6,
            }),
            True,
        ),
        (
            "wrong_window",
            json.dumps({
                "verified": False,
                "expected": "VS Code editor visible",
                "observed": "Excel spreadsheet visible, no VS Code anywhere",
                "mismatch_reason": "Excel is in focus instead of VS Code",
                "confidence": 0.95,
            }),
            False,
        ),
    ],
)
def test_verify_step_input_variations(
    scenario: str, canned_json: str, expected_verified: bool,
) -> None:
    planner = LocalMockPlanner(canned_json)
    result = planner.verify_step(
        step=_make_step(),
        observed_screenshot="fake_b64",
    )
    assert result.verified is expected_verified, f"scenario {scenario} failed"


class TestVerifyStepJsonFallbackLayers:
    """TC3 (review test-coverage gap): verify_step's 3-level JSON parsing
    fallback (direct → markdown extraction → regex). The PR description
    explicitly cites this as a key behavior; previously only direct-parse
    was tested."""

    def test_parses_json_in_markdown_fence(self) -> None:
        """Layer 2: ```json ... ``` fence around valid JSON."""
        canned = (
            "Sure, here's my verdict:\n"
            "```json\n"
            '{"verified": false, "expected": "Notepad open", '
            '"observed": "Desktop only", '
            '"mismatch_reason": "Notepad never opened", '
            '"confidence": 0.9}\n'
            "```\n"
            "Hope that helps!"
        )
        planner = LocalMockPlanner(canned)
        result = planner.verify_step(
            step=_make_step(number=10),
            observed_screenshot="fake_b64",
        )
        assert result.verified is False, (
            "markdown-fenced JSON must be parsed; LLMs commonly wrap "
            "structured output in code blocks"
        )
        assert "Notepad never opened" in (result.mismatch_reason or "")

    def test_parses_json_with_prose_prefix(self) -> None:
        """Layer 3: regex-extracted JSON inside surrounding prose."""
        # Bare JSON object embedded in narration, no fence.
        canned = (
            "After examining the screenshot, here is my conclusion: "
            '{"verified": true, "expected": "Chrome visible", '
            '"observed": "Chrome window in focus", '
            '"mismatch_reason": null, "confidence": 0.85} '
            "End of analysis."
        )
        planner = LocalMockPlanner(canned)
        result = planner.verify_step(
            step=_make_step(number=11),
            observed_screenshot="fake_b64",
        )
        assert result.verified is True
        assert result.confidence == 0.85


class TestVerifyStepForceFailEnvVar:
    """TC4 (review test-coverage gap): the CYBERRACCOON_FORCE_VERIFY_FAIL
    test-only override is a production code path. Without coverage, a
    misconfigured shell environment could short-circuit verification
    silently. This pair pins both directions of the env-var check."""

    def test_force_fail_env_var_short_circuits(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When env var is '1', return verified=False without calling LLM."""
        monkeypatch.setenv("CYBERRACCOON_FORCE_VERIFY_FAIL", "1")
        # Even if the mock would return verified=True, the override wins.
        planner = LocalMockPlanner(
            '{"verified": true, "confidence": 1.0, '
            '"expected": "x", "observed": "x", "mismatch_reason": null}',
        )
        result = planner.verify_step(
            step=_make_step(number=20),
            observed_screenshot="fake_b64",
        )
        assert result.verified is False
        # Mismatch reason should mention the override mechanism so the user
        # understands why their otherwise-OK step is being failed.
        assert "CYBERRACCOON_FORCE_VERIFY_FAIL" in (result.mismatch_reason or "")
        # The LLM mock should NOT have been invoked (last_user_text empty).
        assert planner._last_user_text == ""

    def test_unset_env_var_does_not_short_circuit(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Without the env var, normal verify_step path runs."""
        monkeypatch.delenv("CYBERRACCOON_FORCE_VERIFY_FAIL", raising=False)
        planner = LocalMockPlanner(
            '{"verified": true, "confidence": 1.0, '
            '"expected": "x", "observed": "x", "mismatch_reason": null}',
        )
        result = planner.verify_step(
            step=_make_step(number=21),
            observed_screenshot="fake_b64",
        )
        assert result.verified is True
        # LLM was invoked (mock recorded the prompt).
        assert planner._last_user_text != ""

    def test_zero_value_does_not_short_circuit(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Only literal '1' triggers; '0' / 'true' / etc. do not."""
        monkeypatch.setenv("CYBERRACCOON_FORCE_VERIFY_FAIL", "0")
        planner = LocalMockPlanner(
            '{"verified": true, "confidence": 1.0, '
            '"expected": "x", "observed": "x", "mismatch_reason": null}',
        )
        result = planner.verify_step(
            step=_make_step(number=22),
            observed_screenshot="fake_b64",
        )
        assert result.verified is True
        assert planner._last_user_text != ""
