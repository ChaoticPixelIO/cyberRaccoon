"""Shared JSON parsing utilities for prompt-based protocols.

Provides a 4-level fallback parser for extracting JSON action objects
from unstructured LLM text responses.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger("M3.parsing")


def parse_json_actions(
    raw_text: str, valid_actions: set[str],
) -> list[dict[str, Any]] | None:
    """Extract a list of valid action dicts from LLM response text.

    Supports both a single JSON object and a JSON array of objects.
    Uses a multi-level fallback similar to :func:`parse_json_action`:

    1. Direct JSON parse — array or object
    2. Markdown code block extraction — array or object
    3. Fall through to :func:`parse_json_action` (single object), wrap in list

    Returns None if nothing valid is found.
    """
    text = raw_text.strip()

    # Level 1: Direct parse (array or object)
    result = _try_parse_actions(text, valid_actions)
    if result is not None:
        return result

    # Level 2: Markdown code block (array or object)
    code_block = re.search(r"```(?:json)?\s*(\[.*?\]|\{.*?\})\s*```", text, re.DOTALL)
    if code_block:
        result = _try_parse_actions(code_block.group(1), valid_actions)
        if result is not None:
            return result

    # Level 3: Fall through to single-object parser
    single = parse_json_action(raw_text, valid_actions)
    if single is not None:
        logger.debug(
            "Array parsing failed, fell back to single-object extraction",
        )
        return [single]

    return None


def _try_parse_actions(
    text: str, valid_actions: set[str],
) -> list[dict[str, Any]] | None:
    """Try to parse text as a JSON array of actions or a single action object."""
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None

    if isinstance(obj, list):
        if not obj:
            return None
        actions: list[dict[str, Any]] = []
        for item in obj:
            if not isinstance(item, dict):
                return None
            if item.get("action") not in valid_actions:
                logger.warning("Invalid action in array: %s", item.get("action"))
                return None
            actions.append(item)
        return actions

    if isinstance(obj, dict) and obj.get("action") in valid_actions:
        return [obj]

    return None


def parse_json_action(
    raw_text: str, valid_actions: set[str],
) -> dict[str, Any] | None:
    """Extract a valid action dict from LLM response text.

    Uses a 4-level fallback strategy:
    1. Direct JSON parse of entire text
    2. Markdown code block extraction (```json ... ```)
    3. First shallow {...} object (no nested braces)
    4. Greedy first { to last }

    Returns None if no valid action is found.
    """
    text = raw_text.strip()

    # Level 1: Direct parse
    cmd = try_parse_json(text, valid_actions)
    if cmd is not None:
        return cmd

    # Level 2: Markdown code block extraction
    code_block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if code_block:
        cmd = try_parse_json(code_block.group(1), valid_actions)
        if cmd is not None:
            return cmd

    # Level 3: Regex extract first {...} object (no nested braces)
    json_match = re.search(r"\{[^{}]*\}", text)
    if json_match:
        cmd = try_parse_json(json_match.group(0), valid_actions)
        if cmd is not None:
            return cmd

    # Level 4: Greedy extract from first { to last }
    json_match = re.search(r"\{.*\}", text, re.DOTALL)
    if json_match:
        cmd = try_parse_json(json_match.group(0), valid_actions)
        if cmd is not None:
            return cmd

    return None


def try_parse_json(
    text: str, valid_actions: set[str],
) -> dict[str, Any] | None:
    """Attempt to parse text as JSON and validate it has a valid action."""
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None

    if not isinstance(obj, dict):
        return None

    action = obj.get("action")
    if action not in valid_actions:
        logger.warning("Invalid action in parsed JSON: %s", action)
        return None

    return obj


# Valid completion statuses for the structured completion signal.
# "escalate" is the Path C trigger — the LLM signals it cannot continue without
# human input (CAPTCHA, 2FA, ambiguous credentials, structural blocker). It must
# be in this whitelist for Anthropic CU / OpenAI CU done-text to reach
# WorkflowRunner's escalation gate.
VALID_STATUSES: set[str] = {"success", "gave_up", "stuck", "escalate"}


def extract_completion_status(text: str) -> str:
    """Extract completion status from LLM done message text.

    Searches for JSON objects containing a ``"status"`` field with a valid
    completion value (``"success"``, ``"gave_up"``, ``"stuck"``, or
    ``"escalate"``).

    Returns ``"success"`` if not found (safe default per D-07).
    """
    if not text:
        return "success"

    # Try shallow JSON objects first: {...} with no nested braces
    for m in re.finditer(r"\{[^{}]*\}", text):
        try:
            obj = json.loads(m.group(0))
        except (json.JSONDecodeError, TypeError):
            continue
        status = obj.get("status")
        if status in VALID_STATUSES:
            return status

    # Try markdown code block extraction
    code_block = re.search(
        r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL,
    )
    if code_block:
        try:
            obj = json.loads(code_block.group(1))
            status = obj.get("status")
            if status in VALID_STATUSES:
                return status
        except (json.JSONDecodeError, TypeError):
            pass

    return "success"
