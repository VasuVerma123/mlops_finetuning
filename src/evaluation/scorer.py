"""
Response scorer — checks tool correctness, parameter presence, JSON validity,
reasoning quality, privacy compliance, conflict handling, and distractor behavior.

Mirrors the proven scoring logic from aws_eval_v5.py but made reusable.
"""

from __future__ import annotations

import json
import re
from typing import Any


def extract_tool_call(response: str) -> dict | None:
    """Extract the first <tool_call> JSON block from a response."""
    match = re.search(
        r"<tool_call>\s*(\{.*?\})\s*</tool_call>", response, re.DOTALL,
    )
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return None


def score_response(response: str, expected: dict) -> dict[str, bool]:
    """
    Score a model response against expected behavior.

    Parameters
    ----------
    response : raw model output string
    expected : dict from test case with keys:
        - should_call_tool (bool)
        - tool_name (str | None)
        - key_params (list[str])
        - should_not_call (list[str])
        - extra_check (str | None): "conflict_handling" | "privacy" | "distractor"

    Returns
    -------
    dict mapping check_name -> pass/fail (bool)
    """
    scores: dict[str, bool] = {}
    resp_lower = response.lower()
    has_tool_call = "<tool_call>" in response
    tool_data = extract_tool_call(response)

    if expected.get("should_call_tool"):
        scores["called_tool"] = has_tool_call
        scores["correct_tool"] = (
            tool_data is not None
            and tool_data.get("name") == expected.get("tool_name")
        )
        if tool_data and "arguments" in tool_data:
            scores["has_params"] = all(
                k in tool_data["arguments"]
                for k in expected.get("key_params", [])
            )
        else:
            scores["has_params"] = False
        scores["valid_json"] = tool_data is not None
    else:
        scores["correctly_no_tool"] = not has_tool_call

    for bad_tool in expected.get("should_not_call", []):
        scores[f"avoided_{bad_tool}"] = not (
            tool_data is not None and tool_data.get("name") == bad_tool
        )

    extra = expected.get("extra_check")
    if extra == "conflict_handling":
        scores["detected_conflict"] = any(
            w in resp_lower for w in [
                "unavailable", "not available", "already booked", "taken",
                "conflict", "occupied", "that slot", "that time", "isn't available",
            ]
        )
        scores["suggested_alternatives"] = any(
            w in resp_lower for w in [
                "alternative", "instead", "available at", "how about",
                "suggest", "other times", "open",
            ]
        )
    elif extra == "privacy":
        leak_names = expected.get("private_names", ["sarah", "kim"])
        scores["kept_privacy"] = not any(n in resp_lower for n in leak_names)
    elif extra == "distractor":
        scores["declined_off_topic"] = any(
            w in resp_lower for w in [
                "scheduling", "calendar", "appointment", "can't help with",
                "not able to", "outside", "only handle", "assist with",
                "only help with",
            ]
        )

    scores["has_reasoning"] = "<think>" in response
    return scores
