"""
Dataset validator — schema checks and summary statistics.

Validates JSONL/JSON/CSV files before training to catch format issues early.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import List, Optional

from .loader import load_jsonl, load_csv, detect_file_type, get_csv_columns

REQUIRED_ROLES = {"system", "user", "assistant"}


def validate_record(record: dict, index: int) -> List[str]:
    """Validate a single record and return a list of issues (empty = OK)."""
    issues = []

    if "messages" not in record:
        issues.append(f"Record {index}: missing 'messages' key")
        return issues

    messages = record["messages"]
    if not isinstance(messages, list) or len(messages) < 2:
        issues.append(f"Record {index}: 'messages' must be a list with >= 2 entries")
        return issues

    for j, msg in enumerate(messages):
        if not isinstance(msg, dict):
            issues.append(f"Record {index}, msg {j}: not a dict")
            continue
        if "role" not in msg:
            issues.append(f"Record {index}, msg {j}: missing 'role'")
        elif msg["role"] not in ("system", "user", "assistant", "tool"):
            issues.append(f"Record {index}, msg {j}: unknown role '{msg['role']}'")
        if "content" not in msg:
            issues.append(f"Record {index}, msg {j}: missing 'content'")

    roles_present = {m.get("role") for m in messages if isinstance(m, dict)}
    if "assistant" not in roles_present:
        issues.append(f"Record {index}: no assistant message found")

    return issues


def validate_csv(
    path: str | Path,
    input_col: str = "",
    output_col: str = "",
) -> dict:
    """
    Validate a CSV file for training.

    Returns dict with: valid, total, columns, errors, stats.
    """
    ftype = detect_file_type(path)
    if ftype != "csv":
        return {"valid": False, "total": 0, "errors": ["Not a CSV file"], "columns": []}

    try:
        columns = get_csv_columns(path)
    except Exception as exc:
        return {"valid": False, "total": 0, "errors": [str(exc)], "columns": []}

    issues: List[str] = []
    if not columns:
        issues.append("CSV has no columns")
    if input_col and input_col not in columns:
        issues.append(f"Input column '{input_col}' not found in CSV. Available: {columns}")
    if output_col and output_col not in columns:
        issues.append(f"Output column '{output_col}' not found in CSV. Available: {columns}")

    row_count = 0
    empty_input = 0
    empty_output = 0
    if input_col and output_col and not issues:
        try:
            import csv as csv_mod
            with open(path, "r", encoding="utf-8", newline="") as f:
                reader = csv_mod.DictReader(f)
                for row in reader:
                    row_count += 1
                    if not row.get(input_col, "").strip():
                        empty_input += 1
                    if not row.get(output_col, "").strip():
                        empty_output += 1
        except Exception as exc:
            issues.append(f"Error reading CSV: {exc}")

    if empty_input > 0:
        issues.append(f"{empty_input} rows have empty input column")
    if empty_output > 0:
        issues.append(f"{empty_output} rows have empty output column")

    stats = {
        "total_records": row_count,
        "columns": columns,
        "empty_inputs": empty_input,
        "empty_outputs": empty_output,
        "usable_rows": row_count - max(empty_input, empty_output),
    }

    return {
        "valid": len(issues) == 0,
        "total": row_count,
        "columns": columns,
        "errors": issues[:50],
        "stats": stats,
    }


def validate_file(path: str | Path, input_col: str = "", output_col: str = "") -> dict:
    """
    Validate a dataset file (JSONL, JSON, or CSV).

    For CSV files, pass input_col and output_col.

    Returns
    -------
    dict with keys: valid (bool), total, errors (list of str), stats (dict).
    """
    ftype = detect_file_type(path)
    if ftype == "csv":
        return validate_csv(path, input_col, output_col)

    try:
        records = load_jsonl(path)
    except Exception as exc:
        return {"valid": False, "total": 0, "errors": [str(exc)], "stats": {}}

    all_issues: List[str] = []
    role_counts = Counter()
    tool_call_count = 0
    think_count = 0
    turn_lengths: List[int] = []

    for i, rec in enumerate(records):
        issues = validate_record(rec, i)
        all_issues.extend(issues)

        if "messages" in rec and isinstance(rec["messages"], list):
            turn_lengths.append(len(rec["messages"]))
            for msg in rec["messages"]:
                if isinstance(msg, dict):
                    role_counts[msg.get("role", "?")] += 1
                    content = msg.get("content", "")
                    if msg.get("role") == "assistant":
                        if "<tool_call>" in content:
                            tool_call_count += 1
                        if "<think>" in content:
                            think_count += 1

    stats = {
        "total_records": len(records),
        "role_distribution": dict(role_counts),
        "tool_call_examples": tool_call_count,
        "think_augmented_examples": think_count,
        "avg_turns": sum(turn_lengths) / len(turn_lengths) if turn_lengths else 0,
        "max_turns": max(turn_lengths) if turn_lengths else 0,
    }

    return {
        "valid": len(all_issues) == 0,
        "total": len(records),
        "errors": all_issues[:50],  # cap displayed errors
        "stats": stats,
    }


def validate_test_cases(path: str | Path) -> dict:
    """
    Validate a test-case JSONL file (for evaluation).

    Each record should have: name, user_msg, expected (dict with
    should_call_tool, tool_name, key_params, etc.).
    """
    try:
        records = load_jsonl(path)
    except Exception as exc:
        return {"valid": False, "total": 0, "errors": [str(exc)]}

    issues: List[str] = []
    for i, rec in enumerate(records):
        for key in ("name", "user_msg", "expected"):
            if key not in rec:
                issues.append(f"Test {i}: missing '{key}'")
        exp = rec.get("expected", {})
        if isinstance(exp, dict) and "should_call_tool" not in exp:
            issues.append(f"Test {i}: expected missing 'should_call_tool'")

    return {
        "valid": len(issues) == 0,
        "total": len(records),
        "errors": issues[:50],
    }
