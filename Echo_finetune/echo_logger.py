"""
Echo Agent Session Logger
=========================
Writes a structured JSONL log file for every session.

Log location: logs/echo_YYYYMMDD_HHMMSS.jsonl
Each line = one JSON event (session_start | turn | tool_call | tool_result | error | session_end)

Usage:
    from echo_logger import EchoLogger
    logger = EchoLogger()          # call once at session init
    logger.log_turn(...)           # call after each chat exchange
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path


LOG_DIR = Path(__file__).parent / "logs"


class EchoLogger:
    def __init__(self, session_id: str = ""):
        LOG_DIR.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_id = session_id or ts
        self.log_path = LOG_DIR / f"echo_{ts}.jsonl"
        self.turn_num = 0
        self.session_start = time.time()
        self._write({
            "event":      "session_start",
            "session_id": self.session_id,
            "timestamp":  self._ts(),
            "model":      "Qwen/Qwen3-4B + V4.6 merge + V4.7 LoRA",
        })

    # ── public API ─────────────────────────────────────────────────────

    def log_turn(
        self,
        user_input: str,
        thinking: str | None,
        tool_name: str | None,
        tool_args: dict | None,
        tool_result: dict | None,
        response: str,
        duration_ms: int = 0,
    ):
        self.turn_num += 1
        entry = {
            "event":       "turn",
            "turn":        self.turn_num,
            "timestamp":   self._ts(),
            "duration_ms": duration_ms,
            "user_input":  user_input,
            "thinking":    (thinking or "")[:2000],   # cap at 2000 chars
            "tool_call":   {"name": tool_name, "args": tool_args} if tool_name else None,
            "tool_result": tool_result,
            "response":    response,
            "outcome":     self._outcome(tool_name, tool_result),
        }
        self._write(entry)
        # also append a human-readable summary to a .txt file
        self._write_readable(entry)

    def log_error(self, context: str, error: str):
        self._write({
            "event":     "error",
            "turn":      self.turn_num,
            "timestamp": self._ts(),
            "context":   context,
            "error":     error,
        })

    def log_calendar_state(self, calendar: dict):
        """Snapshot the calendar at any point (e.g. after a booking)."""
        self._write({
            "event":     "calendar_snapshot",
            "turn":      self.turn_num,
            "timestamp": self._ts(),
            "calendar":  calendar,
        })

    def close(self):
        elapsed = round(time.time() - self.session_start, 1)
        self._write({
            "event":      "session_end",
            "timestamp":  self._ts(),
            "turns":      self.turn_num,
            "elapsed_s":  elapsed,
        })

    # ── internals ──────────────────────────────────────────────────────

    def _ts(self) -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _outcome(self, tool_name, tool_result) -> str:
        if not tool_name:
            return "direct_response"
        if tool_result and tool_result.get("status") == "success":
            return f"tool_success:{tool_name}"
        if tool_result and tool_result.get("status") == "error":
            return f"tool_error:{tool_name}"
        return f"tool_called:{tool_name}"

    def _write(self, data: dict):
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

    def _write_readable(self, entry: dict):
        """Append a human-friendly block to a .txt companion file."""
        txt_path = self.log_path.with_suffix(".txt")
        lines = [
            f"\n{'='*70}",
            f"TURN {entry['turn']}  [{entry['timestamp']}]  ({entry['duration_ms']} ms)",
            f"{'='*70}",
            f"USER   : {entry['user_input']}",
        ]
        if entry.get("thinking"):
            preview = entry["thinking"][:300].replace("\n", " ")
            lines.append(f"THINK  : {preview}{'...' if len(entry['thinking']) > 300 else ''}")
        if entry.get("tool_call"):
            tc = entry["tool_call"]
            lines.append(f"TOOL   : {tc['name']}({json.dumps(tc['args'])})")
        if entry.get("tool_result"):
            tr = entry["tool_result"]
            lines.append(f"RESULT : [{tr.get('status','?')}] {tr.get('message', json.dumps(tr))}")
        lines.append(f"ECHO   : {entry['response'][:500]}")
        lines.append(f"OUTCOME: {entry['outcome']}")
        with open(txt_path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
