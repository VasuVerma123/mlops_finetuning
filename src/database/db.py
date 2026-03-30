"""
Local SQLite metadata store for datasets, model results, and evaluations.

DB file: outputs/mlops.db  (override via DATABASE_PATH env var)
No server required — SQLite is built into Python.

Tables:
  datasets     — dataset metadata
  models       — training run results (all metrics)
  eval_results — evaluation results per model/run
"""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent.parent          # mlops_finetuning/
DB_PATH = os.environ.get("DATABASE_PATH", str(ROOT / "outputs" / "mlops.db"))


# ── Connection ─────────────────────────────────────────────────────────

def get_conn() -> sqlite3.Connection:
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


# ── Schema ─────────────────────────────────────────────────────────────

def init_db() -> None:
    """Create tables if they don't exist. Safe to call multiple times."""
    with get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS datasets (
                dataset_id         TEXT PRIMARY KEY,
                name               TEXT,
                s3_link            TEXT DEFAULT '',
                file_type          TEXT DEFAULT '',
                num_examples       INTEGER DEFAULT 0,
                tool_call_examples INTEGER DEFAULT 0,
                think_examples     INTEGER DEFAULT 0,
                avg_turns          REAL    DEFAULT 0,
                uploaded_at        TEXT    DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS models (
                model_id          TEXT PRIMARY KEY,
                experiment_name   TEXT UNIQUE,
                base_model        TEXT,
                adapter_path      TEXT DEFAULT '',
                dataset_id        TEXT REFERENCES datasets(dataset_id) ON DELETE SET NULL,
                quantization      TEXT DEFAULT '4bit',
                lora_r            INTEGER DEFAULT 16,
                lora_alpha        INTEGER DEFAULT 32,
                epochs            INTEGER DEFAULT 3,
                learning_rate     REAL    DEFAULT 0.0002,
                final_train_loss  REAL,
                best_eval_loss    REAL,
                perplexity        REAL,
                val_accuracy_pct  REAL,
                training_time_s   REAL,
                trainable_params  INTEGER,
                stacked_on        TEXT DEFAULT '',
                created_at        TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS eval_results (
                eval_id              TEXT PRIMARY KEY,
                model_id             TEXT REFERENCES models(model_id) ON DELETE SET NULL,
                experiment_name      TEXT DEFAULT '',
                test_file            TEXT DEFAULT '',
                overall_accuracy_pct REAL DEFAULT 0,
                avg_latency_s        REAL DEFAULT 0,
                avg_tokens_per_sec   REAL DEFAULT 0,
                p95_latency_s        REAL DEFAULT 0,
                ttft_s               REAL DEFAULT 0,
                scenario_results     TEXT DEFAULT '[]',
                created_at           TEXT DEFAULT (datetime('now'))
            );
        """)
        conn.commit()


# ── Write ──────────────────────────────────────────────────────────────

def save_dataset(
    name: str,
    s3_link: str = "",
    file_type: str = "",
    num_examples: int = 0,
    tool_call_examples: int = 0,
    think_examples: int = 0,
    avg_turns: float = 0.0,
) -> str:
    dataset_id = str(uuid.uuid4())
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO datasets
                (dataset_id, name, s3_link, file_type, num_examples,
                 tool_call_examples, think_examples, avg_turns)
            VALUES (?,?,?,?,?,?,?,?)
            """,
            (dataset_id, name, s3_link, file_type, num_examples,
             tool_call_examples, think_examples, avg_turns),
        )
        conn.commit()
    return dataset_id


def save_model_result(
    experiment_name: str,
    base_model: str,
    adapter_path: str = "",
    dataset_id: Optional[str] = None,
    quantization: str = "4bit",
    lora_r: int = 16,
    lora_alpha: int = 32,
    epochs: int = 3,
    learning_rate: float = 2e-4,
    final_train_loss: Optional[float] = None,
    best_eval_loss: Optional[float] = None,
    perplexity: Optional[float] = None,
    val_accuracy_pct: Optional[float] = None,
    training_time_s: Optional[float] = None,
    trainable_params: Optional[int] = None,
    stacked_on: Optional[str] = None,
) -> str:
    model_id = str(uuid.uuid4())
    with get_conn() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO models
                (model_id, experiment_name, base_model, adapter_path, dataset_id,
                 quantization, lora_r, lora_alpha, epochs, learning_rate,
                 final_train_loss, best_eval_loss, perplexity, val_accuracy_pct,
                 training_time_s, trainable_params, stacked_on)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (model_id, experiment_name, base_model, adapter_path, dataset_id,
             quantization, lora_r, lora_alpha, epochs, learning_rate,
             final_train_loss, best_eval_loss, perplexity, val_accuracy_pct,
             training_time_s, trainable_params, stacked_on or ""),
        )
        conn.commit()
    return model_id


def save_eval_result(
    test_file: str,
    overall_accuracy_pct: float,
    avg_latency_s: float,
    avg_tokens_per_sec: float,
    model_id: Optional[str] = None,
    experiment_name: str = "",
    p95_latency_s: float = 0.0,
    ttft_s: float = 0.0,
    scenario_results: Optional[list] = None,
) -> str:
    eval_id = str(uuid.uuid4())
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO eval_results
                (eval_id, model_id, experiment_name, test_file, overall_accuracy_pct,
                 avg_latency_s, avg_tokens_per_sec, p95_latency_s, ttft_s, scenario_results)
            VALUES (?,?,?,?,?,?,?,?,?,?)
            """,
            (eval_id, model_id, experiment_name, test_file, overall_accuracy_pct,
             avg_latency_s, avg_tokens_per_sec, p95_latency_s, ttft_s,
             json.dumps(scenario_results or [])),
        )
        conn.commit()
    return eval_id


# ── Read ───────────────────────────────────────────────────────────────

def get_datasets() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM datasets ORDER BY uploaded_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]


def get_models() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM models ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]


def get_model_by_experiment(experiment_name: str) -> Optional[dict]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM models WHERE experiment_name = ?", (experiment_name,)
        ).fetchone()
        return dict(row) if row else None


def get_eval_results() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM eval_results ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]


def get_evals_for_model(model_id: str) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM eval_results WHERE model_id = ? ORDER BY created_at DESC",
            (model_id,)
        ).fetchall()
        return [dict(r) for r in rows]


def get_full_run_table() -> list[dict]:
    """
    Join models + eval_results for a complete metrics table.
    Returns one row per training run, with the latest eval result merged in.
    """
    models = get_models()
    evals = get_eval_results()

    eval_by_model: dict[str, dict] = {}
    eval_by_exp: dict[str, dict] = {}
    for e in evals:
        mid = e.get("model_id")
        exp = e.get("experiment_name", "")
        if mid and mid not in eval_by_model:
            eval_by_model[mid] = e
        if exp and exp not in eval_by_exp:
            eval_by_exp[exp] = e

    rows = []
    for m in models:
        ev = eval_by_model.get(m["model_id"]) or eval_by_exp.get(m["experiment_name"], {})
        rows.append({
            "Run":        m["experiment_name"],
            "Base Model": m["base_model"],
            "Quant":      m["quantization"],
            "LoRA r":     m["lora_r"],
            "Epochs":     m["epochs"],
            "LR":         m["learning_rate"],
            "Train Loss": m["final_train_loss"],
            "Eval Loss":  m["best_eval_loss"],
            "Perplexity": m["perplexity"],
            "Val Acc %":  m["val_accuracy_pct"],
            "Test Acc %": ev.get("overall_accuracy_pct"),
            "Latency s":  ev.get("avg_latency_s"),
            "Tok/s":      ev.get("avg_tokens_per_sec"),
            "TTFT s":     ev.get("ttft_s"),
            "Duration s": m["training_time_s"],
            "Params":     m["trainable_params"],
            "Stacked On": m["stacked_on"] or "",
            "Created":    m["created_at"],
        })
    return rows
