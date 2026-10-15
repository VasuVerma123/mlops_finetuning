"""
Dataset loader — JSONL / JSON / CSV ingestion with train/val split and tokenization.

Supported formats:
  - JSONL: one JSON object per line with a "messages" key
  - JSON:  a JSON array of objects with a "messages" key
  - CSV:   tabular data — user specifies input/output columns, converted to messages
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import List, Optional, Tuple

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # only for annotations; `from __future__ import annotations` keeps
    from datasets import Dataset  # them unevaluated, so this is never imported at runtime


def load_jsonl(path: str | Path) -> List[dict]:
    """Read a JSONL file and return a list of parsed records."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {lineno} of {path}: {exc}")
    return records


def load_csv(
    path: str | Path,
    input_col: str,
    output_col: str,
    system_prompt: str = "",
) -> List[dict]:
    """
    Read a CSV and convert rows to chat-message format.

    Each row becomes:
      {"messages": [
          {"role": "system", "content": system_prompt},   # if provided
          {"role": "user", "content": <input_col value>},
          {"role": "assistant", "content": <output_col value>},
      ]}
    """
    records = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if input_col not in reader.fieldnames:
            raise ValueError(f"Input column '{input_col}' not found. Available: {reader.fieldnames}")
        if output_col not in reader.fieldnames:
            raise ValueError(f"Output column '{output_col}' not found. Available: {reader.fieldnames}")

        for i, row in enumerate(reader):
            user_text = row[input_col].strip()
            assistant_text = row[output_col].strip()
            if not user_text or not assistant_text:
                continue
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_text})
            messages.append({"role": "assistant", "content": assistant_text})
            records.append({"messages": messages})
    return records


def detect_file_type(path: str | Path) -> str:
    """Detect file type from extension."""
    ext = Path(path).suffix.lower()
    if ext == ".csv":
        return "csv"
    elif ext == ".jsonl":
        return "jsonl"
    elif ext == ".json":
        return "json"
    return "unknown"


def get_csv_columns(path: str | Path) -> List[str]:
    """Read CSV header and return column names."""
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or [])


def load_file(
    path: str | Path,
    input_col: str = "",
    output_col: str = "",
    system_prompt: str = "",
) -> List[dict]:
    """
    Auto-detect file type and load records.

    For CSV files, input_col and output_col are required.
    For JSONL/JSON, records must have a "messages" key.
    """
    ftype = detect_file_type(path)
    if ftype == "csv":
        if not input_col or not output_col:
            raise ValueError("CSV files require input_col and output_col to be specified")
        return load_csv(path, input_col, output_col, system_prompt)
    elif ftype in ("jsonl", "json"):
        return load_jsonl(path)
    else:
        raise ValueError(f"Unsupported file type: {Path(path).suffix}. Use .jsonl, .json, or .csv")


def train_val_split(
    records: List[dict],
    val_fraction: float = 0.1,
    seed: int = 42,
) -> Tuple[List[dict], List[dict]]:
    """Randomly split records into (train, val) lists."""
    rng = random.Random(seed)
    shuffled = records.copy()
    rng.shuffle(shuffled)
    val_size = max(1, int(len(shuffled) * val_fraction))
    return shuffled[val_size:], shuffled[:val_size]


def records_to_dataset(records: List[dict]) -> "Dataset":
    """Convert a list of message dicts to a HuggingFace Dataset."""
    # Imported here, not at module scope, so that dataset validation (`run.py validate`)
    # works with only pyyaml installed -- no torch, no datasets.
    from datasets import Dataset
    return Dataset.from_list(records)


def format_for_training(
    dataset: Dataset,
    tokenizer,
    enable_thinking: bool = True,
) -> Dataset:
    """Apply chat template to produce a 'text' column for SFTTrainer."""
    def _apply(batch):
        texts = []
        for msgs in batch["messages"]:
            kwargs = dict(tokenize=False, add_generation_prompt=False)
            if enable_thinking:
                kwargs["enable_thinking"] = True
            try:
                text = tokenizer.apply_chat_template(msgs, **kwargs)
            except TypeError:
                # Older tokenizer without enable_thinking kwarg
                text = tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=False,
                )
            texts.append(text + tokenizer.eos_token)
        batch["text"] = texts
        return batch

    dataset = dataset.map(_apply, batched=True, desc="Formatting chat templates")
    return dataset.select_columns(["text"])


def compute_token_stats(
    dataset: Dataset,
    tokenizer,
    sample_size: int = 200,
) -> dict:
    """Return token length statistics over a sample of the dataset."""
    n = min(sample_size, len(dataset))
    lengths = [len(tokenizer.encode(t)) for t in dataset["text"][:n]]
    lengths_sorted = sorted(lengths)
    return {
        "sampled": n,
        "avg": sum(lengths) / len(lengths),
        "max": max(lengths),
        "min": min(lengths),
        "p50": lengths_sorted[len(lengths) // 2],
        "p95": lengths_sorted[int(len(lengths) * 0.95)],
    }


def load_and_prepare(
    train_path: str,
    val_path: Optional[str],
    tokenizer,
    val_fraction: float = 0.1,
    seed: int = 42,
    enable_thinking: bool = True,
    input_col: str = "",
    output_col: str = "",
    system_prompt: str = "",
) -> Tuple[Dataset, Dataset, dict]:
    """
    End-to-end: load any supported format, optionally split, format, return
    (train_ds, val_ds, token_stats).

    For CSV files, pass input_col and output_col.
    """
    train_raw = load_file(train_path, input_col, output_col, system_prompt)

    if val_path and Path(val_path).exists():
        val_raw = load_file(val_path, input_col, output_col, system_prompt)
    else:
        train_raw, val_raw = train_val_split(train_raw, val_fraction, seed)

    train_ds = format_for_training(
        records_to_dataset(train_raw), tokenizer, enable_thinking,
    )
    val_ds = format_for_training(
        records_to_dataset(val_raw), tokenizer, enable_thinking,
    )
    stats = compute_token_stats(train_ds, tokenizer)
    return train_ds, val_ds, stats
