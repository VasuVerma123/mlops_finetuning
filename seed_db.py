"""
Seed the local SQLite database with realistic dummy data.
Run: python seed_db.py

This populates datasets, model results, and evaluation results
so you can test the Experiments tab without running real training.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.database.db import (
    init_db, save_dataset, save_model_result, save_eval_result,
    get_datasets, get_models, get_eval_results,
)

print("Initialising database...")
init_db()

# ── Datasets ───────────────────────────────────────────────────────────

print("\nSeeding datasets...")

ds1 = save_dataset(
    name="customer_support_train.jsonl",
    s3_link="s3://my-bucket/datasets/customer_support_train.jsonl",
    file_type="jsonl",
    num_examples=1240,
    tool_call_examples=312,
    think_examples=890,
    avg_turns=4.2,
)

ds2 = save_dataset(
    name="tool_calling_v2.jsonl",
    s3_link="s3://my-bucket/datasets/tool_calling_v2.jsonl",
    file_type="jsonl",
    num_examples=580,
    tool_call_examples=580,
    think_examples=0,
    avg_turns=2.8,
)

ds3 = save_dataset(
    name="reasoning_finetune.csv",
    s3_link="",
    file_type="csv",
    num_examples=3200,
    tool_call_examples=0,
    think_examples=3200,
    avg_turns=1.0,
)

print(f"  [OK] {ds1[:8]}  customer_support_train.jsonl  (1240 examples)")
print(f"  [OK] {ds2[:8]}  tool_calling_v2.jsonl         (580 examples)")
print(f"  [OK] {ds3[:8]}  reasoning_finetune.csv         (3200 examples)")

# ── Models ─────────────────────────────────────────────────────────────

print("\nSeeding model training results...")

m1 = save_model_result(
    experiment_name="exp_seed_qwen3_4b_support",
    base_model="Qwen/Qwen3-4B",
    adapter_path="outputs/exp_seed_qwen3_4b_support/final_adapter",
    dataset_id=ds1,
    quantization="4bit",
    lora_r=16,
    lora_alpha=32,
    epochs=3,
    learning_rate=2e-4,
    final_train_loss=0.4821,
    best_eval_loss=0.5103,
    perplexity=1.666,
    val_accuracy_pct=72.0,
    training_time_s=4320.0,
    trainable_params=5242880,
    stacked_on=None,
)

m2 = save_model_result(
    experiment_name="exp_seed_qwen3_4b_tools",
    base_model="Qwen/Qwen3-4B",
    adapter_path="outputs/exp_seed_qwen3_4b_tools/final_adapter",
    dataset_id=ds2,
    quantization="4bit",
    lora_r=32,
    lora_alpha=64,
    epochs=5,
    learning_rate=1e-4,
    final_train_loss=0.3217,
    best_eval_loss=0.3894,
    perplexity=1.476,
    val_accuracy_pct=85.0,
    training_time_s=2880.0,
    trainable_params=10485760,
    stacked_on="outputs/exp_seed_qwen3_4b_support/final_adapter",
)

m3 = save_model_result(
    experiment_name="exp_seed_llama3_8b_reason",
    base_model="meta-llama/Meta-Llama-3-8B-Instruct",
    adapter_path="outputs/exp_seed_llama3_8b_reason/final_adapter",
    dataset_id=ds3,
    quantization="4bit",
    lora_r=16,
    lora_alpha=32,
    epochs=2,
    learning_rate=2e-4,
    final_train_loss=0.6104,
    best_eval_loss=0.6512,
    perplexity=1.918,
    val_accuracy_pct=61.0,
    training_time_s=9600.0,
    trainable_params=8388608,
    stacked_on=None,
)

print(f"  [OK] {m1[:8]}  exp_seed_qwen3_4b_support  loss=0.4821  ppl=1.666  val_acc=72%")
print(f"  [OK] {m2[:8]}  exp_seed_qwen3_4b_tools    loss=0.3217  ppl=1.476  val_acc=85%  (stacked)")
print(f"  [OK] {m3[:8]}  exp_seed_llama3_8b_reason  loss=0.6104  ppl=1.918  val_acc=61%")

# ── Eval results ───────────────────────────────────────────────────────

print("\nSeeding eval results...")

scenarios_support = [
    {"name": "greeting",          "accuracy_pct": 100.0, "latency_s": 1.2},
    {"name": "refund_request",    "accuracy_pct": 87.5,  "latency_s": 2.1},
    {"name": "product_lookup",    "accuracy_pct": 93.3,  "latency_s": 1.8},
    {"name": "escalation",        "accuracy_pct": 75.0,  "latency_s": 2.4},
]

scenarios_tools = [
    {"name": "search_tool_call",  "accuracy_pct": 95.0,  "latency_s": 1.5},
    {"name": "calendar_booking",  "accuracy_pct": 90.0,  "latency_s": 2.0},
    {"name": "multi_tool_chain",  "accuracy_pct": 80.0,  "latency_s": 3.1},
]

e1 = save_eval_result(
    test_file="support_test_cases.jsonl",
    overall_accuracy_pct=88.9,
    avg_latency_s=1.875,
    avg_tokens_per_sec=42.3,
    model_id=m1,
    experiment_name="exp_seed_qwen3_4b_support",
    p95_latency_s=2.8,
    ttft_s=0.32,
    scenario_results=scenarios_support,
)

e2 = save_eval_result(
    test_file="tool_test_cases.jsonl",
    overall_accuracy_pct=91.7,
    avg_latency_s=2.2,
    avg_tokens_per_sec=38.7,
    model_id=m2,
    experiment_name="exp_seed_qwen3_4b_tools",
    p95_latency_s=3.5,
    ttft_s=0.41,
    scenario_results=scenarios_tools,
)

e3 = save_eval_result(
    test_file="support_test_cases.jsonl",
    overall_accuracy_pct=61.2,
    avg_latency_s=3.1,
    avg_tokens_per_sec=28.4,
    model_id=m3,
    experiment_name="exp_seed_llama3_8b_reason",
    p95_latency_s=5.2,
    ttft_s=0.58,
    scenario_results=scenarios_support,
)

print(f"  [OK] {e1[:8]}  support eval  accuracy=88.9%  latency=1.875s  (fine-tuned)")
print(f"  [OK] {e2[:8]}  tools eval    accuracy=91.7%  latency=2.2s    (stacked adapter)")
print(f"  [OK] {e3[:8]}  support eval  accuracy=61.2%  latency=3.1s    (base model)")

# ── Summary ────────────────────────────────────────────────────────────

print("\nDatabase summary:")
print(f"  Datasets:     {len(get_datasets())}")
print(f"  Models:       {len(get_models())}")
print(f"  Eval results: {len(get_eval_results())}")
print(f"\nDB file: outputs/mlops.db")
print("Done. Open the app and go to the Experiments tab to view the records.")
