#!/usr/bin/env python3
"""
MLOps Fine-Tuning Pipeline — CLI Entry Point

Usage:
  python run.py train   --model qwen3_4b.yaml --train data.jsonl [--val val.jsonl] [--name exp1]
  python run.py eval    --model qwen3_4b.yaml --adapter outputs/exp1/final_adapter --test tests.jsonl
  python run.py serve   --model qwen3_4b.yaml --adapter outputs/exp1/final_adapter
  python run.py models                        # list available model configs
  python run.py validate --file data.jsonl    # validate dataset format
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import list_available_models, build_pipeline_config


def cmd_train(args):
    from src.training.trainer import run_training
    from src.tracking.mlflow_tracker import MLflowTracker

    overrides = {}
    if args.epochs:
        overrides["training.epochs"] = args.epochs
    if args.lr:
        overrides["training.learning_rate"] = args.lr
    if args.lora_r:
        overrides["lora.r"] = args.lora_r
    if args.lora_alpha:
        overrides["lora.alpha"] = args.lora_alpha

    cfg = build_pipeline_config(
        model_file=args.model,
        train_file=args.train,
        val_file=args.val or "",
        experiment_name=args.name,
        adapter_path=args.adapter,
        overrides=overrides if overrides else None,
    )

    print(f"\n{'=' * 60}")
    print(f"  MLOps Fine-Tuning Pipeline — TRAIN")
    print(f"  Model:      {cfg.model.display_name}")
    print(f"  Experiment: {cfg.experiment_name}")
    print(f"  Train file: {cfg.train_file}")
    print(f"  Output:     {cfg.output_dir}")
    if cfg.adapter_path:
        print(f"  Stacking:   {cfg.adapter_path}")
    print(f"{'=' * 60}\n")

    tracker = MLflowTracker(experiment_name=cfg.experiment_name)
    tracker.start_run(run_name=cfg.experiment_name)
    tracker.log_pipeline_config(cfg)

    def on_log(entry):
        step = entry.get("step", 0)
        tracker.log_training_step(step, entry)
        if "loss" in entry:
            print(f"  Step {step:5d} | Loss: {entry['loss']:.4f} | "
                  f"LR: {entry.get('lr', 0):.2e}")

    summary = run_training(cfg, on_log=on_log)
    tracker.log_training_summary(summary)
    tracker.log_adapter(summary.get("adapter_path", ""))
    tracker.end_run()

    print(f"\n{'=' * 60}")
    print(f"  Training complete!")
    print(f"  Final loss:    {summary.get('final_train_loss', '?')}")
    print(f"  Best eval:     {summary.get('best_eval_loss', '?')}")
    print(f"  Adapter saved: {summary.get('adapter_path', '?')}")
    print(f"{'=' * 60}\n")


def cmd_eval(args):
    from src.evaluation.harness import run_evaluation
    from src.evaluation.metrics import format_comparison_table
    from src.tracking.mlflow_tracker import MLflowTracker

    cfg = build_pipeline_config(
        model_file=args.model,
        train_file="",
        test_file=args.test,
        experiment_name="eval",
    )

    print(f"\n{'=' * 60}")
    print(f"  MLOps Fine-Tuning Pipeline — EVALUATE")
    print(f"  Model:   {cfg.model.display_name}")
    print(f"  Adapter: {args.adapter}")
    print(f"  Tests:   {args.test}")
    print(f"{'=' * 60}\n")

    def on_progress(info):
        print(f"  [{info['step']}/{info['total']}] {info['name']}")

    results = run_evaluation(
        cfg,
        adapter_path=args.adapter,
        test_file=args.test,
        system_prompt=args.system_prompt or "",
        on_progress=on_progress,
    )

    print(f"\n{format_comparison_table(results)}")

    # Log to MLflow
    tracker = MLflowTracker(experiment_name="evaluations")
    tracker.start_run(run_name=f"eval_{Path(args.adapter).parent.name}")
    tracker.log_evaluation_results(results)
    tracker.end_run()

    print(f"\n  Results saved to: {cfg.output_dir}/eval_results.json")


def cmd_serve(args):
    from src.serving.inference import InferenceEngine

    cfg = build_pipeline_config(
        model_file=args.model,
        train_file="",
        experiment_name="serve",
    )

    print(f"\n{'=' * 60}")
    print(f"  MLOps Fine-Tuning Pipeline — SERVE")
    print(f"  Model:   {cfg.model.display_name}")
    print(f"  Adapter: {args.adapter}")
    print(f"{'=' * 60}\n")
    print("  Loading model... (this may take a minute)")

    engine = InferenceEngine()
    engine.load(cfg, args.adapter)
    print("  Model loaded! Type your messages below (Ctrl+C to exit).\n")

    system_prompt = args.system_prompt or ""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    try:
        while True:
            user_input = input("You > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit"):
                break

            messages.append({"role": "user", "content": user_input})
            result = engine.generate(messages)

            response = result["response"]
            print(f"\nAssistant > {response}")
            print(f"  [{result['tokens_generated']} tokens, "
                  f"{result['latency_s']}s, "
                  f"{result['tokens_per_sec']} tok/s]\n")

            messages.append({"role": "assistant", "content": response})

    except KeyboardInterrupt:
        print("\n  Shutting down.")
    finally:
        engine.unload()


def cmd_models(args):
    models = list_available_models()
    print(f"\n  Available Models ({len(models)}):")
    print(f"  {'File':<25s} {'Display Name':<25s} {'Family':<10s} {'Size':<6s}")
    print(f"  {'-' * 70}")
    for m in models:
        print(f"  {m['file']:<25s} {m['display_name']:<25s} "
              f"{m['family']:<10s} {m['size']:<6s}")
    print()


def cmd_validate(args):
    from src.data.validator import validate_file, validate_test_cases

    path = args.file
    print(f"\n  Validating: {path}")

    if args.test_cases:
        result = validate_test_cases(path)
    else:
        result = validate_file(path)

    if result["valid"]:
        print(f"  VALID — {result['total']} records")
        if "stats" in result:
            stats = result["stats"]
            print(f"  Tool-call examples:    {stats.get('tool_call_examples', 0)}")
            print(f"  Think-augmented:       {stats.get('think_augmented_examples', 0)}")
            print(f"  Avg turns per example: {stats.get('avg_turns', 0):.1f}")
    else:
        print(f"  INVALID — {len(result['errors'])} errors:")
        for e in result["errors"][:10]:
            print(f"    {e}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="MLOps Fine-Tuning Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", help="Command to run")

    # ── train ──
    p_train = sub.add_parser("train", help="Fine-tune a model")
    p_train.add_argument("--model", required=True, help="Model YAML file (e.g. qwen3_4b.yaml)")
    p_train.add_argument("--train", required=True, help="Training JSONL path")
    p_train.add_argument("--val", help="Validation JSONL path (optional)")
    p_train.add_argument("--name", default="experiment", help="Experiment name")
    p_train.add_argument("--adapter", help="Previous adapter path for stacking")
    p_train.add_argument("--epochs", type=int, help="Override epochs")
    p_train.add_argument("--lr", type=float, help="Override learning rate")
    p_train.add_argument("--lora-r", type=int, help="Override LoRA rank")
    p_train.add_argument("--lora-alpha", type=int, help="Override LoRA alpha")

    # ── eval ──
    p_eval = sub.add_parser("eval", help="Evaluate a model")
    p_eval.add_argument("--model", required=True, help="Model YAML file")
    p_eval.add_argument("--adapter", required=True, help="Adapter directory path")
    p_eval.add_argument("--test", required=True, help="Test cases JSONL path")
    p_eval.add_argument("--system-prompt", help="System prompt for evaluation")

    # ── serve ──
    p_serve = sub.add_parser("serve", help="Interactive chat with a model")
    p_serve.add_argument("--model", required=True, help="Model YAML file")
    p_serve.add_argument("--adapter", required=True, help="Adapter directory path")
    p_serve.add_argument("--system-prompt", help="System prompt for chat")

    # ── models ──
    sub.add_parser("models", help="List available model configs")

    # ── validate ──
    p_val = sub.add_parser("validate", help="Validate a dataset file")
    p_val.add_argument("--file", required=True, help="JSONL file to validate")
    p_val.add_argument("--test-cases", action="store_true", help="Validate as test cases format")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    commands = {
        "train": cmd_train,
        "eval": cmd_eval,
        "serve": cmd_serve,
        "models": cmd_models,
        "validate": cmd_validate,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
