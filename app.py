"""
MLOps Fine-Tuning Hub — Streamlit + MLflow
==========================================
End-to-end local fine-tuning pipeline with experiment tracking and chat.

Tabs:
  1. Train     — pick model, upload dataset, configure, fine-tune on local GPU
  2. Evaluate  — run test scenarios, view accuracy/latency dashboard
  3. Chat      — load any trained adapter and talk to the model live
  4. Experiments — full MLflow run history, compare, inspect artifacts

Sidebar: experiment list, quick-select adapter for chat
"""

import copy, gc, json, math, os, re, sys, time, threading
from datetime import datetime, timedelta
from pathlib import Path

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import (
    list_available_models, build_pipeline_config, build_custom_model_config,
    load_model_config, PipelineConfig, load_defaults, TrainingConfig, EvalConfig,
)
from src.data.validator import validate_file, validate_csv, validate_test_cases
from src.data.loader import detect_file_type, get_csv_columns
from src.database.db import (
    init_db, save_dataset, save_model_result, save_eval_result,
    get_full_run_table, get_model_by_experiment, get_evals_for_model,
)

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# ── Constants ────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = ROOT / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)
_db_path = str(ROOT / "mlflow.db").replace("\\", "/")
MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{_db_path}")

PRESET_MODELS = {
    "Qwen/Qwen3-4B": "Qwen 3 — 4B",
    "Qwen/Qwen3-8B": "Qwen 3 — 8B",
    "Qwen/Qwen2.5-7B-Instruct": "Qwen 2.5 — 7B Instruct",
    "meta-llama/Meta-Llama-3-8B-Instruct": "Llama 3 — 8B Instruct",
    "meta-llama/Llama-3.2-3B-Instruct": "Llama 3.2 — 3B Instruct",
    "meta-llama/Llama-3.1-8B-Instruct": "Llama 3.1 — 8B Instruct",
}


# ── Page setup ───────────────────────────────────────────────────────

st.set_page_config(
    page_title="MLOps Fine-Tuning Hub",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] { gap: 1.5rem; }
    .stTabs [data-baseweb="tab"] { padding: 0.6rem 1.2rem; font-weight: 600; }
    .status-running { color: #FFD54F; font-weight: 700; }
    .status-done { color: #66BB6A; font-weight: 700; }
    .status-error { color: #EF5350; font-weight: 700; }
    .info-box {
        background: #1a1a2e; border-left: 4px solid #4FC3F7;
        padding: 0.7rem 1rem; border-radius: 0 0.5rem 0.5rem 0;
        margin-bottom: 0.5rem; font-size: 0.88rem;
    }
    div[data-testid="stChatMessage"] {
        background: #1e293b; border-radius: 12px;
        border: 1px solid #334155; margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

st.title("MLOps Fine-Tuning Hub")


# ── DB init & backfill ────────────────────────────────────────────────

init_db()   # creates tables if they don't exist (idempotent)


def backfill_db_from_outputs() -> int:
    """
    Scan outputs/ and register any adapters that aren't in the DB yet.
    This automatically populates old runs that were trained before the DB existed.
    Returns the number of runs newly added.
    """
    added = 0
    if not OUTPUTS_DIR.exists():
        return 0
    for exp_dir in sorted(OUTPUTS_DIR.iterdir()):
        adapter = exp_dir / "final_adapter"
        if not adapter.is_dir():
            continue
        if get_model_by_experiment(exp_dir.name):
            continue   # already in DB

        base_model = "unknown"
        config_path = adapter / "adapter_config.json"
        if config_path.exists():
            try:
                cfg = json.loads(config_path.read_text())
                base_model = cfg.get("base_model_name_or_path", "unknown")
            except Exception:
                pass

        final_train_loss = None
        best_eval_loss = None
        perplexity = None
        loss_path = exp_dir / "loss_log.json"
        if loss_path.exists():
            try:
                log = json.loads(loss_path.read_text())
                tl = log.get("train_losses", [])
                el = log.get("eval_losses", [])
                if tl:
                    final_train_loss = tl[-1]["loss"]
                if el:
                    best_eval_loss = min(e["eval_loss"] for e in el)
                    if best_eval_loss is not None:
                        perplexity = math.exp(best_eval_loss)
            except Exception:
                pass

        save_model_result(
            experiment_name=exp_dir.name,
            base_model=base_model,
            adapter_path=str(adapter),
            final_train_loss=final_train_loss,
            best_eval_loss=best_eval_loss,
            perplexity=perplexity,
        )
        added += 1
    return added


# Run backfill once per Streamlit session
if "db_backfilled" not in st.session_state:
    backfill_db_from_outputs()
    st.session_state["db_backfilled"] = True


# ── Helpers ──────────────────────────────────────────────────────────

def save_uploaded(uploaded, dest_dir: str) -> str:
    os.makedirs(dest_dir, exist_ok=True)
    path = os.path.join(dest_dir, uploaded.name)
    with open(path, "wb") as f:
        f.write(uploaded.getbuffer())
    return path


def find_adapters() -> list[dict]:
    """Scan outputs/ and mlruns/ for adapters with metadata."""
    adapters = []
    if OUTPUTS_DIR.exists():
        for exp_dir in sorted(OUTPUTS_DIR.iterdir()):
            adapter = exp_dir / "final_adapter"
            if adapter.is_dir():
                meta = {"path": str(adapter), "name": exp_dir.name, "source": "local"}
                config_path = adapter / "adapter_config.json"
                if config_path.exists():
                    try:
                        cfg = json.loads(config_path.read_text())
                        meta["base_model"] = cfg.get("base_model_name_or_path", "?")
                    except Exception:
                        pass
                loss_path = exp_dir / "loss_log.json"
                if loss_path.exists():
                    try:
                        log = json.loads(loss_path.read_text())
                        tl = log.get("train_losses", [])
                        el = log.get("eval_losses", [])
                        if tl:
                            meta["final_loss"] = tl[-1]["loss"]
                        if el:
                            meta["best_eval_loss"] = min(e["eval_loss"] for e in el)
                    except Exception:
                        pass
                adapters.append(meta)
    return adapters


def get_mlflow_runs() -> list[dict]:
    if not MLFLOW_AVAILABLE:
        return []
    try:
        import math
        mlflow.set_tracking_uri(MLFLOW_URI)
        experiments = mlflow.search_experiments()
        all_runs = []
        for exp in experiments:
            if exp.name == "Default":
                continue
            runs = mlflow.search_runs(
                experiment_ids=[exp.experiment_id],
                order_by=["start_time DESC"],
            )
            if len(runs) > 0:
                for _, run in runs.iterrows():
                    d = run.to_dict()
                    cleaned = {}
                    for k, v in d.items():
                        if isinstance(v, float) and math.isnan(v):
                            cleaned[k] = None
                        else:
                            cleaned[k] = v
                    all_runs.append(cleaned)
        return all_runs
    except Exception:
        return []


# ── MLflow setup ─────────────────────────────────────────────────────

if MLFLOW_AVAILABLE:
    mlflow.set_tracking_uri(MLFLOW_URI)


# ═════════════════════════════════════════════════════════════════════
#  TABS
# ═════════════════════════════════════════════════════════════════════

tab_train, tab_eval, tab_chat, tab_experiments = st.tabs([
    "🏋️ Train", "📊 Evaluate", "💬 Chat", "🔬 Experiments",
])


# ═════════════════════════════════════════════════════════════════════
#  TAB 1: TRAIN
# ═════════════════════════════════════════════════════════════════════

with tab_train:
    st.header("Fine-Tune a Model")

    col_cfg, col_progress = st.columns([1, 1.5])

    with col_cfg:
        # Step 1: Model
        st.subheader("Step 1: Choose Model")
        model_source = st.radio("Model source", ["Preset", "Custom HuggingFace ID"], horizontal=True, key="t_msrc")

        selected_hf_id = ""
        if model_source == "Preset":
            hf_ids = list(PRESET_MODELS.keys())
            sel_idx = st.selectbox("Model", range(len(hf_ids)),
                                   format_func=lambda i: f"{PRESET_MODELS[hf_ids[i]]}  ({hf_ids[i]})")
            selected_hf_id = hf_ids[sel_idx]
        else:
            selected_hf_id = st.text_input("HuggingFace Model ID", placeholder="org/model-name")

        if selected_hf_id:
            st.caption(f"Will use: `{selected_hf_id}`")

        # Step 2: Dataset
        st.markdown("---")
        st.subheader("Step 2: Upload Dataset")
        st.markdown('<div class="info-box">Supports <b>JSONL</b> (chat messages format) and <b>CSV</b> '
                    '(pick input/output columns).</div>', unsafe_allow_html=True)

        train_upload = st.file_uploader("Training Data", type=["jsonl", "json", "csv"], key="t_train")
        val_upload = st.file_uploader("Validation Data (optional)", type=["jsonl", "json", "csv"], key="t_val")

        train_path = ""
        val_path = ""
        input_col = ""
        output_col = ""
        csv_sys_prompt = ""
        exp_name = st.text_input("Experiment Name", value=f"exp_{datetime.now().strftime('%Y%m%d_%H%M')}")
        upload_dir = str(OUTPUTS_DIR / exp_name / "data")

        if train_upload:
            train_path = save_uploaded(train_upload, upload_dir)
            ftype = detect_file_type(train_path)

            if ftype == "csv":
                st.info("CSV detected — map your columns:")
                try:
                    columns = get_csv_columns(train_path)
                    c1, c2 = st.columns(2)
                    with c1:
                        input_col = st.selectbox("Input column", columns, key="t_icol")
                    with c2:
                        output_col = st.selectbox("Output column",
                                                  [c for c in columns if c != input_col], key="t_ocol")
                    csv_sys_prompt = st.text_area("System prompt (prepended to each row, optional)",
                                                  height=60, key="t_sysp")
                    if input_col and output_col:
                        result = validate_csv(train_path, input_col, output_col)
                        if result["valid"]:
                            n = result["stats"]["usable_rows"]
                            st.success(f"{n} usable rows")
                            _ds_id = save_dataset(
                                name=train_upload.name, file_type="csv",
                                num_examples=n,
                            )
                            st.session_state["last_dataset_id"] = _ds_id
                        else:
                            for e in result["errors"][:3]:
                                st.error(e)
                except Exception as exc:
                    st.error(str(exc))
            else:
                result = validate_file(train_path)
                if result["valid"]:
                    st.success(f"{result['total']} examples — valid")
                    s = result["stats"]
                    c1, c2 = st.columns(2)
                    c1.metric("Tool-call", s.get("tool_call_examples", 0))
                    c2.metric("Think-augmented", s.get("think_augmented_examples", 0))
                    _ds_id = save_dataset(
                        name=train_upload.name, file_type=ftype,
                        num_examples=result["total"],
                        tool_call_examples=s.get("tool_call_examples", 0),
                        think_examples=s.get("think_augmented_examples", 0),
                    )
                    st.session_state["last_dataset_id"] = _ds_id
                    st.session_state["last_dataset_stats"] = s
                else:
                    for e in result["errors"][:5]:
                        st.error(e)

        if val_upload:
            val_path = save_uploaded(val_upload, upload_dir)

        # Step 3: Adapter stacking
        st.markdown("---")
        st.subheader("Step 3: Adapter Stacking")
        existing = find_adapters()
        adapter_path = None
        if existing:
            use_stack = st.checkbox("Stack on previous adapter")
            if use_stack:
                adapter_path = st.selectbox("Previous Adapter",
                                            [a["path"] for a in existing],
                                            format_func=lambda p: Path(p).parent.name)
        else:
            st.caption("No previous adapters. Training from scratch.")

        # Step 4: Hyperparameters
        st.markdown("---")
        st.subheader("Step 4: Hyperparameters")
        with st.expander("Training", expanded=False):
            epochs = st.slider("Epochs", 1, 10, 3, key="t_ep")
            lr = st.select_slider("Learning Rate", [5e-5, 1e-4, 2e-4, 3e-4, 5e-4], value=2e-4, key="t_lr")
            grad_accum = st.select_slider("Gradient Accumulation", [1, 2, 4, 8, 16], value=8, key="t_ga")
            max_seq = st.slider("Max Sequence Length", 512, 4096, 1536, 128, key="t_seq")
        with st.expander("QLoRA / LoRA", expanded=False):
            lora_r = st.select_slider("LoRA Rank", [8, 16, 32, 64], value=16, key="t_lr_r")
            lora_alpha = st.select_slider("LoRA Alpha", [16, 32, 64, 128], value=32, key="t_lr_a")
            quant = st.selectbox("Quantization", ["4bit", "8bit", "none"], key="t_quant")
            st.caption({"4bit": "QLoRA — 4-bit NF4 + LoRA (recommended, lowest VRAM)",
                        "8bit": "8-bit quantization + LoRA",
                        "none": "Full precision LoRA (needs lots of VRAM)"}[quant])

        # Step 5: Test Cases (optional — for auto-eval after training)
        st.markdown("---")
        st.subheader("Step 5: Test Cases (optional)")
        st.caption("Upload test cases to automatically evaluate accuracy after fine-tuning.")
        test_upload_train = st.file_uploader("Test Cases JSONL", type=["jsonl", "json"], key="t_test")
        test_path_train = ""
        if test_upload_train:
            test_path_train = save_uploaded(test_upload_train, upload_dir)
            r = validate_test_cases(test_path_train)
            if r["valid"]:
                st.success(f"{r['total']} test cases — will auto-evaluate after training")
            else:
                for e in r["errors"][:3]:
                    st.error(e)
                test_path_train = ""

        eval_sys_prompt_train = ""
        if test_path_train:
            eval_sys_prompt_train = st.text_area("Eval System Prompt (optional)", height=60, key="t_eval_sysp")

        # Step 6: Launch
        st.markdown("---")
        can_train = bool(selected_hf_id and train_path)

        launch = st.button("Launch Training", type="primary", disabled=not can_train, width="stretch")

    with col_progress:
        st.subheader("Training Progress")

        if launch and can_train:
            model_cfg = build_custom_model_config(
                selected_hf_id, quantization=quant,
                lora_r=lora_r, lora_alpha=lora_alpha, max_seq_length=max_seq,
            )
            defaults = load_defaults()
            train_data = defaults.get("training", {})
            train_data.update(epochs=epochs, learning_rate=lr,
                              gradient_accumulation=grad_accum)
            eval_data = defaults.get("evaluation", {})

            cfg = PipelineConfig(
                model=model_cfg,
                training=TrainingConfig(**train_data),
                evaluation=EvalConfig(**eval_data),
                output_dir=str(OUTPUTS_DIR / exp_name),
                experiment_name=exp_name,
                adapter_path=adapter_path,
                train_file=train_path,
                val_file=val_path,
                input_col=input_col,
                output_col=output_col,
                system_prompt=csv_sys_prompt,
            )

            status_box = st.status("Training in progress...", expanded=True)
            chart_slot = st.empty()
            metrics_slot = st.empty()

            train_logs = []

            try:
                from src.training.trainer import run_training

                if MLFLOW_AVAILABLE:
                    mlflow.set_tracking_uri(MLFLOW_URI)
                    mlflow.set_experiment(exp_name)
                    mlflow.start_run(run_name=exp_name, tags={
                        "model": selected_hf_id, "quantization": quant,
                    })
                    mlflow.log_params({
                        "model_name": selected_hf_id, "quantization": quant,
                        "lora_r": lora_r, "lora_alpha": lora_alpha,
                        "epochs": epochs, "learning_rate": lr,
                        "gradient_accumulation": grad_accum,
                        "max_seq_length": max_seq,
                        "adapter_stacking": adapter_path is not None,
                        "train_file": Path(train_path).name,
                    })

                def on_log(entry):
                    train_logs.append(entry)
                    if MLFLOW_AVAILABLE:
                        step = entry.get("step", 0)
                        m = {}
                        if "loss" in entry:
                            m["train_loss"] = entry["loss"]
                        if "lr" in entry:
                            m["learning_rate"] = entry["lr"]
                        if "eval_loss" in entry:
                            m["eval_loss"] = entry["eval_loss"]
                        if m:
                            mlflow.log_metrics(m, step=step)

                    losses = [l for l in train_logs if "loss" in l]
                    evals = [l for l in train_logs if "eval_loss" in l]
                    if losses:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=[l["step"] for l in losses], y=[l["loss"] for l in losses],
                            mode="lines+markers", name="Train Loss", line=dict(color="#4FC3F7")))
                        if evals:
                            fig.add_trace(go.Scatter(
                                x=[l["step"] for l in evals], y=[l["eval_loss"] for l in evals],
                                mode="lines+markers", name="Eval Loss", line=dict(color="#FF7043")))
                        fig.update_layout(title="Loss Curve (live)", template="plotly_dark", height=300,
                                          margin=dict(l=40, r=20, t=40, b=30))
                        chart_slot.plotly_chart(fig, width="stretch")

                    last = losses[-1] if losses else {}
                    with metrics_slot.container():
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Step", last.get("step", 0))
                        c2.metric("Loss", f"{last.get('loss', 0):.4f}")
                        gpu_str = ""
                        try:
                            import torch as _t
                            if _t.cuda.is_available():
                                mem = _t.cuda.memory_allocated() / 1024**3
                                tot = _t.cuda.get_device_properties(0).total_memory / 1024**3
                                gpu_str = f"{mem:.1f}/{tot:.1f} GB"
                        except Exception:
                            pass
                        c3.metric("GPU", gpu_str or "N/A")

                    status_box.update(label=f"Step {last.get('step', 0)} — loss {last.get('loss', 0):.4f}")

                summary = run_training(cfg, on_log=on_log)

                # ── Save training result to local DB ──
                _ds_stats = st.session_state.get("last_dataset_stats", {})
                _db_model_id = save_model_result(
                    experiment_name=exp_name,
                    base_model=selected_hf_id,
                    adapter_path=summary.get("adapter_path", ""),
                    dataset_id=st.session_state.get("last_dataset_id"),
                    quantization=quant,
                    lora_r=lora_r,
                    lora_alpha=lora_alpha,
                    epochs=epochs,
                    learning_rate=lr,
                    final_train_loss=summary.get("final_train_loss"),
                    best_eval_loss=summary.get("best_eval_loss") or summary.get("eval_loss"),
                    perplexity=summary.get("perplexity"),
                    val_accuracy_pct=summary.get("val_accuracy_pct"),
                    training_time_s=summary.get("total_time_s"),
                    trainable_params=summary.get("trainable_params"),
                    stacked_on=adapter_path,
                )
                st.session_state["last_db_model_id"] = _db_model_id

                # ── Auto-evaluate if test cases provided ──
                eval_results = None
                if test_path_train:
                    status_box.update(label="Evaluating on test cases...", state="running")
                    try:
                        from src.evaluation.harness import run_evaluation
                        eval_cfg = PipelineConfig(
                            model=model_cfg,
                            training=TrainingConfig(**train_data),
                            evaluation=EvalConfig(**eval_data),
                            output_dir=str(OUTPUTS_DIR / exp_name),
                            test_file=test_path_train,
                        )
                        eval_results = run_evaluation(
                            eval_cfg, summary.get("adapter_path", ""),
                            test_file=test_path_train,
                            system_prompt=eval_sys_prompt_train,
                        )
                        summary["eval_results"] = eval_results
                        # Save eval to DB
                        save_eval_result(
                            test_file=Path(test_path_train).name,
                            overall_accuracy_pct=eval_results.get("overall_accuracy_pct", 0),
                            avg_latency_s=eval_results.get("avg_latency_s", 0),
                            avg_tokens_per_sec=eval_results.get("avg_tokens_per_sec", 0),
                            model_id=st.session_state.get("last_db_model_id"),
                            experiment_name=exp_name,
                            ttft_s=eval_results.get("avg_ttft_s", 0),
                            scenario_results=eval_results.get("scenarios", []),
                        )
                    except Exception as eval_exc:
                        st.warning(f"Auto-evaluation failed: {eval_exc}")

                if MLFLOW_AVAILABLE:
                    metrics = {}
                    if summary.get("final_train_loss") is not None:
                        metrics["final_train_loss"] = summary["final_train_loss"]
                    if summary.get("best_eval_loss") is not None:
                        metrics["best_eval_loss"] = summary["best_eval_loss"]
                    if summary.get("eval_loss") is not None:
                        metrics["eval_loss"] = summary["eval_loss"]
                    if summary.get("perplexity") is not None:
                        metrics["perplexity"] = summary["perplexity"]
                    if summary.get("val_accuracy_pct") is not None:
                        metrics["accuracy_pct"] = summary["val_accuracy_pct"]
                    if summary.get("total_time_s"):
                        metrics["training_time_s"] = summary["total_time_s"]
                    if summary.get("trainable_params"):
                        metrics["trainable_params"] = summary["trainable_params"]
                    if eval_results:
                        metrics["accuracy_pct"] = eval_results.get("overall_accuracy_pct", 0)
                        metrics["avg_latency_s"] = eval_results.get("avg_latency_s", 0)
                        metrics["avg_tokens_per_sec"] = eval_results.get("avg_tokens_per_sec", 0)
                        metrics["ttft_s"] = eval_results.get("avg_ttft_s", 0)
                    if metrics:
                        mlflow.log_metrics(metrics)
                    ap = summary.get("adapter_path", "")
                    if ap and os.path.isdir(ap):
                        mlflow.log_artifacts(ap, "adapter")
                    loss_log = os.path.join(cfg.output_dir, "loss_log.json")
                    if os.path.exists(loss_log):
                        mlflow.log_artifact(loss_log)
                    mlflow.end_run()

                status_box.update(label="Training complete!", state="complete", expanded=True)

                st.balloons()
                fl = summary.get("final_train_loss") or 0
                el = summary.get("best_eval_loss") or summary.get("eval_loss") or 0
                ppl = summary.get("perplexity") or 0
                val_acc = summary.get("val_accuracy_pct")
                time_s = summary.get("total_time_s") or 0

                if eval_results:
                    acc = eval_results.get("overall_accuracy_pct", 0)
                    c1, c2, c3, c4, c5 = st.columns(5)
                    c1.metric("Train Loss", f"{fl:.4f}")
                    c2.metric("Eval Loss", f"{el:.4f}")
                    c3.metric("Accuracy", f"{acc:.1f}%")
                    c4.metric("Perplexity", f"{ppl:.1f}")
                    c5.metric("Duration", f"{time_s // 60:.0f}m {time_s % 60:.0f}s")
                    lat = eval_results.get("avg_latency_s", 0)
                    tps = eval_results.get("avg_tokens_per_sec", 0)
                    st.info(f"Latency: {lat:.2f}s | Tokens/sec: {tps:.1f}")
                else:
                    c1, c2, c3, c4, c5 = st.columns(5)
                    c1.metric("Train Loss", f"{fl:.4f}")
                    c2.metric("Eval Loss", f"{el:.4f}")
                    c3.metric("Accuracy", f"{val_acc:.1f}%" if val_acc is not None else "—")
                    c4.metric("Perplexity", f"{ppl:.1f}" if ppl else "—")
                    c5.metric("Duration", f"{time_s // 60:.0f}m {time_s % 60:.0f}s")

                st.success(f"Adapter saved to: `{summary.get('adapter_path', '')}`")
                st.info("Go to the **Chat** tab to talk to your fine-tuned model!")

                st.session_state["train_status"] = "done"
                st.session_state["train_summary"] = summary

            except Exception as exc:
                if MLFLOW_AVAILABLE:
                    try:
                        mlflow.end_run(status="FAILED")
                    except Exception:
                        pass
                status_box.update(label="Training failed", state="error")
                st.error(f"Error: {exc}")

        elif st.session_state.get("train_status") == "done":
            s = st.session_state.get("train_summary", {})
            st.markdown('<span class="status-done">Last training completed!</span>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            fl = s.get("final_train_loss") or 0
            el = s.get("best_eval_loss") or 0
            time_s = s.get("total_time_s") or 0
            c1.metric("Final Loss", f"{fl:.4f}")
            c2.metric("Best Eval", f"{el:.4f}")
            c3.metric("Duration", f"{time_s // 60:.0f}m {time_s % 60:.0f}s")
            st.success(f"Adapter: `{s.get('adapter_path', '')}`")
        else:
            st.info("Configure training on the left and click **Launch Training**.")


# ═════════════════════════════════════════════════════════════════════
#  TAB 2: EVALUATE
# ═════════════════════════════════════════════════════════════════════

with tab_eval:
    st.header("Evaluate a Model")
    col_ecfg, col_eres = st.columns([1, 1.5])

    with col_ecfg:
        eval_adapters = find_adapters()
        if eval_adapters:
            eval_sel = st.selectbox("Adapter", range(len(eval_adapters)),
                                    format_func=lambda i: f"{eval_adapters[i]['name']} ({eval_adapters[i].get('base_model', '?')})",
                                    key="e_adapter")
            eval_adapter_path = eval_adapters[eval_sel]["path"]
        else:
            st.warning("No adapters found. Train a model first.")
            eval_adapter_path = None

        eval_model_id = st.text_input("Base Model ID", value="Qwen/Qwen3-4B", key="e_model")

        st.markdown("---")
        test_upload = st.file_uploader("Test Cases JSONL", type=["jsonl", "json"], key="e_test")
        test_path = ""
        if test_upload:
            test_path = save_uploaded(test_upload, str(OUTPUTS_DIR / "eval_data"))
            r = validate_test_cases(test_path)
            if r["valid"]:
                st.success(f"{r['total']} test cases loaded")
            else:
                st.error(str(r["errors"][:3]))

        eval_sys_prompt = st.text_area("System Prompt", height=100, key="e_sysp",
                                       placeholder="System prompt for eval (optional)")

        can_eval = bool(eval_adapter_path and test_path and eval_model_id)
        if st.button("Run Evaluation", type="primary", disabled=not can_eval, width="stretch"):
            with st.spinner("Evaluating..."):
                try:
                    from src.evaluation.harness import run_evaluation
                    cfg = PipelineConfig(
                        model=build_custom_model_config(eval_model_id),
                        training=TrainingConfig(),
                        evaluation=EvalConfig(),
                        output_dir=str(OUTPUTS_DIR / "eval_run"),
                        test_file=test_path,
                    )
                    results = run_evaluation(cfg, eval_adapter_path, test_file=test_path,
                                             system_prompt=eval_sys_prompt)
                    st.session_state["eval_results"] = results

                    # Save to DB
                    _eval_exp = Path(eval_adapter_path).parent.name
                    _eval_model = get_model_by_experiment(_eval_exp)
                    save_eval_result(
                        test_file=Path(test_path).name,
                        overall_accuracy_pct=results.get("overall_accuracy_pct", 0),
                        avg_latency_s=results.get("avg_latency_s", 0),
                        avg_tokens_per_sec=results.get("avg_tokens_per_sec", 0),
                        model_id=_eval_model["model_id"] if _eval_model else None,
                        experiment_name=_eval_exp,
                        ttft_s=results.get("avg_ttft_s", 0),
                        scenario_results=results.get("scenarios", []),
                    )

                    if MLFLOW_AVAILABLE:
                        mlflow.set_tracking_uri(MLFLOW_URI)
                        mlflow.set_experiment("evaluations")
                        mlflow.start_run(run_name=f"eval_{Path(eval_adapter_path).parent.name}")
                        mlflow.log_metrics({
                            "overall_accuracy_pct": results.get("overall_accuracy_pct", 0),
                            "avg_latency_s": results.get("avg_latency_s", 0),
                            "avg_tokens_per_sec": results.get("avg_tokens_per_sec", 0),
                        })
                        mlflow.end_run()
                except Exception as exc:
                    st.error(f"Evaluation failed: {exc}")

    with col_eres:
        st.subheader("Results")
        results = st.session_state.get("eval_results")
        if results:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", f"{results['overall_accuracy_pct']:.1f}%")
            c2.metric("Avg Latency", f"{results['avg_latency_s']:.2f}s")
            c3.metric("Tok/sec", f"{results['avg_tokens_per_sec']:.1f}")
            c4.metric("TTFT", f"{results['avg_ttft_s']:.3f}s")

            scenarios = results.get("scenarios", [])
            if scenarios:
                fig = px.bar(x=[s["name"] for s in scenarios], y=[s["accuracy_pct"] for s in scenarios],
                             color=[s["accuracy_pct"] for s in scenarios],
                             color_continuous_scale=["#EF5350", "#FFD54F", "#66BB6A"], range_color=[0, 100],
                             labels={"x": "Scenario", "y": "Accuracy %"}, title="Per-Scenario Accuracy")
                fig.update_layout(template="plotly_dark", height=350, showlegend=False)
                st.plotly_chart(fig, width="stretch")

                with st.expander("Details"):
                    for sc in scenarios:
                        icon = "✅" if sc["accuracy_pct"] == 100 else "⚠️" if sc["accuracy_pct"] > 0 else "❌"
                        st.markdown(f"**{icon} {sc['name']}** — {sc['accuracy_pct']:.0f}%")
                        for k, v in sc.get("checks", {}).items():
                            st.text(f"  {'✅' if v == 'PASS' else '❌'} {k}")
        else:
            st.info("Run an evaluation to see results.")


# ═════════════════════════════════════════════════════════════════════
#  TAB 3: CHAT
# ═════════════════════════════════════════════════════════════════════

with tab_chat:
    st.header("Chat with a Fine-Tuned Model")

    adapters = find_adapters()
    if not adapters:
        st.warning("No trained adapters found. Fine-tune a model first in the **Train** tab.")
    else:
        chat_col_cfg, chat_col_main = st.columns([1, 2])

        with chat_col_cfg:
            st.subheader("Model Selection")
            def _adapter_label(i):
                a = adapters[i]
                label = a["name"]
                if "final_loss" in a:
                    label += f" (loss: {a['final_loss']:.4f})"
                return label

            chat_sel = st.selectbox(
                "Select Adapter",
                range(len(adapters)),
                format_func=_adapter_label,
                key="chat_adapter_sel",
            )
            chat_adapter = adapters[chat_sel]

            chat_model_id = st.text_input(
                "Base Model ID",
                value=chat_adapter.get("base_model", "Qwen/Qwen3-4B"),
                key="chat_model_id",
            )

            chat_system_prompt = st.text_area(
                "System Prompt",
                value="You are a helpful assistant. Think step-by-step before answering. "
                      "Wrap reasoning in <think></think> tags.",
                height=150,
                key="chat_sys_prompt",
            )

            chat_temp = st.slider("Temperature", 0.1, 1.5, 0.3, 0.05, key="chat_temp")
            chat_max_tokens = st.slider("Max Tokens", 100, 2000, 800, 50, key="chat_maxtok")

            st.markdown("---")
            st.markdown(f"**Adapter:** `{chat_adapter['name']}`")
            if "final_loss" in chat_adapter:
                st.markdown(f"**Final Loss:** {chat_adapter['final_loss']:.4f}")
            if "best_eval_loss" in chat_adapter:
                st.markdown(f"**Best Eval:** {chat_adapter['best_eval_loss']:.4f}")

            if st.button("Load Model", type="primary", width="stretch", key="chat_load"):
                st.session_state["chat_load_request"] = {
                    "model_id": chat_model_id,
                    "adapter_path": chat_adapter["path"],
                    "system_prompt": chat_system_prompt,
                }
                st.session_state["chat_messages"] = []

            if st.button("Clear Chat", width="stretch", key="chat_clear"):
                st.session_state["chat_messages"] = []
                st.rerun()

        with chat_col_main:
            if "chat_messages" not in st.session_state:
                st.session_state["chat_messages"] = []

            load_req = st.session_state.get("chat_load_request")
            model_loaded = st.session_state.get("chat_model_loaded", False)

            if load_req and not model_loaded:
                with st.spinner(f"Loading {load_req['model_id']} + adapter... (first time takes ~2 min)"):
                    try:
                        import torch
                        from peft import PeftModel
                        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

                        bnb = BitsAndBytesConfig(
                            load_in_4bit=True, bnb_4bit_quant_type="nf4",
                            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
                        )
                        model = AutoModelForCausalLM.from_pretrained(
                            load_req["model_id"], quantization_config=bnb,
                            device_map="auto", trust_remote_code=True,
                        )
                        tok = AutoTokenizer.from_pretrained(
                            load_req["model_id"], trust_remote_code=True, padding_side="right",
                        )
                        if tok.pad_token is None:
                            tok.pad_token = tok.eos_token

                        model = PeftModel.from_pretrained(model, load_req["adapter_path"])
                        model.eval()
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                        st.session_state["chat_model"] = model
                        st.session_state["chat_tokenizer"] = tok
                        st.session_state["chat_model_loaded"] = True
                        st.session_state["chat_loaded_info"] = {
                            "model_id": load_req["model_id"],
                            "adapter": Path(load_req["adapter_path"]).parent.name,
                        }
                        st.success("Model loaded!")
                    except Exception as exc:
                        st.error(f"Failed to load model: {exc}")

            if model_loaded:
                info = st.session_state.get("chat_loaded_info", {})
                st.caption(f"Model: `{info.get('model_id', '?')}` | Adapter: `{info.get('adapter', '?')}`")

            chat_container = st.container(height=500)
            with chat_container:
                if not st.session_state["chat_messages"]:
                    with st.chat_message("assistant", avatar="🤖"):
                        st.markdown("Hello! I'm ready to chat. What would you like to discuss?")

                for msg in st.session_state["chat_messages"]:
                    with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🤖"):
                        if msg.get("thinking"):
                            with st.expander("🧠 Reasoning"):
                                st.code(msg["thinking"])
                        st.markdown(msg["content"])

            user_input = st.chat_input("Type your message...", key="chat_input")

            if user_input and model_loaded:
                st.session_state["chat_messages"].append({"role": "user", "content": user_input})

                model = st.session_state["chat_model"]
                tok = st.session_state["chat_tokenizer"]
                sys_p = st.session_state.get("chat_load_request", {}).get("system_prompt", "")

                msgs_for_model = []
                if sys_p:
                    msgs_for_model.append({"role": "system", "content": sys_p})
                for m in st.session_state["chat_messages"]:
                    msgs_for_model.append({"role": m["role"], "content": m["content"]})

                import torch
                try:
                    prompt = tok.apply_chat_template(msgs_for_model, tokenize=False,
                                                     add_generation_prompt=True, enable_thinking=True)
                except TypeError:
                    prompt = tok.apply_chat_template(msgs_for_model, tokenize=False,
                                                     add_generation_prompt=True)

                inputs = tok(prompt, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=chat_max_tokens,
                                         temperature=chat_temp, top_p=0.9, do_sample=True,
                                         repetition_penalty=1.1)
                resp = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                del inputs, out
                torch.cuda.empty_cache()

                think_match = re.search(r"<think>(.*?)</think>", resp, re.DOTALL)
                thinking = think_match.group(1).strip() if think_match else None
                visible = re.sub(r"<think>.*?</think>", "", resp, flags=re.DOTALL).strip()

                st.session_state["chat_messages"].append({
                    "role": "assistant", "content": visible, "thinking": thinking,
                })
                st.rerun()

            elif user_input and not model_loaded:
                st.warning("Click **Load Model** first to start chatting.")


# ═════════════════════════════════════════════════════════════════════
#  TAB 4: EXPERIMENTS
# ═════════════════════════════════════════════════════════════════════

with tab_experiments:
    # Refresh backfill every visit to pick up any newly trained adapters
    backfill_db_from_outputs()

    db_runs = get_full_run_table()   # always from DB — complete metrics for all runs

    if not db_runs:
        st.info("No experiments yet. Train a model to see history here.")
    else:
        # ── Overview ──────────────────────────────────────────────────
        all_losses = [r["Train Loss"] for r in db_runs if r["Train Loss"] is not None]
        all_val_acc = [r["Val Acc %"] for r in db_runs if r["Val Acc %"] is not None]
        all_test_acc = [r["Test Acc %"] for r in db_runs if r["Test Acc %"] is not None]
        best_acc = max(all_test_acc) if all_test_acc else (max(all_val_acc) if all_val_acc else None)

        st.markdown("### Overview")
        ov1, ov2, ov3, ov4 = st.columns(4)
        ov1.metric("Total Runs", len(db_runs))
        ov2.metric("With Eval", len([r for r in db_runs if r["Test Acc %"] is not None]))
        ov3.metric("Best Train Loss", f"{min(all_losses):.4f}" if all_losses else "—")
        ov4.metric("Best Accuracy", f"{best_acc:.1f}%" if best_acc is not None else "—")

        st.markdown("---")

        # ── All Runs table (DB-backed — always complete) ──────────────
        st.markdown("### All Runs")
        st.caption("Source: local DB — all metrics stored explicitly for every run.")

        def _fmt(val, kind="f4"):
            if val is None:
                return "—"
            if kind == "f4":
                return f"{val:.4f}"
            if kind == "f2":
                return f"{val:.2f}"
            if kind == "f1":
                return f"{val:.1f}"
            if kind == "pct":
                return f"{val:.1f}%"
            if kind == "s":
                return f"{val:.2f}s"
            if kind == "m":
                return f"{val / 60:.0f}m"
            if kind == "M":
                return f"{val / 1e6:.1f}M"
            return str(val)

        display = []
        for r in db_runs:
            ds_label = r["Dataset"] if r["Dataset"] != "—" else "—"
            if ds_label != "—" and r.get("DS Examples"):
                ds_label = f"{ds_label} ({int(r['DS Examples'])} rows)"
            display.append({
                "Run":          r["Run"],
                "Model":        r["Base Model"],
                "Dataset":      ds_label,
                "Quant":        r["Quant"],
                "Epochs":       r["Epochs"],
                "LR":           r["LR"],
                "Train Loss":   _fmt(r["Train Loss"], "f4"),
                "Eval Loss":    _fmt(r["Eval Loss"], "f4"),
                "Perplexity":   _fmt(r["Perplexity"], "f2"),
                "Val Acc %":    _fmt(r["Val Acc %"], "pct"),
                "Test Acc %":   _fmt(r["Test Acc %"], "pct"),
                "Latency s":    _fmt(r["Latency s"], "s"),
                "Tok/s":        _fmt(r["Tok/s"], "f1"),
                "TTFT s":       _fmt(r["TTFT s"], "f2"),
                "Duration":     _fmt(r["Duration s"], "m"),
                "Params":       _fmt(r["Params"], "M"),
                "Stacked On":   (r["Stacked On"] or "—")[:30],
            })

        st.dataframe(display, width="stretch", hide_index=True)

        st.markdown("---")

        # ── Comparison charts ─────────────────────────────────────────
        st.markdown("### Compare Runs")

        chart_col1, chart_col2 = st.columns(2)

        loss_runs = [r for r in db_runs if r["Train Loss"] is not None]
        with chart_col1:
            if loss_runs:
                names  = [r["Run"] for r in loss_runs]
                losses = [r["Train Loss"] for r in loss_runs]
                colors = ["#EF5350" if l == max(losses) else "#66BB6A" if l == min(losses) else "#4FC3F7" for l in losses]
                fig = go.Figure(go.Bar(x=names, y=losses, marker_color=colors,
                                       text=[f"{l:.4f}" for l in losses], textposition="outside"))
                fig.update_layout(title="Train Loss (lower is better)", template="plotly_dark",
                                  height=320, yaxis_title="Loss", margin=dict(t=40, b=30))
                st.plotly_chart(fig, width="stretch")
            else:
                st.caption("No train loss data yet.")

        # Prefer test accuracy; fall back to val accuracy
        acc_runs = [r for r in db_runs if r["Test Acc %"] is not None or r["Val Acc %"] is not None]
        with chart_col2:
            if acc_runs:
                names = [r["Run"] for r in acc_runs]
                accs  = [r["Test Acc %"] if r["Test Acc %"] is not None else r["Val Acc %"] for r in acc_runs]
                colors = ["#66BB6A" if a == max(accs) else "#EF5350" if a == min(accs) else "#4FC3F7" for a in accs]
                fig = go.Figure(go.Bar(x=names, y=accs, marker_color=colors,
                                       text=[f"{a:.1f}%" for a in accs], textposition="outside"))
                fig.update_layout(title="Accuracy % (higher is better)", template="plotly_dark",
                                  height=320, yaxis_title="Accuracy %", yaxis_range=[0, 105],
                                  margin=dict(t=40, b=30))
                st.plotly_chart(fig, width="stretch")
            else:
                st.caption("No accuracy data yet. Upload test cases during training to get accuracy.")

        chart_col3, chart_col4 = st.columns(2)

        with chart_col3:
            ppl_runs = [r for r in db_runs if r["Perplexity"] is not None]
            if ppl_runs:
                names = [r["Run"] for r in ppl_runs]
                ppls  = [r["Perplexity"] for r in ppl_runs]
                colors = ["#EF5350" if p == max(ppls) else "#66BB6A" if p == min(ppls) else "#FF7043" for p in ppls]
                fig = go.Figure(go.Bar(x=names, y=ppls, marker_color=colors,
                                       text=[f"{p:.2f}" for p in ppls], textposition="outside"))
                fig.update_layout(title="Perplexity (lower is better)", template="plotly_dark",
                                  height=320, yaxis_title="Perplexity", margin=dict(t=40, b=30))
                st.plotly_chart(fig, width="stretch")

        with chart_col4:
            lat_runs = [r for r in db_runs if r["Latency s"] is not None]
            if lat_runs:
                names = [r["Run"] for r in lat_runs]
                lats  = [r["Latency s"] for r in lat_runs]
                fig = go.Figure(go.Bar(x=names, y=lats, marker_color="#AB47BC",
                                       text=[f"{l:.2f}s" for l in lats], textposition="outside"))
                fig.update_layout(title="Avg Latency (lower is better)", template="plotly_dark",
                                  height=320, yaxis_title="Seconds", margin=dict(t=40, b=30))
                st.plotly_chart(fig, width="stretch")

        st.markdown("---")

        # ── Run Details ───────────────────────────────────────────────
        st.markdown("### Run Details")
        run_names = [r["Run"] for r in db_runs]
        sel_name = st.selectbox("Select Run", run_names, key="exp_sel_db")
        sel_db = next((r for r in db_runs if r["Run"] == sel_name), None)

        if sel_db:
            # Metric cards
            metric_defs = [
                ("Train Loss",  sel_db["Train Loss"],   "f4"),
                ("Eval Loss",   sel_db["Eval Loss"],    "f4"),
                ("Perplexity",  sel_db["Perplexity"],   "f2"),
                ("Val Acc %",   sel_db["Val Acc %"],    "pct"),
                ("Test Acc %",  sel_db["Test Acc %"],   "pct"),
                ("Latency s",   sel_db["Latency s"],    "s"),
                ("Tok/s",       sel_db["Tok/s"],        "f1"),
                ("TTFT s",      sel_db["TTFT s"],       "f2"),
                ("Duration",    sel_db["Duration s"],   "m"),
                ("Params",      sel_db["Params"],       "M"),
            ]
            st.markdown("#### Metrics")
            cols = st.columns(5)
            for i, (label, val, kind) in enumerate(metric_defs):
                cols[i % 5].metric(label, _fmt(val, kind))

            # Run config
            st.markdown("#### Config")
            cfg_cols = st.columns(4)
            cfg_cols[0].markdown(f"**Model:** `{sel_db['Base Model']}`")
            cfg_cols[1].markdown(f"**Quant:** `{sel_db['Quant']}`")
            cfg_cols[2].markdown(f"**LoRA r:** `{sel_db['LoRA r']}`")
            cfg_cols[3].markdown(f"**Epochs:** `{sel_db['Epochs']}`")
            if sel_db["Stacked On"]:
                st.markdown(f"**Stacked on:** `{sel_db['Stacked On']}`")

            # Dataset info
            if sel_db["Dataset"] and sel_db["Dataset"] != "—":
                st.markdown("#### Dataset")
                ds_cols = st.columns(4)
                ds_cols[0].markdown(f"**File:** `{sel_db['Dataset']}`")
                ds_cols[1].markdown(f"**Type:** `{sel_db.get('DS Type', '—')}`")
                if sel_db.get("DS Examples"):
                    ds_cols[2].markdown(f"**Examples:** `{int(sel_db['DS Examples'])}`")
                s3 = sel_db.get("DS S3", "")
                if s3:
                    ds_cols[3].markdown(f"**S3:** [{s3[:30]}...]({s3})")
                else:
                    _local = OUTPUTS_DIR / sel_name / "data"
                    ds_cols[3].markdown(f"**Local:** `outputs/{sel_name}/data/`")

            # Loss curve
            loss_log_path = OUTPUTS_DIR / sel_name / "loss_log.json"
            if loss_log_path.exists():
                st.markdown("#### Loss Curve")
                try:
                    ld = json.loads(loss_log_path.read_text())
                    tl = ld.get("train_losses", [])
                    el = ld.get("eval_losses", [])
                    if tl or el:
                        fig = go.Figure()
                        if tl:
                            fig.add_trace(go.Scatter(
                                x=[l["step"] for l in tl], y=[l["loss"] for l in tl],
                                mode="lines+markers", name="Train Loss",
                                line=dict(color="#4FC3F7", width=2)))
                        if el:
                            fig.add_trace(go.Scatter(
                                x=[l["step"] for l in el], y=[l["eval_loss"] for l in el],
                                mode="lines+markers", name="Eval Loss",
                                line=dict(color="#FF7043", width=2)))
                        if tl and len(tl) > 1:
                            fig.add_trace(go.Scatter(
                                x=[tl[0]["step"], tl[-1]["step"]],
                                y=[tl[0]["loss"], tl[-1]["loss"]],
                                mode="lines", name="Trend",
                                line=dict(color="#66BB6A", width=1, dash="dash")))
                        fig.update_layout(template="plotly_dark", height=350,
                                          xaxis_title="Step", yaxis_title="Loss",
                                          legend=dict(orientation="h", y=1.12),
                                          margin=dict(t=30, b=40))
                        st.plotly_chart(fig, width="stretch")
                        if tl and len(tl) > 1:
                            drop = (tl[0]["loss"] - tl[-1]["loss"]) / tl[0]["loss"] * 100
                            st.caption(f"Loss drop: {tl[0]['loss']:.4f} → {tl[-1]['loss']:.4f} "
                                       f"({drop:.1f}% reduction over {len(tl)} steps)")
                except Exception:
                    pass

            # Eval scenarios
            _db_m = get_model_by_experiment(sel_name)
            if _db_m:
                _evals = get_evals_for_model(_db_m["model_id"])
                if _evals:
                    st.markdown("#### Evaluation History")
                    for ev in _evals:
                        scenarios = json.loads(ev.get("scenario_results", "[]"))
                        with st.expander(f"{ev['test_file']}  |  {ev['overall_accuracy_pct']:.1f}%  |  {ev['created_at'][:16]}"):
                            ec1, ec2, ec3, ec4 = st.columns(4)
                            ec1.metric("Accuracy",  f"{ev['overall_accuracy_pct']:.1f}%")
                            ec2.metric("Latency",   f"{ev['avg_latency_s']:.2f}s")
                            ec3.metric("Tok/s",     f"{ev['avg_tokens_per_sec']:.1f}")
                            ec4.metric("TTFT",      f"{ev['ttft_s']:.3f}s")
                            if scenarios:
                                fig = px.bar(
                                    x=[s["name"] for s in scenarios],
                                    y=[s.get("accuracy_pct", 0) for s in scenarios],
                                    color=[s.get("accuracy_pct", 0) for s in scenarios],
                                    color_continuous_scale=["#EF5350", "#FFD54F", "#66BB6A"],
                                    range_color=[0, 100],
                                    labels={"x": "Scenario", "y": "Accuracy %"},
                                    title="Per-Scenario Accuracy",
                                )
                                fig.update_layout(template="plotly_dark", height=280, showlegend=False)
                                st.plotly_chart(fig, width="stretch")

        # ── MLflow section (detailed run history if available) ─────────
        if MLFLOW_AVAILABLE:
            with st.expander("MLflow Raw Run Data"):
                mlf_runs = get_mlflow_runs()
                if mlf_runs:
                    mlf_display = []
                    for run in mlf_runs:
                        fl  = run.get("metrics.final_train_loss")
                        el  = run.get("metrics.eval_loss") or run.get("metrics.best_eval_loss")
                        ac  = run.get("metrics.accuracy_pct") or run.get("metrics.overall_accuracy_pct")
                        ts  = run.get("metrics.training_time_s")
                        mlf_display.append({
                            "Run":        run.get("tags.mlflow.runName", "?"),
                            "Status":     run.get("status", "?"),
                            "Model":      run.get("params.model_name", "?"),
                            "Train Loss": f"{fl:.4f}" if fl is not None else "—",
                            "Eval Loss":  f"{el:.4f}" if el is not None else "—",
                            "Accuracy":   f"{ac:.1f}%" if ac is not None else "—",
                            "Duration":   f"{ts / 60:.0f}m" if ts is not None else "—",
                        })
                    st.dataframe(mlf_display, hide_index=True)
                else:
                    st.caption("No MLflow runs found.")


# ── Sidebar ──────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### MLOps Hub")

    adapters = find_adapters()
    st.markdown(f"**Trained Adapters:** {len(adapters)}")
    for a in adapters:
        loss_str = f" — loss {a['final_loss']:.4f}" if "final_loss" in a else ""
        st.caption(f"• {a['name']}{loss_str}")

    if MLFLOW_AVAILABLE:
        runs = get_mlflow_runs()
        st.markdown(f"**MLflow Runs:** {len(runs)}")

    st.markdown("---")
    st.markdown("### How It Works")
    st.markdown(
        "1. **Train** — upload data, pick model, fine-tune locally with QLoRA\n"
        "2. **Evaluate** — run test scenarios, see accuracy/latency\n"
        "3. **Chat** — load any adapter and talk to the model\n"
        "4. **Experiments** — browse all MLflow-tracked runs"
    )
    st.markdown("---")
    st.markdown("**Tracking:** MLflow (local `mlruns/`)")
    st.markdown("**Method:** QLoRA (4-bit NF4 + LoRA)")
    st.markdown("[HuggingFace Models](https://huggingface.co/models)")
