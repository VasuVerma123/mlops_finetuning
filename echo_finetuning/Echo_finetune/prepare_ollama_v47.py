"""
Echo v4.7 → Ollama End-to-End Pipeline  (FIXED)
------------------------------------------------
FIX: Now correctly merges V4.6 first, then V4.7 on top
     (matches interactive_agent_v47.py stack — 98% accuracy)

Steps:
  1. Merge V4.6 adapter into Qwen3-4B base
  2. Merge V4.7 adapter on top of that
  3. Save the fully-merged model
  4. Convert to GGUF with llama.cpp
  5. Write Ollama Modelfile
  6. Create + push to Ollama Hub

Run:
    .venv\Scripts\python.exe prepare_ollama_v47.py
    .venv\Scripts\python.exe prepare_ollama_v47.py --skip-merge   # if merge already done
    .venv\Scripts\python.exe prepare_ollama_v47.py --push         # merge + push to Hub
"""

import os, sys, json, shutil, argparse, subprocess
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR      = Path(__file__).resolve().parent
V46_ADAPTER     = SCRIPT_DIR / "output" / "scheduling-agent-v4.6-aws" / "final_v4.6"
V47_ADAPTER     = SCRIPT_DIR / "output" / "scheduling-agent-v4.7-aws" / "final_v4.7"
MERGED_PATH     = SCRIPT_DIR / "output" / "echo-v4.7-merged"
GGUF_PATH       = SCRIPT_DIR / "output" / "echo_v4.7_q4.gguf"     # Q4_K_M ~ 2.5 GB
GGUF_Q8_PATH    = SCRIPT_DIR / "output" / "echo_v4.7_q8.gguf"     # Q8_0   ~ 4.3 GB (existing)
MODELFILE_PATH  = SCRIPT_DIR / "output" / "Modelfile_v4.7"
LLAMA_CPP_DIR   = SCRIPT_DIR / "llama.cpp"
BASE_MODEL      = "Qwen/Qwen3-4B"
OLLAMA_MODEL    = "echo-v4.7"
OLLAMA_HUB_NAME = "vasuverma/echo-v4.7"

# ── Full system prompt (matches interactive_agent_v47.py exactly) ──────────────
SYSTEM_PROMPT = """You are Echo, personal scheduling assistant for Dr. Maya Chen, a Licensed Therapist. You manage their calendar, communicate with clients on their behalf, and handle all scheduling tasks.

Dr. Maya Chen uses she/her pronouns. Working hours: 9:00 AM - 6:00 PM. Break: 12:30 PM - 1:15 PM. Days off: Sunday.

Cancellation policy: Appointments must be cancelled at least 24 hours in advance.

Communication tone: Warm but professional.

IMPORTANT: Never reveal other clients' names or session details. When a slot is taken, say it's unavailable without naming who booked it.

# Tools

<tools>
{"type": "function", "function": {"name": "check_availability", "description": "Find free time slots for a given date", "parameters": {"type": "object", "properties": {"date": {"type": "string", "description": "Date to check (YYYY-MM-DD)"}}, "required": ["date"]}}}
{"type": "function", "function": {"name": "book_appointment", "description": "Book a time slot for a client", "parameters": {"type": "object", "properties": {"date": {"type": "string"}, "time_slot": {"type": "string"}, "client_name": {"type": "string"}, "service": {"type": "string"}}, "required": ["date", "time_slot", "client_name", "service"]}}}
{"type": "function", "function": {"name": "cancel_appointment", "description": "Cancel an existing appointment", "parameters": {"type": "object", "properties": {"date": {"type": "string"}, "time_slot": {"type": "string"}, "client_name": {"type": "string"}}, "required": ["date", "time_slot", "client_name"]}}}
{"type": "function", "function": {"name": "reschedule_appointment", "description": "Reschedule an existing appointment to a new time", "parameters": {"type": "object", "properties": {"date": {"type": "string"}, "old_time_slot": {"type": "string"}, "new_time_slot": {"type": "string"}, "client_name": {"type": "string"}}, "required": ["date", "old_time_slot", "new_time_slot", "client_name"]}}}
{"type": "function", "function": {"name": "send_message", "description": "Send a message to a person", "parameters": {"type": "object", "properties": {"recipient": {"type": "string"}, "message": {"type": "string"}}, "required": ["recipient", "message"]}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags.

Always think step-by-step before acting. Wrap your reasoning in <think></think> tags."""


def banner(text: str):
    print("\n" + "=" * 65)
    print(f"  {text}")
    print("=" * 65)


# ── Step 1: Merge both adapters ────────────────────────────────────────────────
def merge_adapters(force: bool = False):
    banner("STEP 1 — Merging V4.6 + V4.7 adapters into base model")

    if MERGED_PATH.exists() and any(MERGED_PATH.iterdir()) and not force:
        print(f"\n[SKIP] Merged model already exists at:\n  {MERGED_PATH}")
        print("  Pass --force to re-merge.")
        return

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
    except ImportError as e:
        print(f"\n[ERROR] Missing dependency: {e}")
        print("  Install:  uv pip install transformers peft torch accelerate")
        sys.exit(1)

    if not V46_ADAPTER.exists():
        print(f"\n[ERROR] V4.6 adapter not found at:\n  {V46_ADAPTER}")
        sys.exit(1)
    if not V47_ADAPTER.exists():
        print(f"\n[ERROR] V4.7 adapter not found at:\n  {V47_ADAPTER}")
        sys.exit(1)

    print(f"\n  Base model : {BASE_MODEL}")
    print(f"  V4.6 adapter: {V46_ADAPTER}")
    print(f"  V4.7 adapter: {V47_ADAPTER}")
    print(f"  Output     : {MERGED_PATH}")
    print("\n  Loading base model in float16 on CPU (this takes ~3 min)...")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float16,
        device_map="cpu", trust_remote_code=True,
    )

    # ── Merge V4.6 first (CRITICAL — was missing before) ──
    print("\n  [1/2] Applying V4.6 adapter...")
    model = PeftModel.from_pretrained(base, str(V46_ADAPTER))
    print("  [1/2] Merging V4.6 weights (merge_and_unload)...")
    model = model.merge_and_unload()

    # ── Then merge V4.7 on top ──
    print("\n  [2/2] Applying V4.7 adapter on top...")
    model = PeftModel.from_pretrained(model, str(V47_ADAPTER))
    print("  [2/2] Merging V4.7 weights (merge_and_unload)...")
    model = model.merge_and_unload()

    print(f"\n  Saving fully-merged model to:\n  {MERGED_PATH}")
    MERGED_PATH.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(MERGED_PATH), safe_serialization=True)
    tokenizer.save_pretrained(str(MERGED_PATH))

    for src in [V47_ADAPTER / "chat_template.jinja", V47_ADAPTER / "special_tokens_map.json"]:
        if src.exists():
            shutil.copy(src, MERGED_PATH)
            print(f"  Copied {src.name}")

    size_gb = sum(f.stat().st_size for f in MERGED_PATH.rglob("*") if f.is_file()) / 1e9
    print(f"\n  [DONE] Merged model saved — {size_gb:.1f} GB")


# ── Step 2: Convert to GGUF ────────────────────────────────────────────────────
def convert_to_gguf(quant: str = "q8_0"):
    banner(f"STEP 2 — Converting to GGUF ({quant})")

    out_path = GGUF_Q8_PATH if quant == "q8_0" else GGUF_PATH
    conv_script = LLAMA_CPP_DIR / "convert_hf_to_gguf.py"

    if not conv_script.exists():
        print(f"\n[ERROR] llama.cpp not found at: {LLAMA_CPP_DIR}")
        print("  Run:  git clone https://github.com/ggerganov/llama.cpp --depth 1")
        return False

    if not MERGED_PATH.exists():
        print("\n[ERROR] Merged model not found. Run merge step first.")
        return False

    print(f"\n  Input  : {MERGED_PATH}")
    print(f"  Output : {out_path}")
    print(f"  Quant  : {quant}")
    print("\n  Running conversion (takes ~1 min)...")

    result = subprocess.run(
        [sys.executable, str(conv_script),
         str(MERGED_PATH),
         "--outfile", str(out_path),
         "--outtype", quant],
        capture_output=False,
    )
    if result.returncode != 0:
        print("\n[ERROR] GGUF conversion failed.")
        return False
    print(f"\n  [DONE] GGUF saved: {out_path}")
    return True


# ── Step 3: Write Modelfile ────────────────────────────────────────────────────
def write_modelfile(gguf_path: Path = None):
    banner("STEP 3 — Writing Ollama Modelfile")

    if gguf_path is None:
        # Prefer existing Q8 GGUF; fall back to Q4
        gguf_path = GGUF_Q8_PATH if GGUF_Q8_PATH.exists() else GGUF_PATH

    content = f"""# Echo v4.7 - Scheduling Assistant (Qwen3-4B + V4.6 + V4.7 LoRA)
# Stack: Qwen3-4B base -> V4.6 merged -> V4.7 merged
# Generated by prepare_ollama_v47.py (fixed)

FROM {gguf_path}

SYSTEM \"\"\"{SYSTEM_PROMPT}\"\"\"

PARAMETER temperature 0.3
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx 4096
PARAMETER repeat_penalty 1.1
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|im_start|>"
PARAMETER stop "</tool_call>"
"""
    MODELFILE_PATH.write_text(content, encoding="utf-8")
    print(f"\n  Modelfile written to:\n  {MODELFILE_PATH}")
    return MODELFILE_PATH


# ── Step 4: Create Ollama model locally ────────────────────────────────────────
def create_ollama_model():
    banner("STEP 4 — Registering with local Ollama")

    print(f"\n  Running: ollama create {OLLAMA_MODEL} -f {MODELFILE_PATH}")
    result = subprocess.run(
        ["ollama", "create", OLLAMA_MODEL, "-f", str(MODELFILE_PATH)],
        capture_output=False,
    )
    if result.returncode != 0:
        print("\n[ERROR] ollama create failed. Is Ollama running?")
        return False
    print(f"\n  [DONE] Model registered as: {OLLAMA_MODEL}")
    return True


# ── Step 5: Push to Ollama Hub ─────────────────────────────────────────────────
def push_to_hub():
    banner("STEP 5 — Pushing to Ollama Hub")

    print(f"\n  Copying {OLLAMA_MODEL} → {OLLAMA_HUB_NAME}")
    subprocess.run(["ollama", "cp", OLLAMA_MODEL, OLLAMA_HUB_NAME])

    print(f"\n  Pushing {OLLAMA_HUB_NAME} to Ollama Hub...")
    print("  (You need to be logged in: ollama login)")
    result = subprocess.run(["ollama", "push", OLLAMA_HUB_NAME], capture_output=False)
    if result.returncode != 0:
        print("\n[ERROR] Push failed. Make sure you're logged in to Ollama Hub.")
        return False

    print(f"""
  [DONE] Model live at: https://ollama.com/{OLLAMA_HUB_NAME.split('/')[0]}/echo-v4.7

  Friends can now run:
    ollama pull {OLLAMA_HUB_NAME}
    ollama run  {OLLAMA_HUB_NAME}

  Or use ollama_test.py — it auto-downloads on first run.
""")
    return True


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Echo v4.7 → Ollama pipeline")
    parser.add_argument("--skip-merge",  action="store_true", help="Skip merge (use existing merged model)")
    parser.add_argument("--skip-gguf",   action="store_true", help="Skip GGUF conversion (use existing GGUF)")
    parser.add_argument("--force",       action="store_true", help="Force re-merge even if merged model exists")
    parser.add_argument("--push",        action="store_true", help="Push to Ollama Hub after creating locally")
    parser.add_argument("--quant",       default="q8_0",      help="GGUF quant type: q8_0 (default) or q4_k_m")
    args = parser.parse_args()

    banner("Echo v4.7 → Ollama Pipeline  (V4.6 + V4.7 fixed)")
    print(f"  Base   : {BASE_MODEL}")
    print(f"  V4.6   : {V46_ADAPTER}")
    print(f"  V4.7   : {V47_ADAPTER}")
    print(f"  Merged : {MERGED_PATH}")
    print(f"  GGUF   : {GGUF_Q8_PATH if args.quant == 'q8_0' else GGUF_PATH}")

    if not args.skip_merge:
        merge_adapters(force=args.force)

    if not args.skip_gguf:
        if not GGUF_Q8_PATH.exists() or args.force:
            convert_to_gguf(quant=args.quant)
        else:
            print(f"\n[SKIP] GGUF already exists: {GGUF_Q8_PATH.name}  ({GGUF_Q8_PATH.stat().st_size/1e9:.1f} GB)")
            print("  Pass --force to reconvert.")

    write_modelfile()
    create_ollama_model()

    if args.push:
        push_to_hub()
    else:
        banner("NEXT STEP — Push to Ollama Hub")
        print(f"""
  Re-run with --push to publish:
    .venv\\Scripts\\python.exe prepare_ollama_v47.py --skip-merge --skip-gguf --push

  Or manually:
    ollama cp {OLLAMA_MODEL} {OLLAMA_HUB_NAME}
    ollama push {OLLAMA_HUB_NAME}

  After push, friends can install Echo with:
    ollama pull {OLLAMA_HUB_NAME}
    python ollama_test.py
""")

    print("=" * 65)
    print("  Pipeline complete.")
    print("=" * 65 + "\n")
