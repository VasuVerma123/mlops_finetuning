"""
Streamlit Echo Agent -- V4.7-AWS (98% accuracy)
3-panel layout: Calendar | System Prompt | Chat
"""

import streamlit as st
import json, re, time, gc, os, copy
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from echo_logger import EchoLogger

# =====================================================================
#  CONFIG
# =====================================================================

MODEL_NAME = "Qwen/Qwen3-4B"
V46_ADAPTER = "output/scheduling-agent-v4.6-aws/final_v4.6"
V47_ADAPTER = "output/scheduling-agent-v4.7-aws/final_v4.7"

INITIAL_CALENDAR = {
    "2026-03-24": {
        "day": "Tuesday",
        "slots": {
            "9:00 AM":  {"status": "booked", "client": "Sarah Kim", "service": "Individual Therapy"},
            "9:30 AM":  {"status": "booked", "client": "Sarah Kim", "service": "Individual Therapy"},
            "10:00 AM": {"status": "free"},
            "10:30 AM": {"status": "free"},
            "11:00 AM": {"status": "booked", "client": "James Rivera", "service": "Intake Session"},
            "11:30 AM": {"status": "free"},
            "12:00 PM": {"status": "free"},
            "12:30 PM": {"status": "break"},
            "1:00 PM":  {"status": "break"},
            "1:15 PM":  {"status": "free"},
            "1:30 PM":  {"status": "free"},
            "2:00 PM":  {"status": "booked", "client": "Anika Patel", "service": "Couples Therapy"},
            "2:30 PM":  {"status": "booked", "client": "Anika Patel", "service": "Couples Therapy"},
            "3:00 PM":  {"status": "free"},
            "3:30 PM":  {"status": "free"},
            "4:00 PM":  {"status": "booked", "client": "Tom Wilson", "service": "Individual Therapy"},
            "4:30 PM":  {"status": "free"},
            "5:00 PM":  {"status": "free"},
            "5:30 PM":  {"status": "free"},
        }
    },
    "2026-03-25": {
        "day": "Wednesday",
        "slots": {
            "9:00 AM":  {"status": "free"},
            "9:30 AM":  {"status": "free"},
            "10:00 AM": {"status": "booked", "client": "Elena Rodriguez", "service": "CBT Session"},
            "10:30 AM": {"status": "free"},
            "11:00 AM": {"status": "free"},
            "11:30 AM": {"status": "booked", "client": "Kevin Park", "service": "Follow-up Session"},
            "12:00 PM": {"status": "free"},
            "12:30 PM": {"status": "break"},
            "1:00 PM":  {"status": "break"},
            "1:15 PM":  {"status": "free"},
            "1:30 PM":  {"status": "free"},
            "2:00 PM":  {"status": "free"},
            "2:30 PM":  {"status": "free"},
            "3:00 PM":  {"status": "booked", "client": "Marcus Webb", "service": "EMDR Session"},
            "3:30 PM":  {"status": "free"},
            "4:00 PM":  {"status": "free"},
            "4:30 PM":  {"status": "free"},
            "5:00 PM":  {"status": "free"},
            "5:30 PM":  {"status": "free"},
        }
    },
    "2026-03-26": {
        "day": "Thursday",
        "slots": {
            "9:00 AM":  {"status": "free"},
            "9:30 AM":  {"status": "free"},
            "10:00 AM": {"status": "free"},
            "10:30 AM": {"status": "free"},
            "11:00 AM": {"status": "free"},
            "11:30 AM": {"status": "free"},
            "12:00 PM": {"status": "free"},
            "12:30 PM": {"status": "break"},
            "1:00 PM":  {"status": "break"},
            "1:15 PM":  {"status": "free"},
            "1:30 PM":  {"status": "free"},
            "2:00 PM":  {"status": "free"},
            "2:30 PM":  {"status": "free"},
            "3:00 PM":  {"status": "free"},
            "3:30 PM":  {"status": "free"},
            "4:00 PM":  {"status": "free"},
            "4:30 PM":  {"status": "free"},
            "5:00 PM":  {"status": "free"},
            "5:30 PM":  {"status": "free"},
        }
    },
}

SYSTEM_PROMPT = (
    "You are Echo, personal scheduling assistant for Dr. Maya Chen, a Licensed Therapist. "
    "You manage their calendar, communicate with clients on their behalf, and handle all scheduling tasks.\n\n"
    "Dr. Maya Chen uses she/her pronouns. Working hours: 9:00 AM - 6:00 PM. "
    "Break: 12:30 PM - 1:15 PM. Days off: Sunday.\n\n"
    "Cancellation policy: Appointments must be cancelled at least 24 hours in advance.\n\n"
    "Communication tone: Warm but professional.\n\n"
    "IMPORTANT: Never reveal other clients' names or session details. "
    "When a slot is taken, say it's unavailable without naming who booked it.\n\n"
    "# Tools\n\n<tools>\n"
    '{"type": "function", "function": {"name": "check_availability", "description": "Find free time slots for a given date", '
    '"parameters": {"type": "object", "properties": {"date": {"type": "string", "description": "Date to check (YYYY-MM-DD)"}}, "required": ["date"]}}}\n'
    '{"type": "function", "function": {"name": "book_appointment", "description": "Book a time slot for a client", '
    '"parameters": {"type": "object", "properties": {"date": {"type": "string"}, "time_slot": {"type": "string"}, '
    '"client_name": {"type": "string"}, "service": {"type": "string"}}, "required": ["date", "time_slot", "client_name", "service"]}}}\n'
    '{"type": "function", "function": {"name": "cancel_appointment", "description": "Cancel an existing appointment", '
    '"parameters": {"type": "object", "properties": {"date": {"type": "string"}, "time_slot": {"type": "string"}, '
    '"client_name": {"type": "string"}}, "required": ["date", "time_slot", "client_name"]}}}\n'
    '{"type": "function", "function": {"name": "send_message", "description": "Send a message to a person", '
    '"parameters": {"type": "object", "properties": {"recipient": {"type": "string"}, "message": {"type": "string"}}, '
    '"required": ["recipient", "message"]}}}\n'
    '{"type": "function", "function": {"name": "get_weather", "description": "Get weather forecast for a location", '
    '"parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}}}\n'
    '{"type": "function", "function": {"name": "reschedule_appointment", "description": "Reschedule an existing appointment to a new time", '
    '"parameters": {"type": "object", "properties": {"date": {"type": "string"}, "old_time_slot": {"type": "string"}, '
    '"new_time_slot": {"type": "string"}, "client_name": {"type": "string"}}, "required": ["date", "old_time_slot", "new_time_slot", "client_name"]}}}\n'
    "</tools>\n\n"
    "For each function call, return a json object with function name and arguments within "
    "<tool_call></tool_call> XML tags.\n\n"
    "Always think step-by-step before acting. Wrap your reasoning in <think></think> tags."
)

TOOLS_DISPLAY = [
    {"name": "check_availability", "desc": "Find free time slots for a given date", "params": "date"},
    {"name": "book_appointment", "desc": "Book a time slot for a client", "params": "date, time_slot, client_name, service"},
    {"name": "cancel_appointment", "desc": "Cancel an existing appointment", "params": "date, time_slot, client_name"},
    {"name": "reschedule_appointment", "desc": "Reschedule to a new time", "params": "date, old_time_slot, new_time_slot, client_name"},
    {"name": "send_message", "desc": "Send a message to a person", "params": "recipient, message"},
    {"name": "get_weather", "desc": "Get weather forecast (distractor)", "params": "location"},
]

# =====================================================================
#  TOOL FUNCTIONS
# =====================================================================

def norm_time(t, calendar):
    t = t.strip()
    for date_data in calendar.values():
        for k in date_data["slots"]:
            if t.replace(" ", "").lower() == k.replace(" ", "").lower():
                return k
    if "-" in t:
        return norm_time(t.split("-")[0], calendar)
    return t


def tool_check_availability(args, calendar):
    d = args.get("date", "")
    if d not in calendar:
        return {"status": "error", "message": f"No schedule data for {d}. Available: {list(calendar.keys())}"}
    free = [t for t, v in calendar[d]["slots"].items() if v["status"] == "free"]
    booked_count = sum(1 for v in calendar[d]["slots"].values() if v["status"] == "booked")
    return {"status": "success", "date": d, "day": calendar[d]["day"],
            "free_slots": free, "total_free": len(free), "total_booked": booked_count}


def tool_book_appointment(args, calendar):
    d, ts = args.get("date", ""), norm_time(args.get("time_slot", ""), calendar)
    client, service = args.get("client_name", ""), args.get("service", "")
    if d not in calendar:
        return {"status": "error", "message": f"No schedule for {d}"}
    slots = calendar[d]["slots"]
    if ts not in slots:
        return {"status": "error", "message": f"Slot '{ts}' not found. Available: {list(slots.keys())}"}
    if slots[ts]["status"] != "free":
        return {"status": "error", "message": f"{ts} is already {slots[ts]['status']}"}
    slots[ts] = {"status": "booked", "client": client, "service": service}
    return {"status": "success", "message": f"Booked {client} for {service} at {ts} on {d}"}


def tool_cancel_appointment(args, calendar):
    d, ts = args.get("date", ""), norm_time(args.get("time_slot", ""), calendar)
    if d not in calendar:
        return {"status": "error", "message": f"No schedule for {d}"}
    slots = calendar[d]["slots"]
    if ts not in slots or slots[ts]["status"] != "booked":
        return {"status": "error", "message": f"No booking at {ts} on {d}"}
    old = slots[ts].get("client", "?")
    slots[ts] = {"status": "free"}
    return {"status": "success", "message": f"Cancelled {old}'s appointment at {ts} on {d}. Slot is now free."}


def tool_reschedule_appointment(args, calendar):
    d = args.get("date", "")
    old_t = norm_time(args.get("old_time_slot", ""), calendar)
    new_t = norm_time(args.get("new_time_slot", ""), calendar)
    client = args.get("client_name", "")
    if d not in calendar:
        return {"status": "error", "message": f"No schedule for {d}"}
    slots = calendar[d]["slots"]
    if old_t not in slots or slots[old_t]["status"] != "booked":
        return {"status": "error", "message": f"No booking at {old_t}"}
    if new_t not in slots or slots[new_t]["status"] != "free":
        return {"status": "error", "message": f"{new_t} is not available"}
    svc = slots[old_t].get("service", "Session")
    c = slots[old_t].get("client", client)
    slots[old_t] = {"status": "free"}
    slots[new_t] = {"status": "booked", "client": c, "service": svc}
    return {"status": "success", "message": f"Rescheduled {c} from {old_t} to {new_t} on {d}"}


def tool_send_message(args, calendar):
    recipient = args.get("recipient", "")
    message = args.get("message", "")
    if "message_log" not in st.session_state:
        st.session_state.message_log = []
    st.session_state.message_log.append({"to": recipient, "message": message, "time": time.strftime("%H:%M:%S")})
    return {"status": "success", "message": f"Message sent to {recipient}"}


TOOL_FUNCS = {
    "check_availability": tool_check_availability,
    "book_appointment": tool_book_appointment,
    "cancel_appointment": tool_cancel_appointment,
    "reschedule_appointment": tool_reschedule_appointment,
    "send_message": tool_send_message,
}


# =====================================================================
#  HELPERS
# =====================================================================

def detect_date(text):
    text_l = text.lower()
    import re as _re
    day_match = _re.search(r'(?:march\s*)?(\d{1,2})(?:st|nd|rd|th)?(?:\s*march)?', text_l)
    if day_match:
        day = int(day_match.group(1))
        if day == 25:
            return "2026-03-25"
        if day == 26:
            return "2026-03-26"
        if day == 24:
            return "2026-03-24"
    if "wednesday" in text_l:
        return "2026-03-25"
    if "thursday" in text_l:
        return "2026-03-26"
    return "2026-03-24"


def render_schedule_for_model(date_key, calendar):
    if date_key not in calendar:
        return ""
    cal = calendar[date_key]
    lines = [f"[CURRENT STATE]\nSchedule for {date_key} ({cal['day']}):"]
    for t, info in cal["slots"].items():
        if info["status"] == "booked":
            lines.append(f"  {t}: BOOKED - {info['client']} ({info['service']})")
        elif info["status"] == "break":
            lines.append(f"  {t}: BREAK")
        else:
            lines.append(f"  {t}: FREE")
    return "\n".join(lines)


def extract_tool_call(resp):
    """
    Robustly extract a tool call from the model response.
    Handles all formats the model may produce:
      1. <tool_call>{"name": ..., "arguments": {...}}</tool_call>
      2. <tool_call>{"function": ..., "arguments": {...}}</tool_call>
      3. Bare JSON block: {"function": ..., "arguments": {...}}
      4. {"name": ..., "parameters": {...}}   (some fine-tuned variants)
    """
    KNOWN_TOOLS = {
        "check_availability", "book_appointment", "cancel_appointment",
        "reschedule_appointment", "send_message", "get_weather",
    }

    def _parse(blob: str):
        blob = blob.strip()
        try:
            d = json.loads(blob)
        except json.JSONDecodeError:
            return None, None
        name = d.get("name") or d.get("function") or d.get("tool") or ""
        args = d.get("arguments") or d.get("parameters") or d.get("args") or {}
        if name in KNOWN_TOOLS:
            return name, args
        return None, None

    # ── Format 1 & 2: inside <tool_call> tags ──
    m = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", resp, re.DOTALL)
    if m:
        name, args = _parse(m.group(1))
        if name:
            return name, args

    # ── Format 3 & 4: bare JSON block anywhere in the response ──
    # find every {...} block and test each one
    for m in re.finditer(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)?\}", resp, re.DOTALL):
        name, args = _parse(m.group(0))
        if name:
            return name, args

    return None, None


# =====================================================================
#  MODEL LOADING
# =====================================================================

@st.cache_resource(show_spinner=False)
def load_model():
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=bnb, device_map="auto",
        trust_remote_code=True, dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, padding_side="right")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if os.path.exists(V46_ADAPTER):
        model = PeftModel.from_pretrained(model, V46_ADAPTER)
        model = model.merge_and_unload()
        gc.collect()
        torch.cuda.empty_cache()

    if os.path.exists(V47_ADAPTER):
        model = PeftModel.from_pretrained(model, V47_ADAPTER)

    model.eval()
    gc.collect()
    torch.cuda.empty_cache()
    return model, tok


def generate(model, tok, messages):
    prompt = tok.apply_chat_template(messages, tokenize=False,
                                      add_generation_prompt=True, enable_thinking=True)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=800, temperature=0.3,
                             top_p=0.9, do_sample=True, repetition_penalty=1.1)
    resp = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    del inputs, out
    torch.cuda.empty_cache()
    return resp


# =====================================================================
#  STREAMLIT PAGE CONFIG
# =====================================================================

st.set_page_config(
    page_title="Echo Agent - V4.7",
    page_icon="📅",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    .stApp { background-color: #0f172a; }
    .main .block-container { padding-top: 1rem; max-width: 100%; }

    /* Calendar slot styles */
    .slot-free { background: #064e3b; color: #6ee7b7; padding: 4px 8px; border-radius: 4px;
        margin: 2px 0; font-size: 0.8em; font-family: monospace; }
    .slot-booked { background: #7f1d1d; color: #fca5a5; padding: 4px 8px; border-radius: 4px;
        margin: 2px 0; font-size: 0.8em; font-family: monospace; }
    .slot-break { background: #1e293b; color: #64748b; padding: 4px 8px; border-radius: 4px;
        margin: 2px 0; font-size: 0.8em; font-family: monospace; }

    .cal-header { background: linear-gradient(135deg, #1e3a5f, #1e293b); color: #93c5fd;
        padding: 8px 12px; border-radius: 8px 8px 0 0; font-weight: 700; font-size: 0.9em;
        text-align: center; border: 1px solid #334155; border-bottom: none; }
    .cal-body { background: #0f172a; border: 1px solid #334155; border-radius: 0 0 8px 8px;
        padding: 8px; }

    .tool-card { background: #1e293b; border: 1px solid #334155; border-radius: 8px;
        padding: 10px; margin: 6px 0; }
    .tool-name { color: #60a5fa; font-weight: 700; font-family: monospace; font-size: 0.85em; }
    .tool-desc { color: #94a3b8; font-size: 0.78em; }
    .tool-params { color: #64748b; font-size: 0.72em; font-family: monospace; }

    .info-box { background: #1e293b; border: 1px solid #334155; border-radius: 12px;
        padding: 16px; margin-bottom: 12px; }
    .info-label { color: #64748b; font-size: 0.75em; text-transform: uppercase;
        letter-spacing: 0.5px; margin-bottom: 4px; }
    .info-value { color: #e2e8f0; font-size: 0.95em; font-weight: 600; }

    .hero-badge { background: linear-gradient(135deg, #059669, #34d399); color: white;
        padding: 4px 14px; border-radius: 12px; font-size: 0.75em; font-weight: 700;
        display: inline-block; margin-bottom: 4px; }

    div[data-testid="stChatMessage"] { background: #1e293b; border-radius: 12px;
        border: 1px solid #334155; margin-bottom: 8px; }
</style>
""", unsafe_allow_html=True)


# =====================================================================
#  SESSION STATE INIT
# =====================================================================

if "calendar" not in st.session_state:
    st.session_state.calendar = copy.deepcopy(INITIAL_CALENDAR)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "message_log" not in st.session_state:
    st.session_state.message_log = []
if "logger" not in st.session_state:
    st.session_state.logger = EchoLogger()


# =====================================================================
#  LAYOUT
# =====================================================================

left_col, mid_col, right_col = st.columns([1, 1, 1.5], gap="medium")

# ----- LEFT: CALENDAR -----
with left_col:
    st.markdown('<div class="hero-badge">LIVE CALENDAR</div>', unsafe_allow_html=True)
    st.markdown("#### Dr. Maya Chen's Schedule")

    cal = st.session_state.calendar
    for date_key in cal:
        day_data = cal[date_key]
        day_label = date_key.split("-")[2]
        st.markdown(
            f'<div class="cal-header">📅 March {int(day_label)}, 2026 ({day_data["day"]})</div>',
            unsafe_allow_html=True,
        )

        slots_html = ""
        for t, info in day_data["slots"].items():
            if info["status"] == "booked":
                slots_html += f'<div class="slot-booked">🔴 {t} &nbsp; {info["client"]} ({info["service"]})</div>'
            elif info["status"] == "break":
                slots_html += f'<div class="slot-break">⏸ {t} &nbsp; BREAK</div>'
            else:
                slots_html += f'<div class="slot-free">🟢 {t} &nbsp; Available</div>'

        st.markdown(f'<div class="cal-body">{slots_html}</div>', unsafe_allow_html=True)
        st.markdown("")

    if st.button("🔄 Reset Calendar", use_container_width=True):
        st.session_state.logger.close()
        st.session_state.calendar = copy.deepcopy(INITIAL_CALENDAR)
        st.session_state.chat_history = []
        st.session_state.message_log = []
        st.session_state.logger = EchoLogger()
        st.rerun()


# ----- MIDDLE: SYSTEM PROMPT & CONFIG -----
with mid_col:
    st.markdown('<div class="hero-badge">MODEL & CONFIGURATION</div>', unsafe_allow_html=True)
    st.markdown("#### Echo Agent V4.7-AWS")

    st.markdown("""
    <div class="info-box">
        <div class="info-label">Model</div>
        <div class="info-value">Qwen3-4B + V4.6 (merge) + V4.7 LoRA</div>
    </div>
    <div class="info-box">
        <div class="info-label">Accuracy</div>
        <div class="info-value" style="color:#34d399; font-size:1.4em;">98% (47/48 checks)</div>
    </div>
    <div class="info-box">
        <div class="info-label">Inference</div>
        <div class="info-value">enable_thinking=True | temp=0.3</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("##### Dr. Chen's Rules")
    st.markdown("""
    <div class="info-box">
        <div style="color:#94a3b8; font-size:0.85em; line-height:1.7;">
        <strong style="color:#60a5fa;">Working Hours:</strong> 9:00 AM - 6:00 PM<br>
        <strong style="color:#60a5fa;">Break:</strong> 12:30 PM - 1:15 PM<br>
        <strong style="color:#60a5fa;">Days Off:</strong> Sunday<br>
        <strong style="color:#60a5fa;">Cancellation:</strong> 24 hours in advance<br>
        <strong style="color:#f87171;">Privacy:</strong> Never reveal client names/details
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("##### Available Tools")
    for tool in TOOLS_DISPLAY:
        emoji = "🔍" if "check" in tool["name"] else "📝" if "book" in tool["name"] else "❌" if "cancel" in tool["name"] else "🔄" if "reschedule" in tool["name"] else "💬" if "send" in tool["name"] else "🌤"
        st.markdown(f"""
        <div class="tool-card">
            <div class="tool-name">{emoji} {tool['name']}</div>
            <div class="tool-desc">{tool['desc']}</div>
            <div class="tool-params">params: {tool['params']}</div>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("📜 Full System Prompt", expanded=False):
        st.code(SYSTEM_PROMPT, language=None)

    # ── Session Log Viewer ──
    st.markdown("---")
    st.markdown("##### Session Log")
    logger = st.session_state.get("logger")
    if logger and logger.log_path.exists():
        log_lines = logger.log_path.read_text(encoding="utf-8").strip().splitlines()
        turns = [json.loads(l) for l in log_lines if '"event": "turn"' in l]
        st.caption(f"Log: `{logger.log_path.name}`  |  {len(turns)} turn(s)")
        with st.expander(f"📋 Turn History ({len(turns)})", expanded=False):
            for t in turns:
                outcome_color = "#34d399" if "success" in t["outcome"] else "#f87171" if "error" in t["outcome"] else "#94a3b8"
                st.markdown(
                    f"**T{t['turn']}** `{t['timestamp'][-8:]}` "
                    f"<span style='color:{outcome_color}'>{t['outcome']}</span> "
                    f"— {t['duration_ms']}ms",
                    unsafe_allow_html=True,
                )
                st.caption(f"User: {t['user_input'][:80]}")
                if t.get("tool_call"):
                    tc = t["tool_call"]
                    st.code(f"{tc['name']}({json.dumps(tc['args'])})", language="json")
                if t.get("tool_result"):
                    tr = t["tool_result"]
                    st.caption(f"Result: [{tr.get('status')}] {tr.get('message','')}")
                st.markdown("---")

        # Download buttons
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button(
                "⬇ Download JSONL",
                data=logger.log_path.read_bytes(),
                file_name=logger.log_path.name,
                mime="application/jsonl",
                use_container_width=True,
            )
        txt_path = logger.log_path.with_suffix(".txt")
        if txt_path.exists():
            with col_dl2:
                st.download_button(
                    "⬇ Download TXT",
                    data=txt_path.read_bytes(),
                    file_name=txt_path.name,
                    mime="text/plain",
                    use_container_width=True,
                )
    else:
        st.caption("No turns logged yet.")


# ----- RIGHT: CHAT -----
with right_col:
    st.markdown('<div class="hero-badge">CHAT WITH ECHO</div>', unsafe_allow_html=True)
    st.markdown("#### Talk as a client")

    chat_container = st.container(height=620)

    with chat_container:
        if not st.session_state.chat_history:
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(
                    "Hello! I'm **Echo**, Dr. Maya Chen's scheduling assistant. "
                    "I can help you book, cancel, or reschedule appointments. "
                    "What can I do for you today?"
                )

        for entry in st.session_state.chat_history:
            if entry["role"] == "user":
                with st.chat_message("user", avatar="👤"):
                    st.markdown(entry["content"])
            else:
                with st.chat_message("assistant", avatar="🤖"):
                    if entry.get("thinking"):
                        with st.expander("🧠 Echo's reasoning", expanded=False):
                            st.markdown(f"```\n{entry['thinking']}\n```")
                    if entry.get("tool_call"):
                        tc = entry["tool_call"]
                        st.info(f"**🔧 Tool Call:** `{tc['name']}`\n```json\n{json.dumps(tc['args'], indent=2)}\n```")
                    if entry.get("tool_result"):
                        tr = entry["tool_result"]
                        color = "success" if tr.get("status") == "success" else "error"
                        if color == "success":
                            st.success(f"**Result:** {tr.get('message', json.dumps(tr))}")
                        else:
                            st.error(f"**Result:** {tr.get('message', json.dumps(tr))}")
                    st.markdown(entry["content"])

    user_input = st.chat_input("Type your message to Echo...", key="chat_input")

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        logger = st.session_state.logger
        t_start = time.time()

        with st.spinner("Loading V4.7-AWS model (first time takes ~2 min)..."):
            model, tok = load_model()

        date_key = detect_date(user_input)
        sched = render_schedule_for_model(date_key, st.session_state.calendar)
        full_msg = f"{user_input}\n\n{sched}"

        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": full_msg}]

        try:
            with st.spinner("Echo is thinking..."):
                resp = generate(model, tok, msgs)
        except Exception as exc:
            logger.log_error("generate_first_pass", str(exc))
            st.error(f"Model error: {exc}")
            st.stop()

        think_match = re.search(r"<think>(.*?)</think>", resp, re.DOTALL)
        thinking_text = think_match.group(1).strip() if think_match else None

        tool_name, tool_args = extract_tool_call(resp)

        entry = {"role": "assistant", "content": "", "thinking": thinking_text,
                 "tool_call": None, "tool_result": None}

        result = None
        visible = ""

        if tool_name and tool_name in TOOL_FUNCS:
            entry["tool_call"] = {"name": tool_name, "args": tool_args}

            # Deep-copy → mutate → explicitly reassign so Streamlit detects the change
            _cal = copy.deepcopy(st.session_state.calendar)
            result = TOOL_FUNCS[tool_name](tool_args, _cal)
            st.session_state.calendar = _cal   # ← this triggers Streamlit to re-render calendar

            entry["tool_result"] = result

            # ── calendar-mutating tools: snapshot the updated state ──
            if tool_name in ("book_appointment", "cancel_appointment", "reschedule_appointment"):
                logger.log_calendar_state(copy.deepcopy(st.session_state.calendar))

            msgs.append({"role": "assistant", "content": resp})
            msgs.append({"role": "user", "content": f"<tool_response>\n{json.dumps(result)}\n</tool_response>"})

            try:
                with st.spinner("Echo is composing response..."):
                    resp2 = generate(model, tok, msgs)
            except Exception as exc:
                logger.log_error("generate_second_pass", str(exc))
                resp2 = "I've processed your request."

            visible = re.sub(r"<think>.*?</think>", "", resp2, flags=re.DOTALL).strip()
            entry["content"] = visible

        elif tool_name:
            entry["tool_call"] = {"name": tool_name, "args": tool_args}
            visible = re.sub(r"<think>.*?</think>", "", resp, flags=re.DOTALL).strip()
            entry["content"] = visible

        else:
            visible = re.sub(r"<think>.*?</think>", "", resp, flags=re.DOTALL).strip()
            entry["content"] = visible

        duration_ms = int((time.time() - t_start) * 1000)

        # ── write to log ──
        logger.log_turn(
            user_input=user_input,
            thinking=thinking_text,
            tool_name=tool_name,
            tool_args=tool_args,
            tool_result=result,
            response=visible,
            duration_ms=duration_ms,
        )

        st.session_state.chat_history.append(entry)
        st.rerun()
