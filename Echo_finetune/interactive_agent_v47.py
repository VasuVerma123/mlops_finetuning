"""
INTERACTIVE AGENT - Chat with Echo (V4.7-AWS, 98% accuracy)
=============================================================
Type messages in natural language. The model will:
  - Think step-by-step (shown in console)
  - Call real tools against a live calendar
  - Get tool results and respond naturally

Type 'calendar' to see current calendar state
Type 'calendar 25' to see March 25 calendar
Type 'messages' to see sent messages log
Type 'quit' to exit

Stack: Base Qwen3-4B -> V4.6-AWS (merge) -> V4.7 LoRA
Run:   py -3 interactive_agent_v47.py
"""

import json, re, time, gc, os
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_NAME = "Qwen/Qwen3-4B"
V46_ADAPTER = "output/scheduling-agent-v4.6-aws/final_v4.6"
V47_ADAPTER = "output/scheduling-agent-v4.7-aws/final_v4.7"

CALENDAR = {
    "2026-03-24": {
        "day": "Tuesday",
        "slots": {
            "9:00 AM":  {"status": "booked", "client": "Sarah Kim", "service": "Individual Therapy Session"},
            "9:30 AM":  {"status": "booked", "client": "Sarah Kim", "service": "Individual Therapy Session"},
            "10:00 AM": {"status": "free"},
            "10:30 AM": {"status": "free"},
            "11:00 AM": {"status": "booked", "client": "James Rivera", "service": "Intake Session"},
            "11:30 AM": {"status": "free"},
            "12:00 PM": {"status": "free"},
            "12:30 PM": {"status": "break"},
            "1:00 PM":  {"status": "break"},
            "1:15 PM":  {"status": "free"},
            "1:30 PM":  {"status": "free"},
            "2:00 PM":  {"status": "booked", "client": "Anika Patel", "service": "Couples Therapy Session"},
            "2:30 PM":  {"status": "booked", "client": "Anika Patel", "service": "Couples Therapy Session"},
            "3:00 PM":  {"status": "free"},
            "3:30 PM":  {"status": "free"},
            "4:00 PM":  {"status": "booked", "client": "Tom Wilson", "service": "Individual Therapy Session"},
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

MESSAGE_LOG = []


def norm_time(t):
    t = t.strip()
    for date_data in CALENDAR.values():
        for k in date_data["slots"]:
            if t.replace(" ", "").lower() == k.replace(" ", "").lower():
                return k
    if "-" in t:
        return norm_time(t.split("-")[0])
    return t


def tool_check_availability(args):
    d = args.get("date", "")
    if d not in CALENDAR:
        return {"status": "error", "message": f"No schedule data for {d}. Available dates: {list(CALENDAR.keys())}"}
    free = [t for t, v in CALENDAR[d]["slots"].items() if v["status"] == "free"]
    booked_count = sum(1 for v in CALENDAR[d]["slots"].values() if v["status"] == "booked")
    return {"status": "success", "date": d, "day": CALENDAR[d]["day"],
            "free_slots": free, "total_free": len(free), "total_booked": booked_count}


def tool_book_appointment(args):
    d, ts = args.get("date", ""), norm_time(args.get("time_slot", ""))
    client = args.get("client_name", "")
    service = args.get("service", "")
    if d not in CALENDAR:
        return {"status": "error", "message": f"No schedule for {d}"}
    slots = CALENDAR[d]["slots"]
    if ts not in slots:
        return {"status": "error", "message": f"Slot '{ts}' not found. Available: {list(slots.keys())}"}
    if slots[ts]["status"] != "free":
        return {"status": "error", "message": f"{ts} is already {slots[ts]['status']}"}
    slots[ts] = {"status": "booked", "client": client, "service": service}
    return {"status": "success", "message": f"Booked {client} for {service} at {ts} on {d}"}


def tool_cancel_appointment(args):
    d, ts = args.get("date", ""), norm_time(args.get("time_slot", ""))
    if d not in CALENDAR:
        return {"status": "error", "message": f"No schedule for {d}"}
    slots = CALENDAR[d]["slots"]
    if ts not in slots or slots[ts]["status"] != "booked":
        return {"status": "error", "message": f"No booking at {ts} on {d}"}
    old = slots[ts].get("client", "?")
    slots[ts] = {"status": "free"}
    return {"status": "success", "message": f"Cancelled {old}'s appointment at {ts} on {d}. Slot is now free."}


def tool_reschedule_appointment(args):
    d = args.get("date", "")
    old_t, new_t = norm_time(args.get("old_time_slot", "")), norm_time(args.get("new_time_slot", ""))
    client = args.get("client_name", "")
    if d not in CALENDAR:
        return {"status": "error", "message": f"No schedule for {d}"}
    slots = CALENDAR[d]["slots"]
    if old_t not in slots or slots[old_t]["status"] != "booked":
        return {"status": "error", "message": f"No booking at {old_t}"}
    if new_t not in slots or slots[new_t]["status"] != "free":
        return {"status": "error", "message": f"{new_t} is not available"}
    svc = slots[old_t].get("service", "Session")
    c = slots[old_t].get("client", client)
    slots[old_t] = {"status": "free"}
    slots[new_t] = {"status": "booked", "client": c, "service": svc}
    return {"status": "success", "message": f"Rescheduled {c} from {old_t} to {new_t} on {d}"}


def tool_send_message(args):
    recipient = args.get("recipient", "")
    message = args.get("message", "")
    MESSAGE_LOG.append({"to": recipient, "message": message, "time": time.strftime("%H:%M:%S")})
    return {"status": "success", "message": f"Message sent to {recipient}"}


TOOLS = {
    "check_availability": tool_check_availability,
    "book_appointment": tool_book_appointment,
    "cancel_appointment": tool_cancel_appointment,
    "reschedule_appointment": tool_reschedule_appointment,
    "send_message": tool_send_message,
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


def print_calendar(date_key=None):
    dates = [date_key] if date_key else list(CALENDAR.keys())
    for dk in dates:
        cal = CALENDAR[dk]
        print(f"\n  Schedule: {dk} ({cal['day']})")
        print(f"  {'-'*45}")
        for t, info in cal["slots"].items():
            if info["status"] == "booked":
                print(f"    {t:>8s}  BOOKED  {info['client']} ({info['service']})")
            elif info["status"] == "break":
                print(f"    {t:>8s}  BREAK")
            else:
                print(f"    {t:>8s}  FREE")
    print()


def render_schedule_for_model(date_key):
    if date_key not in CALENDAR:
        return ""
    cal = CALENDAR[date_key]
    lines = [f"[CURRENT STATE]\nSchedule for {date_key} ({cal['day']}):"]
    for t, info in cal["slots"].items():
        if info["status"] == "booked":
            lines.append(f"  {t}: BOOKED - {info['client']} ({info['service']})")
        elif info["status"] == "break":
            lines.append(f"  {t}: BREAK")
        else:
            lines.append(f"  {t}: FREE")
    return "\n".join(lines)


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


def load_model():
    print("\n  Loading base model (4-bit quantized)...")
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.bfloat16,
                              bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=bnb, device_map="auto",
        trust_remote_code=True, dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, padding_side="right")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if os.path.exists(V46_ADAPTER):
        print("  Merging V4.6-AWS adapter...")
        model = PeftModel.from_pretrained(model, V46_ADAPTER)
        model = model.merge_and_unload()
        gc.collect()
        torch.cuda.empty_cache()
    else:
        print(f"  [WARN] V4.6 adapter not found: {V46_ADAPTER}")

    if os.path.exists(V47_ADAPTER):
        print("  Loading V4.7-AWS adapter on top...")
        model = PeftModel.from_pretrained(model, V47_ADAPTER)
    else:
        print(f"  [WARN] V4.7 adapter not found: {V47_ADAPTER}")

    model.eval()
    gc.collect()
    torch.cuda.empty_cache()
    print("  Model ready! (V4.7-AWS, 98% accuracy)\n")
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


def extract_tool_call(resp):
    m = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", resp, re.DOTALL)
    if m:
        try:
            d = json.loads(m.group(1))
            return d.get("name") or d.get("function", ""), d.get("arguments", {})
        except json.JSONDecodeError:
            pass
    return None, None


def chat_turn(model, tok, user_input):
    date_key = detect_date(user_input)
    sched = render_schedule_for_model(date_key)
    full_msg = f"{user_input}\n\n{sched}"

    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": full_msg}]

    print("\n  Echo is thinking...")
    t0 = time.time()
    resp = generate(model, tok, msgs)

    think = re.search(r"<think>(.*?)</think>", resp, re.DOTALL)
    if think:
        print(f"\n  [THINKING]")
        for line in think.group(1).strip().split("\n"):
            print(f"    {line}")

    tool_name, tool_args = extract_tool_call(resp)

    if tool_name and tool_name in TOOLS:
        print(f"\n  [TOOL CALL] {tool_name}")
        print(f"  [ARGS] {json.dumps(tool_args, indent=2)}")

        result = TOOLS[tool_name](tool_args)
        print(f"  [RESULT] {json.dumps(result, indent=2)}")

        msgs.append({"role": "assistant", "content": resp})
        msgs.append({"role": "user", "content": f"<tool_response>\n{json.dumps(result)}\n</tool_response>"})

        print("\n  Echo is composing final response...")
        resp2 = generate(model, tok, msgs)
        visible = re.sub(r"<think>.*?</think>", "", resp2, flags=re.DOTALL).strip()
        elapsed = time.time() - t0

        print(f"\n  {'='*55}")
        print(f"  ECHO ({elapsed:.1f}s):")
        print(f"  {'='*55}")
        for line in visible.split("\n"):
            print(f"  {line}")
        print()

    elif tool_name:
        visible = re.sub(r"<think>.*?</think>", "", resp, flags=re.DOTALL).strip()
        elapsed = time.time() - t0
        print(f"\n  [TOOL CALL] {tool_name} (not in environment)")
        print(f"\n  {'='*55}")
        print(f"  ECHO ({elapsed:.1f}s):")
        print(f"  {'='*55}")
        for line in visible.split("\n"):
            print(f"  {line}")
        print()

    else:
        visible = re.sub(r"<think>.*?</think>", "", resp, flags=re.DOTALL).strip()
        elapsed = time.time() - t0
        print(f"\n  {'='*55}")
        print(f"  ECHO ({elapsed:.1f}s):")
        print(f"  {'='*55}")
        for line in visible.split("\n"):
            print(f"  {line}")
        print()


def main():
    print("=" * 60)
    print("  INTERACTIVE AGENT - Chat with Echo")
    print("  Model: V4.7-AWS (98% accuracy)")
    print("  Stack: Base -> V4.6 (merge) -> V4.7 LoRA")
    print("  Real tools execute against a live calendar")
    print("=" * 60)
    print()
    print("  Available dates: March 24 (Tue), 25 (Wed), 26 (Thu)")
    print("  Commands:")
    print("    'calendar'    - Show current calendar")
    print("    'calendar 25' - Show March 25 calendar")
    print("    'messages'    - Show sent messages log")
    print("    'quit'        - Exit")
    print()
    print("  Try messages like:")
    print('    "Hi, I\'m Raj. Book me a CBT session at 3 PM on March 24"')
    print('    "What times are free on March 25?"')
    print('    "Move Anika Patel\'s 2 PM to 3 PM on March 24"')
    print('    "Cancel Tom Wilson\'s 4 PM appointment"')
    print('    "Who has the 9 AM slot on Tuesday?"')
    print('    "What\'s the weather in NYC?"')
    print()

    model, tok = load_model()

    while True:
        try:
            user_input = input("  YOU > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() == "quit":
            print("  Goodbye!")
            break

        if user_input.lower().startswith("calendar"):
            parts = user_input.split()
            if len(parts) > 1:
                day = parts[1]
                key = f"2026-03-{int(day):02d}" if day.isdigit() else None
                if key and key in CALENDAR:
                    print_calendar(key)
                else:
                    print(f"  Unknown date. Available: {list(CALENDAR.keys())}")
            else:
                print_calendar()
            continue

        if user_input.lower() == "messages":
            if MESSAGE_LOG:
                print("\n  Sent Messages:")
                for m in MESSAGE_LOG:
                    print(f"    [{m['time']}] To: {m['to']} - {m['message']}")
            else:
                print("  No messages sent yet.")
            print()
            continue

        chat_turn(model, tok, user_input)


if __name__ == "__main__":
    main()
