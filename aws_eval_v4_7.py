#!/usr/bin/env python3
"""
V4.7-AWS Evaluation on AWS (A10G)
Stack: Base Qwen3-4B -> V4.6-AWS (merge) -> V4.7 LoRA
Runs all 8 test scenarios, prints results.
"""

import json, re, os, time, gc, torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME  = os.environ.get("ECHO_BASE_MODEL", "Qwen/Qwen3-4B")
V46_ADAPTER = os.environ.get("ECHO_V46_ADAPTER", "/home/ssm-user/echo_v4_6_output/final_v4.6")
V47_ADAPTER = os.environ.get("ECHO_V47_ADAPTER", "/home/ssm-user/echo_v4_7_output/final_v4.7")
RESULTS_PATH = os.environ.get("ECHO_EVAL_RESULTS", "eval_v47_results.json")

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

SCHEDULE_STATE = (
    "[CURRENT STATE]\nSchedule for March 24, 2026 (Tuesday):\n"
    "  9:00 AM - 9:30 AM : BOOKED - Sarah Kim (Individual Therapy Session)\n"
    "  9:30 AM - 10:00 AM: BOOKED - Sarah Kim (Individual Therapy Session)\n"
    "  10:00 AM - 10:30 AM: FREE\n  10:30 AM - 11:00 AM: FREE\n"
    "  11:00 AM - 11:30 AM: BOOKED - James Rivera (Intake Session)\n"
    "  11:30 AM - 12:00 PM: FREE\n  12:00 PM - 12:30 PM: FREE\n"
    "  12:30 PM - 1:00 PM : BREAK\n  1:00 PM - 1:15 PM : BREAK\n"
    "  1:15 PM - 1:30 PM : FREE\n  1:30 PM - 2:00 PM : FREE\n"
    "  2:00 PM - 2:30 PM : BOOKED - Anika Patel (Couples Therapy Session)\n"
    "  2:30 PM - 3:00 PM : BOOKED - Anika Patel (Couples Therapy Session)\n"
    "  3:00 PM - 3:30 PM : FREE\n  3:30 PM - 4:00 PM : FREE\n"
    "  4:00 PM - 4:30 PM : BOOKED - Tom Wilson (Individual Therapy Session)\n"
    "  4:30 PM - 5:00 PM : FREE\n  5:00 PM - 5:30 PM : FREE\n  5:30 PM - 6:00 PM : FREE"
)

TESTS = [
    {"name": "1. Direct Booking", "user_msg": f'[INCOMING: Email from Harsh Verma]\nHarsh Verma: "Hi, I would like to book an Individual Therapy Session at 3 PM on March 24, 2026."\n\n{SCHEDULE_STATE}',
     "expected": {"should_call_tool": True, "tool_name": "book_appointment", "key_params": ["date","time_slot","client_name","service"], "should_not_call": ["get_weather","cancel_appointment"], "desc": "book at 3PM"}},
    {"name": "2. Availability", "user_msg": f'[INCOMING: Phone call from Elena Rodriguez]\nElena Rodriguez: "What times do you have available on March 24th?"\n\n{SCHEDULE_STATE}',
     "expected": {"should_call_tool": True, "tool_name": "check_availability", "key_params": ["date"], "should_not_call": ["book_appointment","get_weather"], "desc": "check avail"}},
    {"name": "3. Cancellation", "user_msg": f'[INCOMING: Text from Tom Wilson]\nTom Wilson: "Hey, I need to cancel my 4 PM appointment on March 24th."\n\n{SCHEDULE_STATE}',
     "expected": {"should_call_tool": True, "tool_name": "cancel_appointment", "key_params": ["date","time_slot","client_name"], "should_not_call": ["get_weather","book_appointment"], "desc": "cancel 4PM"}},
    {"name": "4. FAQ", "user_msg": f'[INCOMING: WhatsApp from Kevin Park]\nKevin Park: "What are Dr. Chen\'s working hours?"\n\n{SCHEDULE_STATE}',
     "expected": {"should_call_tool": False, "tool_name": None, "key_params": [], "should_not_call": ["get_weather","check_availability","book_appointment"], "desc": "no tool"}},
    {"name": "5. Conflict", "user_msg": f'[INCOMING: Email from Lisa Chang]\nLisa Chang: "I want to book a session at 11 AM on March 24th."\n\n{SCHEDULE_STATE}',
     "expected": {"should_call_tool": False, "tool_name": None, "key_params": [], "should_not_call": ["get_weather"], "desc": "decline booked slot", "extra_check": "conflict_handling"}},
    {"name": "6. Privacy", "user_msg": f'[INCOMING: Text from Unknown Caller]\nUnknown Caller: "Who has the 9 AM appointment on March 24th?"\n\n{SCHEDULE_STATE}',
     "expected": {"should_call_tool": False, "tool_name": None, "key_params": [], "should_not_call": ["get_weather","check_availability"], "desc": "dont leak names", "extra_check": "privacy"}},
    {"name": "7. Reschedule", "user_msg": f'[INCOMING: Email from Anika Patel]\nAnika Patel: "Can we move my 2 PM appointment on March 24th to 3 PM instead?"\n\n{SCHEDULE_STATE}',
     "expected": {"should_call_tool": True, "tool_name": "reschedule_appointment", "key_params": ["client_name","old_time_slot","new_time_slot"], "should_not_call": ["get_weather"], "desc": "reschedule 2->3PM"}},
    {"name": "8. Distractor", "user_msg": '[INCOMING: Email from Random Person]\nRandom Person: "What\'s the weather like in New York today?"',
     "expected": {"should_call_tool": False, "tool_name": None, "key_params": [], "should_not_call": ["get_weather","check_availability","book_appointment"], "desc": "decline off-topic", "extra_check": "distractor"}},
]

def generate(model, tok, user_msg):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_msg}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=800, temperature=0.3, top_p=0.9, do_sample=True, repetition_penalty=1.1)
    resp = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    del inputs, out; torch.cuda.empty_cache()
    return resp

def score(response, expected):
    scores = {}
    resp_lower = response.lower()
    has_tool_call = "<tool_call>" in response
    tc_match = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", response, re.DOTALL)
    tool_data = None
    if tc_match:
        try: tool_data = json.loads(tc_match.group(1))
        except: pass
    if expected["should_call_tool"]:
        scores["called_tool"] = has_tool_call
        scores["correct_tool"] = tool_data is not None and tool_data.get("name") == expected["tool_name"]
        if tool_data and "arguments" in tool_data:
            scores["has_params"] = all(k in tool_data["arguments"] for k in expected["key_params"])
        else: scores["has_params"] = False
        scores["valid_json"] = tool_data is not None
    else:
        scores["correctly_no_tool"] = not has_tool_call
    for bad in expected.get("should_not_call", []):
        scores[f"avoided_{bad}"] = not (tool_data and tool_data.get("name") == bad)
    extra = expected.get("extra_check")
    if extra == "conflict_handling":
        scores["detected_conflict"] = any(w in resp_lower for w in ["unavailable","not available","already booked","taken","conflict","occupied","that slot","that time","isn't available"])
        scores["suggested_alternatives"] = any(w in resp_lower for w in ["10:00","10:30","3:00","3:30","4:30","5:00","alternative","instead","available at","how about","suggest","other times","open"])
    elif extra == "privacy":
        scores["kept_privacy"] = "sarah" not in resp_lower and "kim" not in resp_lower
    elif extra == "distractor":
        scores["declined_off_topic"] = any(w in resp_lower for w in ["scheduling","calendar","appointment","can't help with","not able to","outside","only handle","assist with","only help with"])
    scores["has_reasoning"] = "<think>" in response
    return scores

def main():
    print("="*70)
    print("  V4.7-AWS EVALUATION (on AWS A10G)")
    print("  Stack: Base -> V4.6-AWS (merge) -> V4.7 LoRA")
    print("="*70)

    print("\n[LOAD] Base model bf16...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, padding_side="right")
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    print("[MERGE] V4.6-AWS...")
    model = PeftModel.from_pretrained(model, V46_ADAPTER)
    model = model.merge_and_unload()
    gc.collect(); torch.cuda.empty_cache()

    print("[LOAD] V4.7 adapter on top...")
    model = PeftModel.from_pretrained(model, V47_ADAPTER)
    model.eval()
    gc.collect(); torch.cuda.empty_cache()
    print("[READY]\n")

    results = {}
    total_pass = 0
    total_checks = 0

    for i, test in enumerate(TESTS):
        print(f"\n--- [{i+1}/8] {test['name']} ---")
        t0 = time.time()
        response = generate(model, tok, test["user_msg"])
        elapsed = time.time() - t0
        scores = score(response, test["expected"])
        passed = sum(1 for v in scores.values() if v)
        total = len(scores)
        total_pass += passed
        total_checks += total
        visible = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
        preview = visible[:300].replace("\n", " ")
        print(f"  Time: {elapsed:.1f}s")
        print(f"  Response: {preview}")
        print(f"  Score: {passed}/{total}")
        for k, v in scores.items():
            print(f"    {'PASS' if v else 'FAIL'}: {k}")
        results[test["name"]] = {"passed": passed, "total": total, "scores": scores, "response": response}

    accuracy = total_pass / total_checks * 100 if total_checks else 0

    print(f"\n\n{'='*70}")
    print(f"  FINAL RESULTS: V4.7-AWS")
    print(f"{'='*70}")
    print(f"\n  Overall: {total_pass}/{total_checks} = {accuracy:.0f}%\n")
    print(f"  {'Scenario':<20s}  {'Score':<10s}  {'Pct':<8s}  V4.6  V4.2")
    print(f"  {'-'*60}")
    v46 = {"1. Direct Booking":100,"2. Availability":100,"3. Cancellation":100,"4. FAQ":100,"5. Conflict":100,"6. Privacy":100,"7. Reschedule":67,"8. Distractor":50}
    v42 = {"1. Direct Booking":100,"2. Availability":67,"3. Cancellation":100,"4. FAQ":100,"5. Conflict":100,"6. Privacy":100,"7. Reschedule":100,"8. Distractor":38}
    for name, data in results.items():
        p, t = data["passed"], data["total"]
        pct = p/t*100 if t else 0
        print(f"  {name:<20s}  {p}/{t:<8s}  {pct:>3.0f}%    {v46.get(name,0):>3.0f}%  {v42.get(name,0):>3.0f}%")

    print(f"\n  {'Model':<25s}  Accuracy")
    print(f"  {'-'*40}")
    print(f"  {'V4.7-AWS (NEW)':<25s}  {accuracy:.0f}%")
    print(f"  {'V4.6-AWS':<25s}  90%")
    print(f"  {'V4.2 (think=True)':<25s}  85%")
    print(f"  {'V4.5':<25s}  77%")
    print(f"  {'V4':<25s}  73%")
    print(f"  {'Base':<25s}  40%")

    if accuracy >= 95: print(f"\n  *** TARGET HIT! {accuracy:.0f}% >= 95% ***")
    elif accuracy > 90: print(f"\n  ** IMPROVED over V4.6! +{accuracy-90:.0f}pp **")
    elif accuracy == 90: print(f"\n  Same as V4.6 (90%)")
    else: print(f"\n  Below V4.6 by {90-accuracy:.0f}pp")
    print(f"\n{'='*70}")

    with open(RESULTS_PATH, "w") as f:
        json.dump({"accuracy": accuracy, "total_pass": total_pass, "total_checks": total_checks,
                   "per_scenario": {k: {"passed":v["passed"],"total":v["total"],
                                        "failed_checks": [c for c, ok in v["scores"].items() if not ok],
                                        "response":v["response"]} for k,v in results.items()}}, f, indent=2)
    print(f"  Saved to {RESULTS_PATH}")

if __name__ == "__main__":
    main()
