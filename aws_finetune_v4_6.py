#!/usr/bin/env python3
"""
V4.6 AWS Fine-tuning Script (Self-contained)
=============================================
Runs on g5.2xlarge (A10G 24GB VRAM, 32GB RAM).

This is a SELF-CONTAINED script: generates dataset + runs training.
No need to upload separate files.

Stack: Qwen3-4B -> V4.6 LoRA (fresh, no prior adapters on AWS)
  - Since prior adapters (V4, V4.1, V4.2) are only on the local laptop,
    we train V4.6 directly on the base model with the V4.6 dataset.
  - The dataset teaches all the behaviors we need.

Optimized for A10G:
  - Larger batch size (2 instead of 1)
  - Larger LoRA rank (32 for better quality)
  - Full bf16 (A10G has great bf16 support)
  - No 4-bit quantization needed (enough VRAM)

Usage: python3 aws_finetune_v4_6.py
"""

import json
import os
import random
import time
import gc
from datetime import datetime, timedelta

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import SFTTrainer, SFTConfig

random.seed(42)

# =====================================================================
#  DATASET GENERATION (embedded — same as build_v4_6_dataset.py)
# =====================================================================

FIRST_NAMES = [
    "Alex", "Maria", "David", "Sarah", "James", "Lisa", "Tom", "Anika",
    "Emma", "Noah", "Zara", "Marcus", "Yuki", "Jordan", "Priya", "Carlos",
    "Fatima", "Michael", "Lucia", "Robert", "Mei", "Andre", "Sophia",
    "Raj", "Elena", "Kevin", "Hana", "Omar", "Chloe", "Victor",
    "Aisha", "Daniel", "Nadia", "Leo", "Jasmine", "Hassan", "Grace",
    "Dmitri", "Rosa", "Ian", "Lena", "Felix", "Maya", "Kai", "Nina",
    "Oscar", "Suki", "Theo", "Amara", "Ravi",
]
LAST_NAMES = [
    "Chen", "Kim", "Rivera", "Patel", "Wilson", "Turner", "Santos",
    "Park", "Rodriguez", "Lee", "Nakamura", "O'Brien", "Al-Rashid",
    "Huang", "Garcia", "Baker", "Webb", "Chang", "Sullivan", "Ahmed",
    "Tanaka", "Miller", "Grey", "Quinn", "Torres", "Adams", "Cooper",
    "Shah", "Kelly", "Moreau", "Berg", "Okafor", "Petrov", "Singh",
]
SERVICES = [
    "Individual Therapy Session", "CBT Session", "Intake Session",
    "Couples Therapy", "Family Therapy", "Follow-up Session",
    "EMDR Session", "Group Therapy Session",
]
CHANNELS = ["Email", "Phone call", "Text message", "WhatsApp"]
DATES = [
    ("March 23, 2026", "Monday"), ("March 24, 2026", "Tuesday"),
    ("March 25, 2026", "Wednesday"), ("March 26, 2026", "Thursday"),
    ("March 27, 2026", "Friday"), ("March 28, 2026", "Saturday"),
]
TIME_SLOTS = [
    "9:00 AM", "9:30 AM", "10:00 AM", "10:30 AM", "11:00 AM", "11:30 AM",
    "12:00 PM", "1:15 PM", "1:30 PM", "2:00 PM", "2:30 PM",
    "3:00 PM", "3:30 PM", "4:00 PM", "4:30 PM", "5:00 PM", "5:30 PM",
]
BREAK_SLOTS = ["12:30 PM", "1:00 PM"]

used_names = set()
def get_name():
    for _ in range(500):
        n = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
        if n not in used_names:
            used_names.add(n)
            return n
    return f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"


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


def make_schedule(num_booked=3):
    available = list(TIME_SLOTS)
    random.shuffle(available)
    booked_times = available[:num_booked]
    free_times = [t for t in TIME_SLOTS if t not in booked_times]
    date_str, day_str = random.choice(DATES)
    bookings = {}
    for t in booked_times:
        bookings[t] = (get_name(), random.choice(SERVICES))
    lines = [f"[CURRENT STATE]\nSchedule for {date_str} ({day_str}):"]
    for t in TIME_SLOTS:
        if t in bookings:
            name, svc = bookings[t]
            lines.append(f"  {t}: BOOKED - {name} ({svc})")
        elif t in BREAK_SLOTS:
            lines.append(f"  {t}: BREAK")
        else:
            lines.append(f"  {t}: FREE")
    return "\n".join(lines), bookings, free_times, date_str


def date_to_iso(date_str):
    day_num = date_str.split()[1].rstrip(",")
    return f"2026-03-{int(day_num):02d}"


def make_availability_examples(count=40):
    examples = []
    phrases = [
        "What times do you have available on {date}?",
        "I'm looking for a free slot on {date}.",
        "Can I see what's open on {date}?",
        "Any openings on {date}?",
        "What times can I book on {date}?",
        "Do you have any availability on {date}?",
        "I'd like to know what slots are free on {date}.",
        "When can I come in on {date}?",
        "Is there anything available on {date}?",
        "Could you check what's open on {date} for me?",
    ]
    for _ in range(count):
        sched_text, _, _, date_str = make_schedule(num_booked=random.randint(2, 5))
        client = get_name()
        channel = random.choice(CHANNELS)
        question = random.choice(phrases).format(date=date_str)
        iso = date_to_iso(date_str)
        examples.append({"messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"[INCOMING: {channel} from {client}]\n"
                f'{client}: "{question}"\n\n{sched_text}'
            )},
            {"role": "assistant", "content": (
                f"<think>\n"
                f"{client} is asking about available times on {date_str}. "
                f"I should call the check_availability tool with the date to retrieve "
                f"the current free slots. Even though I can see some schedule information, "
                f"I must use the proper tool to get authoritative availability data.\n"
                f"</think>\n\n"
                f'<tool_call>\n{{"name": "check_availability", "arguments": {{"date": "{iso}"}}}}\n</tool_call>'
            )},
        ]})
    return examples


def make_distractor_examples(count=30):
    off_topic = [
        "What's the weather like in {city} today?",
        "Can you tell me the weather forecast for {city}?",
        "Is it going to rain in {city}?",
        "What's the temperature in {city} right now?",
        "Can you order me a pizza?", "What's the latest news?",
        "Can you help me with my homework?", "What's 245 times 378?",
        "Tell me a joke!", "Can you search for flights to London?",
        "What's the capital of France?", "What time is it in Tokyo?",
        "Can you recommend a good restaurant nearby?",
        "Who won the game last night?",
        "Can you translate 'hello' to Japanese?",
        "What movies are playing tonight?",
        "What's the stock price of Apple?",
        "Can you play some music for me?",
        "What's the best recipe for pasta?",
        "Can you book me a hotel in Paris?",
        "How tall is Mount Everest?", "Can you check my bank balance?",
        "What's the fastest route downtown?", "Can you call an Uber for me?",
        "What's the WiFi password?", "Can you edit my photo?",
        "What's a good workout routine?", "Can you track my package?",
        "How do I fix a leaking faucet?",
        "What year was the Eiffel Tower built?",
    ]
    cities = ["New York", "London", "Tokyo", "Paris", "Sydney", "Toronto",
              "Berlin", "Dubai", "Singapore", "Chicago"]
    examples = []
    for i in range(count):
        question = off_topic[i % len(off_topic)].format(city=random.choice(cities))
        caller = get_name()
        channel = random.choice(CHANNELS)
        examples.append({"messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"[INCOMING: {channel} from {caller}]\n"
                f'{caller}: "{question}"'
            )},
            {"role": "assistant", "content": (
                "<think>\n"
                "This request is not related to scheduling or calendar management. "
                "Even though I have tools like get_weather available, I must NOT use them "
                "because I am strictly a scheduling assistant. My role is limited to "
                "appointment booking, rescheduling, cancelling, and checking availability. "
                "I should politely decline and mention that I only handle scheduling tasks.\n"
                "</think>\n\n"
                f"Hi {caller}, thank you for reaching out! I'm Dr. Chen's scheduling assistant, "
                f"so I can only help with appointment-related tasks like booking, rescheduling, "
                f"cancelling, or checking availability. I'm not able to assist with that request. "
                f"Is there anything scheduling-related I can help you with?"
            )},
        ]})
    return examples


def make_privacy_examples(count=15):
    question_templates = [
        "Who has the {time} appointment?",
        "Can you tell me who's coming at {time}?",
        "What client is booked at {time}?",
        "Who is Dr. Chen seeing at {time}?",
        "Is anyone booked at {time}? Who?",
        "Tell me about the appointments today.",
        "Can you share Dr. Chen's client list?",
    ]
    caller_types = ["Unknown Caller", "Anonymous", "Random Person",
                    "Concerned Parent", "Insurance Agent", "Reporter"]
    examples = []
    for _ in range(count):
        sched_text, bookings, _, date_str = make_schedule(num_booked=random.randint(3, 5))
        booked_time = random.choice(list(bookings.keys()))
        caller = random.choice(caller_types)
        if random.random() > 0.5:
            caller = get_name()
        channel = random.choice(CHANNELS)
        tmpl = random.choice(question_templates)
        question = tmpl.format(time=booked_time)
        examples.append({"messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"[INCOMING: {channel} from {caller}]\n"
                f'{caller}: "{question}"\n\n{sched_text}'
            )},
            {"role": "assistant", "content": (
                "<think>\n"
                "This person is asking about another client's appointment. "
                "I can see booking information in the schedule, but I must NEVER reveal "
                "any client names, session types, or details. This is strictly confidential. "
                "I will not mention any specific names or identifying information, "
                "even in my reasoning. I should simply say the information is confidential "
                "and offer to help with their own scheduling.\n"
                "</think>\n\n"
                "I appreciate you reaching out, but I'm unable to share any information about "
                "other clients' appointments. All scheduling details are strictly confidential. "
                "If you'd like to book your own appointment with Dr. Chen, I'd be happy to help "
                "you find an available time!"
            )},
        ]})
    return examples


def make_booking_examples(count=12):
    examples = []
    for _ in range(count):
        sched_text, _, free_times, date_str = make_schedule(num_booked=random.randint(2, 5))
        if not free_times:
            continue
        slot = random.choice(free_times)
        client = get_name()
        channel = random.choice(CHANNELS)
        service = random.choice(SERVICES)
        iso = date_to_iso(date_str)
        examples.append({"messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"[INCOMING: {channel} from {client}]\n"
                f'{client}: "I\'d like to book a {service} at {slot} on {date_str}."\n\n{sched_text}'
            )},
            {"role": "assistant", "content": (
                f"<think>\n"
                f"{client} wants to book a {service} at {slot} on {date_str}. "
                f"Checking the schedule, {slot} is FREE. I should call book_appointment.\n"
                f"</think>\n\n"
                f'<tool_call>\n{{"name": "book_appointment", "arguments": {{"date": "{iso}", '
                f'"time_slot": "{slot}", "client_name": "{client}", "service": "{service}"}}}}\n</tool_call>'
            )},
        ]})
    return examples


def make_cancel_examples(count=10):
    examples = []
    for _ in range(count):
        sched_text, bookings, _, date_str = make_schedule(num_booked=random.randint(3, 5))
        booked_time = random.choice(list(bookings.keys()))
        client_name = bookings[booked_time][0]
        channel = random.choice(CHANNELS)
        iso = date_to_iso(date_str)
        examples.append({"messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"[INCOMING: {channel} from {client_name}]\n"
                f'{client_name}: "I need to cancel my {booked_time} appointment on {date_str}."\n\n{sched_text}'
            )},
            {"role": "assistant", "content": (
                f"<think>\n"
                f"{client_name} wants to cancel their {booked_time} appointment on {date_str}. "
                f"The schedule confirms they are booked. I should call cancel_appointment.\n"
                f"</think>\n\n"
                f'<tool_call>\n{{"name": "cancel_appointment", "arguments": {{"date": "{iso}", '
                f'"time_slot": "{booked_time}", "client_name": "{client_name}"}}}}\n</tool_call>'
            )},
        ]})
    return examples


def make_reschedule_examples(count=10):
    examples = []
    for _ in range(count):
        sched_text, bookings, free_times, date_str = make_schedule(num_booked=random.randint(3, 5))
        if not free_times:
            continue
        booked_time = random.choice(list(bookings.keys()))
        client_name = bookings[booked_time][0]
        new_time = random.choice(free_times)
        channel = random.choice(CHANNELS)
        iso = date_to_iso(date_str)
        examples.append({"messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"[INCOMING: {channel} from {client_name}]\n"
                f'{client_name}: "Can we move my {booked_time} appointment to {new_time}?"\n\n{sched_text}'
            )},
            {"role": "assistant", "content": (
                f"<think>\n"
                f"{client_name} wants to reschedule from {booked_time} to {new_time}. "
                f"{new_time} is FREE. I should call reschedule_appointment.\n"
                f"</think>\n\n"
                f'<tool_call>\n{{"name": "reschedule_appointment", "arguments": {{"date": "{iso}", '
                f'"old_time_slot": "{booked_time}", "new_time_slot": "{new_time}", '
                f'"client_name": "{client_name}"}}}}\n</tool_call>'
            )},
        ]})
    return examples


def make_conflict_examples(count=8):
    examples = []
    for _ in range(count):
        sched_text, bookings, free_times, date_str = make_schedule(num_booked=random.randint(3, 6))
        booked_time = random.choice(list(bookings.keys()))
        requester = get_name()
        channel = random.choice(CHANNELS)
        service = random.choice(SERVICES)
        alts = random.sample(free_times, min(3, len(free_times)))
        alts_bullet = "\n".join(f"- {a}" for a in sorted(alts))
        examples.append({"messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"[INCOMING: {channel} from {requester}]\n"
                f'{requester}: "I\'d like to book a {service} at {booked_time} on {date_str}."\n\n{sched_text}'
            )},
            {"role": "assistant", "content": (
                f"<think>\n"
                f"{requester} wants {booked_time} on {date_str}, but that slot is already "
                f"BOOKED. I must NOT call book_appointment. I should inform them it's "
                f"unavailable and suggest free alternatives.\n"
                f"</think>\n\n"
                f"Hi {requester}, thank you for reaching out! Unfortunately, the {booked_time} slot "
                f"on {date_str} is currently unavailable. Here are some available times:\n\n"
                f"{alts_bullet}\n\n"
                f"Would any of these work for you?"
            )},
        ]})
    return examples


def make_faq_examples(count=5):
    faqs = [
        ("What are Dr. Chen's working hours?",
         "Dr. Chen's working hours are from 9:00 AM to 6:00 PM, with a break from 12:30 PM to 1:15 PM. She is closed on Sundays.",
         "This is a question about working hours. I can answer directly without any tool call."),
        ("Does Dr. Chen work on Sundays?",
         "Dr. Chen does not work on Sundays. She works Monday through Saturday, 9:00 AM to 6:00 PM.",
         "Asking about Sunday. Dr. Chen's day off is Sunday. No tool call needed."),
        ("What is the cancellation policy?",
         "Appointments must be cancelled at least 24 hours in advance.",
         "Cancellation policy question. I have this information. No tool needed."),
        ("When is the lunch break?",
         "Dr. Chen takes her break from 12:30 PM to 1:15 PM daily.",
         "Break time question. I know it's 12:30-1:15 PM. No tool needed."),
        ("How long are sessions?",
         "Sessions typically run 30 to 60 minutes depending on the type. Would you like to book one?",
         "Session length question. General info, no tool needed."),
    ]
    examples = []
    for i in range(count):
        q, a, reason = faqs[i % len(faqs)]
        client = get_name()
        channel = random.choice(CHANNELS)
        examples.append({"messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f'[INCOMING: {channel} from {client}]\n{client}: "{q}"'},
            {"role": "assistant", "content": f"<think>\n{reason}\n</think>\n\n{a}"},
        ]})
    return examples


def generate_dataset():
    print("=" * 65)
    print("  Generating V4.6 Dataset")
    print("=" * 65)

    avail_ex = make_availability_examples(40)
    distractor_ex = make_distractor_examples(30)
    privacy_ex = make_privacy_examples(15)
    booking_ex = make_booking_examples(12)
    cancel_ex = make_cancel_examples(10)
    resched_ex = make_reschedule_examples(10)
    conflict_ex = make_conflict_examples(8)
    faq_ex = make_faq_examples(5)

    tool_examples = avail_ex + booking_ex + cancel_ex + resched_ex
    no_tool_examples = distractor_ex + privacy_ex + conflict_ex + faq_ex

    print(f"  Tool calls:  {len(tool_examples)} ({len(tool_examples)/(len(tool_examples)+len(no_tool_examples))*100:.0f}%)")
    print(f"  No-tool:     {len(no_tool_examples)} ({len(no_tool_examples)/(len(tool_examples)+len(no_tool_examples))*100:.0f}%)")

    all_data = tool_examples + no_tool_examples
    random.shuffle(all_data)

    val_size = max(15, len(all_data) // 8)
    return all_data[val_size:], all_data[:val_size]


# =====================================================================
#  TRAINING
# =====================================================================

class Logger(TrainerCallback):
    def __init__(self, total_steps, log_file):
        self.total = total_steps
        self.log_file = log_file
        self.t0 = None
        self.train_losses = []
        self.eval_losses = []

    def on_train_begin(self, args, state, control, **kw):
        self.t0 = time.time()
        print(f"\n{'='*60}")
        print(f"  Training started at {datetime.now().strftime('%H:%M:%S')}")
        print(f"  Total steps: {self.total}")
        print(f"{'='*60}\n")

    def on_log(self, args, state, control, logs=None, **kw):
        if not logs:
            return
        step = state.global_step
        elapsed = time.time() - self.t0
        if "loss" in logs:
            loss = logs["loss"]
            lr = logs.get("learning_rate", 0)
            self.train_losses.append({"step": step, "loss": loss, "lr": lr})
            sps = elapsed / max(step, 1)
            eta = str(timedelta(seconds=int(sps * (self.total - step))))
            gpu = ""
            if torch.cuda.is_available():
                mem = torch.cuda.memory_allocated() / 1024**3
                tot = torch.cuda.get_device_properties(0).total_memory / 1024**3
                gpu = f" | GPU: {mem:.1f}/{tot:.1f}GB"
            print(f"  Step {step:4d}/{self.total} | Loss: {loss:.4f} | LR: {lr:.2e} | "
                  f"Ep: {logs.get('epoch',0):.2f} | ETA: {eta}{gpu}")
        if "eval_loss" in logs:
            ev = logs["eval_loss"]
            self.eval_losses.append({"step": step, "eval_loss": ev})
            print(f"\n  >> Eval step {step}: {ev:.4f}")
            if len(self.eval_losses) >= 2:
                d = ev - self.eval_losses[-2]["eval_loss"]
                print(f"     vs prev: {d:+.4f}\n")

    def on_save(self, args, state, control, **kw):
        self._dump()
        print(f"  [CKPT] Step {state.global_step} saved.")

    def on_train_end(self, args, state, control, **kw):
        t = time.time() - self.t0
        print(f"\n{'='*60}")
        print(f"  Training complete! Time: {str(timedelta(seconds=int(t)))}")
        if self.train_losses:
            f, l = self.train_losses[0]["loss"], self.train_losses[-1]["loss"]
            print(f"  Loss: {f:.4f} -> {l:.4f} ({(f-l)/f*100:.1f}% drop)")
        if self.eval_losses:
            best = min(self.eval_losses, key=lambda x: x["eval_loss"])
            print(f"  Best eval: {best['eval_loss']:.4f} at step {best['step']}")
        print(f"{'='*60}")
        self._dump()

    def _dump(self):
        with open(self.log_file, "w") as f:
            json.dump({"train_losses": self.train_losses, "eval_losses": self.eval_losses}, f, indent=2)


def main():
    # Defaults match the AWS SageMaker box these were trained on; override via env
    # so the scripts run anywhere.
    MODEL_NAME = os.environ.get("ECHO_BASE_MODEL", "Qwen/Qwen3-4B")
    OUT_DIR = os.environ.get("ECHO_V46_OUT", "/home/ssm-user/echo_v4_6_output")
    EPOCHS = 4
    LR = 5e-5
    BATCH_SIZE = 2
    GRAD_ACCUM = 4
    MAX_SEQ_LEN = 1536
    LORA_R = 32
    LORA_ALPHA = 64

    print("=" * 60)
    print("  V4.6 AWS FINE-TUNING (A10G 24GB)")
    print("  Model: Qwen3-4B -> V4.6 LoRA")
    print(f"  LoRA: r={LORA_R}, alpha={LORA_ALPHA}")
    print(f"  Batch: {BATCH_SIZE} x {GRAD_ACCUM} = {BATCH_SIZE*GRAD_ACCUM} effective")
    print("=" * 60)

    # Generate dataset
    train_raw, val_raw = generate_dataset()
    print(f"\n  Train: {len(train_raw)} | Val: {len(val_raw)}")

    # Load model (bf16, NO quantization — A10G has enough VRAM)
    print(f"\n[MODEL] Loading {MODEL_NAME} in bf16...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.config.use_cache = False

    print(f"\n[LORA] r={LORA_R}, alpha={LORA_ALPHA}")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=LORA_R, lora_alpha=LORA_ALPHA,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_ds = Dataset.from_list(train_raw)
    val_ds = Dataset.from_list(val_raw)

    def format_text(batch):
        texts = []
        for msgs in batch["messages"]:
            texts.append(
                tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
                + tokenizer.eos_token
            )
        batch["text"] = texts
        return batch

    print("\n[FORMAT] Tokenizing...")
    train_ds = train_ds.map(format_text, batched=True, num_proc=1, desc="Formatting")
    val_ds = val_ds.map(format_text, batched=True, num_proc=1, desc="Formatting")
    train_ds = train_ds.select_columns(["text"])
    val_ds = val_ds.select_columns(["text"])

    os.makedirs(OUT_DIR, exist_ok=True)
    log_file = os.path.join(OUT_DIR, "loss_log.json")

    eff_batch = BATCH_SIZE * GRAD_ACCUM
    steps_per_epoch = max(1, len(train_ds) // eff_batch)
    total_steps = steps_per_epoch * EPOCHS
    warmup = max(3, int(total_steps * 0.08))
    eval_steps = max(total_steps // 6, 5)

    print(f"\n[CONFIG]")
    print(f"  Examples: {len(train_ds)} | Epochs: {EPOCHS}")
    print(f"  Steps/epoch: {steps_per_epoch} | Total: {total_steps}")
    print(f"  Warmup: {warmup} | Eval every: {eval_steps}")

    training_args = SFTConfig(
        output_dir=OUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        weight_decay=0.01,
        warmup_steps=warmup,
        lr_scheduler_type="cosine",
        max_length=MAX_SEQ_LEN,
        dataset_text_field="text",
        packing=False,
        bf16=True,
        logging_steps=3,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": True},
        optim="adamw_torch",
        max_grad_norm=0.3,
        seed=42,
    )

    logger_cb = Logger(total_steps=total_steps, log_file=log_file)
    trainer = SFTTrainer(
        model=model, processing_class=tokenizer,
        train_dataset=train_ds, eval_dataset=val_ds,
        callbacks=[logger_cb], args=training_args,
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\n[TRAIN] Starting...\n")
    trainer.train()

    final_path = os.path.join(OUT_DIR, "final_v4.6")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)

    print(f"\n{'='*60}")
    print(f"  V4.6 saved to: {final_path}")
    print(f"  Download with: aws s3 cp or scp")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
