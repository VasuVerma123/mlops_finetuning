#!/usr/bin/env python3
"""
V4.7 AWS Fine-tuning — Surgical Fix for Reschedule + Distractor
================================================================
V4.6-AWS scored 90%. Missing 5 points:
  - Reschedule (2 fails): Model called cancel_appointment instead of reschedule_appointment
  - Distractor (3 fails): Model called get_weather instead of declining

Fix strategy:
  - 40 reschedule examples (heavy — model must learn THIS specific tool)
  - 35 distractor examples (model must NEVER call get_weather/off-topic tools)
  - 30 retention examples (booking, cancel, availability, conflict, privacy, FAQ)
  - Balance: ~57% tool / ~43% no-tool
  - All examples include <think> reasoning

Stack: V4.6-AWS (merge) -> V4.7 LoRA
Runs on g5.2xlarge (A10G 24GB).

Usage: python3 aws_finetune_v4_7.py
"""

import json, os, random, time, gc
from datetime import datetime, timedelta

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import SFTTrainer, SFTConfig

random.seed(42)

# =====================================================================
#  DATASET GENERATION
# =====================================================================

FIRST_NAMES = [
    "Alex", "Maria", "David", "Sarah", "James", "Lisa", "Tom", "Anika",
    "Emma", "Noah", "Zara", "Marcus", "Yuki", "Jordan", "Priya", "Carlos",
    "Fatima", "Michael", "Lucia", "Robert", "Mei", "Andre", "Sophia",
    "Raj", "Elena", "Kevin", "Hana", "Omar", "Chloe", "Victor",
    "Aisha", "Daniel", "Nadia", "Leo", "Jasmine", "Hassan", "Grace",
    "Dmitri", "Rosa", "Ian", "Lena", "Felix", "Maya", "Kai", "Nina",
    "Oscar", "Suki", "Theo", "Amara", "Ravi", "Chen", "Bruno", "Aria",
    "Ethan", "Lila", "Marco", "Yuna", "Diego", "Iris", "Soren",
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


# ---- FIX 1: RESCHEDULE (40 examples) ----

def make_reschedule_examples(count=40):
    """
    Model MUST call reschedule_appointment (NOT cancel + book separately).
    Diverse phrasing to hammer this in.
    """
    phrases = [
        "Can we move my {old} appointment to {new}?",
        "I need to reschedule my {old} session to {new} instead.",
        "Could you change my {old} appointment to {new}?",
        "Hey, can I switch my {old} slot to {new}?",
        "I'd like to move my appointment from {old} to {new}.",
        "Please reschedule me from {old} to {new}.",
        "Is it possible to shift my {old} booking to {new}?",
        "My {old} time doesn't work anymore. Can we do {new}?",
        "I can't make {old}. Can you move me to {new}?",
        "Would it be okay to change from {old} to {new}?",
        "Something came up at {old}. Can I come at {new} instead?",
        "Hi, I need my {old} appointment moved to {new} please.",
        "Reschedule my {old} to {new} please.",
        "Can you push my {old} session to {new}?",
        "I want to move from {old} to {new} on {date}.",
    ]
    examples = []
    for i in range(count):
        sched_text, bookings, free_times, date_str = make_schedule(num_booked=random.randint(3, 6))
        if not free_times or not bookings:
            continue
        booked_time = random.choice(list(bookings.keys()))
        client_name = bookings[booked_time][0]
        new_time = random.choice(free_times)
        channel = random.choice(CHANNELS)
        iso = date_to_iso(date_str)

        phrase = phrases[i % len(phrases)].format(old=booked_time, new=new_time, date=date_str)

        think_text = (
            f"{client_name} wants to reschedule their appointment from {booked_time} to {new_time} on {date_str}. "
            f"I can see that {booked_time} is currently BOOKED by {client_name} and {new_time} is FREE. "
            f"I must use the reschedule_appointment tool for this - NOT cancel_appointment followed by book_appointment. "
            f"The reschedule_appointment tool handles both the cancellation and rebooking in one step. "
            f"Parameters: date={iso}, old_time_slot={booked_time}, new_time_slot={new_time}, client_name={client_name}."
        )

        examples.append({"messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"[INCOMING: {channel} from {client_name}]\n"
                f'{client_name}: "{phrase}"\n\n{sched_text}'
            )},
            {"role": "assistant", "content": (
                f"<think>\n{think_text}\n</think>\n\n"
                f'<tool_call>\n{{"name": "reschedule_appointment", "arguments": {{"date": "{iso}", '
                f'"old_time_slot": "{booked_time}", "new_time_slot": "{new_time}", '
                f'"client_name": "{client_name}"}}}}\n</tool_call>'
            )},
        ]})
    return examples


# ---- FIX 2: DISTRACTOR (35 examples) ----

def make_distractor_examples(count=35):
    """Model must NOT call any tool. Must politely decline and mention scheduling."""
    off_topic = [
        "What's the weather like in {city} today?",
        "Can you tell me the weather forecast for {city}?",
        "Is it going to rain in {city}?",
        "What's the temperature in {city} right now?",
        "Can you order me a pizza?",
        "What's the latest news?",
        "Can you help me with my homework?",
        "What's 245 times 378?",
        "Tell me a joke!",
        "Can you search for flights to London?",
        "What's the capital of France?",
        "What time is it in Tokyo?",
        "Can you recommend a good restaurant nearby?",
        "Who won the game last night?",
        "Can you translate 'hello' to Japanese?",
        "What movies are playing tonight?",
        "What's the stock price of Apple?",
        "Can you play some music for me?",
        "What's the best recipe for pasta?",
        "Can you book me a hotel in Paris?",
        "How tall is Mount Everest?",
        "Can you check my bank balance?",
        "What's the fastest route downtown?",
        "Can you call an Uber for me?",
        "What's the WiFi password?",
        "Can you edit my photo?",
        "What's a good workout routine?",
        "Can you track my package?",
        "How do I fix a leaking faucet?",
        "What year was the Eiffel Tower built?",
        "What's the weather going to be this weekend in {city}?",
        "Will it snow in {city} tomorrow?",
        "Give me the 5-day forecast for {city}.",
        "How's the weather looking for {city} next week?",
        "Check the weather in {city} for me please.",
    ]
    cities = ["New York", "London", "Tokyo", "Paris", "Sydney", "Toronto",
              "Berlin", "Dubai", "Singapore", "Chicago", "Mumbai", "Seoul"]

    decline_responses = [
        "Hi {name}, thank you for reaching out! I'm Dr. Chen's scheduling assistant, "
        "so I can only help with appointment-related tasks like booking, rescheduling, "
        "cancelling, or checking availability. I'm not able to assist with that request. "
        "Is there anything scheduling-related I can help you with?",

        "Hello {name}! I appreciate you contacting us. However, I'm specifically designed "
        "to handle Dr. Chen's scheduling needs - booking appointments, checking availability, "
        "rescheduling, and cancellations. I can't help with that particular request. "
        "Would you like to schedule an appointment instead?",

        "Hi {name}! Thanks for your message. As Dr. Chen's scheduling assistant, my capabilities "
        "are limited to appointment management - I can help you book, cancel, reschedule, or check "
        "availability. Unfortunately, I can't assist with that. Is there a scheduling matter I can help with?",
    ]

    examples = []
    for i in range(count):
        question = off_topic[i % len(off_topic)].format(city=random.choice(cities))
        caller = get_name()
        channel = random.choice(CHANNELS)
        decline = decline_responses[i % len(decline_responses)].format(name=caller)

        think_text = (
            "This request is not related to scheduling or calendar management. "
            "Even though I have tools like get_weather available in my tool list, "
            "I must NOT use them because I am strictly a scheduling assistant. "
            "My role is limited to appointment booking, rescheduling, cancelling, "
            "and checking availability for Dr. Chen's practice. "
            "I should politely decline and mention that I only handle scheduling tasks. "
            "I must NOT call any tool for this request."
        )

        examples.append({"messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"[INCOMING: {channel} from {caller}]\n"
                f'{caller}: "{question}"'
            )},
            {"role": "assistant", "content": f"<think>\n{think_text}\n</think>\n\n{decline}"},
        ]})
    return examples


# ---- RETENTION: Keep existing strengths ----

def make_booking_examples(count=8):
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
                f"<think>\n{client} wants to book a {service} at {slot} on {date_str}. "
                f"Checking the schedule, {slot} is FREE. I should call book_appointment.\n</think>\n\n"
                f'<tool_call>\n{{"name": "book_appointment", "arguments": {{"date": "{iso}", '
                f'"time_slot": "{slot}", "client_name": "{client}", "service": "{service}"}}}}\n</tool_call>'
            )},
        ]})
    return examples


def make_cancel_examples(count=6):
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
                f"<think>\n{client_name} wants to cancel their {booked_time} appointment. "
                f"This is a cancellation, NOT a reschedule. I should call cancel_appointment.\n</think>\n\n"
                f'<tool_call>\n{{"name": "cancel_appointment", "arguments": {{"date": "{iso}", '
                f'"time_slot": "{booked_time}", "client_name": "{client_name}"}}}}\n</tool_call>'
            )},
        ]})
    return examples


def make_availability_examples(count=6):
    examples = []
    phrases = [
        "What times do you have available on {date}?",
        "Any openings on {date}?",
        "What's free on {date}?",
        "Can I see what's open on {date}?",
        "Do you have any availability on {date}?",
        "When can I come in on {date}?",
    ]
    for i in range(count):
        sched_text, _, _, date_str = make_schedule(num_booked=random.randint(2, 5))
        client = get_name()
        channel = random.choice(CHANNELS)
        question = phrases[i % len(phrases)].format(date=date_str)
        iso = date_to_iso(date_str)
        examples.append({"messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"[INCOMING: {channel} from {client}]\n"
                f'{client}: "{question}"\n\n{sched_text}'
            )},
            {"role": "assistant", "content": (
                f"<think>\n{client} is asking about available times. "
                f"I should call check_availability to get the authoritative data.\n</think>\n\n"
                f'<tool_call>\n{{"name": "check_availability", "arguments": {{"date": "{iso}"}}}}\n</tool_call>'
            )},
        ]})
    return examples


def make_conflict_examples(count=5):
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
                f"<think>\n{requester} wants {booked_time} but it's already BOOKED. "
                f"I must NOT book over it. I should inform them and suggest alternatives.\n</think>\n\n"
                f"Hi {requester}, unfortunately the {booked_time} slot on {date_str} is currently "
                f"unavailable. Here are some available times:\n\n{alts_bullet}\n\n"
                f"Would any of these work for you?"
            )},
        ]})
    return examples


def make_privacy_examples(count=3):
    examples = []
    for _ in range(count):
        sched_text, bookings, _, _ = make_schedule(num_booked=random.randint(3, 5))
        booked_time = random.choice(list(bookings.keys()))
        caller = random.choice(["Unknown Caller", "Anonymous", get_name()])
        channel = random.choice(CHANNELS)
        examples.append({"messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"[INCOMING: {channel} from {caller}]\n"
                f'{caller}: "Who has the {booked_time} appointment?"\n\n{sched_text}'
            )},
            {"role": "assistant", "content": (
                "<think>\nThis person is asking about another client's appointment. "
                "I must NEVER reveal any client names or details. This is confidential. "
                "I should decline and offer to help with their own scheduling.\n</think>\n\n"
                "I appreciate you reaching out, but I'm unable to share any information about "
                "other clients' appointments. All scheduling details are strictly confidential. "
                "If you'd like to book your own appointment with Dr. Chen, I'd be happy to help!"
            )},
        ]})
    return examples


def make_faq_examples(count=2):
    faqs = [
        ("What are Dr. Chen's working hours?",
         "Dr. Chen's working hours are from 9:00 AM to 6:00 PM, with a break from 12:30 PM to 1:15 PM. She is closed on Sundays.",
         "Working hours question. I can answer directly without any tool call."),
        ("What is the cancellation policy?",
         "Appointments must be cancelled at least 24 hours in advance.",
         "Cancellation policy question. No tool needed."),
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
    print("=" * 60)
    print("  Generating V4.7 Surgical Dataset")
    print("  FIX: Reschedule + Distractor")
    print("=" * 60)

    resched_ex = make_reschedule_examples(40)
    distractor_ex = make_distractor_examples(35)
    booking_ex = make_booking_examples(8)
    cancel_ex = make_cancel_examples(6)
    avail_ex = make_availability_examples(6)
    conflict_ex = make_conflict_examples(5)
    privacy_ex = make_privacy_examples(3)
    faq_ex = make_faq_examples(2)

    tool_examples = resched_ex + booking_ex + cancel_ex + avail_ex
    no_tool_examples = distractor_ex + conflict_ex + privacy_ex + faq_ex

    print(f"\n  FIX targets:")
    print(f"    Reschedule (TOOL):    {len(resched_ex)}")
    print(f"    Distractor (NO tool): {len(distractor_ex)}")
    print(f"\n  Retention:")
    print(f"    Booking:      {len(booking_ex)}")
    print(f"    Cancel:       {len(cancel_ex)}")
    print(f"    Availability: {len(avail_ex)}")
    print(f"    Conflict:     {len(conflict_ex)}")
    print(f"    Privacy:      {len(privacy_ex)}")
    print(f"    FAQ:          {len(faq_ex)}")

    total = len(tool_examples) + len(no_tool_examples)
    print(f"\n  BALANCE:")
    print(f"    Tool calls:  {len(tool_examples)} ({len(tool_examples)/total*100:.0f}%)")
    print(f"    No-tool:     {len(no_tool_examples)} ({len(no_tool_examples)/total*100:.0f}%)")

    all_data = tool_examples + no_tool_examples
    random.shuffle(all_data)

    val_size = max(10, len(all_data) // 8)
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
        print(f"  V4.7 Training started at {datetime.now().strftime('%H:%M:%S')}")
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
        print(f"  V4.7 Training complete! Time: {str(timedelta(seconds=int(t)))}")
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
    MODEL_NAME = "Qwen/Qwen3-4B"
    V46_ADAPTER = "/home/ssm-user/echo_v4_6_output/final_v4.6"
    OUT_DIR = "/home/ssm-user/echo_v4_7_output"
    EPOCHS = 5
    LR = 4e-5
    BATCH_SIZE = 2
    GRAD_ACCUM = 4
    MAX_SEQ_LEN = 1536
    LORA_R = 16
    LORA_ALPHA = 32

    print("=" * 60)
    print("  V4.7 AWS FINE-TUNING")
    print("  Stack: V4.6-AWS (merge) -> V4.7 LoRA")
    print("  Target: 95%+ (fix Reschedule + Distractor)")
    print("=" * 60)

    train_raw, val_raw = generate_dataset()
    print(f"\n  Train: {len(train_raw)} | Val: {len(val_raw)}")

    print(f"\n[MODEL] Loading {MODEL_NAME} in bf16...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Merge V4.6-AWS adapter
    if os.path.exists(V46_ADAPTER):
        print(f"  Merging V4.6-AWS adapter...")
        model = PeftModel.from_pretrained(model, V46_ADAPTER)
        model = model.merge_and_unload()
        gc.collect()
        torch.cuda.empty_cache()
    else:
        print(f"  [WARN] V4.6 adapter not found: {V46_ADAPTER}")

    model.config.use_cache = False

    print(f"\n[LORA] r={LORA_R}, alpha={LORA_ALPHA} (small - surgical fix)")
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

    print(f"\n[TRAIN] Starting V4.7...\n")
    trainer.train()

    final_path = os.path.join(OUT_DIR, "final_v4.7")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)

    print(f"\n{'='*60}")
    print(f"  V4.7 saved to: {final_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
