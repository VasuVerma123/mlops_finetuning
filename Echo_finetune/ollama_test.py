"""
Echo v4.7 - AI Scheduling Assistant (Test Script)
====================================================
Fine-tuned on Qwen3-4B by Vasu Verma.
Hosted on Ollama Hub: https://ollama.com/vasuverma/echo-v4.7

Echo is Dr. Maya Chen's personal scheduling assistant.
She handles bookings, cancellations, rescheduling, and client communication.

SETUP (one-time, 3 steps):

  Step 1 - Install Ollama
      Windows/Mac/Linux: https://ollama.com/download

  Step 2 - Install Python dependency
      pip install ollama colorama

  Step 3 - Run this script
      python ollama_test.py
      (The script will auto-offer to download the model ~4.3 GB on first run)

USAGE:

  - Type any message and press Enter to chat with Echo
  - Type a number (1-8) to fire a quick sample test
  - Type  clear   to reset the conversation
  - Type  exit    to quit

SAMPLE PROMPTS:

  1. "Book John at 10am on 2026-04-01 for therapy"
  2. "What are Dr. Chen's working hours?"
  3. "Cancel Sarah's 2pm appointment on 2026-04-02"
  4. "Check availability for 2026-04-05"
  5. "Reschedule Mike from 9am to 11am on 2026-04-03"
  6. "Send a reminder to David about his session tomorrow"
  7. "What is the cancellation policy?"
  8. "Book an emergency session for Lisa today at 3pm"
"""

import sys
import json
import re

try:
    import ollama
except ImportError:
    print("\n[ERROR] 'ollama' package not found.")
    print("Install it with:  pip install ollama")
    sys.exit(1)

try:
    from colorama import Fore, Style, init
    init(autoreset=True)
except ImportError:
    class Fore:
        CYAN = GREEN = YELLOW = RED = MAGENTA = BLUE = WHITE = ""
    class Style:
        RESET_ALL = BRIGHT = DIM = ""

# ── Config ──────────────────────────────────────────────────────────────────

MODEL = "vasuverma/echo-v4.7"

# ── Sample test prompts (numbered shortcuts) ─────────────────────────────────

QUICK_TESTS = [
    ("Book appointment",          "Book John at 10am on 2026-04-01 for therapy"),
    ("Working hours",             "What are Dr. Chen's working hours?"),
    ("Cancel appointment",        "Cancel Sarah's 2pm appointment on 2026-04-02"),
    ("Check availability",        "Check availability for 2026-04-05"),
    ("Reschedule appointment",    "Reschedule Mike from 9am to 11am on 2026-04-03"),
    ("Send message",              "Send a reminder to David about his session tomorrow"),
    ("Cancellation policy",       "What is the cancellation policy?"),
    ("Emergency booking",         "Book an emergency session for Lisa today at 3pm"),
]

# ── Banner ───────────────────────────────────────────────────────────────────

BANNER = f"""
{Fore.CYAN}{Style.BRIGHT}
======================================================
      Echo v4.7 -- AI Scheduling Assistant
      Fine-tuned on Qwen3-4B  |  by Vasu Verma
======================================================
{Style.RESET_ALL}
  {Fore.WHITE}Echo manages Dr. Maya Chen's therapy practice:{Style.RESET_ALL}
    - Booking / cancelling / rescheduling appointments
    - Checking calendar availability
    - Sending messages to clients
    - Answering policy questions

  {Fore.YELLOW}Type a message  |  number 1-8 for quick tests  |  'clear' to reset  |  'exit' to quit{Style.RESET_ALL}
"""


# ── Model check / auto-pull ──────────────────────────────────────────────────

def check_model():
    """Verify model is available locally; offer to pull from Ollama Hub if not."""
    try:
        models = ollama.list()
        available = [m.model for m in models.models]
        if not any(MODEL in m for m in available):
            print(f"\n{Fore.YELLOW}  Model '{MODEL}' is not installed locally.{Style.RESET_ALL}")
            print(f"  It will be downloaded from: {Fore.CYAN}https://ollama.com/vasuverma/echo-v4.7{Style.RESET_ALL}")
            answer = input(f"\n  Download now? (~4.3 GB) [yes/no]: ").strip().lower()
            if answer in ("yes", "y"):
                print(f"\n  Pulling {MODEL} — this may take a few minutes...\n")
                for chunk in ollama.pull(MODEL, stream=True):
                    if hasattr(chunk, "status") and chunk.status:
                        completed = getattr(chunk, "completed", None)
                        total = getattr(chunk, "total", None)
                        if completed and total:
                            pct = int(completed / total * 100)
                            bar = "#" * (pct // 5) + "." * (20 - pct // 5)
                            print(f"\r  [{bar}] {pct}% - {chunk.status}", end="", flush=True)
                        else:
                            print(f"\r  {chunk.status}...", end="", flush=True)
                print(f"\n\n{Fore.GREEN}  Model downloaded and ready!{Style.RESET_ALL}\n")
            else:
                print(f"\n  To install manually, run:")
                print(f"    {Fore.CYAN}ollama pull {MODEL}{Style.RESET_ALL}")
                sys.exit(0)
    except ollama.ResponseError as e:
        print(f"\n{Fore.RED}  [ERROR] Ollama returned an error: {e}{Style.RESET_ALL}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Fore.RED}  [ERROR] Could not connect to Ollama.{Style.RESET_ALL}")
        print(f"  Make sure Ollama is running. Start it with:  {Fore.CYAN}ollama serve{Style.RESET_ALL}")
        print(f"  Details: {e}")
        sys.exit(1)


# ── Quick test menu ──────────────────────────────────────────────────────────

def show_quick_tests():
    print(f"\n{Fore.MAGENTA}{Style.BRIGHT}  Quick test prompts:{Style.RESET_ALL}")
    for i, (label, prompt) in enumerate(QUICK_TESTS, 1):
        print(f"  {Fore.CYAN}[{i}]{Style.RESET_ALL} {Fore.WHITE}{label:<26}{Style.RESET_ALL}  {Style.DIM}{prompt}{Style.RESET_ALL}")
    print()


# ── Tool call display ────────────────────────────────────────────────────────

def display_tool_call(response_text: str):
    """
    If the response contains a <tool_call> block, pretty-print it
    so the user can see what action Echo decided to take.
    """
    pattern = re.compile(r"<tool_call>(.*?)(?:</tool_call>|<tool_call>|$)", re.DOTALL)
    match = pattern.search(response_text)
    if match:
        raw = match.group(1).strip()
        try:
            data = json.loads(raw)
            func = data.get("function") or data.get("name", "unknown")
            args = data.get("arguments") or data.get("parameters") or data.get("args", {})
            print(f"\n  {Fore.YELLOW}⚙  Tool called:{Style.RESET_ALL} {Fore.CYAN}{func}{Style.RESET_ALL}")
            for k, v in args.items():
                print(f"     {Fore.WHITE}{k:<15}{Style.RESET_ALL} {v}")
        except json.JSONDecodeError:
            print(f"\n  {Fore.YELLOW}⚙  Tool call detected (raw):{Style.RESET_ALL} {raw[:120]}")


# ── Main chat loop ───────────────────────────────────────────────────────────

def chat():
    history = []

    print(BANNER)
    check_model()
    show_quick_tests()

    print(f"  {Fore.GREEN}{Style.BRIGHT}Echo is ready. Start chatting!{Style.RESET_ALL}\n")
    print("  " + "─" * 54 + "\n")

    while True:
        try:
            user_input = input(f"  {Fore.CYAN}You:{Style.RESET_ALL} ").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n\n  {Fore.YELLOW}Goodbye!{Style.RESET_ALL}\n")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit"):
            print(f"\n  {Fore.YELLOW}Goodbye!{Style.RESET_ALL}\n")
            break

        if user_input.lower() == "clear":
            history.clear()
            print(f"\n  {Fore.GREEN}Conversation cleared.{Style.RESET_ALL}\n")
            show_quick_tests()
            continue

        if user_input.lower() in ("help", "?"):
            show_quick_tests()
            continue

        # Number shortcut: user types "1" → runs quick test #1
        if user_input.isdigit():
            idx = int(user_input) - 1
            if 0 <= idx < len(QUICK_TESTS):
                _, user_input = QUICK_TESTS[idx]
                print(f"  {Fore.CYAN}You:{Style.RESET_ALL} {user_input}")
            else:
                print(f"  {Fore.RED}No quick test #{int(user_input)}. Valid range: 1–{len(QUICK_TESTS)}{Style.RESET_ALL}")
                continue

        history.append({"role": "user", "content": user_input})

        print(f"\n  {Fore.GREEN}Echo:{Style.RESET_ALL} ", end="", flush=True)

        try:
            full_response = ""
            for chunk in ollama.chat(
                model=MODEL,
                messages=history,
                stream=True,
                think=False,          # Disable Qwen3 internal thinking for faster replies
            ):
                token = chunk.message.content
                if token:
                    print(token, end="", flush=True)
                    full_response += token

            print("\n")

            # Show tool call summary if Echo decided to call a function
            display_tool_call(full_response)

            history.append({"role": "assistant", "content": full_response})

        except ollama.ResponseError as e:
            print(f"\n  {Fore.RED}[ERROR] {e}{Style.RESET_ALL}")
            print(f"  Make sure the model is pulled: {Fore.CYAN}ollama pull {MODEL}{Style.RESET_ALL}\n")
        except Exception as e:
            print(f"\n  {Fore.RED}[ERROR] {e}{Style.RESET_ALL}\n")

        print("  " + "─" * 54 + "\n")


if __name__ == "__main__":
    chat()
