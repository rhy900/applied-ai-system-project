"""
Music Recommender AI — Agentic System
--------------------------------------
Usage:
  python -m src.main                        # interactive chat mode
  python -m src.main --demo                 # 3 preset demo inputs
  python -m src.main --eval                 # run test harness (no API needed)
  python -m src.main --no-browser           # skip auto-opening YouTube
"""

import argparse
import os
import sys

from dotenv import load_dotenv
load_dotenv()

from src.logger import setup_logger
from src.recommender import load_songs_from_csv

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "songs.csv")

DEMO_INPUTS = [
    "I want something chill and relaxing for studying, not too intense",
    "I'm hyped up and want high energy rock music to work out to",
    "Give me a feel-good pop song, something happy and danceable",
]


def run_demo(agent, open_browser: bool) -> None:
    for text in DEMO_INPUTS:
        print(f"\n{'─'*62}")
        print(f"INPUT: \"{text}\"")
        agent.run(text, open_browser=open_browser)


def run_interactive(agent, open_browser: bool) -> None:
    print("\nMusic Recommender AI  |  type 'quit' to exit")
    print("─" * 50)
    while True:
        try:
            user_input = input("\nWhat kind of music do you want? → ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        agent.run(user_input, open_browser=open_browser)


def run_eval(songs) -> None:
    from src.evaluator import run_all_tests, print_summary
    results = run_all_tests(songs)
    print_summary(results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Music Recommender Agentic AI")
    parser.add_argument("--demo",       action="store_true", help="Run 3 demo inputs")
    parser.add_argument("--eval",       action="store_true", help="Run test harness (no API)")
    parser.add_argument("--no-browser", action="store_true", help="Don't open YouTube")
    args = parser.parse_args()

    logger = setup_logger()
    logger.info("System starting")

    songs = load_songs_from_csv(DATA_PATH)

    if args.eval:
        run_eval(songs)
        return

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY is not set.")
        print("  1. Copy .env.example → .env")
        print("  2. Paste your Gemini API key into .env")
        print("  Tip: run --eval to test the scoring engine without an API key.")
        sys.exit(1)

    from src.agent import MusicAgent
    agent = MusicAgent(songs, api_key=api_key)
    open_browser = not args.no_browser

    if args.demo:
        run_demo(agent, open_browser)
    else:
        run_interactive(agent, open_browser)


if __name__ == "__main__":
    main()
