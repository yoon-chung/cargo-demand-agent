"""
CLI entry point for the cargo demand agent.

Usage:
    python main.py "다음달 LA Port 처리량과 이유"
    python main.py "현재 FBX 지수는?" --verbose
"""
import argparse
import logging

from dotenv import load_dotenv


def main():
    parser = argparse.ArgumentParser(
        description="Cargo Demand Agent: 자연어 질의 → 예측 + 근거 + 권장 액션"
    )
    parser.add_argument("query", help="User query.")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show INFO-level logs.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="[%(name)s] %(message)s",
    )
    load_dotenv()  # must precede agent import (LLM clients read env on init)

    from src.cargo_demand_agent.agent import get_app

    state = get_app().invoke({"user_query": args.query})

    print(state.get("final_answer") or "(no answer)")
    citations = state.get("citations", [])
    if citations:
        print("\nCitations:")
        for c in citations:
            print(f"  - {c}")


if __name__ == "__main__":
    main()