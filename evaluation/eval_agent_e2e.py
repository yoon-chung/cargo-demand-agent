"""
End-to-end agent evaluation: 3 scenarios across factual/forecast intents.

Run from project root:
    python evaluation/eval_agent_e2e.py

Each scenario triggers the full LangGraph pipeline (parse + branch + synthesize).
Cost: ~$0.01 per scenario (Solar-Pro 2 calls + Tavily 1 call + RAG).
"""
import logging
import sys
from pathlib import Path

# Make `src.cargo_demand_agent.*` importable when this file is run directly.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")

from dotenv import load_dotenv

load_dotenv(".env")

from src.cargo_demand_agent.agent import get_app

SCENARIOS = [
    {"name": "factual", "user_query": "최근 미국 디젤 가격은?"},
    {"name": "forecast (specific)", "user_query": "다음달 LA Port 컨테이너 처리량과 드라이버 분석"},
    {"name": "forecast (3-month)", "user_query": "3개월 후 LA Port 처리량 전망"},
]


def main() -> None:
    app = get_app()
    for sc in SCENARIOS:
        print()
        print("=" * 70)
        print(f"SCENARIO: {sc['name']}")
        print(f"QUERY: {sc['user_query']}")
        print("=" * 70)
        result = app.invoke({"user_query": sc["user_query"]})

        print(f"\n[INTENT]  {result.get('intent')}")
        print(f"[HORIZON] {result.get('horizon_months')}")

        fr = result.get("forecast_result")
        if fr is not None:
            print(
                f"[FORECAST] {fr.forecast_teu:,.0f} TEU @ "
                f"{fr.forecast_period}  ({fr.model_name})"
            )
            drivers = [(d.feature, round(d.shap_value, 4)) for d in fr.top_drivers]
            print(f"[DRIVERS] {drivers}")

        print(f"[RAG_DOCS]    {len(result.get('rag_documents', []))} chunks")
        print(f"[WEB_RESULTS] {len(result.get('web_results', []))}")

        print("\n--- FINAL ANSWER ---")
        print(result.get("final_answer"))
        print("\n--- CITATIONS ---")
        for c in result.get("citations", [])[:6]:
            print(f"  {c}")


if __name__ == "__main__":
    main()
