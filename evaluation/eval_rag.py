"""
RAG retrieval quality evaluation: 5 manual queries simulating SHAP-driven
RAG calls in the agent's pipeline.

Run from project root:
    python evaluation/eval_rag.py

Outputs markdown tables (one per query) for human (domain expert) judgment.
See evaluation/results_d1.md for the analyzed D1 results and design decisions.
"""
import logging
import sys
from pathlib import Path

# Make `src.cargo_demand_agent.*` importable when this file is run directly.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logging.basicConfig(level=logging.WARNING)  # quiet retrieval logs

from src.cargo_demand_agent.retriever import get_retriever

QUERIES = [
    "12월 LA Port 컨테이너 피크시즌 영향",
    "FBX 지수 상승이 컨테이너 운임에 주는 의미",
    "PMI 50 돌파가 화물 수요에 미치는 영향",
    "Trans-Pacific 해상 컨테이너 시장 점유율 경쟁사",
    "GRI(General Rate Increase) 승인 절차",
]


def main() -> None:
    retriever = get_retriever()
    for qi, q in enumerate(QUERIES, 1):
        print(f"\n### Q{qi}: {q}\n")
        print("| Rank | Source | Route | Type | Preview (80 chars) |")
        print("|---|---|---|---|---|")
        results = retriever.invoke(q)
        for ri, doc in enumerate(results, 1):
            src = doc.metadata.get("source", "?").split("\\")[-1]
            route = doc.metadata.get("route", "?")
            rtype = doc.metadata.get("report_type", "?")
            preview = (
                doc.page_content[:80]
                .replace("\n", " ")
                .replace("|", "\\|")
                .strip()
            )
            print(f"| [{ri}] | {src} | {route} | {rtype} | {preview}... |")


if __name__ == "__main__":
    main()
