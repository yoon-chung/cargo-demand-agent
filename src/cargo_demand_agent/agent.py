"""
Cargo Demand Agent: LangGraph StateGraph implementation.

Pipeline (single conditional edge at parse_query):

    START -> parse_query (Solar-Pro intent classifier)
              ├── intent == 'factual'  -> web_search_factual ─┐
              └── intent == 'forecast' -> forecast            │
                                          -> rag              │
                                          -> web_search_forecast
                                          ─────────────────────┤
                                                synthesize ────┘
                                                    -> END

Public API:
    get_app()  -> compiled LangGraph (cached). Call .invoke({"user_query": ...}).
"""
from datetime import date
from functools import lru_cache
import logging
import os
import re

import numpy as np
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_upstage import ChatUpstage
from langgraph.graph import END, START, StateGraph

from src.cargo_demand_agent import prompts
from src.cargo_demand_agent.schemas import AgentState, ForecastResult, IntentClassification
from src.cargo_demand_agent.tools import (
    EXOG_COLS,
    VALID_HORIZONS,
    forecast_demand,
    rag_search_internal_reports,
    web_search_external_indicator,
)

logger = logging.getLogger(__name__)

# SHAP feature -> natural-language phrase for RAG/Web query reformulation.
# Hardcoded since the 17 features are fixed by the model schema.
# Exogenous indicators use English-first phrasing for better web search hits;
# Korean keywords stay in for RAG matching against the ko-corpus reports.
DRIVER_PHRASES: dict[str, str] = {
    "freightos_index": "FBX Freightos Baltic Index trans-pacific spot rate 컨테이너 운임",
    "diesel_price_usd": "US on-highway diesel retail price 디젤 가격",
    "gscpi": "GSCPI Global Supply Chain Pressure Index 글로벌 공급망 압력",
    "us_manuf_pmi": "ISM US manufacturing PMI 제조업 선행지표",
    "peak_season": "trans-pacific cargo peak season 피크시즌",
    **{
        f"teu_log_lag_{i}": f"TEU 처리량 {i}개월 전 historical pattern"
        for i in range(1, 13)
    },
}

# Features that web search can validate (exogenous, externally observable).
# lag features are POLA's own historical TEU values, which web cannot validate.
_WEB_SEARCHABLE_FEATURES: set[str] = set(EXOG_COLS)


@lru_cache(maxsize=1)
def _get_llm() -> ChatUpstage:
    """Cached Solar-Pro client; API key resolved on first call."""
    return ChatUpstage(
        api_key=os.environ["UPSTAGE_API_KEY"],
        model="solar-pro",
        temperature=0.0,  # deterministic intent classification + synthesis
    )


# Korean/English horizon expressions -> integer months.
# Used as a regex fallback when Solar-Pro's structured-output mentions
# the horizon in `reasoning` but leaves the horizon_months field None.
_HORIZON_PATTERNS: list[tuple[re.Pattern[str], int | None]] = [
    (re.compile(r"다음\s*달|next\s+month", re.IGNORECASE), 1),
    (re.compile(r"반\s*년|half\s+year", re.IGNORECASE), 6),
    (re.compile(r"1\s*년|one\s+year|a\s+year", re.IGNORECASE), 12),
    # Generic numeric pattern: capture int and validate against VALID_HORIZONS.
    (re.compile(r"(\d+)\s*개월|(\d+)\s+months?", re.IGNORECASE), None),
]


def _extract_horizon_fallback(text: str) -> int | None:
    """Pattern-match a horizon (1/3/6/12) from natural-language text. None if no match."""
    if not text:
        return None
    for pattern, fixed in _HORIZON_PATTERNS:
        match = pattern.search(text)
        if match is None:
            continue
        if fixed is not None:
            return fixed
        for group in match.groups():
            if group and int(group) in VALID_HORIZONS:
                return int(group)
    return None


# ============================================================
# Node implementations
# ============================================================


def parse_query_node(state: AgentState) -> dict:
    """
    Classify intent + extract horizon. The horizon is interpreted
    model-relative: "다음달" maps to h=1 = "1 month after the last training
    data row", not "1 month after today". This keeps every natural-language
    horizon ({1,3,6,12}) cleanly aligned with a trained model (no snapping
    needed). The trade-off (forecast_period may be stale relative to today)
    is surfaced as a "data staleness gap" line in the forecast context.
    """
    classifier = _get_llm().with_structured_output(IntentClassification)
    result: IntentClassification = classifier.invoke(
        [
            SystemMessage(content=prompts.INTENT_CLASSIFICATION_SYSTEM),
            HumanMessage(content=state.user_query),
        ]
    )

    horizon = result.horizon_months
    if result.intent == "forecast" and horizon is None:
        horizon = (
            _extract_horizon_fallback(state.user_query)
            or _extract_horizon_fallback(result.reasoning)
        )

    logger.info(
        "parse_query: intent=%s horizon=%s reasoning=%s",
        result.intent, horizon, result.reasoning,
    )
    return {"intent": result.intent, "horizon_months": horizon}


def forecast_node(state: AgentState) -> dict:
    """LightGBM + SHAP for LA-Port. Defaults horizon=1 if classifier didn't set it."""
    horizon = state.horizon_months or 1
    result = forecast_demand.invoke({"horizon_months": horizon})
    return {"forecast_result": result, "horizon_months": horizon}


def _build_driver_query(driver_features: list[str]) -> str:
    return " ".join(DRIVER_PHRASES.get(f, f) for f in driver_features)


def rag_node(state: AgentState) -> dict:
    """Hybrid retriever search using reformulated top-3 driver query."""
    if state.forecast_result is None:
        return {"rag_documents": []}
    driver_features = [d.feature for d in state.forecast_result.top_drivers]
    query = _build_driver_query(driver_features)
    docs = rag_search_internal_reports.invoke({"query": query})
    return {"rag_documents": docs}


def web_search_factual_node(state: AgentState) -> dict:
    """Pass the user query directly to Tavily, biased toward recent results."""
    result = web_search_external_indicator.invoke(
        {"query": state.user_query, "time_range": "month"}
    )
    return {"web_results": result.get("results", [])}


def web_search_forecast_node(state: AgentState) -> dict:
    """
    Tavily search for the latest external values of top-3 drivers.

    Filters to exogenous drivers only; lag features are POLA's own historical
    TEU values that no public source can validate, so including them in the
    query just dilutes the result slots. Also adds year + value-seeking
    keywords so Tavily prefers pages with current numeric values over
    definition/explainer pages.
    """
    if state.forecast_result is None:
        return {"web_results": []}
    exog_drivers = [
        d.feature for d in state.forecast_result.top_drivers
        if d.feature in _WEB_SEARCHABLE_FEATURES
    ]
    if not exog_drivers:
        return {"web_results": []}
    current_year = date.today().year
    query = (
        _build_driver_query(exog_drivers)
        + f" current value spot rate {current_year} latest"
    )
    result = web_search_external_indicator.invoke(
        {"query": query, "time_range": "month"}
    )
    return {"web_results": result.get("results", [])}


def synthesize_node(state: AgentState) -> dict:
    """LLM synthesis; prompt + context shape branch on intent."""
    if state.intent == "forecast":
        system_prompt = prompts.SYNTHESIZE_FORECAST_SYSTEM
        user_content = _format_forecast_context(state)
    else:
        system_prompt = prompts.SYNTHESIZE_FACTUAL_SYSTEM
        user_content = _format_factual_context(state)

    response = _get_llm().invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content),
        ]
    )
    return {
        "final_answer": response.content,
        "citations": _extract_citations(state),
    }


# ============================================================
# Context formatters
# ============================================================


def _shift_month(period: str, months_back: int) -> str:
    """Shift a YYYY-MM string backwards by N months. e.g. ('2025-12', 10) -> '2025-02'."""
    year, month = map(int, period.split("-"))
    total = year * 12 + (month - 1) - months_back
    new_year, new_month_idx = divmod(total, 12)
    return f"{new_year:04d}-{new_month_idx + 1:02d}"


def _annotate_feature(feature: str, fr: ForecastResult) -> str:
    """
    Tag a SHAP feature name with the calendar date it represents, so the LLM
    can match RAG citations to the correct vintage instead of grabbing nearby
    keywords from any era.
    """
    if feature.startswith("teu_log_lag_"):
        n = int(feature.rsplit("_", 1)[-1])
        lag_period = _shift_month(fr.latest_observed_period, n - 1)
        return f"{feature} ({lag_period} 시점 처리량 패턴)"
    if feature == "peak_season":
        return f"{feature} (예측 대상월 {fr.forecast_period}의 피크시즌 여부)"
    return feature


def _format_drivers(fr: ForecastResult) -> str:
    """SHAP top-3 as '  - feature (date hint): +X.XX% vs baseline' lines."""
    return "\n".join(
        f"  - {_annotate_feature(d.feature, fr)}: "
        f"{np.expm1(d.shap_value) * 100:+.2f}% vs baseline"
        for d in fr.top_drivers
    )


def _format_rag_excerpts(documents: list[Document]) -> str:
    """RAG chunks as '[title] (period: ...): preview...' lines, top 5."""
    if not documents:
        return "  (no internal context retrieved)"
    return "\n".join(
        f"  - [{d.metadata.get('title') or d.metadata.get('source', '?')}]"
        f" (period: {d.metadata.get('period', 'unspecified')}): "
        f"{d.page_content[:250].strip()}..."
        for d in documents[:5]
    )


def _format_web_results(results: list[dict], max_chars: int = 200) -> str:
    """Tavily results as '[url] title: snippet' lines, top 4."""
    if not results:
        return "  (no web results)"
    return "\n".join(
        f"  - [{r.get('url', '?')}] {r.get('title', '?')}: "
        f"{(r.get('content') or '')[:max_chars].strip()}"
        for r in results[:4]
    )


def _format_baseline_comparisons(fr: ForecastResult) -> str:
    """Latest-observed and year-ago % deltas vs the forecast."""
    mom_pct = (fr.forecast_teu / fr.latest_observed_teu - 1) * 100
    lines = [
        f"Latest observed ({fr.latest_observed_period}): "
        f"{fr.latest_observed_teu:,.0f} TEU  → forecast {mom_pct:+.2f}%"
    ]
    if fr.year_ago_teu is not None:
        yoy_pct = (fr.forecast_teu / fr.year_ago_teu - 1) * 100
        lines.append(
            f"Year ago ({fr.year_ago_period}): {fr.year_ago_teu:,.0f} TEU"
            f"  → forecast {yoy_pct:+.2f}% YoY"
        )
    else:
        lines.append("Year ago: outside training history range.")
    return "\n".join(lines)


def _format_data_gap(latest_observed_period: str) -> str:
    """One-line note: how far today is past the training cutoff."""
    today = date.today()
    last_year, last_month = map(int, latest_observed_period.split("-"))
    gap = (today.year - last_year) * 12 + (today.month - last_month)
    return (
        f"Data gap: {gap} months "
        f"(today {today.strftime('%Y-%m')} vs latest_observed {latest_observed_period}). "
        f"Forecast Period is measured from latest_observed."
    )


def _format_forecast_context(state: AgentState) -> str:
    fr = state.forecast_result
    return (
        f"현재 날짜: {date.today().isoformat()}\n"
        f"User query: {state.user_query}\n"
        f"{_format_data_gap(fr.latest_observed_period)}\n\n"
        f"=== Forecast (LightGBM + SHAP) ===\n"
        f"Route: {fr.route}\n"
        f"Period: {fr.forecast_period}  (horizon = {fr.horizon_months} months)\n"
        f"Forecast TEU: {fr.forecast_teu:,.0f}\n"
        f"{_format_baseline_comparisons(fr)}\n"
        f"Model: {fr.model_name}  (MAPE {fr.model_mape_pct:.2f}%)\n"
        f"Top-3 SHAP drivers:\n{_format_drivers(fr)}\n\n"
        f"=== Internal RAG context (사내 보고서) ===\n"
        f"{_format_rag_excerpts(state.rag_documents)}\n\n"
        f"=== External web search ===\n{_format_web_results(state.web_results)}\n"
    )


def _format_factual_context(state: AgentState) -> str:
    return (
        f"현재 날짜: {date.today().isoformat()}\n"
        f"User query: {state.user_query}\n\n"
        f"=== External web search ===\n"
        f"{_format_web_results(state.web_results, max_chars=300)}\n"
    )


def _extract_citations(state: AgentState) -> list[str]:
    """Compact citation strings for the answer's footer."""
    cites: list[str] = []
    for d in state.rag_documents[:5]:
        title = d.metadata.get("title") or d.metadata.get("source", "?")
        cites.append(f"internal: {title}")
    for r in state.web_results[:4]:
        url = r.get("url")
        if url:
            cites.append(f"web: {url}")
    return cites


# ============================================================
# Routing + graph builder
# ============================================================


def route_by_intent(state: AgentState) -> str:
    """Conditional edge target name. Returns 'factual' or 'forecast'."""
    return state.intent


def build_graph():
    """Compile the StateGraph (do not call directly; use get_app())."""
    graph = StateGraph(AgentState)

    graph.add_node("parse_query", parse_query_node)
    graph.add_node("forecast", forecast_node)
    graph.add_node("rag", rag_node)
    graph.add_node("web_search_factual", web_search_factual_node)
    graph.add_node("web_search_forecast", web_search_forecast_node)
    graph.add_node("synthesize", synthesize_node)

    graph.add_edge(START, "parse_query")
    graph.add_conditional_edges(
        "parse_query",
        route_by_intent,
        {
            "factual": "web_search_factual",
            "forecast": "forecast",
        },
    )
    # forecast path
    graph.add_edge("forecast", "rag")
    graph.add_edge("rag", "web_search_forecast")
    graph.add_edge("web_search_forecast", "synthesize")
    # factual path
    graph.add_edge("web_search_factual", "synthesize")
    # convergence
    graph.add_edge("synthesize", END)

    return graph.compile()


@lru_cache(maxsize=1)
def get_app():
    """Compiled LangGraph app, cached after first build."""
    return build_graph()
