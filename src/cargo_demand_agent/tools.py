"""
Tools for the cargo demand agent, exposed to LangGraph via @tool decorators.

    forecast_demand:                 LightGBM h={1,3,6,12} prediction + SHAP top-3 drivers.
    web_search_external_indicator:   Tavily wrap for real-time external facts.
    rag_search_internal_reports:     Hybrid retriever wrap for internal domain context.
"""
from functools import lru_cache
from pathlib import Path
from typing import Optional
import json
import logging
import os

import joblib
import numpy as np
import pandas as pd
import shap
from langchain_core.documents import Document
from langchain_core.tools import tool
from tavily import TavilyClient

from src.cargo_demand_agent.schemas import FeatureContribution, ForecastResult

logger = logging.getLogger(__name__)

# Resolve paths relative to the project root (this file's grandparent of the
# package dir), so the agent works from any CWD (notebook in practice/, etc.).
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = _PROJECT_ROOT / "data" / "models"
HISTORY_CSV = _PROJECT_ROOT / "data" / "pola_teu.csv"
EXOG_COLS = [
    "freightos_index",
    "diesel_price_usd",
    "gscpi",
    "us_manuf_pmi",
    "peak_season",
]
LAG_COLS = [f"teu_log_lag_{i}" for i in range(1, 13)]
FEATURE_NAMES = LAG_COLS + EXOG_COLS  # 17 features in this exact order
VALID_HORIZONS = (1, 3, 6, 12)

# Phase 1 fixed scope: LA-Port (POLA) only. Multi-route is phase 2.
FIXED_ROUTE = "LA-PORT"


@lru_cache(maxsize=1)
def _load_metadata() -> dict:
    with (MODELS_DIR / "metadata.json").open(encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=4)
def _load_model(horizon: int):
    """Load and cache the LightGBM model for one horizon."""
    return joblib.load(MODELS_DIR / f"lightgbm_h{horizon}.pkl")


@lru_cache(maxsize=1)
def _load_scaler():
    return joblib.load(MODELS_DIR / "exog_scaler.pkl")


@lru_cache(maxsize=1)
def _load_history() -> pd.DataFrame:
    df = pd.read_csv(HISTORY_CSV, encoding="utf-8")
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def _peak_season_for_month(month: int) -> int:
    """Peak season flag matching the training-set encoding: 1 for Aug-Oct, else 0."""
    return int(8 <= month <= 10)


def _compute_baseline_comparisons(
    history: pd.DataFrame, target_period: pd.Timestamp
) -> tuple[float, str, Optional[float], Optional[str]]:
    """Pull the latest observed TEU and the year-ago TEU (if in range) for synthesis."""
    last_row = history.iloc[-1]
    latest_teu = float(last_row["total_teu"])
    latest_period = pd.Timestamp(last_row["date"]).strftime("%Y-%m")

    year_ago_date = target_period - pd.DateOffset(months=12)
    match = history[history["date"] == year_ago_date]
    if len(match) > 0:
        return (
            latest_teu, latest_period,
            float(match.iloc[0]["total_teu"]),
            year_ago_date.strftime("%Y-%m"),
        )
    return latest_teu, latest_period, None, None


def _build_feature_row(
    horizon_months: int,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    """
    Build the 1x17 feature DataFrame for prediction and resolve target period.

    The "current" timestamp is the last row of pola_teu.csv. Exog values from
    that row are taken as-is (Web Search override is a future enhancement).
    peak_season is replaced by the value appropriate for the target month.
    """
    history = _load_history()
    last_row = history.iloc[-1]
    target_period = last_row["date"] + pd.DateOffset(months=horizon_months)

    # Lags: lag_1 = most recent month, lag_12 = 12 months ago.
    last_12_teu = history.iloc[-12:]["total_teu"].to_numpy()
    log_lags = np.log1p(last_12_teu)[::-1]
    lag_dict = dict(zip(LAG_COLS, log_lags))

    # Exog: current values, then override peak_season for the target month.
    exog_raw = {col: float(last_row[col]) for col in EXOG_COLS}
    exog_raw["peak_season"] = _peak_season_for_month(target_period.month)

    scaler = _load_scaler()
    exog_array = np.array([[exog_raw[c] for c in EXOG_COLS]])
    exog_scaled = scaler.transform(exog_array)[0]
    exog_dict = dict(zip(EXOG_COLS, exog_scaled))

    feature_row = pd.DataFrame([{**lag_dict, **exog_dict}])[FEATURE_NAMES]
    return feature_row, target_period


@tool
def forecast_demand(horizon_months: int) -> ForecastResult:
    """
    Predict LA-Port (POLA) monthly TEU using LightGBM (one model per horizon),
    then extract the top-3 SHAP drivers. horizon_months must be in {1,3,6,12}.
    """
    if horizon_months not in VALID_HORIZONS:
        raise ValueError(
            f"horizon_months must be one of {VALID_HORIZONS}, "
            f"got {horizon_months}"
        )

    metadata = _load_metadata()
    model = _load_model(horizon_months)
    feature_row, target_period = _build_feature_row(horizon_months)

    (latest_observed_teu, latest_observed_period,
     year_ago_teu, year_ago_period) = _compute_baseline_comparisons(
        _load_history(), target_period,
    )

    log_pred = float(model.predict(feature_row)[0])
    forecast_teu = float(np.expm1(log_pred))

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(feature_row)[0]
    contributions = sorted(
        (
            FeatureContribution(feature=f, shap_value=float(v))
            for f, v in zip(FEATURE_NAMES, shap_values)
        ),
        key=lambda c: abs(c.shap_value),
        reverse=True,
    )
    top3 = contributions[:3]

    mape_pct = metadata["rolling_origin_backtest"][
        f"h{horizon_months}"
    ]["MAPE_pct"]

    logger.info(
        "Forecast LA-PORT h=%d: TEU=%.0f (log=%.3f), drivers=%s",
        horizon_months,
        forecast_teu,
        log_pred,
        [(c.feature, round(c.shap_value, 3)) for c in top3],
    )
    return ForecastResult(
        route=FIXED_ROUTE,
        horizon_months=horizon_months,
        forecast_period=target_period.strftime("%Y-%m"),
        forecast_teu=forecast_teu,
        latest_observed_teu=latest_observed_teu,
        latest_observed_period=latest_observed_period,
        year_ago_teu=year_ago_teu,
        year_ago_period=year_ago_period,
        top_drivers=top3,
        model_name=f"lightgbm_h{horizon_months}",
        model_mape_pct=mape_pct,
    )


# ---------- Web Search (Tavily) ----------

DEFAULT_WEB_MAX_RESULTS = 4


@lru_cache(maxsize=1)
def _tavily_client() -> TavilyClient:
    """Cached Tavily client; API key resolved at first call from env."""
    return TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


@tool
def web_search_external_indicator(
    query: str, time_range: Optional[str] = None,
) -> dict:
    """
    Tavily search for short-lived facts (latest FBX print, ISM PMI release).
    `time_range` ("day"/"week"/"month"/"year") biases toward recent results.
    """
    kwargs = {
        "query": query,
        "max_results": DEFAULT_WEB_MAX_RESULTS,
        "search_depth": "basic",
    }
    if time_range:
        kwargs["time_range"] = time_range
    try:
        response = _tavily_client().search(**kwargs)
        return {
            "query": query,
            "results": response.get("results", []),
            "answer": response.get("answer"),
        }
    except Exception as exc:
        logger.warning("Tavily search failed for %r: %s", query, exc)
        return {"query": query, "results": [], "answer": None, "error": str(exc)}


# ---------- Internal RAG (hybrid retriever) ----------


@lru_cache(maxsize=1)
def _cached_retriever():
    """
    Cache the hybrid retriever (dense BGE-M3 + sparse BM25 + Kiwi) after the
    first call so the in-memory BM25 index isn't rebuilt on each invocation
    within the same process.
    """
    from src.cargo_demand_agent.retriever import get_retriever

    return get_retriever()


@tool
def rag_search_internal_reports(query: str) -> list[Document]:
    """
    Hybrid search (BGE-M3 dense + BM25 sparse + Kiwi tokenizer, fused via RRF)
    over the internal report corpus. Returns Documents with metadata preserved
    for citation in the final answer.
    """
    try:
        return _cached_retriever().invoke(query)
    except Exception as exc:
        logger.warning("RAG search failed for %r: %s", query, exc)
        return []
