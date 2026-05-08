"""
Pydantic v2 schemas: the contract between Tools and Agent nodes.

Centralizing these lets LangGraph/LangChain auto-derive JSON schemas for
@tool calls and gives a single place to evolve the data shapes.
"""
from typing import Literal, Optional

from langchain_core.documents import Document
from pydantic import BaseModel, ConfigDict, Field


class FeatureContribution(BaseModel):
    """A single feature's SHAP contribution to a forecast."""

    feature: str = Field(
        ...,
        description="Feature name, e.g. 'us_manuf_pmi' or 'teu_log_lag_1'.",
    )
    shap_value: float = Field(
        ...,
        description="Log-scale SHAP value. Positive pushes forecast up, negative down.",
    )


class ForecastResult(BaseModel):
    """Output of `forecast_demand`: prediction + top-3 SHAP drivers."""

    route: str = Field(..., description="Fixed to 'LA-PORT' in Phase 1.")
    horizon_months: int = Field(..., description="1, 3, 6, or 12.")
    forecast_period: str = Field(
        ..., description="Target month in YYYY-MM, e.g. '2026-01'."
    )
    forecast_teu: float = Field(
        ...,
        description="Predicted TEU after expm1 inverse-log transform.",
    )
    latest_observed_teu: float = Field(..., description="Most recent observed TEU.")
    latest_observed_period: str = Field(..., description="YYYY-MM of latest_observed_teu.")
    year_ago_teu: Optional[float] = Field(
        default=None,
        description="Observed TEU 12 months before forecast_period; None if out of range.",
    )
    year_ago_period: Optional[str] = Field(default=None, description="YYYY-MM of year_ago_teu.")
    top_drivers: list[FeatureContribution] = Field(
        ...,
        description="Top-3 features by absolute SHAP value, descending.",
    )
    model_name: str = Field(..., description="e.g. 'lightgbm_h1'.")
    model_mape_pct: float = Field(
        ..., description="Rolling-origin backtest MAPE (%)."
    )


class IntentClassification(BaseModel):
    """Output of `parse_query`, used via `llm.with_structured_output(...)`."""

    intent: Literal["factual", "forecast"] = Field(
        ...,
        description=(
            "'factual' for single-fact lookups (current FBX, latest PMI release); "
            "'forecast' for LA-Port demand prediction + driver explanation."
        ),
    )
    horizon_months: Optional[int] = Field(
        default=None,
        description="One of 1, 3, 6, 12 if extractable from the query, else null.",
    )
    reasoning: str = Field(
        ..., description="One-line rationale for the classification."
    )


class AgentState(BaseModel):
    """
    LangGraph state for the 6-node pipeline + 1 conditional branch.

    Flow:
        parse_query (Solar-Pro intent classifier)
          ├── intent == 'factual'  -> web_search_factual  -> synthesize
          └── intent == 'forecast' -> forecast -> rag -> web_search_forecast -> synthesize

    The conditional edge in agent.py routes from parse_query based on `intent`.
    Fields populated by later nodes start as None / empty list.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # ---- Input ----
    user_query: str = Field(..., description="Raw user query.")

    # ---- parse_query node output ----
    horizon_months: Optional[int] = Field(
        default=None,
        description=(
            "Horizon in {1,3,6,12}, interpreted model-relative: 'N months "
            "after the last training-data row', not 'N months after today'. "
            "None for factual queries."
        ),
    )
    intent: Literal["factual", "forecast"] = Field(
        default="forecast",
        description=(
            "Classified by Solar-Pro at parse_query. 'factual' short-circuits "
            "to web_search_factual only; 'forecast' runs the full 5-step path."
        ),
    )

    # ---- forecast_demand node output (None when intent == 'factual') ----
    forecast_result: Optional[ForecastResult] = None

    # ---- rag_search node output (empty when intent == 'factual') ----
    rag_documents: list[Document] = Field(default_factory=list)

    # ---- web_search node output (populated for both intents) ----
    web_results: list[dict] = Field(default_factory=list)

    # ---- synthesize_answer node output ----
    final_answer: Optional[str] = None
    citations: list[str] = Field(default_factory=list)
