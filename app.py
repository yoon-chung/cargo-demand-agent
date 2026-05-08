"""
Streamlit demo UI for the cargo demand agent.

Run from project root:
    streamlit run app.py
"""
import logging
import re

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.WARNING)


def _sanitize_markdown(text: str) -> str:
    """Replace lone '~' (e.g. '2025~2026') with en-dash to avoid GFM strikethrough."""
    return re.sub(r"(?<!~)~(?!~)", "–", text)


@st.cache_resource(show_spinner="Building agent (BGE-M3 + Chroma + LangGraph)...")
def _load_app():
    """Compile the agent once per Streamlit process; cached across reruns."""
    from src.cargo_demand_agent.agent import get_app

    return get_app()


@st.cache_data(show_spinner=False)
def _pipeline_png_bytes() -> bytes:
    """LangGraph as PNG via mermaid.ink, cached after first call."""
    return _load_app().get_graph().draw_mermaid_png()


# ============================================================
# Page setup
# ============================================================
st.set_page_config(page_title="Cargo Demand Agent", layout="wide")
st.title("Cargo Demand Agent")
st.caption(
    "LightGBM + SHAP + Hybrid RAG (BGE-M3 dense + BM25 Kiwi) + "
    "Web Search + LLM " \
    "orchestrated by LangGraph"
)

# ============================================================
# Sidebar: example queries + pipeline note
# ============================================================
with st.sidebar:
    st.header("예시 질문")
    EXAMPLES = {
        "factual: 디젤 가격": "최근 미국 디젤 가격은?",
        "factual: ISM PMI": "최근 ISM 제조업 PMI 발표값은?",
        "forecast: 다음달 LA Port": "다음달 LA Port 컨테이너 처리량과 드라이버 분석",
        "forecast: 3개월 후 LA Port": "3개월 후 LA Port 처리량 전망",
    }
    for label, q in EXAMPLES.items():
        if st.button(label, use_container_width=True):
            st.session_state["query_input"] = q

    st.divider()
    st.markdown("**Persona**")
    st.caption(
        "LA 항만 컨테이너 물동량 수요예측 전문가"
    )

    st.divider()
    st.markdown("**데이터 시점**")
    st.caption(
        "학습 데이터는 2025-12까지. '다음달/3개월 후' 등 horizon은 "
        "학습 데이터 기준 1/3/6/12개월 후 시점을 의미."
    )

    st.divider()
    st.markdown("**Pipeline**")
    st.caption(
        "parse_query → "
        "(factual: web) / (forecast: forecast → rag → web) → "
        "synthesize"
    )

# ============================================================
# Main: query input + run
# ============================================================
query = st.text_input(
    "질문을 입력하세요",
    value=st.session_state.get("query_input", ""),
    placeholder="예: 다음달 LA 항만 컨테이너 처리량과 드라이버 분석",
)

def _render_forecast_section(state: dict) -> None:
    """Forecast metrics + Top-3 SHAP driver bar chart. Skipped for factual intent."""
    fr = state.get("forecast_result")
    if fr is None:
        return

    st.subheader("Forecast")
    f1, f2, f3, f4 = st.columns(4)
    f1.metric("Horizon", f"{state.get('horizon_months') or '-'} months")
    f2.metric("예측 TEU", f"{fr.forecast_teu:,.0f}")
    f3.metric("Period", fr.forecast_period)
    f4.metric("Validation MAPE", f"{fr.model_mape_pct:.2f}%")

    st.markdown(
        "**Top-3 드라이버**: 각 요인이 이번 예측 물동량을 "
        "상향(+) / 하향(−)시킨 정도 (%, baseline 대비)"
    )
    df = pd.DataFrame(
        [
            {
                "feature": d.feature,
                "% impact": np.expm1(d.shap_value) * 100,
            }
            for d in fr.top_drivers
        ]
    )
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("% impact:Q", title="% impact (vs baseline)"),
            y=alt.Y("feature:N", sort="-x", title=None),
            color=alt.condition(
                alt.datum["% impact"] > 0,
                alt.value("#2ca02c"),
                alt.value("#d62728"),
            ),
        )
        .properties(height=240)
    )
    chart_col, _ = st.columns([3, 1])  # chart at ~75% width
    chart_col.altair_chart(chart, use_container_width=True)
    st.caption(
        f"**baseline** = {fr.horizon_months}개월 후 예측 모델이 "
        "학습 데이터(LA항만 물동량 2009~2025년)에 대해 산출하는 평균 예측치."
    )


def _render_citations(citations: list[str]) -> None:
    """Collapsible citation list, hidden when empty."""
    if not citations:
        return
    with st.expander(f"Citations ({len(citations)})"):
        for c in citations:
            st.write(f"- {c}")


def _render_pipeline_expander(state: dict, app) -> None:
    """LangGraph pipeline diagram (PNG)."""
    with st.expander("Pipeline (LangGraph)"):
        st.caption(f"intent = `{state.get('intent') or '-'}`")
        img_col, _ = st.columns([3, 2])  # diagram occupies ~60% width
        with img_col:
            try:
                st.image(_pipeline_png_bytes(), use_container_width=True)
            except Exception as exc:
                st.caption(f"PNG render failed ({exc}); raw source:")
                st.code(app.get_graph().draw_mermaid(), language="mermaid")


if st.button("Agent 실행", type="primary", disabled=not query):
    app = _load_app()
    with st.spinner("LangGraph pipeline 실행 중..."):
        state = app.invoke({"user_query": query})

    _render_forecast_section(state)

    st.subheader("Answer")
    st.markdown(_sanitize_markdown(state.get("final_answer") or "_(no answer)_"))

    _render_citations(state.get("citations", []))
    _render_pipeline_expander(state, app)
