"""
LLM prompts for the cargo demand agent's LLM-using nodes:

    parse_query        -> INTENT_CLASSIFICATION_SYSTEM
    synthesize_answer  -> SYNTHESIZE_FORECAST_SYSTEM   (forecast intent)
                       -> SYNTHESIZE_FACTUAL_SYSTEM    (factual intent)

All prompt text lives in `i18n.py` and is resolved at module-import time
based on `APP_LOCALE`. Phase 2 English version flips that env var; no
prompt-side code changes needed.
"""
from src.cargo_demand_agent.i18n import t

INTENT_CLASSIFICATION_SYSTEM: str = t("intent_classification_system")
SYNTHESIZE_FORECAST_SYSTEM: str = t("synthesize_forecast_system")
SYNTHESIZE_FACTUAL_SYSTEM: str = t("synthesize_factual_system")
