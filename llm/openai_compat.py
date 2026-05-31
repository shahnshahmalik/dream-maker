"""OpenAI-compatible LLM provider."""

from __future__ import annotations

import json
import logging

from config import Config
from llm.base import LLMProvider, REVIEW_SYSTEM_PROMPT, RuleBasedLLMProvider, parse_review_json
from llm.chat_completions import chat_completion
from models.review import AIReviewRequest, AIReviewResponse

log = logging.getLogger("dream_maker.llm.openai")


class OpenAICompatProvider(LLMProvider):
    name = "openai_compat"

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._base = cfg.openai_base_url.rstrip("/")
        self._model = cfg.openai_model

    def review_trade(self, request: AIReviewRequest) -> AIReviewResponse:
        if not self.cfg.openai_api_key:
            return RuleBasedLLMProvider().review_trade(request)

        user_msg = self._build_prompt(request)
        try:
            content = chat_completion(
                base_url=self._base,
                api_key=self.cfg.openai_api_key,
                model=self._model,
                system=REVIEW_SYSTEM_PROMPT,
                user=user_msg,
                temperature=0.2,
            )
            return parse_review_json(content)
        except Exception as e:
            log.warning("OpenAI review failed: %s — falling back to rules", e)
            return RuleBasedLLMProvider().review_trade(request)

    @staticmethod
    def _build_prompt(request: AIReviewRequest) -> str:
        c15 = [{"c": c.close, "h": c.high, "l": c.low} for c in request.candles_15m[-20:]]
        c1h = [{"c": c.close} for c in request.candles_1h[-20:]]
        return json.dumps({
            "symbol": request.symbol,
            "direction": request.direction,
            "original_rationale": request.original_rationale,
            "entry_price": request.entry_price,
            "current_price": request.current_price,
            "stop_loss": request.stop_loss,
            "macro_env": request.macro_env,
            "divergence_reasons": request.divergence_reasons,
            "candles_15m": c15,
            "candles_1h": c1h,
            "question": "Should we hold, tighten SL, partially exit, or close immediately?",
        })
