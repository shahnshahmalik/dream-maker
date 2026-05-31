"""OpenAI-compatible LLM provider."""

from __future__ import annotations

import json
import logging

import httpx

from config import Config
from llm.base import LLMProvider, REVIEW_SYSTEM_PROMPT, parse_review_json
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
            from llm.base import RuleBasedLLMProvider
            return RuleBasedLLMProvider().review_trade(request)

        user_msg = self._build_prompt(request)
        try:
            resp = httpx.post(
                f"{self._base}/chat/completions",
                headers={"Authorization": f"Bearer {self.cfg.openai_api_key}"},
                json={
                    "model": self._model,
                    "messages": [
                        {"role": "system", "content": REVIEW_SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    "temperature": 0.2,
                },
                timeout=60,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            return parse_review_json(content)
        except Exception as e:
            log.warning("OpenAI review failed: %s — falling back to rules", e)
            from llm.base import RuleBasedLLMProvider
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
