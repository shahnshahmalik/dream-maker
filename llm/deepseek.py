"""DeepSeek LLM provider (OpenAI-compatible API)."""

from __future__ import annotations

import logging

from config import Config
from llm.base import LLMProvider, REVIEW_SYSTEM_PROMPT, RuleBasedLLMProvider, parse_review_json
from llm.chat_completions import chat_completion
from llm.openai_compat import OpenAICompatProvider
from models.review import AIReviewRequest, AIReviewResponse

log = logging.getLogger("dream_maker.llm.deepseek")


class DeepSeekProvider(LLMProvider):
    name = "deepseek"

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._base = cfg.deepseek_base_url.rstrip("/")
        self._model = cfg.deepseek_model

    def review_trade(self, request: AIReviewRequest) -> AIReviewResponse:
        if not self.cfg.deepseek_api_key:
            return RuleBasedLLMProvider().review_trade(request)

        user_msg = OpenAICompatProvider._build_prompt(request)
        try:
            content = chat_completion(
                base_url=self._base,
                api_key=self.cfg.deepseek_api_key,
                model=self._model,
                system=REVIEW_SYSTEM_PROMPT,
                user=user_msg,
                temperature=0.2,
            )
            return parse_review_json(content)
        except Exception as e:
            log.warning("DeepSeek review failed: %s — falling back to rules", e)
            return RuleBasedLLMProvider().review_trade(request)
