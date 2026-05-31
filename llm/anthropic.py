"""Anthropic Messages API LLM provider."""

from __future__ import annotations

import json
import logging

import httpx

from config import Config
from llm.base import LLMProvider, REVIEW_SYSTEM_PROMPT, parse_review_json
from llm.openai_compat import OpenAICompatProvider
from models.review import AIReviewRequest, AIReviewResponse

log = logging.getLogger("dream_maker.llm.anthropic")


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._model = cfg.anthropic_model

    def review_trade(self, request: AIReviewRequest) -> AIReviewResponse:
        if not self.cfg.anthropic_api_key:
            from llm.base import RuleBasedLLMProvider
            return RuleBasedLLMProvider().review_trade(request)

        user_msg = OpenAICompatProvider._build_prompt(request)
        try:
            resp = httpx.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.cfg.anthropic_api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": self._model,
                    "max_tokens": 512,
                    "system": REVIEW_SYSTEM_PROMPT,
                    "messages": [{"role": "user", "content": user_msg}],
                },
                timeout=60,
            )
            resp.raise_for_status()
            content = resp.json()["content"][0]["text"]
            return parse_review_json(content)
        except Exception as e:
            log.warning("Anthropic review failed: %s — falling back to rules", e)
            from llm.base import RuleBasedLLMProvider
            return RuleBasedLLMProvider().review_trade(request)
