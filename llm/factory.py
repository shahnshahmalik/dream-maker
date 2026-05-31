"""LLM provider factory."""

from __future__ import annotations

from config import Config
from llm.anthropic import AnthropicProvider
from llm.base import LLMProvider, RuleBasedLLMProvider
from llm.openai_compat import OpenAICompatProvider


def get_llm(cfg: Config) -> LLMProvider:
    if cfg.active_llm == "openai_compat":
        return OpenAICompatProvider(cfg)
    if cfg.active_llm == "anthropic":
        return AnthropicProvider(cfg)
    if cfg.active_llm in {"none", "rules"}:
        return RuleBasedLLMProvider()
    raise ValueError(f"Unknown LLM provider: {cfg.active_llm}")
