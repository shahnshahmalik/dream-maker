"""Shared OpenAI-compatible chat completion helper."""

from __future__ import annotations

import logging

import httpx

log = logging.getLogger("dream_maker.llm.chat")


def chat_completion(
    *,
    base_url: str,
    api_key: str,
    model: str,
    system: str,
    user: str,
    temperature: float = 0.2,
    timeout: float = 60.0,
) -> str:
    resp = httpx.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def chat_config_for_provider(cfg, llm) -> tuple[str, str, str] | None:
    """Return (base_url, api_key, model) for chat-capable providers."""
    name = getattr(llm, "name", "")
    if name == "openai_compat":
        return cfg.openai_base_url, cfg.openai_api_key, cfg.openai_model
    if name == "deepseek":
        return cfg.deepseek_base_url, cfg.deepseek_api_key, cfg.deepseek_model
    return None
