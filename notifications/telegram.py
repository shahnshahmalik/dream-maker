"""Telegram Bot API notifier."""

from __future__ import annotations

import logging

import httpx

log = logging.getLogger("dream_maker.telegram")


class TelegramNotifier:
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token.strip()
        self.chat_id = chat_id.strip()

    @property
    def enabled(self) -> bool:
        return bool(self.bot_token and self.chat_id)

    def send(self, text: str) -> None:
        if not self.enabled:
            return
        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            resp = httpx.post(
                url,
                json={
                    "chat_id": self.chat_id,
                    "text": text[:4096],
                    "disable_web_page_preview": True,
                },
                timeout=15.0,
            )
            resp.raise_for_status()
        except Exception as e:
            log.warning("Telegram notify failed: %s", e)
