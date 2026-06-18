"""Signal messenger notifier via CallMeBot API.

Zero-infrastructure Signal notifications. Registration:
  1. Open Signal, send "I allow callmebot to send me messages"
     to +34 693 96 02 64
  2. They reply with your API key.
  3. Set in .env:
       SIGNAL_PHONE=+1234567890    (your Signal-registered phone)
       SIGNAL_API_KEY=<key>

API reference: https://www.callmebot.com/blog/free-api-signal-send-messages/
"""

from __future__ import annotations

import logging

import httpx

log = logging.getLogger("dream_maker.signal")

_CALLMEBOT_URL = "https://signal.callmebot.com/signal/send.php"
_MAX_CHARS = 1600  # CallMeBot message limit


class SignalNotifier:
    def __init__(self, phone: str, api_key: str):
        self.phone = phone.strip()
        self.api_key = api_key.strip()

    @property
    def enabled(self) -> bool:
        return bool(self.phone and self.api_key)

    def send(self, text: str) -> None:
        """Send a plain-text message via CallMeBot Signal API."""
        if not self.enabled:
            return
        try:
            resp = httpx.get(
                _CALLMEBOT_URL,
                params={
                    "phone": self.phone,
                    "apikey": self.api_key,
                    "text": text[:_MAX_CHARS],
                },
                timeout=15.0,
            )
            resp.raise_for_status()
        except Exception as exc:
            log.warning("Signal notify failed: %s", exc)
