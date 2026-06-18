"""Slack incoming-webhook notifier.

Uses Slack's Incoming Webhooks — no OAuth, no bot token needed.
Create one at: https://api.slack.com/messaging/webhooks
Set SLACK_WEBHOOK_URL in .env.
"""

from __future__ import annotations

import logging

import httpx

log = logging.getLogger("dream_maker.slack")


class SlackNotifier:
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url.strip()

    @property
    def enabled(self) -> bool:
        return bool(self.webhook_url)

    def send(self, text: str) -> None:
        """Send a plain-text message to the Slack channel via webhook."""
        if not self.enabled:
            return
        try:
            resp = httpx.post(
                self.webhook_url,
                json={"text": text},
                timeout=15.0,
            )
            resp.raise_for_status()
        except Exception as exc:
            log.warning("Slack notify failed: %s", exc)
