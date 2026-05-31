"""Macro context — risk-on/off classification from headlines."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum

import httpx

log = logging.getLogger("dream_maker.macro")

RISK_ON_KEYWORDS = {"rally", "gains", "record high", "rate cut", "stimulus", "risk-on", "bullish"}
RISK_OFF_KEYWORDS = {"selloff", "crash", "recession", "war", "rate hike", "vix spike", "risk-off", "bearish", "inflation surge"}
RBI_KEYWORDS = {"rbi", "repo rate", "monetary policy"}
FED_KEYWORDS = {"fed", "fomc", "powell"}


class MacroEnvironment(str, Enum):
    RISK_ON = "RISK_ON"
    RISK_OFF = "RISK_OFF"
    NEUTRAL = "NEUTRAL"


@dataclass
class MacroContext:
    environment: MacroEnvironment
    headlines: list[str]
    summary: str
    opposing_long: bool = False
    opposing_short: bool = False


def fetch_headlines(limit: int = 20) -> list[str]:
    headlines: list[str] = []
    try:
        resp = httpx.get(
            "https://news.google.com/rss/search",
            params={"q": "India stock market NIFTY RBI Fed VIX when:1d", "hl": "en-IN"},
            timeout=15,
            follow_redirects=True,
        )
        resp.raise_for_status()
        titles = re.findall(r"<title>([^<]+)</title>", resp.text)
        headlines = [t for t in titles[1 : limit + 1] if t.strip()]
    except Exception as e:
        log.warning("Headline fetch failed: %s", e)
        headlines = ["Market awaits RBI policy", "US futures mixed ahead of data"]
    return headlines


def classify_macro(headlines: list[str] | None = None) -> MacroContext:
    headlines = headlines or fetch_headlines()
    text = " ".join(headlines).lower()
    on_score = sum(1 for k in RISK_ON_KEYWORDS if k in text)
    off_score = sum(1 for k in RISK_OFF_KEYWORDS if k in text)
    rbi = any(k in text for k in RBI_KEYWORDS)
    fed = any(k in text for k in FED_KEYWORDS)

    if off_score > on_score + 1:
        env = MacroEnvironment.RISK_OFF
    elif on_score > off_score + 1:
        env = MacroEnvironment.RISK_ON
    else:
        env = MacroEnvironment.NEUTRAL

    summary_parts = [f"Macro: {env.value}"]
    if rbi:
        summary_parts.append("RBI event active")
    if fed:
        summary_parts.append("Fed/FOMC context")

    return MacroContext(
        environment=env,
        headlines=headlines[:5],
        summary="; ".join(summary_parts),
        opposing_long=env == MacroEnvironment.RISK_OFF,
        opposing_short=env == MacroEnvironment.RISK_ON,
    )
