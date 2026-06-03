"""Fundamental check — news and sector flags."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import httpx

log = logging.getLogger("dream_maker.fundamental")

NEGATIVE_KEYWORDS = {"sebi action", "fraud", "default", "downgrade", "investigation", "scam"}
POSITIVE_KEYWORDS = {"beat estimates", "record profit", "upgrade", "dividend", "buyback"}

SECTOR_MAP = {
    "RELIANCE": "Energy",
    "TCS": "IT",
    "INFY": "IT",
    "HDFCBANK": "Banking",
    "ICICIBANK": "Banking",
}


@dataclass
class FundamentalContext:
    symbol: str
    sector: str
    news_flags: list[str]
    is_weak: bool
    summary: str


def check_fundamental(symbol: str) -> FundamentalContext:
    base = symbol.split("25")[0].split("FUT")[0].upper()
    sector = SECTOR_MAP.get(base, "Unknown")
    flags: list[str] = []
    text = ""

    try:
        resp = httpx.get(
            "https://news.google.com/rss/search",
            params={"q": f"{base} stock India when:7d", "hl": "en-IN"},
            timeout=10,
            follow_redirects=True,
        )
        resp.raise_for_status()
        titles = re.findall(r"<title>([^<]+)</title>", resp.text)
        text = " ".join(titles[1:6]).lower()
        
        # If no relevant content, try a broader search
        if not text.strip() or base.lower() not in text:
            log.debug("No specific news for %s, trying broader search", base)
            resp = httpx.get(
                "https://news.google.com/rss/search",
                params={"q": f"NSE stock market {sector} when:3d", "hl": "en-IN"},
                timeout=8,
                follow_redirects=True,
            )
            resp.raise_for_status()
            titles = re.findall(r"<title>([^<]+)</title>", resp.text)
            text = " ".join(titles[1:4]).lower()
            
    except Exception as e:
        log.debug("Fundamental news fetch failed for %s: %s", base, e)

    for k in NEGATIVE_KEYWORDS:
        if k in text:
            flags.append(f"negative:{k}")
    for k in POSITIVE_KEYWORDS:
        if k in text:
            flags.append(f"positive:{k}")

    is_weak = any(f.startswith("negative:") for f in flags)
    summary = f"{base} in {sector}; flags={flags or 'none'}"
    return FundamentalContext(symbol=symbol, sector=sector, news_flags=flags, is_weak=is_weak, summary=summary)
