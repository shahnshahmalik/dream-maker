"""Yahoo Finance data provider — used as fallback when broker chart APIs fail.

Maps dream-maker symbols (NIFTY26JUN23350CE, NIFTY50IDX) to Yahoo tickers
(^NSEI, ^NSEBANK) and returns OHLCV candles in the standard format.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

try:
    import yfinance as yf
except ImportError:  # optional fallback provider — degrade gracefully
    yf = None

from models.orders import OHLCV

log = logging.getLogger("dream_maker.yahoo")

IST = ZoneInfo("Asia/Kolkata")

# Map dream-maker underlying bases → Yahoo Finance ticker
_SYMBOL_MAP: dict[str, str] = {
    "NIFTY": "^NSEI",
    "BANKNIFTY": "^NSEBANK",
    "SENSEX": "^BSESN",
    "FINNIFTY": "NIFTY_FIN_SERVICE.NS",
}

# timeframe → yfinance interval
_INTERVAL_MAP: dict[str, str] = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "1h": "1h",
    "4h": "1h",    # yfinance doesn't have 4h — use 1h and trim later
    "1d": "1d",
    "1w": "1wk",
}

# timeframe → yfinance period (how far back to fetch)
_PERIOD_MAP: dict[str, str] = {
    "1m": "7d",
    "5m": "7d",
    "15m": "7d",
    "1h": "1mo",
    "4h": "1mo",
    "1d": "1y",
    "1w": "2y",
}


def _resolve_ticker(symbol: str) -> str | None:
    """Map a dream-maker symbol to a Yahoo Finance ticker.

    Handles:
        - NIFTY26JUN23350CE  → ^NSEI  (options → underlying)
        - NIFTY50IDX          → ^NSEI
        - BANKNIFTY26JUN52000CE → ^NSEBANK
        - DIXON, JUBLFOOD etc  → DIXON.NS (stock underlying)
    """
    upper = symbol.upper().replace(" ", "")

    # Try index mappings first — check longer keys first to avoid
    # "NIFTY" matching before "BANKNIFTY"
    for base, ticker in sorted(_SYMBOL_MAP.items(), key=lambda x: -len(x[0])):
        if base in upper:
            return ticker

    # If not an index, try as a stock (append .NS for NSE)
    # Strip option suffix (strike+CE/PE) to get underlying
    import re
    stock_match = re.match(r"^([A-Z]+)\d+(CE|PE)", upper)
    if stock_match:
        return f"{stock_match.group(1)}.NS"

    return None


def fetch_ohlcv(symbol: str, timeframe: str, limit: int) -> list[OHLCV]:
    """Fetch OHLCV candles from Yahoo Finance.

    Args:
        symbol: Dream-maker symbol (e.g. NIFTY26JUN23350CE, NIFTY50IDX)
        timeframe: One of 1m/5m/15m/1h/4h/1d/1w
        limit: Max candles to return

    Returns:
        List of OHLCV candles, or empty list on failure.
    """
    if yf is None:
        log.debug("yfinance not installed — Yahoo fallback unavailable")
        return []

    ticker = _resolve_ticker(symbol)
    if ticker is None:
        log.debug("No Yahoo ticker mapping for %s", symbol)
        return []

    yf_interval = _INTERVAL_MAP.get(timeframe)
    yf_period = _PERIOD_MAP.get(timeframe, "1mo")

    if yf_interval is None:
        log.debug("Unsupported timeframe %s for Yahoo", timeframe)
        return []

    try:
        t = yf.Ticker(ticker)
        df = t.history(period=yf_period, interval=yf_interval)

        if df.empty:
            log.debug("Yahoo returned empty data for %s (%s)", ticker, timeframe)
            return []

        # Convert DataFrame rows → OHLCV list
        candles: list[OHLCV] = []
        for idx, row in df.iterrows():
            ts = idx.to_pydatetime()
            # Ensure timezone-aware (IST)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=IST)
            candles.append(OHLCV(
                timestamp=ts,
                open=float(row["Open"]),
                high=float(row["High"]),
                low=float(row["Low"]),
                close=float(row["Close"]),
                volume=int(row["Volume"]),
            ))

        # Trim to requested limit
        if len(candles) > limit:
            candles = candles[-limit:]

        log.debug(
            "Yahoo: %s (%s) → %s %s → %d candles",
            symbol, ticker, timeframe, yf_period, len(candles),
        )
        return candles

    except Exception as e:
        log.warning("Yahoo fetch failed for %s (%s): %s", symbol, ticker, e)
        return []
