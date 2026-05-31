"""Technical analysis — HTF/LTF trend, EMA, S/R, projections."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd

from models.orders import OHLCV
from models.trade_plan import TradeDirection


class Trend(str, Enum):
    UPTREND = "uptrend"
    DOWNTREND = "downtrend"
    RANGE = "range"


@dataclass
class TechnicalContext:
    htf_trend: Trend
    ltf_trend: Trend
    above_200ema: bool
    support: float
    resistance: float
    ltf_aligned: bool
    entry: float
    stop_loss: float
    tp1: float
    tp2: float
    rr_ratio: float
    direction: TradeDirection
    bias_source: str
    signal_strength: float = 0.0


def _ohlcv_to_df(candles: list[OHLCV]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [c.open for c in candles],
            "high": [c.high for c in candles],
            "low": [c.low for c in candles],
            "close": [c.close for c in candles],
            "volume": [c.volume for c in candles],
        }
    )


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def detect_trend(df: pd.DataFrame) -> Trend:
    if len(df) < 10:
        return Trend.RANGE
    highs = df["high"].tail(5)
    lows = df["low"].tail(5)
    if highs.is_monotonic_increasing and lows.is_monotonic_increasing:
        return Trend.UPTREND
    if highs.is_monotonic_decreasing and lows.is_monotonic_decreasing:
        return Trend.DOWNTREND
    return Trend.RANGE


def analyze_technical(
    htf_candles: list[OHLCV],
    ltf_candles: list[OHLCV],
    *,
    min_rr: float = 2.0,
) -> TechnicalContext | None:
    if len(htf_candles) < 20 or len(ltf_candles) < 20:
        return None

    htf = _ohlcv_to_df(htf_candles)
    ltf = _ohlcv_to_df(ltf_candles)
    htf_trend = detect_trend(htf)
    ltf_trend = detect_trend(ltf)
    ema200 = _ema(htf["close"], min(200, len(htf)))
    last_close = float(htf["close"].iloc[-1])
    above_200 = last_close > float(ema200.iloc[-1]) if len(ema200) else True

    support = float(htf["low"].tail(20).min())
    resistance = float(htf["high"].tail(20).max())
    entry = float(ltf["close"].iloc[-1])

    if htf_trend == Trend.UPTREND and above_200:
        direction = TradeDirection.LONG
        stop_loss = support
        tp1 = entry + (entry - stop_loss) * min_rr
        tp2 = entry + (entry - stop_loss) * (min_rr * 1.5)
        bias = "HTF uptrend + above 200 EMA"
    elif htf_trend == Trend.DOWNTREND and not above_200:
        direction = TradeDirection.SHORT
        stop_loss = resistance
        tp1 = entry - (stop_loss - entry) * min_rr
        tp2 = entry - (stop_loss - entry) * (min_rr * 1.5)
        bias = "HTF downtrend + below 200 EMA"
    else:
        return None

    sl_dist = abs(entry - stop_loss)
    tp_dist = abs(tp1 - entry)
    rr = tp_dist / sl_dist if sl_dist else 0
    ltf_aligned = (
        (direction == TradeDirection.LONG and ltf_trend == Trend.UPTREND)
        or (direction == TradeDirection.SHORT and ltf_trend == Trend.DOWNTREND)
    )
    if not ltf_aligned:
        return None

    strength = 0.0
    strength += 0.35
    strength += 0.25 if ltf_aligned else 0.0
    strength += 0.20 if rr >= min_rr else 0.0
    strength += 0.20 if (direction == TradeDirection.LONG and above_200) or (
        direction == TradeDirection.SHORT and not above_200
    ) else 0.0

    return TechnicalContext(
        htf_trend=htf_trend,
        ltf_trend=ltf_trend,
        above_200ema=above_200,
        support=support,
        resistance=resistance,
        ltf_aligned=ltf_aligned,
        entry=entry,
        stop_loss=stop_loss,
        tp1=tp1,
        tp2=tp2,
        rr_ratio=rr,
        direction=direction,
        bias_source=bias,
        signal_strength=min(1.0, strength),
    )


def check_ltf_structure_break(
    candles: list[OHLCV],
    direction: TradeDirection,
) -> bool:
    if len(candles) < 5:
        return False
    df = _ohlcv_to_df(candles)
    recent = df.tail(5)
    if direction == TradeDirection.LONG:
        return float(recent["low"].iloc[-1]) < float(recent["low"].iloc[-3])
    return float(recent["high"].iloc[-1]) > float(recent["high"].iloc[-3])
