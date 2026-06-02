"""Technical analysis — swing setups, momentum scalps, trailing helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import pandas as pd

from models.orders import OHLCV
from models.trade_plan import TradeDirection


class Trend(str, Enum):
    UPTREND = "uptrend"
    DOWNTREND = "downtrend"
    RANGE = "range"


class SetupType(str, Enum):
    SWING = "swing"
    MOMENTUM_SCALP = "momentum_scalp"


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
    setup_type: SetupType = SetupType.SWING
    confirmations: list[str] = field(default_factory=list)


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


def momentum_confirmation(
    ltf_candles: list[OHLCV],
    direction: TradeDirection,
    *,
    min_confirmations: int = 2,
) -> tuple[bool, list[str]]:
    """Require multiple independent momentum signals before a scalp entry."""
    if len(ltf_candles) < 12:
        return False, []

    ltf = _ohlcv_to_df(ltf_candles)
    ema9 = _ema(ltf["close"], 9)
    ema21 = _ema(ltf["close"], 21)
    last = ltf.iloc[-1]
    prev = ltf.tail(6).iloc[:-1]
    avg_vol = float(ltf["volume"].tail(10).mean()) or 1.0
    reasons: list[str] = []

    if direction == TradeDirection.LONG:
        if float(ema9.iloc[-1]) > float(ema21.iloc[-1]):
            reasons.append("ema9>ema21")
        if float(last["close"]) > float(last["open"]):
            reasons.append("bullish_candle")
        if float(last["close"]) > float(prev["high"].max()):
            reasons.append("micro_breakout")
        if float(last["volume"]) >= avg_vol * 0.85:
            reasons.append("volume_confirm")
        if float(ltf["close"].iloc[-1]) > float(ltf["close"].iloc[-2]):
            reasons.append("momentum_tick")
    else:
        if float(ema9.iloc[-1]) < float(ema21.iloc[-1]):
            reasons.append("ema9<ema21")
        if float(last["close"]) < float(last["open"]):
            reasons.append("bearish_candle")
        if float(last["close"]) < float(prev["low"].min()):
            reasons.append("micro_breakdown")
        if float(last["volume"]) >= avg_vol * 0.85:
            reasons.append("volume_confirm")
        if float(ltf["close"].iloc[-1]) < float(ltf["close"].iloc[-2]):
            reasons.append("momentum_tick")

    return len(reasons) >= min_confirmations, reasons


def _analyze_swing(
    htf: pd.DataFrame,
    ltf: pd.DataFrame,
    *,
    min_rr: float,
) -> TechnicalContext | None:
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
    if sl_dist <= 0:
        return None

    if direction == TradeDirection.LONG and stop_loss >= entry:
        return None
    if direction == TradeDirection.SHORT and stop_loss <= entry:
        return None
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
        setup_type=SetupType.SWING,
    )


def _analyze_momentum_scalp(
    htf: pd.DataFrame,
    ltf: pd.DataFrame,
    ltf_candles: list[OHLCV],
    *,
    min_rr: float,
    max_sl_pct: float,
    min_confirmations: int,
) -> TechnicalContext | None:
    htf_trend = detect_trend(htf)
    ltf_trend = detect_trend(ltf)
    entry = float(ltf["close"].iloc[-1])
    support = float(htf["low"].tail(20).min())
    resistance = float(htf["high"].tail(20).max())
    ema200 = _ema(htf["close"], min(200, len(htf)))
    last_close = float(htf["close"].iloc[-1])
    above_200 = last_close > float(ema200.iloc[-1]) if len(ema200) else True

    if ltf_trend == Trend.UPTREND:
        direction = TradeDirection.LONG
    elif ltf_trend == Trend.DOWNTREND:
        direction = TradeDirection.SHORT
    else:
        return None

    if direction == TradeDirection.LONG and htf_trend == Trend.DOWNTREND:
        return None
    if direction == TradeDirection.SHORT and htf_trend == Trend.UPTREND:
        return None

    confirmed, reasons = momentum_confirmation(
        ltf_candles,
        direction,
        min_confirmations=min_confirmations,
    )
    if not confirmed:
        return None

    max_sl_frac = max(0.05, max_sl_pct) / 100.0
    if direction == TradeDirection.LONG:
        structural_sl = float(ltf["low"].tail(3).min())
        cap_sl = entry * (1 - max_sl_frac)
        stop_loss = max(structural_sl, cap_sl)
        if stop_loss >= entry:
            return None
        sl_dist = entry - stop_loss
        tp1 = entry + sl_dist * min_rr
        tp2 = entry + sl_dist * (min_rr * 1.5)
        bias = f"LTF momentum scalp ({', '.join(reasons[:3])})"
    else:
        structural_sl = float(ltf["high"].tail(3).max())
        cap_sl = entry * (1 + max_sl_frac)
        stop_loss = min(structural_sl, cap_sl)
        if stop_loss <= entry:
            return None
        sl_dist = stop_loss - entry
        tp1 = entry - sl_dist * min_rr
        tp2 = entry - sl_dist * (min_rr * 1.5)
        bias = f"LTF momentum scalp ({', '.join(reasons[:3])})"

    if sl_dist <= 0:
        return None

    rr = min_rr
    ltf_aligned = True
    strength = 0.35
    strength += min(0.35, 0.07 * len(reasons))
    strength += 0.15 if htf_trend != Trend.RANGE else 0.05
    strength += 0.15 if (direction == TradeDirection.LONG and above_200) or (
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
        setup_type=SetupType.MOMENTUM_SCALP,
        confirmations=reasons,
    )


def analyze_technical(
    htf_candles: list[OHLCV],
    ltf_candles: list[OHLCV],
    *,
    min_rr: float = 2.0,
    scalp_enabled: bool = True,
    scalp_min_rr: float = 1.2,
    scalp_max_sl_pct: float = 0.35,
    scalp_min_confirmations: int = 2,
) -> TechnicalContext | None:
    if len(htf_candles) < 20 or len(ltf_candles) < 20:
        return None

    htf = _ohlcv_to_df(htf_candles)
    ltf = _ohlcv_to_df(ltf_candles)

    swing = _analyze_swing(htf, ltf, min_rr=min_rr)
    if swing is not None:
        return swing

    if not scalp_enabled:
        return None

    return _analyze_momentum_scalp(
        htf,
        ltf,
        ltf_candles,
        min_rr=scalp_min_rr,
        max_sl_pct=scalp_max_sl_pct,
        min_confirmations=scalp_min_confirmations,
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
