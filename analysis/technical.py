"""Technical analysis — swing setups, momentum scalps, range scalps, trailing helpers."""

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
    RANGE_SCALP = "range_scalp"
    CANDLESTICK_SCALP = "candlestick_scalp"


from analysis.candlestick_patterns import (
    PatternSignal,
    scan_candlestick_patterns,
    find_recent_doji,
    detect_pullback,
)


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
    ltf_range: float = 0.0  # 14-bar LTF high-low range (volatility proxy for entry zone)


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
    """Detect trend using EMA crossover and EMA direction.

    Relaxed from strict close-above-EMA and higher-high requirements.
    Markets oscillate around EMAs; a single candle dip shouldn't kill the trend.

    Includes a range-width check — if the recent price range is very narrow
    (<0.3% of price), classify as RANGE even if EMAs show a micro-trend.
    """
    if len(df) < 10:
        return Trend.RANGE

    close = df["close"]
    ema_short = _ema(close, 9)
    ema_long = _ema(close, 21)

    ema_short_now = float(ema_short.iloc[-1])
    ema_long_now = float(ema_long.iloc[-1])

    # Range-width check: if recent range < 0.3% of price → flat market
    recent = df.tail(15)
    price_range = float(recent["high"].max()) - float(recent["low"].min())
    avg_price = float(recent["close"].mean())
    if avg_price > 0 and price_range / avg_price < 0.003:
        return Trend.RANGE

    # EMA direction: compare now vs 6 bars ago
    if len(df) >= 8:
        ema_short_prev = float(ema_short.iloc[-8])
        ema_short_rising = ema_short_now > ema_short_prev
        ema_short_falling = ema_short_now < ema_short_prev
    else:
        ema_short_rising = True
        ema_short_falling = False

    # Bullish: EMA9 above EMA21 AND EMA9 is rising
    if ema_short_now > ema_long_now and ema_short_rising:
        return Trend.UPTREND

    # Bearish: EMA9 below EMA21 AND EMA9 is falling
    if ema_short_now < ema_long_now and ema_short_falling:
        return Trend.DOWNTREND

    return Trend.RANGE


def momentum_confirmation(
    ltf_candles: list[OHLCV],
    direction: TradeDirection,
    *,
    min_confirmations: int = 1,
) -> tuple[bool, list[str]]:
    """Require momentum signals before a scalp entry.

    Default min_confirmations=1 for faster scalping — a single clean
    momentum signal with tight SL is sufficient.
    """
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

    # Cap SL distance — HTF 20-bar extremes can be unreasonably far.
    # Use LTF recent range as a volatility proxy (max of ATR-like range).
    ltf_range = float(ltf["high"].tail(14).max()) - float(ltf["low"].tail(14).min())
    max_sl_distance = min(ltf_range, entry * 0.02)
    if max_sl_distance <= 0:
        max_sl_distance = entry * 0.015

    if htf_trend == Trend.UPTREND:
        direction = TradeDirection.LONG
        stop_loss = max(support, entry - max_sl_distance)
        tp1 = entry + (entry - stop_loss) * min_rr
        tp2 = entry + (entry - stop_loss) * (min_rr * 1.5)
        bias = f"HTF uptrend{'+ above 200 EMA' if above_200 else ''}"
    elif htf_trend == Trend.DOWNTREND:
        direction = TradeDirection.SHORT
        stop_loss = min(resistance, entry + max_sl_distance)
        tp1 = entry - (stop_loss - entry) * min_rr
        tp2 = entry - (stop_loss - entry) * (min_rr * 1.5)
        bias = f"HTF downtrend{' + below 200 EMA' if not above_200 else ''}"
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
    if tp1 <= 0 or tp2 <= 0:
        return None

    strength = 0.0
    strength += 0.40  # Base strength for having a trend
    strength += 0.20 if ltf_aligned else 0.0  # LTF alignment bonus
    strength += 0.15 if rr >= min_rr else 0.05  # RR bonus
    strength += 0.25 if (direction == TradeDirection.LONG and above_200) or (
        direction == TradeDirection.SHORT and not above_200
    ) else 0.10  # EMA bonus

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
        ltf_range=ltf_range,
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
    ltf_range = float(ltf["high"].tail(14).max()) - float(ltf["low"].tail(14).min())
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
    if tp1 <= 0 or tp2 <= 0:
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
        ltf_range=ltf_range,
    )


def _analyze_range_scalp(
    htf: pd.DataFrame,
    ltf: pd.DataFrame,
    ltf_candles: list[OHLCV],
    *,
    min_rr: float,
    max_sl_pct: float,
    min_confirmations: int,
) -> TechnicalContext | None:
    """Range scalp: trade mean-reversion bounces off support/resistance.

    When both HTF and LTF are ranging, look for entries near the edges
    of the range — buy near support, sell near resistance.  Tight SL
    just beyond the range edge, TP back toward the range midpoint.

    This is where scalpers make most of their money — markets range ~70%
    of the time.
    """
    htf_trend = detect_trend(htf)
    ltf_trend = detect_trend(ltf)
    entry = float(ltf["close"].iloc[-1])
    ltf_range = float(ltf["high"].tail(14).max()) - float(ltf["low"].tail(14).min())
    support = float(htf["low"].tail(20).min())
    resistance = float(htf["high"].tail(20).max())
    ema200 = _ema(htf["close"], min(200, len(htf)))
    last_close = float(htf["close"].iloc[-1])
    above_200 = last_close > float(ema200.iloc[-1]) if len(ema200) else True

    # Only trigger when BOTH timeframes are ranging
    if htf_trend != Trend.RANGE or ltf_trend != Trend.RANGE:
        return None

    # Range must be wide enough to trade (min 0.5% of price)
    range_width = resistance - support
    if range_width <= 0 or range_width / support < 0.005:
        return None

    mid = (support + resistance) / 2
    range_width_pct = range_width / mid

    # ── Direction: mean reversion from range edges ──
    # Buy when price is near support (bottom of range)
    # Sell when price is near resistance (top of range)
    proximity_to_support = (entry - support) / range_width
    proximity_to_resistance = (resistance - entry) / range_width

    # Entry zone: bottom 30% of range → LONG; top 30% → SHORT
    if proximity_to_support < 0.30:
        direction = TradeDirection.LONG
        # SL just below support
        stop_loss = support - range_width * 0.05
        if stop_loss >= entry:
            return None
        sl_dist = entry - stop_loss
        tp1 = entry + sl_dist * min_rr
        tp2 = mid  # target range midpoint
        bias_pre = "Range scalp LONG near support"
    elif proximity_to_resistance < 0.30:
        direction = TradeDirection.SHORT
        # SL just above resistance
        stop_loss = resistance + range_width * 0.05
        if stop_loss <= entry:
            return None
        sl_dist = stop_loss - entry
        tp1 = entry - sl_dist * min_rr
        tp2 = mid  # target range midpoint
        bias_pre = "Range scalp SHORT near resistance"
    else:
        return None  # Don't trade in the middle of the range

    # Momentum confirmation for the direction
    confirmed, reasons = momentum_confirmation(
        ltf_candles,
        direction,
        min_confirmations=min_confirmations,
    )
    if not confirmed:
        return None

    # Cap SL using max_sl_pct
    max_sl_frac = max(0.05, max_sl_pct) / 100.0
    if direction == TradeDirection.LONG:
        cap_sl = entry * (1 - max_sl_frac)
        stop_loss = max(stop_loss, cap_sl)
    else:
        cap_sl = entry * (1 + max_sl_frac)
        stop_loss = min(stop_loss, cap_sl)

    sl_dist = abs(entry - stop_loss)
    if sl_dist <= 0:
        return None

    rr = min_rr
    ltf_aligned = False  # range by definition
    bias = f"{bias_pre} ({', '.join(reasons[:3])}) w={range_width_pct:.2%}"

    strength = 0.30  # Base — range scalps are lower confidence than trend trades
    strength += min(0.30, 0.07 * len(reasons))
    # Bonus for being near range edge (closer = better)
    edge_proximity = max(proximity_to_support, proximity_to_resistance)
    strength += 0.15 * (1.0 - edge_proximity)  # up to 0.15 for being very close to edge
    strength += 0.15 if (direction == TradeDirection.LONG and above_200) or (
        direction == TradeDirection.SHORT and not above_200
    ) else 0.05

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
        setup_type=SetupType.RANGE_SCALP,
        confirmations=reasons,
        ltf_range=ltf_range,
    )


def _analyze_candlestick_scalp(
    htf: pd.DataFrame,
    ltf: pd.DataFrame,
    ltf_candles: list[OHLCV],
    *,
    min_rr: float,
    max_sl_pct: float,
) -> TechnicalContext | None:
    """Candlestick pattern scalp: hammer, engulfing, doji with confirmation.

    Scans LTF candles for reversal patterns aligned with HTF trend:
    - Hammer in downtrend → LONG (bullish reversal)
    - Inverted hammer in uptrend → SHORT (bearish reversal)
    - Bullish engulfing in downtrend → LONG
    - Bearish engulfing in uptrend → SHORT
    - Doji → watch for next hammer/engulfing

    SL = confirmation candle low (LONG) or high (SHORT).
    Target = 1:2 minimum R:R from SL distance.
    """
    htf_trend = detect_trend(htf)
    ltf_trend = detect_trend(ltf)
    entry = float(ltf["close"].iloc[-1])
    ltf_range = float(ltf["high"].tail(14).max()) - float(ltf["low"].tail(14).min())

    # Use LTF trend for pattern context, HTF for alignment
    trend_str = ltf_trend.value.upper()

    pattern = scan_candlestick_patterns(
        ltf_candles,
        trend_str,
        min_confidence=0.45,
    )
    if pattern is None:
        return None

    # Validate entry vs current price — don't enter on stale patterns
    price_change = abs(entry - pattern.entry) / max(pattern.entry, 0.01)
    if price_change > 0.015:  # price moved >1.5% from pattern entry
        return None

    # SL from confirmation candle
    stop_loss = pattern.stop_loss
    sl_dist = pattern.sl_distance
    if sl_dist <= 0:
        return None

    # Cap SL with max_sl_pct
    max_sl_frac = max(0.05, max_sl_pct) / 100.0
    if pattern.direction == "LONG":
        cap_sl = entry * (1 - max_sl_frac)
        stop_loss = max(stop_loss, cap_sl)
        tp1 = entry + sl_dist * min_rr
        tp2 = entry + sl_dist * (min_rr * 2.0)
    else:
        cap_sl = entry * (1 + max_sl_frac)
        stop_loss = min(stop_loss, cap_sl)
        tp1 = entry - sl_dist * min_rr
        tp2 = entry - sl_dist * (min_rr * 2.0)

    direction = TradeDirection.LONG if pattern.direction == "LONG" else TradeDirection.SHORT

    if tp1 <= 0 or tp2 <= 0:
        return None

    # Check for doji context — stronger signal if preceded by doji
    doji_idx = find_recent_doji(ltf_candles, lookback=5)
    if doji_idx is not None:
        pattern.description += " | preceded by doji (indecision → resolution)"

    strength = pattern.strength
    bias = f"Candlestick {pattern.pattern.value}: {pattern.description}"

    return TechnicalContext(
        htf_trend=htf_trend,
        ltf_trend=ltf_trend,
        above_200ema=True,  # not critical for candle patterns
        support=float(htf["low"].tail(20).min()),
        resistance=float(htf["high"].tail(20).max()),
        ltf_aligned=True,
        entry=entry,
        stop_loss=stop_loss,
        tp1=tp1,
        tp2=tp2,
        rr_ratio=min_rr,
        direction=direction,
        bias_source=bias,
        signal_strength=strength,
        setup_type=SetupType.CANDLESTICK_SCALP,
        confirmations=[pattern.pattern.value],
        ltf_range=ltf_range,
    )


def analyze_technical(
    htf_candles: list[OHLCV],
    ltf_candles: list[OHLCV],
    *,
    min_rr: float = 2.0,
    scalp_enabled: bool = True,
    scalp_min_rr: float = 1.2,
    scalp_max_sl_pct: float = 0.35,
    scalp_min_confirmations: int = 1,
) -> TechnicalContext | None:
    if len(htf_candles) < 20 or len(ltf_candles) < 20:
        return None

    htf = _ohlcv_to_df(htf_candles)
    ltf = _ohlcv_to_df(ltf_candles)

    # 1. Try swing setup (HTF trend)
    swing = _analyze_swing(htf, ltf, min_rr=min_rr)
    if swing is not None:
        return swing

    if not scalp_enabled:
        return None

    # 2. Try momentum scalp (LTF trend, HTF not opposing)
    momentum = _analyze_momentum_scalp(
        htf,
        ltf,
        ltf_candles,
        min_rr=scalp_min_rr,
        max_sl_pct=scalp_max_sl_pct,
        min_confirmations=scalp_min_confirmations,
    )
    if momentum is not None:
        return momentum

    # 2.5 Try candlestick pattern scalp (hammer, engulfing, doji)
    candle = _analyze_candlestick_scalp(
        htf,
        ltf,
        ltf_candles,
        min_rr=scalp_min_rr,
        max_sl_pct=scalp_max_sl_pct,
    )
    if candle is not None:
        return candle

    # 3. Try range scalp (both ranging, mean reversion)
    return _analyze_range_scalp(
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


# ═══════════════════════════════════════════════════════════════
# Precision scalping: setup scoring + entry quality checks
# ═══════════════════════════════════════════════════════════════

def check_pullback_to_ema(
    ltf_candles: list[OHLCV],
    direction: TradeDirection,
    *,
    ema_period: int = 9,
    proximity_pct: float = 0.01,  # within 1% of EMA
    option_premium: float = 0.0,  # if >0, use option-appropriate tolerance
) -> tuple[bool, str]:
    """Check if price has pulled back close to EMA before entry.

    For LONG: price should be near or slightly below EMA9 (buy the dip).
    For SHORT: price should be near or slightly above EMA9 (sell the rip).

    Returns (is_pullback, description).
    """
    if len(ltf_candles) < ema_period + 2:
        return False, "insufficient candles for EMA"

    df = _ohlcv_to_df(ltf_candles)
    ema = _ema(df["close"], ema_period)
    last_close = float(df["close"].iloc[-1])
    ema_now = float(ema.iloc[-1])

    if ema_now <= 0:
        return False, "EMA is zero"

    distance_pct = abs(last_close - ema_now) / ema_now

    # Option-appropriate tolerance: cheap options (<₹50) whipsaw 10-30% per candle.
    # Requiring 2% proximity is impossible — use wider or skip entirely.
    if option_premium > 0 and option_premium < 50:
        # For penny options, only fail extreme outliers (>50% from EMA)
        if distance_pct > 0.50:
            return False, f"price {distance_pct:.0%} from EMA9 — extreme drift, wait"
        return True, f"pullback waived for penny option (dist {distance_pct:.0%})"

    if direction == TradeDirection.LONG:
        if last_close > ema_now * (1 + proximity_pct * 2):
            return False, f"price {distance_pct:.1%} above EMA9 — wait for pullback"
        if last_close >= ema_now * (1 - proximity_pct):
            return True, f"pullback to EMA9 (dist {distance_pct:.1%})"
        return True, f"below EMA9 (dist {distance_pct:.1%}) — dip entry"
    else:
        # For SHORT: price should be near or above EMA (selling the rip)
        if last_close < ema_now * (1 - proximity_pct * 2):
            return False, f"price {distance_pct:.1%} below EMA9 — wait for pullback"
        if last_close <= ema_now * (1 + proximity_pct):
            return True, f"pullback to EMA9 (dist {distance_pct:.1%})"
        return True, f"above EMA9 (dist {distance_pct:.1%}) — rip entry"


def check_volume_surge(
    ltf_candles: list[OHLCV],
    *,
    lookback: int = 10,
    surge_multiplier: float = 1.2,
) -> tuple[bool, str]:
    """Check if recent volume is above average (confirms participation).

    Returns (has_surge, description).
    """
    if len(ltf_candles) < lookback + 1:
        return False, "insufficient candles for volume check"

    df = _ohlcv_to_df(ltf_candles)
    avg_vol = float(df["volume"].tail(lookback).mean()) or 1.0
    last_vol = float(df["volume"].iloc[-1])

    if last_vol >= avg_vol * surge_multiplier:
        return True, f"volume surge {last_vol / avg_vol:.1f}x avg"
    return False, f"volume {last_vol / avg_vol:.1f}x avg (need {surge_multiplier}x)"


def score_setup(ctx: TechnicalContext, *, pullback_ok: bool, volume_ok: bool) -> float:
    """Compute unified setup quality score (0.0–1.0).

    Weights:
    - Base signal strength from technical analysis: 40%
    - R:R ratio quality: 20%
    - Trend alignment (HTF+LTF): 15%
    - Pullback precision bonus: 15%
    - Volume confirmation: 10%

    A score ≥ 0.70 indicates a high-conviction setup suitable for scalping.
    """
    score = 0.0

    # 1. Base signal strength (40%)
    score += ctx.signal_strength * 0.40

    # 2. R:R quality (20%) — score higher R:R better
    #   1.5 R:R → 0.10, 2.0 R:R → 0.15, 3.0 R:R → 0.20
    rr_bonus = min(0.20, (ctx.rr_ratio - 1.0) * 0.10)
    score += max(0.0, rr_bonus)

    # 3. Trend alignment (15%)
    if ctx.ltf_aligned:
        score += 0.15
    elif ctx.above_200ema:
        score += 0.08

    # 4. Pullback precision (15%)
    if pullback_ok:
        score += 0.15

    # 5. Volume confirmation (10%)
    if volume_ok:
        score += 0.10

    return min(1.0, score)
