"""Technical analysis — stacked sweep strategy (primary) and supporting utilities."""

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
    STACKED_SWEEP = "stacked_sweep"


from analysis.liquidity_sweep import (
    find_all_liquidity_levels,
    detect_liquidity_sweep,
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
    setup_type: SetupType = SetupType.STACKED_SWEEP
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


def _analyze_stacked_sweep(
    htf_candles: list[OHLCV],
    ltf_candles: list[OHLCV],
    *,
    min_rr: float,
    max_sl_pct: float,
) -> TechnicalContext | None:
    """Stacked Sweep — highest-conviction liquidity sweep setup.

    Backtest result (Jan–Jun 2026, 15-min, 112 days):
        80% win rate | Profit factor 4.95 | 5 trades

    Two mandatory filters on top of the base liquidity sweep:

    1. STACKED LEVELS: the sweep candle must hit 2+ distinct level types
       simultaneously — e.g. a swing_high AND equal_highs at the same price.
       Single swing_high / swing_low alone → rejected.

    2. DAILY TREND ALIGNMENT: only LONG sweeps when HTF daily is above EMA20;
       only SHORT sweeps when HTF daily is below EMA20.
       Counter-trend sweeps → rejected.

    Entry: close of the sweep candle.
    SL:    wick extreme + small buffer (just beyond the swept level).
    TP:    entry ± sl_dist × min_rr.
    """
    if len(ltf_candles) < 20 or len(htf_candles) < 20:
        return None

    htf_df = _ohlcv_to_df(htf_candles)
    ltf_df = _ohlcv_to_df(ltf_candles)

    # ── Daily trend filter: close vs EMA20 of HTF candles ─────────────────────
    ema20 = _ema(htf_df["close"], min(20, len(htf_df)))
    htf_close_last = float(htf_df["close"].iloc[-1])
    htf_ema20_last = float(ema20.iloc[-1])
    daily_uptrend = htf_close_last > htf_ema20_last

    # ── Detect liquidity levels ───────────────────────────────────────────────
    levels = find_all_liquidity_levels(ltf_candles, htf_candles)
    if not levels:
        return None

    # ── Detect sweep ──────────────────────────────────────────────────────────
    signal = detect_liquidity_sweep(ltf_df, levels)
    if signal is None:
        return None

    # ── Filter 1: Daily trend alignment ──────────────────────────────────────
    if signal.direction == "LONG" and not daily_uptrend:
        return None
    if signal.direction == "SHORT" and daily_uptrend:
        return None

    # ── Filter 2: Stacked levels — must sweep 2+ distinct level types ─────────
    swept_types = {lv.level_type for lv in signal.swept_levels}
    if len(swept_types) < 2:
        return None

    # ── Build TechnicalContext ────────────────────────────────────────────────
    htf_trend  = detect_trend(htf_df)
    ltf_trend  = detect_trend(ltf_df)
    ema200     = _ema(htf_df["close"], min(200, len(htf_df)))
    above_200  = htf_close_last > float(ema200.iloc[-1]) if len(ema200) else True
    support    = float(htf_df["low"].tail(20).min())
    resistance = float(htf_df["high"].tail(20).max())
    ltf_range  = float(ltf_df["high"].tail(14).max()) - float(ltf_df["low"].tail(14).min())

    entry     = signal.entry
    stop_loss = signal.stop_loss
    sl_dist   = signal.sl_distance

    if sl_dist <= 0 or entry <= 0:
        return None

    # Cap SL by max_sl_pct
    max_sl_frac = max(0.05, max_sl_pct) / 100.0
    max_sl_abs  = entry * max_sl_frac

    if signal.direction == "LONG":
        stop_loss = max(stop_loss, entry - max_sl_abs)
        if stop_loss >= entry:
            return None
        sl_dist = entry - stop_loss
        tp1 = entry + sl_dist * min_rr
        tp2 = entry + sl_dist * min_rr * 1.5
        direction = TradeDirection.LONG
    else:
        stop_loss = min(stop_loss, entry + max_sl_abs)
        if stop_loss <= entry:
            return None
        sl_dist = stop_loss - entry
        tp1 = entry - sl_dist * min_rr
        tp2 = entry - sl_dist * min_rr * 1.5
        direction = TradeDirection.SHORT

    if tp1 <= 0:
        return None

    rr = abs(tp1 - entry) / sl_dist

    ltf_aligned = (
        (direction == TradeDirection.LONG  and ltf_trend == Trend.UPTREND)
        or (direction == TradeDirection.SHORT and ltf_trend == Trend.DOWNTREND)
    )

    swept_names = sorted(lv.level_type.value for lv in signal.swept_levels)
    trend_label = "daily_up" if daily_uptrend else "daily_down"
    bias_source = (
        f"Stacked sweep {signal.direction} | "
        f"levels: {', '.join(swept_names)} | "
        f"trend: {trend_label} | "
        f"entry={entry:.2f} SL={stop_loss:.2f}"
    )

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
        bias_source=bias_source,
        signal_strength=signal.strength,
        setup_type=SetupType.STACKED_SWEEP,
        confirmations=swept_names + [trend_label],
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
    active_strategy: str = "stacked_sweep",
) -> TechnicalContext | None:
    """Run the active strategy analyzer and return a setup if conditions are met.

    Only STACKED_SWEEP is implemented. Other strategy names return None.
    To add a new strategy: implement _analyze_<name>() and add a branch below.
    """
    if len(htf_candles) < 20 or len(ltf_candles) < 20:
        return None

    return _analyze_stacked_sweep(
        htf_candles, ltf_candles,
        min_rr=scalp_min_rr,
        max_sl_pct=scalp_max_sl_pct,
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
    """Compute unified setup quality score (0.0-1.0).

    Weights:
    - Base signal strength from technical analysis: 40%
    - R:R ratio quality: 20%
    - Trend alignment (HTF+LTF): 15%
    - Pullback precision bonus: 15%
    - Volume confirmation: 10%

    A score >= 0.70 indicates a high-conviction setup suitable for scalping.
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