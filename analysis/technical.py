"""Technical analysis — strategy implementations and supporting utilities."""

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
    STACKED_SWEEP = "stacked_sweep"      # V-Reversal days — liquidity sweep reversal
    BB_ORB_BREAKOUT = "bb_orb_breakout"  # Trend/Gap days — BB + ORB momentum
    VWAP_PULLBACK = "vwap_pullback"      # Pullback-to-VWAP in trend direction
    QUAD_STOCH_DIV = "quad_stoch_div"    # Quad stochastic divergence (Holy Grail / HPS)


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
    """Stacked Sweep — highest-conviction reversal setup.

    Used on V-Reversal days (75% of NIFTY trading days).
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


def _analyze_bb_orb_breakout(
    htf_candles: list[OHLCV],
    ltf_candles: list[OHLCV],
    *,
    min_rr: float,
    max_sl_pct: float,
    allowed_direction: TradeDirection,
) -> TechnicalContext | None:
    """BB + ORB Breakout — momentum confirmation for Trend/Gap days.

    Used on Trend Day Up/Down and Gap days (13–14% of NIFTY trading days).
    Backtest result (Jan–Jun 2026, with day-type filter):
        BB + ORB combo: PF 3.75 | 62% WR | N=13
        VWAP + ORB:     PF 3.41 | 67% WR | N=18

    Entry logic:
      - Price has broken the 15-min Opening Range in allowed_direction
      - AND price is at/crossing BB(20,2) band in allowed_direction
      - SL: opposite side of OR range + small buffer
      - TP: entry +/- sl_dist * min_rr

    This is a MOMENTUM setup — fire in the direction of the day trend.
    The day gate already locked the direction; this just confirms the setup.
    """
    if len(ltf_candles) < 22 or len(htf_candles) < 20:
        return None

    # ── BB confirmation ───────────────────────────────────────────────────────
    bb_ok, bb_reason = check_bb_confirmation(ltf_candles, allowed_direction)
    if not bb_ok:
        return None

    # ── ORB confirmation ──────────────────────────────────────────────────────
    orb_ok, orb_reason = check_orb_confirmation(ltf_candles, allowed_direction)
    if not orb_ok:
        return None

    # ── Build levels from OR and ATR ──────────────────────────────────────────
    ltf_df = _ohlcv_to_df(ltf_candles)
    htf_df = _ohlcv_to_df(htf_candles)

    # ATR for SL sizing
    h, l, pc = ltf_df["high"], ltf_df["low"], ltf_df["close"].shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr = float(tr.ewm(span=14, adjust=False).mean().iloc[-1])

    if atr <= 0:
        return None

    # OR boundaries (first 3 × 5m candles = 15 min)
    _OR_BARS = 3
    or_high = max(c.high for c in ltf_candles[:_OR_BARS])
    or_low  = min(c.low  for c in ltf_candles[:_OR_BARS])

    entry = float(ltf_df["close"].iloc[-1])
    max_sl_abs = entry * max(0.05, max_sl_pct) / 100.0

    if allowed_direction == TradeDirection.LONG:
        # SL just below OR low (momentum stopped if OR low breaks)
        raw_sl = or_low - atr * 0.2
        stop_loss = max(raw_sl, entry - max_sl_abs)
        if stop_loss >= entry:
            return None
        sl_dist = entry - stop_loss
    else:
        # SL just above OR high
        raw_sl = or_high + atr * 0.2
        stop_loss = min(raw_sl, entry + max_sl_abs)
        if stop_loss <= entry:
            return None
        sl_dist = stop_loss - entry

    if sl_dist <= 0:
        return None

    if allowed_direction == TradeDirection.LONG:
        tp1 = entry + sl_dist * min_rr
        tp2 = entry + sl_dist * min_rr * 1.5
    else:
        tp1 = entry - sl_dist * min_rr
        tp2 = entry - sl_dist * min_rr * 1.5

    if tp1 <= 0:
        return None

    rr = abs(tp1 - entry) / sl_dist

    htf_trend = detect_trend(htf_df)
    ltf_trend = detect_trend(ltf_df)
    ema200    = _ema(htf_df["close"], min(200, len(htf_df)))
    above_200 = float(htf_df["close"].iloc[-1]) > float(ema200.iloc[-1]) if len(ema200) else True
    support   = float(htf_df["low"].tail(20).min())
    resistance= float(htf_df["high"].tail(20).max())
    ltf_range = float(ltf_df["high"].tail(14).max()) - float(ltf_df["low"].tail(14).min())

    ltf_aligned = (
        (allowed_direction == TradeDirection.LONG  and ltf_trend == Trend.UPTREND)
        or (allowed_direction == TradeDirection.SHORT and ltf_trend == Trend.DOWNTREND)
    )

    # Signal strength: base 0.60 + ltf aligned bonus
    strength = 0.60 + (0.10 if ltf_aligned else 0.0)

    bias_source = (
        f"BB+ORB breakout {allowed_direction.value} | "
        f"OR={or_low:.2f}-{or_high:.2f} | "
        f"entry={entry:.2f} SL={stop_loss:.2f} | "
        f"{bb_reason} | {orb_reason}"
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
        direction=allowed_direction,
        bias_source=bias_source,
        signal_strength=min(1.0, strength),
        setup_type=SetupType.BB_ORB_BREAKOUT,
        confirmations=["bb_confirmed", "orb_confirmed"],
        ltf_range=ltf_range,
    )


def _compute_session_vwap(candles: list[OHLCV]) -> tuple[float, float, float]:
    """Compute session VWAP and ±1σ bands from intraday candles.

    Returns (vwap, upper_1sigma, lower_1sigma).
    Uses typical price × volume weighting, standard deviation of typical prices
    weighted by volume for the band width.
    Returns (0, 0, 0) if candles are insufficient.
    """
    if len(candles) < 5:
        return 0.0, 0.0, 0.0

    tp_vol_sum = 0.0
    vol_sum    = 0.0
    tp_sq_sum  = 0.0

    for c in candles:
        tp = (c.high + c.low + c.close) / 3.0
        v  = float(c.volume) or 1.0
        tp_vol_sum += tp * v
        vol_sum    += v
        tp_sq_sum  += tp * tp * v

    if vol_sum <= 0:
        return 0.0, 0.0, 0.0

    vwap     = tp_vol_sum / vol_sum
    variance = max(0.0, (tp_sq_sum / vol_sum) - vwap * vwap)
    sigma    = variance ** 0.5

    return vwap, vwap + sigma, vwap - sigma


def check_vwap_pullback(
    ltf_candles: list[OHLCV],
    direction: TradeDirection,
    *,
    proximity_pct: float = 0.003,
) -> tuple[bool, str]:
    """Check whether the last candle is pulling back to VWAP in trend direction.

    Returns (ok, reason).

    LONG: price must be above VWAP (trend confirmed) and current low
          touched within proximity_pct of VWAP (pullback occurred).
    SHORT: price must be below VWAP and current high touched VWAP.
    """
    if len(ltf_candles) < 10:
        return False, "too_few_candles"

    vwap, _, _ = _compute_session_vwap(ltf_candles)
    if vwap <= 0:
        return False, "vwap_zero"

    last  = ltf_candles[-1]
    close = last.close

    if direction == TradeDirection.LONG:
        if close <= vwap:
            return False, f"close {close:.2f} below VWAP {vwap:.2f} — not in uptrend"
        proximity = abs(last.low - vwap) / vwap
        if proximity > proximity_pct:
            return False, f"low {last.low:.2f} not close enough to VWAP {vwap:.2f} ({proximity:.3%})"
        return True, f"LONG pullback to VWAP {vwap:.2f} confirmed"
    else:
        if close >= vwap:
            return False, f"close {close:.2f} above VWAP {vwap:.2f} — not in downtrend"
        proximity = abs(last.high - vwap) / vwap
        if proximity > proximity_pct:
            return False, f"high {last.high:.2f} not close enough to VWAP {vwap:.2f} ({proximity:.3%})"
        return True, f"SHORT pullback to VWAP {vwap:.2f} confirmed"


def _analyze_vwap_pullback(
    htf_candles: list[OHLCV],
    ltf_candles: list[OHLCV],
    *,
    min_rr: float,
    max_sl_pct: float,
    allowed_direction: TradeDirection | None = None,
) -> TechnicalContext | None:
    """VWAP Pullback — trend-following entry on pullback to session VWAP.

    Works on any day type; best on trending days (V-Reversal Bull/Bear,
    Trend Up/Down). Avoid in flat/range sessions (VWAP and price oscillate
    through each other constantly — no edge).

    Setup logic:
      1. HTF trend (EMA9/21) defines direction.
      2. VWAP computed from all today's LTF candles.
      3. Price pulled back and tagged VWAP (last candle low/high within 0.3%).
      4. Last candle closed back in trend direction (rejection wick).
      5. Volume above 20-bar average (not a dead-zone touch).

    SL: just beyond VWAP (opposite side), capped by max_sl_pct.
    TP: entry + sl_dist × min_rr.
    """
    if len(ltf_candles) < 20 or len(htf_candles) < 20:
        return None

    htf_df = _ohlcv_to_df(htf_candles)
    ltf_df = _ohlcv_to_df(ltf_candles)
    htf_trend = detect_trend(htf_df)
    ltf_trend = detect_trend(ltf_df)

    # Determine trade direction from HTF trend; respect day gate lock if set
    if allowed_direction is not None:
        direction = allowed_direction
    elif htf_trend == Trend.UPTREND:
        direction = TradeDirection.LONG
    elif htf_trend == Trend.DOWNTREND:
        direction = TradeDirection.SHORT
    else:
        return None  # range HTF — no edge

    # VWAP pullback check
    vwap_ok, vwap_reason = check_vwap_pullback(ltf_candles, direction)
    if not vwap_ok:
        return None

    vwap, vwap_upper, vwap_lower = _compute_session_vwap(ltf_candles)

    # Volume confirmation
    avg_vol = float(ltf_df["volume"].tail(20).mean()) or 1.0
    last_vol = float(ltf_df["volume"].iloc[-1])
    vol_ok = last_vol >= avg_vol * 1.1

    # Rejection candle: must close back in trend direction
    last = ltf_candles[-1]
    if direction == TradeDirection.LONG and last.close <= last.open:
        return None  # bearish candle at VWAP — rejection not confirmed
    if direction == TradeDirection.SHORT and last.close >= last.open:
        return None  # bullish candle at VWAP — rejection not confirmed

    entry = last.close
    max_sl_abs = entry * max(0.05, max_sl_pct) / 100.0

    # SL: just beyond VWAP on the wrong side + small ATR buffer
    h, l, pc = ltf_df["high"], ltf_df["low"], ltf_df["close"].shift(1)
    tr  = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr = float(tr.ewm(span=14, adjust=False).mean().iloc[-1])

    if direction == TradeDirection.LONG:
        raw_sl   = vwap - atr * 0.3
        stop_loss = max(raw_sl, entry - max_sl_abs)
        if stop_loss >= entry:
            return None
        sl_dist = entry - stop_loss
        tp1 = entry + sl_dist * min_rr
        tp2 = entry + sl_dist * min_rr * 1.5
    else:
        raw_sl   = vwap + atr * 0.3
        stop_loss = min(raw_sl, entry + max_sl_abs)
        if stop_loss <= entry:
            return None
        sl_dist   = stop_loss - entry
        tp1 = entry - sl_dist * min_rr
        tp2 = entry - sl_dist * min_rr * 1.5

    if sl_dist <= 0 or tp1 <= 0:
        return None

    rr = abs(tp1 - entry) / sl_dist

    ema200    = _ema(htf_df["close"], min(200, len(htf_df)))
    above_200 = float(htf_df["close"].iloc[-1]) > float(ema200.iloc[-1]) if len(ema200) else True
    support   = float(htf_df["low"].tail(20).min())
    resistance= float(htf_df["high"].tail(20).max())
    ltf_range = float(ltf_df["high"].tail(14).max()) - float(ltf_df["low"].tail(14).min())
    ltf_aligned = (
        (direction == TradeDirection.LONG  and ltf_trend == Trend.UPTREND)
        or (direction == TradeDirection.SHORT and ltf_trend == Trend.DOWNTREND)
    )

    # Signal strength: base 0.55, bonuses for ltf alignment and volume
    strength = 0.55
    strength += 0.10 if ltf_aligned else 0.0
    strength += 0.05 if vol_ok else 0.0

    confirmations = ["vwap_pullback", vwap_reason]
    if vol_ok:
        confirmations.append("volume_ok")
    if ltf_aligned:
        confirmations.append("ltf_aligned")

    bias_source = (
        f"VWAP pullback {direction.value} | "
        f"VWAP={vwap:.2f} | "
        f"entry={entry:.2f} SL={stop_loss:.2f} | "
        f"{vwap_reason}"
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
        signal_strength=min(1.0, strength),
        setup_type=SetupType.VWAP_PULLBACK,
        confirmations=confirmations,
        ltf_range=ltf_range,
    )


def _stochastic(
    df: pd.DataFrame,
    k_period: int,
    d_period: int,
    *,
    smooth: int = 1,
) -> tuple[pd.Series, pd.Series]:
    """Classic stochastic %K / %D (TradingView-compatible).

    raw_%K = 100 * (close - lowest_low) / (highest_high - lowest_low)
    %K     = SMA(raw_%K, smooth)   # smooth=1 → unsmoothed / Full
    %D     = SMA(%K, d_period)
    """
    lowest = df["low"].rolling(k_period).min()
    highest = df["high"].rolling(k_period).max()
    denom = (highest - lowest).replace(0, pd.NA)
    raw_k = 100.0 * (df["close"] - lowest) / denom
    k = raw_k.rolling(smooth).mean() if smooth > 1 else raw_k
    d = k.rolling(d_period).mean()
    return k, d


def _swing_low_indices(lows: pd.Series, order: int = 2) -> list[int]:
    """Indices of local swing lows (strict local minima over ±order bars)."""
    vals = lows.to_numpy(dtype=float)
    n = len(vals)
    out: list[int] = []
    for i in range(order, n - order):
        left_ok = all(vals[i] <= vals[i - j] for j in range(1, order + 1))
        right_ok = all(vals[i] < vals[i + j] for j in range(1, order + 1))
        if left_ok and right_ok:
            out.append(i)
    return out


def _swing_high_indices(highs: pd.Series, order: int = 2) -> list[int]:
    """Indices of local swing highs (strict local maxima over ±order bars)."""
    vals = highs.to_numpy(dtype=float)
    n = len(vals)
    out: list[int] = []
    for i in range(order, n - order):
        left_ok = all(vals[i] >= vals[i - j] for j in range(1, order + 1))
        right_ok = all(vals[i] > vals[i + j] for j in range(1, order + 1))
        if left_ok and right_ok:
            out.append(i)
    return out


def _is_bullish_reversal_candle(c: OHLCV) -> bool:
    """Hammer / rejection wick: lower wick >= body, closes in upper half."""
    body = abs(c.close - c.open)
    lower_wick = min(c.open, c.close) - c.low
    upper_wick = c.high - max(c.open, c.close)
    rng = c.high - c.low
    if rng <= 0:
        return False
    closes_upper = c.close >= c.low + rng * 0.55
    return lower_wick >= max(body, rng * 0.35) and lower_wick >= upper_wick and closes_upper


def _is_bearish_reversal_candle(c: OHLCV) -> bool:
    """Shooting star / rejection wick: upper wick >= body, closes in lower half."""
    body = abs(c.close - c.open)
    lower_wick = min(c.open, c.close) - c.low
    upper_wick = c.high - max(c.open, c.close)
    rng = c.high - c.low
    if rng <= 0:
        return False
    closes_lower = c.close <= c.high - rng * 0.55
    return upper_wick >= max(body, rng * 0.35) and upper_wick >= lower_wick and closes_lower


def _organized_pullback(
    df: pd.DataFrame,
    start: int,
    end: int,
    *,
    atr: float,
    max_chaos_atr: float = 2.5,
) -> bool:
    """Reject chaotic Stage1→Stage2 drops (single panic candle or huge bar).

    Interior bars only — Stage 1/2 extremes often carry long rejection wicks
    by design and must not fail the organization filter.
    """
    if end - start < 2 or atr <= 0:
        return False
    segment = df.iloc[start + 1 : end]
    if segment.empty:
        return True
    max_bar_range = float((segment["high"] - segment["low"]).max())
    return max_bar_range <= atr * max_chaos_atr


@dataclass
class _StochDivSignal:
    direction: TradeDirection
    stage1_idx: int
    stage2_idx: int
    stage1_price: float
    stage2_price: float
    stage1_stoch: float
    stage2_stoch: float
    confirmations: list[str]
    strength: float


def detect_quad_stoch_divergence(
    ltf_candles: list[OHLCV],
    *,
    require_lower_extreme: bool = False,
    min_bars_between: int = 3,
    max_bars_between: int = 20,
    lookback: int = 50,
    swing_order: int = 2,
) -> _StochDivSignal | None:
    """Detect bullish/bearish quad-stochastic divergence ending near the last bar.

    Bullish (mirror for bearish):
      Stage 1 — all 4 stochs < 20 near a swing low; then bounce above 20.
      Stage 2 — price equal/lower low while fast (9,3) holds > 20 and makes a
                higher stoch low; entry when 9-3 turns up off that low.

    Tunables:
      require_lower_extreme — if True, Stage 2 must be strictly lower/higher
                              (not equal) than Stage 1 price extreme.
    """
    if len(ltf_candles) < 80:
        return None

    df = _ohlcv_to_df(ltf_candles)
    k9, d9 = _stochastic(df, 9, 3)
    k14, _ = _stochastic(df, 14, 3)
    k40, _ = _stochastic(df, 40, 4)
    k60, _ = _stochastic(df, 60, 10, smooth=1)

    # Need valid stoch readings
    if any(pd.isna(x.iloc[-1]) for x in (k9, d9, k14, k40, k60)):
        return None

    h, l, pc = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr = float(tr.ewm(span=14, adjust=False).mean().iloc[-1])
    if atr <= 0:
        return None

    n = len(df)
    start = max(swing_order + 1, n - lookback)
    end_scan = n - swing_order  # leave room for swing confirmation

    # Fast stoch must be turning up/down on the latest bar
    k_now = float(k9.iloc[-1])
    k_prev = float(k9.iloc[-2])
    d_now = float(d9.iloc[-1])
    bull_turn = k_now > k_prev and k_now >= d_now
    bear_turn = k_now < k_prev and k_now <= d_now
    if not bull_turn and not bear_turn:
        return None

    def _all_oversold_near(idx: int, window: int = 4) -> bool:
        lo = max(0, idx - window)
        hi = min(n, idx + 1)
        for i in range(lo, hi):
            vals = [k9.iloc[i], k14.iloc[i], k40.iloc[i], k60.iloc[i]]
            if any(pd.isna(v) for v in vals):
                continue
            if all(float(v) < 20.0 for v in vals):
                return True
        return False

    def _all_overbought_near(idx: int, window: int = 4) -> bool:
        lo = max(0, idx - window)
        hi = min(n, idx + 1)
        for i in range(lo, hi):
            vals = [k9.iloc[i], k14.iloc[i], k40.iloc[i], k60.iloc[i]]
            if any(pd.isna(v) for v in vals):
                continue
            if all(float(v) > 80.0 for v in vals):
                return True
        return False

    def _bounced_above_20(s1: int, s2: int) -> bool:
        mid = k9.iloc[s1 : s2 + 1]
        return bool((mid > 20.0).any())

    def _dipped_below_80(s1: int, s2: int) -> bool:
        mid = k9.iloc[s1 : s2 + 1]
        return bool((mid < 80.0).any())

    # ── Bullish path ──────────────────────────────────────────────────────────
    if bull_turn:
        lows = _swing_low_indices(df["low"], order=swing_order)
        lows = [i for i in lows if start <= i < end_scan]
        # Prefer Stage 2 near the end (within last few bars before turn)
        for j in range(len(lows) - 1, 0, -1):
            s2 = lows[j]
            s1 = lows[j - 1]
            # Stage 2 should be recent relative to the turn bar
            if s2 < n - 6:
                continue
            gap = s2 - s1
            if gap < min_bars_between or gap > max_bars_between:
                continue

            p1 = float(df["low"].iloc[s1])
            p2 = float(df["low"].iloc[s2])
            if require_lower_extreme:
                if p2 >= p1:
                    continue
            elif p2 > p1:
                continue

            sk1 = float(k9.iloc[s1])
            sk2 = float(k9.iloc[s2])
            if pd.isna(sk1) or pd.isna(sk2):
                continue
            # Stage 2 fast stoch holds above 20 and prints a higher low
            if sk2 <= 20.0 or sk2 <= sk1:
                continue
            if not _all_oversold_near(s1):
                continue
            if not _bounced_above_20(s1, s2):
                continue
            if not _organized_pullback(df, s1, s2, atr=atr):
                continue

            conf = ["stoch_div_bull", "quad_oversold_s1", "fast_holds_above_20"]
            strength = 0.55
            if sk2 - sk1 >= 5.0:
                strength += 0.08
                conf.append("strong_stoch_hl")
            if float(k14.iloc[s2]) > float(k14.iloc[s1]) if not pd.isna(k14.iloc[s1]) else False:
                strength += 0.05
                conf.append("k14_confirms")
            c2 = ltf_candles[s2]
            if _is_bullish_reversal_candle(c2):
                strength += 0.08
                conf.append("reversal_candle")

            return _StochDivSignal(
                direction=TradeDirection.LONG,
                stage1_idx=s1,
                stage2_idx=s2,
                stage1_price=p1,
                stage2_price=p2,
                stage1_stoch=sk1,
                stage2_stoch=sk2,
                confirmations=conf,
                strength=min(1.0, strength),
            )

    # ── Bearish path ──────────────────────────────────────────────────────────
    if bear_turn:
        highs = _swing_high_indices(df["high"], order=swing_order)
        highs = [i for i in highs if start <= i < end_scan]
        for j in range(len(highs) - 1, 0, -1):
            s2 = highs[j]
            s1 = highs[j - 1]
            if s2 < n - 6:
                continue
            gap = s2 - s1
            if gap < min_bars_between or gap > max_bars_between:
                continue

            p1 = float(df["high"].iloc[s1])
            p2 = float(df["high"].iloc[s2])
            if require_lower_extreme:
                if p2 <= p1:
                    continue
            elif p2 < p1:
                continue

            sk1 = float(k9.iloc[s1])
            sk2 = float(k9.iloc[s2])
            if pd.isna(sk1) or pd.isna(sk2):
                continue
            if sk2 >= 80.0 or sk2 >= sk1:
                continue
            if not _all_overbought_near(s1):
                continue
            if not _dipped_below_80(s1, s2):
                continue
            if not _organized_pullback(df, s1, s2, atr=atr):
                continue

            conf = ["stoch_div_bear", "quad_overbought_s1", "fast_holds_below_80"]
            strength = 0.55
            if sk1 - sk2 >= 5.0:
                strength += 0.08
                conf.append("strong_stoch_lh")
            if float(k14.iloc[s2]) < float(k14.iloc[s1]) if not pd.isna(k14.iloc[s1]) else False:
                strength += 0.05
                conf.append("k14_confirms")
            c2 = ltf_candles[s2]
            if _is_bearish_reversal_candle(c2):
                strength += 0.08
                conf.append("reversal_candle")

            return _StochDivSignal(
                direction=TradeDirection.SHORT,
                stage1_idx=s1,
                stage2_idx=s2,
                stage1_price=p1,
                stage2_price=p2,
                stage1_stoch=sk1,
                stage2_stoch=sk2,
                confirmations=conf,
                strength=min(1.0, strength),
            )

    return None


def _analyze_quad_stoch_div(
    htf_candles: list[OHLCV],
    ltf_candles: list[OHLCV],
    *,
    min_rr: float,
    max_sl_pct: float,
    allowed_direction: TradeDirection | None = None,
    require_lower_extreme: bool = False,
) -> TechnicalContext | None:
    """Quad Stochastic Divergence — Holy Grail / HPS momentum divergence.

    Four stacked stochastics (9-3, 14-3, 40-4, 60-10) + EMA/VWAP context.
    Triggers when price makes an equal/lower low but fast stoch makes a higher
    low while holding above 20 (exhaustion → reversal). Mirror for shorts.

    Entry: close of the bar where 9-3 turns up off the Stage 2 higher stoch low.
    SL:    1–2 ticks (ATR-scaled buffer) beyond the Stage 2 extreme.
    TP1:   entry ± sl_dist × min_rr  (proxy for first partial near stoch 80).
    TP2:   1.5× that distance for trend-context runners.

    Exit guidance (encoded in confirmations / bias_source):
      - Counter-trend (below 200 EMA): take profits fast, tighten to BE.
      - With-trend (above 200 EMA / VWAP): trail using 60-10 rotations.
    """
    if len(ltf_candles) < 80 or len(htf_candles) < 20:
        return None

    signal = detect_quad_stoch_divergence(
        ltf_candles,
        require_lower_extreme=require_lower_extreme,
    )
    if signal is None:
        return None

    if allowed_direction is not None and signal.direction != allowed_direction:
        return None

    htf_df = _ohlcv_to_df(htf_candles)
    ltf_df = _ohlcv_to_df(ltf_candles)
    htf_trend = detect_trend(htf_df)
    ltf_trend = detect_trend(ltf_df)

    entry = float(ltf_candles[-1].close)
    max_sl_abs = entry * max(0.05, max_sl_pct) / 100.0

    h, l, pc = ltf_df["high"], ltf_df["low"], ltf_df["close"].shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr = float(tr.ewm(span=14, adjust=False).mean().iloc[-1])
    tick_buf = max(atr * 0.05, entry * 0.00005)  # ~1–2 ticks scaled to ATR

    if signal.direction == TradeDirection.LONG:
        raw_sl = signal.stage2_price - tick_buf
        stop_loss = max(raw_sl, entry - max_sl_abs)
        if stop_loss >= entry:
            return None
        sl_dist = entry - stop_loss
        tp1 = entry + sl_dist * min_rr
        tp2 = entry + sl_dist * min_rr * 1.5
    else:
        raw_sl = signal.stage2_price + tick_buf
        stop_loss = min(raw_sl, entry + max_sl_abs)
        if stop_loss <= entry:
            return None
        sl_dist = stop_loss - entry
        tp1 = entry - sl_dist * min_rr
        tp2 = entry - sl_dist * min_rr * 1.5

    if sl_dist <= 0 or tp1 <= 0:
        return None

    rr = abs(tp1 - entry) / sl_dist

    ema200 = _ema(htf_df["close"], min(200, len(htf_df)))
    above_200 = float(htf_df["close"].iloc[-1]) > float(ema200.iloc[-1]) if len(ema200) else True
    vwap, _, _ = _compute_session_vwap(ltf_candles)
    above_vwap = entry > vwap if vwap > 0 else above_200

    # Trend context for exit policy (spec: weak vs high-confidence)
    confirmations = list(signal.confirmations)
    strength = signal.strength
    if signal.direction == TradeDirection.LONG:
        if above_200 or above_vwap:
            confirmations.append("uptrend_context")
            strength += 0.07
        else:
            confirmations.append("counter_trend_fast_exit")
    else:
        if (not above_200) or (vwap > 0 and entry < vwap):
            confirmations.append("downtrend_context")
            strength += 0.07
        else:
            confirmations.append("counter_trend_fast_exit")

    support = float(htf_df["low"].tail(20).min())
    resistance = float(htf_df["high"].tail(20).max())
    ltf_range = float(ltf_df["high"].tail(14).max()) - float(ltf_df["low"].tail(14).min())
    ltf_aligned = (
        (signal.direction == TradeDirection.LONG and ltf_trend == Trend.UPTREND)
        or (signal.direction == TradeDirection.SHORT and ltf_trend == Trend.DOWNTREND)
    )
    if ltf_aligned:
        confirmations.append("ltf_aligned")
        strength += 0.05

    bias_source = (
        f"Quad stoch div {signal.direction.value} | "
        f"S1 price={signal.stage1_price:.2f} K={signal.stage1_stoch:.1f} | "
        f"S2 price={signal.stage2_price:.2f} K={signal.stage2_stoch:.1f} | "
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
        direction=signal.direction,
        bias_source=bias_source,
        signal_strength=min(1.0, strength),
        setup_type=SetupType.QUAD_STOCH_DIV,
        confirmations=confirmations,
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
    active_strategy: str = "quad_stoch_div",
    day_type: str = "unknown",
    allowed_direction: TradeDirection | None = None,
) -> TechnicalContext | None:
    """Route exclusively by active_strategy — no day-type strategy switching.

    Default strategy: quad_stoch_div (Holy Grail / HPS divergence).

    active_strategy selects exactly one analyzer:
      'quad_stoch_div' | 'stacked_sweep' | 'bb_orb_breakout' | 'vwap_pullback'

    day_type is accepted for pipeline compatibility / logging but never
    overrides the strategy. ORB runs only when active_strategy == 'bb_orb_breakout'.
    """
    _ = day_type  # unused — kept for call-site compatibility

    if len(htf_candles) < 20 or len(ltf_candles) < 20:
        return None

    if active_strategy == "bb_orb_breakout":
        if allowed_direction is None:
            return None  # breakout requires a locked direction from day gate
        return _analyze_bb_orb_breakout(
            htf_candles, ltf_candles,
            min_rr=scalp_min_rr,
            max_sl_pct=scalp_max_sl_pct,
            allowed_direction=allowed_direction,
        )

    if active_strategy == "vwap_pullback":
        return _analyze_vwap_pullback(
            htf_candles, ltf_candles,
            min_rr=scalp_min_rr,
            max_sl_pct=scalp_max_sl_pct,
            allowed_direction=allowed_direction,
        )

    if active_strategy == "stacked_sweep":
        return _analyze_stacked_sweep(
            htf_candles, ltf_candles,
            min_rr=scalp_min_rr,
            max_sl_pct=scalp_max_sl_pct,
        )

    # Default: quad stochastic divergence
    return _analyze_quad_stoch_div(
        htf_candles, ltf_candles,
        min_rr=scalp_min_rr,
        max_sl_pct=scalp_max_sl_pct,
        allowed_direction=allowed_direction,
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


def check_bb_confirmation(
    ltf_candles: list[OHLCV],
    direction: TradeDirection,
    *,
    period: int = 20,
    std_dev: float = 2.0,
) -> tuple[bool, str]:
    """Check if price is breaking out of Bollinger Bands in the trade direction.

    For LONG: close crossed above upper band on the last 1–2 candles.
    For SHORT: close crossed below lower band on the last 1–2 candles.

    Backtest edge: BB(20,2) + ORB 15m combo → PF 3.75, 62% WR, N=13 (with day filter).

    Returns (confirmed, reason).
    """
    if len(ltf_candles) < period + 2:
        return False, "insufficient candles for BB"

    df = _ohlcv_to_df(ltf_candles)
    mid = df["close"].rolling(period).mean()
    std = df["close"].rolling(period).std()
    upper = mid + std_dev * std
    lower = mid - std_dev * std

    last_close = float(df["close"].iloc[-1])
    prev_close = float(df["close"].iloc[-2])
    last_upper = float(upper.iloc[-1])
    last_lower = float(lower.iloc[-1])
    prev_upper = float(upper.iloc[-2])
    prev_lower = float(lower.iloc[-2])

    if direction == TradeDirection.LONG:
        crossed = last_close > last_upper and prev_close <= prev_upper
        near = last_close > last_upper * 0.998  # within 0.2% of upper band
        if crossed:
            return True, f"BB breakout LONG: close {last_close:.2f} > upper {last_upper:.2f}"
        if near:
            return True, f"BB near upper band: close {last_close:.2f} ≈ {last_upper:.2f}"
        return False, f"BB: close {last_close:.2f} below upper {last_upper:.2f}"
    else:
        crossed = last_close < last_lower and prev_close >= prev_lower
        near = last_close < last_lower * 1.002
        if crossed:
            return True, f"BB breakout SHORT: close {last_close:.2f} < lower {last_lower:.2f}"
        if near:
            return True, f"BB near lower band: close {last_close:.2f} ≈ {last_lower:.2f}"
        return False, f"BB: close {last_close:.2f} above lower {last_lower:.2f}"


def check_orb_confirmation(
    ltf_candles: list[OHLCV],
    direction: TradeDirection,
    *,
    or_bars: int = 3,  # 3 × 5m = 15-min OR
) -> tuple[bool, str]:
    """Check if price has broken the Opening Range (OR) in the trade direction.

    OR = high/low of the first `or_bars` 5-min candles (default 15 min).
    For LONG: current close above OR high.
    For SHORT: current close below OR low.

    Backtest edge: BB + ORB combo → PF 3.75, 62% WR (with day filter).
    VWAP + ORB combo → PF 3.41, 67% WR (with day filter).

    Returns (confirmed, reason).
    """
    if len(ltf_candles) < or_bars + 2:
        return False, "insufficient candles for ORB"

    # Assume candles are sorted oldest→newest (intraday from market open)
    # The first `or_bars` are the opening range
    or_slice = ltf_candles[:or_bars]
    or_high = max(c.high for c in or_slice)
    or_low = min(c.low for c in or_slice)

    # Only check ORB if we're past the OR period
    if len(ltf_candles) <= or_bars:
        return False, "still inside OR period"

    last_close = ltf_candles[-1].close

    if direction == TradeDirection.LONG:
        if last_close > or_high:
            return True, f"ORB breakout LONG: close {last_close:.2f} > OR high {or_high:.2f}"
        gap_pct = (or_high - last_close) / or_high * 100
        return False, f"ORB: close {last_close:.2f} below OR high {or_high:.2f} ({gap_pct:.1f}% away)"
    else:
        if last_close < or_low:
            return True, f"ORB breakdown SHORT: close {last_close:.2f} < OR low {or_low:.2f}"
        gap_pct = (last_close - or_low) / or_low * 100
        return False, f"ORB: close {last_close:.2f} above OR low {or_low:.2f} ({gap_pct:.1f}% away)"



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