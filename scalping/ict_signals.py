"""ICT/SMC signal detection: Order Block identification and OTE zone math.

Three-step ICT entry model
--------------------------
1. Liquidity Sweep   — handled upstream by analysis/liquidity_sweep.py
2. Order Block (OB)  — last candle whose body opposes the displacement direction,
                       found immediately before the displacement impulse.
3. OTE Entry         — Fibonacci 61.8 %–79 % retracement of the displacement leg
                       back into the OB zone.

All functions are pure (no I/O, no side-effects) so they are trivially testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Fibonacci levels defining the OTE zone (ICT standard)
OTE_FIB_LOW  = 0.618   # 61.8 % retracement
OTE_FIB_HIGH = 0.790   # 79.0 % retracement


@dataclass
class OrderBlock:
    """An un-mitigated Order Block with its derived risk levels."""

    direction: str          # "LONG" or "SHORT"
    ob_high: float          # top of the OB candle
    ob_low: float           # bottom of the OB candle
    ote_high: float         # upper edge of OTE zone (61.8 % retracement)
    ote_low: float          # lower edge of OTE zone (79.0 % retracement)
    sl_price: float         # SL 1-buffer below/above OB boundary
    tp_price: float         # TP at swept-liquidity origin
    displacement_high: float
    displacement_low: float
    ob_candle_index: int    # index in the candle list
    fresh: bool = True      # invalidated once price closes through OB
    strength: float = 0.5   # 0.0–1.0, from sweep strength + OB quality

    @property
    def ote_midpoint(self) -> float:
        return (self.ote_high + self.ote_low) / 2

    def price_in_ote(self, price: float) -> bool:
        """Return True when price touches the OTE zone."""
        return self.ote_low <= price <= self.ote_high

    def price_in_ob(self, price: float) -> bool:
        return self.ob_low <= price <= self.ob_high

    def check_mitigation(self, candle_close: float) -> None:
        """Mark OB as mitigated (spent) if price closes through it."""
        if not self.fresh:
            return
        if self.direction == "LONG" and candle_close < self.ob_low:
            self.fresh = False
        elif self.direction == "SHORT" and candle_close > self.ob_high:
            self.fresh = False


# ── Order Block detection ─────────────────────────────────────────────────────

def detect_order_block(
    candles: list[dict],
    direction: str,
    *,
    displacement_lookback: int = 8,
    sl_buffer_pct: float = 0.001,
    min_ob_body_pct: float = 0.003,
    sweep_strength: float = 0.5,
) -> OrderBlock | None:
    """Detect the most recent un-mitigated Order Block for the given direction.

    Parameters
    ----------
    candles:
        List of OHLCV dicts with keys open/high/low/close/volume.
        Must be in chronological order, last element = most recent closed bar.
    direction:
        "LONG" — look for a bullish setup after a sweep of lows.
        "SHORT" — look for a bearish setup after a sweep of highs.
    displacement_lookback:
        How many recent bars to scan for a displacement impulse.
    sl_buffer_pct:
        Percentage buffer beyond the OB boundary for the stop-loss.
    min_ob_body_pct:
        Minimum body size as a fraction of close price — filters doji OBs.
    sweep_strength:
        Strength score from the upstream sweep detector (0.0–1.0).
    """
    if len(candles) < displacement_lookback + 3:
        return None

    closes = [c["close"] for c in candles]
    opens  = [c["open"]  for c in candles]
    highs  = [c["high"]  for c in candles]
    lows   = [c["low"]   for c in candles]

    n = len(candles)
    scan_end = n - 1          # exclude the currently forming bar
    scan_start = max(0, scan_end - displacement_lookback)

    if direction == "LONG":
        # Find the most recent strong bullish displacement candle
        # (close well above open, range above average)
        disp_idx = _find_displacement(opens, closes, highs, lows, scan_start, scan_end, "bull")
        if disp_idx is None:
            return None

        # Walk back from the displacement to find the last bearish body candle
        # (this is the Order Block — institutions placed buy orders here)
        ob_idx = _last_opposite_body(opens, closes, disp_idx, "bear")
        if ob_idx is None:
            return None

        ob_open  = opens[ob_idx]
        ob_close = closes[ob_idx]
        ob_high  = highs[ob_idx]
        ob_low   = lows[ob_idx]

        # Minimum body filter
        body = abs(ob_close - ob_open)
        if body / max(ob_close, 1e-9) < min_ob_body_pct:
            return None

        # Displacement range (from OB close to displacement high)
        disp_high = max(highs[ob_idx: disp_idx + 1])
        disp_low  = min(lows[ob_idx: disp_idx + 1])

        # OTE: 61.8–79 % retracement from disp_high back toward ob_low
        span       = disp_high - ob_low
        ote_high   = round(disp_high - span * OTE_FIB_LOW,  2)   # 61.8 % retrace
        ote_low    = round(disp_high - span * OTE_FIB_HIGH, 2)   # 79.0 % retrace

        # OTE zone must overlap the OB (ICT requirement)
        if ote_high < ob_low or ote_low > ob_high:
            # Relax: use ob itself as the OTE zone
            ote_high = ob_high
            ote_low  = ob_low

        sl_price = round(ob_low * (1 - sl_buffer_pct), 2)
        tp_price = round(disp_high, 2)   # target: liquidity origin high

        quality = _ob_quality(ob_open, ob_close, ob_high, ob_low, candles, ob_idx, "LONG")
        strength = min(1.0, sweep_strength * 0.5 + quality * 0.5)

        return OrderBlock(
            direction="LONG",
            ob_high=round(ob_high, 2),
            ob_low=round(ob_low, 2),
            ote_high=ote_high,
            ote_low=ote_low,
            sl_price=sl_price,
            tp_price=tp_price,
            displacement_high=round(disp_high, 2),
            displacement_low=round(disp_low, 2),
            ob_candle_index=ob_idx,
            strength=round(strength, 3),
        )

    else:  # direction == "SHORT"
        disp_idx = _find_displacement(opens, closes, highs, lows, scan_start, scan_end, "bear")
        if disp_idx is None:
            return None

        ob_idx = _last_opposite_body(opens, closes, disp_idx, "bull")
        if ob_idx is None:
            return None

        ob_open  = opens[ob_idx]
        ob_close = closes[ob_idx]
        ob_high  = highs[ob_idx]
        ob_low   = lows[ob_idx]

        body = abs(ob_close - ob_open)
        if body / max(ob_close, 1e-9) < min_ob_body_pct:
            return None

        disp_high = max(highs[ob_idx: disp_idx + 1])
        disp_low  = min(lows[ob_idx: disp_idx + 1])

        span       = ob_high - disp_low
        ote_low    = round(disp_low + span * OTE_FIB_LOW,  2)   # 61.8 % retrace up
        ote_high   = round(disp_low + span * OTE_FIB_HIGH, 2)   # 79.0 % retrace up

        if ote_low > ob_high or ote_high < ob_low:
            ote_low  = ob_low
            ote_high = ob_high

        sl_price = round(ob_high * (1 + sl_buffer_pct), 2)
        tp_price = round(disp_low, 2)

        quality = _ob_quality(ob_open, ob_close, ob_high, ob_low, candles, ob_idx, "SHORT")
        strength = min(1.0, sweep_strength * 0.5 + quality * 0.5)

        return OrderBlock(
            direction="SHORT",
            ob_high=round(ob_high, 2),
            ob_low=round(ob_low, 2),
            ote_high=ote_high,
            ote_low=ote_low,
            sl_price=sl_price,
            tp_price=tp_price,
            displacement_high=round(disp_high, 2),
            displacement_low=round(disp_low, 2),
            ob_candle_index=ob_idx,
            strength=round(strength, 3),
        )


# ── OTE retracement math ──────────────────────────────────────────────────────

def compute_ote_zone(
    swing_low: float,
    swing_high: float,
    direction: str,
) -> tuple[float, float]:
    """Return (ote_low, ote_high) for a given displacement leg.

    For LONG: measures 61.8–79 % retracement from swing_high back toward swing_low.
    For SHORT: measures 61.8–79 % retracement from swing_low back toward swing_high.

    Example
    -------
    >>> lo, hi = compute_ote_zone(100.0, 110.0, "LONG")
    >>> assert 103 < lo < 105   # 79 % retrace → 110 - 10*0.79 = 102.1
    """
    span = swing_high - swing_low
    if span <= 0:
        return swing_low, swing_high

    if direction == "LONG":
        ote_high = round(swing_high - span * OTE_FIB_LOW,  2)
        ote_low  = round(swing_high - span * OTE_FIB_HIGH, 2)
        return ote_low, ote_high
    else:
        ote_low  = round(swing_low + span * OTE_FIB_LOW,  2)
        ote_high = round(swing_low + span * OTE_FIB_HIGH, 2)
        return ote_low, ote_high


def price_in_ote(price: float, ote_low: float, ote_high: float) -> bool:
    """Return True when price is inside the OTE zone."""
    return ote_low <= price <= ote_high


# ── Internal helpers ──────────────────────────────────────────────────────────

def _find_displacement(
    opens: list[float],
    closes: list[float],
    highs: list[float],
    lows: list[float],
    start: int,
    end: int,
    side: str,   # "bull" or "bear"
) -> int | None:
    """Find the most recent strong impulsive displacement candle in [start, end].

    A bull displacement: close > open, body >= 40 % of bar range, close is
    the highest close in the window.
    A bear displacement: close < open, body >= 40 % of bar range, close is
    the lowest close in the window.
    """
    best_idx: int | None = None
    best_size: float = 0.0

    for i in range(start, end + 1):
        body  = abs(closes[i] - opens[i])
        rng   = highs[i] - lows[i]
        if rng <= 0:
            continue
        body_ratio = body / rng
        if body_ratio < 0.40:
            continue

        if side == "bull" and closes[i] > opens[i] and body > best_size:
            best_idx, best_size = i, body
        elif side == "bear" and closes[i] < opens[i] and body > best_size:
            best_idx, best_size = i, body

    return best_idx


def _last_opposite_body(
    opens: list[float],
    closes: list[float],
    before_index: int,
    body_side: str,   # "bull" or "bear"
    lookback: int = 10,
) -> int | None:
    """Find the last candle with a body in body_side direction before before_index."""
    start = max(0, before_index - lookback)
    for i in range(before_index - 1, start - 1, -1):
        if body_side == "bear" and closes[i] < opens[i]:
            return i
        if body_side == "bull" and closes[i] > opens[i]:
            return i
    return None


def _ob_quality(
    ob_open: float,
    ob_close: float,
    ob_high: float,
    ob_low: float,
    candles: list[dict],
    ob_idx: int,
    direction: str,
) -> float:
    """Score OB quality 0.0–1.0 based on body-to-range ratio and position.

    Higher score = cleaner OB (large body, minimal wicks on the correct side).
    """
    rng = ob_high - ob_low
    if rng <= 0:
        return 0.5

    body_ratio = abs(ob_close - ob_open) / rng   # 0–1, higher = cleaner

    # Wick quality: for a LONG OB (bearish body), lower wick should be small
    if direction == "LONG":
        lower_wick = min(ob_open, ob_close) - ob_low
        wick_ratio = 1.0 - (lower_wick / rng)    # low lower wick = better
    else:
        upper_wick = ob_high - max(ob_open, ob_close)
        wick_ratio = 1.0 - (upper_wick / rng)

    # Volume confirmation: OB bar volume vs recent average
    vol_score = 0.5
    vols = [c["volume"] for c in candles[max(0, ob_idx - 10): ob_idx]]
    if vols:
        avg_vol = sum(vols) / len(vols)
        ob_vol = candles[ob_idx]["volume"]
        if avg_vol > 0:
            vol_score = min(1.0, ob_vol / avg_vol * 0.5)

    return round((body_ratio * 0.5 + wick_ratio * 0.3 + vol_score * 0.2), 3)
