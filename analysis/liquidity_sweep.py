"""Liquidity sweep detection — swing highs/lows, equal highs/lows, session levels.

A liquidity sweep: price wicks BEYOND a known liquidity pool (stop cluster)
but CLOSES BACK inside. Signals institutional absorption before reversal.

Three pool types detected:
1. Swing high/low — N-bar extremes where retail stops cluster.
2. Equal highs/lows (EQH/EQL) — SMC/ICT engineered liquidity at clustered levels.
3. Session levels — previous day high/low (high-volume reference points).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd

from models.orders import OHLCV


class LiquidityLevelType(str, Enum):
    SWING_HIGH = "swing_high"
    SWING_LOW = "swing_low"
    EQUAL_HIGHS = "equal_highs"    # EQH — 2+ swing highs within tolerance
    EQUAL_LOWS = "equal_lows"      # EQL — 2+ swing lows within tolerance
    SESSION_HIGH = "session_high"  # Previous day high
    SESSION_LOW = "session_low"    # Previous day low


# Strength bonus contributed per level type — session and equal levels are higher-conviction.
_LEVEL_STRENGTH_BONUS: dict[LiquidityLevelType, float] = {
    LiquidityLevelType.SWING_HIGH: 0.10,
    LiquidityLevelType.SWING_LOW: 0.10,
    LiquidityLevelType.EQUAL_HIGHS: 0.20,
    LiquidityLevelType.EQUAL_LOWS: 0.20,
    LiquidityLevelType.SESSION_HIGH: 0.20,
    LiquidityLevelType.SESSION_LOW: 0.20,
}

_LOW_LEVEL_TYPES = frozenset({
    LiquidityLevelType.SWING_LOW,
    LiquidityLevelType.EQUAL_LOWS,
    LiquidityLevelType.SESSION_LOW,
})

_HIGH_LEVEL_TYPES = frozenset({
    LiquidityLevelType.SWING_HIGH,
    LiquidityLevelType.EQUAL_HIGHS,
    LiquidityLevelType.SESSION_HIGH,
})


@dataclass
class LiquidityLevel:
    price: float
    level_type: LiquidityLevelType
    candle_index: int  # index in the LTF candle list where the level was formed


@dataclass
class LiquiditySweepSignal:
    direction: str                  # "LONG" or "SHORT"
    entry: float
    stop_loss: float
    swept_levels: list[LiquidityLevel]
    strength: float                 # 0.0–1.0
    description: str

    @property
    def sl_distance(self) -> float:
        return abs(self.entry - self.stop_loss)


# ── DataFrame helper ──────────────────────────────────────────────────────────

def _to_df(candles: list[OHLCV]) -> pd.DataFrame:
    return pd.DataFrame({
        "open":   [c.open   for c in candles],
        "high":   [c.high   for c in candles],
        "low":    [c.low    for c in candles],
        "close":  [c.close  for c in candles],
        "volume": [c.volume for c in candles],
    })


# ── Level detectors ───────────────────────────────────────────────────────────

def detect_swing_levels(
    df: pd.DataFrame,
    swing_lookback: int = 5,
    max_levels: int = 10,
) -> list[LiquidityLevel]:
    """Find recent swing highs and lows — N-bar extremes where stops cluster.

    A swing high at index i: df.high[i] is the max of the ±swing_lookback window.
    A swing low  at index i: df.low[i]  is the min of the ±swing_lookback window.

    Returns at most max_levels most-recent swing points.
    """
    levels: list[LiquidityLevel] = []
    n = swing_lookback

    for i in range(n, len(df) - n):
        window = df.iloc[i - n: i + n + 1]

        if float(df["high"].iloc[i]) == float(window["high"].max()):
            levels.append(LiquidityLevel(
                price=float(df["high"].iloc[i]),
                level_type=LiquidityLevelType.SWING_HIGH,
                candle_index=i,
            ))

        if float(df["low"].iloc[i]) == float(window["low"].min()):
            levels.append(LiquidityLevel(
                price=float(df["low"].iloc[i]),
                level_type=LiquidityLevelType.SWING_LOW,
                candle_index=i,
            ))

    # Return most-recent levels first (ancient history is less relevant).
    levels.sort(key=lambda x: x.candle_index, reverse=True)
    return levels[:max_levels]


def detect_equal_levels(
    df: pd.DataFrame,
    tolerance_pct: float = 0.002,  # 0.2 % — two levels are "equal" if within this
    lookback: int = 40,
) -> list[LiquidityLevel]:
    """Detect Equal Highs (EQH) and Equal Lows (EQL) — SMC engineered liquidity.

    EQH: two or more swing highs within tolerance_pct of each other → stop cluster above.
    EQL: two or more swing lows  within tolerance_pct of each other → stop cluster below.
    """
    scan_df = df.tail(lookback).reset_index(drop=True)
    n = 3  # narrower lookback for equal-level swing detection
    highs: list[float] = []
    lows:  list[float] = []

    for i in range(n, len(scan_df) - n):
        window = scan_df.iloc[i - n: i + n + 1]
        if float(scan_df["high"].iloc[i]) == float(window["high"].max()):
            highs.append(float(scan_df["high"].iloc[i]))
        if float(scan_df["low"].iloc[i]) == float(window["low"].min()):
            lows.append(float(scan_df["low"].iloc[i]))

    levels: list[LiquidityLevel] = []
    last_idx = len(df) - 1

    eqh = _find_cluster(highs, tolerance_pct)
    if eqh is not None:
        levels.append(LiquidityLevel(
            price=eqh,
            level_type=LiquidityLevelType.EQUAL_HIGHS,
            candle_index=last_idx,
        ))

    eql = _find_cluster(lows, tolerance_pct)
    if eql is not None:
        levels.append(LiquidityLevel(
            price=eql,
            level_type=LiquidityLevelType.EQUAL_LOWS,
            candle_index=last_idx,
        ))

    return levels


def _find_cluster(prices: list[float], tolerance_pct: float) -> float | None:
    """Return the mean price of the most prominent cluster (≥2 prices within tolerance)."""
    if len(prices) < 2:
        return None

    sorted_prices = sorted(prices)
    best_cluster: list[float] = []

    i = 0
    while i < len(sorted_prices):
        cluster = [sorted_prices[i]]
        j = i + 1
        while j < len(sorted_prices):
            if abs(sorted_prices[j] - sorted_prices[i]) / max(sorted_prices[i], 1e-9) <= tolerance_pct:
                cluster.append(sorted_prices[j])
                j += 1
            else:
                break
        if len(cluster) >= 2 and len(cluster) > len(best_cluster):
            best_cluster = cluster
        i = j if j > i else i + 1

    return sum(best_cluster) / len(best_cluster) if best_cluster else None


def detect_session_levels(htf_daily_candles: list[OHLCV]) -> list[LiquidityLevel]:
    """Extract previous day high/low from daily HTF candles.

    Returns SESSION_HIGH and SESSION_LOW for the prior completed day.
    Returns empty list when fewer than 2 daily candles are available.
    """
    if len(htf_daily_candles) < 2:
        return []

    prev = htf_daily_candles[-2]  # second-to-last = previous completed day
    return [
        LiquidityLevel(
            price=float(prev.high),
            level_type=LiquidityLevelType.SESSION_HIGH,
            candle_index=0,
        ),
        LiquidityLevel(
            price=float(prev.low),
            level_type=LiquidityLevelType.SESSION_LOW,
            candle_index=0,
        ),
    ]


def find_all_liquidity_levels(
    ltf_candles: list[OHLCV],
    htf_daily_candles: list[OHLCV],
) -> list[LiquidityLevel]:
    """Aggregate all liquidity levels: swing, equal, and session."""
    if len(ltf_candles) < 15:
        return []

    ltf_df = _to_df(ltf_candles)
    levels: list[LiquidityLevel] = []
    levels.extend(detect_swing_levels(ltf_df))
    levels.extend(detect_equal_levels(ltf_df))
    levels.extend(detect_session_levels(htf_daily_candles))
    return levels


# ── Sweep detector ────────────────────────────────────────────────────────────

def detect_liquidity_sweep(
    ltf_df: pd.DataFrame,
    levels: list[LiquidityLevel],
    min_sweep_pct: float = 0.001,  # wick must breach level by ≥ 0.1 %
) -> LiquiditySweepSignal | None:
    """Detect if a recent candle swept a liquidity level and closed back inside.

    LONG setup (sweep of lows):
      candle.low   < level.price × (1 − min_sweep_pct)  [wicked below]
      candle.close > level.price                         [closed back above]

    SHORT setup (sweep of highs):
      candle.high  > level.price × (1 + min_sweep_pct)  [wicked above]
      candle.close < level.price                         [closed back below]

    Checks the 2nd and 3rd most-recent candles (skip the forming candle).
    Returns the most recent valid sweep signal found.
    """
    if len(ltf_df) < 5 or not levels:
        return None

    for candle_offset in [2, 3]:
        idx = len(ltf_df) - candle_offset
        if idx < 0:
            continue

        candle       = ltf_df.iloc[idx]
        candle_low   = float(candle["low"])
        candle_high  = float(candle["high"])
        candle_close = float(candle["close"])

        # ── LONG sweep: swept lows, closed above ──
        swept_lows = [
            lv for lv in levels
            if lv.level_type in _LOW_LEVEL_TYPES
            and candle_low   < lv.price * (1 - min_sweep_pct)
            and candle_close > lv.price
        ]
        if swept_lows:
            stop_loss = candle_low * 0.998  # small buffer below wick extreme
            entry = candle_close
            if entry <= stop_loss:
                continue
            strength = _compute_sweep_strength(swept_lows, candle, ltf_df, "LONG")
            return LiquiditySweepSignal(
                direction="LONG",
                entry=entry,
                stop_loss=stop_loss,
                swept_levels=swept_lows,
                strength=strength,
                description=_describe_sweep("LONG", swept_lows, entry, stop_loss),
            )

        # ── SHORT sweep: swept highs, closed below ──
        swept_highs = [
            lv for lv in levels
            if lv.level_type in _HIGH_LEVEL_TYPES
            and candle_high  > lv.price * (1 + min_sweep_pct)
            and candle_close < lv.price
        ]
        if swept_highs:
            stop_loss = candle_high * 1.002
            entry = candle_close
            if entry >= stop_loss:
                continue
            strength = _compute_sweep_strength(swept_highs, candle, ltf_df, "SHORT")
            return LiquiditySweepSignal(
                direction="SHORT",
                entry=entry,
                stop_loss=stop_loss,
                swept_levels=swept_highs,
                strength=strength,
                description=_describe_sweep("SHORT", swept_highs, entry, stop_loss),
            )

    return None


# ── Strength scoring ──────────────────────────────────────────────────────────

def _compute_sweep_strength(
    swept_levels: list[LiquidityLevel],
    candle: pd.Series,
    ltf_df: pd.DataFrame,
    direction: str,
) -> float:
    """Compute sweep signal strength (0.0–1.0).

    Weights:
    - Base: 0.45 (above swing's 0.40 — sweeps are precision setups)
    - Best level type bonus: +0.10–0.20
    - Multiple levels swept simultaneously: +0.10
    - Wick rejection quality (wick beyond level / candle range): 0.0–0.15
    - Volume surge (last candle vs 10-bar avg): +0.05
    """
    strength = 0.45

    best_bonus = max(
        _LEVEL_STRENGTH_BONUS.get(lv.level_type, 0.0)
        for lv in swept_levels
    )
    strength += best_bonus

    if len(swept_levels) >= 2:
        strength += 0.10  # stacked liquidity — stronger conviction

    candle_range = float(candle["high"]) - float(candle["low"])
    if candle_range > 0:
        level_price = swept_levels[0].price
        if direction == "LONG":
            wick_beyond = max(0.0, level_price - float(candle["low"]))
        else:
            wick_beyond = max(0.0, float(candle["high"]) - level_price)
        strength += min(0.15, (wick_beyond / candle_range) * 0.30)

    avg_vol = float(ltf_df["volume"].tail(10).mean()) or 1.0
    if float(candle["volume"]) >= avg_vol * 1.2:
        strength += 0.05

    return min(1.0, strength)


def _describe_sweep(
    direction: str,
    swept_levels: list[LiquidityLevel],
    entry: float,
    stop_loss: float,
) -> str:
    names  = ", ".join(lv.level_type.value.replace("_", " ") for lv in swept_levels)
    prices = ", ".join(f"{lv.price:.2f}" for lv in swept_levels)
    return (
        f"Liquidity sweep {direction} | swept: {names} @ {prices} "
        f"| entry={entry:.2f} SL={stop_loss:.2f}"
    )
