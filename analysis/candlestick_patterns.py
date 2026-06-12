"""Candlestick pattern detection — hammer, engulfing, doji, confirmation.

Pure functions operating on OHLCV lists. No side effects, no state.
Designed for 15m/5m scalping on index options.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd

from models.orders import OHLCV


class CandlestickPattern(str, Enum):
    HAMMER = "hammer"
    INVERTED_HAMMER = "inverted_hammer"
    BULLISH_ENGULFING = "bullish_engulfing"
    BEARISH_ENGULFING = "bearish_engulfing"
    DOJI = "doji"


@dataclass
class PatternSignal:
    """A detected candlestick pattern with confirmation."""

    pattern: CandlestickPattern
    pattern_index: int  # index in the candle list of the pattern candle
    confirmation_index: int  # index of the confirmation candle
    direction: str  # "LONG" or "SHORT"
    entry: float
    stop_loss: float
    strength: float  # 0.0-1.0
    description: str

    @property
    def sl_distance(self) -> float:
        return abs(self.entry - self.stop_loss)


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


# ── Core pattern detectors ───────────────────────────────────────────


def _body(row) -> float:
    """Absolute body size."""
    return abs(row["close"] - row["open"])


def _upper_wick(row) -> float:
    return row["high"] - max(row["close"], row["open"])


def _lower_wick(row) -> float:
    return min(row["close"], row["open"]) - row["low"]


def _total_range(row) -> float:
    return row["high"] - row["low"]


def _is_bullish(row) -> bool:
    return row["close"] > row["open"]


def _is_bearish(row) -> bool:
    return row["close"] < row["open"]


def is_hammer(candle: pd.Series) -> bool:
    """Detect a hammer: small body at top, long lower wick ≥2x body, small upper wick.

    Hammer signals reversal from downtrend → bullish.
    """
    body = _body(candle)
    total = _total_range(candle)
    if total <= 0 or body <= 0:
        return False

    lower = _lower_wick(candle)
    upper = _upper_wick(candle)

    # Lower wick must be ≥ 2x body
    if lower < body * 2.0:
        return False
    # Upper wick must be ≤ 30% of total range (small head)
    if upper > total * 0.30:
        return False
    # Body must be in upper third of range
    body_center = (candle["open"] + candle["close"]) / 2
    if body_center < candle["low"] + total * 0.6:
        return False
    # Body must be at least 0.05% of price (avoid micro-noise)
    avg_price = (candle["high"] + candle["low"]) / 2
    if body < avg_price * 0.0005:
        return False

    return True


def is_inverted_hammer(candle: pd.Series) -> bool:
    """Detect an inverted hammer / shooting star: small body at bottom,
    long upper wick ≥2x body, small lower wick.

    Signals reversal from uptrend → bearish.
    """
    body = _body(candle)
    total = _total_range(candle)
    if total <= 0 or body <= 0:
        return False

    upper = _upper_wick(candle)
    lower = _lower_wick(candle)

    if upper < body * 2.0:
        return False
    if lower > total * 0.30:
        return False

    body_center = (candle["open"] + candle["close"]) / 2
    if body_center > candle["low"] + total * 0.4:
        return False

    avg_price = (candle["high"] + candle["low"]) / 2
    if body < avg_price * 0.0005:
        return False

    return True


def is_bullish_engulfing(prev: pd.Series, curr: pd.Series) -> bool:
    """Bullish engulfing: red candle followed by larger green candle
    that fully engulfs the red body.

    Signals reversal from downtrend → bullish.
    """
    if not _is_bearish(prev):
        return False
    if not _is_bullish(curr):
        return False

    # Current body must fully engulf previous body
    if curr["open"] >= prev["close"]:
        return False  # didn't gap below
    if curr["close"] <= prev["open"]:
        return False  # didn't close above

    # Current body should be ≥ 1.2x previous body (conviction)
    if _body(curr) < _body(prev) * 1.1:
        return False

    # Volume should be higher (if available)
    if curr.get("volume", 0) > 0 and prev.get("volume", 0) > 0:
        if curr["volume"] < prev["volume"]:
            return False

    return True


def is_bearish_engulfing(prev: pd.Series, curr: pd.Series) -> bool:
    """Bearish engulfing: green candle followed by larger red candle
    that fully engulfs the green body.

    Signals reversal from uptrend → bearish.
    """
    if not _is_bullish(prev):
        return False
    if not _is_bearish(curr):
        return False

    if curr["open"] <= prev["close"]:
        return False
    if curr["close"] >= prev["open"]:
        return False

    if _body(curr) < _body(prev) * 1.1:
        return False

    if curr.get("volume", 0) > 0 and prev.get("volume", 0) > 0:
        if curr["volume"] < prev["volume"]:
            return False

    return True


def is_doji(candle: pd.Series) -> bool:
    """Detect a doji: very small body relative to total range.

    Body ≤ 10% of total range, or ≤ 0.05% of price.
    """
    body = _body(candle)
    total = _total_range(candle)
    if total <= 0:
        return False

    avg_price = (candle["high"] + candle["low"]) / 2

    # Doji: body ≤ 10% of range OR body ≤ 0.05% of price
    if body <= total * 0.10:
        return True
    if avg_price > 0 and body <= avg_price * 0.0005:
        return True

    return False


# ── Confirmation helpers ────────────────────────────────────────────


def _confirms_hammer(confirmation: pd.Series, hammer: pd.Series) -> bool:
    """Confirmation candle must be bullish and close above hammer's body."""
    if not _is_bullish(confirmation):
        return False
    hammer_body_high = max(hammer["close"], hammer["open"])
    return confirmation["close"] > hammer_body_high


def _confirms_inverted_hammer(confirmation: pd.Series, inv_hammer: pd.Series) -> bool:
    """Confirmation must be bearish (red) and close below inverted hammer's body."""
    if not _is_bearish(confirmation):
        return False
    inv_hammer_body_low = min(inv_hammer["close"], inv_hammer["open"])
    return confirmation["close"] < inv_hammer_body_low


def _confirms_engulfing_long(confirmation: pd.Series, engulfing: pd.Series) -> bool:
    """Confirmation must close above the engulfing candle's body."""
    return confirmation["close"] > max(engulfing["close"], engulfing["open"])


def _confirms_engulfing_short(confirmation: pd.Series, engulfing: pd.Series) -> bool:
    """Confirmation must close below the engulfing candle's body."""
    return confirmation["close"] < min(engulfing["close"], engulfing["open"])


def _confirms_doji(confirmation: pd.Series) -> bool:
    """Doji confirmation: a clear directional candle (not another doji)."""
    body = _body(confirmation)
    total = _total_range(confirmation)
    if total <= 0:
        return False
    # Must have real body > 20% of range
    return body > total * 0.20


# ── Pullback detection (for trailing) ───────────────────────────────


def detect_pullback(
    candles: list[OHLCV],
    direction: str,
    from_index: int = 0,
) -> int | None:
    """Find the first pullback candle after from_index.

    For LONG: a candle whose low is below the previous candle's low.
    For SHORT: a candle whose high is above the previous candle's high.

    Returns the index of the pullback candle, or None if no pullback found.
    """
    if len(candles) < from_index + 3:
        return None

    df = _ohlcv_to_df(candles)
    for i in range(from_index + 2, len(df)):
        if direction == "LONG":
            if df["low"].iloc[i] < df["low"].iloc[i - 1]:
                return i
        else:
            if df["high"].iloc[i] > df["high"].iloc[i - 1]:
                return i
    return None


# ── Main scanner: scan for patterns ─────────────────────────────────


def scan_candlestick_patterns(
    candles: list[OHLCV],
    trend: str,  # "UPTREND" | "DOWNTREND" | "RANGE"
    *,
    min_confidence: float = 0.5,
) -> PatternSignal | None:
    """Scan the last few candles for actionable candlestick patterns.

    Args:
        candles: OHLCV candle list (15m or 5m)
        trend: current market trend from detect_trend()
        min_confidence: minimum pattern strength to accept

    Returns:
        PatternSignal if found with confirmation, else None.
    """
    if len(candles) < 5:
        return None

    df = _ohlcv_to_df(candles)
    last_idx = len(df) - 1

    # We need at least: pattern candle + confirmation candle + current candle
    # So we look at candles[last-2] for pattern, candles[last-1] for confirmation
    if last_idx < 3:
        return None

    trend_upper = trend.upper()

    # ── Scan for patterns at recent candle positions ──
    # We check: pattern at i, confirmation at i+1
    for pattern_idx in [last_idx - 1, last_idx - 2]:
        if pattern_idx < 0:
            continue

        pattern_candle = df.iloc[pattern_idx]
        conf_idx = pattern_idx + 1
        if conf_idx > last_idx:
            continue
        confirmation = df.iloc[conf_idx]

        # ── 1. HAMMER (downtrend → bullish reversal) ──
        if trend_upper in ("DOWNTREND", "RANGE"):
            if is_hammer(pattern_candle) and _confirms_hammer(confirmation, pattern_candle):
                entry = confirmation["close"]
                stop_loss = confirmation["low"]
                if entry > stop_loss:
                    strength = _compute_strength(pattern_candle, "hammer", trend)
                    if strength >= min_confidence:
                        return PatternSignal(
                            pattern=CandlestickPattern.HAMMER,
                            pattern_index=pattern_idx,
                            confirmation_index=conf_idx,
                            direction="LONG",
                            entry=entry,
                            stop_loss=stop_loss,
                            strength=strength,
                            description=f"Hammer at {pattern_candle['low']:.2f} confirmed by bullish candle closing at {entry:.2f}",
                        )

        # ── 2. INVERTED HAMMER (uptrend → bearish reversal) ──
        if trend_upper in ("UPTREND", "RANGE"):
            if is_inverted_hammer(pattern_candle) and _confirms_inverted_hammer(confirmation, pattern_candle):
                entry = confirmation["close"]
                stop_loss = confirmation["high"]
                if stop_loss > entry:
                    strength = _compute_strength(pattern_candle, "inverted_hammer", trend)
                    if strength >= min_confidence:
                        return PatternSignal(
                            pattern=CandlestickPattern.INVERTED_HAMMER,
                            pattern_index=pattern_idx,
                            confirmation_index=conf_idx,
                            direction="SHORT",
                            entry=entry,
                            stop_loss=stop_loss,
                            strength=strength,
                            description=f"Inverted hammer at {pattern_candle['high']:.2f} confirmed by bearish candle closing at {entry:.2f}",
                        )

        # ── 3. BULLISH ENGULFING (downtrend → bullish) ──
        if trend_upper in ("DOWNTREND", "RANGE"):
            # Engulfing needs two candles: prev (red) and curr (green)
            prev_idx = pattern_idx - 1
            if prev_idx >= 0:
                prev = df.iloc[prev_idx]
                if is_bullish_engulfing(prev, pattern_candle):
                    if confirmation["close"] > max(pattern_candle["close"], pattern_candle["open"]):
                        entry = confirmation["close"]
                        stop_loss = confirmation["low"]
                        if entry > stop_loss:
                            strength = _compute_strength(pattern_candle, "bullish_engulfing", trend)
                            if strength >= min_confidence:
                                return PatternSignal(
                                    pattern=CandlestickPattern.BULLISH_ENGULFING,
                                    pattern_index=pattern_idx,
                                    confirmation_index=conf_idx,
                                    direction="LONG",
                                    entry=entry,
                                    stop_loss=stop_loss,
                                    strength=strength,
                                    description=f"Bullish engulfing confirmed, entry={entry:.2f} SL={stop_loss:.2f}",
                                )

        # ── 4. BEARISH ENGULFING (uptrend → bearish) ──
        if trend_upper in ("UPTREND", "RANGE"):
            prev_idx = pattern_idx - 1
            if prev_idx >= 0:
                prev = df.iloc[prev_idx]
                if is_bearish_engulfing(prev, pattern_candle):
                    if confirmation["close"] < min(pattern_candle["close"], pattern_candle["open"]):
                        entry = confirmation["close"]
                        stop_loss = confirmation["high"]
                        if stop_loss > entry:
                            strength = _compute_strength(pattern_candle, "bearish_engulfing", trend)
                            if strength >= min_confidence:
                                return PatternSignal(
                                    pattern=CandlestickPattern.BEARISH_ENGULFING,
                                    pattern_index=pattern_idx,
                                    confirmation_index=conf_idx,
                                    direction="SHORT",
                                    entry=entry,
                                    stop_loss=stop_loss,
                                    strength=strength,
                                    description=f"Bearish engulfing confirmed, entry={entry:.2f} SL={stop_loss:.2f}",
                                )

    # ── 5. DOJI → wait for next pattern (signal to prepare) ──
    # Doji itself doesn't generate a trade — it signals to watch for hammer/engulfing.
    # We return None here; the caller should note the doji and watch next candles.
    # (Doji logic is handled in the _analyze_candlestick_scalp wrapper)

    return None


def find_recent_doji(candles: list[OHLCV], lookback: int = 5) -> int | None:
    """Find the most recent doji candle index within lookback range.

    Returns index or None. Used to set context for upcoming pattern detection.
    """
    if len(candles) < 3:
        return None
    df = _ohlcv_to_df(candles)
    start = max(0, len(df) - lookback)
    for i in range(len(df) - 1, start - 1, -1):
        if is_doji(df.iloc[i]):
            return i
    return None


# ── Strength computation ─────────────────────────────────────────────


def _compute_strength(
    candle: pd.Series,
    pattern_name: str,
    trend: str,
) -> float:
    """Compute a 0-1 signal strength for the detected pattern."""
    strength = 0.40  # base for any detected pattern

    body = _body(candle)
    total = _total_range(candle)

    if total > 0:
        # Larger wick-to-body ratio = stronger signal
        if "hammer" in pattern_name:
            wick_ratio = _lower_wick(candle) / max(body, 0.001) if "inverted" not in pattern_name else _upper_wick(candle) / max(body, 0.001)
            strength += min(0.25, wick_ratio * 0.05)

        if "engulfing" in pattern_name:
            strength += 0.15  # engulfing is inherently stronger

    # Trend alignment bonus
    trend_upper = trend.upper()
    if pattern_name in ("hammer", "bullish_engulfing") and trend_upper == "DOWNTREND":
        strength += 0.20  # reversal in downtrend = strong
    elif pattern_name in ("inverted_hammer", "bearish_engulfing") and trend_upper == "UPTREND":
        strength += 0.20
    elif trend_upper == "RANGE":
        strength += 0.05  # range patterns are weaker

    # Volume: higher volume = stronger (if available)
    if candle.get("volume", 0) > 0:
        strength += 0.10

    return min(1.0, strength)
