"""Tests for the Stacked Sweep strategy.

Coverage:
- Happy path: stacked LONG sweep (swing_low + equal_lows) in uptrend → signal
- Happy path: stacked SHORT sweep (swing_high + equal_highs) in downtrend → signal
- Rejected: single level type only (not stacked)
- Rejected: stacked sweep but counter-trend (long in downtrend)
- Rejected: stacked sweep but counter-trend (short in uptrend)
- Rejected: fewer than 20 candles
- analyze_technical with active_strategy='stacked_sweep' returns STACKED_SWEEP type
- analyze_technical with active_strategy='stacked_sweep', no signal → None
"""

from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

from analysis.liquidity_sweep import LiquidityLevel, LiquidityLevelType, LiquiditySweepSignal
from analysis.technical import (
    SetupType,
    Trend,
    _analyze_stacked_sweep,
    analyze_technical,
)
from models.orders import OHLCV
from models.trade_plan import TradeDirection


# ── Helpers ───────────────────────────────────────────────────────────────────

def _candle(o: float, h: float, l: float, c: float, v: float = 1000.0) -> OHLCV:
    return OHLCV(open=o, high=h, low=l, close=c, volume=v, timestamp="2026-01-01T00:00:00")


def _trending_up(n: int = 30, start: float = 100.0, step: float = 0.5) -> list[OHLCV]:
    """Monotonically rising candles so HTF EMA20 trend = uptrend."""
    candles = []
    for i in range(n):
        base = start + i * step
        candles.append(_candle(base, base + 0.8, base - 0.3, base + 0.5))
    return candles


def _trending_down(n: int = 30, start: float = 130.0, step: float = 0.5) -> list[OHLCV]:
    """Monotonically falling candles so HTF EMA20 trend = downtrend."""
    candles = []
    for i in range(n):
        base = start - i * step
        candles.append(_candle(base, base + 0.3, base - 0.8, base - 0.5))
    return candles


def _make_sweep_signal(direction: str, entry: float, stop_loss: float,
                       level_types: list[LiquidityLevelType]) -> LiquiditySweepSignal:
    levels = [LiquidityLevel(price=entry, level_type=lt, candle_index=5) for lt in level_types]
    strength = 0.65
    return LiquiditySweepSignal(
        direction=direction,
        entry=entry,
        stop_loss=stop_loss,
        swept_levels=levels,
        strength=strength,
        description=f"Test sweep {direction}",
    )


# ── Happy path ────────────────────────────────────────────────────────────────

def test_stacked_long_sweep_uptrend_returns_signal():
    """swing_low + equal_lows in uptrend → LONG signal."""
    htf = _trending_up(30)
    ltf = _trending_up(25)

    signal = _make_sweep_signal(
        "LONG", entry=115.0, stop_loss=113.5,
        level_types=[LiquidityLevelType.SWING_LOW, LiquidityLevelType.EQUAL_LOWS]
    )

    with patch("analysis.technical.find_all_liquidity_levels", return_value=signal.swept_levels), \
         patch("analysis.technical.detect_liquidity_sweep", return_value=signal):
        result = _analyze_stacked_sweep(htf, ltf, min_rr=1.5, max_sl_pct=5.0)

    assert result is not None
    assert result.direction == TradeDirection.LONG
    assert result.setup_type == SetupType.STACKED_SWEEP
    assert result.entry == pytest.approx(115.0)
    assert result.tp1 > result.entry
    assert result.rr_ratio >= 1.5 - 1e-3
    assert "daily_up" in result.confirmations
    assert "swing_low" in result.confirmations
    assert "equal_lows" in result.confirmations


def test_stacked_short_sweep_downtrend_returns_signal():
    """swing_high + equal_highs in downtrend → SHORT signal."""
    htf = _trending_down(30)
    ltf = _trending_down(25)

    signal = _make_sweep_signal(
        "SHORT", entry=115.0, stop_loss=116.5,
        level_types=[LiquidityLevelType.SWING_HIGH, LiquidityLevelType.EQUAL_HIGHS]
    )

    with patch("analysis.technical.find_all_liquidity_levels", return_value=signal.swept_levels), \
         patch("analysis.technical.detect_liquidity_sweep", return_value=signal):
        result = _analyze_stacked_sweep(htf, ltf, min_rr=1.5, max_sl_pct=5.0)

    assert result is not None
    assert result.direction == TradeDirection.SHORT
    assert result.setup_type == SetupType.STACKED_SWEEP
    assert result.tp1 < result.entry
    assert "daily_down" in result.confirmations


# ── Rejection: not stacked ────────────────────────────────────────────────────

def test_single_level_type_rejected():
    """Only swing_low (one level type) → rejected."""
    htf = _trending_up(30)
    ltf = _trending_up(25)

    signal = _make_sweep_signal(
        "LONG", entry=115.0, stop_loss=113.5,
        level_types=[LiquidityLevelType.SWING_LOW]  # only 1 type
    )

    with patch("analysis.technical.find_all_liquidity_levels", return_value=signal.swept_levels), \
         patch("analysis.technical.detect_liquidity_sweep", return_value=signal):
        result = _analyze_stacked_sweep(htf, ltf, min_rr=1.5, max_sl_pct=5.0)

    assert result is None


def test_same_level_type_twice_rejected():
    """Two swing_low levels (same type repeated) → still only 1 distinct type → rejected."""
    htf = _trending_up(30)
    ltf = _trending_up(25)

    signal = _make_sweep_signal(
        "LONG", entry=115.0, stop_loss=113.5,
        level_types=[LiquidityLevelType.SWING_LOW, LiquidityLevelType.SWING_LOW]
    )

    with patch("analysis.technical.find_all_liquidity_levels", return_value=signal.swept_levels), \
         patch("analysis.technical.detect_liquidity_sweep", return_value=signal):
        result = _analyze_stacked_sweep(htf, ltf, min_rr=1.5, max_sl_pct=5.0)

    assert result is None


# ── Rejection: counter-trend ──────────────────────────────────────────────────

def test_long_sweep_in_downtrend_rejected():
    """Stacked LONG sweep but HTF is downtrend → rejected."""
    htf = _trending_down(30)
    ltf = _trending_down(25)

    signal = _make_sweep_signal(
        "LONG", entry=115.0, stop_loss=113.5,
        level_types=[LiquidityLevelType.SWING_LOW, LiquidityLevelType.EQUAL_LOWS]
    )

    with patch("analysis.technical.find_all_liquidity_levels", return_value=signal.swept_levels), \
         patch("analysis.technical.detect_liquidity_sweep", return_value=signal):
        result = _analyze_stacked_sweep(htf, ltf, min_rr=1.5, max_sl_pct=5.0)

    assert result is None


def test_short_sweep_in_uptrend_rejected():
    """Stacked SHORT sweep but HTF is uptrend → rejected."""
    htf = _trending_up(30)
    ltf = _trending_up(25)

    signal = _make_sweep_signal(
        "SHORT", entry=115.0, stop_loss=116.5,
        level_types=[LiquidityLevelType.SWING_HIGH, LiquidityLevelType.EQUAL_HIGHS]
    )

    with patch("analysis.technical.find_all_liquidity_levels", return_value=signal.swept_levels), \
         patch("analysis.technical.detect_liquidity_sweep", return_value=signal):
        result = _analyze_stacked_sweep(htf, ltf, min_rr=1.5, max_sl_pct=5.0)

    assert result is None


# ── Rejection: insufficient candles ──────────────────────────────────────────

def test_too_few_candles_rejected():
    htf = _trending_up(30)
    ltf = _trending_up(10)  # < 20 required

    result = _analyze_stacked_sweep(htf, ltf, min_rr=1.5, max_sl_pct=5.0)
    assert result is None


def test_no_levels_rejected():
    htf = _trending_up(30)
    ltf = _trending_up(25)

    with patch("analysis.technical.find_all_liquidity_levels", return_value=[]):
        result = _analyze_stacked_sweep(htf, ltf, min_rr=1.5, max_sl_pct=5.0)

    assert result is None


# ── analyze_technical integration ────────────────────────────────────────────

def test_analyze_technical_stacked_sweep_mode():
    """analyze_technical with active_strategy='stacked_sweep' returns STACKED_SWEEP."""
    htf = _trending_up(30)
    ltf = _trending_up(25)

    signal = _make_sweep_signal(
        "LONG", entry=115.0, stop_loss=113.5,
        level_types=[LiquidityLevelType.SWING_LOW, LiquidityLevelType.EQUAL_LOWS]
    )

    with patch("analysis.technical.find_all_liquidity_levels", return_value=signal.swept_levels), \
         patch("analysis.technical.detect_liquidity_sweep", return_value=signal):
        result = analyze_technical(htf, ltf, active_strategy="stacked_sweep")

    assert result is not None
    assert result.setup_type == SetupType.STACKED_SWEEP


def test_analyze_technical_stacked_sweep_no_signal_returns_none():
    """analyze_technical with active_strategy='stacked_sweep', no sweep → None."""
    htf = _trending_up(30)
    ltf = _trending_up(25)

    with patch("analysis.technical.find_all_liquidity_levels", return_value=[]), \
         patch("analysis.technical.detect_liquidity_sweep", return_value=None):
        result = analyze_technical(htf, ltf, active_strategy="stacked_sweep")

    assert result is None
