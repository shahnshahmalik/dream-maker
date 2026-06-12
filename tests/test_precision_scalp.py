"""Tests for precision scalping technical checks."""

import pytest
from models.orders import OHLCV
from models.trade_plan import TradeDirection
from analysis.technical import (
    check_pullback_to_ema,
    check_volume_surge,
    score_setup,
    TechnicalContext,
    SetupType,
    Trend,
)


def _make_candles(prices: list[float], volumes: list[float] | None = None) -> list[OHLCV]:
    """Build OHLCV candles from close prices. OHLC all set to same price for simplicity."""
    from datetime import datetime, timezone, timedelta
    if volumes is None:
        volumes = [100] * len(prices)
    candles = []
    base_ts = datetime(2026, 6, 5, 10, 0, tzinfo=timezone.utc)
    for i, p in enumerate(prices):
        candles.append(OHLCV(
            timestamp=base_ts + timedelta(minutes=15 * i),
            open=p, high=p * 1.01, low=p * 0.99, close=p,
            volume=volumes[i],
        ))
    return candles


def _make_ctx(**kwargs) -> TechnicalContext:
    defaults = dict(
        htf_trend=Trend.UPTREND,
        ltf_trend=Trend.UPTREND,
        above_200ema=True,
        support=100,
        resistance=200,
        ltf_aligned=True,
        entry=50.0,
        stop_loss=45.0,
        tp1=60.0,
        tp2=65.0,
        rr_ratio=2.0,
        direction=TradeDirection.LONG,
        bias_source="test",
        signal_strength=0.60,
        setup_type=SetupType.MOMENTUM_SCALP,
        confirmations=["test"],
        ltf_range=5.0,
    )
    defaults.update(kwargs)
    return TechnicalContext(**defaults)


# ── check_pullback_to_ema ──

def test_pullback_long_near_ema():
    """Price near EMA9 → pullback OK for LONG."""
    # EMA9 will be ~50 with these prices
    candles = _make_candles([50] * 15)
    ok, reason = check_pullback_to_ema(candles, TradeDirection.LONG)
    assert ok, reason


def test_pullback_long_far_above_ema():
    """Price far above EMA9 → pullback rejected for LONG."""
    # First 14 candles at 50, last one at 55 (+10%)
    prices = [50] * 14 + [55]
    candles = _make_candles(prices)
    ok, reason = check_pullback_to_ema(candles, TradeDirection.LONG)
    assert not ok
    assert "wait for pullback" in reason.lower()


def test_pullback_short_near_ema():
    """Price near EMA9 → pullback OK for SHORT."""
    candles = _make_candles([50] * 15)
    ok, reason = check_pullback_to_ema(candles, TradeDirection.SHORT)
    assert ok, reason


def test_pullback_short_far_below_ema():
    """Price far below EMA9 → pullback rejected for SHORT."""
    prices = [50] * 14 + [45]  # 10% below
    candles = _make_candles(prices)
    ok, reason = check_pullback_to_ema(candles, TradeDirection.SHORT)
    assert not ok
    assert "wait for pullback" in reason.lower()


def test_pullback_insufficient_candles():
    candles = _make_candles([50] * 5)
    ok, reason = check_pullback_to_ema(candles, TradeDirection.LONG)
    assert not ok
    assert "insufficient" in reason.lower()


# ── check_volume_surge ──

def test_volume_surge_detected():
    candles = _make_candles([50] * 15, volumes=[100] * 14 + [150])  # 1.5x surge
    ok, reason = check_volume_surge(candles)
    assert ok
    assert "surge" in reason.lower()


def test_volume_no_surge():
    candles = _make_candles([50] * 15, volumes=[100] * 15)
    ok, reason = check_volume_surge(candles)
    assert not ok


def test_volume_insufficient_candles():
    candles = _make_candles([50] * 5, volumes=[100] * 5)
    ok, reason = check_volume_surge(candles)
    assert not ok
    assert "insufficient" in reason.lower()


# ── score_setup ──

def test_score_perfect_setup():
    ctx = _make_ctx(signal_strength=0.80, rr_ratio=2.5, ltf_aligned=True)
    score = score_setup(ctx, pullback_ok=True, volume_ok=True)
    assert score > 0.70
    assert score <= 1.0


def test_score_weak_setup():
    ctx = _make_ctx(signal_strength=0.30, rr_ratio=1.1, ltf_aligned=False)
    score = score_setup(ctx, pullback_ok=False, volume_ok=False)
    assert score < 0.40


def test_score_no_pullback_penalty():
    ctx = _make_ctx(signal_strength=0.70, rr_ratio=2.0, ltf_aligned=True)
    score_with = score_setup(ctx, pullback_ok=True, volume_ok=True)
    score_without = score_setup(ctx, pullback_ok=False, volume_ok=False)
    assert score_with > score_without
    assert score_with - score_without == pytest.approx(0.25, abs=0.01)


def test_score_capped_at_one():
    ctx = _make_ctx(signal_strength=1.0, rr_ratio=5.0, ltf_aligned=True)
    score = score_setup(ctx, pullback_ok=True, volume_ok=True)
    assert score == 1.0
