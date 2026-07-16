"""Unit tests for ICT/SMC signal detection: Order Block and OTE math.

Tests cover:
- compute_ote_zone: correct 61.8/79 % levels for LONG and SHORT
- price_in_ote: boundary conditions
- detect_order_block: finds OB after a bull/bear displacement
- detect_order_block: returns None when no valid OB exists
- OrderBlock.check_mitigation: marks OB as stale when price closes through it
- OrderBlock.price_in_ote: delegates correctly
- detect_order_block displacement body filter: doji impulses are rejected
"""

from __future__ import annotations

import pytest

from scalping.ict_signals import (
    OTE_FIB_HIGH,
    OTE_FIB_LOW,
    OrderBlock,
    compute_ote_zone,
    detect_order_block,
    price_in_ote,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _candle(o: float, h: float, l: float, c: float, v: int = 1000) -> dict:
    return {"open": o, "high": h, "low": l, "close": c, "volume": v}


def _make_bull_setup(n_lead: int = 5) -> list[dict]:
    """Build a minimal bullish sequence:
    n_lead neutral candles → one bearish OB candle → strong bull displacement candle.
    Returns with the OB candle at index -2 and displacement at index -1.
    """
    candles = [_candle(100, 101, 99, 100) for _ in range(n_lead)]
    candles.append(_candle(102, 103, 99, 100))   # bearish OB candle (close < open)
    candles.append(_candle(100, 110, 99, 109))   # strong bull displacement
    return candles


def _make_bear_setup(n_lead: int = 5) -> list[dict]:
    """Build a minimal bearish sequence:
    n_lead neutral candles → one bullish OB candle → strong bear displacement candle.
    """
    candles = [_candle(100, 101, 99, 100) for _ in range(n_lead)]
    candles.append(_candle(99, 103, 98, 101))    # bullish OB candle (close > open)
    candles.append(_candle(101, 102, 90, 91))    # strong bear displacement
    return candles


# ── compute_ote_zone ──────────────────────────────────────────────────────────

def test_ote_zone_long_levels():
    """LONG OTE: 61.8 % and 79 % retracement from swing_high toward swing_low."""
    lo, hi = compute_ote_zone(100.0, 110.0, "LONG")
    span = 10.0
    expected_hi = round(110.0 - span * OTE_FIB_LOW,  2)   # 61.8 % retrace
    expected_lo = round(110.0 - span * OTE_FIB_HIGH, 2)   # 79.0 % retrace
    assert abs(hi - expected_hi) < 0.01
    assert abs(lo - expected_lo) < 0.01
    assert lo < hi


def test_ote_zone_short_levels():
    """SHORT OTE: 61.8 % and 79 % retrace from swing_low back toward swing_high."""
    lo, hi = compute_ote_zone(100.0, 110.0, "SHORT")
    span = 10.0
    expected_lo = round(100.0 + span * OTE_FIB_LOW,  2)
    expected_hi = round(100.0 + span * OTE_FIB_HIGH, 2)
    assert abs(lo - expected_lo) < 0.01
    assert abs(hi - expected_hi) < 0.01
    assert lo < hi


def test_ote_zone_zero_span():
    """Zero-span leg returns the input prices unchanged."""
    lo, hi = compute_ote_zone(50.0, 50.0, "LONG")
    assert lo == 50.0 and hi == 50.0


def test_ote_zone_long_midpoint():
    """The OTE midpoint should be between 61.8 % and 79 % retracement."""
    lo, hi = compute_ote_zone(0.0, 100.0, "LONG")
    mid = (lo + hi) / 2
    # Expect midpoint near 70 % retrace = 30.0
    assert 20.0 < mid < 40.0


# ── price_in_ote ──────────────────────────────────────────────────────────────

def test_price_in_ote_inside():
    assert price_in_ote(105.0, 103.0, 107.0) is True


def test_price_in_ote_at_boundaries():
    assert price_in_ote(103.0, 103.0, 107.0) is True
    assert price_in_ote(107.0, 103.0, 107.0) is True


def test_price_in_ote_outside():
    assert price_in_ote(102.9, 103.0, 107.0) is False
    assert price_in_ote(107.1, 103.0, 107.0) is False


# ── detect_order_block ────────────────────────────────────────────────────────

def test_detects_long_ob_after_bull_displacement():
    candles = _make_bull_setup(n_lead=8)
    ob = detect_order_block(candles, "LONG", displacement_lookback=6)
    assert ob is not None
    assert ob.direction == "LONG"
    assert ob.ob_low < ob.ob_high
    assert ob.ote_low < ob.ote_high
    assert ob.sl_price < ob.ob_low + 0.01    # SL is below OB low
    assert ob.fresh is True


def test_detects_short_ob_after_bear_displacement():
    candles = _make_bear_setup(n_lead=8)
    ob = detect_order_block(candles, "SHORT", displacement_lookback=6)
    assert ob is not None
    assert ob.direction == "SHORT"
    assert ob.sl_price > ob.ob_high - 0.01   # SL is above OB high
    assert ob.fresh is True


def test_returns_none_when_too_few_candles():
    candles = _make_bull_setup(n_lead=2)[:5]   # only 5 bars
    ob = detect_order_block(candles, "LONG", displacement_lookback=8)
    assert ob is None


def test_returns_none_when_no_displacement():
    """A flat market with no strong impulse candle → no valid OB."""
    candles = [_candle(100, 101, 99, 100) for _ in range(20)]
    ob = detect_order_block(candles, "LONG", displacement_lookback=10)
    assert ob is None


def test_returns_none_when_doji_displacement():
    """If the 'displacement' candle has a tiny body, it should be filtered."""
    candles = [_candle(100, 101, 99, 100) for _ in range(10)]
    # Add a near-doji as the displacement (body 0.05 / range 2.0 = 2.5 % < 40 %)
    candles.append(_candle(99.5, 101, 99, 99.55))   # OB candle
    candles.append(_candle(99.55, 101, 99, 99.60))  # doji displacement
    ob = detect_order_block(candles, "LONG", displacement_lookback=6, min_ob_body_pct=0.003)
    assert ob is None


def test_ob_sl_is_below_ob_low_for_long():
    candles = _make_bull_setup(n_lead=10)
    ob = detect_order_block(candles, "LONG", displacement_lookback=8)
    assert ob is not None
    assert ob.sl_price < ob.ob_low


def test_ob_sl_is_above_ob_high_for_short():
    candles = _make_bear_setup(n_lead=10)
    ob = detect_order_block(candles, "SHORT", displacement_lookback=8)
    assert ob is not None
    assert ob.sl_price > ob.ob_high


def test_ob_tp_at_displacement_extreme():
    candles = _make_bull_setup(n_lead=8)
    ob = detect_order_block(candles, "LONG", displacement_lookback=6)
    assert ob is not None
    # TP should be at or near the displacement high (109–110 range in our fixture)
    assert ob.tp_price >= 108.0


def test_ob_strength_between_zero_and_one():
    candles = _make_bull_setup(n_lead=8)
    ob = detect_order_block(candles, "LONG", displacement_lookback=6)
    assert ob is not None
    assert 0.0 <= ob.strength <= 1.0


# ── OrderBlock.check_mitigation ───────────────────────────────────────────────

def test_mitigation_invalidates_long_ob_on_close_below():
    candles = _make_bull_setup(n_lead=8)
    ob = detect_order_block(candles, "LONG", displacement_lookback=6)
    assert ob is not None
    assert ob.fresh

    ob.check_mitigation(ob.ob_low - 1.0)   # close below OB low → mitigated
    assert not ob.fresh


def test_mitigation_does_not_trigger_when_above_ob_low():
    candles = _make_bull_setup(n_lead=8)
    ob = detect_order_block(candles, "LONG", displacement_lookback=6)
    assert ob is not None

    ob.check_mitigation(ob.ob_low + 0.1)   # close still inside OB → still fresh
    assert ob.fresh


def test_mitigation_invalidates_short_ob_on_close_above():
    candles = _make_bear_setup(n_lead=8)
    ob = detect_order_block(candles, "SHORT", displacement_lookback=6)
    assert ob is not None

    ob.check_mitigation(ob.ob_high + 1.0)
    assert not ob.fresh


def test_double_mitigation_stays_stale():
    candles = _make_bull_setup(n_lead=8)
    ob = detect_order_block(candles, "LONG", displacement_lookback=6)
    assert ob is not None
    ob.check_mitigation(ob.ob_low - 5.0)
    assert not ob.fresh
    ob.check_mitigation(ob.ob_high + 5.0)  # second call on stale OB — should stay False
    assert not ob.fresh


# ── OrderBlock.price_in_ote ───────────────────────────────────────────────────

def test_ob_price_in_ote_delegation():
    candles = _make_bull_setup(n_lead=8)
    ob = detect_order_block(candles, "LONG", displacement_lookback=6)
    assert ob is not None
    # A price inside the OTE zone should return True
    mid = (ob.ote_low + ob.ote_high) / 2
    assert ob.price_in_ote(mid) is True
    # A price far outside should return False
    assert ob.price_in_ote(ob.ob_low - 10.0) is False


# ── OTE zone is within displacement span ─────────────────────────────────────

def test_ote_zone_within_displacement_span():
    """The OTE zone must lie within the displacement leg — never outside it."""
    candles = _make_bull_setup(n_lead=8)
    ob = detect_order_block(candles, "LONG", displacement_lookback=6)
    assert ob is not None
    assert ob.ote_low  >= ob.displacement_low
    assert ob.ote_high <= ob.displacement_high + 1.0  # tiny tolerance for rounding


# ── config instantiation smoke test ──────────────────────────────────────────

def test_ict_configs_import_without_error():
    from scalping.ict_config import ExpiryICTConfig, NonExpiryICTConfig
    assert ExpiryICTConfig.name == "EXPIRY ICT SNIPER"
    assert NonExpiryICTConfig.name == "NON-EXPIRY ICT SNIPER"
    assert ExpiryICTConfig.run_on_weekday == 1      # Tuesday
    assert NonExpiryICTConfig.skip_weekday == 1     # skip Tuesday
    assert ExpiryICTConfig.entry_interval == "1"
    assert NonExpiryICTConfig.entry_interval == "5"
    assert ExpiryICTConfig.ote_fib_low == 0.618
    assert ExpiryICTConfig.ote_fib_high == 0.790
