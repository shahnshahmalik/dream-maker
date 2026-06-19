"""Tests for VWAP Pullback strategy.

Coverage:
- _compute_session_vwap: correct weighting, sigma band, edge cases
- check_vwap_pullback: LONG/SHORT happy paths, rejection cases
- _analyze_vwap_pullback: full signal (LONG + SHORT), all rejection paths
- analyze_technical: active_strategy='vwap_pullback' routes correctly
"""

from __future__ import annotations

import pytest
from datetime import datetime

from analysis.technical import (
    SetupType,
    Trend,
    _analyze_vwap_pullback,
    _compute_session_vwap,
    analyze_technical,
    check_vwap_pullback,
)
from models.orders import OHLCV
from models.trade_plan import TradeDirection


# ── Helpers ───────────────────────────────────────────────────────────────────

def _c(o: float, h: float, l: float, c: float, v: float = 10_000.0) -> OHLCV:
    return OHLCV(open=o, high=h, low=l, close=c, volume=v, timestamp=datetime(2026, 6, 17))


def _uptrend(n: int = 30, start: float = 24000.0, step: float = 5.0) -> list[OHLCV]:
    """Monotonically rising candles — HTF trend = UPTREND."""
    return [_c(start + i * step, start + i * step + 8, start + i * step - 3, start + i * step + 5) for i in range(n)]


def _downtrend(n: int = 30, start: float = 24300.0, step: float = 5.0) -> list[OHLCV]:
    """Monotonically falling candles — HTF trend = DOWNTREND."""
    return [_c(start - i * step, start - i * step + 3, start - i * step - 8, start - i * step - 5) for i in range(n)]


def _vwap_pullback_long_candles(vwap_approx: float = 24100.0) -> list[OHLCV]:
    """LTF candles that create a VWAP near vwap_approx and end with a pullback long candle."""
    # 20 stable candles clustered around vwap_approx to anchor VWAP
    candles = [_c(vwap_approx, vwap_approx + 10, vwap_approx - 10, vwap_approx + 5) for _ in range(20)]
    # Final candle: price above VWAP, low touches VWAP, bullish close (rejection wick)
    pull = _c(vwap_approx + 5, vwap_approx + 15, vwap_approx - 0.2, vwap_approx + 12, v=15_000.0)
    return candles + [pull]


def _vwap_pullback_short_candles(vwap_approx: float = 24100.0) -> list[OHLCV]:
    """LTF candles that create a VWAP near vwap_approx and end with a pullback short candle."""
    candles = [_c(vwap_approx, vwap_approx + 10, vwap_approx - 10, vwap_approx - 5) for _ in range(20)]
    # Final candle: price below VWAP, high touches VWAP, bearish close
    pull = _c(vwap_approx - 5, vwap_approx + 0.2, vwap_approx - 15, vwap_approx - 12, v=15_000.0)
    return candles + [pull]


# ── _compute_session_vwap ─────────────────────────────────────────────────────

def test_vwap_weighted_average():
    """VWAP = sum(typical_price × volume) / sum(volume)."""
    # Use 5+ candles (minimum) — first 3 are neutral, last 2 have distinct weights
    base = [_c(100, 110, 90, 100, v=1000) for _ in range(3)]
    weighted = [
        _c(100, 110, 90, 105, v=1000),  # TP = (110+90+105)/3 ≈ 101.67
        _c(105, 115, 95, 110, v=2000),  # TP = (115+95+110)/3 ≈ 106.67
    ]
    candles = base + weighted
    vwap, upper, lower = _compute_session_vwap(candles)
    assert vwap > 0
    # Heavier volume on the second weighted candle should pull VWAP toward its TP
    tp1 = (110 + 90 + 105) / 3.0
    tp2 = (115 + 95 + 110) / 3.0
    assert tp1 < vwap < tp2 + 1  # VWAP is biased toward the 2000-volume candle


def test_vwap_bands_upper_above_lower():
    candles = _uptrend(20)
    vwap, upper, lower = _compute_session_vwap(candles)
    assert upper > vwap > lower


def test_vwap_too_few_candles_returns_zeros():
    vwap, upper, lower = _compute_session_vwap([_c(100, 110, 90, 105)])
    assert vwap == 0.0
    assert upper == 0.0
    assert lower == 0.0


# ── check_vwap_pullback ───────────────────────────────────────────────────────

def test_vwap_pullback_long_confirmed():
    ltf = _vwap_pullback_long_candles(24100.0)
    ok, reason = check_vwap_pullback(ltf, TradeDirection.LONG)
    assert ok, reason


def test_vwap_pullback_short_confirmed():
    ltf = _vwap_pullback_short_candles(24100.0)
    ok, reason = check_vwap_pullback(ltf, TradeDirection.SHORT)
    assert ok, reason


def test_vwap_pullback_long_fails_when_close_below_vwap():
    """Price below VWAP — not an uptrend, reject LONG."""
    candles = [_c(24100, 24110, 24090, 24095) for _ in range(20)]
    # Last candle: close below VWAP
    candles.append(_c(24090, 24102, 24080, 24085))
    ok, reason = check_vwap_pullback(candles, TradeDirection.LONG)
    assert not ok
    assert "below VWAP" in reason


def test_vwap_pullback_short_fails_when_close_above_vwap():
    """Price above VWAP — not a downtrend, reject SHORT."""
    candles = [_c(24100, 24110, 24090, 24105) for _ in range(20)]
    candles.append(_c(24110, 24120, 24098, 24115))
    ok, reason = check_vwap_pullback(candles, TradeDirection.SHORT)
    assert not ok
    assert "above VWAP" in reason


def test_vwap_pullback_fails_wick_too_far():
    """Low is far from VWAP — not a pullback, just trending away."""
    candles = [_c(24100, 24110, 24090, 24105) for _ in range(20)]
    # Last candle: close above VWAP but low is 2% away — too far
    candles.append(_c(24200, 24250, 24200, 24240))  # low 24200, VWAP ~24100 → 0.46% proximity fails 0.3%
    ok, reason = check_vwap_pullback(candles, TradeDirection.LONG)
    assert not ok


def test_vwap_pullback_fails_too_few_candles():
    ok, reason = check_vwap_pullback([_c(100, 110, 90, 105)] * 5, TradeDirection.LONG)
    assert not ok
    assert "too_few_candles" in reason


# ── _analyze_vwap_pullback ────────────────────────────────────────────────────

def test_vwap_pullback_long_signal():
    """Full LONG setup: uptrend HTF + pullback LTF → VWAP_PULLBACK signal."""
    htf = _uptrend(30)
    ltf = _vwap_pullback_long_candles(24100.0)
    ctx = _analyze_vwap_pullback(htf, ltf, min_rr=1.5, max_sl_pct=2.0)
    assert ctx is not None
    assert ctx.setup_type == SetupType.VWAP_PULLBACK
    assert ctx.direction == TradeDirection.LONG
    assert ctx.stop_loss < ctx.entry
    assert ctx.tp1 > ctx.entry
    assert ctx.rr_ratio >= 1.5
    assert "vwap_pullback" in ctx.confirmations


def test_vwap_pullback_short_signal():
    """Full SHORT setup: downtrend HTF + pullback LTF → VWAP_PULLBACK signal."""
    htf = _downtrend(30)
    ltf = _vwap_pullback_short_candles(24100.0)
    ctx = _analyze_vwap_pullback(htf, ltf, min_rr=1.5, max_sl_pct=2.0)
    assert ctx is not None
    assert ctx.setup_type == SetupType.VWAP_PULLBACK
    assert ctx.direction == TradeDirection.SHORT
    assert ctx.stop_loss > ctx.entry
    assert ctx.tp1 < ctx.entry


def test_vwap_pullback_rejects_range_htf():
    """Flat HTF — no trend direction, should return None."""
    flat = [_c(24100, 24110, 24090, 24100) for _ in range(30)]
    ltf  = _vwap_pullback_long_candles(24100.0)
    ctx  = _analyze_vwap_pullback(flat, ltf, min_rr=1.5, max_sl_pct=2.0)
    assert ctx is None


def test_vwap_pullback_rejects_bearish_candle_on_long():
    """Last candle is bearish — rejection not confirmed for LONG."""
    htf = _uptrend(30)
    ltf = _vwap_pullback_long_candles(24100.0)
    # Replace last candle with a bearish one (close < open)
    ltf[-1] = _c(24112, 24115, 24099, 24100)  # bearish, low near VWAP, but close < open
    ctx = _analyze_vwap_pullback(htf, ltf, min_rr=1.5, max_sl_pct=2.0)
    assert ctx is None


def test_vwap_pullback_rejects_too_few_candles():
    ctx = _analyze_vwap_pullback(_uptrend(5), _vwap_pullback_long_candles(), min_rr=1.5, max_sl_pct=2.0)
    assert ctx is None


def test_vwap_pullback_respects_allowed_direction():
    """allowed_direction overrides HTF trend direction."""
    htf = _uptrend(30)  # uptrend → would normally pick LONG
    ltf = _vwap_pullback_short_candles(24100.0)
    ctx = _analyze_vwap_pullback(htf, ltf, min_rr=1.5, max_sl_pct=2.0, allowed_direction=TradeDirection.SHORT)
    assert ctx is not None
    assert ctx.direction == TradeDirection.SHORT


def test_vwap_signal_strength_range():
    """Signal strength must be between 0 and 1."""
    htf = _uptrend(30)
    ltf = _vwap_pullback_long_candles(24100.0)
    ctx = _analyze_vwap_pullback(htf, ltf, min_rr=1.5, max_sl_pct=2.0)
    assert ctx is not None
    assert 0.0 <= ctx.signal_strength <= 1.0


# ── analyze_technical router ──────────────────────────────────────────────────

def test_analyze_technical_routes_vwap_pullback():
    """active_strategy='vwap_pullback' returns a VWAP_PULLBACK context."""
    htf = _uptrend(30)
    ltf = _vwap_pullback_long_candles(24100.0)
    ctx = analyze_technical(
        htf, ltf,
        active_strategy="vwap_pullback",
        scalp_min_rr=1.5,
        scalp_max_sl_pct=2.0,
    )
    assert ctx is not None
    assert ctx.setup_type == SetupType.VWAP_PULLBACK


def test_analyze_technical_vwap_returns_none_on_no_signal():
    """active_strategy='vwap_pullback' with flat candles → None."""
    flat = [_c(24100, 24110, 24090, 24100) for _ in range(30)]
    ctx  = analyze_technical(
        flat, flat,
        active_strategy="vwap_pullback",
        scalp_min_rr=1.5,
        scalp_max_sl_pct=2.0,
    )
    assert ctx is None
