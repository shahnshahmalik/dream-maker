"""Tests for Quad Stochastic Divergence (Holy Grail / HPS) strategy.

Coverage:
- _stochastic / swing helpers / reversal candles
- detect_quad_stoch_divergence: LONG happy path + rejections
- _analyze_quad_stoch_div: full signal, direction lock, candle count
- analyze_technical: active_strategy='quad_stoch_div' routes correctly
"""

from __future__ import annotations

from datetime import datetime, timedelta

from analysis.technical import (
    SetupType,
    _analyze_quad_stoch_div,
    _is_bearish_reversal_candle,
    _is_bullish_reversal_candle,
    _ohlcv_to_df,
    _stochastic,
    _swing_high_indices,
    _swing_low_indices,
    analyze_technical,
    detect_quad_stoch_divergence,
)
from models.orders import OHLCV
from models.trade_plan import TradeDirection


def _c(o: float, h: float, l: float, cl: float, v: int = 10_000, i: int = 0) -> OHLCV:
    return OHLCV(
        timestamp=datetime(2026, 6, 17) + timedelta(minutes=i),
        open=o,
        high=h,
        low=l,
        close=cl,
        volume=v,
    )


def _uptrend_htf(n: int = 30, start: float = 24000.0, step: float = 5.0) -> list[OHLCV]:
    out: list[OHLCV] = []
    px = start
    for i in range(n):
        o = px
        cl = px + step
        out.append(_c(o, cl + 2, o - 1, cl, i=i))
        px = cl
    return out


def _bullish_div_ltf() -> list[OHLCV]:
    """LTF series with a completed bullish quad-stoch divergence at the end."""
    candles: list[OHLCV] = []
    px = 24500.0
    for i in range(50):
        o = px
        cl = px + 2
        candles.append(_c(o, cl + 1, o - 1, cl, i=i))
        px = cl
    for i in range(30):
        o = px
        cl = px - 15
        candles.append(_c(o, o + 1, cl - 2, cl, i=50 + i))
        px = cl
    s1_low = px - 10
    candles.append(_c(px, px + 4, s1_low, px + 3, i=80))
    px = candles[-1].close
    for i in range(7):
        o = px
        cl = px + 10
        candles.append(_c(o, cl + 2, o - 1, cl, i=81 + i))
        px = cl
    target = s1_low - 3
    for i in range(6):
        o = px
        cl = px - (px - target) / 6
        candles.append(_c(o, o + 2, min(cl, o) - 1, cl, i=88 + i))
        px = cl
    candles.append(_c(px, px + 8, target - 1, px + 7, i=94))
    px = candles[-1].close
    for i in range(3):
        o = px
        cl = px + 5
        candles.append(_c(o, cl + 2, o - 1, cl, i=95 + i))
        px = cl
    return candles


# ── Helpers ───────────────────────────────────────────────────────────────────

def test_stochastic_bounds():
    candles = _bullish_div_ltf()
    df = _ohlcv_to_df(candles)
    k, d = _stochastic(df, 9, 3)
    valid = k.dropna()
    assert valid.min() >= 0.0
    assert valid.max() <= 100.0
    assert not d.dropna().empty


def test_swing_low_indices_finds_local_minima():
    lows = [10, 9, 8, 7, 8, 9, 6, 7, 8]  # minima at idx 3 and 6
    import pandas as pd

    idxs = _swing_low_indices(pd.Series(lows, dtype=float), order=2)
    assert 3 in idxs
    assert 6 in idxs


def test_swing_high_indices_finds_local_maxima():
    highs = [5, 6, 7, 8, 7, 6, 9, 8, 7]
    import pandas as pd

    idxs = _swing_high_indices(pd.Series(highs, dtype=float), order=2)
    assert 3 in idxs
    assert 6 in idxs


def test_bullish_reversal_candle():
    hammer = _c(100, 102, 90, 101)  # long lower wick, close near high
    assert _is_bullish_reversal_candle(hammer)
    bear = _c(100, 101, 99, 99.5)
    assert not _is_bullish_reversal_candle(bear)


def test_bearish_reversal_candle():
    star = _c(100, 110, 99, 100.5)  # long upper wick
    assert _is_bearish_reversal_candle(star)
    bull = _c(100, 105, 99, 104)
    assert not _is_bearish_reversal_candle(bull)


# ── detect_quad_stoch_divergence ──────────────────────────────────────────────

def test_detect_bullish_divergence():
    sig = detect_quad_stoch_divergence(_bullish_div_ltf())
    assert sig is not None
    assert sig.direction == TradeDirection.LONG
    assert sig.stage2_price <= sig.stage1_price
    assert sig.stage2_stoch > sig.stage1_stoch
    assert sig.stage2_stoch > 20.0
    assert "stoch_div_bull" in sig.confirmations


def test_detect_rejects_too_few_candles():
    assert detect_quad_stoch_divergence([_c(100, 101, 99, 100)] * 20) is None


def test_detect_rejects_flat_noise():
    flat = [_c(24100, 24105, 24095, 24100, i=i) for i in range(90)]
    assert detect_quad_stoch_divergence(flat) is None


# ── _analyze_quad_stoch_div ───────────────────────────────────────────────────

def test_analyze_long_signal():
    htf = _uptrend_htf()
    ltf = _bullish_div_ltf()
    ctx = _analyze_quad_stoch_div(htf, ltf, min_rr=1.5, max_sl_pct=2.0)
    assert ctx is not None
    assert ctx.setup_type == SetupType.QUAD_STOCH_DIV
    assert ctx.direction == TradeDirection.LONG
    assert ctx.stop_loss < ctx.entry
    assert ctx.tp1 > ctx.entry
    assert ctx.rr_ratio >= 1.5
    assert "stoch_div_bull" in ctx.confirmations
    assert 0.0 < ctx.signal_strength <= 1.0


def test_analyze_respects_allowed_direction():
    htf = _uptrend_htf()
    ltf = _bullish_div_ltf()
    ctx = _analyze_quad_stoch_div(
        htf, ltf, min_rr=1.5, max_sl_pct=2.0, allowed_direction=TradeDirection.SHORT,
    )
    assert ctx is None


def test_analyze_rejects_too_few_candles():
    ctx = _analyze_quad_stoch_div(
        _uptrend_htf(10), _bullish_div_ltf()[:40], min_rr=1.5, max_sl_pct=2.0,
    )
    assert ctx is None


def test_analyze_sl_beyond_stage2_low():
    htf = _uptrend_htf()
    ltf = _bullish_div_ltf()
    sig = detect_quad_stoch_divergence(ltf)
    assert sig is not None
    ctx = _analyze_quad_stoch_div(htf, ltf, min_rr=1.5, max_sl_pct=2.0)
    assert ctx is not None
    assert ctx.stop_loss < sig.stage2_price


# ── Router ────────────────────────────────────────────────────────────────────

def test_analyze_technical_routes_quad_stoch_div():
    htf = _uptrend_htf()
    ltf = _bullish_div_ltf()
    ctx = analyze_technical(htf, ltf, active_strategy="quad_stoch_div")
    assert ctx is not None
    assert ctx.setup_type == SetupType.QUAD_STOCH_DIV


def test_analyze_technical_quad_returns_none_on_no_signal():
    flat_htf = [_c(24100, 24105, 24095, 24100, i=i) for i in range(30)]
    flat_ltf = [_c(24100, 24105, 24095, 24100, i=i) for i in range(90)]
    ctx = analyze_technical(flat_htf, flat_ltf, active_strategy="quad_stoch_div")
    assert ctx is None
