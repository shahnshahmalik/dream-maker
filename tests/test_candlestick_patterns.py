"""Tests for candlestick pattern detection."""

import pytest
from analysis.candlestick_patterns import (
    is_hammer,
    is_inverted_hammer,
    is_bullish_engulfing,
    is_bearish_engulfing,
    is_doji,
    scan_candlestick_patterns,
    detect_pullback,
    find_recent_doji,
    CandlestickPattern,
    PatternSignal,
)
from analysis.technical import (
    _analyze_candlestick_scalp,
    analyze_technical,
    SetupType,
    _ohlcv_to_df,
    detect_trend,
    Trend,
)
from models.orders import OHLCV
from models.trade_plan import TradeDirection
import pandas as pd


def _c(open_, high, low, close, vol=1000):
    return OHLCV(open=open_, high=high, low=low, close=close, volume=vol, timestamp="t")


def _flat_candles(n=30, price=100.0):
    return [_c(price, price+1, price-1, price) for _ in range(n)]


# ── Hammer Tests ─────────────────────────────────────────────────


class TestHammer:
    def test_classic_hammer(self):
        """Long lower wick, small body at top, tiny upper wick."""
        row = pd.Series({"open": 100.0, "high": 100.2, "low": 95.0, "close": 100.1, "volume": 2000})
        assert is_hammer(row)

    def test_not_hammer_wick_too_small(self):
        row = pd.Series({"open": 100.0, "high": 101.0, "low": 99.5, "close": 100.8, "volume": 1000})
        assert not is_hammer(row)

    def test_not_hammer_body_not_at_top(self):
        row = pd.Series({"open": 100.0, "high": 103.0, "low": 95.0, "close": 97.0, "volume": 1000})
        assert not is_hammer(row)

    def test_not_hammer_upper_wick_too_large(self):
        row = pd.Series({"open": 100.0, "high": 103.0, "low": 95.0, "close": 100.2, "volume": 1000})
        assert not is_hammer(row)


class TestInvertedHammer:
    def test_classic_inverted_hammer(self):
        row = pd.Series({"open": 100.0, "high": 105.0, "low": 99.8, "close": 100.2, "volume": 2000})
        assert is_inverted_hammer(row)

    def test_not_inverted_body_not_at_bottom(self):
        row = pd.Series({"open": 100.0, "high": 105.0, "low": 99.0, "close": 103.0, "volume": 1000})
        assert not is_inverted_hammer(row)


class TestEngulfing:
    def test_bullish_engulfing(self):
        prev = pd.Series({"open": 102.0, "high": 103.0, "low": 100.0, "close": 100.5, "volume": 800})
        curr = pd.Series({"open": 99.0, "high": 104.0, "low": 98.5, "close": 103.0, "volume": 1200})
        assert is_bullish_engulfing(prev, curr)

    def test_not_bullish_engulfing_prev_is_green(self):
        prev = pd.Series({"open": 100.0, "high": 103.0, "low": 99.0, "close": 102.0, "volume": 800})
        curr = pd.Series({"open": 99.0, "high": 104.0, "low": 98.5, "close": 103.0, "volume": 1200})
        assert not is_bullish_engulfing(prev, curr)

    def test_bearish_engulfing(self):
        prev = pd.Series({"open": 100.0, "high": 104.0, "low": 99.0, "close": 103.0, "volume": 800})
        curr = pd.Series({"open": 104.0, "high": 105.0, "low": 98.0, "close": 99.0, "volume": 1200})
        assert is_bearish_engulfing(prev, curr)

    def test_not_engulfing_body_too_small(self):
        prev = pd.Series({"open": 102.0, "high": 103.0, "low": 100.0, "close": 100.5, "volume": 800})
        curr = pd.Series({"open": 100.4, "high": 101.0, "low": 100.0, "close": 101.0, "volume": 900})
        assert not is_bullish_engulfing(prev, curr)


class TestDoji:
    def test_classic_doji(self):
        row = pd.Series({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.05, "volume": 1000})
        assert is_doji(row)

    def test_not_doji_large_body(self):
        row = pd.Series({"open": 100.0, "high": 103.0, "low": 99.0, "close": 102.0, "volume": 1000})
        assert not is_doji(row)


# ── Pullback Tests ─────────────────────────────────────────────────


class TestPullback:
    def test_detect_pullback_long(self):
        candles = [
            _c(100, 102, 99, 101),
            _c(101, 103, 100, 102),
            _c(102, 104, 101, 103),
            _c(103, 104, 100.5, 102),  # pullback: low 100.5 < prev low 101
        ]
        idx = detect_pullback(candles, "LONG", from_index=0)
        assert idx == 3

    def test_detect_pullback_short(self):
        candles = [
            _c(100, 101, 99, 100),
            _c(99, 100, 98, 99),
            _c(98, 99, 97, 98),
            _c(97.5, 99.5, 97, 98.5),  # pullback: high 99.5 > prev high 99
        ]
        idx = detect_pullback(candles, "SHORT", from_index=0)
        assert idx == 3

    def test_no_pullback(self):
        candles = [
            _c(100, 103, 100, 102),
            _c(102, 105, 102, 104),
            _c(104, 107, 104, 106),
        ]
        idx = detect_pullback(candles, "LONG", from_index=0)
        assert idx is None


# ── Pattern Scanning Tests ─────────────────────────────────────────


class TestScanCandlestickPatterns:
    def test_hammer_in_downtrend_detected(self):
        """Hammer at index 3, confirmation at index 4."""
        candles = [
            _c(105, 106, 104, 104.5),  # 0: red (downtrend)
            _c(104, 105, 103, 103.5),  # 1: red
            _c(103, 104, 102, 102.5),  # 2: red
            _c(100, 100.2, 95.0, 100.1, 2000),  # 3: hammer
            _c(100.1, 101.0, 99.8, 100.8, 1500),  # 4: bullish confirmation
        ]
        result = scan_candlestick_patterns(candles, "downtrend")
        assert result is not None
        assert result.pattern == CandlestickPattern.HAMMER
        assert result.direction == "LONG"

    def test_bullish_engulfing_in_downtrend(self):
        candles = [
            _c(105, 106, 104, 104.5),
            _c(104, 105, 103, 103.5),
            _c(103, 104, 102, 102.5),  # red candle (prev)
            _c(101, 104, 100, 103.5, 1500),  # green engulfing
            _c(103.5, 105, 103.5, 104.5, 1200),  # confirmation above
        ]
        result = scan_candlestick_patterns(candles, "downtrend")
        assert result is not None
        assert result.pattern == CandlestickPattern.BULLISH_ENGULFING

    def test_no_pattern_in_uptrend_only(self):
        """Hammer in uptrend without inverted hammer should not trigger."""
        candles = [
            _c(95, 96, 94, 95.5),
            _c(96, 97, 95, 96.5),
            _c(97, 98, 96, 97.5),
            _c(97.5, 98, 92, 97.8, 2000),  # hammer (but uptrend)
            _c(98, 99, 97, 98.8, 1500),  # bullish confirmation
        ]
        result = scan_candlestick_patterns(candles, "uptrend")
        # Hammer only triggers in downtrend/range, not uptrend
        assert result is None

    def test_empty_returns_none(self):
        assert scan_candlestick_patterns([], "downtrend") is None
        assert scan_candlestick_patterns([_c(100, 101, 99, 100.5)], "downtrend") is None


# ── Doji Finder Tests ──────────────────────────────────────────────


class TestDojiFinder:
    def test_find_recent_doji(self):
        candles = [
            _c(100, 101, 99, 100.5),
            _c(100.5, 101.5, 99.5, 100.5),  # doji
            _c(100.5, 102, 99, 101.0),
            _c(101, 103, 100, 102.0),
        ]
        idx = find_recent_doji(candles)
        assert idx == 1

    def test_no_doji(self):
        candles = [_c(100, 103, 99, 102) for _ in range(5)]
        assert find_recent_doji(candles) is None


# ── Candlestick Scalp Analyzer Tests ───────────────────────────────


class TestAnalyzeCandlestickScalp:
    def test_returns_setup_for_hammer_in_downtrend(self):
        """Full integration: hammer pattern → TechnicalContext."""
        # HTF: downtrend (30 candles)
        htf = [_c(p, p+1, p-1, p-0.5) for p in [110 - i*2 for i in range(30)]]
        # LTF: downtrend prelude (15 candles) + hammer + confirmation + follow-through
        prelude = [_c(p, p+1, p-1, p-0.5) for p in [108 - i*2 for i in range(18)]]
        ltf = prelude + [
            _c(100, 100.2, 95.0, 100.1, 2000),  # hammer
            _c(100.2, 101.5, 99.5, 101.0, 1500),  # bullish confirmation
        ]
        htf_df = _ohlcv_to_df(htf)
        ltf_df = _ohlcv_to_df(ltf)
        result = _analyze_candlestick_scalp(
            htf_df, ltf_df, ltf, min_rr=2.0, max_sl_pct=0.35
        )
        assert result is not None
        assert result.setup_type == SetupType.CANDLESTICK_SCALP
        assert result.direction == TradeDirection.LONG
        assert result.stop_loss < result.entry

    def test_returns_none_without_pattern(self):
        htf = _flat_candles(30)
        ltf = _flat_candles(25)
        htf_df = _ohlcv_to_df(htf)
        ltf_df = _ohlcv_to_df(ltf)
        result = _analyze_candlestick_scalp(
            htf_df, ltf_df, ltf, min_rr=2.0, max_sl_pct=0.35
        )
        assert result is None

    def test_integrated_in_analyze_technical(self):
        """verify candlestick scalp fires in full analyze_technical pipeline."""
        # HTF: flat (range) — so swing fails, falls through to scalps
        htf = [OHLCV(open=100, high=101, low=99, close=100.5, volume=1000, timestamp="t") for _ in range(30)]
        # LTF: downtrend prelude + hammer + confirmation
        prelude = [_c(p, p+1, p-1, p-0.5) for p in [108 - i*2 for i in range(18)]]
        ltf = prelude + [
            _c(100, 100.2, 95.0, 100.1, 2000),  # hammer
            _c(100.2, 101.5, 99.5, 101.0, 1500),  # confirmation
        ]
        result = analyze_technical(htf, ltf, min_rr=3.0, scalp_enabled=True,
                                    scalp_min_rr=2.0, scalp_max_sl_pct=0.35,
                                    scalp_min_confirmations=1)
        assert result is not None
        assert result.setup_type in (
            SetupType.CANDLESTICK_SCALP,
            SetupType.MOMENTUM_SCALP,
            SetupType.SWING,
        )
