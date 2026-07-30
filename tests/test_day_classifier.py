"""Tests for analysis/day_classifier.py.

Covers:
- Quiet Opening Range does NOT block before midday (bug fix)
- Range/Inside Day → blocked only after 11:00 IST when developing day range < 0.8%
- Gap Down & Rally → LONG only
- Gap Down & Trend → SHORT only
- Gap Up & Trend → LONG only
- V-Reversal Bull → LONG only (no gap, bullish OR structure)
- V-Reversal Bear → SHORT only (no gap, bearish OR structure)
- Unknown (< 3 candles) → allows any
- direction gate: allows() / is_blocked
- DayGate cache: second call returns same result without re-fetch
- DayGate does not permanently cache UNKNOWN
"""
from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import pytest

from analysis.day_classifier import (
    DayGate,
    DayType,
    classify,
)
from models.orders import OHLCV
from models.trade_plan import TradeDirection

IST = ZoneInfo("Asia/Kolkata")


# ── Helpers ──────────────────────────────────────────────────────────

def _c(o: float, h: float, l: float, c: float, v: int = 100_000,
       hour: int = 9, minute: int = 15) -> OHLCV:
    # Use today's date so DayGate's today-filter doesn't discard test candles
    from datetime import date
    today = date.today()
    return OHLCV(
        timestamp=datetime(today.year, today.month, today.day, hour, minute, tzinfo=IST),
        open=o, high=h, low=l, close=c, volume=v,
    )


def _morning(hour: int = 9, minute: int = 30) -> datetime:
    from datetime import date
    today = date.today()
    return datetime(today.year, today.month, today.day, hour, minute, tzinfo=IST)


def _afternoon(hour: int = 11, minute: int = 30) -> datetime:
    return _morning(hour, minute)


def _or_candles_bullish(base: float = 24000.0) -> list[OHLCV]:
    """3 candles making HH/HL — bullish OR structure, range ~1.1% (>0.8% threshold)."""
    return [
        _c(base, base + 80, base - 20, base + 70),
        _c(base + 70, base + 150, base + 40, base + 140),
        _c(base + 140, base + 250, base + 110, base + 240),
    ]


def _or_candles_bearish(base: float = 24000.0) -> list[OHLCV]:
    """3 candles making LH/LL — bearish OR structure, range ~1.1% (>0.8% threshold)."""
    return [
        _c(base, base + 20, base - 80, base - 70),
        _c(base - 70, base - 40, base - 150, base - 140),
        _c(base - 140, base - 110, base - 250, base - 240),
    ]


def _flat_candles(base: float = 24000.0) -> list[OHLCV]:
    """3 nearly-flat candles — OR range < 0.3% → Range/Inside Day."""
    return [
        _c(base, base + 20, base - 10, base + 5),
        _c(base + 5, base + 25, base - 5, base + 10),
        _c(base + 10, base + 30, base, base + 15),
    ]


# ── Tests: classify() ────────────────────────────────────────────────

class TestClassifyRangeDay:
    def test_quiet_or_does_not_block_before_midday(self):
        """Regression: OR range < 0.8% used to hard-block most normal opens."""
        base = 24000.0
        candles = _flat_candles(base)
        result = classify(candles, prev_close=base, now_ist=_morning(9, 30))

        assert result.day_type != DayType.RANGE_INSIDE
        assert result.is_blocked is False
        # Flat OR with HH/HL → V-Reversal Bull, LONG only
        assert result.day_type == DayType.V_REVERSAL_BULL
        assert result.allows(TradeDirection.LONG) is True

    def test_quiet_developing_day_blocked_after_midday(self):
        base = 24000.0
        candles = _flat_candles(base)
        result = classify(candles, prev_close=base, now_ist=_afternoon(11, 30))

        assert result.day_type == DayType.RANGE_INSIDE
        assert result.is_blocked is True
        assert result.allowed_directions is None
        assert result.allows(TradeDirection.LONG) is False
        assert result.allows(TradeDirection.SHORT) is False

    def test_wide_day_not_blocked_after_midday(self):
        base = 24000.0
        # Developing day range ~1.1% (> 0.8%) even with quiet OR structure shape
        candles = _or_candles_bullish(base)
        result = classify(candles, prev_close=base, now_ist=_afternoon(11, 30))

        assert result.day_type != DayType.RANGE_INSIDE
        assert result.is_blocked is False


class TestClassifyGapDays:
    def test_gap_down_bullish_or_is_gap_down_rally(self):
        base = 24000.0
        open_price = base * (1 - 0.006)  # -0.6% gap
        # Bullish OR structure, range ~1.1%
        candles = [
            _c(open_price, open_price + 80, open_price - 20, open_price + 70),
            _c(open_price + 70, open_price + 150, open_price + 40, open_price + 140),
            _c(open_price + 140, open_price + 250, open_price + 110, open_price + 240),
        ]
        result = classify(candles, prev_close=base)

        assert result.day_type == DayType.GAP_DOWN_RALLY
        assert result.is_blocked is False
        assert TradeDirection.LONG in (result.allowed_directions or [])
        assert result.allows(TradeDirection.LONG) is True
        assert result.allows(TradeDirection.SHORT) is False

    def test_gap_down_bearish_or_is_gap_down_trend(self):
        base = 24000.0
        open_price = base * (1 - 0.006)
        # Bearish OR structure, range ~1.1%
        candles = [
            _c(open_price, open_price + 20, open_price - 80, open_price - 70),
            _c(open_price - 70, open_price - 40, open_price - 150, open_price - 140),
            _c(open_price - 140, open_price - 110, open_price - 250, open_price - 240),
        ]
        result = classify(candles, prev_close=base)

        assert result.day_type == DayType.GAP_DOWN_TREND
        assert result.allows(TradeDirection.SHORT) is True
        assert result.allows(TradeDirection.LONG) is False

    def test_gap_up_trend(self):
        base = 24000.0
        open_price = base * (1 + 0.005)  # +0.5% gap
        # Bullish OR structure, range ~1.1%
        candles = [
            _c(open_price, open_price + 80, open_price - 20, open_price + 70),
            _c(open_price + 70, open_price + 150, open_price + 40, open_price + 140),
            _c(open_price + 140, open_price + 250, open_price + 110, open_price + 240),
        ]
        result = classify(candles, prev_close=base)

        assert result.day_type == DayType.GAP_UP_TREND
        assert result.allows(TradeDirection.LONG) is True
        assert result.allows(TradeDirection.SHORT) is False

    def test_gap_below_threshold_not_classified_as_gap(self):
        base = 24000.0
        open_price = base * (1 + 0.001)  # +0.1% — below 0.3% threshold
        candles = _or_candles_bullish(open_price)
        result = classify(candles, prev_close=base)

        assert result.is_gap_day is False


class TestClassifyVReversalDays:
    def test_v_reversal_bull_no_gap_bullish_or(self):
        base = 24000.0
        candles = _or_candles_bullish(base)  # open == prev_close
        result = classify(candles, prev_close=base)

        assert result.day_type == DayType.V_REVERSAL_BULL
        assert result.allows(TradeDirection.LONG) is True
        assert result.allows(TradeDirection.SHORT) is False
        assert result.is_gap_day is False

    def test_v_reversal_bear_no_gap_bearish_or(self):
        base = 24000.0
        candles = _or_candles_bearish(base)
        result = classify(candles, prev_close=base)

        assert result.day_type == DayType.V_REVERSAL_BEAR
        assert result.allows(TradeDirection.SHORT) is True
        assert result.allows(TradeDirection.LONG) is False


class TestClassifyUnknown:
    def test_fewer_than_3_candles_returns_unknown(self):
        result = classify([_c(24000, 24050, 23980, 24020)], prev_close=24000.0)
        assert result.day_type == DayType.UNKNOWN
        assert result.allows_any_direction is True
        assert result.allows(TradeDirection.LONG) is True
        assert result.allows(TradeDirection.SHORT) is True

    def test_zero_prev_close_returns_unknown(self):
        result = classify(_or_candles_bullish(), prev_close=0.0)
        assert result.day_type == DayType.UNKNOWN

    def test_empty_candles_returns_unknown(self):
        result = classify([], prev_close=24000.0)
        assert result.day_type == DayType.UNKNOWN


# ── Tests: DayGate ───────────────────────────────────────────────────

class TestDayGate:
    def _make_broker(self, daily_closes=(24100.0, 24000.0), today_5m_base=24000.0):
        broker = MagicMock()
        daily = [
            OHLCV(datetime(2026, 6, 16), 24000, 24200, 23950, c, 1_000_000)
            for c in daily_closes
        ]
        broker.get_ohlcv.side_effect = lambda sym, tf, limit: (
            daily if tf == "1d" else _or_candles_bullish(today_5m_base)
        )
        return broker

    def test_gate_disabled_allows_any(self):
        broker = MagicMock()
        gate = DayGate(broker, enabled=False)
        ok, _ = gate.allows(TradeDirection.LONG)
        assert ok is True
        ok, _ = gate.allows(TradeDirection.SHORT)
        assert ok is True

    def _freeze_now(self, monkeypatch, when: datetime):
        import analysis.day_classifier as dc

        class _FakeDateTime:
            @staticmethod
            def now(tz=None):
                return when if tz is None else when.astimezone(tz)

        monkeypatch.setattr(dc, "datetime", _FakeDateTime)

    def test_gate_blocks_on_quiet_day_after_midday(self, monkeypatch):
        broker = MagicMock()
        daily = [
            OHLCV(datetime(2026, 6, 16, tzinfo=IST), 24000, 24200, 23950, 24100.0, 1_000_000),
            OHLCV(datetime(2026, 6, 17, tzinfo=IST), 24000, 24200, 23950, 24000.0, 1_000_000),
        ]
        broker.get_ohlcv.side_effect = lambda sym, tf, limit: (
            daily if tf == "1d" else _flat_candles(24000.0)
        )
        self._freeze_now(monkeypatch, _afternoon(11, 30))

        gate = DayGate(broker, enabled=True)
        ok, reason = gate.allows(TradeDirection.LONG)
        assert ok is False
        assert "RANGE_INSIDE" in reason.upper() or "range" in reason.lower()

    def test_gate_does_not_block_quiet_or_in_morning(self, monkeypatch):
        broker = MagicMock()
        daily = [
            OHLCV(datetime(2026, 6, 16, tzinfo=IST), 24000, 24200, 23950, 24100.0, 1_000_000),
            OHLCV(datetime(2026, 6, 17, tzinfo=IST), 24000, 24200, 23950, 24000.0, 1_000_000),
        ]
        broker.get_ohlcv.side_effect = lambda sym, tf, limit: (
            daily if tf == "1d" else _flat_candles(24000.0)
        )
        self._freeze_now(monkeypatch, _morning(9, 30))

        gate = DayGate(broker, enabled=True)
        ok, reason = gate.allows(TradeDirection.LONG)
        assert ok is True, reason

    def test_gate_caches_result_second_call_no_extra_fetch(self, monkeypatch):
        self._freeze_now(monkeypatch, _morning(9, 30))
        broker = self._make_broker()
        gate = DayGate(broker, enabled=True)

        gate.allows(TradeDirection.LONG)
        call_count_after_first = broker.get_ohlcv.call_count
        gate.allows(TradeDirection.SHORT)
        assert broker.get_ohlcv.call_count == call_count_after_first  # no extra fetch

    def test_gate_does_not_cache_unknown(self, monkeypatch):
        self._freeze_now(monkeypatch, _morning(9, 16))

        broker = MagicMock()
        daily = [
            OHLCV(datetime(2026, 6, 16, tzinfo=IST), 24000, 24200, 23950, 24100.0, 1_000_000),
            OHLCV(datetime(2026, 6, 17, tzinfo=IST), 24000, 24200, 23950, 24000.0, 1_000_000),
        ]
        # First call: only 1 candle → UNKNOWN; second: full OR → classifiable
        calls = {"n": 0}

        def _ohlcv(sym, tf, limit):
            if tf == "1d":
                return daily
            calls["n"] += 1
            if calls["n"] == 1:
                return [_c(24000, 24050, 23980, 24020)]
            return _or_candles_bullish(24000.0)

        broker.get_ohlcv.side_effect = _ohlcv
        gate = DayGate(broker, enabled=True)

        first = gate.classify_today()
        assert first.day_type == DayType.UNKNOWN
        second = gate.classify_today()
        # Must re-fetch and produce a real directional classification (not stuck on UNKNOWN)
        assert second.day_type != DayType.UNKNOWN
        assert second.is_blocked is False

    def test_gate_fetch_error_allows_any(self, monkeypatch):
        self._freeze_now(monkeypatch, _morning(9, 30))
        broker = MagicMock()
        broker.get_ohlcv.side_effect = Exception("Dhan 502")
        gate = DayGate(broker, enabled=True)
        ok, _ = gate.allows(TradeDirection.LONG)
        assert ok is True  # fail open — don't block on broker error

    def test_gate_insufficient_daily_candles_allows_any(self, monkeypatch):
        self._freeze_now(monkeypatch, _morning(9, 30))
        broker = MagicMock()
        broker.get_ohlcv.side_effect = lambda sym, tf, limit: [] if tf == "1d" else _or_candles_bullish()
        gate = DayGate(broker, enabled=True)
        ok, _ = gate.allows(TradeDirection.LONG)
        assert ok is True

    def test_gate_reset_forces_reclassification(self, monkeypatch):
        self._freeze_now(monkeypatch, _morning(9, 30))
        broker = self._make_broker()
        gate = DayGate(broker, enabled=True)

        gate.allows(TradeDirection.LONG)
        first_count = broker.get_ohlcv.call_count

        gate.reset()
        gate.allows(TradeDirection.SHORT)
        assert broker.get_ohlcv.call_count > first_count  # re-fetched after reset
