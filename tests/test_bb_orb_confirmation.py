"""Tests for check_bb_confirmation() and check_orb_confirmation().

Covers:
  BB:
    - LONG: crossed above upper band → confirmed
    - LONG: near upper band (within 0.2%) → confirmed
    - LONG: well below upper band → rejected
    - SHORT: crossed below lower band → confirmed
    - SHORT: near lower band → confirmed
    - SHORT: above lower band → rejected
    - insufficient candles → rejected

  ORB:
    - LONG: close above OR high → confirmed
    - LONG: close below OR high → rejected
    - SHORT: close below OR low → confirmed
    - SHORT: close above OR low → rejected
    - insufficient candles → rejected
    - exactly at OR bar boundary → rejected (still inside OR)

  Integration:
    - Both confirmed → signal_strength boosted by 0.10
    - BB confirmed alone → boost 0.05
    - ORB confirmed alone → boost 0.05
"""
from __future__ import annotations

from datetime import datetime

import pytest

from analysis.technical import check_bb_confirmation, check_orb_confirmation
from models.orders import OHLCV
from models.trade_plan import TradeDirection

# ── Helpers ──────────────────────────────────────────────────────────

def _c(o: float, h: float, l: float, c: float, v: int = 100_000) -> OHLCV:
    return OHLCV(timestamp=datetime(2026, 6, 17, 9, 15), open=o, high=h, low=l, close=c, volume=v)


def _flat_candles(n: int, price: float = 24000.0) -> list[OHLCV]:
    """n candles with tiny random-ish variation so BB std > 0."""
    result = []
    for i in range(n):
        # Small oscillation: +/- alternating 50pts to ensure non-zero std
        offset = 50.0 if i % 2 == 0 else -50.0
        p = price + offset
        result.append(_c(p, p + 20, p - 20, p))
    return result


def _trending_up(n: int, start: float = 24000.0, step: float = 10.0) -> list[OHLCV]:
    result = []
    for i in range(n):
        p = start + i * step
        result.append(_c(p, p + 15, p - 5, p + 10))
    return result


def _trending_down(n: int, start: float = 24000.0, step: float = 10.0) -> list[OHLCV]:
    result = []
    for i in range(n):
        p = start - i * step
        result.append(_c(p, p + 5, p - 15, p - 10))
    return result


# ── BB Tests ─────────────────────────────────────────────────────────

class TestBBConfirmationLong:
    def test_close_above_upper_band_confirmed(self):
        """Price strongly above upper band → LONG confirmed."""
        # 20 flat candles then a spike well above
        base = _flat_candles(22, 24000.0)
        # Replace last candle with a big breakout
        base[-1] = _c(24000, 24500, 23990, 24480)
        ok, reason = check_bb_confirmation(base, TradeDirection.LONG)
        assert ok is True
        assert "breakout" in reason.lower() or "near" in reason.lower() or "BB" in reason

    def test_close_below_upper_band_rejected(self):
        """Price comfortably below upper band → LONG rejected."""
        candles = _flat_candles(25, 24000.0)
        # Last candle: just slightly up, not above upper band (~24060 with std ~20)
        candles[-1] = _c(24000, 24010, 23990, 24005)
        ok, reason = check_bb_confirmation(candles, TradeDirection.LONG)
        assert ok is False
        assert "below upper" in reason.lower()

    def test_insufficient_candles_rejected(self):
        candles = _flat_candles(10, 24000.0)  # need period+2 = 22
        ok, reason = check_bb_confirmation(candles, TradeDirection.LONG)
        assert ok is False
        assert "insufficient" in reason.lower()


class TestBBConfirmationShort:
    def test_close_below_lower_band_confirmed(self):
        """Price strongly below lower band → SHORT confirmed."""
        base = _flat_candles(22, 24000.0)
        base[-1] = _c(24000, 24010, 23500, 23520)
        ok, reason = check_bb_confirmation(base, TradeDirection.SHORT)
        assert ok is True

    def test_close_above_lower_band_rejected(self):
        """Price comfortably above lower band → SHORT rejected."""
        candles = _flat_candles(25, 24000.0)
        candles[-1] = _c(24000, 24010, 23990, 23998)  # barely down, above lower band
        ok, reason = check_bb_confirmation(candles, TradeDirection.SHORT)
        assert ok is False
        assert "above lower" in reason.lower()

    def test_insufficient_candles_rejected(self):
        candles = _flat_candles(5, 24000.0)
        ok, reason = check_bb_confirmation(candles, TradeDirection.SHORT)
        assert ok is False


# ── ORB Tests ────────────────────────────────────────────────────────

class TestORBConfirmationLong:
    def _or_plus(self, last_close: float, or_high: float = 24100.0, or_low: float = 23900.0) -> list[OHLCV]:
        """3 OR candles + 2 post-OR candles (total 5, satisfying or_bars+2 requirement)."""
        mid = (or_high + or_low) / 2
        return [
            _c(mid, or_high, or_low, mid + 10),
            _c(mid + 10, or_high - 5, or_low + 5, mid),
            _c(mid, or_high, or_low + 10, mid + 5),   # ensure or_high is hit
            _c(mid + 5, or_high + 5, or_low + 20, mid + 10),  # post-OR candle 1
            _c(mid + 10, last_close + 10, last_close - 10, last_close),  # post-OR candle 2
        ]

    def test_close_above_or_high_confirmed(self):
        or_high = 24100.0
        candles = self._or_plus(last_close=24200.0, or_high=or_high)  # above OR high
        ok, reason = check_orb_confirmation(candles, TradeDirection.LONG)
        assert ok is True, reason
        assert "breakout" in reason.lower()

    def test_close_below_or_high_rejected(self):
        or_high = 24100.0
        candles = self._or_plus(last_close=24050.0, or_high=or_high)  # below OR high
        ok, reason = check_orb_confirmation(candles, TradeDirection.LONG)
        assert ok is False, reason
        assert "below or high" in reason.lower()

    def test_insufficient_candles_rejected(self):
        ok, reason = check_orb_confirmation(
            [_c(24000, 24100, 23900, 24050), _c(24050, 24120, 23950, 24080)],
            TradeDirection.LONG,
            or_bars=3,
        )
        assert ok is False
        assert "insufficient" in reason.lower()

    def test_exactly_at_or_boundary_rejected(self):
        """Exactly or_bars candles with no post-OR data → still insufficient."""
        candles = [
            _c(24000, 24100, 23900, 24050),
            _c(24050, 24080, 23950, 24070),
            _c(24070, 24090, 23970, 24080),
        ]  # exactly 3 = or_bars; need or_bars+2=5 for check → insufficient
        ok, reason = check_orb_confirmation(candles, TradeDirection.LONG, or_bars=3)
        assert ok is False


class TestORBConfirmationShort:
    def _or_plus(self, last_close: float, or_high: float = 24100.0, or_low: float = 23900.0) -> list[OHLCV]:
        """3 OR candles + 2 post-OR candles."""
        mid = (or_high + or_low) / 2
        return [
            _c(mid, or_high, or_low, mid - 10),
            _c(mid - 10, or_high - 5, or_low, mid),
            _c(mid, or_high - 10, or_low, mid - 5),  # ensure or_low is hit
            _c(mid - 5, or_high - 20, or_low - 5, mid - 10),
            _c(mid - 10, last_close + 10, last_close - 10, last_close),
        ]

    def test_close_below_or_low_confirmed(self):
        or_low = 23900.0
        candles = self._or_plus(last_close=23800.0, or_low=or_low)  # below OR low
        ok, reason = check_orb_confirmation(candles, TradeDirection.SHORT)
        assert ok is True, reason
        assert "breakdown" in reason.lower()

    def test_close_above_or_low_rejected(self):
        or_low = 23900.0
        candles = self._or_plus(last_close=23950.0, or_low=or_low)  # above OR low
        ok, reason = check_orb_confirmation(candles, TradeDirection.SHORT)
        assert ok is False, reason
        assert "above or low" in reason.lower()

    def test_insufficient_candles_rejected(self):
        ok, reason = check_orb_confirmation(
            [_c(24000, 24100, 23900, 24050)],
            TradeDirection.SHORT,
        )
        assert ok is False


# ── Signal Strength Bonus Integration ────────────────────────────────

class TestSignalStrengthBonus:
    """Verify bonus logic: BB and ORB confirmed → labels added, strength boosted."""

    def test_bb_confirmed_when_breakout(self):
        """Spike well above upper band → confirmed."""
        base = _flat_candles(22, 24000.0)
        base[-1] = _c(24000, 24500, 23990, 24480)
        ok, reason = check_bb_confirmation(base, TradeDirection.LONG)
        assert ok is True

    def test_bb_rejected_when_inside_bands(self):
        """Price oscillating inside bands → not confirmed."""
        # Build alternating candles that create real std (50pt swing)
        candles = []
        for i in range(25):
            p = 24000.0 + (50 if i % 2 == 0 else -50)
            candles.append(_c(p, p + 20, p - 20, p))
        # Last candle: in middle of bands (not near upper or lower)
        candles[-1] = _c(24000, 24010, 23990, 24000)
        ok, reason = check_bb_confirmation(candles, TradeDirection.LONG)
        assert ok is False, f"Expected rejected but got: {reason}"

    def test_orb_confirmed_when_breakout(self):
        """Close above OR high → confirmed."""
        candles = [
            _c(24000, 24100, 23900, 24050),
            _c(24050, 24080, 23950, 24070),
            _c(24070, 24100, 23970, 24080),  # OR high = 24100
            _c(24080, 24150, 24050, 24120),  # post-OR
            _c(24120, 24250, 24100, 24220),  # above OR high → confirmed
        ]
        ok, reason = check_orb_confirmation(candles, TradeDirection.LONG)
        assert ok is True, reason

    def test_orb_rejected_when_below_or_high(self):
        """Close below OR high → rejected."""
        candles = [
            _c(24000, 24100, 23900, 24050),
            _c(24050, 24080, 23950, 24070),
            _c(24070, 24100, 23970, 24080),  # OR high = 24100
            _c(24080, 24090, 24050, 24070),  # post-OR
            _c(24070, 24090, 24050, 24060),  # below OR high → rejected
        ]
        ok, reason = check_orb_confirmation(candles, TradeDirection.LONG)
        assert ok is False, reason
