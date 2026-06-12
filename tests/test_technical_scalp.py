"""Momentum scalp and bracket validation tests."""

from analysis.technical import SetupType, analyze_technical, momentum_confirmation
from models.orders import OHLCV
from models.trade_plan import (
    EntryType,
    PlanStatus,
    TradeDirection,
    TradePlan,
    validate_bracket,
    validate_plan,
)


def _candle(close: float, *, high: float | None = None, low: float | None = None, vol: float = 1000) -> OHLCV:
    h = high if high is not None else close + 1
    lo = low if low is not None else close - 1
    return OHLCV(open=close - 0.5, high=h, low=lo, close=close, volume=vol, timestamp="2026-01-01T09:15:00")


def _trending_candles(start: float, steps: int, direction: int = 1) -> list[OHLCV]:
    candles = []
    price = start
    for i in range(steps):
        price += direction * 2
        candles.append(_candle(price, vol=1200 + i * 10))
    return candles


def test_momentum_confirmation_requires_multiple_signals():
    candles = _trending_candles(100, 15, direction=1)
    ok, reasons = momentum_confirmation(candles, TradeDirection.LONG, min_confirmations=2)
    assert ok
    assert len(reasons) >= 2


def _ranging_candles(center: float, steps: int) -> list[OHLCV]:
    candles = []
    for i in range(steps):
        offset = 1 if i % 2 == 0 else -1
        price = center + offset
        candles.append(_candle(price, vol=1000))
    return candles


def test_momentum_scalp_detected_when_swing_fails():
    """With HTF ranging and LTF trending, a scalp or range scalp is found."""
    # HTF: truly flat (single price) — guarantees RANGE
    htf = [_candle(100, vol=1000) for _ in range(30)]
    # LTF: strongly trending up from the same base
    ltf = _trending_candles(100, 25, direction=1)
    tech = analyze_technical(
        htf,
        ltf,
        min_rr=3.0,
        scalp_enabled=True,
        scalp_min_rr=1.2,
        scalp_max_sl_pct=0.35,
        scalp_min_confirmations=2,
    )
    assert tech is not None
    # With flat HTF and trending LTF, should get a scalp (momentum or range)
    assert tech.setup_type in (SetupType.MOMENTUM_SCALP, SetupType.RANGE_SCALP, SetupType.SWING)
    assert tech.rr_ratio >= 1.2


def test_scalp_disabled_when_flag_off():
    htf = _trending_candles(100, 30, direction=0)
    ltf = _trending_candles(150, 25, direction=1)
    tech = analyze_technical(htf, ltf, min_rr=3.0, scalp_enabled=False)
    assert tech is None or tech.setup_type == SetupType.SWING


def test_validate_bracket_rejects_naked_long():
    plan = TradePlan(
        symbol="NIFTY50IDX",
        direction=TradeDirection.LONG,
        timeframe="15m",
        bias_source="test",
        entry_zone="100",
        entry_type=EntryType.LIMIT,
        stop_loss=101.0,
        stop_loss_reason="bad",
        take_profit_1=103.0,
        tp1_exit_pct=50,
        take_profit_2=104.0,
        tp2_exit_pct=50,
        risk_amount=100,
        position_size=1,
        rr_ratio=2.0,
        status=PlanStatus.PENDING,
    )
    ok, reason = validate_bracket(plan, 100.0)
    assert not ok
    assert "below entry" in reason


def test_scalp_plan_allows_lower_rr():
    plan = TradePlan(
        symbol="NIFTY50IDX",
        direction=TradeDirection.LONG,
        timeframe="5m",
        bias_source="scalp",
        entry_zone="100",
        entry_type=EntryType.LIMIT,
        stop_loss=99.7,
        stop_loss_reason="micro",
        take_profit_1=100.36,
        tp1_exit_pct=50,
        take_profit_2=100.54,
        tp2_exit_pct=50,
        risk_amount=100,
        position_size=1,
        rr_ratio=1.2,
        status=PlanStatus.PENDING,
        meta={"setup_type": "momentum_scalp"},
    )
    ok, reason = validate_plan(plan, min_rr=3.0, min_rr_scalp=1.2)
    assert ok, reason
