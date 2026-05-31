"""Trade plan validation tests."""

from models.trade_plan import (
    EntryType,
    PlanStatus,
    TradeDirection,
    TradePlan,
    validate_plan,
)


def _sample_plan(**kwargs) -> TradePlan:
    defaults = dict(
        symbol="RELIANCE",
        direction=TradeDirection.LONG,
        timeframe="15m",
        bias_source="test",
        entry_zone="2500",
        entry_type=EntryType.LIMIT,
        stop_loss=2450.0,
        stop_loss_reason="support",
        take_profit_1=2600.0,
        tp1_exit_pct=50.0,
        take_profit_2=2700.0,
        tp2_exit_pct=50.0,
        risk_amount=1500.0,
        position_size=10,
        rr_ratio=2.0,
        status=PlanStatus.PENDING,
    )
    defaults.update(kwargs)
    return TradePlan(**defaults)


def test_valid_plan():
    plan = _sample_plan()
    ok, reason = validate_plan(plan, min_rr=2.0)
    assert ok
    assert reason == "ok"


def test_rejects_low_rr():
    plan = _sample_plan(rr_ratio=1.5)
    ok, reason = validate_plan(plan, min_rr=2.0)
    assert not ok
    assert "R:R" in reason


def test_rejects_zero_size():
    plan = _sample_plan(position_size=0)
    ok, reason = validate_plan(plan, min_rr=2.0)
    assert not ok
