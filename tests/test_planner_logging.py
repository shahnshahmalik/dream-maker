"""Trade planner logging helpers."""

from models.trade_plan import EntryType, PlanStatus, TradeDirection, TradePlan
from agent.planner import TradePlanner


def test_log_levels_does_not_raise(caplog):
    import logging

    caplog.set_level(logging.INFO)
    plan = TradePlan(
        symbol="NIFTY50IDX",
        direction=TradeDirection.LONG,
        timeframe="15m",
        bias_source="HTF uptrend",
        entry_zone="24500.00",
        entry_type=EntryType.LIMIT,
        stop_loss=24400.0,
        stop_loss_reason="HTF support",
        take_profit_1=24800.0,
        tp1_exit_pct=50.0,
        take_profit_2=24950.0,
        tp2_exit_pct=50.0,
        risk_amount=100.0,
        position_size=25,
        rr_ratio=3.0,
        status=PlanStatus.WAITING_ENTRY,
        entry_price_low=24480.0,
        entry_price_high=24520.0,
        signal_strength=0.72,
        rationale="test setup",
    )
    TradePlanner._log_levels(
        "AI levels approved",
        plan,
        insight="Momentum aligned with macro",
        provider="deepseek",
    )
    assert "AI levels approved" in caplog.text
    assert "24500" in caplog.text or "24480" in caplog.text
    assert "Momentum aligned with macro" in caplog.text
