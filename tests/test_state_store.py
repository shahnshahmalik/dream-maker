"""State store persistence tests."""

import json
from pathlib import Path

from audit.state_store import StateStore
from models.trade_plan import EntryType, PlanStatus, TradeDirection, TradePlan


def test_save_and_load_plans(tmp_path: Path):
    store = StateStore(tmp_path, tmp_path / "trade_log.jsonl")
    plan = TradePlan(
        symbol="NIFTY50IDX",
        direction=TradeDirection.LONG,
        timeframe="15m",
        bias_source="test",
        entry_zone="24000",
        entry_type=EntryType.LIMIT,
        stop_loss=23900,
        stop_loss_reason="support",
        take_profit_1=24300,
        tp1_exit_pct=50,
        take_profit_2=24600,
        tp2_exit_pct=50,
        risk_amount=1500,
        position_size=25,
        rr_ratio=3.0,
        status=PlanStatus.WAITING_ENTRY,
        entry_price_low=23980,
        entry_price_high=24020,
    )
    store.save_plans([plan])
    loaded = store.load_plans()
    assert len(loaded) == 1
    assert loaded[0].plan_id == plan.plan_id
    assert loaded[0].status == PlanStatus.WAITING_ENTRY


def test_load_from_trade_log(tmp_path: Path):
    log_path = tmp_path / "trade_log.jsonl"
    plan = TradePlan(
        symbol="NIFTY50IDX",
        direction=TradeDirection.LONG,
        timeframe="15m",
        bias_source="test",
        entry_zone="24000",
        entry_type=EntryType.LIMIT,
        stop_loss=23900,
        stop_loss_reason="support",
        take_profit_1=24300,
        tp1_exit_pct=50,
        take_profit_2=24600,
        tp2_exit_pct=50,
        risk_amount=1500,
        position_size=25,
        rr_ratio=3.0,
        status=PlanStatus.ACTIVE,
    )
    entry = {
        "action": "ORDER",
        "symbol": "NIFTY50IDX",
        "details": {"success": True, "plan": plan.to_dict()},
    }
    log_path.write_text(json.dumps(entry) + "\n", encoding="utf-8")
    store = StateStore(tmp_path, log_path)
    recovered = store.load_from_trade_log("NIFTY50IDX")
    assert recovered is not None
    assert recovered.plan_id == plan.plan_id
