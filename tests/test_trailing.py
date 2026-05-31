"""Trailing SL/TP manager tests."""

from models.trade_plan import TradeDirection
from risk.trailing import TrailingBracketManager


def _manager(**kwargs) -> TrailingBracketManager:
    defaults = dict(
        direction=TradeDirection.LONG,
        entry=100.0,
        initial_sl=99.0,
        initial_tp1=103.0,
        initial_tp2=104.5,
        trail_activate_pct=0.5,
        trail_sl_distance_pct=0.35,
        trail_tp_reward_pct=1.0,
        breakeven_progress_pct=20.0,
        breakeven_buffer_pct=0.05,
    )
    defaults.update(kwargs)
    return TrailingBracketManager(**defaults)


def test_breakeven_at_20pct_progress_long():
    """Entry 100, TP1 103 → 20% progress = 100.6 → SL to cost."""
    mgr = _manager()
    update = mgr.on_price(100.6)
    assert update.new_sl is not None
    assert update.new_sl >= 100.0
    assert mgr.state.breakeven_done
    assert "breakeven" in update.reason
    assert "20%" in update.reason


def test_no_breakeven_before_20pct_progress():
    mgr = _manager()
    update = mgr.on_price(100.5)
    assert update.new_sl is None
    assert not mgr.state.breakeven_done


def test_breakeven_short():
    mgr = _manager(
        direction=TradeDirection.SHORT,
        entry=100.0,
        initial_sl=101.0,
        initial_tp1=97.0,
        initial_tp2=95.5,
    )
    update = mgr.on_price(99.2)
    assert update.new_sl is not None
    assert update.new_sl <= 100.0
    assert mgr.state.breakeven_done


def test_trail_sl_and_tp_long():
    mgr = _manager()
    mgr.on_price(100.6)
    update = mgr.on_price(101.0)
    assert update.new_sl is not None or update.new_tp is not None
    if update.new_sl:
        assert update.new_sl > 99.0
    if update.new_tp:
        assert update.new_tp > 103.0


def test_no_trail_when_adverse():
    mgr = _manager()
    update = mgr.on_price(99.5)
    assert update.new_sl is None
    assert update.new_tp is None
