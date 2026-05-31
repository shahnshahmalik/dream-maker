"""Strike selector tests."""

from models.trade_plan import TradeDirection
from risk.strike_selector import select_strike_and_size


def test_futures_lot_sizing():
    sel = select_strike_and_size(
        "NIFTY25JUNFUT",
        TradeDirection.LONG,
        entry=24000,
        stop_loss=23900,
        available_funds=200_000,
        risk_pct=1.5,
        ltp=24000,
    )
    assert sel.affordable
    assert sel.qty >= 25


def test_unaffordable_returns_zero():
    sel = select_strike_and_size(
        "NIFTY25JUNFUT",
        TradeDirection.LONG,
        entry=24000,
        stop_loss=23900,
        available_funds=100,
        risk_pct=1.5,
        ltp=24000,
    )
    assert not sel.affordable
    assert sel.qty == 0
