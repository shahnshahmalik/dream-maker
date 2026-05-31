"""Risk manager tests."""

from models.trade_plan import TradeDirection
from risk.manager import RiskManager


def test_compute_size_valid():
    rm = RiskManager(risk_pct_per_trade=1.5, min_rr_ratio=3.0)
    result = rm.compute_size(capital=100_000, entry=100, stop_loss=99, take_profit=103)
    assert result.qty > 0
    assert result.rr_ratio >= 3.0
    assert result.reason == "ok"


def test_compute_size_rejects_low_rr():
    rm = RiskManager(risk_pct_per_trade=1.5, min_rr_ratio=3.0)
    result = rm.compute_size(capital=100_000, entry=100, stop_loss=99, take_profit=101)
    assert result.qty == 0
    assert "R:R" in result.reason


def test_bracket_long():
    rm = RiskManager(1.5, 2.0)
    bp = rm.bracket(100, TradeDirection.LONG, sl_pct=1.0, tp1_pct=2.0, tp2_pct=4.0)
    assert bp.stop_loss == 99.0
    assert bp.take_profit_1 == 102.0
    assert bp.take_profit_2 == 104.0


def test_bracket_short():
    rm = RiskManager(1.5, 2.0)
    bp = rm.bracket(100, TradeDirection.SHORT, sl_pct=1.0, tp1_pct=2.0, tp2_pct=4.0)
    assert bp.stop_loss == 101.0
    assert bp.take_profit_1 == 98.0
