"""Tests for balance_manager.py — BUY_ONLY vs FULL capability."""

import pytest
from risk.balance_manager import BalanceManager, Capability, Decision


class TestBalanceManager:
    def test_buy_only_below_threshold(self):
        mgr = BalanceManager(min_sell_balance=100_000, sell_buffer=1.5)
        decision = mgr.evaluate(50_000)  # below 150K threshold
        assert decision.capability == Capability.BUY_ONLY

    def test_full_above_threshold(self):
        mgr = BalanceManager(min_sell_balance=100_000, sell_buffer=1.5)
        decision = mgr.evaluate(200_000)
        assert decision.capability == Capability.FULL

    def test_buy_only_at_exact_threshold(self):
        mgr = BalanceManager(min_sell_balance=100_000, sell_buffer=1.0)
        decision = mgr.evaluate(100_000)
        assert decision.capability == Capability.FULL  # >= threshold

    def test_buy_only_just_below(self):
        mgr = BalanceManager(min_sell_balance=100_000, sell_buffer=1.0)
        decision = mgr.evaluate(99_999)
        assert decision.capability == Capability.BUY_ONLY

    def test_reason_includes_balance(self):
        mgr = BalanceManager(min_sell_balance=100_000, sell_buffer=1.5)
        decision = mgr.evaluate(10_000)
        assert "₹10,000" in decision.reason
        assert "BUY only" in decision.reason

    def test_default_thresholds(self):
        mgr = BalanceManager()
        assert mgr.min_sell == 100_000
        assert mgr.buffer == 1.5

    def test_get_side_buy_only_returns_buy(self):
        from models.trade_plan import TradeDirection
        mgr = BalanceManager(min_sell_balance=100_000, sell_buffer=1.5)
        # Low balance: always BUY regardless of direction
        assert mgr.get_side(TradeDirection.LONG, 10_000) == "BUY"
        assert mgr.get_side(TradeDirection.SHORT, 10_000) == "BUY"

    def test_get_side_full_returns_correct(self):
        from models.trade_plan import TradeDirection
        mgr = BalanceManager(min_sell_balance=100_000, sell_buffer=1.0)
        # High balance: LONG→BUY, SHORT→SELL
        assert mgr.get_side(TradeDirection.LONG, 200_000) == "BUY"
        assert mgr.get_side(TradeDirection.SHORT, 200_000) == "SELL"

    def test_custom_threshold(self):
        mgr = BalanceManager(min_sell_balance=50_000, sell_buffer=2.0)
        decision = mgr.evaluate(90_000)
        assert decision.capability == Capability.BUY_ONLY  # 90K < 100K (50K × 2)
        decision2 = mgr.evaluate(110_000)
        assert decision2.capability == Capability.FULL  # 110K ≥ 100K

    def test_capability_enum_values(self):
        assert Capability.BUY_ONLY.value == "buy_only"
        assert Capability.FULL.value == "full"

    def test_decision_dataclass(self):
        d = Decision(Capability.BUY_ONLY, "test reason")
        assert d.capability == Capability.BUY_ONLY
        assert d.reason == "test reason"
