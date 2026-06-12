"""Tests for SessionTracker — daily discipline."""

import pytest
from risk.session_tracker import SessionTracker, SessionState


class FakeConfig:
    max_trades_per_day = 4
    daily_profit_target_pct = 3.0
    daily_loss_limit_inr = 1000


def _make_tracker() -> SessionTracker:
    cfg = FakeConfig()
    st = SessionTracker(cfg)
    st.set_initial_capital(16000)
    return st


def test_can_trade_initially():
    st = _make_tracker()
    assert st.can_trade is True
    assert st.state == SessionState.ACTIVE


def test_register_entry_increments_count():
    st = _make_tracker()
    st.register_entry("NIFTY26JUN24000CE", 200)
    assert st.trade_count == 1


def test_trade_cap_stops_at_max():
    st = _make_tracker()
    for i in range(4):
        st.register_entry(f"TEST{i}", 200)
    assert st.trade_count == 4
    assert st.state == SessionState.TRADE_CAP
    assert st.can_trade is False


def test_profit_target_hit():
    st = _make_tracker()
    st.set_initial_capital(16000)
    st.register_entry("TEST", 200)
    st.register_close(pnl=500)  # 500/16000 = 3.125% > 3.0%
    assert st.state == SessionState.PROFIT_TARGET_HIT
    assert st.can_trade is False


def test_profit_target_not_hit_below_threshold():
    st = _make_tracker()
    st.set_initial_capital(16000)
    st.register_entry("TEST", 200)
    st.register_close(pnl=400)  # 400/16000 = 2.5% < 3.0%
    assert st.state == SessionState.ACTIVE
    assert st.can_trade is True


def test_daily_loss_limit_hit():
    """Loss guard triggers when daily P&L crosses -₹1000 (not by loss count)."""
    st = _make_tracker()
    st.register_entry("T1", 200)
    st.register_close(pnl=-500)  # P&L: -500 (under ₹1000 limit)
    assert st.state == SessionState.ACTIVE
    st.register_entry("T2", 200)
    st.register_close(pnl=-400)  # P&L: -900 (still under)
    assert st.state == SessionState.ACTIVE
    st.register_entry("T3", 200)
    st.register_close(pnl=-200)  # P&L: -1100 → triggers loss guard
    assert st.state == SessionState.LOSS_GUARD
    assert st.can_trade is False
    assert "₹" in st.stop_reason


def test_win_improves_daily_pnl():
    """A winning trade improves daily P&L, keeping loss guard from triggering."""
    st = _make_tracker()
    st.register_entry("T1", 200)
    st.register_close(pnl=-500)  # P&L: -500
    assert st.daily_pnl == -500
    st.register_entry("T2", 200)
    st.register_close(pnl=300)  # P&L: -200 (win partially recovered)
    assert st.daily_pnl == -200
    assert st.state == SessionState.ACTIVE


def test_size_multiplier_after_loss():
    st = _make_tracker()
    assert st.size_multiplier == 1.0
    st.register_entry("T1", 200)
    st.register_close(pnl=-500)
    assert st.size_multiplier == 0.5
    st.register_entry("T2", 200)
    st.register_close(pnl=300)  # win → back to 1.0
    assert st.size_multiplier == 1.0


def test_day_rollover_resets():
    st = _make_tracker()
    st.register_entry("T1", 200)
    st.register_close(pnl=-500)
    st.register_close(pnl=-600)  # P&L: -1100 → loss guard
    st._state = SessionState.LOSS_GUARD
    # Simulate day change
    st._today = "2025-01-01"
    st._check_day_rollover()
    assert st.trade_count == 0
    assert st.daily_pnl == 0.0
    assert st._consecutive_losses == 0
    assert st.state == SessionState.ACTIVE


def test_status_summary():
    st = _make_tracker()
    st.register_entry("T1", 200)
    st.register_close(pnl=300)
    summary = st.status_summary()
    assert "1/4 trades" in summary
    assert "+300" in summary
    assert "active" in summary.lower()
