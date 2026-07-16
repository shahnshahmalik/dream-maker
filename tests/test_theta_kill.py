"""Tests for ThetaKillConfig parameters and exit_after_theta_kill engine behavior."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from scalping.config import ExpirySniperConfig, NonExpirySniperConfig, ThetaKillConfig
from scalping.core import (
    EngineState,
    SignalResult,
    SniperEngine,
    compute_trail_sl,
    pick_direction,
)


# ── ThetaKillConfig sanity checks ─────────────────────────────────────────────

def test_theta_kill_config_is_tuesday_only():
    assert ThetaKillConfig.run_on_weekday == 1


def test_theta_kill_config_entries_from_14_15():
    assert ThetaKillConfig.entries_from == (14, 15)


def test_theta_kill_config_theta_kill_15_15():
    assert ThetaKillConfig.theta_kill == (15, 15)


def test_theta_kill_tp_is_100_pct():
    assert ThetaKillConfig.tp_pct == 1.00


def test_theta_kill_sl_wider_than_expiry():
    """Phase 2 SL must be wider than Phase 1 — cheap options need more room."""
    assert ThetaKillConfig.sl_pct > ExpirySniperConfig.sl_pct


def test_theta_kill_max_trades_2():
    assert ThetaKillConfig.max_trades == 2


def test_theta_kill_premium_ceiling_lower_than_expiry():
    """Phase 2 targets cheap options only."""
    assert ThetaKillConfig.max_premium < ExpirySniperConfig.max_premium


def test_theta_kill_vix_and_oi_gates_off():
    """VIX/OI gates are irrelevant in the final hour — must be disabled."""
    assert ThetaKillConfig.use_vix_gate is False
    assert ThetaKillConfig.use_oi_gate is False


def test_theta_kill_htf_align_off():
    """HTF EMAs are stale by 14:15 — require_htf_align must be False."""
    assert ThetaKillConfig.require_htf_align is False


def test_theta_kill_same_day_expiry_ok():
    assert ThetaKillConfig.same_day_expiry_ok is True


def test_theta_kill_exit_after_theta_kill():
    assert ThetaKillConfig.exit_after_theta_kill is True


def test_expiry_phase1_exit_after_theta_kill():
    """Phase 1 must hand off to Phase 2 by setting exit_after_theta_kill=True."""
    assert ExpirySniperConfig.exit_after_theta_kill is True


def test_non_expiry_does_not_exit_after_theta_kill():
    """Non-expiry engine should idle until market close, not hand off."""
    assert NonExpirySniperConfig.exit_after_theta_kill is False


def test_theta_kill_daily_max_loss_tight():
    """Phase 2 should halt quickly on loss — Phase 1 gains must be protected."""
    assert ThetaKillConfig.daily_max_loss <= 600.0


def test_theta_kill_loop_secs_faster_than_expiry():
    """Final-hour moves happen in seconds — Phase 2 loop must be faster."""
    assert ThetaKillConfig.loop_secs < ExpirySniperConfig.loop_secs


# ── exit_after_theta_kill engine behavior ─────────────────────────────────────

def _make_engine_past_theta_kill(cfg, *, active: bool = False) -> SniperEngine:
    """Return an engine whose inner loop will immediately hit the theta kill branch."""
    engine = SniperEngine(cfg)
    engine.state.active = active
    return engine


def test_exit_after_theta_kill_breaks_loop():
    """Engine with exit_after_theta_kill=True must return from run() at theta kill time."""
    engine = _make_engine_past_theta_kill(ExpirySniperConfig)

    call_count = 0

    def _fake_hm():
        nonlocal call_count
        call_count += 1
        # First call: past theta kill (14, 30). Subsequent calls still past.
        return (14, 30)

    with (
        patch("scalping.core.hm", side_effect=_fake_hm),
        patch("scalping.core.past_theta_kill", return_value=True),
        patch("scalping.core.past_market_close", return_value=False),
        patch("scalping.core.SniperEngine.calendar_allows", return_value=True),
        patch("scalping.core.SniperEngine.restore_position"),
        patch("time.sleep"),
    ):
        engine.run()

    # The engine should have broken out — not looped indefinitely
    assert not engine.state.active


def test_exit_after_theta_kill_false_does_not_break():
    """Engine with exit_after_theta_kill=False must NOT break at theta kill — it idles."""
    engine = _make_engine_past_theta_kill(NonExpirySniperConfig)

    tick = 0

    def _fake_past_theta_kill(_cfg=None):
        return True

    def _fake_past_market_close():
        nonlocal tick
        tick += 1
        # Allow a few idle ticks then signal market close so the test ends
        return tick >= 3

    with (
        patch("scalping.core.past_theta_kill", side_effect=_fake_past_theta_kill),
        patch("scalping.core.past_market_close", side_effect=_fake_past_market_close),
        patch("scalping.core.SniperEngine.calendar_allows", return_value=True),
        patch("scalping.core.SniperEngine.restore_position"),
        patch("time.sleep"),
    ):
        engine.run()

    # Engine ran multiple ticks in idle (not an early break)
    assert tick >= 3


# ── Trail math on Phase 2 cheap options ───────────────────────────────────────

def test_theta_kill_trail_activates_at_30_pct():
    cfg = ThetaKillConfig
    entry = 12.0
    peak_below = entry * 1.20   # +20% — below 30% activate
    assert compute_trail_sl(entry, peak_below, entry * (1 - cfg.sl_pct), cfg) is None


def test_theta_kill_trail_activates_above_30_pct():
    cfg = ThetaKillConfig
    entry = 12.0
    peak = entry * 1.35   # +35% — above 30% activate
    sl = compute_trail_sl(entry, peak, entry * (1 - cfg.sl_pct), cfg)
    assert sl is not None
    assert sl > entry  # SL is above entry (breakeven protection)
    assert sl < peak   # SL never above peak (no instant stop-out)


# ── pick_direction with ThetaKillConfig ───────────────────────────────────────

def test_theta_kill_direction_requires_primary():
    """require_primary=True in ThetaKillConfig — alignment alone must not fire."""
    sig = SignalResult(
        bull=3, bear=0,
        reasons=["EMA_align_bull", "HTF_bull", "HH_HL_struct"],
        htf_bias="bull",
    )
    assert pick_direction(sig, ThetaKillConfig) is None


def test_theta_kill_direction_fires_on_primary():
    """A primary trigger (EMA cross) + score >= 3 must yield a direction."""
    sig = SignalResult(
        bull=4, bear=0,
        reasons=["EMA_cross_bull", "ORB_bull_break>24000", "HH_HL_struct", "HTF_bull"],
        htf_bias="bull",
    )
    assert pick_direction(sig, ThetaKillConfig) == "CE"


def test_theta_kill_htf_conflict_does_not_block():
    """require_htf_align=False in ThetaKillConfig — HTF conflict must be ignored."""
    sig = SignalResult(
        bull=3, bear=1,
        reasons=["EMA_cross_bull", "ORB_bull_break>24000", "HTF_bear"],
        htf_bias="bear",
    )
    # With require_htf_align=False, HTF conflict does not block a score-3 CE
    direction = pick_direction(sig, ThetaKillConfig)
    assert direction == "CE"
