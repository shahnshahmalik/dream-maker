"""Unit tests for sniper trail math, volume gate, and direction gates."""

from __future__ import annotations

from scalping.config import ExpirySniperConfig, NonExpirySniperConfig
from scalping.core import (
    SignalResult,
    compute_trail_sl,
    capital_target_rupees,
    pick_direction,
    score_signal,
)


def test_trail_never_above_peak_on_cheap_premium():
    """Regression: old ₹8 absolute buffer set SL above LTP on ₹35 options."""
    cfg = ExpirySniperConfig
    entry = 35.40
    peak = 41.75  # +17.9%
    hard_sl = round(entry * (1 - cfg.sl_pct), 2)

    new_sl = compute_trail_sl(entry, peak, hard_sl, cfg)
    assert new_sl is not None
    assert new_sl < peak
    # Must be near breakeven+, not entry+8 (=43.40 which was the bug)
    assert new_sl < entry + 8
    assert new_sl >= entry * (1 + cfg.trail_be_buffer_pct) - 0.05


def test_trail_inactive_before_activate_pct():
    cfg = ExpirySniperConfig
    entry = 100.0
    peak = 110.0  # +10% < 12% activate
    assert compute_trail_sl(entry, peak, 82.0, cfg) is None


def test_trail_ratchets_with_peak():
    cfg = ExpirySniperConfig
    entry = 100.0
    peak = 130.0  # +30%
    be = entry * (1 + cfg.trail_be_buffer_pct)
    sl1 = compute_trail_sl(entry, peak, be, cfg)
    assert sl1 is not None
    # peak trail = 130 * 0.92 = 119.6, capped at peak*0.995
    assert sl1 >= be
    assert abs(sl1 - min(peak * (1 - cfg.trail_from_peak_pct), peak * 0.995)) < 0.02


def test_capital_target_scales_with_position():
    cfg = ExpirySniperConfig
    # Cheap small: floor
    assert capital_target_rupees(30, 65, cfg) == cfg.capital_target_floor
    # Large: capped
    assert capital_target_rupees(150, 65, cfg) == cfg.capital_target_cap


def _bars(n: int = 20, *, vol_last_closed: float = 2000, vol_avg: float = 1000) -> list[dict]:
    """Synthetic rising bars; volumes set so closed-bar gate can pass/fail."""
    out = []
    for i in range(n):
        base = 24000 + i
        # volumes: most bars at vol_avg; bar -2 (closed) = vol_last_closed; last forming smaller
        if i == n - 2:
            v = vol_last_closed
        elif i == n - 1:
            v = vol_avg * 0.3
        else:
            v = vol_avg
        out.append({
            "hh": 10, "mm": i % 60, "time": f"10:{i % 60:02d}",
            "open": float(base), "high": float(base + 5),
            "low": float(base - 2), "close": float(base + 3),
            "volume": float(v),
        })
    return out


def test_volume_gate_uses_closed_bar_not_forming():
    cfg = NonExpirySniperConfig
    # Forming bar is low vol but closed bar is strong → should NOT block
    candles = _bars(20, vol_last_closed=2000, vol_avg=1000)
    htf = _bars(10, vol_last_closed=1000, vol_avg=1000)
    sig = score_signal(
        candles, htf,
        or_high=23900.0, or_low=23800.0, or_set=True,
        vol_mult=cfg.vol_mult,
    )
    assert sig.blocked is None


def test_volume_gate_blocks_weak_closed_bar():
    cfg = NonExpirySniperConfig
    candles = _bars(20, vol_last_closed=500, vol_avg=1000)
    htf = _bars(10, vol_last_closed=1000, vol_avg=1000)
    sig = score_signal(
        candles, htf,
        or_high=23900.0, or_low=23800.0, or_set=True,
        vol_mult=cfg.vol_mult,
    )
    assert sig.blocked is not None
    assert "low_vol" in sig.blocked


def test_pick_direction_requires_primary():
    cfg = ExpirySniperConfig
    # Align + HTF only — no primary cross/ORB/VWAP
    sig = SignalResult(bull=3, bear=0, reasons=["EMA_align_bull", "HTF_bull", "HH_HL_struct"], htf_bias="bull")
    assert pick_direction(sig, cfg) is None

    sig2 = SignalResult(
        bull=4, bear=0,
        reasons=["EMA_cross_bull", "HTF_bull", "HH_HL_struct"],
        htf_bias="bull",
    )
    assert pick_direction(sig2, cfg) == "CE"


def test_pick_direction_blocks_htf_conflict():
    cfg = ExpirySniperConfig
    sig = SignalResult(
        bull=3, bear=1,
        reasons=["EMA_cross_bull", "ORB_above", "HTF_bear"],
        htf_bias="bear",
    )
    # score 3 < min_signal+2 → blocked
    assert pick_direction(sig, cfg) is None

    sig_strong = SignalResult(
        bull=5, bear=1,
        reasons=["EMA_cross_bull", "ORB_bull_break>100", "VWAP_reject_bull@1", "HTF_bear"],
        htf_bias="bear",
    )
    assert pick_direction(sig_strong, cfg) == "CE"
