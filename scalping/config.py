"""Mode-specific sniper configs. Fewer trades, higher conviction."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SniperConfig:
    name: str
    version: str
    # Calendar
    run_on_weekday: int | None  # 0=Mon … 6=Sun; None = any except skip_weekday
    skip_weekday: int | None

    # Bars
    entry_interval: str  # "1" or "5"
    lookback_entry: int
    lookback_htf: int

    # ORB
    orb_start: tuple[int, int]  # inclusive
    orb_end_mm: tuple[int, int]  # inclusive last minute of OR window
    entries_from: tuple[int, int]

    # Signals
    min_signal: int
    vol_mult: float
    require_primary: bool  # EMA cross / fresh ORB / VWAP reject
    require_htf_align: bool

    # Risk
    tp_pct: float
    sl_pct: float
    max_trades: int
    exceptional_score: int | None  # allow +1 trade at this score; None = hard cap
    daily_target: float
    daily_max_loss: float
    max_consecutive_losses: int
    capital_target_pct: float  # fraction of position cost
    capital_target_floor: float
    capital_target_cap: float

    # Trail (percent-based — never absolute rupees)
    trail_activate_pct: float
    trail_be_buffer_pct: float
    trail_from_peak_pct: float
    high_momentum_score: int
    momentum_refresh_ticks: int

    # Premium filters (ATM sniper — skip lottery / whale premiums)
    min_premium: float
    max_premium: float

    # Session
    theta_kill: tuple[int, int]
    dead_start: tuple[int, int]
    dead_end: tuple[int, int]
    loop_secs: int
    api_delay: float
    cooldown_win_secs: int
    cooldown_loss_secs: int

    # Option resolution
    same_day_expiry_ok: bool  # True on expiry day
    use_vix_gate: bool
    use_oi_gate: bool

    # Phase control
    exit_after_theta_kill: bool  # True → run() breaks when idle past theta_kill (Phase 1 handoff)

    # Logging
    log_filename: str


# Expiry Tuesday: gamma moves fast — 1m sniper, early theta kill, tight confluence.
ExpirySniperConfig = SniperConfig(
    name="EXPIRY SNIPER",
    version="v3",
    run_on_weekday=1,  # Tuesday
    skip_weekday=None,
    entry_interval="1",
    lookback_entry=120,  # cover ORB from 9:15 even mid-morning
    lookback_htf=20,
    orb_start=(9, 15),
    orb_end_mm=(9, 29),
    entries_from=(9, 30),
    min_signal=3,
    vol_mult=1.15,
    require_primary=True,
    require_htf_align=True,
    tp_pct=0.50,
    sl_pct=0.18,
    max_trades=4,
    exceptional_score=6,
    daily_target=2000.0,
    daily_max_loss=1500.0,
    max_consecutive_losses=2,
    capital_target_pct=0.22,
    capital_target_floor=600.0,
    capital_target_cap=1500.0,
    trail_activate_pct=0.12,
    trail_be_buffer_pct=0.02,
    trail_from_peak_pct=0.08,
    high_momentum_score=4,
    momentum_refresh_ticks=3,
    min_premium=25.0,
    max_premium=180.0,
    theta_kill=(14, 15),
    dead_start=(12, 0),
    dead_end=(12, 0),
    loop_secs=25,
    api_delay=1.5,
    cooldown_win_secs=90,
    cooldown_loss_secs=180,
    same_day_expiry_ok=True,
    use_vix_gate=True,
    use_oi_gate=True,
    exit_after_theta_kill=True,
    log_filename="expiry_scalper.log",
)


# Non-expiry: slower tape — 5m sniper, stricter filters, next-week ATM.
NonExpirySniperConfig = SniperConfig(
    name="NON-EXPIRY SNIPER",
    version="v2",
    run_on_weekday=None,
    skip_weekday=1,  # never on Tuesday
    entry_interval="5",
    lookback_entry=40,
    lookback_htf=20,
    orb_start=(9, 15),
    orb_end_mm=(9, 44),
    entries_from=(9, 45),
    min_signal=3,
    vol_mult=1.35,
    require_primary=True,
    require_htf_align=True,
    tp_pct=0.35,
    sl_pct=0.15,
    max_trades=3,
    exceptional_score=6,
    daily_target=2000.0,
    daily_max_loss=1200.0,
    max_consecutive_losses=2,
    capital_target_pct=0.18,
    capital_target_floor=500.0,
    capital_target_cap=1200.0,
    trail_activate_pct=0.12,
    trail_be_buffer_pct=0.02,
    trail_from_peak_pct=0.08,
    high_momentum_score=4,
    momentum_refresh_ticks=2,
    min_premium=30.0,
    max_premium=200.0,
    theta_kill=(15, 0),
    dead_start=(12, 0),
    dead_end=(12, 0),
    loop_secs=45,
    api_delay=1.5,
    cooldown_win_secs=120,
    cooldown_loss_secs=240,
    same_day_expiry_ok=False,
    use_vix_gate=True,
    use_oi_gate=True,
    exit_after_theta_kill=False,
    log_filename="non_expiry_scalper.log",
)


# Theta Kill Phase — expiry Tuesday only, 14:15–15:15.
# ATM options at this hour are ₹5–40. Any 50-pt Nifty move = 100–300% gain on gamma.
# Phase 2 engine is spawned by scripts/sniper.py after Phase 1 exits at theta_kill.
# ORB range and daily_pnl are inherited from Phase 1 at runtime.
ThetaKillConfig = SniperConfig(
    name="THETA KILL",
    version="v1",
    run_on_weekday=1,       # Tuesday only
    skip_weekday=None,
    entry_interval="1",
    lookback_entry=120,     # reuse full morning history for ORB reference
    lookback_htf=20,
    orb_start=(9, 15),      # inherited from Phase 1 at runtime; lock call is a no-op if or_set=True
    orb_end_mm=(9, 29),
    entries_from=(14, 15),  # start trading immediately after Phase 1 exits
    min_signal=3,
    vol_mult=1.0,           # volume dries up end-of-day — informational only
    require_primary=True,
    require_htf_align=False,  # HTF EMAs stale by 14:15; don't block on afternoon context
    tp_pct=1.00,            # 100% TP — cheap options either double or die
    sl_pct=0.35,            # wider SL — a ₹10 option swings ₹3 per bar
    max_trades=2,           # lottery territory — 2 shots maximum
    exceptional_score=None,
    daily_target=5000.0,    # overridden at runtime from Phase 1 state (not enforced here)
    daily_max_loss=500.0,   # halt Phase 2 if it loses ₹500 (Phase 1 gains are protected)
    max_consecutive_losses=1,
    capital_target_pct=0.50,   # book half the option cost as profit if momentum weak
    capital_target_floor=200.0,
    capital_target_cap=800.0,
    trail_activate_pct=0.30,   # activate trail only after +30% (give cheap options room)
    trail_be_buffer_pct=0.05,
    trail_from_peak_pct=0.15,
    high_momentum_score=4,
    momentum_refresh_ticks=3,
    min_premium=2.0,        # accept very cheap options
    max_premium=60.0,       # don't buy expensive options in Phase 2
    theta_kill=(15, 15),    # hard exit 15 min before NSE close (avoid settlement auction)
    dead_start=(15, 15),    # no dead zone in Phase 2 — entire window is active
    dead_end=(15, 15),
    loop_secs=20,           # faster loop — moves happen in seconds at expiry end
    api_delay=1.0,
    cooldown_win_secs=60,
    cooldown_loss_secs=120,
    same_day_expiry_ok=True,
    use_vix_gate=False,     # VIX irrelevant at 14:15 — option pricing already reflects it
    use_oi_gate=False,      # OI walls dissolve in the final hour
    exit_after_theta_kill=True,
    log_filename="expiry_scalper.log",  # append to same log as Phase 1
)
