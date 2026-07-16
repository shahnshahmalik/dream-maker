"""ICT/SMC scalper configurations — Order Block + Liquidity Sweep + OTE.

Config differences vs SniperConfig
------------------------------------
• kill_zone_open   / kill_zone_close   — ICT kill-zone windows (replace ORB-based entry window)
• ob_lookback      — how many bars to scan for a displacement when building an OB
• ob_sl_buffer_pct — percent buffer beyond OB boundary for SL placement
• ote_fib_low/high — Fibonacci levels defining the OTE zone (default 0.618/0.79)
• sweep_min_strength — minimum sweep signal strength (0.0–1.0) to proceed to OB detection
• ob_max_age_bars  — discard OB if price hasn't retraced within this many bars (stale setup)
• tp_mode          — "swept_level" (target the origin liquidity) or "rr" (fixed R:R)
• tp_rr_multiple   — used when tp_mode == "rr"
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ICTConfig:
    name: str
    version: str

    # Calendar
    run_on_weekday: int | None  # 0=Mon…6=Sun; None = any except skip_weekday
    skip_weekday: int | None

    # Chart timeframe for entry candles
    entry_interval: str  # "1" or "5"
    lookback_entry: int  # bars to fetch for entry-TF
    lookback_htf: int    # bars to fetch for 15m HTF

    # ICT kill zones (two windows: open drive + afternoon)
    kill_zone_open_start: tuple[int, int]   # e.g. (9, 30)
    kill_zone_open_end:   tuple[int, int]   # e.g. (11, 0)
    kill_zone_pm_start:   tuple[int, int]   # e.g. (13, 0)
    kill_zone_pm_end:     tuple[int, int]   # e.g. (14, 15) or (14, 45)

    # Dead zone (no trading)
    dead_start: tuple[int, int]
    dead_end:   tuple[int, int]

    # Theta / EOD
    theta_kill: tuple[int, int]

    # Order Block parameters
    ob_lookback:          int    # displacement scan window
    ob_sl_buffer_pct:     float  # % buffer beyond OB for SL
    ob_max_age_bars:      int    # bars before an un-touched OB is discarded
    ob_min_body_pct:      float  # minimum OB body % of close to filter dojis

    # Sweep gate
    sweep_min_strength: float  # 0.0–1.0 — minimum sweep strength to act on

    # OTE zone
    ote_fib_low:  float  # 61.8 % default
    ote_fib_high: float  # 79.0 % default

    # TP
    tp_mode:       str    # "swept_level" or "rr"
    tp_rr_multiple: float  # used when tp_mode == "rr"

    # Volume gate
    vol_mult: float  # entry-bar volume must be >= vol_mult × trimmed baseline

    # Risk / position
    max_trades:            int
    daily_target:          float
    daily_max_loss:        float
    max_consecutive_losses: int
    cooldown_win_secs:     int
    cooldown_loss_secs:    int

    # Premium bounds (ATM option filter)
    min_premium: float
    max_premium: float

    # Option resolution
    same_day_expiry_ok: bool

    # VIX / OI gates
    use_vix_gate: bool
    use_oi_gate:  bool

    # Trailing SL (percent-based, same as sniper)
    trail_activate_pct:  float
    trail_be_buffer_pct: float
    trail_from_peak_pct: float

    # Session
    loop_secs:   int
    api_delay:   float
    log_filename: str


# ─────────────────────────────────────────────────────────────────────────────
# Expiry ICT: 1-minute chart, Tuesday only
# Gamma is fast — target the open drive kill zone and afternoon reversal zone.
# OB tends to be tighter on 1m; fewer trades, tighter buffers.
# ─────────────────────────────────────────────────────────────────────────────
ExpiryICTConfig = ICTConfig(
    name="EXPIRY ICT SNIPER",
    version="v1",
    run_on_weekday=1,   # Tuesday
    skip_weekday=None,
    entry_interval="1",
    lookback_entry=120,
    lookback_htf=20,
    # Kill zones (IST)
    kill_zone_open_start=(9, 30),
    kill_zone_open_end=(11, 0),
    kill_zone_pm_start=(13, 0),
    kill_zone_pm_end=(14, 15),  # theta kill on expiry day
    dead_start=(11, 0),
    dead_end=(13, 0),
    theta_kill=(14, 15),
    # Order Block
    ob_lookback=8,
    ob_sl_buffer_pct=0.001,     # 0.1 %
    ob_max_age_bars=20,         # discard after 20 × 1m = 20 minutes
    ob_min_body_pct=0.003,
    # Sweep gate
    sweep_min_strength=0.55,
    # OTE
    ote_fib_low=0.618,
    ote_fib_high=0.790,
    # TP
    tp_mode="swept_level",
    tp_rr_multiple=2.0,
    # Volume
    vol_mult=1.15,
    # Risk
    max_trades=3,
    daily_target=2000.0,
    daily_max_loss=1500.0,
    max_consecutive_losses=2,
    cooldown_win_secs=90,
    cooldown_loss_secs=180,
    # Premium
    min_premium=25.0,
    max_premium=180.0,
    # Option
    same_day_expiry_ok=True,
    use_vix_gate=True,
    use_oi_gate=True,
    # Trail
    trail_activate_pct=0.12,
    trail_be_buffer_pct=0.02,
    trail_from_peak_pct=0.08,
    # Session
    loop_secs=25,
    api_delay=1.5,
    log_filename="ict_expiry_scalper.log",
)

# ─────────────────────────────────────────────────────────────────────────────
# Non-Expiry ICT: 5-minute chart, Mon/Wed/Thu/Fri
# Slower tape — displacement moves are larger but take longer to form.
# Wider OB tolerance, more time for retracement, stricter sweep threshold.
# ─────────────────────────────────────────────────────────────────────────────
NonExpiryICTConfig = ICTConfig(
    name="NON-EXPIRY ICT SNIPER",
    version="v1",
    run_on_weekday=None,
    skip_weekday=1,   # never on Tuesday
    entry_interval="5",
    lookback_entry=40,
    lookback_htf=20,
    # Kill zones (IST) — 5m moves are slower so windows are slightly wider
    kill_zone_open_start=(9, 45),
    kill_zone_open_end=(11, 30),
    kill_zone_pm_start=(13, 0),
    kill_zone_pm_end=(14, 45),
    dead_start=(11, 30),
    dead_end=(13, 0),
    theta_kill=(15, 0),
    # Order Block
    ob_lookback=6,
    ob_sl_buffer_pct=0.0015,    # 0.15 % — slightly wider on 5m
    ob_max_age_bars=12,         # 12 × 5m = 60 minutes
    ob_min_body_pct=0.003,
    # Sweep gate
    sweep_min_strength=0.60,    # stricter on non-expiry (noise is lower conviction)
    # OTE
    ote_fib_low=0.618,
    ote_fib_high=0.790,
    # TP
    tp_mode="swept_level",
    tp_rr_multiple=2.0,
    # Volume
    vol_mult=1.35,
    # Risk
    max_trades=2,
    daily_target=2000.0,
    daily_max_loss=1200.0,
    max_consecutive_losses=2,
    cooldown_win_secs=120,
    cooldown_loss_secs=240,
    # Premium
    min_premium=30.0,
    max_premium=200.0,
    # Option
    same_day_expiry_ok=False,
    use_vix_gate=True,
    use_oi_gate=True,
    # Trail
    trail_activate_pct=0.12,
    trail_be_buffer_pct=0.02,
    trail_from_peak_pct=0.08,
    # Session
    loop_secs=45,
    api_delay=1.5,
    log_filename="ict_nonexpiry_scalper.log",
)
