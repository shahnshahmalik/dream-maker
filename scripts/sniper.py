"""
Unified Sniper — NIFTY Options Scalper (Expiry + Non-Expiry).

Replaces scripts/expiry_scalper.py and scripts/non_expiry_scalper.py.

Design
------
- Runs 24x7 as a daemon. Sleeps between sessions using market_hours utilities.
- On session open, detects day type and runs the appropriate engine(s):

  Tuesday (expiry):
    Phase 1 — ExpirySniperConfig  (09:30 – 14:15, 1m bars, 4 trades max)
    Phase 2 — ThetaKillConfig     (14:15 – 15:15, 1m bars, 2 trades max)
    ORB range and daily PnL are handed off from Phase 1 to Phase 2.

  Mon / Wed / Thu / Fri (non-expiry):
    SniperEngine(NonExpirySniperConfig)  (09:45 – 15:00, 5m bars, 3 trades max)

- A heartbeat file is written every outer loop tick so the guardian can detect stalls.
- An instance lock prevents duplicate processes.

Usage
-----
  python scripts/sniper.py          # start the 24x7 daemon
  python scripts/sniper.py --once   # run one market session and exit (for testing)
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scalping.config import ExpirySniperConfig, NonExpirySniperConfig, ThetaKillConfig
from scalping.core import SniperEngine, ist_now
from scalping.logfmt import setup_colored_logging
from utils.market_hours import (
    compute_idle_sleep_seconds,
    format_next_open,
    get_market_session,
)

IST = ZoneInfo("Asia/Kolkata")
MARKET_HOURS = "09:15-15:30"
HEARTBEAT_PATH = ROOT / "state" / "sniper_heartbeat.json"
LOCK_PATH = ROOT / "state" / "sniper.lock"

log = logging.getLogger("sniper")
_running = True


# ── Heartbeat ─────────────────────────────────────────────────────────────────

def _write_heartbeat(phase: str, daily_pnl: float = 0.0) -> None:
    try:
        HEARTBEAT_PATH.parent.mkdir(parents=True, exist_ok=True)
        HEARTBEAT_PATH.write_text(
            json.dumps({
                "timestamp": datetime.now(IST).isoformat(),
                "phase": phase,
                "daily_pnl": daily_pnl,
            }),
            encoding="utf-8",
        )
    except OSError:
        pass


# ── Instance lock ─────────────────────────────────────────────────────────────

def _acquire_lock() -> bool:
    try:
        LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
        if LOCK_PATH.exists():
            pid = LOCK_PATH.read_text().strip()
            if pid.isdigit():
                import os
                try:
                    os.kill(int(pid), 0)  # check if process is alive
                    log.error(
                        "Another sniper instance is already running (pid=%s). "
                        "Stop it first or delete %s.",
                        pid, LOCK_PATH,
                    )
                    return False
                except OSError:
                    pass  # stale lock
        import os
        LOCK_PATH.write_text(str(os.getpid()))
        return True
    except OSError as e:
        log.warning("Could not acquire instance lock: %s — proceeding without lock", e)
        return True


def _release_lock() -> None:
    try:
        LOCK_PATH.unlink(missing_ok=True)
    except OSError:
        pass


# ── Signal handling ───────────────────────────────────────────────────────────

def _handle_signal(signum: int, _frame: object) -> None:
    global _running
    log.info("Signal %d received — shutting down after current tick", signum)
    _running = False


# ── Core session runner ───────────────────────────────────────────────────────

def _run_expiry_session() -> float:
    """Run Phase 1 (normal) + Phase 2 (theta kill) on expiry day. Returns final daily PnL."""
    log.info("=" * 60)
    log.info("EXPIRY DAY — Phase 1: ExpirySniperConfig (09:30–14:15)")
    log.info("=" * 60)

    e1 = SniperEngine(ExpirySniperConfig)
    e1.run()

    daily_pnl = e1.state.daily_pnl
    log.info("Phase 1 complete — daily_pnl=Rs%.0f trades=%d", daily_pnl, e1.state.trades)

    # Only enter Phase 2 if we haven't blown past theta kill already
    # (e.g. if the script started late in the day)
    h, m = ist_now().hour, ist_now().minute
    if (h, m) >= (15, 15):
        log.info("Already past 15:15 — skipping Phase 2")
        return daily_pnl

    log.info("=" * 60)
    log.info("EXPIRY DAY — Phase 2: ThetaKillConfig (14:15–15:15)")
    log.info("Phase 2 inherits ORB H=%.2f L=%.2f | daily_pnl=Rs%.0f",
             e1.state.or_high or 0, e1.state.or_low or 0, daily_pnl)
    log.info("=" * 60)

    e2 = SniperEngine(ThetaKillConfig)
    # Hand off state from Phase 1
    e2.state.daily_pnl = daily_pnl
    e2.state.or_high   = e1.state.or_high   # morning ORB is still the reference range
    e2.state.or_low    = e1.state.or_low
    e2.state.or_set    = e1.state.or_set
    e2.run()

    final_pnl = e2.state.daily_pnl
    log.info("Phase 2 complete — daily_pnl=Rs%.0f trades=%d", final_pnl, e2.state.trades)
    return final_pnl


def _run_non_expiry_session() -> float:
    """Run non-expiry session. Returns final daily PnL."""
    log.info("=" * 60)
    log.info("NON-EXPIRY DAY — NonExpirySniperConfig (09:45–15:00)")
    log.info("=" * 60)

    engine = SniperEngine(NonExpirySniperConfig)
    engine.run()
    return engine.state.daily_pnl


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    setup_colored_logging(ROOT / "logs", "sniper.log")

    ap = argparse.ArgumentParser(description="Unified NIFTY Sniper — 24x7 daemon")
    ap.add_argument(
        "--once",
        action="store_true",
        help="Run one market session (or wait for it to open) then exit",
    )
    ap.add_argument(
        "--no-lock",
        action="store_true",
        help="Skip instance lock (for testing / multiple terminals)",
    )
    args = ap.parse_args()

    if not args.no_lock:
        if not _acquire_lock():
            return 3

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    log.info("Sniper daemon started (pid=%d once=%s)", __import__("os").getpid(), args.once)
    session_ran_today: str | None = None  # ISO date string — one session per calendar day

    try:
        while _running:
            now = ist_now()
            today_str = now.strftime("%Y-%m-%d")

            session = get_market_session(MARKET_HOURS)
            _write_heartbeat("idle" if not session.is_open else "watching", 0.0)

            if not session.is_open:
                log.info(
                    "Market %s — next open: %s",
                    session.reason,
                    format_next_open(session),
                )
                # Reset daily tracker when a new calendar day begins
                if session_ran_today and session_ran_today != today_str:
                    session_ran_today = None

                secs = compute_idle_sleep_seconds(
                    session,
                    monitor_interval=30,
                    closed_poll_interval=300,
                )
                time.sleep(secs)
                continue

            # Market is open — run at most one session per calendar day
            if session_ran_today == today_str:
                # Already traded today; idle until close
                time.sleep(60)
                continue

            _write_heartbeat("trading", 0.0)
            is_expiry = now.weekday() == 1  # Tuesday since Sep 2025

            try:
                if is_expiry:
                    final_pnl = _run_expiry_session()
                else:
                    final_pnl = _run_non_expiry_session()
            except Exception as exc:
                log.exception("Session crashed: %s", exc)
                final_pnl = 0.0

            session_ran_today = today_str
            _write_heartbeat("done", final_pnl)
            log.info("Session complete — final daily PnL=Rs%.0f", final_pnl)

            if args.once:
                log.info("--once flag set — exiting")
                break

            # Sleep a bit before the outer loop checks the session state again
            time.sleep(60)

    finally:
        _release_lock()
        log.info("Sniper daemon stopped")

    return 0


if __name__ == "__main__":
    sys.exit(main())
