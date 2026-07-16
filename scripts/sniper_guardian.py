#!/usr/bin/env python3
"""Sniper Guardian — crash monitor and auto-restart for scripts/sniper.py.

Responsibilities
----------------
1. Launch ``python scripts/sniper.py`` as a subprocess.
2. Monitor the heartbeat written to state/sniper_heartbeat.json every outer loop tick.
3. If the heartbeat is stale (> STALE_TIMEOUT seconds), kill and restart the child.
4. If the process exits non-zero, restart with exponential backoff.
5. Cap restarts: max 5 in any 10-minute window, then halt and alert.
6. Send Telegram notifications on crash, restart, and recovery.

Usage
-----
  python scripts/sniper_guardian.py          # normal 24x7 operation
  systemctl start sniper                     # via systemd (see deploy/sniper.service)

Kill with SIGINT or SIGTERM — forwarded to the child before shutdown.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

PROJECT_DIR    = Path(__file__).resolve().parents[1]
HEARTBEAT_FILE = PROJECT_DIR / "state" / "sniper_heartbeat.json"
CHILD_CMD      = [sys.executable, str(PROJECT_DIR / "scripts" / "sniper.py")]
LOG_FILE       = PROJECT_DIR / "logs" / "sniper_guardian.log"
ENV_FILE       = PROJECT_DIR / ".env"

STALE_TIMEOUT          = 300   # seconds — heartbeat staleness limit
RESTART_WINDOW         = 600   # 10-minute rolling window
MAX_RESTARTS_IN_WINDOW = 5     # halt after this many crashes in the window
DELAY_BASE             = 2     # exponential backoff base (seconds)
DELAY_MAX              = 60    # backoff cap

BOT_TOKEN = ""
CHAT_ID   = ""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_telegram_creds() -> None:
    global BOT_TOKEN, CHAT_ID
    if not ENV_FILE.exists():
        return
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if line.startswith("TELEGRAM_BOT_TOKEN="):
            BOT_TOKEN = line.split("=", 1)[1].strip().strip('"').strip("'")
        elif line.startswith("TELEGRAM_CHAT_ID="):
            CHAT_ID = line.split("=", 1)[1].strip().strip('"').strip("'")


def _send_telegram(message: str) -> None:
    if not BOT_TOKEN or not CHAT_ID:
        return
    try:
        httpx.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            json={"chat_id": CHAT_ID, "text": message, "parse_mode": "HTML"},
            timeout=5.0,
        )
    except Exception:
        pass


def _fmt_ts(dt: datetime | None = None) -> str:
    dt = dt or datetime.now(tz=timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M:%S UTC")


def _log(msg: str) -> None:
    """Write to stdout and append to guardian log."""
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    line = f"[{ts}] [guardian] {msg}"
    print(line, flush=True)
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


# ── Heartbeat ─────────────────────────────────────────────────────────────────

def _read_heartbeat_age() -> float | None:
    """Return age in seconds of the sniper heartbeat, or None if unavailable."""
    try:
        if not HEARTBEAT_FILE.exists():
            return None
        data = json.loads(HEARTBEAT_FILE.read_text(encoding="utf-8"))
        ts = data.get("timestamp", "")
        if not ts:
            return None
        hb_dt = datetime.fromisoformat(ts)
        if hb_dt.tzinfo is None:
            hb_dt = hb_dt.replace(tzinfo=timezone.utc)
        age = (datetime.now(timezone.utc) - hb_dt).total_seconds()
        return max(0.0, age)
    except (OSError, json.JSONDecodeError, ValueError, KeyError):
        return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    _load_telegram_creds()

    restart_times: list[float] = []
    attempt      = 0
    child_pid: int | None = None
    child: subprocess.Popen | None = None  # type: ignore[type-arg]
    running      = True

    def _handle_signal(signum: int, _frame: object) -> None:
        nonlocal running
        _log(f"Received signal {signum} — shutting down")
        running = False
        if child_pid is not None:
            try:
                os.kill(child_pid, signum)
            except ProcessLookupError:
                pass

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    _log(f"Sniper Guardian started")
    _log(f"Child command: {' '.join(CHILD_CMD)}")
    _log(f"Heartbeat: {HEARTBEAT_FILE}")
    _log(f"Telegram: {'enabled' if BOT_TOKEN else 'disabled'}")

    while running:
        # ── Crash-limit check ────────────────────────────────────────────────
        now = time.time()
        window_start = now - RESTART_WINDOW
        restart_times = [t for t in restart_times if t > window_start]
        if len(restart_times) >= MAX_RESTARTS_IN_WINDOW:
            msg = (
                f"🛑 <b>SNIPER GUARDIAN: HALTED</b>\n"
                f"{MAX_RESTARTS_IN_WINDOW} crashes in the last "
                f"{RESTART_WINDOW // 60} minutes — giving up.\n"
                f"Time: {_fmt_ts()}\nManual intervention required."
            )
            _log(msg)
            _send_telegram(msg)
            return 1

        # ── Launch child ─────────────────────────────────────────────────────
        attempt += 1
        if attempt > 1:
            delay = min(DELAY_BASE ** min(attempt - 1, 5), DELAY_MAX)
            _log(f"Backing off {delay}s before attempt {attempt}...")
            time.sleep(delay)

        try:
            HEARTBEAT_FILE.unlink(missing_ok=True)
        except OSError:
            pass

        sniper_log = open(PROJECT_DIR / "logs" / "sniper_stdout.log", "a")
        _log(f"Launching sniper (attempt {attempt})...")
        child = subprocess.Popen(
            CHILD_CMD,
            cwd=str(PROJECT_DIR),
            env={**os.environ, "SNIPER_GUARDIAN_PID": str(os.getpid())},
            stdout=sniper_log,
            stderr=subprocess.STDOUT,
        )
        child_pid = child.pid
        _log(f"Child PID: {child_pid}")

        if attempt > 1:
            _send_telegram(
                f"🔄 <b>SNIPER: Restarting</b>\n"
                f"Attempt {attempt} | PID {child_pid}\nTime: {_fmt_ts()}"
            )

        # ── Monitor loop ─────────────────────────────────────────────────────
        while running and child.poll() is None:
            age = _read_heartbeat_age()
            if age is not None and age > STALE_TIMEOUT:
                _log(f"Heartbeat STALE ({age:.0f}s > {STALE_TIMEOUT}s) — killing PID {child_pid}")
                _send_telegram(
                    f"💀 <b>SNIPER: STALE heartbeat</b>\n"
                    f"Age: {age:.0f}s (limit: {STALE_TIMEOUT}s)\n"
                    f"Time: {_fmt_ts()}\nKilling PID {child_pid} and restarting..."
                )
                try:
                    child.kill()
                    child.wait(timeout=10)
                except (subprocess.TimeoutExpired, ProcessLookupError):
                    try:
                        child.terminate()
                        child.wait(timeout=5)
                    except Exception:
                        pass
                break

            time.sleep(10)

        if not running:
            break

        # ── Child exited ─────────────────────────────────────────────────────
        exit_code = child.returncode
        child_pid = None
        restart_times.append(time.time())

        if exit_code in (0, -signal.SIGTERM, None):
            label = {0: "clean exit", None: "None"}.get(exit_code, "SIGTERM")
            _log(f"Child exited cleanly ({label})")
            if running:
                _send_telegram(
                    f"🔄 <b>SNIPER: Restarting after clean exit</b>\n"
                    f"Exit: {label}\nTime: {_fmt_ts()}\nAttempt: {attempt}"
                )
            continue

        # ── Crash ────────────────────────────────────────────────────────────
        msg = (
            f"🚨 <b>SNIPER: CRASHED</b>\n"
            f"Exit code: {exit_code}\n"
            f"Time: {_fmt_ts()}\n"
            f"Crash #{len(restart_times)} in last {RESTART_WINDOW // 60}m\n"
            f"Next attempt: {attempt + 1}"
        )
        _log(msg)
        _send_telegram(msg)

    # ── Shutdown ──────────────────────────────────────────────────────────────
    if child is not None and child.poll() is None:
        _log(f"Terminating child PID {child.pid}...")
        child.terminate()
        try:
            child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()

    _log("Guardian shutdown complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
