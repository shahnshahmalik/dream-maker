#!/usr/bin/env python3
"""Guardian — crash monitor and auto-restart for dream-maker trading engine.

Responsibilities:
  1. Launch ``python main.py`` as a subprocess.
  2. Monitor heartbeat at state/heartbeat.json (written every loop cycle).
  3. If heartbeat is stale (> 2× poll interval, max 300s), kill and restart.
  4. If process exits non-zero, restart with exponential backoff.
  5. Cap restarts: max 5 in any 10-minute window.
  6. Send Telegram notifications on crash, restart, and recovery.

Usage:
    python guardian.py

Kill with SIGINT or SIGTERM — forwarded to child before shutdown.
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

# ------------------------------------------------------------------ #
# Constants
# ------------------------------------------------------------------ #
HEARTBEAT_FILE = Path("state/heartbeat.json")
PROJECT_DIR = Path(__file__).resolve().parent
STALE_TIMEOUT = 300         # seconds — heartbeat staleness limit
RESTART_WINDOW = 600        # 10-minute rolling window
MAX_RESTARTS_IN_WINDOW = 5  # max restarts before giving up
DELAY_BASE = 2              # exponential backoff base (seconds)
DELAY_MAX = 60              # backoff cap
ENV_FILE = PROJECT_DIR / ".env"
BOT_TOKEN = ""
CHAT_ID = ""

# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #
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
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        httpx.post(
            url,
            json={"chat_id": CHAT_ID, "text": message, "parse_mode": "HTML"},
            timeout=5.0,
        )
    except Exception:
        pass


def _fmt_ts(dt: datetime | None = None) -> str:
    dt = dt or datetime.now(tz=timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M:%S UTC")


# ------------------------------------------------------------------ #
# Heartbeat
# ------------------------------------------------------------------ #
def _read_heartbeat_age() -> float | None:
    """Return age in seconds of the heartbeat, or None if unavailable."""
    try:
        if not HEARTBEAT_FILE.exists():
            return None
        data = json.loads(HEARTBEAT_FILE.read_text(encoding="utf-8"))
        ts = data.get("timestamp", "")
        if not ts:
            return None
        hb_dt = datetime.fromisoformat(ts)
        age = (datetime.now(timezone.utc) - hb_dt).total_seconds()
        return max(0.0, age)
    except (OSError, json.JSONDecodeError, ValueError, KeyError):
        return None


# ------------------------------------------------------------------ #
# Main guardian loop
# ------------------------------------------------------------------ #
def main() -> int:
    _load_telegram_creds()

    restart_times: list[float] = []
    attempt = 0
    child_pid: int | None = None
    child: subprocess.Popen | None = None
    running = True

    def _handle_signal(signum: int, _frame: object) -> None:
        nonlocal running
        print(f"\n[guardian] Received signal {signum}, shutting down...")
        running = False
        if child_pid is not None:
            try:
                os.kill(child_pid, signum)
            except ProcessLookupError:
                pass

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    print(f"[guardian] Watching heartbeat: {HEARTBEAT_FILE}")
    print(f"[guardian] Project dir: {PROJECT_DIR}")
    print(f"[guardian] Telegram: {'enabled' if BOT_TOKEN else 'disabled'}")

    failure_count = 0

    while running:
        # --- Crash-limit check ---
        now = time.time()
        window_start = now - RESTART_WINDOW
        restart_times = [t for t in restart_times if t > window_start]
        if len(restart_times) >= MAX_RESTARTS_IN_WINDOW:
            msg = (
                f"🛑 <b>dream-maker GUARDIAN: HALTED</b>\n"
                f"{MAX_RESTARTS_IN_WINDOW} crashes in the last "
                f"{RESTART_WINDOW // 60} minutes — giving up.\n"
                f"Time: {_fmt_ts()}\n"
                f"Manual intervention required."
            )
            print(f"[guardian] {msg}")
            _send_telegram(msg)
            return 1

        # --- Launch child ---
        attempt += 1
        delay = 0
        if attempt > 1:
            delay = min(DELAY_BASE ** min(attempt - 1, 5), DELAY_MAX)
            print(f"[guardian] Backing off {delay}s before attempt {attempt}...")
            time.sleep(delay)

        # Clear stale heartbeat from previous run
        try:
            HEARTBEAT_FILE.unlink(missing_ok=True)
        except OSError:
            pass

        print(f"[guardian] Launching dream-maker (attempt {attempt})...")
        child = subprocess.Popen(
            [sys.executable, "main.py"],
            cwd=str(PROJECT_DIR),
            env={**os.environ, "GUARDIAN_PID": str(os.getpid())},
            stdout=open("/tmp/dream-maker-engine.log", "a"),
            stderr=subprocess.STDOUT,
        )
        child_pid = child.pid
        print(f"[guardian] Child PID: {child_pid}")

        # --- Monitor loop ---
        while running and child.poll() is None:
            age = _read_heartbeat_age()
            if age is not None and age > STALE_TIMEOUT:
                print(
                    f"[guardian] Heartbeat STALE ({age:.0f}s > {STALE_TIMEOUT}s) "
                    f"— killing child PID {child_pid}"
                )
                _send_telegram(
                    f"💀 <b>dream-maker: STALE heartbeat</b>\n"
                    f"Age: {age:.0f}s (limit: {STALE_TIMEOUT}s)\n"
                    f"Time: {_fmt_ts()}\n"
                    f"Killing PID {child_pid} and restarting..."
                )
                try:
                    child.kill()
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.terminate()
                    child.wait(timeout=5)
                except ProcessLookupError:
                    pass
                failure_count += 1
                break

            time.sleep(10)

        if not running:
            break

        # --- Child exited ---
        exit_code = child.returncode
        child_pid = None
        restart_times.append(time.time())

        if exit_code == 0 or exit_code == -signal.SIGTERM:
            label = "SIGTERM" if exit_code == -signal.SIGTERM else f"code {exit_code}"
            print(f"[guardian] Child exited cleanly ({label}).")
            if running:
                _send_telegram(
                    f"🔄 <b>dream-maker: Restarting after clean exit</b>\n"
                    f"Exit: {label}\nTime: {_fmt_ts()}\nAttempt: {attempt}"
                )
            continue

        # --- Crash ---
        failure_count += 1
        msg = (
            f"🚨 <b>dream-maker: CRASHED</b>\n"
            f"Exit code: {exit_code}\n"
            f"Time: {_fmt_ts()}\n"
            f"Crash #{failure_count} | Restart #{len(restart_times)} in "
            f"last {RESTART_WINDOW // 60}m\n"
            f"Next attempt: {attempt + 1}"
        )
        print(f"[guardian] {msg}")
        _send_telegram(msg)

    # --- Shutdown ---
    if child is not None and child.poll() is None:
        print(f"[guardian] Terminating child PID {child.pid}...")
        child.terminate()
        try:
            child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()

    print("[guardian] Shutdown complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
