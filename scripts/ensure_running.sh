#!/usr/bin/env bash
# Safe supervisor helper — starts dream-maker only if not already running.
# Use ONE of: Docker restart policy OR this script in cron — not both blindly.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

WATCHDOG_LOCK="${STATE_DIR:-$ROOT/state}/watchdog.lock"
INSTANCE_LOCK="${STATE_DIR:-$ROOT/state}/dream-maker.lock"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"

mkdir -p "$(dirname "$WATCHDOG_LOCK")"

exec 9>"$WATCHDOG_LOCK"
if ! flock -n 9; then
  echo "watchdog: another ensure_running.sh is already running"
  exit 0
fi

if [ -f "$INSTANCE_LOCK" ]; then
  if flock -n "$INSTANCE_LOCK" true 2>/dev/null; then
    : # lock not held — stale file, safe to start
  else
    pid="$(head -n1 "$INSTANCE_LOCK" 2>/dev/null || true)"
    echo "dream-maker already running (lock held${pid:+, pid $pid})"
    exit 0
  fi
fi

if [ ! -x "$PYTHON" ]; then
  PYTHON="python3"
fi

exec "$PYTHON" "$ROOT/main.py" "$@"
