"""Autonomous trading agent CLI."""

from __future__ import annotations

import argparse
import sys

from agent.engine import TradingEngine, setup_logging
from config import load_config


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Dream Maker — Autonomous Trading Agent")
    ap.add_argument("--live", action="store_true", help="Disable simulation mode (live orders)")
    ap.add_argument("--scan-only", action="store_true", help="Analysis and plans only, no orders")
    ap.add_argument("--once", action="store_true", help="Run one cycle then exit")
    ap.add_argument(
        "--no-lock",
        action="store_true",
        help="Allow multiple instances (not recommended with a watchdog)",
    )
    return ap.parse_args()


def main() -> int:
    setup_logging()
    args = parse_args()
    cfg = load_config()

    instance_lock = None
    if not args.no_lock:
        from utils.instance_lock import acquire_instance_lock, release_instance_lock

        lock_path = cfg.state_dir / "dream-maker.lock"
        instance_lock = acquire_instance_lock(lock_path)
        if instance_lock is None:
            print(
                "ERROR: another dream-maker instance is already running "
                f"(lock: {lock_path}). Stop it first or use --no-lock for testing.",
                file=sys.stderr,
            )
            return 3

    if args.live:
        if not sys.stdin.isatty():
            print("ERROR: --live requires interactive confirmation", file=sys.stderr)
            return 2
        confirm = input("SIMULATION_MODE will be disabled. Type 'LIVE' to confirm: ")
        if confirm.strip() != "LIVE":
            print("Aborted.")
            return 1
        from dataclasses import replace
        cfg = replace(cfg, simulation_mode=False)

    engine = TradingEngine(cfg, scan_only=args.scan_only)

    import signal
    signal.signal(signal.SIGINT, engine.stop)
    signal.signal(signal.SIGTERM, engine.stop)

    try:
        if args.once:
            engine.bootstrap()
            engine.run_once()
            engine.broker.close()
        else:
            engine.run()
    except Exception as e:
        import logging
        logging.getLogger("dream_maker").exception("Fatal: %s", e)
        return 1
    finally:
        if instance_lock is not None:
            from utils.instance_lock import release_instance_lock

            release_instance_lock()
    return 0


if __name__ == "__main__":
    sys.exit(main())
