"""Autonomous trading agent CLI."""

from __future__ import annotations

import argparse
import re
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


# ── Premium affordability helper (used by symbol picker) ──

MARGIN_BUFFER = 1.3  # must match SymbolPicker.MARGIN_BUFFER


def _find_affordable_option(
    picker,
    broker,
    best,          # InstrumentCandidate
    resolved: str,  # initial ATM resolution
    spot: float,
    balance: float,
    candidates: list,
    fallback: str,
    log,
) -> str:
    """Check that the resolved option premium fits the balance.

    If ATM is too expensive, try increasingly OTM strikes on the same
    underlying, then fall back to the next candidate (different symbol).
    """
    # ── Check ATM premium ──
    if _premium_fits(broker, resolved, best.lot_size, balance, log, spot=spot):
        return resolved

    log.info(
        "Option %s premium exceeds balance ₹%.0f — trying OTM strikes...",
        resolved, balance,
    )

    # ── Try OTM strikes on same underlying ──
    if spot > 0 or best.underlying in picker.STOCK_SPOTS:
        effective_spot = spot if spot > 0 else picker.STOCK_SPOTS.get(best.underlying.upper(), 0)
        if effective_spot > 0:
            # Try intelligent strike selector first (CE + PE + multiple expiries)
            from agent.intelligent_strike_selector import find_affordable_strike
            smart_pick = find_affordable_strike(
                broker=broker,
                underlying=best.underlying,
                spot=effective_spot,
                balance=balance,
                preferred_direction="CE",
            )
            if smart_pick:
                log.info("Found affordable strike (intelligent): %s", smart_pick)
                return smart_pick

            # Fall back to simple OTM CE scan
            otm_symbols = picker._resolve_otm_strikes(best.underlying, effective_spot, max_otm_steps=6)
            for otm_sym in otm_symbols:
                if _premium_fits(broker, otm_sym, best.lot_size, balance, log, spot=effective_spot):
                    log.info("Found affordable OTM strike: %s", otm_sym)
                    return otm_sym

    log.warning(
        "No affordable OTM strike for %s — trying next candidate...",
        best.underlying,
    )

    # ── Fall back to next candidate ──
    # Remove this candidate and re-rank
    remaining = [c for c in candidates if c.symbol != best.symbol]
    next_best = picker._best_fit(remaining, balance)
    if next_best:
        spot2 = 0.0
        if next_best.is_option:
            try:
                spot2 = broker.get_index_spot(next_best.underlying)
            except Exception:
                try:
                    q = broker.get_quote(next_best.underlying)
                    spot2 = q.ltp
                except Exception:
                    pass
        next_resolved = picker.resolve(next_best.symbol, spot_price=spot2)
        if next_best.is_option:
            next_resolved = _find_affordable_option(
                picker, broker, next_best, next_resolved, spot2,
                balance, remaining, fallback, log,
            )
        return next_resolved

    log.warning("No affordable candidate found — using fallback %s", fallback)
    return fallback


def _premium_fits(
    broker,
    option_symbol: str,
    lot_size: int,
    balance: float,
    log,
    spot: float = 0.0,
) -> bool:
    """Return True if the option's estimated premium × lot_size fits within balance.

    Dhan's ``get_quote`` returns spot/underlying prices for options, not the
    actual option premium.  We estimate the premium from the strike distance
    and typical ATM pricing instead.
    """
    try:
        # Extract strike from symbol (e.g., "NIFTY26JUN23400CE" → 23400)
        m = re.search(r"(\d{4,5})(CE|PE)", option_symbol)
        if not m:
            log.warning("Cannot parse strike from %s — assuming affordable", option_symbol)
            return True
        strike = int(m.group(1))

        # Need spot price for estimation
        if spot <= 0:
            # Try to extract underlying and fetch spot
            underlying = option_symbol.split(str(strike))[0]
            # Remove date code (e.g., "NIFTY26JUN" → "NIFTY")
            underlying_clean = re.sub(r"\d{2}[A-Z]{3}$", "", underlying)
            try:
                spot = broker.get_index_spot(underlying_clean)
            except Exception:
                try:
                    q = broker.get_quote(underlying_clean)
                    spot = q.ltp
                except Exception:
                    log.warning("Cannot get spot for %s — assuming affordable", underlying_clean)
                    return True

        # Estimate ATM premium as % of spot (empirical for Indian index options)
        underlying_upper = option_symbol[:6].upper()
        if "NIFTY" in underlying_upper and "BANK" not in underlying_upper:
            atm_pct = 0.008   # ~0.8% for NIFTY ATM
        elif "BANKNIFTY" in underlying_upper:
            atm_pct = 0.006   # ~0.6% for BANKNIFTY ATM
        elif "SENSEX" in underlying_upper:
            atm_pct = 0.005   # ~0.5% for SENSEX ATM
        else:
            atm_pct = 0.010   # ~1.0% for stocks

        # Premium decays as we go OTM. Each 1% away from spot → ~15% premium drop
        distance_pct = abs(strike - spot) / spot  # e.g., 0.02 = 2% OTM
        decay = max(0.15, 1.0 - distance_pct * 15)  # 15% drop per 1% distance, floor 0.15
        estimated_premium = spot * atm_pct * decay
        # Floor: at least ₹8 per unit (deep OTM still has some value)
        estimated_premium = max(estimated_premium, 8.0)

        cost = estimated_premium * lot_size
        required = cost * MARGIN_BUFFER

        if balance >= required:
            log.info(
                "Premium est: %s strike=%d spot=%.0f → premium≈₹%.0f × %d lot = ₹%.0f "
                "(need ₹%.0f, have ₹%.0f) ✅",
                option_symbol, strike, spot, estimated_premium, lot_size, cost, required, balance,
            )
            return True
        else:
            log.info(
                "Premium est: %s strike=%d spot=%.0f → premium≈₹%.0f × %d lot = ₹%.0f "
                "> balance ₹%.0f ❌",
                option_symbol, strike, spot, estimated_premium, lot_size, cost, balance,
            )
            return False
    except Exception as e:
        log.warning("Cannot estimate premium for %s: %s — assuming affordable", option_symbol, e)
        return True  # Don't block on unforeseen errors


def main() -> int:
    setup_logging()
    args = parse_args()
    cfg = load_config()

    # --- Instance lock (prevent duplicate runs) ---
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

    # --- Auto symbol picker: select best F&O instrument based on balance ---
    if cfg.auto_symbol_picker:
        import logging as _log
        import os as _os
        _picker_log = _log.getLogger("dream_maker")
        try:
            from agent.symbol_picker import SymbolPicker
            from providers.dhan import DhanProvider

            broker = DhanProvider(cfg)
            funds = broker.get_funds()
            balance = funds.available

            preferred_raw = _os.getenv("PREFERRED_SYMBOLS", "")
            preferred = (
                [p.strip() for p in preferred_raw.split(",") if p.strip()]
                if preferred_raw
                else ["SENSEX", "BANKNIFTY", "NIFTY50IDX"]
            )

            picker = SymbolPicker(
                preferred=preferred,
                fallback=cfg.trading_symbol,
            )
            candidates = picker.build_candidates(balance)
            best = picker._best_fit(candidates, balance)
            if best:
                spot = 0.0
                if best.is_option:
                    # Fetch spot price for ATM option resolution
                    try:
                        spot = broker.get_index_spot(best.underlying)
                    except Exception:
                        # Stock option: try get_quote
                        try:
                            quote = broker.get_quote(best.underlying)
                            spot = quote.ltp
                            # Detect Dhan synthetic fallback (~100) for stocks
                            # that should be >500. Use 0 to trigger hardcoded spots.
                            known_low = best.underlying.upper() in {
                                "DIXON", "KFINTECH", "JUBLFOOD", "HPCL",
                                "INDUSTOWER", "EXIDEIND", "ITC", "TATASTEEL",
                            }
                            if known_low and spot < 500:
                                spot = 0.0
                        except Exception as qe:
                            _picker_log.warning(
                                "Could not fetch spot for %s: %s",
                                best.underlying, qe,
                            )
                resolved = picker.resolve(best.symbol, spot_price=spot)

                # ── Premium affordability check ──
                if best.is_option:
                    resolved = _find_affordable_option(
                        picker, broker, best, resolved, spot,
                        balance, candidates, cfg.trading_symbol,
                        _picker_log,
                    )
            else:
                resolved = cfg.trading_symbol

            if resolved != cfg.trading_symbol:
                _picker_log.info(
                    "Symbol picker: %s → %s (balance ₹%.0f)",
                    cfg.trading_symbol, resolved, balance,
                )
                from dataclasses import replace
                cfg = replace(cfg, trading_symbol=resolved)
            broker.close()
        except Exception as e:
            _picker_log.warning(
                "Symbol picker failed (%s) — using TRADING_SYMBOL=%s",
                e, cfg.trading_symbol,
            )

    # --- Live mode confirmation ---
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
