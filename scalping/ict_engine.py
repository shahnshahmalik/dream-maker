"""ICT/SMC scalper engine — Order Block + Liquidity Sweep + OTE entry.

Trading loop (per tick)
-----------------------
1. Fetch candles + spot price.
2. Check session gates (kill zone, dead zone, theta kill, risk halts).
3. If in position: monitor LTP, update trail SL, check TP/SL/EOD exit.
4. If not in position:
   a. Detect liquidity sweep via analysis/liquidity_sweep.py.
   b. If sweep strength >= threshold → detect Order Block via scalping/ict_signals.py.
   c. Cache the OB. Each subsequent tick checks if price has entered the OTE zone.
   d. OTE touch → enter ATM option, set SL at OB boundary, TP at sweep origin.
5. Govern with daily max-loss, consecutive-loss halts, per-trade cooldowns.

Reused from scalping/core.py (unchanged):
  get_candles, get_spot, get_option_ltp, place_order, resolve_option,
  confirm_fill_price, ema, vwap, compute_trail_sl, ist_now, hm,
  past_market_close, expiry_to_nse_str, _oi_vix_ok (inlined below)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from models.orders import OHLCV
from scalping.ict_config import ICTConfig
from scalping.ict_signals import OrderBlock, detect_order_block
from scalping.logfmt import C, fmt_pct, fmt_rupees, paint, setup_colored_logging

# Reuse all API helpers and indicator functions from the shared sniper core
from scalping.core import (
    IST,
    compute_trail_sl as _compute_trail_sl_sniper,
    confirm_fill_price,
    ema,
    expiry_to_nse_str,
    fetch_open_nifty_long,
    get_balance,
    get_candles,
    get_option_ltp,
    get_spot,
    place_order,
    resolve_option,
    vwap,
)

# analysis/liquidity_sweep.py uses OHLCV dataclass
from analysis.liquidity_sweep import (
    LiquiditySweepSignal,
    detect_liquidity_sweep,
    find_all_liquidity_levels,
)

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs"

log = logging.getLogger("ict_sniper")


# ── ICT-specific trail helper (wraps core's sniper trail with ICT config) ─────

def _compute_trail_sl(
    entry: float,
    peak: float,
    current_sl: float,
    cfg: ICTConfig,
) -> float | None:
    """Percent-based trail reusing core logic via duck-typed config."""

    class _Proxy:
        trail_activate_pct  = cfg.trail_activate_pct
        trail_be_buffer_pct = cfg.trail_be_buffer_pct
        trail_from_peak_pct = cfg.trail_from_peak_pct

    return _compute_trail_sl_sniper(entry, peak, current_sl, _Proxy())  # type: ignore[arg-type]


# ── Time helpers ──────────────────────────────────────────────────────────────

def _hm() -> tuple[int, int]:
    n = datetime.now(IST)
    return n.hour, n.minute


def _in_kill_zone(cfg: ICTConfig) -> bool:
    """Return True when current time is inside one of the two ICT kill zones."""
    h, m = _hm()
    cur = (h, m)
    in_open = cfg.kill_zone_open_start <= cur < cfg.kill_zone_open_end
    in_pm   = cfg.kill_zone_pm_start   <= cur < cfg.kill_zone_pm_end
    return in_open or in_pm


def _in_dead_zone(cfg: ICTConfig) -> bool:
    return cfg.dead_start <= _hm() < cfg.dead_end


def _past_theta(cfg: ICTConfig) -> bool:
    return _hm() >= cfg.theta_kill


def _past_close() -> bool:
    return _hm() >= (15, 30)


def _candles_to_ohlcv(candles: list[dict]) -> list[OHLCV]:
    """Convert the raw candle dicts from get_candles() into OHLCV dataclasses."""
    IST_tz = IST
    today_str = date.today().strftime("%Y-%m-%d")
    result = []
    for c in candles:
        ts_str = f"{today_str}T{c['time']}:00"
        try:
            ts = datetime.fromisoformat(ts_str).replace(tzinfo=IST_tz)
        except ValueError:
            ts = datetime.now(IST_tz)
        result.append(OHLCV(
            timestamp=ts,
            open=float(c["open"]),
            high=float(c["high"]),
            low=float(c["low"]),
            close=float(c["close"]),
            volume=int(c["volume"]),
        ))
    return result


# ── Engine state ──────────────────────────────────────────────────────────────

@dataclass
class ICTEngineState:
    trades: int = 0

    # Active position
    active: bool = False
    symbol: str | None = None
    sid: int | None = None
    entry: float | None = None
    tp: float | None = None
    sl: float | None = None
    qty: int | None = None
    peak_ltp: float | None = None
    direction: str | None = None   # "CE" or "PE"

    # Cached order block waiting for OTE touch
    pending_ob: OrderBlock | None = None
    pending_ob_direction: str | None = None   # "CE" or "PE"
    pending_ob_age: int = 0   # ticks since OB was cached

    # Daily risk
    daily_pnl: float = 0.0
    consecutive_losses: int = 0
    cooldown_until: float = 0.0
    halted: bool = False
    halt_reason: str = ""


# ── ICTEngine ─────────────────────────────────────────────────────────────────

class ICTEngine:
    """Main loop for ICT/SMC scalper (expiry and non-expiry)."""

    def __init__(self, cfg: ICTConfig) -> None:
        self.cfg = cfg
        self.state = ICTEngineState()
        setup_colored_logging(LOG_DIR, cfg.log_filename)

    # ── Calendar ──────────────────────────────────────────────────────────────

    def calendar_allows(self) -> bool:
        today = datetime.now(IST)
        wd = today.weekday()
        if self.cfg.run_on_weekday is not None and wd != self.cfg.run_on_weekday:
            log.info(
                "Not scheduled day (today=%s). %s runs on weekday=%s only. Exiting.",
                today.strftime("%A"), self.cfg.name, self.cfg.run_on_weekday,
            )
            return False
        if self.cfg.skip_weekday is not None and wd == self.cfg.skip_weekday:
            log.info(
                "Skip day (today=%s). %s does not run on weekday=%s. Exiting.",
                today.strftime("%A"), self.cfg.name, self.cfg.skip_weekday,
            )
            return False
        return True

    # ── Risk halts ────────────────────────────────────────────────────────────

    def _risk_halted(self) -> bool:
        if self.state.halted:
            return True
        s = self.state
        if s.daily_pnl <= -self.cfg.daily_max_loss:
            s.halted, s.halt_reason = True, f"daily_max_loss(Rs{s.daily_pnl:.0f})"
            log.warning("HALT - %s", s.halt_reason)
            return True
        if s.consecutive_losses >= self.cfg.max_consecutive_losses:
            s.halted, s.halt_reason = True, f"consecutive_losses({s.consecutive_losses})"
            log.warning("HALT - %s", s.halt_reason)
            return True
        if s.daily_pnl >= self.cfg.daily_target:
            s.halted, s.halt_reason = True, f"daily_target(Rs{s.daily_pnl:.0f})"
            log.info("HALT - %s", s.halt_reason)
            return True
        return False

    # ── Close helpers ─────────────────────────────────────────────────────────

    def _register_close(self, pnl: float, *, entry: float, exit_px: float) -> None:
        self.state.daily_pnl += pnl
        self.state.active = False
        self.state.pending_ob = None
        self.state.pending_ob_age = 0
        if pnl < 0:
            self.state.consecutive_losses += 1
            self.state.cooldown_until = time.time() + self.cfg.cooldown_loss_secs
        else:
            self.state.consecutive_losses = 0
            self.state.cooldown_until = time.time() + self.cfg.cooldown_win_secs
        log.info(
            "CLOSED entry=%.2f exit=%.2f %s | daily %s | streak_loss=%d",
            entry, exit_px,
            fmt_rupees(pnl, bold=True),
            fmt_rupees(self.state.daily_pnl, bold=True),
            self.state.consecutive_losses,
        )

    def _exit_position(self, tag: str, ltp: float) -> bool:
        assert self.state.sid and self.state.qty and self.state.entry is not None
        tag_u = tag.upper()
        if "TP" in tag_u or "TRAIL" in tag_u:
            painted = paint(f"EXIT {tag}", C.BOLD, C.GREEN)
        elif "SL" in tag_u:
            painted = paint(f"EXIT {tag}", C.BOLD, C.RED)
        else:
            painted = paint(f"EXIT {tag}", C.BOLD, C.YELLOW)
        log.info("%s @ Rs%.2f", painted, ltp)
        oid = place_order(self.state.sid, self.state.qty, "SELL")
        if not oid:
            return False
        fill = confirm_fill_price(oid, self.state.sid, ltp)
        pnl  = (fill - self.state.entry) * self.state.qty
        self._register_close(pnl, entry=self.state.entry, exit_px=fill)
        return True

    # ── Position monitoring ────────────────────────────────────────────────────

    def _monitor(self) -> None:
        assert self.state.sid and self.state.entry is not None
        ltp = get_option_ltp(self.state.sid)
        if ltp <= 0:
            log.warning("LTP=0 — retry next tick")
            return

        if self.state.peak_ltp is None or ltp > self.state.peak_ltp:
            self.state.peak_ltp = ltp

        # Percent-based trail SL
        new_sl = _compute_trail_sl(
            self.state.entry,
            self.state.peak_ltp,
            self.state.sl or 0.0,
            self.cfg,
        )
        if new_sl is not None and self.state.sl is not None:
            log.info(
                "%s %.2f -> %s  (peak=%.2f %s)",
                paint("TRAIL SL", C.GREEN),
                self.state.sl,
                paint(f"{new_sl:.2f}", C.BOLD, C.GREEN),
                self.state.peak_ltp,
                fmt_pct((self.state.peak_ltp - self.state.entry) / self.state.entry),
            )
            self.state.sl = new_sl

        pnl_pct = (ltp - self.state.entry) / self.state.entry
        pnl_rs  = (ltp - self.state.entry) * (self.state.qty or 0)
        log.info(
            "MONITOR %s  LTP=%.2f  %s (%s)  TP=%.2f  SL=%.2f  peak=%.2f",
            self.state.symbol, ltp,
            fmt_pct(pnl_pct, bold=True),
            fmt_rupees(pnl_rs, bold=True),
            self.state.tp, self.state.sl, self.state.peak_ltp,
        )

        hit_tp = self.state.tp is not None and ltp >= self.state.tp
        hit_sl = self.state.sl is not None and ltp <= self.state.sl
        force  = _past_theta(self.cfg) or _past_close()

        if force:
            self._exit_position("EOD/Theta", ltp)
        elif hit_tp:
            self._exit_position("TP", ltp)
        elif hit_sl:
            be_level = self.state.entry * (1 + self.cfg.trail_be_buffer_pct)
            tag = "TRAIL" if (self.state.sl and self.state.sl >= be_level - 0.01) else "SL"
            self._exit_position(tag, ltp)

    # ── OI/VIX gate (same logic as SniperEngine) ──────────────────────────────

    def _oi_vix_ok(self, direction: str, spot: float, expiry_ymd: str) -> bool:
        if self.cfg.use_vix_gate:
            try:
                from utils.market_data import get_india_vix, vix_allows_entry
                vix = get_india_vix()
                ok, reason = vix_allows_entry(vix)
                if not ok:
                    log.warning("VIX GATE: %s", reason)
                    return False
                log.info("VIX: %s", reason)
            except Exception as exc:
                log.warning("VIX check failed (allow): %s", exc)

        if not self.cfg.use_oi_gate:
            return True
        try:
            from utils.market_data import get_option_chain_context, max_pain_bias
            oc = get_option_chain_context("NIFTY", spot, expiry_to_nse_str(expiry_ymd))
            if oc is None:
                return True
            mp = max_pain_bias(spot, oc.max_pain)
            log.info(
                "OI: max_pain=%.0f CE_wall=%.0f PE_wall=%.0f PCR=%.2f bias=%s mp=%s",
                oc.max_pain, oc.ce_wall, oc.pe_wall, oc.pcr, oc.bias, mp,
            )
            if direction == "CE" and spot >= oc.ce_wall - 25:
                log.warning("OI WALL: spot near CE_wall=%.0f — skip CE", oc.ce_wall)
                return False
            if direction == "PE" and spot <= oc.pe_wall + 25:
                log.warning("OI WALL: spot near PE_wall=%.0f — skip PE", oc.pe_wall)
                return False
        except Exception as exc:
            log.warning("OI context failed (non-fatal): %s", exc)
        return True

    # ── Sweep + OB detection ──────────────────────────────────────────────────

    def _detect_setup(self, candles: list[dict]) -> tuple[str | None, OrderBlock | None]:
        """Run sweep detection then OB detection. Return (direction, ob) or (None, None)."""
        if len(candles) < 15:
            return None, None

        ohlcv_list = _candles_to_ohlcv(candles)
        all_levels = find_all_liquidity_levels(ohlcv_list, [])

        if not all_levels:
            return None, None

        import pandas as pd
        ltf_df = pd.DataFrame({
            "open":   [c.open   for c in ohlcv_list],
            "high":   [c.high   for c in ohlcv_list],
            "low":    [c.low    for c in ohlcv_list],
            "close":  [c.close  for c in ohlcv_list],
            "volume": [c.volume for c in ohlcv_list],
        })

        sweep: LiquiditySweepSignal | None = detect_liquidity_sweep(ltf_df, all_levels)
        if sweep is None:
            return None, None

        if sweep.strength < self.cfg.sweep_min_strength:
            log.info(
                "SWEEP detected but strength=%.2f < threshold=%.2f — skip",
                sweep.strength, self.cfg.sweep_min_strength,
            )
            return None, None

        direction = "CE" if sweep.direction == "LONG" else "PE"
        log.info(
            "%s %s (strength=%.2f) — scanning for OB",
            paint("SWEEP", C.BOLD, C.YELLOW),
            sweep.description[:80],
            sweep.strength,
        )

        ob = detect_order_block(
            candles,
            sweep.direction,  # "LONG" or "SHORT"
            displacement_lookback=self.cfg.ob_lookback,
            sl_buffer_pct=self.cfg.ob_sl_buffer_pct,
            min_ob_body_pct=self.cfg.ob_min_body_pct,
            sweep_strength=sweep.strength,
        )
        if ob is None:
            log.info("No valid OB found after sweep — skip")
            return None, None

        log.info(
            "%s %s  OB=[%.2f–%.2f]  OTE=[%.2f–%.2f]  SL=%.2f  TP=%.2f  strength=%.2f",
            paint("OB FOUND", C.BOLD, C.CYAN),
            direction,
            ob.ob_low, ob.ob_high,
            ob.ote_low, ob.ote_high,
            ob.sl_price, ob.tp_price,
            ob.strength,
        )
        return direction, ob

    # ── OTE entry check ───────────────────────────────────────────────────────

    def _try_ote_entry(self, spot: float, ltp: float) -> bool:
        """Check if spot has retraced into the pending OB's OTE zone and enter."""
        ob = self.state.pending_ob
        if ob is None or not ob.fresh:
            return False

        # Age out stale OBs
        self.state.pending_ob_age += 1
        if self.state.pending_ob_age > self.cfg.ob_max_age_bars:
            log.info("OB expired after %d bars — discarding", self.state.pending_ob_age)
            self.state.pending_ob = None
            return False

        # Mitigation check: if price has closed through the OB, invalidate it
        ob.check_mitigation(spot)
        if not ob.fresh:
            log.info("OB mitigated (price closed through it) — discarding")
            self.state.pending_ob = None
            return False

        # OTE touch: underlying price (spot) in OTE zone
        if not ob.price_in_ote(spot):
            return False

        # Validate premium is within acceptable range
        if not (self.cfg.min_premium <= ltp <= self.cfg.max_premium):
            log.warning(
                "OTE touched but premium %.2f outside [%.0f–%.0f] — skip",
                ltp, self.cfg.min_premium, self.cfg.max_premium,
            )
            self.state.pending_ob = None
            return False

        return True

    # ── Main entry logic ──────────────────────────────────────────────────────

    def _try_entry(self, candles: list[dict], spot: float) -> None:
        if self._risk_halted():
            return
        if _in_dead_zone(self.cfg):
            log.info("DEAD ZONE — no new setups")
            return
        if not _in_kill_zone(self.cfg):
            log.info("Outside ICT kill zones — no new setups")
            return
        if time.time() < self.state.cooldown_until:
            log.info("Cooldown %.0fs remaining", self.state.cooldown_until - time.time())
            return
        if self.state.trades >= self.cfg.max_trades:
            log.info("Max trades %d/%d", self.state.trades, self.cfg.max_trades)
            return

        # If we already have a pending OB, check for OTE touch before scanning for a new one
        if self.state.pending_ob is not None:
            result = resolve_option(spot, self.state.pending_ob_direction or "CE", same_day_ok=self.cfg.same_day_expiry_ok)
            if result is None:
                log.warning("Cannot resolve ATM option for pending OB")
                return
            sym, sid, lot, expiry_ymd = result
            time.sleep(self.cfg.api_delay)
            ltp = get_option_ltp(sid)
            if ltp <= 0:
                log.warning("LTP=0 for %s — skip OTE check", sym)
                return
            if self._try_ote_entry(spot, ltp):
                self._enter(sym, sid, lot, expiry_ymd, ltp, self.state.pending_ob, self.state.pending_ob_direction or "CE")
            return

        # No pending OB — run sweep + OB detection
        direction, ob = self._detect_setup(candles)
        if direction is None or ob is None:
            return

        # Resolve option before checking OI/VIX gates
        result = resolve_option(spot, direction, same_day_ok=self.cfg.same_day_expiry_ok)
        if result is None:
            log.warning("Cannot resolve ATM %s — skip", direction)
            return
        sym, sid, lot, expiry_ymd = result
        time.sleep(self.cfg.api_delay)
        ltp = get_option_ltp(sid)
        if ltp <= 0:
            log.warning("LTP=0 for %s — skip", sym)
            return

        # OI/VIX gate
        if not self._oi_vix_ok(direction, spot, expiry_ymd):
            return

        # If price is already in OTE on the same tick, enter immediately
        if self._try_ote_entry(spot, ltp):
            self._enter(sym, sid, lot, expiry_ymd, ltp, ob, direction)
        else:
            # Cache OB and wait for retracement
            self.state.pending_ob = ob
            self.state.pending_ob_direction = direction
            self.state.pending_ob_age = 0
            log.info(
                "OB CACHED — waiting for OTE retracement into [%.2f–%.2f] (spot=%.2f)",
                ob.ote_low, ob.ote_high, spot,
            )

    def _enter(
        self,
        sym: str,
        sid: int,
        lot: int,
        expiry_ymd: str,
        ltp: float,
        ob: OrderBlock,
        direction: str,
    ) -> None:
        balance = get_balance()
        cost = ltp * lot * 1.2
        if cost > balance:
            log.warning("Cannot afford %s Rs%.0f > bal Rs%.0f", sym, cost, balance)
            self.state.pending_ob = None
            return

        log.info(
            "%s %s @ Rs%.2f  lot=%d  cost≈Rs%.0f  bal=Rs%.0f",
            paint("ENTER BUY", C.BOLD, C.GREEN),
            sym, ltp, lot, ltp * lot, balance,
        )
        oid = place_order(sid, lot, "BUY")
        if oid is None:
            return
        fill = confirm_fill_price(oid, sid, ltp)

        # TP from OB: sweep level origin (or 2R fallback if mode == "rr")
        if self.cfg.tp_mode == "rr":
            sl_dist = abs(fill - ob.sl_price)
            tp = round(fill + sl_dist * self.cfg.tp_rr_multiple, 2)
        else:
            tp = ob.tp_price  # swept liquidity origin

        self.state.active = True
        self.state.symbol = sym
        self.state.sid = sid
        self.state.entry = fill
        self.state.tp = tp
        self.state.sl = ob.sl_price
        self.state.qty = lot
        self.state.peak_ltp = fill
        self.state.direction = direction
        self.state.trades += 1
        self.state.pending_ob = None
        self.state.pending_ob_age = 0

        log.info(
            "ACTIVE %s  entry=%.2f  TP=%.2f  SL=%.2f  trade=%d/%d  OTE=[%.2f–%.2f]",
            sym, fill, tp, ob.sl_price,
            self.state.trades, self.cfg.max_trades,
            ob.ote_low, ob.ote_high,
        )

    # ── Main run loop ─────────────────────────────────────────────────────────

    def run(self) -> None:
        cfg = self.cfg
        log.info(
            "=== %s %s starting ===",
            paint(cfg.name, C.BOLD, C.CYAN),
            cfg.version,
        )

        if not self.calendar_allows():
            return

        # Try to restore a position that survived a restart
        pos = fetch_open_nifty_long()
        if pos:
            entry = pos["entry"]
            self.state.active = True
            self.state.symbol = pos["symbol"]
            self.state.sid    = pos["sid"]
            self.state.qty    = pos["qty"]
            self.state.entry  = entry
            # Reconstruct SL/TP from trail config on restore (conservative: hard SL only)
            self.state.sl = round(entry * (1 - cfg.trail_activate_pct), 2)
            self.state.tp = round(entry * (1 + cfg.ote_fib_low * 2), 2)
            self.state.peak_ltp = entry
            self.state.trades = max(1, self.state.trades)
            log.info(
                "RESTORED %s qty=%d entry=%.2f SL=%.2f TP=%.2f",
                pos["symbol"], pos["qty"], entry, self.state.sl, self.state.tp,
            )

        while True:
            try:
                now = datetime.now(IST)

                if _past_close():
                    log.info("Market closed — final PnL=%s trades=%d",
                             fmt_rupees(self.state.daily_pnl, bold=True),
                             self.state.trades)
                    break

                if _past_theta(cfg) and not self.state.active:
                    log.info("Theta kill — no new entries. PnL=%s",
                             fmt_rupees(self.state.daily_pnl, bold=True))
                    time.sleep(60)
                    continue

                log.info(
                    "── %s | trade=%d/%d | active=%s | pnl=%s ──",
                    now.strftime("%H:%M:%S"),
                    self.state.trades, cfg.max_trades,
                    self.state.active,
                    fmt_rupees(self.state.daily_pnl, bold=True),
                )

                if self.state.active:
                    self._monitor()
                    time.sleep(cfg.loop_secs)
                    continue

                # Fetch candles + spot for setup detection
                candles = get_candles(cfg.entry_interval, cfg.lookback_entry)
                time.sleep(cfg.api_delay)
                spot = get_spot()

                if spot <= 0 or len(candles) < 10:
                    log.warning("Bad data spot=%.2f bars=%d", spot, len(candles))
                    time.sleep(cfg.loop_secs)
                    continue

                self._try_entry(candles, spot)

            except KeyboardInterrupt:
                log.info("Interrupted")
                break
            except Exception as exc:
                log.exception("Tick error: %s", exc)

            time.sleep(cfg.loop_secs)

        log.info(
            "=== DONE  PnL=%s  trades=%d ===",
            fmt_rupees(self.state.daily_pnl, bold=True),
            self.state.trades,
        )
