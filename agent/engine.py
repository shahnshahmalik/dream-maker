"""Trading engine — startup sequence and main loop."""

from __future__ import annotations

import logging
import time

from agent.entry_watcher import EntryWatcher
from agent.executor import TradeExecutor
from agent.monitor import PositionMonitor
from agent.planner import TradePlanner
from agent.trailing_service import TrailingService
from analysis.pipeline import AnalysisPipeline
from analysis.scanner import WatchlistScanner
from audit.state_store import StateStore
from config import Config
from llm.factory import get_llm
from audit.trade_logger import TradeLogger
from notifications.service import NotificationService
from models.trade_plan import EntryType, PlanStatus, TradeDirection, TradePlan
from providers.factory import get_broker
from risk.limits import LimitsGuard
from risk.manager import RiskManager
from risk.balance_manager import BalanceManager
from risk.session_tracker import SessionTracker, SessionState
from utils.market_hours import (
    MarketSession,
    compute_idle_sleep_seconds,
    format_next_open,
    get_market_session,
    is_trade_window_ending,
)
from utils.symbols import normalize_symbol

log = logging.getLogger("dream_maker.engine")


def _is_api_error(exc: BaseException) -> bool:
    try:
        import httpx
    except ImportError:
        return False
    return isinstance(exc, (httpx.HTTPError, httpx.TimeoutException))


class TradingEngine:
    def __init__(self, cfg: Config, *, scan_only: bool = False):
        self.cfg = cfg
        self.scan_only = scan_only
        self._stop = False
        self._loop_count = 0
        self._last_session_reason: str | None = None

        self.trade_logger = TradeLogger(cfg.trade_log_path, notifier=NotificationService(cfg))
        self._notifier = self.trade_logger.notifier
        self.state_store = StateStore(cfg.state_dir, cfg.trade_log_path)
        self.broker = get_broker(cfg)
        self.llm = get_llm(cfg)
        self.risk = RiskManager(cfg.risk_pct_per_trade, cfg.min_rr_ratio)
        self.balance = BalanceManager(
            min_sell_balance=cfg.balance_min_sell,
            sell_buffer=cfg.balance_sell_buffer,
        )
        self.session = SessionTracker(cfg)
        self.limits = LimitsGuard(
            cfg.max_open_trades,
            cfg.daily_loss_limit,
            cfg.event_blackout_minutes,
            cfg.scheduled_events,
        )
        self.pipeline = AnalysisPipeline(self.broker, cfg, self.trade_logger, self.risk)
        self.scanner = WatchlistScanner(self.pipeline, cfg, self.broker, self.session, self._notifier)
        self.planner = TradePlanner(self.llm, cfg, self.trade_logger)
        self.entry_watcher = EntryWatcher(self.broker, cfg)
        self.executor = TradeExecutor(
            self.broker, cfg, self.trade_logger, self.limits, self.balance,
            session_tracker=self.session,
        )
        self.trailing = TrailingService(cfg, self.executor, self.trade_logger)
        self.monitor = PositionMonitor(
            self.broker, self.llm, self.executor, cfg, self.trade_logger, self.trailing,
        )
        self.plans: list[TradePlan] = []
        self._entry_drift_cycles: dict[str, int] = {}  # plan_id → consecutive drift cycles
        self._symbol_cooldown_until: dict[str, float] = {}  # symbol → timestamp when cooldown ends
        self._churn_cooldown_seconds: int = 300  # 5 min cooldown after monitor-triggered close
        self._consecutive_auth_failures: int = 0  # counter for DH-901 / 401 errors
        self._max_auth_failures: int = 3  # halt engine after this many consecutive auth failures

    def _recover_state(self) -> None:
        recovered = self.state_store.load_plans()
        if recovered:
            # Filter out stale plans from different trading symbols that could block the scanner
            compatible_plans = []
            for plan in recovered:
                # Check if plan symbol matches current trading symbol or its underlying
                normalized_trading = normalize_symbol(self.cfg.trading_symbol)
                plan_symbol = normalize_symbol(plan.symbol) if plan.symbol else ""
                
                # For option contracts, extract underlying (e.g., NIFTY26JUN23250CE -> NIFTY) 
                import re
                plan_underlying = plan_symbol
                if plan_symbol.upper().endswith("CE") or plan_symbol.upper().endswith("PE"):
                    base = plan_symbol.upper().replace(" ", "")
                    base = re.sub(r"\d{2}[A-Z]{3}.*", "", base)  # strip date+strike
                    plan_underlying = base
                
                # Check compatibility
                trading_base = normalized_trading.replace("50IDX", "").replace("IDX", "")[:5]
                if (plan_symbol == normalized_trading or 
                    plan_underlying.startswith(trading_base) or
                    plan_symbol.startswith(trading_base)):
                    compatible_plans.append(plan)
                else:
                    log.warning(
                        "Discarding stale plan for %s (current symbol: %s)",
                        plan.symbol, self.cfg.trading_symbol
                    )
            
            if compatible_plans:
                log.info("Recovered %s compatible plan(s) from state/active_plans.json", len(compatible_plans))
                self.plans.extend(compatible_plans)
            elif recovered:
                log.info("No compatible plans recovered (discarded %s stale plans)", len(recovered))

        log_plan = self.state_store.load_from_trade_log(self.cfg.trading_symbol)
        if log_plan and log_plan.plan_id not in {p.plan_id for p in self.plans}:
            log.info("Recovered active trade from trade_log: %s", log_plan.plan_id)
            self.plans.append(log_plan)
            self.trade_logger.log(
                "MONITOR", log_plan.symbol, self.broker.name,
                "Crash recovery — resumed active plan from trade log",
                {"plan_id": log_plan.plan_id, "plan": log_plan.to_dict()},
            )

    def bootstrap(self) -> None:
        log.info(
            "Booting dream-maker: broker=%s llm=%s simulation=%s trading_symbol=%s (F&O only)",
            self.broker.name, self.llm.name, self.cfg.simulation_mode, self.cfg.trading_symbol,
        )
        if self.state_store.is_stale():
            log.warning("Previous heartbeat stale or missing — possible crash detected")

        # ── 1. Get broker truth FIRST ──────────────────────────────
        funds = self.broker.get_funds()
        log.info("Funds: available=%.2f total=%.2f %s", funds.available, funds.total, funds.currency)
        self.session.set_initial_capital(funds.available)
        self.trade_logger.log(
            "MONITOR", "SYSTEM", self.broker.name,
            "Startup connectivity OK",
            {"available": funds.available, "total": funds.total},
        )

        broker_positions = self.broker.get_positions()
        broker_active_symbols: set[str] = set()
        for pos in broker_positions:
            sym = normalize_symbol(pos.symbol) if pos.symbol else ""
            # Also track the Dhan-format symbol for fuzzy matching
            broker_active_symbols.add(sym)
            if pos.symbol:
                broker_active_symbols.add(pos.symbol)

        # ── 2. Recover local state ─────────────────────────────────
        self._recover_state()

        # ── 3. Cross-validate: phantom plans → invalidate ──────────
        for plan in list(self.plans):
            if plan.status not in {PlanStatus.ACTIVE, PlanStatus.WAITING_ENTRY}:
                continue
            plan_sym = normalize_symbol(plan.symbol) if plan.symbol else ""
            # Check if any broker position matches (exact or fuzzy underlying match)
            is_real = False
            for broker_sym in broker_active_symbols:
                if plan_sym == broker_sym:
                    is_real = True
                    break
                # Fuzzy match: NIFTY26JUN23400CE vs NIFTY-Jun2026-23400-CE
                if plan_sym.upper().replace(" ", "") == broker_sym.upper().replace(" ", "").replace("-", ""):
                    is_real = True
                    break
                # Match by underlying + strike: extract "NIFTY" + "23400"
                import re
                plan_strike = re.search(r"(\d{5})", plan_sym) if plan_sym else None
                broker_strike = re.search(r"(\d{5})", broker_sym)
                plan_base = re.sub(r"\d{2}[A-Z]{3}.*", "", (plan_sym or "").upper())
                broker_base = re.sub(r"\d{2}[A-Z]{3}.*", "", broker_sym.upper())
                if plan_strike and broker_strike and plan_base and broker_base:
                    if plan_strike.group(1) == broker_strike.group(1) and plan_base in broker_base:
                        is_real = True
                        break
            if not is_real:
                plan.status = PlanStatus.INVALIDATED
                plan.rationale = "stale — position already closed on broker"
                log.warning(
                    "Invalidated phantom plan %s (status was %s) — no matching broker position",
                    plan.symbol, plan.status.name if hasattr(plan.status, 'name') else plan.status,
                )

        # ── 4. Reconstruct broker positions not in plans ───────────
        for pos in broker_positions:
            pos_sym = normalize_symbol(pos.symbol) if pos.symbol else ""
            if pos_sym != self.cfg.trading_symbol and not pos_sym.startswith(
                normalize_symbol(self.cfg.trading_symbol).replace("50IDX", "").replace("IDX", "")[:5]
            ):
                log.info(
                    "Ignoring position %s — not TRADING_SYMBOL (%s)",
                    pos.symbol, self.cfg.trading_symbol,
                )
                continue
            # Check if we already track this position
            already_tracked = False
            for p in self.plans:
                if p.status != PlanStatus.ACTIVE:
                    continue
                p_sym = normalize_symbol(p.symbol) if p.symbol else ""
                if p_sym == pos_sym or (p.symbol and p.symbol == pos.symbol):
                    already_tracked = True
                    break
            if already_tracked:
                continue
            plan = TradePlan(
                symbol=normalize_symbol(pos.symbol) if pos.symbol else self.cfg.trading_symbol,
                direction=TradeDirection.LONG if pos.side == "LONG" else TradeDirection.SHORT,
                timeframe="unknown",
                bias_source="reconstructed from broker",
                entry_zone=str(pos.avg_price),
                entry_type=EntryType.MARKET,
                stop_loss=pos.avg_price * 0.99,
                stop_loss_reason="reconstructed default",
                take_profit_1=pos.avg_price * 1.03,
                tp1_exit_pct=50,
                take_profit_2=pos.avg_price * 1.045,
                tp2_exit_pct=50,
                risk_amount=0,
                position_size=pos.qty,
                rr_ratio=3.0,
                status=PlanStatus.ACTIVE,
                entry_price=pos.avg_price,
                is_intraday=True,
                ai_setup_done=True,
            )
            self.plans.append(plan)
            self.trailing.register(plan)
            log.info("Reconstructed active plan for %s qty=%s", pos.symbol, pos.qty)

        phantom_count = sum(1 for p in self.plans if p.rationale and "stale" in p.rationale)
        if phantom_count:
            log.warning("Invalidated %d phantom plan(s) — positions already closed on broker", phantom_count)

        self.state_store.save_plans(self.plans)

    def _has_open_workflow(self) -> bool:
        return any(
            p.status in {PlanStatus.WAITING_ENTRY, PlanStatus.PENDING, PlanStatus.ACTIVE}
            for p in self.plans
        )

    def _invalidate_stale_entries(self) -> list[TradePlan]:
        """Invalidate WAITING_ENTRY plans whose LTP has drifted too far from the
        entry zone for too many consecutive cycles.  Clean drift counters for
        plans that are no longer WAITING_ENTRY."""
        max_pct = self.cfg.entry_stale_pct / 100.0
        max_cycles = self.cfg.entry_stale_cycles
        invalidated: list[TradePlan] = []

        # Clean up counters for plans that are no longer waiting
        active_ids = {
            p.plan_id for p in self.plans
            if p.status == PlanStatus.WAITING_ENTRY
        }
        stale_keys = [k for k in self._entry_drift_cycles if k not in active_ids]
        for k in stale_keys:
            del self._entry_drift_cycles[k]

        for plan in self.plans:
            if plan.status != PlanStatus.WAITING_ENTRY:
                continue
            if plan.entry_price_low is None or plan.entry_price_high is None:
                continue

            try:
                ltp = self.entry_watcher._get_ltp(plan)
            except Exception:
                continue

            zone_center = (plan.entry_price_low + plan.entry_price_high) / 2
            if zone_center <= 0:
                continue

            drift_pct = abs(ltp - zone_center) / zone_center
            if drift_pct > max_pct:
                cycles = self._entry_drift_cycles.get(plan.plan_id, 0) + 1
                self._entry_drift_cycles[plan.plan_id] = cycles
                if cycles >= max_cycles:
                    plan.status = PlanStatus.INVALIDATED
                    plan.rationale = (
                        f"Entry stale — LTP {ltp:.2f} drifted {drift_pct:.1%} "
                        f"from zone {plan.entry_price_low:.2f}–{plan.entry_price_high:.2f} "
                        f"for {cycles} cycles"
                    )
                    invalidated.append(plan)
                    del self._entry_drift_cycles[plan.plan_id]
                    log.warning("Invalidated stale entry %s: %s", plan.symbol, plan.rationale)
            else:
                # Price came back into range — reset counter
                self._entry_drift_cycles.pop(plan.plan_id, None)

        return invalidated

    def _session(self) -> MarketSession:
        return get_market_session(
            self.cfg.trading_hours_ist,
            holidays=self.cfg.market_holidays,
        )

    def _market_payload(self, session: MarketSession) -> dict[str, object]:
        payload: dict[str, object] = {
            "is_open": session.is_open,
            "reason": session.reason,
            "seconds_until_open": session.seconds_until_open,
        }
        if session.next_open is not None:
            payload["next_open"] = session.next_open.isoformat()
        return payload

    def _log_session_change(self, session: MarketSession) -> None:
        if session.reason == self._last_session_reason:
            return
        if session.is_open:
            log.info("Market open — resuming trading")
        elif self.cfg.stop_at_market_close and session.reason == "after_close":
            log.info(
                "Market session ended — stopping process (next open %s)",
                format_next_open(session),
            )
        elif self.cfg.stop_at_market_close and not self.cfg.wait_for_market_open:
            log.info("Market closed (%s) — stopping process", session.reason)
        else:
            log.info(
                "Market closed (%s) — trading paused until %s (~%s)",
                session.reason,
                format_next_open(session),
                self._format_duration(session.seconds_until_open),
            )
        self._last_session_reason = session.reason

    def _should_stop_engine(self, session: MarketSession) -> bool:
        if session.is_open:
            return False
        if not self.cfg.stop_at_market_close:
            return False
        if session.reason == "after_close":
            return True
        if self.cfg.wait_for_market_open and session.reason in {"before_open", "weekend", "holiday"}:
            return False
        return True

    def _end_session_cleanup(self, session: MarketSession) -> None:
        if not self.scan_only:
            self.executor.square_off_intraday(self.plans)
        cancelled = 0
        for plan in self.plans:
            if plan.status in {PlanStatus.WAITING_ENTRY, PlanStatus.PENDING}:
                plan.status = PlanStatus.INVALIDATED
                cancelled += 1
                log.info("Cancelled pending plan %s — market session ended", plan.symbol)
        if cancelled:
            self.trade_logger.log(
                "PLAN",
                self.cfg.trading_symbol,
                self.broker.name,
                f"Cancelled {cancelled} pending plan(s) at session end",
                {"reason": session.reason},
            )
        self.state_store.save_plans(self.plans)

    @staticmethod
    def _format_duration(seconds: int) -> str:
        if seconds < 60:
            return f"{seconds}s"
        minutes, secs = divmod(seconds, 60)
        if minutes < 60:
            return f"{minutes}m {secs}s"
        hours, minutes = divmod(minutes, 60)
        if hours < 24:
            return f"{hours}h {minutes}m"
        days, hours = divmod(hours, 24)
        return f"{days}d {hours}h"

    def _write_heartbeat(self, session: MarketSession) -> None:
        active_count = sum(1 for p in self.plans if p.status == PlanStatus.ACTIVE)
        self.state_store.write_heartbeat(
            loop_count=self._loop_count,
            active_plans=active_count,
            market=self._market_payload(session),
        )

    def _idle_once(self, session: MarketSession) -> bool:
        """Idle heartbeat when market is closed. Returns False when the engine should stop."""
        self._loop_count += 1
        self._log_session_change(session)
        self._write_heartbeat(session)
        self.trade_logger.log(
            "MONITOR",
            "SYSTEM",
            self.broker.name,
            f"Trading paused ({session.reason})",
            {
                "loop": self._loop_count,
                "next_open": format_next_open(session),
                "seconds_until_open": session.seconds_until_open,
                "stop_at_close": self.cfg.stop_at_market_close,
            },
        )

        if session.reason == "after_close":
            self._end_session_cleanup(session)

        return not self._should_stop_engine(session)

    def run_once(self) -> bool:
        """Run one engine cycle. Returns False when the daemon should exit."""
        session = self._session()
        if not session.is_open:
            return self._idle_once(session)

        if self._last_session_reason != "open":
            self._log_session_change(session)

        self._loop_count += 1
        active_count = sum(1 for p in self.plans if p.status == PlanStatus.ACTIVE)
        self._write_heartbeat(session)
        self.trade_logger.log(
            "MONITOR", "SYSTEM", self.broker.name,
            "Heartbeat",
            {"loop": self._loop_count, "active_plans": active_count, "market": "open"},
        )

        if not self._has_open_workflow() and not self.scan_only:
            # Churn guard: skip scan if a recent monitor-triggered close is cooling down
            import time as _time
            now = _time.monotonic()
            cooldown_blocked = [
                sym for sym, until in self._symbol_cooldown_until.items()
                if now < until
            ]
            if cooldown_blocked:
                remaining = max(0, int(max(self._symbol_cooldown_until.values()) - now))
                log.info(
                    "Scan suppressed — %s in cooldown (%ds remaining)",
                    cooldown_blocked, remaining,
                )
            else:
                results = self.scanner.scan()
                pending = self.scanner.pending_plans(results)
                for plan in pending:
                    if plan.plan_id in {p.plan_id for p in self.plans}:
                        continue
                    tech_summary = {"symbol": self.cfg.trading_symbol, "strength": plan.signal_strength}
                    plan = self.planner.apply_ai_markers(plan, tech_summary)
                    if plan.status == PlanStatus.PAUSED:
                        log.info("Trade paused: %s", plan.rationale)
                        continue
                    self.plans.append(plan)
                    log.info("\n%s", plan.format_plan())
                    self.trade_logger.log("PLAN", plan.symbol, self.broker.name, plan.rationale, plan.to_dict())

        ready = self.entry_watcher.tick(self.plans)
        if ready:
            log.info("ENGINE: %d plan(s) ready for execution: %s",
                     len(ready), [p.symbol for p in ready])
        stale_invalidated = self._invalidate_stale_entries()
        open_count = sum(1 for p in self.plans if p.status == PlanStatus.ACTIVE)

        if not self.scan_only:
            # Snapshot ACTIVE plans BEFORE execution so new entries this
            # cycle can be detected for session tracking.
            pre_exec_active = {p.plan_id for p in self.plans if p.status == PlanStatus.ACTIVE}

            # Single execution pass per cycle: covers plans the entry watcher
            # just marked PENDING and earlier failed placements. A failed
            # placement sets a retry cooldown inside execute_plan, so it is
            # never attempted twice in the same cycle.
            for plan in self.plans:
                if plan.status == PlanStatus.PENDING:
                    self.executor.execute_plan(plan, open_count)
                    open_count = sum(1 for p in self.plans if p.status == PlanStatus.ACTIVE)

            # Track new entries: plans that became ACTIVE this cycle
            for plan in self.plans:
                if plan.status == PlanStatus.ACTIVE and plan.plan_id not in pre_exec_active:
                    self.session.register_entry(plan.symbol, plan.risk_amount)
                    log.info("Session: trade #%d entered — %s", self.session.trade_count, plan.symbol)

            # Plans that were ACTIVE at any point this cycle — used for
            # close/P&L tracking so square-offs are not missed.
            ever_active = pre_exec_active | {
                p.plan_id for p in self.plans if p.status == PlanStatus.ACTIVE
            }

            self.executor.square_off_intraday(self.plans)

            # Trade-window-end square-off: close all positions when 15:15 approaches
            if is_trade_window_ending(self.cfg.trade_window_end_ist):
                active_at_end = [p for p in self.plans if p.status == PlanStatus.ACTIVE]
                if active_at_end:
                    log.warning(
                        "Trade window ending (%s IST) — squaring off %d position(s)",
                        self.cfg.trade_window_end_ist, len(active_at_end),
                    )
                    for plan in active_at_end:
                        self.executor.close_plan(plan, reason=f"Trade window end {self.cfg.trade_window_end_ist} IST")

            # Snapshot ACTIVE plans before monitor tick to detect churn
            active_before_monitor = {p.plan_id for p in self.plans if p.status == PlanStatus.ACTIVE}

            self.monitor.tick(self.plans)
            # Detect plans the monitor just closed — set cooldown to prevent instant re-entry
            closed_by_monitor = [
                p for p in self.plans
                if p.status == PlanStatus.CLOSED and p.plan_id in active_before_monitor
            ]
            for p in closed_by_monitor:
                import time as _time
                self._symbol_cooldown_until[p.symbol] = _time.monotonic() + self._churn_cooldown_seconds
                log.info(
                    "Churn guard: %s closed by monitor — cooldown %ds before re-entry",
                    p.symbol, self._churn_cooldown_seconds,
                )

            # Track closed trades in session (P&L tracking)
            for p in self.plans:
                if p.status in (PlanStatus.CLOSED, PlanStatus.INVALIDATED) and p.plan_id in ever_active:
                    if p.meta.get("_session_tracked"):
                        continue  # only register once per trade
                    # Prefer the realized P&L captured at close (fill price or
                    # live quote); estimate from SL/TP only as a last resort.
                    pnl_raw = p.meta.get("realized_pnl")
                    if pnl_raw is not None:
                        pnl = float(pnl_raw)
                    else:
                        pnl = 0.0
                        if p.entry_price and p.entry_price > 0:
                            exit_px = p.take_profit_1 if p.tp1_hit else p.stop_loss
                            if p.direction == TradeDirection.LONG:
                                pnl = (exit_px - p.entry_price) * p.position_size
                            else:
                                pnl = (p.entry_price - exit_px) * p.position_size
                    self.session.register_close(pnl=pnl)
                    # Feed the daily-loss circuit breaker with real percentages
                    if self.session.initial_capital > 0:
                        self.limits.record_pnl(pnl / self.session.initial_capital * 100.0)
                        if self.limits.state.halted:
                            log.warning("LimitsGuard: %s", self.limits.state.halt_reason)
                    p.meta["_session_tracked"] = True
                    log.info("Session: trade closed — P&L=₹%.0f | %s", pnl, self.session.status_summary())

            # Log session status every cycle
            if self._loop_count % 5 == 0:  # every 5 cycles (~5 min)
                log.info("%s", self.session.status_summary())

        self.state_store.save_plans(self.plans)

        session = self._session()
        if not session.is_open and session.reason == "after_close":
            return self._idle_once(session)

        return True

    def run(self) -> None:
        self.bootstrap()
        interval = self.cfg.monitor_interval
        log.info(
            "Entering 24x7 loop (trade interval=%ss, idle poll=%ss, scan_only=%s)",
            interval,
            self.cfg.market_closed_poll_interval,
            self.scan_only,
        )

        while not self._stop:
            try:
                should_continue = self.run_once()
            except Exception as e:
                log.exception("Loop error: %s", e)
                # Auth circuit breaker: halt on repeated 401 / expired token errors
                exc_str = str(e)
                if any(tag in exc_str for tag in ("DH-901", "Invalid_Authentication", "token is invalid", "token has expired")):
                    self._consecutive_auth_failures += 1
                    if self._consecutive_auth_failures >= self._max_auth_failures:
                        log.critical(
                            "AUTH CIRCUIT BREAKER: %d consecutive auth failures — halting engine. "
                            "Refresh Dhan access token in .env and restart.",
                            self._consecutive_auth_failures,
                        )
                        break
                if _is_api_error(e) and self.limits.record_api_error():
                    log.error("Trading halted: %s", self.limits.state.halt_reason)
                    break
                should_continue = True
            else:
                self.limits.record_api_success()
                self._consecutive_auth_failures = 0  # reset on clean cycle

            if not should_continue:
                log.info("Engine stopping — market session ended")
                break

            session = self._session()
            sleep_for = compute_idle_sleep_seconds(
                session,
                monitor_interval=interval,
                closed_poll_interval=self.cfg.market_closed_poll_interval,
            )
            for _ in range(sleep_for):
                if self._stop:
                    break
                time.sleep(1)

        self.state_store.write_heartbeat(
            loop_count=self._loop_count,
            active_plans=0,
            status="stopped",
            market=self._market_payload(self._session()),
        )
        self.broker.close()
        log.info("Engine stopped cleanly.")

    def stop(self, *_: object) -> None:
        log.info("Stop requested...")
        self._stop = True


def setup_logging() -> None:
    import sys
    import os
    
    # Ensure unbuffered output for background processes
    if not sys.stdout.isatty():
        # Force line buffering for non-TTY (background) execution
        os.environ["PYTHONUNBUFFERED"] = "1"
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    
    # Use FileHandler with immediate flush for background processes
    log_file = os.getenv("LOG_FILE", "/tmp/dream-maker-engine.log")
    
    # Create custom formatter
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S"
    )
    
    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add console handler (for interactive mode)
    if sys.stdout.isatty():
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    else:
        # Add file handler with immediate flush (for background mode)
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        
        # Force immediate flush after each log
        class FlushingFileHandler(logging.FileHandler):
            def emit(self, record):
                super().emit(record)
                self.flush()
        
        flush_handler = FlushingFileHandler(log_file, mode='a', encoding='utf-8')
        flush_handler.setFormatter(formatter)
        flush_handler.setLevel(logging.INFO)
        root_logger.addHandler(flush_handler)
    
    # Reduce noise from HTTP client
    logging.getLogger("httpx").setLevel(logging.WARNING)
