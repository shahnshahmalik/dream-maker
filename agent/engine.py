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
from utils.market_hours import (
    MarketSession,
    compute_idle_sleep_seconds,
    format_next_open,
    get_market_session,
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
        self.state_store = StateStore(cfg.state_dir, cfg.trade_log_path)
        self.broker = get_broker(cfg)
        self.llm = get_llm(cfg)
        self.risk = RiskManager(cfg.risk_pct_per_trade, cfg.min_rr_ratio)
        self.limits = LimitsGuard(
            cfg.max_open_trades,
            cfg.daily_loss_limit,
            cfg.event_blackout_minutes,
            cfg.scheduled_events,
        )
        self.pipeline = AnalysisPipeline(self.broker, cfg, self.trade_logger, self.risk)
        self.scanner = WatchlistScanner(self.pipeline, cfg, self.broker)
        self.planner = TradePlanner(self.llm, cfg, self.trade_logger)
        self.entry_watcher = EntryWatcher(self.broker, cfg)
        self.executor = TradeExecutor(self.broker, cfg, self.trade_logger, self.limits)
        self.trailing = TrailingService(cfg, self.executor, self.trade_logger)
        self.monitor = PositionMonitor(
            self.broker, self.llm, self.executor, cfg, self.trade_logger, self.trailing,
        )
        self.plans: list[TradePlan] = []

    def _recover_state(self) -> None:
        recovered = self.state_store.load_plans()
        if recovered:
            log.info("Recovered %s plan(s) from state/active_plans.json", len(recovered))
            self.plans.extend(recovered)

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

        self._recover_state()

        funds = self.broker.get_funds()
        log.info("Funds: available=%.2f total=%.2f %s", funds.available, funds.total, funds.currency)
        self.trade_logger.log(
            "MONITOR", "SYSTEM", self.broker.name,
            "Startup connectivity OK",
            {"available": funds.available, "total": funds.total},
        )

        positions = self.broker.get_positions()
        for pos in positions:
            pos_sym = normalize_symbol(pos.symbol) if pos.symbol else ""
            if pos_sym != self.cfg.trading_symbol and not pos_sym.startswith(
                normalize_symbol(self.cfg.trading_symbol).replace("50IDX", "").replace("IDX", "")[:5]
            ):
                log.info(
                    "Ignoring position %s — not TRADING_SYMBOL (%s)",
                    pos.symbol, self.cfg.trading_symbol,
                )
                continue
            if any(p.status == PlanStatus.ACTIVE for p in self.plans):
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

        self.state_store.save_plans(self.plans)

    def _has_open_workflow(self) -> bool:
        return any(
            p.status in {PlanStatus.WAITING_ENTRY, PlanStatus.PENDING, PlanStatus.ACTIVE}
            for p in self.plans
        )

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
        open_count = sum(1 for p in self.plans if p.status == PlanStatus.ACTIVE)

        if not self.scan_only:
            for plan in ready:
                self.executor.execute_plan(plan, open_count)
                open_count = sum(1 for p in self.plans if p.status == PlanStatus.ACTIVE)

            for plan in self.plans:
                if plan.status == PlanStatus.PENDING:
                    self.executor.execute_plan(plan, open_count)
                    open_count = sum(1 for p in self.plans if p.status == PlanStatus.ACTIVE)

            self.executor.square_off_intraday(self.plans)
            self.monitor.tick(self.plans)

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
                if _is_api_error(e) and self.limits.record_api_error():
                    log.error("Trading halted: %s", self.limits.state.halt_reason)
                    break
                should_continue = True
            else:
                self.limits.record_api_success()

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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
