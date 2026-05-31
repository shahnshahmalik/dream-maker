"""Trading engine — startup sequence and main loop."""

from __future__ import annotations

import logging
import time

from agent.entry_watcher import EntryWatcher
from agent.executor import TradeExecutor
from agent.monitor import PositionMonitor
from agent.planner import TradePlanner
from analysis.pipeline import AnalysisPipeline
from analysis.scanner import WatchlistScanner
from audit.state_store import StateStore
from config import Config
from llm.factory import get_llm
from audit.trade_logger import TradeLogger
from models.trade_plan import EntryType, PlanStatus, TradeDirection, TradePlan
from providers.factory import get_broker
from risk.limits import LimitsGuard
from risk.manager import RiskManager
from utils.symbols import normalize_symbol

log = logging.getLogger("dream_maker.engine")


class TradingEngine:
    def __init__(self, cfg: Config, *, scan_only: bool = False):
        self.cfg = cfg
        self.scan_only = scan_only
        self._stop = False
        self._loop_count = 0

        self.trade_logger = TradeLogger(cfg.trade_log_path)
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
        self.monitor = PositionMonitor(self.broker, self.llm, self.executor, cfg, self.trade_logger)
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
            log.info("Reconstructed active plan for %s qty=%s", pos.symbol, pos.qty)

        self.state_store.save_plans(self.plans)

    def _has_open_workflow(self) -> bool:
        return any(
            p.status in {PlanStatus.WAITING_ENTRY, PlanStatus.PENDING, PlanStatus.ACTIVE}
            for p in self.plans
        )

    def run_once(self) -> None:
        self._loop_count += 1
        active_count = sum(1 for p in self.plans if p.status == PlanStatus.ACTIVE)
        self.state_store.write_heartbeat(loop_count=self._loop_count, active_plans=active_count)
        self.trade_logger.log(
            "MONITOR", "SYSTEM", self.broker.name,
            "Heartbeat",
            {"loop": self._loop_count, "active_plans": active_count},
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

    def run(self) -> None:
        self.bootstrap()
        interval = self.cfg.monitor_interval
        log.info("Entering main loop (interval=%ss, scan_only=%s)", interval, self.scan_only)

        while not self._stop:
            try:
                self.run_once()
            except Exception as e:
                log.exception("Loop error: %s", e)
                if self.limits.record_api_error():
                    log.error("Trading halted: %s", self.limits.state.halt_reason)
                    break
            else:
                self.limits.record_api_success()

            for _ in range(interval):
                if self._stop:
                    break
                time.sleep(1)

        self.state_store.write_heartbeat(
            loop_count=self._loop_count, active_plans=0, status="stopped"
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
