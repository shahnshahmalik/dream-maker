"""Trade execution — orders, Groww SL/TP monitor."""

from __future__ import annotations

import logging
import threading
from typing import Callable

from config import Config
from audit.trade_logger import TradeLogger
from models.trade_plan import PlanStatus, TradeDirection, TradePlan
from providers.base import BrokerProvider
from providers.groww import GrowwProvider
from risk.limits import LimitsGuard
from utils.market_hours import is_market_open, is_square_off_time

log = logging.getLogger("dream_maker.executor")


class GrowwPriceMonitor:
    """Client-side SL/TP monitoring for Groww equity positions."""

    def __init__(self, broker: GrowwProvider):
        self.broker = broker
        self._monitors: dict[str, threading.Event] = {}
        self._threads: dict[str, threading.Thread] = {}

    def start(
        self,
        plan: TradePlan,
        on_trigger: Callable[[TradePlan, str], None],
        poll_seconds: float = 5.0,
    ) -> None:
        if plan.plan_id in self._threads:
            return
        stop_event = threading.Event()
        self._monitors[plan.plan_id] = stop_event

        def _loop() -> None:
            while not stop_event.is_set():
                try:
                    quote = self.broker.get_quote(plan.symbol)
                    ltp = quote.ltp
                    if plan.direction == TradeDirection.LONG:
                        if ltp <= plan.stop_loss:
                            on_trigger(plan, "SL hit")
                            break
                        if ltp >= plan.take_profit_1 and not plan.tp1_hit:
                            on_trigger(plan, "TP1 hit")
                        if ltp >= plan.take_profit_2:
                            on_trigger(plan, "TP2 hit")
                            break
                    else:
                        if ltp >= plan.stop_loss:
                            on_trigger(plan, "SL hit")
                            break
                        if ltp <= plan.take_profit_1 and not plan.tp1_hit:
                            on_trigger(plan, "TP1 hit")
                        if ltp <= plan.take_profit_2:
                            on_trigger(plan, "TP2 hit")
                            break
                except Exception as e:
                    log.debug("Groww monitor error: %s", e)
                stop_event.wait(poll_seconds)

        t = threading.Thread(target=_loop, daemon=True, name=f"groww-mon-{plan.plan_id}")
        self._threads[plan.plan_id] = t
        t.start()

    def stop(self, plan_id: str) -> None:
        ev = self._monitors.pop(plan_id, None)
        if ev:
            ev.set()
        self._threads.pop(plan_id, None)


class TradeExecutor:
    def __init__(
        self,
        broker: BrokerProvider,
        cfg: Config,
        trade_logger: TradeLogger,
        limits: LimitsGuard,
    ):
        self.broker = broker
        self.cfg = cfg
        self.trade_logger = trade_logger
        self.limits = limits
        self._groww_monitor: GrowwPriceMonitor | None = None
        if isinstance(broker, GrowwProvider):
            self._groww_monitor = GrowwPriceMonitor(broker)

    def execute_plan(self, plan: TradePlan, open_count: int) -> TradePlan:
        if plan.status != PlanStatus.PENDING:
            return plan

        from utils.symbols import normalize_symbol

        if normalize_symbol(plan.symbol) != self.cfg.trading_symbol:
            log.warning(
                "Rejecting plan for %s — only TRADING_SYMBOL=%s is allowed",
                plan.symbol, self.cfg.trading_symbol,
            )
            plan.status = PlanStatus.INVALIDATED
            return plan

        if not is_market_open(self.cfg.trading_hours_ist, holidays=self.cfg.market_holidays):
            log.info("Outside trading hours — skipping %s", plan.symbol)
            return plan

        ok, reason = self.limits.can_open_trade(open_count)
        if not ok:
            log.warning("Cannot open %s: %s", plan.symbol, reason)
            plan.status = PlanStatus.INVALIDATED
            return plan

        funds = self.broker.get_funds()
        if funds.available < plan.risk_amount:
            log.warning("Insufficient funds for %s", plan.symbol)
            plan.status = PlanStatus.INVALIDATED
            return plan

        side = "BUY" if plan.direction == TradeDirection.LONG else "SELL"
        entry_price = float(plan.entry_zone)
        result = self.broker.place_order(
            symbol=plan.symbol,
            side=side,
            qty=plan.position_size,
            order_type=plan.entry_type.value if plan.entry_type.value != "STOP_ENTRY" else "STOP_LOSS",
            price=entry_price,
            sl=plan.stop_loss,
            tp=plan.take_profit_1,
        )

        if not result.success:
            self.limits.record_api_error()
            self.trade_logger.log("ORDER", plan.symbol, self.broker.name, result.message, {"success": False})
            return plan

        if not result.sl_order_id or not result.tp_order_id:
            log.error("Naked entry blocked — SL/TP not confirmed for %s", plan.symbol)
            if result.order_id:
                self.broker.cancel_order(result.order_id)
                flatten = "SELL" if side == "BUY" else "BUY"
                self.broker.place_order(plan.symbol, flatten, plan.position_size, "MARKET")
            plan.status = PlanStatus.INVALIDATED
            self.trade_logger.log(
                "ORDER", plan.symbol, self.broker.name,
                "Entry reversed — bracket SL/TP missing (never naked)",
                {"success": False, "order_id": result.order_id},
            )
            return plan

        self.limits.record_api_success()
        plan.status = PlanStatus.ACTIVE
        plan.order_id = result.order_id
        plan.sl_order_id = result.sl_order_id
        plan.tp_order_id = result.tp_order_id
        plan.entry_price = result.fill_price or entry_price
        self.limits.record_trade_opened()

        self.trade_logger.log(
            "ORDER", plan.symbol, self.broker.name,
            f"Bracketed entry {side} qty={plan.position_size} @ {plan.entry_price}",
            {
                "order_id": result.order_id,
                "sl_order_id": result.sl_order_id,
                "tp_order_id": result.tp_order_id,
                "simulation": self.cfg.simulation_mode,
                "success": True,
                "plan": plan.to_dict(),
            },
        )

        if self._groww_monitor:
            self._groww_monitor.start(plan, self._on_groww_trigger)

        return plan

    def _on_groww_trigger(self, plan: TradePlan, trigger: str) -> None:
        log.warning("Groww monitor trigger for %s: %s", plan.symbol, trigger)
        if trigger == "TP1 hit":
            plan.tp1_hit = True
            self.modify_sl_breakeven(plan)
            return
        self.close_plan(plan, reason=trigger)

    def modify_sl_breakeven(self, plan: TradePlan) -> None:
        if not plan.sl_order_id or not plan.entry_price:
            return
        result = self.broker.modify_order(plan.sl_order_id, sl=plan.entry_price)
        plan.stop_loss = plan.entry_price
        self.trade_logger.log(
            "MODIFY", plan.symbol, self.broker.name,
            f"SL moved to breakeven {plan.entry_price}",
            {"sl_order_id": plan.sl_order_id, "success": result.success},
        )

    def apply_trail(self, plan: TradePlan, update) -> None:
        """Push trailed SL/TP levels to the broker bracket orders."""
        from risk.trailing import TrailUpdate

        if not isinstance(update, TrailUpdate):
            return

        sl_ok = tp_ok = True
        if update.new_sl is not None and plan.sl_order_id:
            result = self.broker.modify_order(plan.sl_order_id, sl=update.new_sl)
            sl_ok = result.success
            if sl_ok:
                plan.stop_loss = update.new_sl

        if update.new_tp is not None and plan.tp_order_id:
            result = self.broker.modify_order(plan.tp_order_id, tp=update.new_tp)
            tp_ok = result.success
            if tp_ok:
                plan.take_profit_1 = update.new_tp

        if update.new_tp2 is not None:
            plan.take_profit_2 = update.new_tp2

        if update.tp1_milestone:
            plan.tp1_hit = True

        if update.new_sl is not None or update.new_tp is not None:
            self.trade_logger.log(
                "MODIFY", plan.symbol, self.broker.name,
                update.reason or "Trailing SL/TP updated",
                {
                    "new_sl": update.new_sl,
                    "new_tp": update.new_tp,
                    "new_tp2": update.new_tp2,
                    "sl_ok": sl_ok,
                    "tp_ok": tp_ok,
                    "plan": plan.to_dict(),
                },
            )
            log.info(
                "Trailed %s: SL=%s TP=%s (%s)",
                plan.symbol, plan.stop_loss, plan.take_profit_1, update.reason,
            )

    def partial_exit(self, plan: TradePlan, pct: float) -> None:
        qty = max(1, int(plan.position_size * pct / 100))
        side = "SELL" if plan.direction == TradeDirection.LONG else "BUY"
        result = self.broker.place_order(plan.symbol, side, qty, "MARKET")
        plan.position_size -= qty
        self.trade_logger.log(
            "CLOSE", plan.symbol, self.broker.name,
            f"Partial exit {pct}% ({qty} units)",
            {"success": result.success},
        )

    def close_plan(self, plan: TradePlan, reason: str = "manual close") -> None:
        if plan.position_size <= 0:
            plan.status = PlanStatus.CLOSED
            return
        side = "SELL" if plan.direction == TradeDirection.LONG else "BUY"
        if plan.order_id:
            self.broker.cancel_order(plan.order_id)
        if plan.sl_order_id:
            self.broker.cancel_order(plan.sl_order_id)
        if plan.tp_order_id:
            self.broker.cancel_order(plan.tp_order_id)
        result = self.broker.place_order(plan.symbol, side, plan.position_size, "MARKET")
        plan.status = PlanStatus.CLOSED
        if self._groww_monitor:
            self._groww_monitor.stop(plan.plan_id)
        self.trade_logger.log(
            "CLOSE", plan.symbol, self.broker.name, reason,
            {"success": result.success, "qty": plan.position_size, "plan": plan.to_dict()},
        )

    def square_off_intraday(self, plans: list[TradePlan]) -> None:
        if not is_square_off_time(self.cfg.intraday_square_off):
            return
        for plan in plans:
            if plan.is_intraday and plan.status == PlanStatus.ACTIVE:
                log.warning("Intraday square-off: closing %s", plan.symbol)
                self.close_plan(plan, reason="Intraday square-off 15:15 IST")
