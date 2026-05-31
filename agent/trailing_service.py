"""Coordinates per-plan trailing managers and broker updates."""

from __future__ import annotations

import logging

from agent.executor import TradeExecutor
from config import Config
from audit.trade_logger import TradeLogger
from models.trade_plan import PlanStatus, TradePlan
from risk.trailing import TrailingBracketManager, TrailingState, TrailUpdate

log = logging.getLogger("dream_maker.trailing_service")


class TrailingService:
    def __init__(self, cfg: Config, executor: TradeExecutor, trade_logger: TradeLogger):
        self.cfg = cfg
        self.executor = executor
        self.trade_logger = trade_logger
        self._managers: dict[str, TrailingBracketManager] = {}

    def register(self, plan: TradePlan) -> None:
        if not self.cfg.trail_enabled or plan.status != PlanStatus.ACTIVE:
            return
        if plan.plan_id in self._managers:
            return
        entry = plan.entry_price or plan.entry_target()
        trail_meta = plan.meta.get("trail", {})
        state = TrailingState.from_dict(
            trail_meta,
            fallback_entry=entry,
            plan_sl=plan.stop_loss,
            plan_tp1=plan.take_profit_1,
            plan_tp2=plan.take_profit_2,
        ) if trail_meta else None

        self._managers[plan.plan_id] = TrailingBracketManager(
            direction=plan.direction,
            entry=entry,
            initial_sl=plan.stop_loss,
            initial_tp1=plan.take_profit_1,
            initial_tp2=plan.take_profit_2,
            trail_activate_pct=self.cfg.trail_activate_pct,
            trail_sl_distance_pct=self.cfg.trail_sl_distance_pct,
            trail_tp_reward_pct=self.cfg.trail_tp_reward_pct,
            breakeven_progress_pct=self.cfg.trail_breakeven_progress_pct,
            breakeven_buffer_pct=self.cfg.trail_breakeven_buffer_pct,
            state=state,
        )
        plan.stop_loss = self._managers[plan.plan_id].state.current_sl
        plan.take_profit_1 = self._managers[plan.plan_id].state.current_tp
        plan.take_profit_2 = self._managers[plan.plan_id].state.current_tp2

    def unregister(self, plan_id: str) -> None:
        self._managers.pop(plan_id, None)

    def tick(self, plan: TradePlan, ltp: float) -> TrailUpdate:
        if not self.cfg.trail_enabled:
            return TrailUpdate()
        if plan.plan_id not in self._managers:
            self.register(plan)
        manager = self._managers.get(plan.plan_id)
        if not manager:
            return TrailUpdate()

        update = manager.on_price(ltp)
        plan.meta["trail"] = manager.state.to_dict()

        if update.new_sl is not None or update.new_tp is not None:
            self.executor.apply_trail(plan, update)
        elif update.tp1_milestone and not plan.tp1_hit:
            plan.tp1_hit = True
            self.trade_logger.log(
                "MONITOR", plan.symbol, self.executor.broker.name,
                "TP1 milestone — trailing active",
                {"ltp": ltp, "tp1": plan.take_profit_1},
            )

        return update
