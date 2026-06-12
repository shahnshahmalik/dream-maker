"""Coordinates per-plan trailing managers and broker updates."""

from __future__ import annotations

import logging

from agent.executor import TradeExecutor
from config import Config
from audit.trade_logger import TradeLogger
from models.trade_plan import PlanStatus, TradeDirection, TradePlan
from risk.trailing import TrailingBracketManager, TrailingState, TrailUpdate

log = logging.getLogger("dream_maker.trailing_service")


class TrailingService:
    def __init__(self, cfg: Config, executor: TradeExecutor, trade_logger: TradeLogger):
        self.cfg = cfg
        self.executor = executor
        self.trade_logger = trade_logger
        self._managers: dict[str, TrailingBracketManager] = {}

    @staticmethod
    def _effective_params(plan: TradePlan) -> tuple[TradeDirection, float, float, float]:
        """Return (direction, sl, tp1, tp2) — mirroring for BUY_ONLY flipped positions.

        When the balance manager flips SELL→BUY for a SHORT bias (buying a put),
        the actual position is LONG the option. Trail direction and SL/TP levels
        must reflect the real position, not the market bias.
        """
        direction = plan.direction
        sl = plan.stop_loss
        tp1 = plan.take_profit_1
        tp2 = plan.take_profit_2

        # BUY_ONLY: SHORT + PE = actually LONG the option (we BUY a put, not sell)
        if direction == TradeDirection.SHORT and plan.symbol.upper().endswith("PE"):
            direction = TradeDirection.LONG
            entry = plan.entry_price or plan.entry_target()
            # Mirror SL/TP around entry: SHORT SL is above entry, LONG SL is below
            sl_dist = abs(entry - sl)
            sl = entry - sl_dist
            tp1 = entry + abs(entry - tp1)
            tp2 = entry + abs(entry - tp2)
            log.info(
                "BUY_ONLY flip: SHORT bias → LONG PE trail. SL %.2f→%.2f TP1 %.2f→%.2f",
                plan.stop_loss, sl, plan.take_profit_1, tp1,
            )

        return direction, sl, tp1, tp2

    def register(self, plan: TradePlan) -> None:
        if not self.cfg.trail_enabled or plan.status != PlanStatus.ACTIVE:
            return

        direction, sl, tp1, tp2 = self._effective_params(plan)

        # Force re-register if direction was wrong (BUY_ONLY flip detected)
        existing = self._managers.get(plan.plan_id)
        if existing and existing.direction != direction:
            log.info("Re-registering trail for %s — direction corrected %s→%s",
                     plan.symbol, existing.direction.value, direction.value)
            self._managers.pop(plan.plan_id, None)

        if plan.plan_id in self._managers:
            return
        entry = plan.entry_price or plan.entry_target()
        trail_meta = plan.meta.get("trail", {})
        is_scalp = plan.meta.get("setup_type") == "momentum_scalp"
        trail_activate = self.cfg.scalp_trail_activate_pct if is_scalp else self.cfg.trail_activate_pct
        trail_sl_distance = self.cfg.scalp_trail_sl_distance_pct if is_scalp else self.cfg.trail_sl_distance_pct
        breakeven_progress = (
            self.cfg.scalp_trail_breakeven_progress_pct if is_scalp else self.cfg.trail_breakeven_progress_pct
        )
        # Only restore previous trail state if direction hasn't changed.
        # When direction flips (BUY_ONLY correction), start fresh — old state is inverted.
        state = TrailingState.from_dict(
            trail_meta,
            fallback_entry=entry,
            plan_sl=sl,
            plan_tp1=tp1,
            plan_tp2=tp2,
        ) if trail_meta and existing is None else None

        self._managers[plan.plan_id] = TrailingBracketManager(
            direction=direction,
            entry=entry,
            initial_sl=sl,
            initial_tp1=tp1,
            initial_tp2=tp2,
            trail_activate_pct=trail_activate,
            trail_sl_distance_pct=trail_sl_distance,
            trail_tp_reward_pct=self.cfg.trail_tp_reward_pct,
            breakeven_progress_pct=breakeven_progress,
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
