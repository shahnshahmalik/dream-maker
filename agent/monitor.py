"""Position monitoring — trailing, rule-first, AI only on cooldown."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from agent.executor import TradeExecutor
from agent.trailing_service import TrailingService
from config import Config
from analysis.technical import check_ltf_structure_break
from llm.base import LLMProvider
from audit.trade_logger import TradeLogger
from models.review import AIReviewRequest, ReviewDecision
from models.trade_plan import PlanStatus, TradeDirection, TradePlan
from providers.base import BrokerProvider

log = logging.getLogger("dream_maker.monitor")


class PositionMonitor:
    def __init__(
        self,
        broker: BrokerProvider,
        llm: LLMProvider,
        executor: TradeExecutor,
        cfg: Config,
        trade_logger: TradeLogger,
        trailing: TrailingService | None = None,
    ):
        self.broker = broker
        self.llm = llm
        self.executor = executor
        self.cfg = cfg
        self.trade_logger = trade_logger
        self.trailing = trailing or TrailingService(cfg, executor, trade_logger)

    def tick(self, plans: list[TradePlan]) -> None:
        active = [p for p in plans if p.status == PlanStatus.ACTIVE]
        for plan in active:
            self.trailing.register(plan)
            self._monitor_one(plan)

    def _monitor_one(self, plan: TradePlan) -> None:
        quote = self.broker.get_quote(plan.symbol)
        ltp = quote.ltp
        entry = plan.entry_price or plan.entry_target()
        sl_dist = abs(entry - plan.stop_loss)
        adverse = 0.0
        favorable = 0.0
        if plan.direction == TradeDirection.LONG:
            if ltp < entry:
                adverse = entry - ltp
            else:
                favorable = ltp - entry
        else:
            if ltp > entry:
                adverse = ltp - entry
            else:
                favorable = entry - ltp

        moving_as_expected = favorable >= adverse and favorable > 0

        if moving_as_expected and self.cfg.trail_enabled:
            self.trailing.tick(plan, ltp)

        candles_15m = self.broker.get_ohlcv(plan.symbol, "15m", 20)
        candles_1h = self.broker.get_ohlcv(plan.symbol, "1h", 20)
        divergence_reasons: list[str] = []

        if sl_dist > 0 and adverse / sl_dist > 0.5:
            divergence_reasons.append("Price moved >50% toward SL")

        if check_ltf_structure_break(candles_15m, plan.direction):
            divergence_reasons.append("LTF structure break")

        if plan.direction == TradeDirection.LONG and ltp < plan.stop_loss * 1.01:
            divergence_reasons.append("Near HTF invalidation/support breach")
        elif plan.direction == TradeDirection.SHORT and ltp > plan.stop_loss * 0.99:
            divergence_reasons.append("Near HTF invalidation/resistance breach")

        self.trade_logger.log(
            "MONITOR", plan.symbol, self.broker.name,
            f"LTP={ltp:.2f} expected={moving_as_expected} SL={plan.stop_loss:.2f} TP={plan.take_profit_1:.2f}",
            {
                "divergence": divergence_reasons,
                "moving_as_expected": moving_as_expected,
                "trail": plan.meta.get("trail"),
            },
        )

        if not divergence_reasons:
            return

        if sl_dist > 0 and adverse / sl_dist > 0.75:
            self.trailing.unregister(plan.plan_id)
            self.executor.close_plan(plan, reason=">75% toward SL — rule-based exit")
            return

        if not self._ai_cooldown_elapsed(plan):
            log.info(
                "Divergence on %s (%s) — AI review on cooldown, holding",
                plan.symbol,
                "; ".join(divergence_reasons),
            )
            return

        log.info(
            "AI review request [%s] for %s — divergence: %s",
            self.llm.name,
            plan.symbol,
            "; ".join(divergence_reasons),
        )
        request = AIReviewRequest(
            symbol=plan.symbol,
            direction=plan.direction.value,
            original_rationale=plan.rationale,
            entry_price=entry,
            current_price=ltp,
            stop_loss=plan.stop_loss,
            macro_env=plan.macro_env,
            candles_15m=candles_15m,
            candles_1h=candles_1h,
            divergence_reasons=divergence_reasons,
        )
        response = self.llm.review_trade(request)
        plan.last_ai_review_at = datetime.now(timezone.utc)
        extra = ""
        if response.new_stop_loss is not None:
            extra = f" | new SL={response.new_stop_loss:.2f}"
        elif response.partial_exit_pct is not None:
            extra = f" | partial exit={response.partial_exit_pct:.0f}%"
        log.info(
            "AI insight [%s] %s — %s: %s%s",
            self.llm.name,
            plan.symbol,
            response.decision.value,
            response.reason,
            extra,
        )
        self.trade_logger.log(
            "AI_REVIEW", plan.symbol, self.llm.name,
            response.reason,
            {"decision": response.decision.value, "divergence": divergence_reasons},
        )
        self._execute_review(plan, response.decision, response)

    def _ai_cooldown_elapsed(self, plan: TradePlan) -> bool:
        if plan.last_ai_review_at is None:
            return True
        elapsed = (datetime.now(timezone.utc) - plan.last_ai_review_at).total_seconds() / 60
        return elapsed >= self.cfg.ai_review_cooldown_minutes

    def _execute_review(self, plan: TradePlan, decision: ReviewDecision, response) -> None:
        if decision == ReviewDecision.HOLD:
            log.info("AI action [%s] %s — no change (HOLD)", self.llm.name, plan.symbol)
            return
        if decision == ReviewDecision.TIGHTEN_SL:
            if response.new_stop_loss:
                log.info(
                    "AI action [%s] %s — tightening SL to %.2f",
                    self.llm.name,
                    plan.symbol,
                    response.new_stop_loss,
                )
                if plan.sl_order_id:
                    self.broker.modify_order(plan.sl_order_id, sl=response.new_stop_loss)
                plan.stop_loss = response.new_stop_loss
            else:
                log.info("AI action [%s] %s — moving SL to breakeven", self.llm.name, plan.symbol)
                self.executor.modify_sl_breakeven(plan)
            return
        if decision == ReviewDecision.PARTIAL_EXIT:
            pct = response.partial_exit_pct or 50.0
            log.info("AI action [%s] %s — partial exit %.0f%%", self.llm.name, plan.symbol, pct)
            self.executor.partial_exit(plan, pct)
            return
        if decision == ReviewDecision.CLOSE:
            log.info("AI action [%s] %s — closing position", self.llm.name, plan.symbol)
            self.trailing.unregister(plan.plan_id)
            self.executor.close_plan(plan, reason=response.reason)
