"""Position monitoring — rule-first, AI only on cooldown."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from agent.executor import TradeExecutor
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
    ):
        self.broker = broker
        self.llm = llm
        self.executor = executor
        self.cfg = cfg
        self.trade_logger = trade_logger

    def tick(self, plans: list[TradePlan]) -> None:
        active = [p for p in plans if p.status == PlanStatus.ACTIVE]
        for plan in active:
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

        candles_15m = self.broker.get_ohlcv(plan.symbol, "15m", 20)
        candles_1h = self.broker.get_ohlcv(plan.symbol, "1h", 20)
        divergence_reasons: list[str] = []
        moving_as_expected = favorable > adverse

        if sl_dist > 0 and adverse / sl_dist > 0.5:
            divergence_reasons.append("Price moved >50% toward SL")

        if check_ltf_structure_break(candles_15m, plan.direction):
            divergence_reasons.append("LTF structure break")

        if plan.direction == TradeDirection.LONG and ltp < plan.stop_loss * 1.01:
            divergence_reasons.append("Near HTF invalidation/support breach")
        elif plan.direction == TradeDirection.SHORT and ltp > plan.stop_loss * 0.99:
            divergence_reasons.append("Near HTF invalidation/resistance breach")

        if not plan.tp1_hit:
            tp1_hit = (
                (plan.direction == TradeDirection.LONG and ltp >= plan.take_profit_1)
                or (plan.direction == TradeDirection.SHORT and ltp <= plan.take_profit_1)
            )
            if tp1_hit:
                plan.tp1_hit = True
                self.executor.modify_sl_breakeven(plan)
                self.trade_logger.log(
                    "MONITOR", plan.symbol, self.broker.name,
                    "TP1 hit — SL moved to breakeven",
                    {"ltp": ltp, "tp1": plan.take_profit_1},
                )

        self.trade_logger.log(
            "MONITOR", plan.symbol, self.broker.name,
            f"LTP={ltp:.2f} expected={moving_as_expected} adverse={adverse:.2f}",
            {"divergence": divergence_reasons, "moving_as_expected": moving_as_expected},
        )

        if not divergence_reasons:
            return

        if sl_dist > 0 and adverse / sl_dist > 0.75:
            self.executor.close_plan(plan, reason=">75% toward SL — rule-based exit")
            return

        if not self._ai_cooldown_elapsed(plan):
            log.info("Divergence detected but AI review on cooldown — holding")
            return

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
            return
        if decision == ReviewDecision.TIGHTEN_SL:
            if response.new_stop_loss:
                if plan.sl_order_id:
                    self.broker.modify_order(plan.sl_order_id, sl=response.new_stop_loss)
                plan.stop_loss = response.new_stop_loss
            else:
                self.executor.modify_sl_breakeven(plan)
            return
        if decision == ReviewDecision.PARTIAL_EXIT:
            pct = response.partial_exit_pct or 50.0
            self.executor.partial_exit(plan, pct)
            return
        if decision == ReviewDecision.CLOSE:
            self.executor.close_plan(plan, reason=response.reason)
