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
        self._entered_at: dict[str, float] = {}  # plan_id → monotonic timestamp
        self._min_hold_seconds: float = 60.0  # no AI close within first 60s

    def tick(self, plans: list[TradePlan]) -> None:
        active = [p for p in plans if p.status == PlanStatus.ACTIVE]
        for plan in active:
            self.trailing.register(plan)
            self._monitor_one(plan)

    def _monitor_one(self, plan: TradePlan) -> None:
        quote = self.broker.get_quote(plan.symbol)
        ltp = quote.ltp
        entry = plan.entry_price or plan.entry_target()

        # LTP sanity check — Dhan returns spot/index level for far-OTM options
        # instead of the actual option premium. Fall back to OHLCV candles + estimation.
        if entry > 0:
            ratio = ltp / entry
            if ratio > 5 or ratio < 0.2:
                log.warning(
                    "LTP sanity check failed for %s: LTP=%.2f vs entry=%.2f (ratio=%.1fx) — "
                    "trying OHLCV fallback",
                    plan.symbol, ltp, entry, ratio,
                )
                ltp = self._fallback_ltp(plan, entry)
                if ltp <= 0:
                    log.warning(
                        "All LTP sources failed for %s — skipping monitor tick",
                        plan.symbol,
                    )
                    return
                log.info("Monitor using fallback LTP=%.2f for %s", ltp, plan.symbol)

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

        # ── Minimum hold: don't let AI close a fresh position on noise ──
        import time as _time
        now_mono = _time.monotonic()
        if plan.plan_id not in self._entered_at:
            self._entered_at[plan.plan_id] = now_mono
        age_seconds = now_mono - self._entered_at[plan.plan_id]

        # ── Minimum adverse threshold: <2% of entry is noise, not divergence ──
        min_adverse_pct = 0.02  # 2% of entry price
        min_adverse_absolute = entry * min_adverse_pct

        if moving_as_expected and self.cfg.trail_enabled:
            self.trailing.tick(plan, ltp)

        # Internal trailing SL: exit when LTP crosses the trail's SL (broker
        # bracket modifications are unreliable — the internal trail is the real stop).
        trail_state = plan.meta.get("trail", {})
        internal_sl = trail_state.get("current_sl")
        if internal_sl:
            entry = plan.entry_price or plan.entry_target()
            if internal_sl <= entry:
                # LONG position (SL at or below entry): exit when LTP drops to/below SL
                if ltp <= internal_sl:
                    log.warning(
                        "Internal trailing SL hit for %s: LTP %.2f <= SL %.2f — exiting",
                        plan.symbol, ltp, internal_sl,
                    )
                    self.trailing.unregister(plan.plan_id)
                    self.executor.close_plan(plan, reason=f"Internal trailing SL {internal_sl:.2f} hit")
                    return
            else:
                # SHORT position (SL above entry): exit when LTP rises above SL
                if ltp >= internal_sl:
                    log.warning(
                        "Internal trailing SL hit for %s: LTP %.2f >= SL %.2f — exiting",
                        plan.symbol, ltp, internal_sl,
                    )
                    self.trailing.unregister(plan.plan_id)
                    self.executor.close_plan(plan, reason=f"Internal trailing SL {internal_sl:.2f} hit")
                    return

        candles_15m = self.broker.get_ohlcv(plan.symbol, "15m", 20)
        candles_1h = self.broker.get_ohlcv(plan.symbol, "1h", 20)
        divergence_reasons: list[str] = []

        # Only flag divergences if adverse move exceeds minimum threshold (2% of entry)
        if sl_dist > 0 and adverse / sl_dist > 0.5 and adverse > min_adverse_absolute:
            divergence_reasons.append("Price moved >50% toward SL")

        if check_ltf_structure_break(candles_15m, plan.direction):
            divergence_reasons.append("LTF structure break")

        # Near-SL detection: only trigger if price within 3% of SL (was 1% — too twitchy)
        near_sl_buffer = 0.03
        if plan.direction == TradeDirection.LONG and ltp < plan.stop_loss * (1 + near_sl_buffer):
            divergence_reasons.append("Near HTF invalidation/support breach")
        elif plan.direction == TradeDirection.SHORT and ltp > plan.stop_loss * (1 - near_sl_buffer):
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

        # ── Minimum hold: don't let AI close a position within the first N seconds ──
        if age_seconds < self._min_hold_seconds:
            log.info(
                "Divergence on %s (%s) — holding (age %.0fs < %ss minimum)",
                plan.symbol,
                "; ".join(divergence_reasons),
                age_seconds,
                self._min_hold_seconds,
            )
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

    def _fallback_ltp(self, plan: TradePlan, entry: float) -> float:
        """Try to get a valid LTP when Dhan returns spot/index-level garbage for options.

        Fallback chain:
        1. Option's 5m OHLCV close (most reliable for actively traded options)
        2. Underlying index LTP → approximate option price via spot ratio
        3. Return 0 = give up
        """
        is_option = plan.symbol.upper().endswith(("CE", "PE"))
        if not is_option:
            return 0.0

        # 1. Try option OHLCV 5m close
        try:
            candles = self.broker.get_ohlcv(plan.symbol, "5m", 2)
            if candles:
                close = candles[-1].close
                # Validate: must be in the same order of magnitude as entry
                if entry > 0 and 0.1 < close / entry < 10:
                    return close
                log.debug(
                    "OHLCV fallback for %s: close=%.2f vs entry=%.2f — rejected (ratio=%.1fx)",
                    plan.symbol, close, entry, close / entry if entry else 0,
                )
        except Exception as e:
            log.debug("OHLCV fallback failed for %s: %s", plan.symbol, e)

        # 2. Underlying index spot ratio
        try:
            underlying = self._extract_underlying(plan.symbol)
            if underlying:
                spot = self.broker.get_index_spot(underlying)
                entry_spot = plan.meta.get("entry_spot")
                if entry_spot and entry_spot > 0 and spot > 0:
                    # Approximate: option moves roughly proportionally to underlying
                    # for near-ATM; for deep OTM use a dampened ratio
                    spot_ratio = spot / entry_spot
                    estimated = entry * spot_ratio
                    log.info(
                        "Spot-ratio LTP for %s: spot %.0f→%.0f (%.1f%%) → est=%.2f",
                        plan.symbol, entry_spot, spot, (spot_ratio - 1) * 100, estimated,
                    )
                    return estimated
        except Exception as e:
            log.debug("Spot-ratio fallback failed for %s: %s", plan.symbol, e)

        return 0.0

    @staticmethod
    def _extract_underlying(symbol: str) -> str | None:
        """Extract index underlying from option symbol. 'NIFTY26JUN23900PE' → 'NIFTY'."""
        import re
        m = re.match(r"^([A-Z]+)\d{2}[A-Z]{3}\d+[CP]E$", symbol.upper())
        return m.group(1) if m else None

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
