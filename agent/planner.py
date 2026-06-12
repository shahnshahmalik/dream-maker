"""One-time AI trade setup — markers and levels, not per-tick review."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from config import Config
from llm.base import LLMProvider, RuleBasedLLMProvider
from models.trade_plan import PlanStatus, TradePlan, validate_bracket, validate_plan
from audit.trade_logger import TradeLogger

log = logging.getLogger("dream_maker.planner")

SETUP_PROMPT = """You are a systematic F&O trade planner. Given technical context, respond ONLY with JSON:
{
  "approve": true|false,
  "reason": "...",
  "entry_low": number,
  "entry_high": number,
  "stop_loss": number,
  "take_profit_1": number,
  "take_profit_2": number
}
approve=false if setup is weak. Swing setups need R:R >= 1:2. Momentum scalps need R:R >= 1:1.2 with tight SL.
Never suggest naked entries — stop_loss and take_profit levels are mandatory."""


class TradePlanner:
    """Runs AI analysis once per setup to refine price markers."""

    def __init__(self, llm: LLMProvider, cfg: Config, trade_logger: TradeLogger):
        self.llm = llm
        self.cfg = cfg
        self.trade_logger = trade_logger
        self._last_setup_at: datetime | None = None

    @staticmethod
    def _log_levels(
        heading: str,
        plan: TradePlan,
        *,
        insight: str,
        provider: str,
    ) -> None:
        entry_lo = plan.entry_price_low if plan.entry_price_low is not None else plan.entry_target()
        entry_hi = plan.entry_price_high if plan.entry_price_high is not None else plan.entry_target()
        log.info(
            "%s [%s] %s\n"
            "  Insight : %s\n"
            "  Entry   : %.2f – %.2f\n"
            "  SL      : %.2f (%s)\n"
            "  TP1     : %.2f (%.0f%% exit)\n"
            "  TP2     : %.2f (%.0f%% exit)\n"
            "  R:R     : 1:%.2f | Signal %.2f",
            heading,
            provider,
            plan.symbol,
            insight,
            entry_lo,
            entry_hi,
            plan.stop_loss,
            plan.stop_loss_reason,
            plan.take_profit_1,
            plan.tp1_exit_pct,
            plan.take_profit_2,
            plan.tp2_exit_pct,
            plan.rr_ratio,
            plan.signal_strength,
        )

    def can_run_ai_setup(self) -> bool:
        if self._last_setup_at is None:
            return True
        elapsed = (datetime.now(timezone.utc) - self._last_setup_at).total_seconds() / 60
        return elapsed >= self.cfg.ai_analysis_cooldown_minutes

    def apply_ai_markers(self, plan: TradePlan, tech_summary: dict) -> TradePlan:
        if plan.ai_setup_done:
            return plan
        if not self.can_run_ai_setup():
            log.info("AI setup on cooldown — using technical markers only for %s", plan.symbol)
            self._apply_default_markers(plan)
            plan.ai_setup_done = True
            self._log_levels(
                "Technical levels (AI cooldown)",
                plan,
                insight=plan.rationale,
                provider="rules",
            )
            return plan

        if isinstance(self.llm, RuleBasedLLMProvider):
            log.info("AI disabled — applying technical levels for %s", plan.symbol)
            self._apply_default_markers(plan)
            plan.ai_setup_done = True
            self._log_levels(
                "Technical levels",
                plan,
                insight=plan.rationale,
                provider=self.llm.name,
            )
            return plan

        user_msg = json.dumps({"plan": plan.to_dict(), "technical": tech_summary})
        # When BUY_ONLY (low balance), SHORT+PE = BUY put (bearish), LONG+CE = BUY call (bullish).
        # The AI needs to know the actual market bias, not the order side.
        # Also mirror SL/TP levels: plan stores SHORT orientation (SL above, TP below),
        # but a BUY PE is LONG the option (SL below, TP above).
        if plan.symbol.upper().endswith("PE") and plan.direction.value == "SHORT":
            plan_dict = plan.to_dict()
            entry = plan.entry_target()
            # Mirror SL/TP around entry for the AI's context
            plan_dict["stop_loss"] = entry - abs(entry - plan.stop_loss)
            plan_dict["take_profit_1"] = entry + abs(entry - plan.take_profit_1)
            plan_dict["take_profit_2"] = entry + abs(entry - plan.take_profit_2)
            user_msg = json.dumps({
                "plan": plan_dict,
                "technical": tech_summary,
                "_note": "This account is BUY_ONLY. SHORT direction + PE symbol = BUYING a put (bearish bet). "
                         "The SL/TP levels shown are already mirrored for a LONG option position. "
                         "SL below entry, TP above entry — the put gains value when the underlying drops.",
            })
        try:
            from llm.chat_completions import chat_completion, chat_config_for_provider

            chat_cfg = chat_config_for_provider(self.cfg, self.llm)
            if not chat_cfg:
                raise RuntimeError("LLM provider does not support chat setup")
            base_url, api_key, model = chat_cfg
            if not api_key:
                raise RuntimeError("LLM API key missing")

            log.info(
                "AI setup request (%s/%s) for %s — refining entry/SL/TP markers",
                self.llm.name,
                model,
                plan.symbol,
            )
            content = chat_completion(
                base_url=base_url,
                api_key=api_key,
                model=model,
                system=SETUP_PROMPT,
                user=user_msg,
                temperature=0.1,
            )
            data = json.loads(content[content.index("{") : content.rindex("}") + 1])
        except Exception as e:
            log.warning("AI setup failed (%s) — using technical markers for %s", e, plan.symbol)
            self._apply_default_markers(plan)
            plan.ai_setup_done = True
            self._log_levels(
                "Technical levels (AI fallback)",
                plan,
                insight=plan.rationale,
                provider="rules",
            )
            return plan

        self._last_setup_at = datetime.now(timezone.utc)
        if not data.get("approve", False):
            reason = str(data.get("reason", "AI rejected setup"))
            plan.status = PlanStatus.PAUSED
            plan.rationale = reason
            log.info(
                "AI rejected setup [%s] %s — %s",
                self.llm.name,
                plan.symbol,
                reason,
            )
            self.trade_logger.log(
                "AI_REVIEW", plan.symbol, self.llm.name, plan.rationale, {"phase": "setup", "approved": False}
            )
            return plan

        entry = plan.entry_target()
        tol = self.cfg.entry_zone_tolerance_pct / 100.0
        plan.entry_price_low = float(data.get("entry_low", entry * (1 - tol)))
        plan.entry_price_high = float(data.get("entry_high", entry * (1 + tol)))

        # Enforce minimum spread: AI sometimes returns entry_low == entry_high
        # (e.g., 184.00–184.00), making the zone impossible to hit.
        ai_spread = abs(plan.entry_price_high - plan.entry_price_low)
        min_spread = entry * tol * 0.5
        if ai_spread < min_spread:
            mid = (plan.entry_price_low + plan.entry_price_high) / 2
            plan.entry_price_low = mid * (1 - tol)
            plan.entry_price_high = mid * (1 + tol)
            log.info(
                "AI gave zero-width zone (%.2f–%.2f) — spread to [%.2f, %.2f]",
                float(data.get("entry_low", 0)), float(data.get("entry_high", 0)),
                plan.entry_price_low, plan.entry_price_high,
            )
        plan.stop_loss = float(data.get("stop_loss", plan.stop_loss))
        plan.take_profit_1 = float(data.get("take_profit_1", plan.take_profit_1))
        plan.take_profit_2 = float(data.get("take_profit_2", plan.take_profit_2))

        # Mirror SL/TP back: AI sees LONG-option levels (SL below, TP above),
        # but plan stores SHORT-orientation (SL above, TP below).
        if plan.symbol.upper().endswith("PE") and plan.direction.value == "SHORT":
            plan.stop_loss = entry + abs(entry - plan.stop_loss)
            plan.take_profit_1 = entry - abs(entry - plan.take_profit_1)
            plan.take_profit_2 = entry - abs(entry - plan.take_profit_2)
        sl_dist = abs(entry - plan.stop_loss)
        tp_dist = abs(plan.take_profit_1 - entry)
        plan.rr_ratio = tp_dist / sl_dist if sl_dist else plan.rr_ratio
        ok, reason = validate_plan(
            plan,
            min_rr=self.cfg.min_rr_ratio,
            min_rr_scalp=self.cfg.scalp_min_rr_ratio,
        )
        if not ok:
            plan.status = PlanStatus.PAUSED
            plan.rationale = f"AI levels rejected: {reason}"
            log.info("AI levels invalid [%s] %s — %s", self.llm.name, plan.symbol, reason)
            self.trade_logger.log(
                "AI_REVIEW", plan.symbol, self.llm.name, plan.rationale, {"phase": "setup", "approved": False}
            )
            return plan

        plan.ai_setup_done = True
        self._apply_default_markers(plan, overwrite=False)
        insight = str(data.get("reason", "AI markers applied"))
        plan.rationale = f"{plan.rationale}; AI: {insight}"
        self._log_levels(
            "AI levels approved",
            plan,
            insight=insight,
            provider=self.llm.name,
        )
        self.trade_logger.log(
            "AI_REVIEW",
            plan.symbol,
            self.llm.name,
            str(data.get("reason", "AI markers applied")),
            {"phase": "setup", "approved": True, "markers": plan.to_dict()},
        )
        return plan

    def _apply_default_markers(self, plan: TradePlan, *, overwrite: bool = True) -> None:
        entry = plan.entry_target()
        tol = self.cfg.entry_zone_tolerance_pct / 100.0
        if overwrite or plan.entry_price_low is None:
            plan.entry_price_low = entry * (1 - tol)
        if overwrite or plan.entry_price_high is None:
            plan.entry_price_high = entry * (1 + tol)
