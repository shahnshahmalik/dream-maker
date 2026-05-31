"""One-time AI trade setup — markers and levels, not per-tick review."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from config import Config
from llm.base import LLMProvider, RuleBasedLLMProvider
from models.trade_plan import PlanStatus, TradePlan
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
approve=false if setup is weak. Minimum R:R is 1:3. Never suggest naked entries."""


class TradePlanner:
    """Runs AI analysis once per setup to refine price markers."""

    def __init__(self, llm: LLMProvider, cfg: Config, trade_logger: TradeLogger):
        self.llm = llm
        self.cfg = cfg
        self.trade_logger = trade_logger
        self._last_setup_at: datetime | None = None

    def can_run_ai_setup(self) -> bool:
        if self._last_setup_at is None:
            return True
        elapsed = (datetime.now(timezone.utc) - self._last_setup_at).total_seconds() / 60
        return elapsed >= self.cfg.ai_analysis_cooldown_minutes

    def apply_ai_markers(self, plan: TradePlan, tech_summary: dict) -> TradePlan:
        if plan.ai_setup_done:
            return plan
        if not self.can_run_ai_setup():
            log.info("AI setup on cooldown — using technical markers only")
            self._apply_default_markers(plan)
            plan.ai_setup_done = True
            return plan

        if isinstance(self.llm, RuleBasedLLMProvider):
            self._apply_default_markers(plan)
            plan.ai_setup_done = True
            return plan

        user_msg = json.dumps({"plan": plan.to_dict(), "technical": tech_summary})
        try:
            from llm.chat_completions import chat_completion, chat_config_for_provider

            chat_cfg = chat_config_for_provider(self.cfg, self.llm)
            if not chat_cfg:
                raise RuntimeError("LLM provider does not support chat setup")
            base_url, api_key, model = chat_cfg
            if not api_key:
                raise RuntimeError("LLM API key missing")

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
            log.warning("AI setup failed (%s) — using technical markers", e)
            self._apply_default_markers(plan)
            plan.ai_setup_done = True
            return plan

        self._last_setup_at = datetime.now(timezone.utc)
        if not data.get("approve", False):
            plan.status = PlanStatus.PAUSED
            plan.rationale = str(data.get("reason", "AI rejected setup"))
            self.trade_logger.log(
                "AI_REVIEW", plan.symbol, self.llm.name, plan.rationale, {"phase": "setup", "approved": False}
            )
            return plan

        entry = plan.entry_target()
        tol = self.cfg.entry_zone_tolerance_pct / 100.0
        plan.entry_price_low = float(data.get("entry_low", entry * (1 - tol)))
        plan.entry_price_high = float(data.get("entry_high", entry * (1 + tol)))
        plan.stop_loss = float(data.get("stop_loss", plan.stop_loss))
        plan.take_profit_1 = float(data.get("take_profit_1", plan.take_profit_1))
        plan.take_profit_2 = float(data.get("take_profit_2", plan.take_profit_2))
        sl_dist = abs(entry - plan.stop_loss)
        tp_dist = abs(plan.take_profit_1 - entry)
        plan.rr_ratio = tp_dist / sl_dist if sl_dist else plan.rr_ratio
        plan.ai_setup_done = True
        self._apply_default_markers(plan, overwrite=False)
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
