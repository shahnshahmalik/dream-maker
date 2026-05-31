"""LLM provider abstract base and rule-based fallback."""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod

from models.review import AIReviewRequest, AIReviewResponse, ReviewDecision


class LLMProvider(ABC):
    name: str = "base"

    @abstractmethod
    def review_trade(self, request: AIReviewRequest) -> AIReviewResponse: ...


def parse_review_json(text: str) -> AIReviewResponse:
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object in LLM response")
    data = json.loads(match.group())
    decision = ReviewDecision(str(data.get("decision", "HOLD")).upper())
    return AIReviewResponse(
        decision=decision,
        reason=str(data.get("reason", "")),
        new_stop_loss=float(data["new_stop_loss"]) if data.get("new_stop_loss") else None,
        partial_exit_pct=float(data["partial_exit_pct"]) if data.get("partial_exit_pct") else None,
        raw=data,
    )


class RuleBasedLLMProvider(LLMProvider):
    name = "none"

    def review_trade(self, request: AIReviewRequest) -> AIReviewResponse:
        sl_dist = abs(request.entry_price - request.stop_loss)
        adverse = abs(request.current_price - request.entry_price)
        if request.direction.upper() == "LONG" and request.current_price < request.entry_price:
            adverse = request.entry_price - request.current_price
        elif request.direction.upper() == "SHORT" and request.current_price > request.entry_price:
            adverse = request.current_price - request.entry_price

        if sl_dist > 0 and adverse / sl_dist > 0.5:
            return AIReviewResponse(
                decision=ReviewDecision.CLOSE,
                reason="Price moved >50% toward SL (rule-based fallback)",
            )
        if request.divergence_reasons:
            return AIReviewResponse(
                decision=ReviewDecision.TIGHTEN_SL,
                reason="Structure divergence detected; tighten SL to breakeven",
                new_stop_loss=request.entry_price,
            )
        return AIReviewResponse(decision=ReviewDecision.HOLD, reason="No action required")


REVIEW_SYSTEM_PROMPT = """You are a disciplined trading risk manager. Given trade context, respond ONLY with JSON:
{"decision": "HOLD|TIGHTEN_SL|PARTIAL_EXIT|CLOSE", "reason": "...", "new_stop_loss": null or number, "partial_exit_pct": null or number}
Never override stop-loss without justification. Capital preservation first."""
