"""LLM JSON parsing tests."""

import pytest

from llm.base import parse_review_json, RuleBasedLLMProvider
from models.review import AIReviewRequest, ReviewDecision


def test_parse_review_json():
    text = 'Some text {"decision": "CLOSE", "reason": "SL breached", "new_stop_loss": null}'
    resp = parse_review_json(text.replace("null", "null"))
    assert resp.decision == ReviewDecision.CLOSE
    assert "SL" in resp.reason


def test_parse_review_tighten_sl():
    text = '{"decision": "TIGHTEN_SL", "reason": "move to BE", "new_stop_loss": 100.5}'
    resp = parse_review_json(text)
    assert resp.decision == ReviewDecision.TIGHTEN_SL
    assert resp.new_stop_loss == 100.5


def test_parse_invalid_raises():
    with pytest.raises(ValueError):
        parse_review_json("no json here")


def test_rule_based_close_on_adverse_move():
    llm = RuleBasedLLMProvider()
    req = AIReviewRequest(
        symbol="RELIANCE",
        direction="LONG",
        original_rationale="test",
        entry_price=100,
        current_price=95,
        stop_loss=90,
        macro_env="NEUTRAL",
        candles_15m=[],
        candles_1h=[],
        divergence_reasons=["Price moved >50% toward SL"],
    )
    resp = llm.review_trade(req)
    assert resp.decision in {ReviewDecision.CLOSE, ReviewDecision.TIGHTEN_SL}
