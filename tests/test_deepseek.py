"""DeepSeek LLM provider tests."""

from llm.deepseek import DeepSeekProvider
from llm.factory import get_llm
from llm.base import RuleBasedLLMProvider


def test_factory_deepseek(monkeypatch):
    monkeypatch.setenv("TRADING_SYMBOL", "NIFTY50IDX")
    monkeypatch.setenv("ACTIVE_BROKER", "dhan")
    monkeypatch.setenv("ACTIVE_LLM", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")

    from config import load_config

    cfg = load_config()
    llm = get_llm(cfg)
    assert isinstance(llm, DeepSeekProvider)
    assert llm.name == "deepseek"
    assert llm._model == "deepseek-chat"


def test_deepseek_fallback_without_key(monkeypatch):
    monkeypatch.setenv("TRADING_SYMBOL", "NIFTY50IDX")
    monkeypatch.setenv("ACTIVE_BROKER", "dhan")
    monkeypatch.setenv("ACTIVE_LLM", "deepseek")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    from config import load_config
    from models.review import AIReviewRequest, ReviewDecision

    cfg = load_config()
    llm = DeepSeekProvider(cfg)
    resp = llm.review_trade(
        AIReviewRequest(
            symbol="NIFTY50IDX",
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
    )
    assert resp.decision in {ReviewDecision.CLOSE, ReviewDecision.TIGHTEN_SL, ReviewDecision.HOLD}
