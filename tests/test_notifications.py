"""Notification formatting tests."""

from dataclasses import replace
from unittest.mock import MagicMock, patch

from config import Config, load_config
from notifications.service import NotificationService


def _cfg(**overrides) -> Config:
    base = load_config()
    return replace(base, **overrides)


def test_ai_setup_message():
    svc = NotificationService(_cfg(telegram_enabled=True, telegram_bot_token="t", telegram_chat_id="1"))
    msg = svc._format_ai_review(
        "NIFTY50IDX",
        "deepseek",
        "Momentum aligned",
        {
            "phase": "setup",
            "approved": True,
            "markers": {
                "entry_price_low": 24480.0,
                "entry_price_high": 24520.0,
                "stop_loss": 24400.0,
                "take_profit_1": 24800.0,
                "take_profit_2": 24950.0,
                "rr_ratio": 3.0,
            },
        },
    )
    assert msg is not None
    assert "AI LEVELS" in msg
    assert "Momentum aligned" in msg
    assert "24400" in msg


def test_trade_entry_message():
    svc = NotificationService(_cfg(simulation_mode=True, telegram_enabled=True))
    msg = svc._format_order(
        "NIFTY50IDX",
        "dhan",
        "Bracketed entry",
        {
            "success": True,
            "order_id": "ORD1",
            "simulation": True,
            "plan": {
                "direction": "LONG",
                "position_size": 25,
                "entry_price": 24500.0,
                "entry_price_low": 24480.0,
                "entry_price_high": 24520.0,
                "stop_loss": 24400.0,
                "take_profit_1": 24800.0,
                "rr_ratio": 3.0,
            },
        },
    )
    assert msg is not None
    assert "TRADE ENTRY" in msg
    assert "[SIM]" in msg


def test_on_audit_log_sends_to_telegram():
    svc = NotificationService(_cfg(telegram_enabled=True, telegram_bot_token="tok", telegram_chat_id="99"))
    svc.telegram.send = MagicMock()
    svc.on_audit_log(
        "AI_REVIEW",
        "NIFTY50IDX",
        "deepseek",
        "Setup weak",
        {"phase": "setup", "approved": False},
    )
    svc.telegram.send.assert_called_once()
    assert "AI PAUSED" in svc.telegram.send.call_args[0][0]


@patch("notifications.telegram.httpx.post")
def test_telegram_send(mock_post):
    from notifications.telegram import TelegramNotifier

    mock_post.return_value.raise_for_status = MagicMock()
    mock_post.return_value.status_code = 200
    notifier = TelegramNotifier("token", "123")
    notifier.send("hello")
    mock_post.assert_called_once()
    assert mock_post.call_args.kwargs["json"]["chat_id"] == "123"
