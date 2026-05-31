"""Format and dispatch notifications for trades and AI insights."""

from __future__ import annotations

import logging
from typing import Any

from config import Config
from notifications.telegram import TelegramNotifier

log = logging.getLogger("dream_maker.notify")


class NotificationService:
    """Routes audit log events to Telegram when configured."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.telegram = TelegramNotifier(cfg.telegram_bot_token, cfg.telegram_chat_id)
        if self.telegram.enabled:
            log.info("Telegram notifications enabled for chat %s", cfg.telegram_chat_id)
        elif cfg.telegram_enabled:
            log.warning(
                "TELEGRAM_ENABLED is true but TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID is missing"
            )

    @property
    def enabled(self) -> bool:
        return self.telegram.enabled

    def on_audit_log(
        self,
        action: str,
        symbol: str,
        provider: str,
        reason: str,
        details: dict[str, Any],
    ) -> None:
        if not self.cfg.telegram_enabled or not self.telegram.enabled:
            return
        if action == "ORDER" and self.cfg.telegram_notify_trades:
            msg = self._format_order(symbol, provider, reason, details)
        elif action in {"CLOSE", "MODIFY"} and self.cfg.telegram_notify_trades:
            msg = self._format_trade_event(action, symbol, provider, reason, details)
        elif action == "AI_REVIEW" and self.cfg.telegram_notify_ai:
            msg = self._format_ai_review(symbol, provider, reason, details)
        else:
            return
        if msg:
            self.telegram.send(msg)

    def _sim_tag(self, details: dict[str, Any]) -> str:
        if self.cfg.simulation_mode or details.get("simulation"):
            return " [SIM]"
        return ""

    @staticmethod
    def _plan_levels(plan: dict[str, Any]) -> str:
        entry_lo = plan.get("entry_price_low")
        entry_hi = plan.get("entry_price_high")
        entry = f"{entry_lo:.2f} – {entry_hi:.2f}" if entry_lo and entry_hi else plan.get("entry_zone", "—")
        sl = plan.get("stop_loss")
        tp1 = plan.get("take_profit_1")
        tp2 = plan.get("take_profit_2")
        rr = plan.get("rr_ratio")
        lines = [
            f"Entry : {entry}",
            f"SL    : {sl:.2f}" if sl else "SL    : —",
            f"TP1   : {tp1:.2f}" if tp1 else "TP1   : —",
        ]
        if tp2:
            lines.append(f"TP2   : {tp2:.2f}")
        if rr:
            lines.append(f"R:R   : 1:{float(rr):.2f}")
        return "\n".join(lines)

    def _format_order(
        self,
        symbol: str,
        provider: str,
        reason: str,
        details: dict[str, Any],
    ) -> str | None:
        if not details.get("success"):
            return None
        plan = details.get("plan") or {}
        direction = plan.get("direction", "—")
        qty = plan.get("position_size", "—")
        entry = plan.get("entry_price") or plan.get("entry_zone", "—")
        lines = [
            f"📈 TRADE ENTRY{self._sim_tag(details)}",
            f"{symbol} {direction} × {qty} @ {entry}",
            self._plan_levels(plan) if plan else reason,
            f"Broker: {provider}",
        ]
        order_id = details.get("order_id")
        if order_id:
            lines.append(f"Order : {order_id}")
        return "\n".join(str(line) for line in lines if line)

    def _format_trade_event(
        self,
        action: str,
        symbol: str,
        provider: str,
        reason: str,
        details: dict[str, Any],
    ) -> str | None:
        if action == "CLOSE":
            qty = details.get("qty", "—")
            return "\n".join(
                [
                    f"📉 TRADE CLOSED{self._sim_tag(details)}",
                    symbol,
                    f"Reason: {reason}",
                    f"Qty   : {qty}",
                    f"Broker: {provider}",
                ]
            )
        if action == "MODIFY":
            new_sl = details.get("new_sl")
            new_tp = details.get("new_tp")
            if new_sl is None and new_tp is None:
                return None
            parts = [f"🔧 TRAIL UPDATE{self._sim_tag(details)}", symbol]
            if new_sl is not None:
                parts.append(f"SL → {float(new_sl):.2f}")
            if new_tp is not None:
                parts.append(f"TP → {float(new_tp):.2f}")
            if reason:
                parts.append(reason)
            return "\n".join(parts)
        return None

    def _format_ai_review(
        self,
        symbol: str,
        provider: str,
        reason: str,
        details: dict[str, Any],
    ) -> str | None:
        phase = details.get("phase")
        if phase == "setup":
            if details.get("approved"):
                plan = details.get("markers") or {}
                return "\n".join(
                    [
                        "🤖 AI LEVELS",
                        f"{symbol} | {provider}",
                        f"Insight: {reason}",
                        self._plan_levels(plan) if plan else "",
                    ]
                ).strip()
            return "\n".join(
                [
                    "🤖 AI PAUSED",
                    symbol,
                    f"Reason: {reason}",
                ]
            )

        decision = details.get("decision", "HOLD")
        divergence = details.get("divergence") or []
        lines = [
            "🤖 AI INSIGHT",
            f"{symbol} | {decision}",
            f"Insight: {reason}",
        ]
        if divergence:
            lines.append(f"Trigger: {'; '.join(str(d) for d in divergence)}")
        lines.append(f"Provider: {provider}")
        return "\n".join(lines)
