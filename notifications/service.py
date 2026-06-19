"""Format and dispatch notifications for trades, AI insights, and indicator events.

Channels:
  - Telegram: trades + AI reviews + indicator events
  - Slack:    trades + indicator events (via incoming webhook)
"""

from __future__ import annotations

import logging
from typing import Any

from config import Config
from notifications.signal_notifier import SignalNotifier
from notifications.telegram import TelegramNotifier

log = logging.getLogger("dream_maker.notify")


class NotificationService:
    """Routes audit log events to Telegram and/or Slack when configured."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.telegram = TelegramNotifier(cfg.telegram_bot_token, cfg.telegram_chat_id)
        self.signal = SignalNotifier(cfg.signal_phone, cfg.signal_api_key)

        if self.telegram.enabled:
            log.info("Telegram notifications enabled for chat %s", cfg.telegram_chat_id)
        elif cfg.telegram_enabled:
            log.warning(
                "TELEGRAM_ENABLED is true but TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID is missing"
            )
        if self.signal.enabled:
            log.info("Signal notifications enabled for %s", cfg.signal_phone)

    @property
    def enabled(self) -> bool:
        return self.telegram.enabled or self.signal.enabled

    def _broadcast(self, msg: str, *, telegram: bool = True, signal: bool = True) -> None:
        """Send msg to all enabled channels."""
        if telegram and self.cfg.telegram_enabled and self.telegram.enabled:
            self.telegram.send(msg)
        if signal and self.signal.enabled:
            self.signal.send(msg)

    def on_audit_log(
        self,
        action: str,
        symbol: str,
        provider: str,
        reason: str,
        details: dict[str, Any],
    ) -> None:
        if action == "ORDER" and self.cfg.telegram_notify_trades:
            msg = self._format_order(symbol, provider, reason, details)
            if msg:
                self._broadcast(msg, signal=self.cfg.signal_notify_trades)
        elif action in {"CLOSE", "MODIFY"} and self.cfg.telegram_notify_trades:
            msg = self._format_trade_event(action, symbol, provider, reason, details)
            if msg:
                self._broadcast(msg, signal=self.cfg.signal_notify_trades)
        elif action == "AI_REVIEW" and self.cfg.telegram_notify_ai:
            msg = self._format_ai_review(symbol, provider, reason, details)
            if msg:
                # AI reviews go to Telegram only (too noisy for Signal)
                self._broadcast(msg, telegram=True, signal=False)

    def on_indicator_event(
        self,
        event_type: str,
        symbol: str,
        details: dict[str, Any],
    ) -> None:
        """Notify on indicator/strategy events — day classification, signal firing, rejection.

        event_type values:
          'day_classified'   — day type determined at OR close
          'strategy_routed'  — which strategy is active today
          'signal_fired'     — a setup passed all gates, plan created
          'signal_rejected'  — setup found but rejected (with reason)
          'bb_orb_signal'    — BB+ORB breakout triggered (Trend/Gap day)
          'sweep_signal'     — Stacked sweep triggered (V-Reversal day)
          'day_blocked'      — Range/Inside day, no trades today
        """
        if not self.cfg.signal_notify_indicators and not self.cfg.telegram_notify_ai:
            return

        msg = self._format_indicator_event(event_type, symbol, details)
        if not msg:
            return

        # Indicator events go to both Telegram and Signal
        self._broadcast(msg, telegram=True, signal=self.cfg.signal_notify_indicators)

    # ── Formatters ────────────────────────────────────────────────────────────

    def _sim_tag(self, details: dict[str, Any]) -> str:
        return " [SIM]" if (self.cfg.simulation_mode or details.get("simulation")) else ""

    @staticmethod
    def _plan_levels(plan: dict[str, Any]) -> str:
        entry_lo = plan.get("entry_price_low")
        entry_hi = plan.get("entry_price_high")
        entry = f"{entry_lo:.2f} – {entry_hi:.2f}" if entry_lo and entry_hi else plan.get("entry_zone", "—")
        sl   = plan.get("stop_loss")
        tp1  = plan.get("take_profit_1")
        tp2  = plan.get("take_profit_2")
        rr   = plan.get("rr_ratio")
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

    def _format_order(self, symbol, provider, reason, details) -> str | None:
        if not details.get("success"):
            return None
        plan = details.get("plan") or {}
        direction = plan.get("direction", "—")
        qty  = plan.get("position_size", "—")
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
        return "\n".join(str(l) for l in lines if l)

    def _format_trade_event(self, action, symbol, provider, reason, details) -> str | None:
        if action == "CLOSE":
            qty = details.get("qty", "—")
            return "\n".join([
                f"📉 TRADE CLOSED{self._sim_tag(details)}",
                symbol,
                f"Reason: {reason}",
                f"Qty   : {qty}",
                f"Broker: {provider}",
            ])
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

    def _format_ai_review(self, symbol, provider, reason, details) -> str | None:
        phase = details.get("phase")
        if phase == "setup":
            if details.get("approved"):
                plan = details.get("markers") or {}
                return "\n".join([
                    "🤖 AI LEVELS",
                    f"{symbol} | {provider}",
                    f"Insight: {reason}",
                    self._plan_levels(plan) if plan else "",
                ]).strip()
            return "\n".join(["🤖 AI PAUSED", symbol, f"Reason: {reason}"])

        decision   = details.get("decision", "HOLD")
        divergence = details.get("divergence") or []
        lines = ["🤖 AI INSIGHT", f"{symbol} | {decision}", f"Insight: {reason}"]
        if divergence:
            lines.append(f"Trigger: {'; '.join(str(d) for d in divergence)}")
        lines.append(f"Provider: {provider}")
        return "\n".join(lines)

    def _format_indicator_event(
        self, event_type: str, symbol: str, details: dict[str, Any]
    ) -> str | None:
        if event_type == "day_classified":
            day_type   = details.get("day_type", "unknown")
            direction  = details.get("allowed_direction", "any")
            or_range   = details.get("or_range_pct", 0.0)
            gap        = details.get("gap_pct", 0.0)
            emoji = {
                "v_reversal_bull":  "🔄",
                "v_reversal_bear":  "🔄",
                "trend_up":         "📈",
                "trend_down":       "📉",
                "gap_up_trend":     "⬆️",
                "gap_down_rally":   "↕️",
                "gap_down_trend":   "⬇️",
                "range_inside":     "⏸️",
                "unknown":          "❓",
            }.get(day_type, "📊")
            lines = [
                f"{emoji} DAY CLASSIFIED — {day_type.replace('_', ' ').upper()}",
                f"Symbol    : {symbol}",
                f"Direction : {direction}",
                f"OR range  : {or_range:.2f}%",
            ]
            if abs(gap) > 0.001:
                lines.append(f"Gap       : {gap:+.2f}%")
            strategy = details.get("strategy", "")
            if strategy:
                lines.append(f"Strategy  : {strategy}")
            return "\n".join(lines)

        if event_type == "day_blocked":
            reason = details.get("reason", "Range/Inside day")
            return "\n".join([
                f"⏸️ NO TRADES TODAY — {symbol}",
                f"Reason: {reason}",
            ])

        if event_type == "strategy_routed":
            strategy  = details.get("strategy", "—")
            day_type  = details.get("day_type", "—")
            direction = details.get("direction", "any")
            return "\n".join([
                f"🧭 STRATEGY ACTIVE — {strategy.upper().replace('_', ' ')}",
                f"Symbol    : {symbol}",
                f"Day type  : {day_type.replace('_', ' ')}",
                f"Direction : {direction}",
            ])

        if event_type == "sweep_signal":
            levels    = details.get("levels", [])
            strength  = details.get("strength", 0.0)
            direction = details.get("direction", "—")
            entry     = details.get("entry", 0.0)
            sl        = details.get("stop_loss", 0.0)
            tp1       = details.get("tp1", 0.0)
            return "\n".join([
                f"🌊 STACKED SWEEP — {symbol} {direction}",
                f"Levels   : {', '.join(levels)}",
                f"Strength : {strength:.2f}",
                f"Entry    : {entry:.2f}",
                f"SL       : {sl:.2f}",
                f"TP1      : {tp1:.2f}",
            ])

        if event_type == "bb_orb_signal":
            direction = details.get("direction", "—")
            entry     = details.get("entry", 0.0)
            sl        = details.get("stop_loss", 0.0)
            tp1       = details.get("tp1", 0.0)
            bb_reason = details.get("bb_reason", "")
            orb_reason= details.get("orb_reason", "")
            return "\n".join([
                f"📊 BB+ORB BREAKOUT — {symbol} {direction}",
                f"Entry : {entry:.2f}",
                f"SL    : {sl:.2f}",
                f"TP1   : {tp1:.2f}",
                f"BB    : {bb_reason}",
                f"ORB   : {orb_reason}",
            ])

        if event_type == "signal_fired":
            strategy  = details.get("strategy", "—")
            direction = details.get("direction", "—")
            score     = details.get("quality_score", 0.0)
            return "\n".join([
                f"✅ SIGNAL APPROVED — {symbol} {direction}",
                f"Strategy : {strategy}",
                f"Score    : {score:.2f}",
            ])

        if event_type == "signal_rejected":
            reason = details.get("reason", "—")
            score  = details.get("quality_score", 0.0)
            return "\n".join([
                f"❌ SIGNAL REJECTED — {symbol}",
                f"Reason : {reason}",
                f"Score  : {score:.2f}" if score else "",
            ]).strip()

        return None
