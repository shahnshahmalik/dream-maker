"""Session tracker — daily trade cap, P&L, consecutive loss guard.

Enforces the "1-4 high-precision scalps per day" discipline:
- Hard cap on total trades per day
- Stop when daily profit target is hit
- Stop after N consecutive losses
- Reduces size after a loss
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

from config import Config

log = logging.getLogger("dream_maker.session")


class SessionState(str, Enum):
    ACTIVE = "active"           # normal trading
    PROFIT_TARGET_HIT = "profit_target"  # daily goal reached — stop
    LOSS_GUARD = "loss_guard"   # consecutive loss limit hit — stop
    TRADE_CAP = "trade_cap"     # max trades/day reached — stop


@dataclass
class SessionTracker:
    """Tracks daily session: trade count, P&L, consecutive losses.

    Single responsibility: enforce the daily trading discipline.
    Does NOT modify plans or place orders — only answers "should we trade?"
    """

    cfg: Config
    _trade_count: int = 0
    _consecutive_losses: int = 0
    _daily_pnl: float = 0.0
    _initial_capital: float = 0.0
    _state: SessionState = SessionState.ACTIVE
    _stop_reason: str = ""
    _today: str = ""  # ISO date string for day-change detection
    _last_trade_pnl: float = 0.0
    # Per-setup scoring: track the best setup seen today
    _best_setup_score: float = 0.0
    _best_setup_symbol: str = ""

    def _check_day_rollover(self) -> None:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self._today:
            self._trade_count = 0
            self._consecutive_losses = 0
            self._daily_pnl = 0.0
            self._state = SessionState.ACTIVE
            self._stop_reason = ""
            self._today = today
            self._last_trade_pnl = 0.0
            self._best_setup_score = 0.0
            self._best_setup_symbol = ""
            log.info("New trading day — session reset")

    def set_initial_capital(self, capital: float) -> None:
        if self._initial_capital <= 0:
            self._initial_capital = capital

    @property
    def trade_count(self) -> int:
        self._check_day_rollover()
        return self._trade_count

    @property
    def daily_pnl(self) -> float:
        self._check_day_rollover()
        return self._daily_pnl

    @property
    def state(self) -> SessionState:
        self._check_day_rollover()
        return self._state

    @property
    def stop_reason(self) -> str:
        return self._stop_reason

    @property
    def can_trade(self) -> bool:
        """Can we open a new trade right now?"""
        self._check_day_rollover()
        return self._state == SessionState.ACTIVE

    @property
    def size_multiplier(self) -> float:
        """Return position size multiplier based on recent performance.
        
        1 win → 1.0x (full size)
        1 loss → 0.5x (half size for recovery)
        2+ wins → 1.0x
        """
        if self._consecutive_losses >= 1:
            return 0.5
        return 1.0

    def register_entry(self, symbol: str, risk_amount: float) -> None:
        """Called when a new trade is entered."""
        self._check_day_rollover()
        self._trade_count += 1
        if self._trade_count >= self.cfg.max_trades_per_day:
            self._state = SessionState.TRADE_CAP
            self._stop_reason = (
                f"Daily trade cap reached ({self._trade_count}/{self.cfg.max_trades_per_day})"
            )
            log.info("Session STOP: %s", self._stop_reason)

    def register_close(self, *, pnl: float) -> None:
        """Called when a trade is closed. Updates P&L, consecutive loss counter."""
        self._check_day_rollover()
        self._daily_pnl += pnl
        self._last_trade_pnl = pnl

        if pnl > 0:
            self._consecutive_losses = 0
        else:
            self._consecutive_losses += 1

        # Check daily profit target
        if self._initial_capital > 0:
            profit_pct = (self._daily_pnl / self._initial_capital) * 100
            if profit_pct >= self.cfg.daily_profit_target_pct:
                self._state = SessionState.PROFIT_TARGET_HIT
                self._stop_reason = (
                    f"Daily profit target hit: +{profit_pct:.1f}% "
                    f"(₹{self._daily_pnl:.0f} on ₹{self._initial_capital:.0f})"
                )
                log.info("Session STOP: %s", self._stop_reason)
                return

        # Check daily loss limit (₹ based, not count based)
        if self._daily_pnl <= -self.cfg.daily_loss_limit_inr:
            self._state = SessionState.LOSS_GUARD
            self._stop_reason = (
                f"Daily loss limit hit: -₹{abs(self._daily_pnl):.0f} "
                f"(limit ₹{self.cfg.daily_loss_limit_inr})"
            )
            log.info("Session STOP: %s", self._stop_reason)

    def track_setup_score(self, symbol: str, score: float) -> None:
        """Track the best setup seen today. Used for quality gating."""
        if score > self._best_setup_score:
            self._best_setup_score = score
            self._best_setup_symbol = symbol

    def is_best_setup_so_far(self, symbol: str, score: float) -> bool:
        """Is this setup at least as good as the best we've seen?"""
        return score >= self._best_setup_score * 0.95  # within 5% of best

    def status_summary(self) -> str:
        """One-line status for logging/telegram."""
        self._check_day_rollover()
        pnl_pct = (self._daily_pnl / self._initial_capital * 100) if self._initial_capital > 0 else 0
        return (
            f"Session: {self._trade_count}/{self.cfg.max_trades_per_day} trades | "
            f"P&L: ₹{self._daily_pnl:+.0f} ({pnl_pct:+.1f}%) | "
            f"Loss limit: ₹{self.cfg.daily_loss_limit_inr} | "
            f"State: {self._state.value}"
        )
