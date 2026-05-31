"""Trading limits — daily loss, max trades, API error halt."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from utils.market_hours import is_event_blackout


@dataclass
class LimitsState:
    halted: bool = False
    halt_reason: str = ""
    daily_pnl_pct: float = 0.0
    trades_today: int = 0
    consecutive_api_errors: int = 0
    session_date: date = field(default_factory=date.today)


class LimitsGuard:
    def __init__(
        self,
        max_open_trades: int,
        daily_loss_limit: float,
        event_blackout_minutes: int,
        scheduled_events: list[str],
    ):
        self.max_open_trades = max_open_trades
        self.daily_loss_limit = daily_loss_limit
        self.event_blackout_minutes = event_blackout_minutes
        self.scheduled_events = scheduled_events
        self.state = LimitsState()

    def _roll_session(self) -> None:
        today = date.today()
        if self.state.session_date != today:
            self.state = LimitsState(session_date=today)

    def can_open_trade(self, open_count: int) -> tuple[bool, str]:
        self._roll_session()
        if self.state.halted:
            return False, self.state.halt_reason
        if open_count >= self.max_open_trades:
            return False, f"Max open trades ({self.max_open_trades}) reached"
        blocked, reason = is_event_blackout(
            self.scheduled_events, self.event_blackout_minutes
        )
        if blocked:
            return False, reason
        if self.state.daily_pnl_pct <= -self.daily_loss_limit:
            self.state.halted = True
            self.state.halt_reason = f"Daily loss limit {self.daily_loss_limit}% breached"
            return False, self.state.halt_reason
        return True, "ok"

    def record_trade_opened(self) -> None:
        self._roll_session()
        self.state.trades_today += 1

    def record_pnl(self, pnl_pct: float) -> None:
        self._roll_session()
        self.state.daily_pnl_pct += pnl_pct
        if self.state.daily_pnl_pct <= -self.daily_loss_limit:
            self.state.halted = True
            self.state.halt_reason = f"Daily loss limit {self.daily_loss_limit}% breached"

    def record_api_error(self) -> bool:
        self.state.consecutive_api_errors += 1
        if self.state.consecutive_api_errors >= 3:
            self.state.halted = True
            self.state.halt_reason = "3 consecutive API errors"
            return True
        return False

    def record_api_success(self) -> None:
        self.state.consecutive_api_errors = 0
