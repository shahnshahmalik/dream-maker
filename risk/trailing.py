"""Trailing SL + TP manager — extends targets when trade moves as expected."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from models.trade_plan import TradeDirection


@dataclass
class TrailUpdate:
    new_sl: float | None = None
    new_tp: float | None = None
    new_tp2: float | None = None
    tp1_milestone: bool = False
    reason: str = ""


@dataclass
class TrailingState:
    peak: float
    current_sl: float
    current_tp: float
    current_tp2: float
    breakeven_done: bool = False
    armed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "peak": self.peak,
            "current_sl": self.current_sl,
            "current_tp": self.current_tp,
            "current_tp2": self.current_tp2,
            "breakeven_done": self.breakeven_done,
            "armed": self.armed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, fallback_entry: float, plan_sl: float, plan_tp1: float, plan_tp2: float) -> TrailingState:
        return cls(
            peak=float(data.get("peak", fallback_entry)),
            current_sl=float(data.get("current_sl", plan_sl)),
            current_tp=float(data.get("current_tp", plan_tp1)),
            current_tp2=float(data.get("current_tp2", plan_tp2)),
            breakeven_done=bool(data.get("breakeven_done", False)),
            armed=bool(data.get("armed", False)),
        )


class TrailingBracketManager:
    """Trails both SL and TP in the direction of profit once trade moves favorably."""

    def __init__(
        self,
        *,
        direction: TradeDirection,
        entry: float,
        initial_sl: float,
        initial_tp1: float,
        initial_tp2: float,
        trail_activate_pct: float,
        trail_sl_distance_pct: float,
        trail_tp_reward_pct: float,
        breakeven_progress_pct: float,
        breakeven_buffer_pct: float,
        state: TrailingState | None = None,
    ):
        self.direction = direction
        self.entry = float(entry)
        self.initial_sl = float(initial_sl)
        self.initial_tp1 = float(initial_tp1)
        self.initial_tp2 = float(initial_tp2)
        self.trail_activate = max(0.0, trail_activate_pct) / 100.0
        self.trail_sl_distance = max(0.0, trail_sl_distance_pct) / 100.0
        self.trail_tp_reward = max(0.0, trail_tp_reward_pct) / 100.0
        self.be_progress = max(0.0, min(100.0, breakeven_progress_pct)) / 100.0
        self.be_buffer = max(0.0, breakeven_buffer_pct) / 100.0

        self.tp1_distance = abs(initial_tp1 - entry)
        self.tp2_distance = abs(initial_tp2 - entry)

        if state:
            self.state = state
        else:
            self.state = TrailingState(
                peak=entry,
                current_sl=initial_sl,
                current_tp=initial_tp1,
                current_tp2=initial_tp2,
            )

    @property
    def is_long(self) -> bool:
        return self.direction == TradeDirection.LONG

    def on_price(self, ltp: float) -> TrailUpdate:
        if ltp <= 0:
            return TrailUpdate()

        self._update_peak(ltp)
        profit_pct = self._profit_pct()

        if not self._moving_as_expected(ltp):
            return TrailUpdate()

        update = TrailUpdate()

        progress = self._progress_toward_tp1(ltp)

        if not self.state.breakeven_done:
            if self.be_progress > 0 and self.tp1_distance > 0 and progress + 1e-9 >= self.be_progress:
                be_sl = self._breakeven_sl()
                if self._sl_improves(be_sl):
                    self.state.current_sl = be_sl
                    self.state.breakeven_done = True
                    update.new_sl = be_sl
                    update.reason = (
                        f"breakeven SL at cost {be_sl:.2f} "
                        f"({min(progress, 1.0) * 100:.0f}% of move toward TP1)"
                    )
                    if self._tp1_crossed(ltp):
                        update.tp1_milestone = True
                    return update
                self.state.breakeven_done = True
            return update

        if profit_pct < self.trail_activate:
            return update

        self.state.armed = True
        candidate_sl = self._trail_sl()
        candidate_tp = self._trail_tp(self.tp1_distance)
        candidate_tp2 = self._trail_tp(self.tp2_distance)

        moved = False
        if self._sl_improves(candidate_sl):
            self.state.current_sl = candidate_sl
            update.new_sl = candidate_sl
            moved = True

        if self._tp_improves(candidate_tp):
            self.state.current_tp = candidate_tp
            update.new_tp = candidate_tp
            moved = True

        if self._tp_improves(candidate_tp2, secondary=True):
            self.state.current_tp2 = candidate_tp2
            update.new_tp2 = candidate_tp2
            moved = True

        if self._tp1_crossed(ltp):
            update.tp1_milestone = True

        if moved:
            update.reason = (
                f"trail peak={self.state.peak:.2f} profit=+{profit_pct * 100:.2f}% "
                f"SL={self.state.current_sl:.2f} TP={self.state.current_tp:.2f}"
            )

        return update

    def _update_peak(self, ltp: float) -> None:
        if self.is_long:
            self.state.peak = max(self.state.peak, ltp)
        else:
            self.state.peak = min(self.state.peak, ltp)

    def _profit_pct(self) -> float:
        if self.is_long:
            return (self.state.peak - self.entry) / self.entry if self.entry else 0.0
        return (self.entry - self.state.peak) / self.entry if self.entry else 0.0

    def _moving_as_expected(self, ltp: float) -> bool:
        if self.is_long:
            return ltp >= self.entry or self.state.peak > self.entry
        return ltp <= self.entry or self.state.peak < self.entry

    def _progress_toward_tp1(self, ltp: float) -> float:
        """How far price has moved from entry toward TP1 (0.0 – 1.0+)."""
        if self.tp1_distance <= 0:
            return 0.0
        if self.is_long:
            return max(0.0, (ltp - self.entry) / self.tp1_distance)
        return max(0.0, (self.entry - ltp) / self.tp1_distance)

    def _breakeven_sl(self) -> float:
        """Stop at cost (entry) with optional fee buffer."""
        if self.is_long:
            return self.entry * (1 + self.be_buffer)
        return self.entry * (1 - self.be_buffer)

    def _trail_sl(self) -> float:
        if self.is_long:
            return self.state.peak * (1 - self.trail_sl_distance)
        return self.state.peak * (1 + self.trail_sl_distance)

    def _trail_tp(self, reward_distance: float) -> float:
        if self.trail_tp_reward > 0:
            dynamic = abs(self.state.peak - self.entry) * self.trail_tp_reward
            reward_distance = max(reward_distance, dynamic)
        if self.is_long:
            return self.state.peak + reward_distance
        return self.state.peak - reward_distance

    def _sl_improves(self, candidate: float) -> bool:
        if self.is_long:
            return candidate > self.state.current_sl
        return candidate < self.state.current_sl

    def _tp_improves(self, candidate: float, *, secondary: bool = False) -> bool:
        current = self.state.current_tp2 if secondary else self.state.current_tp
        if self.is_long:
            return candidate > current
        return candidate < current

    def _tp1_crossed(self, ltp: float) -> bool:
        if self.is_long:
            return ltp >= self.initial_tp1
        return ltp <= self.initial_tp1
