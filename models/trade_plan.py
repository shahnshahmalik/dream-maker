"""Trade plan model and validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4


class TradeDirection(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class EntryType(str, Enum):
    LIMIT = "LIMIT"
    STOP_ENTRY = "STOP_ENTRY"
    MARKET = "MARKET"


class PlanStatus(str, Enum):
    PAUSED = "PAUSED"
    WAITING_ENTRY = "WAITING_ENTRY"
    PENDING = "PENDING"
    ACTIVE = "ACTIVE"
    INVALIDATED = "INVALIDATED"
    CLOSED = "CLOSED"


@dataclass
class TradePlan:
    symbol: str
    direction: TradeDirection
    timeframe: str
    bias_source: str
    entry_zone: str
    entry_type: EntryType
    stop_loss: float
    stop_loss_reason: str
    take_profit_1: float
    tp1_exit_pct: float
    take_profit_2: float
    tp2_exit_pct: float
    risk_amount: float
    position_size: int
    rr_ratio: float
    status: PlanStatus = PlanStatus.PAUSED
    plan_id: str = field(default_factory=lambda: uuid4().hex[:12])
    entry_price: float | None = None
    entry_price_low: float | None = None
    entry_price_high: float | None = None
    strike_price: float | None = None
    order_id: str | None = None
    sl_order_id: str | None = None
    tp_order_id: str | None = None
    tp1_hit: bool = False
    signal_strength: float = 0.0
    macro_env: str = "NEUTRAL"
    rationale: str = ""
    is_intraday: bool = False
    ai_setup_done: bool = False
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_ai_review_at: datetime | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def entry_target(self) -> float:
        if self.entry_price is not None:
            return self.entry_price
        return float(self.entry_zone)

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "symbol": self.symbol,
            "direction": self.direction.value,
            "timeframe": self.timeframe,
            "bias_source": self.bias_source,
            "entry_zone": self.entry_zone,
            "entry_type": self.entry_type.value,
            "stop_loss": self.stop_loss,
            "stop_loss_reason": self.stop_loss_reason,
            "take_profit_1": self.take_profit_1,
            "tp1_exit_pct": self.tp1_exit_pct,
            "take_profit_2": self.take_profit_2,
            "tp2_exit_pct": self.tp2_exit_pct,
            "risk_amount": self.risk_amount,
            "position_size": self.position_size,
            "rr_ratio": self.rr_ratio,
            "status": self.status.value,
            "entry_price": self.entry_price,
            "entry_price_low": self.entry_price_low,
            "entry_price_high": self.entry_price_high,
            "strike_price": self.strike_price,
            "order_id": self.order_id,
            "sl_order_id": self.sl_order_id,
            "tp_order_id": self.tp_order_id,
            "tp1_hit": self.tp1_hit,
            "signal_strength": self.signal_strength,
            "macro_env": self.macro_env,
            "rationale": self.rationale,
            "is_intraday": self.is_intraday,
            "ai_setup_done": self.ai_setup_done,
            "created_at": self.created_at.isoformat(),
            "last_ai_review_at": self.last_ai_review_at.isoformat() if self.last_ai_review_at else None,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TradePlan:
        created = data.get("created_at")
        last_ai = data.get("last_ai_review_at")
        return cls(
            symbol=data["symbol"],
            direction=TradeDirection(data["direction"]),
            timeframe=data.get("timeframe", "15m"),
            bias_source=data.get("bias_source", ""),
            entry_zone=data.get("entry_zone", "0"),
            entry_type=EntryType(data.get("entry_type", "LIMIT")),
            stop_loss=float(data["stop_loss"]),
            stop_loss_reason=data.get("stop_loss_reason", ""),
            take_profit_1=float(data["take_profit_1"]),
            tp1_exit_pct=float(data.get("tp1_exit_pct", 50)),
            take_profit_2=float(data["take_profit_2"]),
            tp2_exit_pct=float(data.get("tp2_exit_pct", 50)),
            risk_amount=float(data.get("risk_amount", 0)),
            position_size=int(data.get("position_size", 0)),
            rr_ratio=float(data.get("rr_ratio", 0)),
            status=PlanStatus(data.get("status", "PAUSED")),
            plan_id=data.get("plan_id", uuid4().hex[:12]),
            entry_price=float(data["entry_price"]) if data.get("entry_price") is not None else None,
            entry_price_low=float(data["entry_price_low"]) if data.get("entry_price_low") is not None else None,
            entry_price_high=float(data["entry_price_high"]) if data.get("entry_price_high") is not None else None,
            strike_price=float(data["strike_price"]) if data.get("strike_price") is not None else None,
            order_id=data.get("order_id"),
            sl_order_id=data.get("sl_order_id"),
            tp_order_id=data.get("tp_order_id"),
            tp1_hit=bool(data.get("tp1_hit", False)),
            signal_strength=float(data.get("signal_strength", 0)),
            macro_env=data.get("macro_env", "NEUTRAL"),
            rationale=data.get("rationale", ""),
            is_intraday=bool(data.get("is_intraday", True)),
            ai_setup_done=bool(data.get("ai_setup_done", False)),
            created_at=datetime.fromisoformat(created) if created else datetime.now(timezone.utc),
            last_ai_review_at=datetime.fromisoformat(last_ai) if last_ai else None,
            meta=dict(data.get("meta", {})),
        )

    def format_plan(self) -> str:
        lines = [
            "TRADE PLAN",
            "──────────────────────────────",
            f"Symbol        : {self.symbol}",
            f"Direction     : {self.direction.value}",
            f"Timeframe     : {self.timeframe}",
            f"Bias Source   : {self.bias_source}",
            f"Entry Zone    : {self.entry_zone}",
            f"Entry Markers : {self.entry_price_low} – {self.entry_price_high}",
            f"Strike        : {self.strike_price}",
            f"Entry Type    : {self.entry_type.value}",
            f"Stop Loss     : {self.stop_loss} — reason: {self.stop_loss_reason}",
            f"Take Profit 1 : {self.take_profit_1} — partial exit: {self.tp1_exit_pct}%",
            f"Take Profit 2 : {self.take_profit_2} — partial exit: {self.tp2_exit_pct}%",
            f"Risk Amount   : {self.risk_amount}",
            f"Position Size : {self.position_size}",
            f"R:R Ratio     : {self.rr_ratio:.2f}",
            f"Signal        : {self.signal_strength:.2f}",
            f"Status        : {self.status.value}",
        ]
        return "\n".join(lines)


def validate_bracket(plan: TradePlan, entry: float | None = None) -> tuple[bool, str]:
    """Ensure SL/TP are on the correct side of entry — never naked."""
    if plan.stop_loss <= 0 or plan.take_profit_1 <= 0 or plan.take_profit_2 <= 0:
        return False, "SL and TP levels must be positive"
    if plan.stop_loss == plan.take_profit_1:
        return False, "SL and TP1 cannot be equal"

    px = entry if entry is not None else plan.entry_target()
    if plan.direction == TradeDirection.LONG:
        if plan.stop_loss >= px:
            return False, "LONG stop loss must be below entry"
        if plan.take_profit_1 <= px:
            return False, "LONG take profit must be above entry"
        if plan.take_profit_2 <= plan.take_profit_1:
            return False, "LONG TP2 must be above TP1"
    else:
        if plan.stop_loss <= px:
            return False, "SHORT stop loss must be above entry"
        if plan.take_profit_1 >= px:
            return False, "SHORT take profit must be below entry"
        if plan.take_profit_2 >= plan.take_profit_1:
            return False, "SHORT TP2 must be below TP1"
    return True, "ok"


def required_min_rr(plan: TradePlan, *, min_rr: float, min_rr_scalp: float | None = None) -> float:
    if plan.meta.get("setup_type") == "momentum_scalp" and min_rr_scalp is not None:
        return min_rr_scalp
    return min_rr


def validate_plan(
    plan: TradePlan,
    *,
    min_rr: float,
    min_rr_scalp: float | None = None,
    max_sl_capital_pct: float = 2.0,
) -> tuple[bool, str]:
    floor = required_min_rr(plan, min_rr=min_rr, min_rr_scalp=min_rr_scalp)
    # Use a small epsilon to avoid floating-point false rejections (e.g., 2.0 < 2.0 → False, but 1.9999 < 2.0 → True)
    if plan.rr_ratio < floor - 0.001:
        return False, f"R:R {plan.rr_ratio:.2f} below minimum {floor}"
    if plan.position_size <= 0:
        return False, "Position size must be positive"
    if plan.risk_amount <= 0:
        return False, "Risk amount must be positive"
    return validate_bracket(plan)


def is_strong_signal(plan: TradePlan, min_strength: float, min_rr: float = 2.0) -> bool:
    return plan.signal_strength >= min_strength and plan.rr_ratio >= min_rr
