"""Risk management — position sizing and bracket prices."""

from __future__ import annotations

from dataclasses import dataclass

from models.trade_plan import TradeDirection


@dataclass
class BracketPrices:
    stop_loss: float
    take_profit_1: float
    take_profit_2: float


@dataclass
class SizingResult:
    qty: int
    risk_amount: float
    rr_ratio: float
    reason: str


class RiskManager:
    def __init__(self, risk_pct_per_trade: float, min_rr_ratio: float):
        self.risk_pct = risk_pct_per_trade
        self.min_rr = min_rr_ratio

    def compute_size(
        self,
        capital: float,
        entry: float,
        stop_loss: float,
        take_profit: float,
        *,
        lot_size: int = 1,
    ) -> SizingResult:
        risk_amount = capital * (self.risk_pct / 100.0)
        sl_distance = abs(entry - stop_loss)
        if sl_distance <= 0:
            return SizingResult(0, 0, 0, "Invalid SL distance")
        raw_qty = int(risk_amount / sl_distance)
        qty = max(lot_size, (raw_qty // lot_size) * lot_size)
        tp_distance = abs(take_profit - entry)
        rr = tp_distance / sl_distance if sl_distance else 0
        if rr < self.min_rr:
            return SizingResult(0, risk_amount, rr, f"R:R {rr:.2f} below minimum {self.min_rr}")
        sl_pct_of_capital = (sl_distance / entry) * qty / capital * 100 if entry else 100
        if sl_pct_of_capital > 2.0:
            return SizingResult(0, risk_amount, rr, f"SL exposure {sl_pct_of_capital:.2f}% exceeds 2% cap")
        return SizingResult(qty, risk_amount, rr, "ok")

    def bracket(
        self,
        entry: float,
        direction: TradeDirection,
        sl_pct: float = 1.0,
        tp1_pct: float = 2.0,
        tp2_pct: float = 4.0,
    ) -> BracketPrices:
        sl = sl_pct / 100.0
        tp1 = tp1_pct / 100.0
        tp2 = tp2_pct / 100.0
        if direction == TradeDirection.LONG:
            return BracketPrices(
                stop_loss=entry * (1 - sl),
                take_profit_1=entry * (1 + tp1),
                take_profit_2=entry * (1 + tp2),
            )
        return BracketPrices(
            stop_loss=entry * (1 + sl),
            take_profit_1=entry * (1 - tp1),
            take_profit_2=entry * (1 - tp2),
        )
