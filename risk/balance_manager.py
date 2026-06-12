"""Balance-aware margin gate — BUY-only vs full SELL capability.

Indian brokers require significantly more margin to SELL/write options
than to BUY them:

    BUY  1 lot NIFTY ATM : ~₹12K (premium only)
    SELL 1 lot NIFTY ATM : ~₹1.2L (SPAN + exposure margin)

Small accounts should never attempt SELL operations — a margin shortfall
would reject the order or trigger a penalty.  Instead, bearish views are
expressed by BUY-ing puts (PE), and bullish views by BUY-ing calls (CE).

This module provides a single decision point used by the executor and
strike selector to enforce the correct side.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import logging

log = logging.getLogger("dream_maker.balance")


class Capability(Enum):
    BUY_ONLY = "buy_only"
    FULL = "full"          # can BUY *and* SELL / write


@dataclass
class Decision:
    capability: Capability
    reason: str


# ── Conservative defaults ──────────────────────────────────────────────
_DEFAULT_MIN_SELL_BALANCE = 100_000   # ₹1L — one lot SELL margin is ~₹1-1.5L
_DEFAULT_SELL_BUFFER      = 1.5       # require 1.5× the minimum before allowing sells


class BalanceManager:
    """Single-responsibility gate: should we allow SELL orders?

    Public API
    ----------
    evaluate(available_balance) → Decision
    get_side(direction, available_balance) → "BUY" | "SELL"
    """

    def __init__(
        self,
        min_sell_balance: float = _DEFAULT_MIN_SELL_BALANCE,
        sell_buffer: float = _DEFAULT_SELL_BUFFER,
    ):
        self.min_sell = min_sell_balance
        self.buffer = sell_buffer

    # ── core evaluation ──────────────────────────────────────────────

    def evaluate(self, available_balance: float) -> Decision:
        threshold = self.min_sell * self.buffer
        if available_balance < threshold:
            return Decision(
                Capability.BUY_ONLY,
                f"Balance ₹{available_balance:,.0f} < ₹{threshold:,.0f} "
                f"({self.buffer:.0f}× sell threshold) — BUY only",
            )
        return Decision(Capability.FULL, "Balance sufficient for full trading")

    # ── side resolver (used by executor) ─────────────────────────────

    def get_side(self, direction, available_balance: float) -> str:
        """Return the broker order side ("BUY" or "SELL").

        When the account is BUY_ONLY, bearish views are expressed by
        BUY-ing puts (the CE→PE flip is handled upstream by the strike
        selector), so we always return "BUY".

        When the account is FULL, the traditional mapping applies:
            LONG  → BUY
            SHORT → SELL
        """
        # Avoid circular import by late-importing the enum
        from models.trade_plan import TradeDirection  # noqa: PLC0415

        decision = self.evaluate(available_balance)
        if decision.capability == Capability.BUY_ONLY:
            log.debug(
                "BUY_ONLY mode: direction=%s forced to BUY (%s)",
                direction, decision.reason,
            )
            return "BUY"
        return "BUY" if direction == TradeDirection.LONG else "SELL"
