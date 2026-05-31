"""Symbol picker — selects the best F&O instrument based on account balance."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

log = logging.getLogger("dream_maker.picker")


@dataclass(frozen=True)
class InstrumentCandidate:
    symbol: str
    lot_size: int
    min_capital_estimate: float  # minimum INR needed for 1 lot
    tier: int  # 1=preferred index futures, 2=options, 3=stock F&O


@dataclass
class SymbolPicker:
    preferred: list[str] = field(
        default_factory=lambda: ["SENSEX", "BANKNIFTY", "NIFTY50IDX"]
    )
    fallback: str = "NIFTY50IDX"

    # Margin buffer — require balance to exceed min_capital by this factor
    MARGIN_BUFFER: float = 1.3

    def select(
        self, balance: float, candidates: list[InstrumentCandidate]
    ) -> str:
        """Pick the best affordable symbol, or fall back."""
        best = self._best_fit(candidates, balance)
        if best:
            log.info(
                "Symbol picker: selected %s (tier %d, "
                "min capital ~₹%.0f, balance ₹%.0f)",
                best.symbol,
                best.tier,
                best.min_capital_estimate,
                balance,
            )
            return best.symbol
        log.warning(
            "Symbol picker: no preferred instrument fits (balance ₹%.0f) — "
            "using fallback %s",
            balance,
            self.fallback,
        )
        return self.fallback

    def _best_fit(
        self,
        candidates: list[InstrumentCandidate],
        balance: float,
    ) -> InstrumentCandidate | None:
        """Return the highest-tier candidate that fits within balance."""
        ranked = sorted(
            candidates,
            key=lambda c: (
                c.tier,
                self._pref_order(c.symbol),
            ),
        )
        for c in ranked:
            required = c.min_capital_estimate * self.MARGIN_BUFFER
            if balance >= required:
                return c
        return None

    def _pref_order(self, symbol: str) -> int:
        """Rank by preferred list index (lower = better)."""
        sym = symbol.upper()
        for i, pref in enumerate(self.preferred):
            if pref.upper() in sym or sym.startswith(pref.upper()):
                return i
        return len(self.preferred)
