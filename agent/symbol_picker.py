"""Symbol picker — selects the best F&O instrument based on account balance."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import ClassVar

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

    # ------------------------------------------------------------------ #
    # Instrument discovery
    # ------------------------------------------------------------------ #
    # Known index futures with estimated MIS intraday margin (approx 1 lot)
    INDEX_FUTURES: ClassVar[dict[str, tuple[int, float]]] = {
        "SENSEX": (10, 45000.0),      # lot 10, ~₹45K MIS margin
        "BANKNIFTY": (15, 35000.0),   # lot 15, ~₹35K MIS margin
        "NIFTY": (25, 30000.0),       # lot 25, ~₹30K MIS margin
        "FINNIFTY": (40, 25000.0),    # lot 40, ~₹25K MIS margin
        "MIDCPNIFTY": (50, 20000.0),  # lot 50, ~₹20K MIS margin
    }

    # Cheap stock F&O — small contract sizes, low margin barrier
    STOCK_FUTURES: ClassVar[list[tuple[str, int, float]]] = [
        ("IDEA", 6000, 5000.0),       # lot 6000, ~₹11/share → under ₹5K margin
        ("ITC", 1600, 8000.0),        # lot 1600, ~₹180/share
        ("TATASTEEL", 500, 6000.0),   # lot 500, ~₹140/share
    ]

    # Balance thresholds
    OPTIONS_THRESHOLD: ClassVar[float] = 25000.0  # below this, add options
    STOCK_FNO_THRESHOLD: ClassVar[float] = 7000.0  # below this, add stock F&O

    # Margin buffer — require balance to exceed min_capital by this factor
    MARGIN_BUFFER: ClassVar[float] = 1.3

    def build_candidates(self, balance: float) -> list[InstrumentCandidate]:
        """Build candidate list: index futures → options → stock F&O."""
        candidates: list[InstrumentCandidate] = []

        # Tier 1: Index futures (only preferred symbols)
        prefs_upper = {p.upper() for p in self.preferred}
        for idx_name, (lot, margin) in self.INDEX_FUTURES.items():
            if idx_name.upper() in prefs_upper or any(
                idx_name.upper() in p.upper() for p in prefs_upper
            ):
                candidates.append(
                    InstrumentCandidate(
                        symbol=f"{idx_name}FUT",
                        lot_size=lot,
                        min_capital_estimate=margin,
                        tier=1,
                    )
                )

        # Tier 2: ATM weekly options (for low balance, or as fallback)
        if balance < self.OPTIONS_THRESHOLD or not any(
            c.tier == 1 and balance >= c.min_capital_estimate * self.MARGIN_BUFFER
            for c in candidates
        ):
            option_indices = ["NIFTY", "BANKNIFTY"]
            for idx in option_indices:
                if idx in prefs_upper or any(
                    idx in p.upper() for p in prefs_upper
                ):
                    candidates.append(
                        InstrumentCandidate(
                            symbol=f"{idx}OPT",
                            lot_size=25 if idx == "NIFTY" else 15,
                            min_capital_estimate=3000.0,
                            tier=2,
                        )
                    )
            # Always add NIFTY options as a fallback even if not in preferences
            if "NIFTYOPT" not in {c.symbol for c in candidates}:
                candidates.append(
                    InstrumentCandidate(
                        symbol="NIFTYOPT",
                        lot_size=25,
                        min_capital_estimate=3000.0,
                        tier=2,
                    )
                )

        # Tier 3: Stock F&O with small contracts (for very low balance)
        if balance < self.STOCK_FNO_THRESHOLD:
            for sym, lot, margin in self.STOCK_FUTURES:
                if margin <= balance * self.MARGIN_BUFFER:
                    candidates.append(
                        InstrumentCandidate(
                            symbol=f"{sym}FUT",
                            lot_size=lot,
                            min_capital_estimate=margin,
                            tier=3,
                        )
                    )

        return candidates

    # ------------------------------------------------------------------ #
    # Symbol resolution
    # ------------------------------------------------------------------ #
    # Month codes for NSE F&O contracts
    _MONTH_CODES: ClassVar[dict[int, str]] = {
        1: "JAN", 2: "FEB", 3: "MAR", 4: "APR", 5: "MAY",
        6: "JUN", 7: "JUL", 8: "AUG", 9: "SEP", 10: "OCT",
        11: "NOV", 12: "DEC",
    }

    @staticmethod
    def resolve(symbol: str) -> str:
        """Map a generic symbol to an actual Dhan trading symbol.

        Generic futures like 'NIFTYFUT' → 'NIFTY25JUNFUT'.
        Generic options like 'NIFTYOPT' → kept as-is (caller enriches).
        Concrete symbols like 'NIFTY25JUNFUT' or 'NIFTY50IDX' pass through.
        """
        import re
        from datetime import datetime

        sym = symbol.upper()

        # Already a concrete contract — return as-is
        if re.search(r"\d{2}[A-Z]{3}FUT", sym):
            return sym
        if re.search(r"\d+(CE|PE)", sym):
            return sym
        # Index IDX — keep as-is (used for chart/market data)
        if sym.endswith("IDX") or sym.endswith("INDEX"):
            return sym

        # Generic futures → current month contract
        if sym.endswith("FUT") and not re.search(r"\d{2}[A-Z]{3}", sym):
            now = datetime.now()
            yy = str(now.year)[-2:]
            month = SymbolPicker._MONTH_CODES[now.month]
            base = sym.replace("FUT", "").replace("50", "")
            return f"{base}{yy}{month}FUT"

        # Options — returned as-is (caller handles strike/expiry enrichment)
        return sym
