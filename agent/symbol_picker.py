"""Symbol picker — selects the best F&O instrument based on account balance.

Now with full option contract resolution: generic placeholders like BANKNIFTYOPT
are resolved to actual Dhan contracts (e.g., BANKNIFTY25060548000CE) using the
spot price to compute ATM strikes and nearest weekly expiry.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import ClassVar

log = logging.getLogger("dream_maker.picker")


@dataclass(frozen=True)
class InstrumentCandidate:
    symbol: str
    lot_size: int
    min_capital_estimate: float  # minimum INR needed for 1 lot
    tier: int  # 1=preferred index futures, 2=options, 3=stock F&O
    is_option: bool = False       # True if this is an option contract
    underlying: str = ""          # e.g., "BANKNIFTY" for a BANKNIFTY option


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

    # Index options — always available when futures don't fit
    INDEX_OPTIONS: ClassVar[list[str]] = ["NIFTY", "BANKNIFTY", "FINNIFTY"]

    # Stock option underlyings (user-requested + defaults)
    STOCK_OPTIONS: ClassVar[dict[str, int]] = {
        # symbol: lot_size
        "DIXON":       300,
        "JUBLFOOD":    250,
        "HPCL":        500,
        "INDUSTOWER":  400,
        "KFINTECH":    200,
        "EXIDEIND":    600,
        "ITC":        1600,
        "TATASTEEL":   500,
    }

    # Balance thresholds
    STOCK_OPTION_THRESHOLD: ClassVar[float] = 7000.0

    # Margin buffer — require balance to exceed min_capital by this factor
    MARGIN_BUFFER: ClassVar[float] = 1.3

    # Month codes
    _MONTH_CODES: ClassVar[dict[int, str]] = {
        1: "JAN", 2: "FEB", 3: "MAR", 4: "APR", 5: "MAY",
        6: "JUN", 7: "JUL", 8: "AUG", 9: "SEP", 10: "OCT",
        11: "NOV", 12: "DEC",
    }

    def build_candidates(self, balance: float) -> list[InstrumentCandidate]:
        """Build candidate list: index futures → index options → stock options."""
        candidates: list[InstrumentCandidate] = []
        prefs_upper = {p.upper() for p in self.preferred}

        # Tier 1: Index futures (only preferred symbols)
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
                        underlying=idx_name,
                    )
                )

        # Tier 2: Index ATM weekly options (for low balance, or as fallback)
        has_affordable_futures = any(
            c.tier == 1 and balance >= c.min_capital_estimate * self.MARGIN_BUFFER
            for c in candidates
        )
        if not has_affordable_futures:
            for idx in self.INDEX_OPTIONS:
                if idx in prefs_upper or any(idx in p.upper() for p in prefs_upper):
                    candidates.append(
                        InstrumentCandidate(
                            symbol=f"{idx}OPT",   # resolved later
                            lot_size=25 if idx in ("NIFTY", "FINNIFTY") else 15,
                            min_capital_estimate=3000.0,
                            tier=2,
                            is_option=True,
                            underlying=idx,
                        )
                    )

        # Tier 3: Stock options (for very low balance)
        if balance < self.STOCK_OPTION_THRESHOLD or not candidates:
            for sym, lot in self.STOCK_OPTIONS.items():
                candidates.append(
                    InstrumentCandidate(
                        symbol=f"{sym}OPT",
                        lot_size=lot,
                        min_capital_estimate=2000.0 if lot <= 300 else 3500.0,
                        tier=3,
                        is_option=True,
                        underlying=sym,
                    )
                )

        return candidates

    # ------------------------------------------------------------------ #
    # Symbol resolution — constructs real Dhan option contracts
    # ------------------------------------------------------------------ #
    @staticmethod
    def resolve(symbol: str, spot_price: float = 0.0) -> str:
        """Map a generic symbol to an actual Dhan trading symbol.

        Generic futures like 'NIFTYFUT' → 'NIFTY25JUNFUT'.
        Generic options like 'BANKNIFTYOPT' → 'BANKNIFTY25060548000CE'
          (ATM CE for nearest Thursday expiry, computed from spot_price).
        Concrete symbols pass through unchanged.
        """
        import re

        sym = symbol.upper()

        # Already a concrete contract — return as-is
        if re.search(r"\d{2}[A-Z]{3}FUT", sym):
            return sym
        if re.search(r"\d+(CE|PE)", sym):
            return sym
        # Index IDX — keep as-is
        if sym.endswith("IDX") or sym.endswith("INDEX"):
            return sym

        # Generic futures → current month contract
        if sym.endswith("FUT") and not re.search(r"\d{2}[A-Z]{3}", sym):
            now = datetime.now()
            yy = str(now.year)[-2:]
            month = SymbolPicker._MONTH_CODES[now.month]
            base = sym.replace("FUT", "").replace("50", "")
            return f"{base}{yy}{month}FUT"

        # Generic options → resolve to actual ATM weekly contract
        if sym.endswith("OPT") and spot_price > 0:
            return SymbolPicker._resolve_option(sym.replace("OPT", ""), spot_price)

        return sym

    @staticmethod
    def _resolve_option(underlying: str, spot: float) -> str:
        """Construct a real Dhan option symbol for the given underlying.

        Returns e.g. 'BANKNIFTY25060548000CE' — ATM CE for nearest Thursday.
        """
        # Determine strike interval
        underlying_upper = underlying.upper()
        if "NIFTY" in underlying_upper and "BANK" not in underlying_upper:
            strike_interval = 50
        elif "BANKNIFTY" in underlying_upper:
            strike_interval = 100
        elif "FINNIFTY" in underlying_upper:
            strike_interval = 50
        elif "SENSEX" in underlying_upper:
            strike_interval = 100
        else:
            # Stock option — determine interval from price
            if spot <= 200:
                strike_interval = 5
            elif spot <= 1000:
                strike_interval = 10
            elif spot <= 5000:
                strike_interval = 50
            else:
                strike_interval = 100

        # Compute ATM strike
        atm_strike = int(round(spot / strike_interval) * strike_interval)

        # Nearest Thursday (NSE weekly expiry)
        today = datetime.now()
        days_until_thursday = (3 - today.weekday()) % 7
        if days_until_thursday == 0 and today.hour >= 15:
            # After market close on Thursday → next week
            days_until_thursday = 7
        expiry = today + timedelta(days=days_until_thursday)

        yy = str(expiry.year)[-2:]
        month_code = SymbolPicker._MONTH_CODES[expiry.month]
        strike_str = str(atm_strike)

        symbol = f"{underlying.upper()}{yy}{month_code}{strike_str}CE"
        log.info("Resolved option: %s → %s (spot=%.1f, strike=%d)", underlying, symbol, spot, atm_strike)
        return symbol
