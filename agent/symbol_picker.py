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
        "BANKNIFTY": (30, 35000.0),   # lot 30, ~₹35K MIS margin
        "NIFTY": (65, 30000.0),       # lot 65, ~₹30K MIS margin
        "FINNIFTY": (60, 25000.0),    # lot 60, ~₹25K MIS margin
        "MIDCPNIFTY": (50, 20000.0),  # lot 50, ~₹20K MIS margin
    }

    # Index options — always available when futures don't fit
    INDEX_OPTIONS: ClassVar[list[str]] = ["NIFTY", "BANKNIFTY", "FINNIFTY"]

    # Realistic option cost estimates (lot_size, min_capital_estimate)
    # Based on actual OTM option premiums: premium × lot_size
    # BALANCE must exceed min_capital × MARGIN_BUFFER (1.3) to be eligible
    INDEX_OPTION_ESTIMATES: ClassVar[dict[str, tuple[int, float]]] = {
        "NIFTY":      (65, 4500.0),    # lot 65, far OTM CE ~₹50-80 × 65 = ₹3.3K-5.2K
        "BANKNIFTY":  (30, 15000.0),   # lot 30, OTM CE ~₹300-600 × 30 = ₹9K-18K
        "FINNIFTY":   (60, 10000.0),   # lot 60, OTM CE ~₹150-300 × 60 = ₹9K-18K
    }

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

    # Approximate spot prices for stock underlyings (used when Dhan LTP
    # returns synthetic data). Updated periodically.
    STOCK_SPOTS: ClassVar[dict[str, float]] = {
        "DIXON":       13900.0,
        "JUBLFOOD":      580.0,
        "HPCL":          380.0,
        "INDUSTOWER":    350.0,
        "KFINTECH":     5400.0,
        "EXIDEIND":      420.0,
        "ITC":           420.0,
        "TATASTEEL":     140.0,
    }

    # Balance thresholds
    STOCK_OPTION_THRESHOLD: ClassVar[float] = 7000.0

    # Skip index options below this balance — but lowered to allow
    # ₹6K accounts since stock options lack Dhan chart/quote support
    SKIP_INDEX_OPTIONS_BELOW: ClassVar[float] = 5000.0

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

        # Tier 2: Index ATM weekly options (only if balance is sufficient)
        has_affordable_futures = any(
            c.tier == 1 and balance >= c.min_capital_estimate * self.MARGIN_BUFFER
            for c in candidates
        )
        can_afford_index_options = balance >= self.SKIP_INDEX_OPTIONS_BELOW
        if not has_affordable_futures and can_afford_index_options:
            for idx in self.INDEX_OPTIONS:
                if idx in prefs_upper or any(idx in p.upper() for p in prefs_upper):
                    lot, est = self.INDEX_OPTION_ESTIMATES.get(idx, (25, 8000.0))
                    candidates.append(
                        InstrumentCandidate(
                            symbol=f"{idx}OPT",   # resolved later
                            lot_size=lot,
                            min_capital_estimate=est,
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
        underlying = sym.replace("OPT", "")
        if sym.endswith("OPT"):
            if spot_price > 0:
                return SymbolPicker._resolve_option(underlying, spot_price)
            # Fall back to hardcoded spot for stocks (Dhan LTP returns synthetic)
            if underlying in SymbolPicker.STOCK_SPOTS:
                return SymbolPicker._resolve_option(underlying, SymbolPicker.STOCK_SPOTS[underlying])

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

    @staticmethod
    def _resolve_otm_strikes(underlying: str, spot: float, max_otm_steps: int = 6) -> list[str]:
        """Generate increasingly OTM option symbols for premium fallback.

        Returns a list of concrete option symbols from nearest-OTM to
        farthest-OTM.  ATM is excluded (use ``_resolve_option`` for that).
        The caller should iterate and check affordability via get_quote.

        Args:
            underlying: e.g. ``"NIFTY"`` or ``"BANKNIFTY"``
            spot: current spot price of the underlying
            max_otm_steps: how many OTM strikes to generate (default 6)

        Returns:
            e.g. ``["NIFTY25060523450CE", "NIFTY25060523500CE", ...]``
        """
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
            if spot <= 200:
                strike_interval = 5
            elif spot <= 1000:
                strike_interval = 10
            elif spot <= 5000:
                strike_interval = 50
            else:
                strike_interval = 100

        atm_strike = int(round(spot / strike_interval) * strike_interval)

        # Expiry
        today = datetime.now()
        days_until_thursday = (3 - today.weekday()) % 7
        if days_until_thursday == 0 and today.hour >= 15:
            days_until_thursday = 7
        expiry = today + timedelta(days=days_until_thursday)
        yy = str(expiry.year)[-2:]
        month_code = SymbolPicker._MONTH_CODES[expiry.month]

        symbols: list[str] = []
        for step in range(1, max_otm_steps + 1):
            otm_strike = atm_strike + step * strike_interval
            sym = f"{underlying_upper}{yy}{month_code}{otm_strike}CE"
            symbols.append(sym)

        return symbols

    @staticmethod
    def _strike_to_sym(underlying: str, strike: int) -> str:
        """Build a concrete option symbol for a specific strike."""
        today = datetime.now()
        days_until_thursday = (3 - today.weekday()) % 7
        if days_until_thursday == 0 and today.hour >= 15:
            days_until_thursday = 7
        expiry = today + timedelta(days=days_until_thursday)
        yy = str(expiry.year)[-2:]
        month_code = SymbolPicker._MONTH_CODES[expiry.month]
        return f"{underlying.upper()}{yy}{month_code}{strike}CE"
