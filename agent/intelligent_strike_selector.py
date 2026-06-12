"""Intelligent strike price selector — finds affordable option contracts.

When balance is too low for ATM options, systematically searches across:
- Both CE and PE (calls and puts)
- Multiple expiry weeks (further expiry = cheaper premium)
- Progressively wider OTM distances

Uses actual broker quotes when available, falls back to estimation.
Scores candidates by affordability, liquidity, and tradability.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import ClassVar

log = logging.getLogger("dream_maker.strike_selector")


# ── Constants ──────────────────────────────────────────────────────────

MONTH_CODES: dict[int, str] = {
    1: "JAN", 2: "FEB", 3: "MAR", 4: "APR", 5: "MAY",
    6: "JUN", 7: "JUL", 8: "AUG", 9: "SEP", 10: "OCT",
    11: "NOV", 12: "DEC",
}

# Premium estimation: ATM premium as % of spot (empirical)
ATM_PREMIUM_PCT: dict[str, float] = {
    "NIFTY": 0.008,
    "BANKNIFTY": 0.006,
    "SENSEX": 0.005,
    "FINNIFTY": 0.005,
}

# Strike intervals
STRIKE_INTERVALS: dict[str, int] = {
    "NIFTY": 50,
    "BANKNIFTY": 100,
    "SENSEX": 100,
    "FINNIFTY": 50,
}

# Minimum premium floor (₹ per unit)
MIN_PREMIUM_FLOOR = 5.0

# Maximum OTM distance as fraction of spot (e.g., 0.15 = 15% away)
MAX_OTM_FRACTION = 0.25

# How many expiry weeks to try
MAX_EXPIRY_WEEKS = 4

# Lot sizes
INDEX_LOT_SIZES: dict[str, int] = {
    "NIFTY": 65,
    "BANKNIFTY": 30,
    "SENSEX": 10,
    "FINNIFTY": 60,
}

# Margin buffer (safety multiplier on estimated cost)
MARGIN_BUFFER = 1.3


# ── Data Class ──────────────────────────────────────────────────────────

@dataclass
class StrikeCandidate:
    """A candidate option contract for trading."""
    symbol: str
    underlying: str
    strike: int
    option_type: str          # "CE" or "PE"
    expiry_date: date
    expiry_week: int          # 0 = this week, 1 = next week, ...
    distance_pct: float       # how far OTM (0 = ATM)
    lot_size: int
    estimated_premium: float  # our estimate
    actual_premium: float | None = None  # from broker quote
    total_cost_estimate: float = 0.0
    score: float = 0.0

    def __post_init__(self) -> None:
        premium = self.actual_premium if self.actual_premium is not None else self.estimated_premium
        self.total_cost_estimate = premium * self.lot_size


# ── Main Selector ───────────────────────────────────────────────────────

class IntelligentStrikeSelector:
    """Finds the best affordable option contract for a given balance."""

    def __init__(
        self,
        underlying: str,
        spot: float,
        balance: float,
        preferred_direction: str = "CE",  # "CE", "PE", or "ANY"
        margin_buffer: float = MARGIN_BUFFER,
    ):
        self.underlying = underlying.upper()
        self.spot = spot
        self.balance = balance
        self.preferred_direction = preferred_direction.upper()
        self.margin_buffer = margin_buffer

        base = re.sub(r"\d+$", "", self.underlying)
        base = re.sub(r"(IDX|INDEX)$", "", base, flags=re.IGNORECASE)
        self._base_underlying = base

        self._strike_interval = self._get_strike_interval()
        self._atm_premium_pct = ATM_PREMIUM_PCT.get(self._base_underlying, 0.008)
        self._lot_size = INDEX_LOT_SIZES.get(self._base_underlying, 25)

    def _get_strike_interval(self) -> int:
        # Sort by key length descending so BANKNIFTY matches before NIFTY
        sorted_keys = sorted(STRIKE_INTERVALS.keys(), key=len, reverse=True)
        for key in sorted_keys:
            if key in self._base_underlying:
                return STRIKE_INTERVALS[key]
        # Stock: derive from price
        if self.spot <= 200:
            return 5
        elif self.spot <= 1000:
            return 10
        elif self.spot <= 5000:
            return 50
        return 100

    # ── Public API ──────────────────────────────────────────────────

    def find_best(
        self,
        broker,  # BrokerProvider — duck-typed to avoid circular import
        max_otm_distance: float = MAX_OTM_FRACTION,
        max_expiry_weeks: int = MAX_EXPIRY_WEEKS,
    ) -> StrikeCandidate | None:
        """Find the best affordable option contract.

        Returns the highest-scored candidate that fits within balance,
        or None if nothing fits.
        """
        candidates = self._generate_candidates(max_otm_distance, max_expiry_weeks)

        if not candidates:
            log.warning("No option candidates generated for %s @ %.0f", self.underlying, self.spot)
            return None

        # Enrich with actual quotes
        self._enrich_quotes(broker, candidates)

        # Score and sort
        scored = self._score_candidates(candidates)
        scored.sort(key=lambda c: c.score, reverse=True)

        # Find first affordable one
        for c in scored:
            required = c.total_cost_estimate * self.margin_buffer
            if self.balance >= required:
                log.info(
                    "Selected %s (score=%.2f): %s strike=%d premium≈₹%.0f cost=₹%.0f need=₹%.0f "
                    "balance=₹%.0f",
                    c.option_type, c.score, c.symbol, c.strike,
                    c.actual_premium or c.estimated_premium,
                    c.total_cost_estimate, required, self.balance,
                )
                return c

        # Nothing fits — log the cheapest option for diagnostics
        cheapest = min(scored, key=lambda c: c.total_cost_estimate)
        log.warning(
            "No affordable option found. Cheapest: %s cost=₹%.0f balance=₹%.0f (need ₹%.0f more)",
            cheapest.symbol, cheapest.total_cost_estimate,
            self.balance, cheapest.total_cost_estimate * self.margin_buffer - self.balance,
        )
        return None

    # ── Candidate Generation ────────────────────────────────────────

    def _generate_candidates(
        self, max_otm_distance: float, max_expiry_weeks: int
    ) -> list[StrikeCandidate]:
        """Generate candidate option symbols across expiries and OTM distances."""
        candidates: list[StrikeCandidate] = []
        atm_strike = int(round(self.spot / self._strike_interval) * self._strike_interval)
        expiry_dates = self._get_expiry_dates(max_expiry_weeks)

        option_types = self._option_types_to_try()

        for week_idx, expiry in enumerate(expiry_dates):
            for ot in option_types:
                # For each expiry + option type, generate progressively OTM strikes
                for step in range(0, self._max_otm_steps(max_otm_distance)):
                    if ot == "CE":
                        strike = atm_strike + step * self._strike_interval
                    else:
                        strike = atm_strike - step * self._strike_interval

                    # Skip negative/nonsensical strikes
                    if strike <= 0:
                        continue

                    distance_pct = abs(strike - self.spot) / self.spot
                    if distance_pct > max_otm_distance:
                        break

                    symbol = self._build_symbol(
                        self._base_underlying, strike, ot, expiry
                    )
                    estimated = self._estimate_premium(distance_pct, week_idx)

                    candidates.append(
                        StrikeCandidate(
                            symbol=symbol,
                            underlying=self._base_underlying,
                            strike=strike,
                            option_type=ot,
                            expiry_date=expiry,
                            expiry_week=week_idx,
                            distance_pct=distance_pct,
                            lot_size=self._lot_size,
                            estimated_premium=estimated,
                        )
                    )

        return candidates

    def _option_types_to_try(self) -> list[str]:
        """Order of option types to try."""
        if self.preferred_direction == "CE":
            return ["CE", "PE"]
        elif self.preferred_direction == "PE":
            return ["PE", "CE"]
        return ["CE", "PE"]  # ANY: try calls first

    def _max_otm_steps(self, max_otm_distance: float) -> int:
        """How many strike intervals until we exceed max_otm_distance."""
        max_strike_move = int(self.spot * max_otm_distance)
        return max(1, max_strike_move // self._strike_interval + 1)

    def _get_expiry_dates(self, max_weeks: int) -> list[date]:
        """Get list of Thursday expiry dates starting from current week."""
        today = datetime.now().date()
        # Find this week's Thursday
        days_until_thu = (3 - today.weekday()) % 7
        if days_until_thu == 0 and datetime.now().hour >= 15:
            days_until_thu = 7  # after market close → next week
        this_thu = today + timedelta(days=days_until_thu)

        expiries = []
        for w in range(max_weeks):
            expiries.append(this_thu + timedelta(weeks=w))
        return expiries

    # ── Symbol Building ─────────────────────────────────────────────

    @staticmethod
    def _build_symbol(
        underlying: str, strike: int, option_type: str, expiry: date
    ) -> str:
        """Build Dhan-format option symbol.

        Example: NIFTY + 23400 + CE + 2026-06-05 → 'NIFTY26JUN23400CE'
        """
        yy = str(expiry.year)[-2:]
        month_code = MONTH_CODES.get(expiry.month, "JAN")
        return f"{underlying.upper()}{yy}{month_code}{strike}{option_type.upper()}"

    # ── Premium Estimation ──────────────────────────────────────────

    def _estimate_premium(self, distance_pct: float, expiry_week: int) -> float:
        """Estimate option premium based on distance from ATM and expiry.

        ATM = spot * atm_pct
        Each 1% OTM → ~15% premium drop (floor at MIN_PREMIUM_FLOOR)
        Each extra week → ~40% premium increase (theta decay)
        """
        atm_premium = self.spot * self._atm_premium_pct
        # Distance decay
        decay = max(0.15, 1.0 - distance_pct * 15)
        estimated = atm_premium * decay
        # Theta: further expiry = more premium (less time decay)
        theta_factor = 1.0 + expiry_week * 0.4
        estimated *= theta_factor
        return max(estimated, MIN_PREMIUM_FLOOR)

    # ── Quote Enrichment ────────────────────────────────────────────

    def _enrich_quotes(
        self, broker, candidates: list[StrikeCandidate]
    ) -> None:
        """Try to fetch actual premiums from broker for top candidates."""
        # Limit to avoid excessive API calls — enrich top 12
        to_check = candidates[:12]
        for c in to_check:
            try:
                quote = broker.get_quote(c.symbol)
                if quote and quote.ltp > 0:
                    # Filter out obvious fallback quotes (round numbers)
                    if quote.ltp < 1.0 or (quote.ltp > 1000 and quote.ltp % 100 == 0):
                        continue
                    c.actual_premium = quote.ltp
                    c.total_cost_estimate = quote.ltp * c.lot_size
                    log.debug(
                        "Quote for %s: ₹%.2f (est was ₹%.2f)",
                        c.symbol, quote.ltp, c.estimated_premium,
                    )
            except Exception as e:
                log.debug("Quote fetch failed for %s: %s", c.symbol, e)

    # ── Scoring ─────────────────────────────────────────────────────

    def _score_candidates(
        self, candidates: list[StrikeCandidate]
    ) -> list[StrikeCandidate]:
        """Score each candidate 0-1. Higher = better.

        Factors:
        - Affordability (40%): cheaper relative to balance = better
        - Proximity to ATM (30%): closer to ATM = better R:R potential
        - Liquidity proxy (15%): higher premium = more liquid
        - Expiry freshness (10%): this week > next week
        - Direction preference (5%): matching preferred type
        """
        max_cost = max((c.total_cost_estimate for c in candidates), default=1)
        max_distance = max((c.distance_pct for c in candidates), default=0.01)
        max_premium = max(
            (c.actual_premium or c.estimated_premium for c in candidates), default=1
        )

        for c in candidates:
            # Affordability: lower cost = better (40%)
            affordability = 1.0 - min(c.total_cost_estimate / max(max_cost, 1), 1.0)

            # Proximity: closer to ATM = better (30%)
            proximity = 1.0 - min(c.distance_pct / max(max_distance, 0.01), 1.0)

            # Liquidity: higher premium = more liquid (15%)
            premium = c.actual_premium or c.estimated_premium
            liquidity = min(premium / max(max_premium, 1), 1.0)

            # Expiry: this week = best (10%)
            expiry_score = max(0.0, 1.0 - c.expiry_week * 0.33)

            # Direction: preferred type = bonus (5%)
            direction_score = 1.0 if c.option_type == self.preferred_direction else 0.5

            # Quote confidence: actual quote > estimate (10% penalty for estimates)
            quote_bonus = 0.10 if c.actual_premium is not None else 0.0

            c.score = (
                0.35 * affordability
                + 0.25 * proximity
                + 0.10 * liquidity
                + 0.10 * expiry_score
                + 0.05 * direction_score
                + quote_bonus
            )

        return candidates


# ── Convenience function ────────────────────────────────────────────────

def find_affordable_strike(
    broker,           # BrokerProvider
    underlying: str,
    spot: float,
    balance: float,
    preferred_direction: str = "CE",
    margin_buffer: float = MARGIN_BUFFER,
) -> str | None:
    """Find the best affordable option symbol. Returns symbol or None."""
    selector = IntelligentStrikeSelector(
        underlying=underlying,
        spot=spot,
        balance=balance,
        preferred_direction=preferred_direction,
        margin_buffer=margin_buffer,
    )
    best = selector.find_best(broker)
    return best.symbol if best else None
