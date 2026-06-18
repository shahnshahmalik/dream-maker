"""Day-type classifier for NIFTY opening-range gate.

Classifies the current trading day into one of 7 types based on:
  - Gap from previous close
  - Opening-range structure (first 15 min = 3×5m candles)
  - OR range size relative to price

Classification rules (derived from Jan–Jun 2026 backtest, 111 days):

  Day Type          %Freq  DirWR   Action
  ─────────────────────────────────────────────
  V-Reversal Bull   38.7%   91%   LONG after midday reversal
  V-Reversal Bear   36.9%   85%   SHORT after midday reversal
  Trend Day Up       5.4%  100%   LONG only (OR high breakout)
  Trend Day Down     8.1%  100%   SHORT only (OR low breakdown)
  Gap Down & Rally   1.8%  100%   LONG (fade the gap)
  Gap Up & Trend     1.8%  100%   LONG (follow the gap)
  Gap Down & Trend   2.7%  100%   SHORT (follow the gap)
  Range/Inside Day   3.6%   25%   SKIP — no trades

Key insight: NIFTY is a V-Reversal market 75% of the time.
Range/Inside Days have 25% DirWR — premium decays, skip them.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from typing import Optional

from models.orders import OHLCV
from models.trade_plan import TradeDirection

log = logging.getLogger("dream_maker.day_classifier")

# Thresholds (all as fractions of price)
_GAP_THRESHOLD = 0.003        # 0.3% gap = meaningful gap
_RANGE_DAY_THRESHOLD = 0.008  # OR range < 0.8% → likely range/inside day
_MIN_OR_CANDLES = 3           # need ≥3 5m candles (15 min) to classify


class DayType(str, Enum):
    TREND_UP = "trend_up"
    TREND_DOWN = "trend_down"
    V_REVERSAL_BULL = "v_reversal_bull"
    V_REVERSAL_BEAR = "v_reversal_bear"
    GAP_DOWN_RALLY = "gap_down_rally"
    GAP_DOWN_TREND = "gap_down_trend"
    GAP_UP_TREND = "gap_up_trend"
    RANGE_INSIDE = "range_inside"
    UNKNOWN = "unknown"       # not enough data yet


# Allowed trading directions per day type.
# None = skip entirely, list = allowed directions, empty list = any direction
_ALLOWED_DIRECTIONS: dict[DayType, Optional[list[TradeDirection]]] = {
    DayType.TREND_UP:         [TradeDirection.LONG],
    DayType.TREND_DOWN:       [TradeDirection.SHORT],
    DayType.V_REVERSAL_BULL:  [TradeDirection.LONG],
    DayType.V_REVERSAL_BEAR:  [TradeDirection.SHORT],
    DayType.GAP_DOWN_RALLY:   [TradeDirection.LONG],
    DayType.GAP_DOWN_TREND:   [TradeDirection.SHORT],
    DayType.GAP_UP_TREND:     [TradeDirection.LONG],
    DayType.RANGE_INSIDE:     None,   # skip — 25% DirWR, premium decays
    DayType.UNKNOWN:          [],     # no data yet — allow all, don't block
}


@dataclass(frozen=True)
class DayClassification:
    day_type: DayType
    allowed_directions: Optional[list[TradeDirection]]  # None = blocked
    gap_pct: float
    or_range_pct: float
    is_gap_day: bool
    bullish_or_structure: bool  # HH/HL in OR
    reason: str

    @property
    def is_blocked(self) -> bool:
        return self.allowed_directions is None

    @property
    def allows_any_direction(self) -> bool:
        return self.allowed_directions is not None and len(self.allowed_directions) == 0

    def allows(self, direction: TradeDirection) -> bool:
        if self.allowed_directions is None:
            return False
        if len(self.allowed_directions) == 0:
            return True  # UNKNOWN — don't block
        return direction in self.allowed_directions


def _or_structure_is_bullish(candles: list[OHLCV]) -> bool:
    """Return True if the OR candles show HH/HL (bullish structure)."""
    if len(candles) < 2:
        return True  # not enough data — don't assume
    highs = [c.high for c in candles]
    lows = [c.low for c in candles]
    # HH/HL: each successive high is higher AND each successive low is higher
    hh = all(highs[i] >= highs[i - 1] for i in range(1, len(highs)))
    hl = all(lows[i] >= lows[i - 1] for i in range(1, len(lows)))
    return hh or hl  # at least one structure leg bullish


def _gap_direction(gap_pct: float) -> str:
    if gap_pct > _GAP_THRESHOLD:
        return "up"
    if gap_pct < -_GAP_THRESHOLD:
        return "down"
    return "flat"


def classify(
    today_candles: list[OHLCV],
    prev_close: float,
) -> DayClassification:
    """Classify the current trading day.

    Args:
        today_candles: 5-min candles for today so far (sorted oldest→newest).
        prev_close: previous trading day's closing price.

    Returns:
        DayClassification with day_type + allowed directions.
    """
    if len(today_candles) < _MIN_OR_CANDLES or prev_close <= 0:
        return DayClassification(
            day_type=DayType.UNKNOWN,
            allowed_directions=[],
            gap_pct=0.0,
            or_range_pct=0.0,
            is_gap_day=False,
            bullish_or_structure=True,
            reason="Not enough candles yet — gate open",
        )

    or_candles = today_candles[:_MIN_OR_CANDLES]
    today_open = or_candles[0].open
    or_high = max(c.high for c in or_candles)
    or_low = min(c.low for c in or_candles)
    or_range_pct = (or_high - or_low) / prev_close if prev_close else 0.0

    gap_pct = (today_open - prev_close) / prev_close
    gap_dir = _gap_direction(gap_pct)
    is_gap_day = gap_dir != "flat"
    bullish_structure = _or_structure_is_bullish(or_candles)

    # ── Range/Inside Day gate (25% DirWR — always skip) ──
    if or_range_pct < _RANGE_DAY_THRESHOLD:
        classification = DayClassification(
            day_type=DayType.RANGE_INSIDE,
            allowed_directions=None,
            gap_pct=gap_pct,
            or_range_pct=or_range_pct,
            is_gap_day=is_gap_day,
            bullish_or_structure=bullish_structure,
            reason=(
                f"Range/Inside Day — OR range {or_range_pct:.2%} < {_RANGE_DAY_THRESHOLD:.2%}. "
                "DirWR 25%: no trades."
            ),
        )
        log.info(
            "Day classified: RANGE_INSIDE | OR range %.2f%% | BLOCKED",
            or_range_pct * 100,
        )
        return classification

    # ── Gap days ──
    if is_gap_day:
        if gap_dir == "down":
            day_type = DayType.GAP_DOWN_TREND if not bullish_structure else DayType.GAP_DOWN_RALLY
        else:
            day_type = DayType.GAP_UP_TREND

        allowed = _ALLOWED_DIRECTIONS[day_type]
        classification = DayClassification(
            day_type=day_type,
            allowed_directions=allowed,
            gap_pct=gap_pct,
            or_range_pct=or_range_pct,
            is_gap_day=True,
            bullish_or_structure=bullish_structure,
            reason=(
                f"Gap {gap_dir} {gap_pct:.2%} | OR structure={'bullish' if bullish_structure else 'bearish'} "
                f"| OR range {or_range_pct:.2%} | DirWR 100%"
            ),
        )
    else:
        # ── Non-gap days: V-Reversal or Trend ──
        # We can't distinguish Trend vs V-Reversal at OR time —
        # treat as V-Reversal with structural direction lock.
        # If OR structure is bullish → V-Reversal Bull (wait for HH/HL post midday)
        # If OR structure is bearish → V-Reversal Bear (wait for LH/LL post midday)
        if bullish_structure:
            day_type = DayType.V_REVERSAL_BULL
        else:
            day_type = DayType.V_REVERSAL_BEAR

        allowed = _ALLOWED_DIRECTIONS[day_type]
        classification = DayClassification(
            day_type=day_type,
            allowed_directions=allowed,
            gap_pct=gap_pct,
            or_range_pct=or_range_pct,
            is_gap_day=False,
            bullish_or_structure=bullish_structure,
            reason=(
                f"No gap ({gap_pct:.2%}) | OR structure={'bullish' if bullish_structure else 'bearish'} "
                f"| OR range {or_range_pct:.2%}"
            ),
        )

    log.info(
        "Day classified: %s | gap=%.2f%% | OR range=%.2f%% | allowed=%s",
        classification.day_type.value,
        gap_pct * 100,
        or_range_pct * 100,
        [d.value for d in (classification.allowed_directions or [])],
    )
    return classification


class DayGate:
    """Stateful wrapper: classifies once per calendar day, caches result.

    The gate fetches today's 5-min candles and yesterday's daily close
    from the broker, then delegates to `classify()`.

    Cache is reset at midnight — subsequent calls on the same date
    return the cached result without hitting the broker.
    """

    def __init__(self, broker, enabled: bool = True) -> None:
        self._broker = broker
        self._enabled = enabled
        self._cached_date: Optional[date] = None
        self._cached_result: Optional[DayClassification] = None

    def reset(self) -> None:
        """Force re-classification on next check (e.g. after new day starts)."""
        self._cached_date = None
        self._cached_result = None

    def classify_today(self) -> DayClassification:
        """Return today's classification. Uses cache if already computed today."""
        if not self._enabled:
            return DayClassification(
                day_type=DayType.UNKNOWN,
                allowed_directions=[],
                gap_pct=0.0,
                or_range_pct=0.0,
                is_gap_day=False,
                bullish_or_structure=True,
                reason="Day gate disabled",
            )

        today = datetime.now().date()
        if self._cached_date == today and self._cached_result is not None:
            return self._cached_result

        result = self._fetch_and_classify()
        self._cached_date = today
        self._cached_result = result
        return result

    def _fetch_and_classify(self) -> DayClassification:
        """Fetch candles from broker and classify."""
        try:
            # Use the underlying NIFTY index (securityId 13) for clean daily data
            daily_candles = self._broker.get_ohlcv("NIFTY50IDX", "1d", 5)
            if not daily_candles or len(daily_candles) < 2:
                log.warning("Day gate: insufficient daily candles — gate open")
                return _unknown_result("Insufficient daily candles")

            prev_close = daily_candles[-2].close  # yesterday's close

            today_5m = self._broker.get_ohlcv("NIFTY50IDX", "5m", 20)
            if not today_5m:
                log.warning("Day gate: no 5m candles yet — gate open")
                return _unknown_result("No intraday candles yet")

            return classify(today_5m, prev_close)

        except Exception as exc:
            log.warning("Day gate fetch failed (%s) — gate open", exc)
            return _unknown_result(f"Fetch error: {exc}")

    def allows(self, direction: TradeDirection) -> tuple[bool, str]:
        """Return (allowed, reason) for the given trade direction today."""
        classification = self.classify_today()
        if classification.allows(direction):
            return True, classification.reason
        return False, (
            f"Day gate blocked: {classification.day_type.value} "
            f"allows only {[d.value for d in (classification.allowed_directions or [])]} "
            f"— {classification.reason}"
        )


def _unknown_result(reason: str) -> DayClassification:
    return DayClassification(
        day_type=DayType.UNKNOWN,
        allowed_directions=[],
        gap_pct=0.0,
        or_range_pct=0.0,
        is_gap_day=False,
        bullish_or_structure=True,
        reason=reason,
    )
