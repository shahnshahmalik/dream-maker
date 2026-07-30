"""Day-type classifier for NIFTY opening-range gate.

Classifies the current trading day into one of 7 types based on:
  - Gap from previous close
  - Opening-range structure (first 15 min = 3×5m candles)
  - Developing day range (for Range/Inside skip — see note below)

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

IMPORTANT — Range/Inside detection:
  The backtest used FULL-DAY range (high-low) < 0.8% of prev_close.
  That metric is unknown at the open. Using the 15-min Opening Range
  with the same 0.8% threshold incorrectly blocks most normal days
  (a quiet NIFTY open is often 0.3–0.6%). Live logic therefore:
    - never blocks on OR range alone
    - only blocks as Range/Inside after midday once the developing
      day range is still < 0.8%
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, time
from enum import Enum
from typing import Optional
from zoneinfo import ZoneInfo

from models.orders import OHLCV
from models.trade_plan import TradeDirection

log = logging.getLogger("dream_maker.day_classifier")

IST = ZoneInfo("Asia/Kolkata")

# Thresholds (all as fractions of price)
_GAP_THRESHOLD = 0.003        # 0.3% gap = meaningful gap
# Full-day range threshold from the backtest — NOT for Opening Range.
_RANGE_DAY_THRESHOLD = 0.008  # developing day range < 0.8% → Range/Inside
_MIN_OR_CANDLES = 3           # need ≥3 5m candles (15 min) to classify
# Earliest IST time at which a quiet developing day may be blocked.
# Before this, Range/Inside is unknowable — OR alone must not block.
_RANGE_BLOCK_AFTER_IST = time(11, 0)


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
    *,
    now_ist: datetime | None = None,
) -> DayClassification:
    """Classify the current trading day.

    Args:
        today_candles: 5-min candles for today so far (sorted oldest→newest).
        prev_close: previous trading day's closing price.
        now_ist: current time in IST (injectable for tests). Used only for the
            midday developing-range Range/Inside check.

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

    # ── Range/Inside Day gate (matches backtest: FULL day range, not OR) ──
    # Only evaluate after midday once enough of the session has printed.
    # A quiet 15-min OR is normal and must NOT hard-block the day.
    now = now_ist or datetime.now(IST)
    if now.tzinfo is None:
        now = now.replace(tzinfo=IST)
    else:
        now = now.astimezone(IST)

    day_high = max(c.high for c in today_candles)
    day_low = min(c.low for c in today_candles)
    day_range_pct = (day_high - day_low) / prev_close if prev_close else 0.0

    if (
        now.timetz().replace(tzinfo=None) >= _RANGE_BLOCK_AFTER_IST
        and day_range_pct < _RANGE_DAY_THRESHOLD
    ):
        classification = DayClassification(
            day_type=DayType.RANGE_INSIDE,
            allowed_directions=None,
            gap_pct=gap_pct,
            or_range_pct=or_range_pct,
            is_gap_day=is_gap_day,
            bullish_or_structure=bullish_structure,
            reason=(
                f"Range/Inside Day — developing day range {day_range_pct:.2%} "
                f"< {_RANGE_DAY_THRESHOLD:.2%} after {_RANGE_BLOCK_AFTER_IST.strftime('%H:%M')} IST. "
                "DirWR 25%: no trades."
            ),
        )
        log.info(
            "Day classified: RANGE_INSIDE | day range %.2f%% | OR %.2f%% | BLOCKED",
            day_range_pct * 100,
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
                f"| OR range {or_range_pct:.2%} | day range {day_range_pct:.2%} | DirWR 100%"
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
                f"| OR range {or_range_pct:.2%} | day range {day_range_pct:.2%}"
            ),
        )

    log.info(
        "Day classified: %s | gap=%.2f%% | OR range=%.2f%% | day range=%.2f%% | allowed=%s",
        classification.day_type.value,
        gap_pct * 100,
        or_range_pct * 100,
        day_range_pct * 100,
        [d.value for d in (classification.allowed_directions or [])],
    )
    return classification


class DayGate:
    """Stateful wrapper around `classify()` with a careful cache.

    Cache rules:
      - UNKNOWN is never cached (re-fetch until the Opening Range forms).
      - Before 11:00 IST, directional classifications are cached but will be
        re-evaluated after midday so a quiet developing day can still be
        blocked as Range/Inside (matching the full-day-range backtest rule).
      - RANGE_INSIDE and any post-midday classification are final for the day.
    """

    def __init__(self, broker, enabled: bool = True) -> None:
        self._broker = broker
        self._enabled = enabled
        self._cached_date: Optional[date] = None
        self._cached_result: Optional[DayClassification] = None
        self._cached_after_midday: bool = False

    def reset(self) -> None:
        """Force re-classification on next check (e.g. after new day starts)."""
        self._cached_date = None
        self._cached_result = None
        self._cached_after_midday = False

    def classify_today(self) -> DayClassification:
        """Return today's classification, refreshing when the cache is stale."""
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

        now_ist = datetime.now(IST)
        today = now_ist.date()
        past_midday = now_ist.timetz().replace(tzinfo=None) >= _RANGE_BLOCK_AFTER_IST

        if (
            self._cached_date == today
            and self._cached_result is not None
            and self._cached_result.day_type != DayType.UNKNOWN
            and (self._cached_after_midday or not past_midday
                 or self._cached_result.day_type == DayType.RANGE_INSIDE)
        ):
            return self._cached_result

        result = self._fetch_and_classify(now_ist=now_ist)
        # Never pin UNKNOWN — keep polling until OR candles arrive.
        if result.day_type != DayType.UNKNOWN:
            self._cached_date = today
            self._cached_result = result
            self._cached_after_midday = past_midday
        return result

    def _fetch_and_classify(self, *, now_ist: datetime | None = None) -> DayClassification:
        """Fetch candles from broker and classify."""
        now_ist = now_ist or datetime.now(IST)
        try:
            # Use the underlying NIFTY index (securityId 13) for clean daily data
            daily_candles = self._broker.get_ohlcv("NIFTY50IDX", "1d", 5)
            if not daily_candles or len(daily_candles) < 2:
                log.warning("Day gate: insufficient daily candles — gate open")
                return _unknown_result("Insufficient daily candles")

            prev_close = daily_candles[-2].close  # yesterday's close

            # Need enough bars for OR + developing day range after midday.
            today_5m_raw = self._broker.get_ohlcv("NIFTY50IDX", "5m", 80)
            if not today_5m_raw:
                log.warning("Day gate: no 5m candles yet — gate open")
                return _unknown_result("No intraday candles yet")

            # Filter to today's candles only (IST calendar date — NSE session).
            today_date = now_ist.date()
            today_5m = []
            for c in today_5m_raw:
                ts = c.timestamp
                if not hasattr(ts, "date"):
                    continue
                if ts.tzinfo is None:
                    ts_date = ts.date()
                else:
                    ts_date = ts.astimezone(IST).date()
                if ts_date == today_date:
                    today_5m.append(c)

            if not today_5m:
                log.warning("Day gate: no candles for today yet — gate open")
                return _unknown_result("No candles for today yet")

            return classify(today_5m, prev_close, now_ist=now_ist)

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
