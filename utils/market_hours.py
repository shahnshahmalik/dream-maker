"""IST market hours, session state, and next-open scheduling."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")


@dataclass(frozen=True)
class MarketSession:
    is_open: bool
    reason: str
    next_open: datetime | None
    seconds_until_open: int


def parse_hours_range(hours_str: str) -> tuple[time, time]:
    start_s, end_s = hours_str.split("-")
    sh, sm = map(int, start_s.strip().split(":"))
    eh, em = map(int, end_s.strip().split(":"))
    return time(sh, sm), time(eh, em)


def now_ist() -> datetime:
    return datetime.now(IST)


def _holiday_set(holidays: list[str] | None) -> set[str]:
    return {h.strip() for h in (holidays or []) if h.strip()}


def is_trading_day(dt: datetime | None = None, holidays: list[str] | None = None) -> bool:
    dt = dt or now_ist()
    if dt.weekday() >= 5:
        return False
    return dt.strftime("%Y-%m-%d") not in _holiday_set(holidays)


def is_within_trading_hours(hours_str: str, dt: datetime | None = None) -> bool:
    dt = dt or now_ist()
    start, end = parse_hours_range(hours_str)
    t = dt.time()
    return start <= t <= end


def is_market_open(
    hours_str: str,
    dt: datetime | None = None,
    holidays: list[str] | None = None,
) -> bool:
    dt = dt or now_ist()
    return is_trading_day(dt, holidays) and is_within_trading_hours(hours_str, dt)


def is_within_trade_window(
    trade_start_ist: str,
    trade_end_ist: str,
    dt: datetime | None = None,
    holidays: list[str] | None = None,
) -> bool:
    """True if now is within the trade execution window AND it's a trading day.

    Trade window is a subset of market hours — e.g. 09:30–15:15 to avoid
    opening-volatility period (09:15–09:30).
    """
    dt = dt or now_ist()
    if not is_trading_day(dt, holidays):
        return False
    start = parse_hours_range(f"{trade_start_ist}-23:59")[0]
    end = parse_hours_range(f"00:00-{trade_end_ist}")[1]
    t = dt.time()
    return start <= t <= end


def is_trade_window_ending(
    trade_end_ist: str,
    dt: datetime | None = None,
    seconds_before: int = 60,
) -> bool:
    """True if we're within *seconds_before* of the trade window ending."""
    dt = dt or now_ist()
    end = parse_hours_range(f"00:00-{trade_end_ist}")[1]
    end_dt = dt.replace(hour=end.hour, minute=end.minute, second=0, microsecond=0)
    remaining = (end_dt - dt).total_seconds()
    return 0 < remaining <= seconds_before


def next_market_open(
    hours_str: str,
    dt: datetime | None = None,
    holidays: list[str] | None = None,
) -> datetime:
    dt = dt or now_ist()
    start_time, _ = parse_hours_range(hours_str)
    open_today = dt.replace(
        hour=start_time.hour,
        minute=start_time.minute,
        second=0,
        microsecond=0,
    )

    if is_trading_day(dt, holidays) and dt < open_today:
        return open_today

    day: date = dt.date()
    for _ in range(14):
        day += timedelta(days=1)
        candidate = datetime.combine(day, start_time, tzinfo=IST)
        if is_trading_day(candidate, holidays):
            return candidate

    raise RuntimeError("Could not find next NSE trading session within 14 days")


def get_market_session(
    hours_str: str,
    dt: datetime | None = None,
    holidays: list[str] | None = None,
) -> MarketSession:
    dt = dt or now_ist()
    start, end = parse_hours_range(hours_str)

    if not is_trading_day(dt, holidays):
        reason = "weekend" if dt.weekday() >= 5 else "holiday"
        nxt = next_market_open(hours_str, dt, holidays)
        return MarketSession(
            is_open=False,
            reason=reason,
            next_open=nxt,
            seconds_until_open=max(0, int((nxt - dt).total_seconds())),
        )

    t = dt.time()
    if t < start:
        open_dt = dt.replace(
            hour=start.hour,
            minute=start.minute,
            second=0,
            microsecond=0,
        )
        return MarketSession(
            is_open=False,
            reason="before_open",
            next_open=open_dt,
            seconds_until_open=max(0, int((open_dt - dt).total_seconds())),
        )

    if t > end:
        nxt = next_market_open(hours_str, dt, holidays)
        return MarketSession(
            is_open=False,
            reason="after_close",
            next_open=nxt,
            seconds_until_open=max(0, int((nxt - dt).total_seconds())),
        )

    return MarketSession(is_open=True, reason="open", next_open=None, seconds_until_open=0)


def compute_idle_sleep_seconds(
    session: MarketSession,
    *,
    monitor_interval: int,
    closed_poll_interval: int,
) -> int:
    """Sleep duration for 24x7 loop — shorter polls near the open."""
    if session.is_open:
        return max(1, monitor_interval)

    poll = max(30, closed_poll_interval)
    if session.seconds_until_open <= poll:
        return max(30, min(session.seconds_until_open, monitor_interval))
    return poll


def format_next_open(session: MarketSession) -> str:
    if session.next_open is None:
        return "now"
    return session.next_open.strftime("%a %d %b %Y %H:%M IST")


def is_square_off_time(square_off: str, dt: datetime | None = None) -> bool:
    dt = dt or now_ist()
    h, m = map(int, square_off.split(":"))
    cutoff = time(h, m)
    return dt.time() >= cutoff


def minutes_until_event(event_time: datetime, dt: datetime | None = None) -> float:
    dt = dt or now_ist()
    if event_time.tzinfo is None:
        event_time = event_time.replace(tzinfo=IST)
    return abs((event_time - dt).total_seconds()) / 60.0


def is_event_blackout(
    scheduled_events: list[str],
    blackout_minutes: int,
    dt: datetime | None = None,
) -> tuple[bool, str]:
    dt = dt or now_ist()
    for ev in scheduled_events:
        try:
            event_dt = datetime.fromisoformat(ev)
            if event_dt.tzinfo is None:
                event_dt = event_dt.replace(tzinfo=IST)
            mins = minutes_until_event(event_dt, dt)
            if mins <= blackout_minutes:
                return True, f"Within {blackout_minutes}m of event {ev}"
        except ValueError:
            continue
    return False, "ok"
