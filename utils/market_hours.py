"""IST market hours and square-off helpers."""

from __future__ import annotations

from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")


def parse_hours_range(hours_str: str) -> tuple[time, time]:
    start_s, end_s = hours_str.split("-")
    sh, sm = map(int, start_s.strip().split(":"))
    eh, em = map(int, end_s.strip().split(":"))
    return time(sh, sm), time(eh, em)


def now_ist() -> datetime:
    return datetime.now(IST)


def is_within_trading_hours(hours_str: str, dt: datetime | None = None) -> bool:
    dt = dt or now_ist()
    start, end = parse_hours_range(hours_str)
    t = dt.time()
    return start <= t <= end


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
