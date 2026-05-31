"""Market hours tests."""

from datetime import datetime
from zoneinfo import ZoneInfo

from utils.market_hours import (
    compute_idle_sleep_seconds,
    get_market_session,
    is_event_blackout,
    is_market_open,
    is_square_off_time,
    is_trading_day,
    is_within_trading_hours,
    next_market_open,
    parse_hours_range,
)

IST = ZoneInfo("Asia/Kolkata")


def test_parse_hours():
    start, end = parse_hours_range("09:15-15:15")
    assert start.hour == 9 and start.minute == 15
    assert end.hour == 15 and end.minute == 15


def test_within_trading_hours():
    dt = datetime(2026, 6, 2, 10, 0, tzinfo=IST)  # Tuesday
    assert is_within_trading_hours("09:15-15:15", dt)


def test_outside_trading_hours():
    dt = datetime(2026, 6, 2, 8, 0, tzinfo=IST)
    assert not is_within_trading_hours("09:15-15:15", dt)


def test_square_off():
    dt = datetime(2026, 6, 2, 15, 20, tzinfo=IST)
    assert is_square_off_time("15:15", dt)


def test_event_blackout():
    dt = datetime(2026, 6, 2, 10, 0, tzinfo=IST)
    events = ["2026-06-02T10:10:00"]
    blocked, _ = is_event_blackout(events, 15, dt)
    assert blocked


def test_weekend_not_trading_day():
    sunday = datetime(2026, 5, 31, 10, 0, tzinfo=IST)
    assert not is_trading_day(sunday)
    assert not is_market_open("09:15-15:15", sunday)


def test_before_open_session():
    dt = datetime(2026, 6, 2, 8, 30, tzinfo=IST)
    session = get_market_session("09:15-15:15", dt)
    assert not session.is_open
    assert session.reason == "before_open"
    assert session.next_open is not None
    assert session.next_open.hour == 9 and session.next_open.minute == 15


def test_after_close_session():
    dt = datetime(2026, 6, 2, 16, 0, tzinfo=IST)
    session = get_market_session("09:15-15:15", dt)
    assert not session.is_open
    assert session.reason == "after_close"
    assert session.next_open is not None
    assert session.next_open.date() == datetime(2026, 6, 3, tzinfo=IST).date()


def test_next_open_skips_weekend():
    friday_close = datetime(2026, 6, 5, 16, 0, tzinfo=IST)
    nxt = next_market_open("09:15-15:15", friday_close)
    assert nxt.weekday() == 0  # Monday
    assert nxt.hour == 9 and nxt.minute == 15


def test_holiday_skipped():
    dt = datetime(2026, 6, 2, 10, 0, tzinfo=IST)
    holidays = ["2026-06-02"]
    assert not is_trading_day(dt, holidays)
    session = get_market_session("09:15-15:15", dt, holidays)
    assert session.reason == "holiday"


def test_idle_sleep_shorter_near_open():
    session = get_market_session(
        "09:15-15:15",
        datetime(2026, 6, 2, 9, 10, tzinfo=IST),
    )
    sleep_for = compute_idle_sleep_seconds(
        session,
        monitor_interval=60,
        closed_poll_interval=300,
    )
    assert sleep_for <= 60


def test_idle_sleep_longer_when_weekend():
    session = get_market_session(
        "09:15-15:15",
        datetime(2026, 5, 31, 10, 0, tzinfo=IST),
    )
    sleep_for = compute_idle_sleep_seconds(
        session,
        monitor_interval=60,
        closed_poll_interval=300,
    )
    assert sleep_for == 300
