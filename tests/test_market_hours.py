"""Market hours tests."""

from datetime import datetime
from zoneinfo import ZoneInfo

from utils.market_hours import (
    is_event_blackout,
    is_square_off_time,
    is_within_trading_hours,
    parse_hours_range,
)

IST = ZoneInfo("Asia/Kolkata")


def test_parse_hours():
    start, end = parse_hours_range("09:15-15:15")
    assert start.hour == 9 and start.minute == 15
    assert end.hour == 15 and end.minute == 15


def test_within_trading_hours():
    dt = datetime(2026, 5, 31, 10, 0, tzinfo=IST)
    assert is_within_trading_hours("09:15-15:15", dt)


def test_outside_trading_hours():
    dt = datetime(2026, 5, 31, 8, 0, tzinfo=IST)
    assert not is_within_trading_hours("09:15-15:15", dt)


def test_square_off():
    dt = datetime(2026, 5, 31, 15, 20, tzinfo=IST)
    assert is_square_off_time("15:15", dt)


def test_event_blackout():
    dt = datetime(2026, 5, 31, 10, 0, tzinfo=IST)
    events = ["2026-05-31T10:10:00"]
    blocked, _ = is_event_blackout(events, 15, dt)
    assert blocked
