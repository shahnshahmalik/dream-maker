"""Engine session stop/wait behaviour."""

from datetime import datetime
from zoneinfo import ZoneInfo

from utils.market_hours import MarketSession, get_market_session

IST = ZoneInfo("Asia/Kolkata")


def _should_stop(session: MarketSession, *, stop_at_close: bool, wait_for_open: bool) -> bool:
    if session.is_open:
        return False
    if not stop_at_close:
        return False
    if session.reason == "after_close":
        return True
    if wait_for_open and session.reason in {"before_open", "weekend", "holiday"}:
        return False
    return True


def test_stop_after_close():
    session = get_market_session("09:15-15:15", datetime(2026, 6, 2, 16, 0, tzinfo=IST))
    assert _should_stop(session, stop_at_close=True, wait_for_open=True)


def test_wait_before_open():
    session = get_market_session("09:15-15:15", datetime(2026, 6, 2, 8, 0, tzinfo=IST))
    assert not _should_stop(session, stop_at_close=True, wait_for_open=True)


def test_wait_over_weekend():
    session = get_market_session("09:15-15:15", datetime(2026, 5, 31, 10, 0, tzinfo=IST))
    assert not _should_stop(session, stop_at_close=True, wait_for_open=True)


def test_no_stop_when_disabled():
    session = get_market_session("09:15-15:15", datetime(2026, 6, 2, 16, 0, tzinfo=IST))
    assert not _should_stop(session, stop_at_close=False, wait_for_open=True)
