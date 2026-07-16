"""Smoke checks for colored log helpers."""

from __future__ import annotations

from scalping.logfmt import (
    ColoredISTFormatter,
    fmt_pct,
    fmt_rupees,
    paint,
    strip_ansi,
    C,
)


def test_fmt_rupees_signs():
    assert "+500" in strip_ansi(fmt_rupees(500))
    assert "-500" in strip_ansi(fmt_rupees(-500))
    assert "Rs" in strip_ansi(fmt_rupees(100))
    assert "\033[" in fmt_rupees(100)
    assert "\033[" not in strip_ansi(fmt_rupees(100))


def test_fmt_pct():
    assert "+12.5%" in strip_ansi(fmt_pct(0.125))
    assert "-8.0%" in strip_ansi(fmt_pct(-0.08))


def test_file_formatter_strips_ansi():
    import logging

    record = logging.LogRecord(
        name="sniper", level=logging.INFO, pathname="", lineno=0,
        msg=paint("CLOSED PnL=₹+500", C.GREEN), args=(), exc_info=None,
    )
    # Simulate plain path: strip after format
    colored = ColoredISTFormatter("%(message)s", "%H:%M:%S").format(record)
    assert "\033[" in colored
    assert "CLOSED" in strip_ansi(colored)
    assert "\033[" not in strip_ansi(colored)
