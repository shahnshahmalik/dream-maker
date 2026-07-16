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
    from scalping.logfmt import ISTFormatter

    class _PlainFileFormatter(ISTFormatter):
        def format(self, record: logging.LogRecord) -> str:
            return strip_ansi(super().format(record))

    formatter = _PlainFileFormatter("%(asctime)s | %(levelname)-7s | %(message)s")
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname="", lineno=0,
        msg=paint("Hello World", C.BOLD, C.GREEN), args=(), exc_info=None,
    )
    result = formatter.format(record)
    assert "\033[" not in result
    assert "Hello World" in result


def test_paint_adds_reset():
    result = paint("text", C.BOLD)
    assert result.startswith(C.BOLD)
    assert result.endswith(C.RESET)


def test_strip_ansi_removes_all_codes():
    s = f"{C.BOLD}{C.RED}hello{C.RESET} world"
    assert strip_ansi(s) == "hello world"


def test_colored_formatter_exit_tp():
    import logging
    fmt = ColoredISTFormatter("%(message)s")
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname="", lineno=0,
        msg="EXIT TP @ Rs152.30", args=(), exc_info=None,
    )
    result = fmt.format(record)
    assert C.GREEN in result


def test_colored_formatter_exit_sl():
    import logging
    fmt = ColoredISTFormatter("%(message)s")
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname="", lineno=0,
        msg="EXIT SL @ Rs28.00", args=(), exc_info=None,
    )
    result = fmt.format(record)
    assert C.RED in result
