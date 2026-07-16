"""Terminal-colored logging for sniper scalpers. File logs stay plain."""

from __future__ import annotations

import logging
import os
import re
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")


def _enable_windows_ansi() -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)
    except Exception:
        pass


def _configure_stdout_utf8() -> None:
    """Avoid cp1252 UnicodeEncodeError for fancy chars on Windows consoles."""
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except Exception:
        pass


_enable_windows_ansi()
_configure_stdout_utf8()


class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    RED     = "\033[91m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    BLUE    = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN    = "\033[96m"
    WHITE   = "\033[97m"
    GRAY    = "\033[90m"


_ANSI_RE = re.compile(r"\033\[[0-9;]*m")


def strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def paint(text: str, *codes: str) -> str:
    if not codes:
        return text
    return f"{''.join(codes)}{text}{C.RESET}"


def pnl_color(value: float) -> str:
    return C.GREEN if value > 0 else C.RED if value < 0 else C.YELLOW


def fmt_rupees(value: float, *, signed: bool = True, bold: bool = False) -> str:
    """Colored Rs amount for terminal (ANSI embedded)."""
    if signed:
        s = f"Rs{value:+,.0f}" if abs(value) >= 1 else f"Rs{value:+.2f}"
    else:
        s = f"Rs{value:,.0f}" if abs(value) >= 1 else f"Rs{value:.2f}"
    style = (C.BOLD,) if bold else ()
    return paint(s, pnl_color(value), *style)


def fmt_pct(value: float, *, bold: bool = False) -> str:
    s = f"{value * 100:+.1f}%"
    style = (C.BOLD,) if bold else ()
    return paint(s, pnl_color(value), *style)


class ISTFormatter(logging.Formatter):
    """Plain IST timestamps — used for file logs."""

    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        ct = datetime.fromtimestamp(record.created, tz=IST)
        return ct.strftime(datefmt or "%H:%M:%S")


class ColoredISTFormatter(ISTFormatter):
    """ANSI colors for stdout only. Auto-tints PnL / exits / signals."""

    _LEVEL = {
        logging.DEBUG:    C.GRAY,
        logging.INFO:     C.CYAN,
        logging.WARNING:  C.YELLOW,
        logging.ERROR:    C.RED,
        logging.CRITICAL: C.BOLD + C.RED,
    }

    def format(self, record: logging.LogRecord) -> str:
        msg = record.getMessage()
        colored_msg = self._colorize_message(msg, record.levelno)

        asctime = self.formatTime(record, self.datefmt)
        level = record.levelname
        level_c = self._LEVEL.get(record.levelno, C.WHITE)

        line = (
            f"{paint(asctime, C.GRAY)} | "
            f"{paint(f'{level:<7}', level_c)} | "
            f"{colored_msg}"
        )
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line

    def _colorize_message(self, msg: str, levelno: int) -> str:
        if "\033[" in msg:
            return msg

        upper = msg.upper()

        if msg.startswith("=") or msg.startswith("--") or msg.startswith("=== "):
            return paint(msg, C.BOLD, C.BLUE)

        if upper.startswith("EXIT TP") or " EXIT TP" in upper:
            return paint(msg, C.BOLD, C.GREEN)
        if upper.startswith("EXIT TRAIL") or upper.startswith("EXIT CAPITAL"):
            return paint(msg, C.BOLD, C.GREEN)
        if upper.startswith("EXIT SL"):
            return paint(msg, C.BOLD, C.RED)
        if upper.startswith("EXIT THETA"):
            return paint(msg, C.YELLOW)

        if upper.startswith("CLOSED"):
            if re.search(r"Rs-\d", msg) or "Rs-" in msg:
                return paint(msg, C.BOLD, C.RED)
            if "Rs+" in msg or "Rs" in msg:
                return paint(msg, C.BOLD, C.GREEN)
            return msg

        if upper.startswith("FINAL"):
            m = re.search(r"Rs([+-]?\d+)", msg)
            if m:
                return paint(msg, C.BOLD, pnl_color(float(m.group(1))))
            return paint(msg, C.BOLD)

        if upper.startswith("ENTER BUY") or upper.startswith("ACTIVE "):
            return paint(msg, C.BOLD, C.CYAN)
        if upper.startswith("SIGNAL "):
            return paint(msg, C.BOLD, C.MAGENTA)
        if upper.startswith("ORDER BUY"):
            return paint(msg, C.CYAN)
        if upper.startswith("ORDER SELL"):
            return paint(msg, C.YELLOW)

        if "GATE" in upper or upper.startswith("SNIPER GATE"):
            return paint(msg, C.YELLOW)
        if upper.startswith("HALT") or upper.startswith("HALTED"):
            if "daily_target" in msg.lower():
                return paint(msg, C.BOLD, C.GREEN)
            return paint(msg, C.BOLD, C.RED)

        if upper.startswith("TRAIL SL"):
            return paint(msg, C.GREEN)

        if upper.startswith("MONITOR "):
            return self._color_monitor(msg)

        if "active=" in msg.lower() and "daily" in msg.lower():
            return self._color_tick_header(msg)

        if levelno >= logging.ERROR:
            return paint(msg, C.RED)
        if levelno >= logging.WARNING:
            return paint(msg, C.YELLOW)

        return msg

    def _color_monitor(self, msg: str) -> str:
        m = re.search(r"([+-]?\d+\.?\d*)%\s*\((Rs[+-]?[\d,.]+)\)", msg)
        if not m:
            return msg
        pct = float(m.group(1))
        rs_m = re.search(r"([+-]?[\d.]+)", m.group(2).replace(",", ""))
        rs = float(rs_m.group(1)) if rs_m else pct
        col = pnl_color(rs if rs != 0 else pct)
        colored_pnl = paint(f"{pct:+.1f}% ({m.group(2)})", col, C.BOLD)
        return msg[: m.start()] + colored_pnl + msg[m.end() :]

    def _color_tick_header(self, msg: str) -> str:
        m = re.search(r"(Rs[+-]?[\d,.]+)", msg)
        if not m:
            return paint(msg, C.DIM)
        rs_m = re.search(r"([+-]?[\d.]+)", m.group(1).replace(",", ""))
        val = float(rs_m.group(1)) if rs_m else 0.0
        colored = paint(m.group(1), pnl_color(val), C.BOLD)
        return paint(msg[: m.start()], C.DIM) + colored + paint(msg[m.end() :], C.DIM)


def setup_colored_logging(log_dir, log_filename: str) -> logging.Logger:
    """Attach plain file handler + colored stream handler."""
    _configure_stdout_utf8()
    _enable_windows_ansi()

    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("sniper")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    class _PlainFileFormatter(ISTFormatter):
        def format(self, record: logging.LogRecord) -> str:
            return strip_ansi(super().format(record))

    fh = logging.FileHandler(log_dir / log_filename, encoding="utf-8")
    fh.setFormatter(_PlainFileFormatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S"))

    use_color = sys.stdout.isatty() or os.getenv("FORCE_COLOR", "").strip().lower() in {
        "1", "true", "yes",
    }
    sh = logging.StreamHandler(sys.stdout)
    if use_color:
        sh.setFormatter(ColoredISTFormatter("%(message)s", "%H:%M:%S"))
    else:
        sh.setFormatter(ISTFormatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S"))

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger
