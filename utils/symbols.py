"""F&O symbol detection and validation."""

from __future__ import annotations

import re

# NSE index / underlying identifiers commonly used in env or broker feeds
KNOWN_FNO_UNDERLYINGS: frozenset[str] = frozenset(
    {
        "NIFTY",
        "NIFTY50",
        "NIFTY50IDX",
        "NIFTYIDX",
        "NIFTY INDEX",
        "BANKNIFTY",
        "BANK NIFTY",
        "FINNIFTY",
        "MIDCPNIFTY",
        "SENSEX",
        "BANKEX",
    }
)

# Major NSE F&O single-stock underlyings (non-exhaustive; extend as needed)
KNOWN_FNO_STOCKS: frozenset[str] = frozenset(
    {
        "RELIANCE",
        "TCS",
        "INFY",
        "HDFCBANK",
        "ICICIBANK",
        "SBIN",
        "AXISBANK",
        "ITC",
        "LT",
        "HINDUNILVR",
        "BHARTIARTL",
        "KOTAKBANK",
        "TATASTEEL",
        "TATAMOTORS",
        "MARUTI",
        "SUNPHARMA",
        "WIPRO",
        "HCLTECH",
        "ASIANPAINT",
        "BAJFINANCE",
        # User-requested stock options
        "DIXON",
        "JUBLFOOD",
        "HPCL",
        "INDUSTOWER",
        "KFINTECH",
        "EXIDEIND",
    }
)

_FNO_CONTRACT_RE = re.compile(
    r"(\d{2}[A-Z]{3}FUT|FUT\d|[\d]{2}(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[\d]{2}(FUT)?|[\d]+CE|[\d]+PE|\bCE\b|\bPE\b|FUT\b)",
    re.IGNORECASE,
)
_INDEX_TOKEN_RE = re.compile(r"(NIFTY|BANKNIFTY|FINNIFTY|MIDCPNIFTY|SENSEX|BANKEX)", re.IGNORECASE)


def normalize_symbol(symbol: str) -> str:
    return symbol.strip().upper().replace(" ", "")


def underlying_base(symbol: str) -> str:
    s = normalize_symbol(symbol)
    s = re.sub(r"\d{2}[A-Z]{3}FUT.*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\d+(CE|PE).*$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"(IDX|INDEX)$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"(OPT|FUT)$", "", s, flags=re.IGNORECASE)  # strip OPT/FUT tags
    return s


def is_fno_eligible(symbol: str) -> bool:
    """Return True if symbol represents an F&O-tradable underlying or contract."""
    s = normalize_symbol(symbol)
    if not s:
        return False

    if s in {u.replace(" ", "") for u in KNOWN_FNO_UNDERLYINGS}:
        return True

    if _FNO_CONTRACT_RE.search(s):
        return True

    if s.endswith("IDX") and _INDEX_TOKEN_RE.search(s):
        return True

    base = underlying_base(s)
    if base in KNOWN_FNO_STOCKS:
        return True

    # Index shorthand without IDX suffix (e.g. BANKNIFTY)
    if base in {u.replace(" ", "") for u in KNOWN_FNO_UNDERLYINGS}:
        return True

    return False


def require_fno_symbol(symbol: str) -> str:
    s = normalize_symbol(symbol)
    if not is_fno_eligible(s):
        raise ValueError(
            f"TRADING_SYMBOL '{symbol}' is not F&O eligible. "
            "Set an index/stock with NSE F&O (e.g. NIFTY50IDX, NIFTY25JUNFUT, BANKNIFTY)."
        )
    return s


def same_underlying_strike(a: str, b: str) -> bool:
    """Check if two option symbols share the same underlying and strike (ignoring CE/PE)."""
    import re as _re

    a = normalize_symbol(a)
    b = normalize_symbol(b)

    def _parts(s: str) -> tuple[str, str] | None:
        m = _re.match(r"^([A-Z]+)(\d{2}[A-Z]{3})(\d+)(CE|PE)$", s)
        if m:
            return (m.group(1) + m.group(2), m.group(3))  # e.g., ("NIFTY26JUN", "23350")
        return None

    pa = _parts(a)
    pb = _parts(b)
    if pa and pb:
        return pa == pb
    return False
