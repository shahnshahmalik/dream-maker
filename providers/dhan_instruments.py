"""Dhan exchange security IDs for indices and symbol resolution."""

from __future__ import annotations

import re
from dataclasses import dataclass

from utils.symbols import normalize_symbol, underlying_base

# NSE index underlyings — used for charts / LTP on TRADING_SYMBOL like NIFTY50IDX
INDEX_INSTRUMENTS: dict[str, dict[str, str | int]] = {
    "NIFTY": {"securityId": "13", "exchangeSegment": "IDX_I", "instrument": "INDEX"},
    "NIFTY50": {"securityId": "13", "exchangeSegment": "IDX_I", "instrument": "INDEX"},
    "BANKNIFTY": {"securityId": "25", "exchangeSegment": "IDX_I", "instrument": "INDEX"},
    "FINNIFTY": {"securityId": "27", "exchangeSegment": "IDX_I", "instrument": "INDEX"},
    "MIDCPNIFTY": {"securityId": "442", "exchangeSegment": "IDX_I", "instrument": "INDEX"},
    "SENSEX": {"securityId": "1", "exchangeSegment": "IDX_I", "instrument": "INDEX"},
}


@dataclass(frozen=True)
class DhanInstrument:
    symbol: str
    security_id: str
    exchange_segment: str
    instrument: str
    lot_size: int = 1


def _index_key(symbol: str) -> str | None:
    base = underlying_base(symbol)
    if base in INDEX_INSTRUMENTS:
        return base
    if "NIFTY" in base and "BANK" not in base and "FIN" not in base and "MIDCP" not in base:
        return "NIFTY"
    if base.startswith("NIFTY") and base.endswith("IDX"):
        return "NIFTY"
    return None


def resolve_market_data_instrument(symbol: str) -> DhanInstrument | None:
    """Map a trading symbol to Dhan chart / LTP parameters."""
    sym = normalize_symbol(symbol)
    key = _index_key(sym)
    if key:
        meta = INDEX_INSTRUMENTS[key]
        lot = 25 if key in {"NIFTY", "NIFTY50", "FINNIFTY", "MIDCPNIFTY"} else 15 if key == "BANKNIFTY" else 1
        return DhanInstrument(
            symbol=sym,
            security_id=str(meta["securityId"]),
            exchange_segment=str(meta["exchangeSegment"]),
            instrument=str(meta["instrument"]),
            lot_size=lot,
        )

    # Explicit F&O contract — security ID must be numeric; caller may enrich via scrip master later.
    if re.search(r"\d+(CE|PE)$", sym, re.IGNORECASE) or "FUT" in sym:
        return DhanInstrument(
            symbol=sym,
            security_id="",
            exchange_segment="NSE_FNO",
            instrument="OPTIDX" if re.search(r"(CE|PE)$", sym, re.IGNORECASE) else "FUTIDX",
            lot_size=25 if "NIFTY" in sym else 1,
        )

    return None
