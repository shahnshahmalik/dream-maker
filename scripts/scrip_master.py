"""Dhan Scrip Master — downloader, parser, and lookup for F&O instruments.

Downloads the daily scrip master CSV from Dhan's public URL, parses
NSE F&O rows (OPTIDX/FUTIDX/OPTSTK/FUTSTK), stores them in SQLite,
and provides fast lookups by underlying + expiry + strike + option_type.

Usage:
    python scripts/scrip_master.py              # download + store
    python scripts/scrip_master.py --lookup     # verify with a test query

From code:
    from scripts.scrip_master import ScripMaster
    sm = ScripMaster()
    sid = sm.get_security_id("BANKNIFTY", "2026-06-05", 54400, "CE")
    # → 12345 (numeric Dhan security ID)
"""
from __future__ import annotations

import csv
import logging
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Iterator

import requests

log = logging.getLogger("scrip_master")

# ------------------------------------------------------------------ #
# Constants
# ------------------------------------------------------------------ #
SCRIP_MASTER_URL = "https://images.dhan.co/api-data/api-scrip-master.csv"
DB_PATH = Path(__file__).resolve().parent.parent / "data" / "scrip_master.db"
CSV_CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "scrip_master.csv"

# We only care about NSE F&O instruments
# SEM_EXM_EXCH_ID=NSE, SEM_SEGMENT=D (NSE_FNO)
TARGET_EXCHANGE = "NSE"
TARGET_SEGMENT = "D"

# Instrument types we want
WANTED_INSTRUMENTS = {"OPTIDX", "FUTIDX", "OPTSTK", "FUTSTK"}

# CSV column indices (0-based) — parsed from header
COL_MAP: dict[str, int] = {}
HEADER_EXPECTED = [
    "SEM_EXM_EXCH_ID",
    "SEM_SEGMENT",
    "SEM_SMST_SECURITY_ID",
    "SEM_INSTRUMENT_NAME",
    "SEM_EXPIRY_CODE",
    "SEM_TRADING_SYMBOL",
    "SEM_LOT_UNITS",
    "SEM_CUSTOM_SYMBOL",
    "SEM_EXPIRY_DATE",
    "SEM_STRIKE_PRICE",
    "SEM_OPTION_TYPE",
    "SEM_TICK_SIZE",
    "SEM_EXPIRY_FLAG",
    "SEM_EXCH_INSTRUMENT_TYPE",
    "SEM_SERIES",
    "SM_SYMBOL_NAME",
]

# Request timeout
REQUEST_TIMEOUT = 60  # seconds — CSV can be 50+ MB


@dataclass
class ScripRecord:
    security_id: int
    trading_symbol: str
    instrument_name: str  # OPTIDX, FUTIDX, OPTSTK, FUTSTK
    expiry_date: str      # "2026-06-05 14:30:00"
    strike_price: float
    option_type: str       # "CE", "PE", or "XX"
    lot_size: int
    tick_size: float
    symbol_name: str       # extracted underlying, e.g., "BANKNIFTY"
    updated_at: str = ""   # ISO timestamp of when this row was stored


class ScripMaster:
    """Download, store, and query Dhan's F&O scrip master."""

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None

    # ------------------------------------------------------------------ #
    # Connection management
    # ------------------------------------------------------------------ #
    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
        return self._conn

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    # ------------------------------------------------------------------ #
    # Schema
    # ------------------------------------------------------------------ #
    def ensure_schema(self) -> None:
        """Create the scrip_master table if it doesn't exist."""
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS scrip_master (
                security_id    INTEGER PRIMARY KEY,
                trading_symbol TEXT    NOT NULL,
                instrument_name TEXT   NOT NULL,
                expiry_date    TEXT    NOT NULL,
                strike_price   REAL    NOT NULL DEFAULT 0,
                option_type    TEXT    NOT NULL DEFAULT '',
                lot_size       INTEGER NOT NULL DEFAULT 1,
                tick_size      REAL    NOT NULL DEFAULT 0.05,
                symbol_name    TEXT    NOT NULL,
                updated_at     TEXT    NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_lookup
                ON scrip_master(symbol_name, expiry_date, strike_price, option_type);

            CREATE INDEX IF NOT EXISTS idx_trading_symbol
                ON scrip_master(trading_symbol);

            CREATE INDEX IF NOT EXISTS idx_instrument_name
                ON scrip_master(instrument_name);
        """)
        self.conn.commit()

    # ------------------------------------------------------------------ #
    # Download + parse
    # ------------------------------------------------------------------ #
    def download(self, force: bool = False) -> Path:
        """Download the scrip master CSV. Returns path to cached file."""
        if not force and CSV_CACHE_PATH.exists():
            age_hours = (datetime.now().timestamp() - CSV_CACHE_PATH.stat().st_mtime) / 3600
            if age_hours < 12:
                log.info("Using cached scrip master (%.1f hours old)", age_hours)
                return CSV_CACHE_PATH

        log.info("Downloading scrip master from %s ...", SCRIP_MASTER_URL)
        resp = requests.get(SCRIP_MASTER_URL, timeout=REQUEST_TIMEOUT,
                            stream=True)
        resp.raise_for_status()

        # Write to cache
        CSV_CACHE_PATH.write_bytes(resp.content)
        size_mb = len(resp.content) / (1024 * 1024)
        log.info("Downloaded %.1f MB → %s", size_mb, CSV_CACHE_PATH)
        return CSV_CACHE_PATH

    def parse_and_store(self, csv_path: Path | None = None) -> int:
        """Parse the CSV and store F&O rows in SQLite. Returns row count."""
        path = csv_path or CSV_CACHE_PATH
        if not path.exists():
            raise FileNotFoundError(f"Scrip master CSV not found: {path}")

        self.ensure_schema()

        # Clear existing data
        self.conn.execute("DELETE FROM scrip_master")

        updated_at = datetime.utcnow().isoformat()
        count = 0

        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header is None:
                raise ValueError("Empty CSV — no header row")

            # Build column index map
            col_map = {name.strip(): i for i, name in enumerate(header)}
            required = ["SEM_EXM_EXCH_ID", "SEM_SEGMENT", "SEM_SMST_SECURITY_ID",
                         "SEM_INSTRUMENT_NAME"]
            missing = [c for c in required if c not in col_map]
            if missing:
                raise ValueError(f"CSV missing required columns: {missing}")

            for row in reader:
                if len(row) < len(required):
                    continue

                exchange = _col(row, col_map, "SEM_EXM_EXCH_ID")
                segment = _col(row, col_map, "SEM_SEGMENT")
                if exchange != TARGET_EXCHANGE or segment != TARGET_SEGMENT:
                    continue

                instrument = _col(row, col_map, "SEM_INSTRUMENT_NAME")
                if instrument not in WANTED_INSTRUMENTS:
                    continue

                try:
                    record = _parse_row(row, col_map, updated_at)
                except (ValueError, IndexError) as e:
                    log.debug("Skipping malformed row: %s", e)
                    continue

                self._insert(record)
                count += 1

        self.conn.commit()
        log.info("Stored %d F&O instruments in %s", count, self.db_path)
        return count

    def _insert(self, rec: ScripRecord) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO scrip_master
               (security_id, trading_symbol, instrument_name, expiry_date,
                strike_price, option_type, lot_size, tick_size,
                symbol_name, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (rec.security_id, rec.trading_symbol, rec.instrument_name,
             rec.expiry_date, rec.strike_price, rec.option_type,
             rec.lot_size, rec.tick_size, rec.symbol_name, rec.updated_at),
        )

    # ------------------------------------------------------------------ #
    # Refresh (one-shot: download + parse + store)
    # ------------------------------------------------------------------ #
    def refresh(self, force_download: bool = False) -> int:
        """Download and store fresh scrip master. Returns row count."""
        csv_path = self.download(force=force_download)
        return self.parse_and_store(csv_path)

    # ------------------------------------------------------------------ #
    # Lookup
    # ------------------------------------------------------------------ #
    def get_security_id(
        self,
        symbol: str,          # e.g., "BANKNIFTY"
        expiry_date: str,     # e.g., "2026-06-05"
        strike: float,        # e.g., 54400
        option_type: str,     # "CE" or "PE"
    ) -> int | None:
        """Look up the Dhan numeric security ID for an option contract."""
        expiry_formatted = _normalize_expiry(expiry_date)
        row = self.conn.execute(
            """SELECT security_id FROM scrip_master
               WHERE symbol_name = ?
                 AND date(expiry_date) = date(?)
                 AND strike_price = ?
                 AND option_type = ?
               LIMIT 1""",
            (symbol.upper(), expiry_formatted, float(strike), option_type.upper()),
        ).fetchone()
        return int(row["security_id"]) if row else None

    def get_by_trading_symbol(self, trading_symbol: str) -> int | None:
        """Look up security ID by exact trading_symbol match (CSV format).

        The trading_symbol in the CSV is like 'BANKNIFTY-Jun2026-65400-CE'.
        Our internal format is 'BANKNIFTY26JUN65400CE'.
        This method normalizes between them.
        """
        # Try exact match first
        row = self.conn.execute(
            "SELECT security_id FROM scrip_master WHERE trading_symbol = ? LIMIT 1",
            (trading_symbol,),
        ).fetchone()
        if row:
            return int(row["security_id"])

        # Convert internal format to CSV format: BANKNIFTY26JUN54400CE → BANKNIFTY-Jun2026-54400-CE
        import re
        m = re.match(
            r"^(?P<underlying>[A-Z]+)(?P<yy>\d{2})(?P<month>[A-Z]{3})"
            r"(?P<strike>\d+)(?P<type>CE|PE)$",
            trading_symbol.upper(),
        )
        if m:
            yy = int(m.group("yy"))
            full_year = 2000 + yy
            month_name = m.group("month").capitalize()
            csv_format = (
                f"{m.group('underlying')}-{month_name}{full_year}-"
                f"{m.group('strike')}-{m.group('type')}"
            )
            row = self.conn.execute(
                "SELECT security_id FROM scrip_master WHERE trading_symbol = ? LIMIT 1",
                (csv_format,),
            ).fetchone()
            if row:
                return int(row["security_id"])

        return None

    def get_by_security_id(self, security_id: int) -> ScripRecord | None:
        """Get full record by security ID."""
        row = self.conn.execute(
            "SELECT * FROM scrip_master WHERE security_id = ?",
            (security_id,),
        ).fetchone()
        if row is None:
            return None
        return ScripRecord(
            security_id=row["security_id"],
            trading_symbol=row["trading_symbol"],
            instrument_name=row["instrument_name"],
            expiry_date=row["expiry_date"],
            strike_price=row["strike_price"],
            option_type=row["option_type"],
            lot_size=row["lot_size"],
            tick_size=row["tick_size"],
            symbol_name=row["symbol_name"],
            updated_at=row["updated_at"],
        )

    def get_lot_size(self, trading_symbol: str) -> int | None:
        """Get lot size from trading_symbol (e.g., 'BANKNIFTY26JUN54400CE').

        Handles format conversion between internal (BANKNIFTY26JUN54400CE)
        and CSV (BANKNIFTY-Jun2026-54400-CE) formats.
        """
        # Try exact match first
        row = self.conn.execute(
            "SELECT lot_size FROM scrip_master WHERE trading_symbol = ? LIMIT 1",
            (trading_symbol,),
        ).fetchone()
        if row:
            return int(row["lot_size"])

        # Convert internal format to CSV format
        import re
        m = re.match(
            r"^(?P<underlying>[A-Z]+)(?P<yy>\d{2})(?P<month>[A-Z]{3})"
            r"(?P<strike>\d+)(?P<type>CE|PE)$",
            trading_symbol.upper(),
        )
        if m:
            yy = int(m.group("yy"))
            full_year = 2000 + yy
            month_name = m.group("month").capitalize()
            csv_format = (
                f"{m.group('underlying')}-{month_name}{full_year}-"
                f"{m.group('strike')}-{m.group('type')}"
            )
            row2 = self.conn.execute(
                "SELECT lot_size FROM scrip_master WHERE trading_symbol = ? LIMIT 1",
                (csv_format,),
            ).fetchone()
            if row2:
                return int(row2["lot_size"])

        return None

    def stats(self) -> dict:
        """Return summary statistics about stored data."""
        total = self.conn.execute(
            "SELECT COUNT(*) FROM scrip_master"
        ).fetchone()[0]
        by_type = {}
        for inst in WANTED_INSTRUMENTS:
            cnt = self.conn.execute(
                "SELECT COUNT(*) FROM scrip_master WHERE instrument_name = ?",
                (inst,),
            ).fetchone()[0]
            if cnt:
                by_type[inst] = cnt
        return {"total": total, "by_type": by_type}


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #
def _col(row: list[str], col_map: dict[str, int], name: str) -> str:
    idx = col_map.get(name)
    if idx is None or idx >= len(row):
        return ""
    return row[idx].strip()


def _normalize_expiry(date_str: str) -> str:
    """Normalize expiry date to 'YYYY-MM-DD' format."""
    # Already in standard format
    if len(date_str) == 10 and date_str[4] == "-":
        return date_str
    # Try parsing common formats
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%d-%b-%Y", "%d-%m-%Y"):
        try:
            return datetime.strptime(date_str, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return date_str


def _extract_symbol_name(trading_symbol: str, instrument_name: str) -> str:
    """Extract the underlying symbol name from a trading symbol.

    'BANKNIFTY-Jun2026-65400-CE' → 'BANKNIFTY'
    'CGPOWER-Jun2026-840-PE' → 'CGPOWER'
    """
    # Strip known suffixes
    parts = trading_symbol.split("-")
    if len(parts) >= 1:
        return parts[0].strip().upper()
    return trading_symbol.upper()


def _parse_row(row: list[str], col_map: dict[str, int],
               updated_at: str) -> ScripRecord:
    trading_symbol = _col(row, col_map, "SEM_TRADING_SYMBOL")
    instrument_name = _col(row, col_map, "SEM_INSTRUMENT_NAME")

    # Parse numeric fields safely
    def _float_or(col_name: str, default: float) -> float:
        val = _col(row, col_map, col_name)
        try:
            return float(val)
        except (ValueError, TypeError):
            return default

    def _int_or(col_name: str, default: int) -> int:
        val = _col(row, col_map, col_name)
        try:
            return int(float(val))
        except (ValueError, TypeError):
            return default

    return ScripRecord(
        security_id=_int_or("SEM_SMST_SECURITY_ID", 0),
        trading_symbol=trading_symbol,
        instrument_name=instrument_name,
        expiry_date=_col(row, col_map, "SEM_EXPIRY_DATE"),
        strike_price=_float_or("SEM_STRIKE_PRICE", 0),
        option_type=_col(row, col_map, "SEM_OPTION_TYPE"),
        lot_size=_int_or("SEM_LOT_UNITS", 1),
        tick_size=_float_or("SEM_TICK_SIZE", 0.05),
        symbol_name=_extract_symbol_name(trading_symbol, instrument_name),
        updated_at=updated_at,
    )


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    sm = ScripMaster()

    if "--lookup" in sys.argv:
        # Test lookup: BANKNIFTY 54400 CE nearest expiry
        sm.ensure_schema()
        count = sm.conn.execute(
            "SELECT COUNT(*) FROM scrip_master"
        ).fetchone()[0]
        if not count:
            print("No data in scrip_master. Run without --lookup first.")
            sys.exit(1)

        # Find a BANKNIFTY option to test
        row = sm.conn.execute(
            """SELECT * FROM scrip_master
               WHERE symbol_name = 'BANKNIFTY'
                 AND option_type = 'CE'
               ORDER BY expiry_date
               LIMIT 1"""
        ).fetchone()
        if row:
            sid = sm.get_security_id(
                row["symbol_name"], row["expiry_date"],
                row["strike_price"], row["option_type"],
            )
            print(f"Lookup test: {row['symbol_name']} "
                  f"expiry={row['expiry_date'][:10]} "
                  f"strike={row['strike_price']:.0f} "
                  f"{row['option_type']} → security_id={sid}")
        else:
            print("No BANKNIFTY CE data found")

        # Show stats
        stats = sm.stats()
        print(f"\nDB stats: {stats['total']} total instruments")
        for typ, cnt in stats.get("by_type", {}).items():
            print(f"  {typ}: {cnt}")
        sys.exit(0)

    # Default: refresh
    try:
        count = sm.refresh()
        print(f"Done: {count} instruments stored")
    except Exception as e:
        log.exception("Refresh failed: %s", e)
        sys.exit(1)
    finally:
        sm.close()
