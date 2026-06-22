"""Indian market data helpers: India VIX, NSE Option Chain (Max Pain + OI walls).

Data sources:
  - India VIX: NSE public API (no auth required)
  - Option Chain: NSE public API /api/option-chain-indices

All functions are fault-tolerant — return None/defaults on failure so
the calling scalper can degrade gracefully instead of crashing.

Usage:
    from utils.market_data import get_india_vix, get_option_chain_context

    vix = get_india_vix()           # float or None
    ctx = get_option_chain_context("NIFTY", spot, expiry_date_str)
    # ctx.max_pain, ctx.ce_wall, ctx.pe_wall, ctx.pcr, ctx.bias
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import requests

log = logging.getLogger("market_data")

# ── Constants ─────────────────────────────────────────────────────────────────

NSE_VIX_URL           = "https://www.nseindia.com/api/allIndices"
NSE_OPTION_CHAIN_URL  = "https://www.nseindia.com/api/option-chain-indices"
NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/option-chain",
}

_SESSION: requests.Session | None = None
_SESSION_WARMED_AT: float = 0.0
SESSION_TTL_SEC = 300  # re-warm every 5 minutes

VIX_BLOCK_THRESHOLD = 22.0   # don't enter new trades above this VIX
VIX_CAUTION_THRESHOLD = 18.0  # reduce confidence above this level


@dataclass
class OptionChainContext:
    """Key option chain metrics for trading decisions."""
    max_pain: float            # strike where max options expire worthless
    ce_wall: float             # highest OI CE strike (resistance)
    pe_wall: float             # highest OI PE strike (support)
    pcr: float                 # Put-Call Ratio (OI-based)
    bias: str                  # "bullish" | "bearish" | "neutral"
    total_ce_oi: int
    total_pe_oi: int


# ── Session management ────────────────────────────────────────────────────────

def _get_session() -> requests.Session:
    """Return a warmed NSE session. Re-warms if TTL expired."""
    global _SESSION, _SESSION_WARMED_AT

    now = time.time()
    if _SESSION is None or (now - _SESSION_WARMED_AT) > SESSION_TTL_SEC:
        sess = requests.Session()
        sess.headers.update(NSE_HEADERS)
        try:
            # Warm the session — NSE requires a homepage hit before API calls
            sess.get("https://www.nseindia.com", timeout=8)
            _SESSION_WARMED_AT = now
        except Exception as exc:
            log.debug("NSE session warm failed (non-fatal): %s", exc)
        _SESSION = sess

    return _SESSION


# ── India VIX ─────────────────────────────────────────────────────────────────

def get_india_vix(timeout: int = 6) -> float | None:
    """Fetch current India VIX from NSE.

    Returns the VIX value (e.g. 14.25) or None on failure.
    """
    try:
        sess = _get_session()
        resp = sess.get(NSE_VIX_URL, timeout=timeout)
        resp.raise_for_status()
        data = resp.json().get("data", [])
        for item in data:
            if item.get("index", "").upper() == "INDIA VIX":
                vix = float(item.get("last", 0))
                log.debug("India VIX: %.2f", vix)
                return vix
        log.warning("India VIX not found in NSE response")
        return None
    except Exception as exc:
        log.warning("get_india_vix failed: %s", exc)
        return None


def vix_allows_entry(vix: float | None) -> tuple[bool, str]:
    """Check if VIX level permits a new entry.

    Returns (allowed, reason).
    """
    if vix is None:
        # Can't fetch — allow entry but log warning
        return True, "vix_unavailable(allow)"

    if vix > VIX_BLOCK_THRESHOLD:
        return False, f"vix_block({vix:.1f}>{VIX_BLOCK_THRESHOLD})"

    if vix < 12.0:
        # Low VIX — options are cheap, entering is fine but premiums thin
        return True, f"vix_low({vix:.1f})"

    return True, f"vix_ok({vix:.1f})"


# ── Option Chain ──────────────────────────────────────────────────────────────

def get_option_chain_context(
    symbol: str,
    spot: float,
    expiry_date: str,
    strike_range: int = 10,
    timeout: int = 8,
) -> OptionChainContext | None:
    """Fetch NSE option chain and compute Max Pain, CE/PE walls, PCR.

    Args:
        symbol: "NIFTY" or "BANKNIFTY"
        spot: current spot price (used to filter relevant strikes)
        expiry_date: "DD-MMM-YYYY" format (e.g. "24-Jun-2025")
        strike_range: number of strikes each side of ATM to consider
        timeout: HTTP timeout in seconds

    Returns:
        OptionChainContext or None on failure.
    """
    try:
        sess = _get_session()
        resp = sess.get(
            NSE_OPTION_CHAIN_URL,
            params={"symbol": symbol},
            timeout=timeout,
        )
        resp.raise_for_status()
        raw = resp.json()

        records = raw.get("records", {})
        data    = records.get("data", [])

        if not data:
            log.warning("Empty option chain for %s", symbol)
            return None

        atm = round(spot / 50) * 50
        strikes_data: dict[float, dict] = {}

        for entry in data:
            exp = entry.get("expiryDate", "")
            if expiry_date not in exp:
                continue
            strike = float(entry.get("strikePrice", 0))
            if abs(strike - atm) > strike_range * 50:
                continue

            ce = entry.get("CE", {})
            pe = entry.get("PE", {})
            strikes_data[strike] = {
                "ce_oi": int(ce.get("openInterest", 0)),
                "pe_oi": int(pe.get("openInterest", 0)),
                "ce_ltp": float(ce.get("lastPrice", 0)),
                "pe_ltp": float(pe.get("lastPrice", 0)),
            }

        if not strikes_data:
            log.warning("No strikes found for expiry %s", expiry_date)
            return None

        return _compute_chain_metrics(strikes_data)

    except Exception as exc:
        log.warning("get_option_chain_context failed: %s", exc)
        return None


def _compute_chain_metrics(strikes_data: dict[float, dict]) -> OptionChainContext:
    """Compute Max Pain, CE/PE walls, and PCR from strikes data."""
    total_ce_oi = sum(v["ce_oi"] for v in strikes_data.values())
    total_pe_oi = sum(v["pe_oi"] for v in strikes_data.values())

    # Max Pain: strike where total option value destroyed is maximum
    max_pain = _compute_max_pain(strikes_data)

    # CE wall: strike with highest CE OI (acts as resistance)
    ce_wall = max(strikes_data, key=lambda s: strikes_data[s]["ce_oi"])

    # PE wall: strike with highest PE OI (acts as support)
    pe_wall = max(strikes_data, key=lambda s: strikes_data[s]["pe_oi"])

    # PCR: put-call ratio by OI
    pcr = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 1.0

    # Bias from PCR
    if pcr > 1.2:
        bias = "bullish"
    elif pcr < 0.8:
        bias = "bearish"
    else:
        bias = "neutral"

    return OptionChainContext(
        max_pain=max_pain,
        ce_wall=ce_wall,
        pe_wall=pe_wall,
        pcr=round(pcr, 3),
        bias=bias,
        total_ce_oi=total_ce_oi,
        total_pe_oi=total_pe_oi,
    )


def _compute_max_pain(strikes_data: dict[float, dict]) -> float:
    """Compute the max pain strike.

    For each possible expiry price (each strike), compute the total payout
    to option holders. The strike with the minimum total payout = max pain
    (where option writers lose the least = options expire most worthless).
    """
    strikes = sorted(strikes_data.keys())
    min_pain = float("inf")
    max_pain_strike = strikes[len(strikes) // 2]  # default to ATM

    for expiry_price in strikes:
        total_pain = 0.0
        for strike, data in strikes_data.items():
            # CE holders profit if expiry > strike
            ce_pain = max(0, expiry_price - strike) * data["ce_oi"]
            # PE holders profit if expiry < strike
            pe_pain = max(0, strike - expiry_price) * data["pe_oi"]
            total_pain += ce_pain + pe_pain

        if total_pain < min_pain:
            min_pain = total_pain
            max_pain_strike = expiry_price

    return max_pain_strike


def max_pain_bias(spot: float, max_pain: float, atm_step: int = 50) -> str:
    """Determine directional bias relative to max pain.

    If price is below max pain → expect pull toward max pain → bullish.
    If price is above max pain → expect pull toward max pain → bearish.
    If price is at max pain (within 1 strike) → neutral.
    """
    diff = max_pain - spot
    if abs(diff) <= atm_step:
        return "neutral"
    return "bullish" if diff > 0 else "bearish"
