"""Strike / lot selection based on available funds."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from models.trade_plan import TradeDirection


@dataclass
class StrikeSelection:
    tradable_symbol: str
    qty: int
    strike: float | None
    margin_required: float
    affordable: bool
    reason: str


_OPTION_RE = re.compile(r"(\d+)(CE|PE)$", re.IGNORECASE)
_INDEX_STEP = {"NIFTY": 50, "BANKNIFTY": 100, "FINNIFTY": 50, "MIDCPNIFTY": 25}


def _index_step(symbol: str) -> int:
    upper = symbol.upper()
    for key, step in _INDEX_STEP.items():
        if key in upper:
            return step
    return 50


def _round_strike(price: float, step: int) -> float:
    return round(price / step) * step


def _margin_rate(symbol: str) -> float:
    upper = symbol.upper()
    if "CE" in upper or "PE" in upper:
        return 1.0
    return 0.12


def _lot_size(symbol: str) -> int:
    """Return the minimum trading lot size for a symbol.

    NSE June 2026 lot sizes (effective from May 2026 expiry onwards):
      NIFTY: 65 (was 25)
      BANKNIFTY: 30 (was 15)
      FINNIFTY: 60 (was 40)
      MIDCPNIFTY: 50 (was 75)
    """
    upper = symbol.upper()
    if "BANKNIFTY" in upper:
        return 30
    if "MIDCPNIFTY" in upper or "MIDCP" in upper:
        return 50
    if "SENSEX" in upper:
        return 10
    if "FINNIFTY" in upper:
        return 60
    if "NIFTY" in upper:
        return 65
    return 1


def _estimate_option_premium(ltp: float, strike: float, direction: TradeDirection) -> float:
    distance = abs(ltp - strike)
    # Realistic option premium: ~0.8% of spot for weekly ATM NIFTY.
    # was 0.015 (1.5%) — overestimated by ~2x, blocking trades on ₹16k balance.
    base = ltp * 0.008
    return max(base, distance * 0.15, 15.0)


def select_strike_and_size(
    trading_symbol: str,
    direction: TradeDirection,
    entry: float,
    stop_loss: float,
    *,
    available_funds: float,
    risk_pct: float,
    ltp: float | None = None,
) -> StrikeSelection:
    spot = ltp or entry
    lot = _lot_size(trading_symbol)
    risk_amount = available_funds * (risk_pct / 100.0)
    sl_dist = abs(entry - stop_loss)
    if sl_dist <= 0:
        return StrikeSelection(trading_symbol, 0, None, 0, False, "Invalid stop distance")

    upper = trading_symbol.upper()
    opt_match = _OPTION_RE.search(upper)

    if opt_match:
        strike = float(opt_match.group(1))
        opt_type = opt_match.group(2).upper()  # "CE" or "PE"

        # Indian retail brokers: you can only BUY options, not SELL/write.
        # LONG → BUY CE, SHORT → BUY PE. If direction conflicts with option
        # type (e.g. SHORT on a CE), flip to the correct instrument.
        needs_ce = direction == TradeDirection.LONG
        if needs_ce and opt_type == "PE":
            trading_symbol = trading_symbol.upper().replace("PE", "CE", 1)
            lot = _lot_size(trading_symbol)  # re-derive since underlying may differ
        elif not needs_ce and opt_type == "CE":
            trading_symbol = trading_symbol.upper().replace("CE", "PE", 1)
            lot = _lot_size(trading_symbol)

        premium = _estimate_option_premium(spot, strike, direction)
        margin_per_lot = premium * lot
        max_lots = int(available_funds // margin_per_lot) if margin_per_lot else 0
        risk_lots = int(risk_amount // margin_per_lot) if margin_per_lot else 0
        lots = min(max_lots, max(1, risk_lots))
        qty = lots * lot
        affordable = max_lots >= 1 and qty > 0
        return StrikeSelection(
            tradable_symbol=trading_symbol,
            qty=qty if affordable else 0,
            strike=strike,
            margin_required=margin_per_lot * lots if affordable else margin_per_lot,
            affordable=affordable,
            reason="ok" if affordable else f"Cannot afford 1 lot (margin ~{margin_per_lot:.0f})",
        )

    if "CE" not in upper and "PE" not in upper and "FUT" not in upper:
        step = _index_step(trading_symbol)
        atm = _round_strike(spot, step)
        if direction == TradeDirection.LONG:
            strike = atm
            suffix = "CE"
        else:
            strike = atm
            suffix = "PE"
        premium = _estimate_option_premium(spot, strike, direction)
        margin_per_lot = premium * lot
        max_lots = int(available_funds // margin_per_lot) if margin_per_lot else 0
        if max_lots >= 1:
            risk_lots = int(risk_amount // margin_per_lot) if margin_per_lot else 0
            lots = min(max_lots, max(1, risk_lots))
            qty = lots * lot
            sym = f"{trading_symbol.replace('IDX', '').replace('50', '')}{int(strike)}{suffix}"
            return StrikeSelection(
                tradable_symbol=sym,
                qty=qty,
                strike=strike,
                margin_required=margin_per_lot * lots,
                affordable=True,
                reason=f"Selected ATM {suffix} strike {strike}",
            )

    margin_rate = _margin_rate(trading_symbol)
    notional_per_lot = spot * lot
    margin_per_lot = notional_per_lot * margin_rate
    max_lots = int(available_funds // margin_per_lot) if margin_per_lot else 0
    risk_lots = int(risk_amount // (sl_dist * lot)) if sl_dist else 0
    lots = min(max_lots, max(1, risk_lots))
    qty = lots * lot
    affordable = max_lots >= 1 and qty > 0
    return StrikeSelection(
        tradable_symbol=trading_symbol,
        qty=qty if affordable else 0,
        strike=None,
        margin_required=margin_per_lot * lots if affordable else margin_per_lot,
        affordable=affordable,
        reason="ok" if affordable else f"Insufficient margin for 1 lot (~{margin_per_lot:.0f})",
    )
