"""Dhan broker provider — primary integration."""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from config import Config
from models.orders import (
    Funds,
    OHLCV,
    Order,
    OrderResult,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    Quote,
)
from providers.base import BrokerProvider
from providers.dhan_instruments import DhanInstrument, resolve_market_data_instrument
from providers.yahoo_finance import fetch_ohlcv as yahoo_fetch_ohlcv
from utils.http import HttpClient

from utils.symbols import is_fno_eligible, normalize_symbol, underlying_base

log = logging.getLogger("dream_maker.dhan")

IST = ZoneInfo("Asia/Kolkata")

# Dhan intraday intervals: 1, 5, 15, 25, 60 (minutes)
TIMEFRAME_MAP = {
    "1m": "1",
    "5m": "5",
    "15m": "15",
    "25m": "25",
    "1h": "60",
    "4h": "60",
    "1d": "D",
    "1w": "D",
}


class DhanProvider(BrokerProvider):
    name = "dhan"

    def __init__(self, cfg: Config):
        self.cfg = cfg
        if not cfg.dhan_access_token and not cfg.simulation_mode:
            raise ValueError("DHAN_ACCESS_TOKEN required for live trading")
        headers: dict[str, str] = {
            "access-token": cfg.dhan_access_token,
            "Content-Type": "application/json",
        }
        if cfg.dhan_client_id:
            headers["client-id"] = cfg.dhan_client_id
        elif not cfg.simulation_mode:
            log.warning(
                "DHAN_CLIENT_ID not set — market LTP/quotes may return 401; "
                "find it in Dhan web/API dashboard"
            )
        self._http = HttpClient(
            "https://api.dhan.co",
            headers=headers,
            max_per_second=25.0,
        )
        self._instrument_cache: dict[str, dict[str, Any]] = {}
        self._synthetic_warned: set[str] = set()
        self._auth_failure_count: int = 0
        self._max_auth_failures: int = 3
        self._auth_blocked: bool = False

    def _is_auth_error(self, error: Exception | str) -> bool:
        """Check if an error is an authentication failure (expired token, etc.)."""
        msg = str(error)
        return any(tag in msg for tag in (
            "DH-901", "Invalid_Authentication", "token is invalid",
            "token has expired", "access token is invalid",
        ))

    def _record_auth_failure(self, context: str) -> None:
        self._auth_failure_count += 1
        if self._auth_failure_count >= self._max_auth_failures and not self._auth_blocked:
            self._auth_blocked = True
            log.critical(
                "AUTH CIRCUIT BREAKER: %s — %d consecutive auth failures. "
                "All broker calls blocked until restart. Refresh Dhan token in .env.",
                context, self._auth_failure_count,
            )

    def close(self) -> None:
        self._http.close()

    def get_index_spot(self, index_name: str) -> float:
        """Get the spot LTP for an index underlying (bypasses _allowed_symbol).

        Used by SymbolPicker to compute ATM option strikes before the
        trading_symbol is finalized.
        """
        from providers.dhan_instruments import INDEX_INSTRUMENTS, _index_key

        key = _index_key(index_name)
        if key is None:
            raise ValueError(f"Not a known index: {index_name}")
        meta = INDEX_INSTRUMENTS[key]
        security_id = str(meta["securityId"])
        segment = str(meta["exchangeSegment"])

        try:
            data = self._http.post(
                "/v2/marketfeed/ltp",
                json={segment: [int(security_id)]},
            )
            ltp = self._extract_ltp(data, segment, security_id)
            if ltp <= 0:
                raise ValueError("empty quote")
            return ltp
        except Exception as e:
            raise ValueError(f"Failed to fetch spot for {index_name}: {e}")

    def _allowed_symbol(self, symbol: str) -> bool:
        sym = normalize_symbol(symbol)
        configured = normalize_symbol(self.cfg.trading_symbol)
        configured_base = underlying_base(configured)
        base = underlying_base(sym)
        # Allow the configured symbol AND any F&O-eligible symbol
        # (needed for symbol picker to query spot prices of candidate underlyings)
        return (
            sym == configured
            or base == configured_base
            or base == configured.replace("50", "")
            or is_fno_eligible(sym)
        )

    def _market_instrument(self, symbol: str) -> DhanInstrument | None:
        sym = normalize_symbol(symbol)
        if not self._allowed_symbol(sym):
            raise ValueError(
                f"Symbol {symbol} is not allowed for TRADING_SYMBOL={self.cfg.trading_symbol}"
            )
        if not is_fno_eligible(sym):
            raise ValueError(f"Symbol {symbol} is not F&O eligible")
        return resolve_market_data_instrument(sym)

    def _resolve_instrument(self, symbol: str) -> dict[str, Any]:
        sym = normalize_symbol(symbol)
        if sym in self._instrument_cache:
            return self._instrument_cache[sym]

        market = self._market_instrument(sym)

        # ── Option contracts: scrip_master always wins ──────────────
        is_option = bool(re.search(r"(CE|PE)$", sym, re.IGNORECASE)) or \
                     (market is not None and market.instrument in ("OPTIDX", "OPTSTK"))
        if is_option:
            scrip_sid = self._lookup_option_security_id(sym)
            if scrip_sid and isinstance(scrip_sid, int) and scrip_sid > 0:
                lot_from_db = self._lookup_option_lot_size(sym)
                lot = lot_from_db if lot_from_db else (market.lot_size if market else 65)
                info = {
                    "symbol": sym,
                    "exchangeSegment": "NSE_FNO",
                    "securityId": str(scrip_sid),
                    "instrument": market.instrument if market and market.instrument in ("OPTIDX", "OPTSTK") else "OPTIDX",
                    "lotSize": lot,
                }
                self._instrument_cache[sym] = info
                log.info("Resolved option via scrip master: %s → sid=%d lot=%d",
                        sym, scrip_sid, lot)
                return info
            # Scrip master failed — fall through to market info below
            log.warning("Option %s not found in scrip master — using market fallback", sym)

        if market and market.security_id:
            info = {
                "symbol": sym,
                "exchangeSegment": market.exchange_segment,
                "securityId": market.security_id,
                "instrument": market.instrument,
                "lotSize": market.lot_size,
            }
            self._instrument_cache[sym] = info
            return info

        # ── Last-resort fallback ───────────────────────────────────
        lot_size: int = 1
        instrument: str = "FUTSTK"
        if market:
            lot_size = market.lot_size
            instrument = market.instrument
        elif "NIFTY" in sym:
            lot_size = 30 if "BANKNIFTY" in sym else 65
            instrument = "FUTIDX"

        info = {
            "symbol": sym,
            "exchangeSegment": "NSE_FNO",
            "securityId": sym,
            "instrument": instrument,
            "lotSize": lot_size,
        }
        self._instrument_cache[sym] = info
        return info

    # ------------------------------------------------------------------ #
    # Scrip master lookup (option security IDs)
    # ------------------------------------------------------------------ #
    _OPTION_SYMBOL_RE = re.compile(
        r"^(?P<underlying>[A-Z]+)(?P<yy>\d{2})(?P<month>[A-Z]{3})"
        r"(?P<strike>\d+)(?P<type>CE|PE)$",
        re.IGNORECASE,
    )

    @staticmethod
    def _parse_option_symbol(sym: str) -> dict | None:
        """Parse a symbol like 'BANKNIFTY26JUN54400CE' into components."""
        match = DhanProvider._OPTION_SYMBOL_RE.match(sym.strip().upper())
        if not match:
            return None
        return {
            "underlying": match.group("underlying"),
            "yy": match.group("yy"),
            "month": match.group("month").upper(),
            "strike": int(match.group("strike")),
            "option_type": match.group("type").upper(),
        }

    @staticmethod
    def _month_to_number(mmm: str) -> int:
        """Convert month abbreviation to number. JAN→1, FEB→2, ..."""
        months = {v: k for k, v in enumerate(
            ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
             "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], start=1)}
        return months.get(mmm.upper(), 0)

    def _lookup_option_security_id(self, symbol: str) -> int | None:
        """Look up the numeric Dhan security ID for an option contract.

        Uses the scrip master SQLite DB if available. Tries exact
        trading_symbol match first (converting between formats).
        """
        try:
            from scripts.scrip_master import ScripMaster
            sm = ScripMaster()
            # Try direct trading_symbol lookup (handles format conversion)
            sid = sm.get_by_trading_symbol(symbol)
            sm.close()
            return sid
        except Exception as e:
            log.debug("Scrip master lookup failed for %s: %s", symbol, e)
            return None

    def _lookup_option_lot_size(self, symbol: str) -> int | None:
        """Look up lot size from scrip master DB."""
        try:
            from scripts.scrip_master import ScripMaster
            sm = ScripMaster()
            lot = sm.get_lot_size(symbol)
            sm.close()
            return lot
        except Exception:
            return None

    @staticmethod
    def _parse_candles(data: dict[str, Any], limit: int) -> list[OHLCV]:
        closes = data.get("close") or []
        if not closes:
            return []
        opens = data.get("open") or []
        highs = data.get("high") or []
        lows = data.get("low") or []
        volumes = data.get("volume") or []
        times = data.get("timestamp") or []
        candles: list[OHLCV] = []
        n = min(len(closes), limit)
        for i in range(-n, 0):
            idx = i if i < 0 else i
            ts_raw = times[i] if times else None
            if ts_raw:
                ts = datetime.fromtimestamp(int(ts_raw), tz=IST)
            else:
                ts = datetime.now(IST)
            candles.append(
                OHLCV(
                    timestamp=ts,
                    open=float(opens[i]),
                    high=float(highs[i]),
                    low=float(lows[i]),
                    close=float(closes[i]),
                    volume=int(volumes[i]) if volumes and i < len(volumes) else 0,
                )
            )
        return candles

    @staticmethod
    def _intraday_dates(limit: int, interval_minutes: int) -> tuple[str, str]:
        now = datetime.now(IST)
        to_date = now.strftime("%Y-%m-%d %H:%M:%S")
        calendar_days = max(5, (limit * interval_minutes) // (5 * 60) + 3)
        from_dt = now - timedelta(days=calendar_days)
        from_date = from_dt.strftime("%Y-%m-%d %H:%M:%S")
        return from_date, to_date

    @staticmethod
    def _daily_dates(limit: int) -> tuple[str, str]:
        now = datetime.now(IST)
        to_date = now.strftime("%Y-%m-%d")
        from_dt = now - timedelta(days=max(limit + 30, 400))
        from_date = from_dt.strftime("%Y-%m-%d")
        return from_date, to_date

    def get_funds(self) -> Funds:
        if self._auth_blocked:
            return Funds(available=0.0, invested=0.0, total=0.0, currency="INR")
        try:
            data = self._http.get("/v2/fundlimit")
            self._auth_failure_count = 0  # reset on success
            return Funds(
                available=float(data.get("availabelBalance") or data.get("availableBalance") or 0),
                invested=float(data.get("utilizedAmount") or 0),
                total=float(data.get("sodLimit") or data.get("totalBalance") or 0),
                currency="INR",
            )
        except Exception as e:
            log.warning("get_funds failed (%s); returning simulation defaults", e)
            if self._is_auth_error(e):
                self._record_auth_failure("get_funds")
            return Funds(available=100_000.0, invested=0.0, total=100_000.0, currency="INR")

    def get_positions(self) -> list[Position]:
        try:
            data = self._http.get("/v2/positions")
            rows = data if isinstance(data, list) else data.get("data", [])
            positions: list[Position] = []
            for row in rows:
                qty = int(row.get("netQty") or row.get("quantity") or 0)
                if qty == 0:
                    continue
                side = "LONG" if qty > 0 else "SHORT"
                positions.append(
                    Position(
                        symbol=str(row.get("tradingSymbol") or row.get("symbol") or ""),
                        security_id=str(row.get("securityId") or ""),
                        exchange_segment=str(row.get("exchangeSegment") or "NSE_EQ"),
                        side=side,
                        qty=abs(qty),
                        avg_price=float(row.get("avgPrice") or row.get("costPrice") or 0),
                        ltp=float(row.get("ltp") or 0),
                        unrealized_pnl=float(row.get("unrealizedProfit") or 0),
                    )
                )
            return positions
        except Exception as e:
            log.warning("get_positions failed: %s", e)
            return []

    def get_orders(self) -> list[Order]:
        try:
            data = self._http.get("/v2/orders")
            rows = data if isinstance(data, list) else data.get("data", [])
            orders: list[Order] = []
            for row in rows:
                orders.append(self._parse_order(row))
            return orders
        except Exception as e:
            log.warning("get_orders failed: %s", e)
            return []

    def _parse_order(self, row: dict[str, Any]) -> Order:
        side_raw = str(row.get("transactionType") or row.get("side") or "BUY").upper()
        type_raw = str(row.get("orderType") or "MARKET").upper()
        status_raw = str(row.get("orderStatus") or row.get("status") or "OPEN").upper()
        return Order(
            order_id=str(row.get("orderId") or row.get("order_id") or ""),
            symbol=str(row.get("tradingSymbol") or ""),
            side=OrderSide.BUY if "BUY" in side_raw else OrderSide.SELL,
            qty=int(row.get("quantity") or row.get("qty") or 0),
            order_type=OrderType(type_raw.replace(" ", "_") if type_raw in OrderType.__members__ else "MARKET"),
            price=float(row["price"]) if row.get("price") else None,
            status=OrderStatus.OPEN if "OPEN" in status_raw else OrderStatus.FILLED if "FILLED" in status_raw else OrderStatus.PENDING,
            trigger_price=float(row["triggerPrice"]) if row.get("triggerPrice") else None,
            raw=row,
        )

    def _place_leg(self, payload: dict[str, Any], label: str, sim_id: str) -> str | None:
        """Place a single bracket leg with one retry.

        Returns the broker order id, a deterministic SIM id when the API
        accepted a paper order without returning an id, or None on failure.
        An API error is never converted into a fake success — even in
        simulation mode — so leg failures surface honestly.
        """
        last_error: str = ""
        for attempt in (1, 2):
            try:
                data = self._http.post("/v2/orders", json=payload)
                oid = str(data.get("orderId") or data.get("order_id") or "")
                if oid:
                    return oid
                if self.cfg.simulation_mode:
                    # Paper order accepted but no id returned — fabricate a
                    # stable id so the bracket can be tracked in simulation.
                    return sim_id
                log.error("%s leg accepted but returned no orderId (attempt %d): %s",
                          label, attempt, data)
                last_error = "no orderId in response"
            except Exception as e:
                last_error = str(e)
                log.error("%s leg failed (attempt %d/2): %s", label, attempt, last_error[:200])
        log.error("%s leg could not be placed after retry: %s", label, last_error[:200])
        return None

    def place_order(
        self,
        symbol: str,
        side: str,
        qty: int,
        order_type: str,
        price: float | None = None,
        sl: float | None = None,
        tp: float | None = None,
    ) -> OrderResult:
        if self._auth_blocked:
            return OrderResult(
                success=False, order_id=None,
                message="Auth blocked — Dhan token expired. Restart after refreshing .env.",
            )
        inst = self._resolve_instrument(symbol)
        payload: dict[str, Any] = {
            "dhanClientId": self.cfg.dhan_client_id or "",
            "transactionType": side.upper(),
            "exchangeSegment": inst["exchangeSegment"],
            "productType": "INTRADAY",
            "orderType": order_type.upper(),
            "validity": "DAY",
            "tradingSymbol": symbol,
            "securityId": inst["securityId"],
            "quantity": qty,
        }
        if price is not None:
            payload["price"] = price
        if sl is not None and order_type.upper() in {"STOP_LOSS", "STOP_LOSS_MARKET"}:
            payload["triggerPrice"] = sl
        if self.cfg.simulation_mode:
            payload["orderFlag"] = "PAPER"

        # ── Entry leg ─────────────────────────────────────────────────
        try:
            log.info("place_order payload: %s", {k: v for k, v in payload.items() if k != "dhanClientId"})
            data = self._http.post("/v2/orders", json=payload)
        except Exception as e:
            detail = str(e)
            if "Invalid IP" in detail or "401" in detail:
                log.error("place_order DENIED: IP not whitelisted or auth invalid — %s", detail[:200])
                self._record_auth_failure(f"place_order {symbol}")
            else:
                log.error("place_order failed: %s", detail[:200])
            return OrderResult(success=False, order_id=None, message=detail)

        order_id = str(data.get("orderId") or data.get("order_id") or "")
        if not order_id:
            if self.cfg.simulation_mode:
                order_id = f"PAPER-{symbol}-{qty}"
            else:
                # Accepted-but-unidentified orders cannot be tracked or
                # cancelled — treat as failure and alert loudly.
                log.critical(
                    "Entry order for %s accepted but Dhan returned no orderId — "
                    "manual reconciliation may be required: %s", symbol, data,
                )
                return OrderResult(success=False, order_id=None, message="No orderId in entry response")

        self._auth_failure_count = 0

        # ── Bracket legs: each isolated, one retry, no fake success ──
        sl_id: str | None = None
        tp_id: str | None = None

        if sl is not None and order_type.upper() not in {"STOP_LOSS", "STOP_LOSS_MARKET"}:
            sl_side = "SELL" if side.upper() == "BUY" else "BUY"
            sl_payload = {
                "dhanClientId": payload["dhanClientId"],
                "transactionType": sl_side,
                "exchangeSegment": payload["exchangeSegment"],
                "productType": payload["productType"],
                "orderType": "STOP_LOSS_MARKET",
                "validity": payload["validity"],
                "tradingSymbol": payload["tradingSymbol"],
                "securityId": payload["securityId"],
                "quantity": qty,
                "triggerPrice": sl,
            }
            if self.cfg.simulation_mode:
                sl_payload["orderFlag"] = "PAPER"
            log.info("SL payload: %s", {k: v for k, v in sl_payload.items() if k != "dhanClientId"})
            sl_id = self._place_leg(sl_payload, "SL", f"SIM-SL-{symbol}-{qty}")

        if tp is not None:
            tp_side = "SELL" if side.upper() == "BUY" else "BUY"
            tp_payload = {
                "dhanClientId": payload["dhanClientId"],
                "transactionType": tp_side,
                "exchangeSegment": payload["exchangeSegment"],
                "productType": payload["productType"],
                "orderType": "LIMIT",
                "validity": payload["validity"],
                "tradingSymbol": payload["tradingSymbol"],
                "securityId": payload["securityId"],
                "quantity": qty,
                "price": tp,
            }
            if self.cfg.simulation_mode:
                tp_payload["orderFlag"] = "PAPER"
            tp_id = self._place_leg(tp_payload, "TP", f"SIM-TP-{symbol}-{qty}")

        # Extract actual fill price from Dhan response for MARKET orders
        fill_price = price  # limit orders use the limit price
        if price is None:   # MARKET orders — extract from broker response
            fill_price = float(
                data.get("averageTradedPrice")
                or data.get("filledPrice")
                or data.get("average_price")
                or data.get("tradedPrice")
                or 0
            ) or None

        # The entry is live, so report success with whatever leg ids exist.
        # Missing SL/TP ids are handled by the executor's naked-entry guard
        # (cancel + flatten) — returning success=False here would leave the
        # entry orphaned and trigger a duplicate placement next cycle.
        legs_ok = (sl is None or sl_id) and (tp is None or tp_id)
        return OrderResult(
            success=True,
            order_id=order_id,
            message="Order placed" if legs_ok else "Entry placed but bracket leg(s) failed",
            fill_price=fill_price,
            sl_order_id=sl_id,
            tp_order_id=tp_id,
            raw=data,
        )

    @staticmethod
    def _is_simulated_order_id(order_id: str) -> bool:
        return order_id.startswith(("SIM-", "PAPER-"))

    def modify_order(
        self,
        order_id: str,
        sl: float | None = None,
        tp: float | None = None,
        qty: int | None = None,
    ) -> OrderResult:
        # Simulated orders don't exist broker-side — succeed locally without
        # an API call instead of masking a guaranteed API error as success.
        if self._is_simulated_order_id(order_id):
            return OrderResult(success=True, order_id=order_id, message="Simulated modify")
        payload: dict[str, Any] = {"orderId": order_id}
        if sl is not None:
            payload["triggerPrice"] = sl
        if tp is not None:
            payload["price"] = tp
        if qty is not None:
            payload["quantity"] = qty
        try:
            data = self._http.put("/v2/orders", json=payload)
            return OrderResult(success=True, order_id=order_id, message="Order modified", raw=data)
        except Exception as e:
            return OrderResult(success=False, order_id=order_id, message=str(e))

    def cancel_order(self, order_id: str) -> bool:
        if self._is_simulated_order_id(order_id):
            return True
        try:
            self._http.delete(f"/v2/orders/{order_id}")
            return True
        except Exception as e:
            log.warning("cancel_order failed: %s", e)
            return False

    def _extract_ltp(self, data: dict[str, Any], segment: str, security_id: str) -> float:
        seg_data = (data.get("data") or {}).get(segment) or {}
        row = seg_data.get(security_id) or seg_data.get(str(security_id)) or {}
        return float(row.get("last_price") or row.get("lastPrice") or 0)

    def get_quote(self, symbol: str) -> Quote:
        inst = self._resolve_instrument(symbol)
        segment = inst["exchangeSegment"]
        security_id = str(inst["securityId"])
        if not security_id.isdigit():
            log.warning("No numeric securityId for %s — cannot fetch LTP", symbol)
            return self._fallback_quote(symbol)

        try:
            sec_int = int(security_id)
            data = self._http.post(
                "/v2/marketfeed/ltp",
                json={segment: [sec_int]},
            )
            log.debug("Raw LTP response for %s (sid=%s): %s", symbol, security_id, data)
            ltp = self._extract_ltp(data, segment, security_id)
            if ltp <= 0:
                log.warning("Empty LTP for %s — raw response: %s", symbol, data)
                raise ValueError("empty quote")
            return Quote(symbol=symbol, ltp=ltp, bid=ltp, ask=ltp, volume=0)
        except Exception as e:
            log.warning("get_quote failed for %s: %s", symbol, e)
            return self._fallback_quote(symbol)

    def _fallback_quote(self, symbol: str) -> Quote:
        base = 24500.0 if "NIFTY" in symbol.upper() else 100.0
        return Quote(symbol=symbol, ltp=base, bid=base * 0.999, ask=base * 1.001, volume=0)

    def get_ohlcv(self, symbol: str, timeframe: str, limit: int) -> list[OHLCV]:
        inst = self._resolve_instrument(symbol)
        security_id = str(inst["securityId"])
        if not security_id.isdigit():
            yahoo_candles = yahoo_fetch_ohlcv(symbol, timeframe, limit)
            if yahoo_candles:
                return yahoo_candles
            return self._synthetic_ohlcv(symbol, limit, timeframe)

        interval = TIMEFRAME_MAP.get(timeframe, "15")
        try:
            if interval == "D":
                from_date, to_date = self._daily_dates(limit)
                data = self._http.post(
                    "/v2/charts/historical",
                    json={
                        "securityId": security_id,
                        "exchangeSegment": inst["exchangeSegment"],
                        "instrument": inst.get("instrument", "INDEX"),
                        "expiryCode": 0,
                        "oi": False,
                        "fromDate": from_date,
                        "toDate": to_date,
                    },
                )
            else:
                interval_minutes = int(interval)
                from_date, to_date = self._intraday_dates(limit, interval_minutes)
                data = self._http.post(
                    "/v2/charts/intraday",
                    json={
                        "securityId": security_id,
                        "exchangeSegment": inst["exchangeSegment"],
                        "instrument": inst.get("instrument", "INDEX"),
                        "interval": interval,
                        "oi": False,
                        "fromDate": from_date,
                        "toDate": to_date,
                    },
                )
            candles = self._parse_candles(data, limit)
            if candles:
                return candles
            raise ValueError("empty candle response")
        except Exception as e:
            if symbol not in self._synthetic_warned:
                log.warning(
                    "Dhan chart data unavailable for %s (%s) — trying Yahoo Finance; "
                    "check DHAN_CLIENT_ID and Data API access",
                    symbol,
                    e,
                )
                self._synthetic_warned.add(symbol)

            # Try Yahoo Finance first
            yahoo_candles = yahoo_fetch_ohlcv(symbol, timeframe, limit)
            if yahoo_candles:
                log.info("Yahoo Finance returned %d candles for %s (%s)", len(yahoo_candles), symbol, timeframe)
                return yahoo_candles

            # Last resort: synthetic candles
            log.warning("Yahoo also failed for %s — using synthetic candles", symbol)
            return self._synthetic_ohlcv(symbol, limit, timeframe)

    @staticmethod
    def _synthetic_ohlcv(symbol: str, limit: int, timeframe: str) -> list[OHLCV]:
        """Consistent synthetic series so HTF/LTF share the same price scale."""
        base = 24500.0 if "NIFTY" in symbol.upper() else 100.0 + (hash(symbol) % 500)
        step_minutes = 15 if timeframe in {"15m", "1h", "4h"} else 1440
        candles: list[OHLCV] = []
        price = base
        now = datetime.now(IST)
        for i in range(limit):
            drift = 0.0015 if i % 2 == 0 else 0.001
            o = price
            h = price * (1 + drift)
            low = price * (1 - 0.0008)
            c = price * (1 + drift * 0.5)
            ts = now - timedelta(minutes=step_minutes * (limit - i))
            candles.append(
                OHLCV(timestamp=ts, open=o, high=h, low=low, close=c, volume=1000 + i)
            )
            price = c
        return candles
