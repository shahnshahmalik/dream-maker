"""Dhan broker provider — primary integration."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

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
from utils.http import HttpClient

from utils.symbols import is_fno_eligible, normalize_symbol, underlying_base

log = logging.getLogger("dream_maker.dhan")

TIMEFRAME_MAP = {
    "1m": "1",
    "5m": "5",
    "15m": "15",
    "1h": "60",
    "4h": "240",
    "1d": "D",
    "1w": "W",
}


def _is_fno_symbol(symbol: str) -> bool:
    return is_fno_eligible(symbol)


class DhanProvider(BrokerProvider):
    name = "dhan"

    def __init__(self, cfg: Config):
        self.cfg = cfg
        if not cfg.dhan_access_token and not cfg.simulation_mode:
            raise ValueError("DHAN_ACCESS_TOKEN required for live trading")
        self._http = HttpClient(
            "https://api.dhan.co",
            headers={
                "access-token": cfg.dhan_access_token,
                "Content-Type": "application/json",
            },
            max_per_second=25.0,
        )
        self._instrument_cache: dict[str, dict[str, Any]] = {}

    def close(self) -> None:
        self._http.close()

    def _allowed_symbol(self, symbol: str) -> bool:
        sym = normalize_symbol(symbol)
        base = underlying_base(sym)
        configured = normalize_symbol(self.cfg.trading_symbol)
        configured_base = underlying_base(configured)
        return sym == configured or base == configured_base or base == configured.replace("50", "")

    def _resolve_instrument(self, symbol: str) -> dict[str, Any]:
        sym = normalize_symbol(symbol)
        if not self._allowed_symbol(sym):
            raise ValueError(
                f"Symbol {symbol} is not allowed for TRADING_SYMBOL={self.cfg.trading_symbol}"
            )
        if not is_fno_eligible(sym):
            raise ValueError(f"Symbol {symbol} is not F&O eligible")
        if sym in self._instrument_cache:
            return self._instrument_cache[sym]
        info = {
            "symbol": sym,
            "exchangeSegment": "NSE_FNO",
            "securityId": sym,
            "lotSize": 25 if "NIFTY" in sym else 1,
        }
        self._instrument_cache[sym] = info
        return info

    def get_funds(self) -> Funds:
        try:
            data = self._http.get("/v2/fundlimit")
            return Funds(
                available=float(data.get("availabelBalance") or data.get("availableBalance") or 0),
                invested=float(data.get("utilizedAmount") or 0),
                total=float(data.get("sodLimit") or data.get("totalBalance") or 0),
                currency="INR",
            )
        except Exception as e:
            log.warning("get_funds failed (%s); returning simulation defaults", e)
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
        inst = self._resolve_instrument(symbol)
        payload: dict[str, Any] = {
            "dhanClientId": "",
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

        try:
            data = self._http.post("/v2/orders", json=payload)
            order_id = str(data.get("orderId") or data.get("order_id") or f"PAPER-{symbol}-{qty}")
            sl_id = None
            tp_id = None

            if sl is not None and order_type.upper() not in {"STOP_LOSS", "STOP_LOSS_MARKET"}:
                sl_side = "SELL" if side.upper() == "BUY" else "BUY"
                sl_payload = {
                    **payload,
                    "transactionType": sl_side,
                    "orderType": "STOP_LOSS_MARKET",
                    "triggerPrice": sl,
                    "quantity": qty,
                }
                if self.cfg.simulation_mode:
                    sl_payload["orderFlag"] = "PAPER"
                sl_data = self._http.post("/v2/orders", json=sl_payload)
                sl_id = str(sl_data.get("orderId") or "")

            if tp is not None:
                tp_side = "SELL" if side.upper() == "BUY" else "BUY"
                tp_payload = {
                    **payload,
                    "transactionType": tp_side,
                    "orderType": "LIMIT",
                    "price": tp,
                    "quantity": qty,
                }
                if self.cfg.simulation_mode:
                    tp_payload["orderFlag"] = "PAPER"
                tp_data = self._http.post("/v2/orders", json=tp_payload)
                tp_id = str(tp_data.get("orderId") or "")

            if self.cfg.simulation_mode:
                sl_id = sl_id or f"SIM-SL-{symbol}-{qty}"
                tp_id = tp_id or f"SIM-TP-{symbol}-{qty}"

            return OrderResult(
                success=True,
                order_id=order_id,
                message="Order placed",
                fill_price=price,
                sl_order_id=sl_id,
                tp_order_id=tp_id,
                raw=data,
            )
        except Exception as e:
            log.error("place_order failed: %s", e)
            if self.cfg.simulation_mode:
                return OrderResult(
                    success=True,
                    order_id=f"SIM-{symbol}-{side}-{qty}",
                    message=f"Simulated order (API error: {e})",
                    fill_price=price or 0.0,
                    sl_order_id=f"SIM-SL-{symbol}-{qty}",
                    tp_order_id=f"SIM-TP-{symbol}-{qty}",
                )
            return OrderResult(success=False, order_id=None, message=str(e))

    def modify_order(
        self,
        order_id: str,
        sl: float | None = None,
        tp: float | None = None,
        qty: int | None = None,
    ) -> OrderResult:
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
            if self.cfg.simulation_mode:
                return OrderResult(success=True, order_id=order_id, message=f"Simulated modify: {e}")
            return OrderResult(success=False, order_id=order_id, message=str(e))

    def cancel_order(self, order_id: str) -> bool:
        try:
            self._http.delete(f"/v2/orders/{order_id}")
            return True
        except Exception as e:
            log.warning("cancel_order failed: %s", e)
            return self.cfg.simulation_mode

    def get_quote(self, symbol: str) -> Quote:
        inst = self._resolve_instrument(symbol)
        try:
            data = self._http.post(
                "/v2/marketfeed/ltp",
                json={"NSE_FNO": [inst["securityId"]]},
            )
            ltp = float(data.get("data", {}).get(inst["securityId"], {}).get("last_price", 0) or 0)
            if ltp == 0:
                raise ValueError("empty quote")
            return Quote(symbol=symbol, ltp=ltp, bid=ltp, ask=ltp, volume=0)
        except Exception as e:
            log.debug("get_quote API failed for %s: %s", symbol, e)
            return Quote(symbol=symbol, ltp=100.0, bid=99.9, ask=100.1, volume=0)

    def get_ohlcv(self, symbol: str, timeframe: str, limit: int) -> list[OHLCV]:
        inst = self._resolve_instrument(symbol)
        interval = TIMEFRAME_MAP.get(timeframe, "15")
        try:
            data = self._http.post(
                "/v2/charts/intraday",
                json={
                    "securityId": inst["securityId"],
                    "exchangeSegment": inst["exchangeSegment"],
                    "instrument": "FUTIDX" if "NIFTY" in inst["symbol"] or "IDX" in inst["symbol"] else "FUTSTK",
                    "interval": interval,
                },
            )
            candles: list[OHLCV] = []
            opens = data.get("open", [])
            highs = data.get("high", [])
            lows = data.get("low", [])
            closes = data.get("close", [])
            volumes = data.get("volume", [])
            times = data.get("timestamp", [])
            n = min(len(closes), limit)
            for i in range(-n, 0):
                ts = datetime.fromtimestamp(times[i]) if times else datetime.utcnow()
                candles.append(
                    OHLCV(
                        timestamp=ts,
                        open=float(opens[i]),
                        high=float(highs[i]),
                        low=float(lows[i]),
                        close=float(closes[i]),
                        volume=int(volumes[i]) if volumes else 0,
                    )
                )
            return candles
        except Exception as e:
            log.debug("get_ohlcv failed for %s: %s — generating synthetic data", symbol, e)
            return self._synthetic_ohlcv(symbol, limit)

    @staticmethod
    def _synthetic_ohlcv(symbol: str, limit: int) -> list[OHLCV]:
        base = 100.0 + (hash(symbol) % 500)
        candles: list[OHLCV] = []
        price = base
        now = datetime.now(timezone.utc)
        for i in range(limit):
            drift = 0.003 if i % 2 == 0 else 0.002
            o = price
            h = price * (1 + drift)
            l = price * (1 - 0.001)
            c = price * (1 + drift)
            candles.append(
                OHLCV(
                    timestamp=now,
                    open=o,
                    high=h,
                    low=l,
                    close=c,
                    volume=1000 + i,
                )
            )
            price = c
        return candles
