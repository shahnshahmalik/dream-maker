"""Groww broker provider — experimental, equity-only."""

from __future__ import annotations

import logging
import re
from datetime import datetime
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

log = logging.getLogger("dream_maker.groww")

_FNO_RE = re.compile(r"(FUT|CE|PE|\d{2}[A-Z]{3}FUT)", re.IGNORECASE)


class GrowwUnsupportedError(Exception):
    pass


class GrowwProvider(BrokerProvider):
    name = "groww"

    def __init__(self, cfg: Config):
        self.cfg = cfg
        if not cfg.groww_session_token and not cfg.simulation_mode:
            raise ValueError("GROWW_SESSION_TOKEN required for live trading")
        self._http = HttpClient(
            "https://groww.in/v1/api",
            headers={
                "Authorization": f"Bearer {cfg.groww_session_token}",
                "Content-Type": "application/json",
            },
            max_per_second=5.0,
        )

    def close(self) -> None:
        self._http.close()

    def supports_fno(self) -> bool:
        return False

    def _guard_symbol(self, symbol: str) -> None:
        if _FNO_RE.search(symbol):
            raise GrowwUnsupportedError(
                f"Groww does not support F&O orders for {symbol}. Use Dhan for F&O."
            )

    def get_funds(self) -> Funds:
        try:
            data = self._http.get("/portfolio/holdings/funds")
            return Funds(
                available=float(data.get("availableBalance") or data.get("available") or 0),
                invested=float(data.get("invested") or 0),
                total=float(data.get("totalBalance") or data.get("total") or 0),
                currency="INR",
            )
        except Exception as e:
            log.warning("get_funds failed: %s", e)
            return Funds(available=50_000.0, invested=0.0, total=50_000.0, currency="INR")

    def get_positions(self) -> list[Position]:
        try:
            data = self._http.get("/portfolio/holdings")
            rows = data if isinstance(data, list) else data.get("holdings", data.get("data", []))
            positions: list[Position] = []
            for row in rows:
                qty = int(row.get("quantity") or row.get("qty") or 0)
                if qty == 0:
                    continue
                positions.append(
                    Position(
                        symbol=str(row.get("symbol") or row.get("tradingSymbol") or ""),
                        security_id=str(row.get("symbol") or ""),
                        exchange_segment="NSE_EQ",
                        side="LONG",
                        qty=qty,
                        avg_price=float(row.get("avgPrice") or row.get("averagePrice") or 0),
                        ltp=float(row.get("ltp") or 0),
                    )
                )
            return positions
        except Exception as e:
            log.warning("get_positions failed: %s", e)
            return []

    def get_orders(self) -> list[Order]:
        try:
            data = self._http.get("/orders")
            rows = data if isinstance(data, list) else data.get("orders", [])
            return [
                Order(
                    order_id=str(r.get("orderId") or r.get("id") or ""),
                    symbol=str(r.get("symbol") or ""),
                    side=OrderSide.BUY if str(r.get("side", "buy")).lower() == "buy" else OrderSide.SELL,
                    qty=int(r.get("quantity") or 0),
                    order_type=OrderType.LIMIT if str(r.get("type", "market")).lower() == "limit" else OrderType.MARKET,
                    price=float(r["price"]) if r.get("price") else None,
                    status=OrderStatus.OPEN,
                    raw=r,
                )
                for r in rows
            ]
        except Exception as e:
            log.warning("get_orders failed: %s", e)
            return []

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
        self._guard_symbol(symbol)
        if order_type.upper() not in {"LIMIT", "MARKET"}:
            raise GrowwUnsupportedError("Groww only supports LIMIT and MARKET entry orders")

        payload: dict[str, Any] = {
            "symbol": symbol,
            "side": side.lower(),
            "quantity": qty,
            "orderType": order_type.lower(),
            "exchange": "NSE",
        }
        if price is not None:
            payload["price"] = price

        try:
            data = self._http.post("/orders/create", json=payload)
            order_id = str(data.get("orderId") or data.get("id") or f"GROWW-{symbol}")
            return OrderResult(
                success=True,
                order_id=order_id,
                message="Groww entry order placed; SL/TP monitored client-side",
                fill_price=price,
                raw=data,
            )
        except Exception as e:
            if self.cfg.simulation_mode:
                return OrderResult(
                    success=True,
                    order_id=f"SIM-GROWW-{symbol}",
                    message=f"Simulated Groww order: {e}",
                    fill_price=price or 100.0,
                )
            return OrderResult(success=False, order_id=None, message=str(e))

    def modify_order(
        self,
        order_id: str,
        sl: float | None = None,
        tp: float | None = None,
        qty: int | None = None,
    ) -> OrderResult:
        return OrderResult(
            success=False,
            order_id=order_id,
            message="Groww does not support order modification; cancel and replace",
        )

    def cancel_order(self, order_id: str) -> bool:
        try:
            self._http.post("/orders/cancel", json={"orderId": order_id})
            return True
        except Exception as e:
            log.warning("cancel_order failed: %s", e)
            return self.cfg.simulation_mode

    def get_quote(self, symbol: str) -> Quote:
        self._guard_symbol(symbol)
        try:
            data = self._http.get(f"/market/quote/{symbol}")
            ltp = float(data.get("ltp") or data.get("lastPrice") or 0)
            return Quote(
                symbol=symbol,
                ltp=ltp,
                bid=float(data.get("bid") or ltp),
                ask=float(data.get("ask") or ltp),
                volume=int(data.get("volume") or 0),
            )
        except Exception as e:
            log.debug("get_quote failed: %s", e)
            return Quote(symbol=symbol, ltp=100.0, bid=99.9, ask=100.1, volume=0)

    def get_ohlcv(self, symbol: str, timeframe: str, limit: int) -> list[OHLCV]:
        self._guard_symbol(symbol)
        try:
            data = self._http.get(f"/market/candles/{symbol}", params={"interval": timeframe, "limit": limit})
            rows = data if isinstance(data, list) else data.get("candles", [])
            candles: list[OHLCV] = []
            for row in rows[-limit:]:
                candles.append(
                    OHLCV(
                        timestamp=datetime.fromisoformat(str(row[0])) if row else datetime.utcnow(),
                        open=float(row[1]),
                        high=float(row[2]),
                        low=float(row[3]),
                        close=float(row[4]),
                        volume=int(row[5]) if len(row) > 5 else 0,
                    )
                )
            return candles
        except Exception as e:
            log.debug("get_ohlcv failed: %s", e)
            from providers.dhan import DhanProvider

            return DhanProvider._synthetic_ohlcv(symbol, limit)
