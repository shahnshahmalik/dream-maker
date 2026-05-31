"""Broker provider abstract base class."""

from __future__ import annotations

from abc import ABC, abstractmethod

from models.orders import Funds, OHLCV, Order, OrderResult, Position, Quote


class BrokerProvider(ABC):
    name: str = "base"

    @abstractmethod
    def get_funds(self) -> Funds: ...

    @abstractmethod
    def get_positions(self) -> list[Position]: ...

    @abstractmethod
    def get_orders(self) -> list[Order]: ...

    @abstractmethod
    def place_order(
        self,
        symbol: str,
        side: str,
        qty: int,
        order_type: str,
        price: float | None = None,
        sl: float | None = None,
        tp: float | None = None,
    ) -> OrderResult: ...

    @abstractmethod
    def modify_order(
        self,
        order_id: str,
        sl: float | None = None,
        tp: float | None = None,
        qty: int | None = None,
    ) -> OrderResult: ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool: ...

    @abstractmethod
    def get_quote(self, symbol: str) -> Quote: ...

    @abstractmethod
    def get_ohlcv(self, symbol: str, timeframe: str, limit: int) -> list[OHLCV]: ...

    def supports_fno(self) -> bool:
        return True

    def close(self) -> None:
        pass
