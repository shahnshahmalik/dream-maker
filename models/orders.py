"""Shared order and market data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    LIMIT = "LIMIT"
    MARKET = "MARKET"
    STOP_LOSS = "STOP_LOSS"
    STOP_LOSS_MARKET = "STOP_LOSS_MARKET"


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    OPEN = "OPEN"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


@dataclass
class Funds:
    available: float
    invested: float
    total: float
    currency: str = "INR"


@dataclass
class OHLCV:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int


@dataclass
class Quote:
    symbol: str
    ltp: float
    bid: float
    ask: float
    volume: int


@dataclass
class Position:
    symbol: str
    security_id: str
    exchange_segment: str
    side: str  # LONG | SHORT
    qty: int
    avg_price: float
    ltp: float = 0.0
    unrealized_pnl: float = 0.0


@dataclass
class Order:
    order_id: str
    symbol: str
    side: OrderSide
    qty: int
    order_type: OrderType
    price: float | None
    status: OrderStatus
    trigger_price: float | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class OrderResult:
    success: bool
    order_id: str | None
    message: str
    fill_price: float | None = None
    sl_order_id: str | None = None
    tp_order_id: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)
