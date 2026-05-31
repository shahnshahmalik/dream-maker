"""AI review request/response models."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from models.orders import OHLCV


class ReviewDecision(str, Enum):
    HOLD = "HOLD"
    TIGHTEN_SL = "TIGHTEN_SL"
    PARTIAL_EXIT = "PARTIAL_EXIT"
    CLOSE = "CLOSE"


@dataclass
class AIReviewRequest:
    symbol: str
    direction: str
    original_rationale: str
    entry_price: float
    current_price: float
    stop_loss: float
    macro_env: str
    candles_15m: list[OHLCV]
    candles_1h: list[OHLCV]
    divergence_reasons: list[str] = field(default_factory=list)


@dataclass
class AIReviewResponse:
    decision: ReviewDecision
    reason: str
    new_stop_loss: float | None = None
    partial_exit_pct: float | None = None
    raw: dict[str, Any] = field(default_factory=dict)
