"""Broker provider factory."""

from __future__ import annotations

from config import Config
from providers.base import BrokerProvider
from providers.dhan import DhanProvider
from providers.groww import GrowwProvider


def get_broker(cfg: Config) -> BrokerProvider:
    if cfg.active_broker == "dhan":
        return DhanProvider(cfg)
    if cfg.active_broker == "groww":
        return GrowwProvider(cfg)
    raise ValueError(f"Unknown broker: {cfg.active_broker}. Use 'dhan' or 'groww'.")
