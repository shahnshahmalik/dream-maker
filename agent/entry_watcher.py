"""Wait for price markers before entry — no immediate market orders."""

from __future__ import annotations

import logging

from config import Config
from models.trade_plan import PlanStatus, TradeDirection, TradePlan
from providers.base import BrokerProvider

log = logging.getLogger("dream_maker.entry_watcher")


class EntryWatcher:
    def __init__(self, broker: BrokerProvider, cfg: Config):
        self.broker = broker
        self.cfg = cfg

    def tick(self, plans: list[TradePlan]) -> list[TradePlan]:
        ready: list[TradePlan] = []
        for plan in plans:
            if plan.status != PlanStatus.WAITING_ENTRY:
                continue
            if self._conditions_met(plan):
                log.info(
                    "Entry markers fulfilled for %s — LTP in zone [%.2f, %.2f]",
                    plan.symbol,
                    plan.entry_price_low,
                    plan.entry_price_high,
                )
                plan.status = PlanStatus.PENDING
                ready.append(plan)
        return ready

    def _conditions_met(self, plan: TradePlan) -> bool:
        if plan.entry_price_low is None or plan.entry_price_high is None:
            return False
        quote = self.broker.get_quote(plan.symbol)
        ltp = quote.ltp
        lo = min(plan.entry_price_low, plan.entry_price_high)
        hi = max(plan.entry_price_low, plan.entry_price_high)
        in_zone = lo <= ltp <= hi
        if not in_zone:
            return False
        if plan.direction == TradeDirection.LONG and ltp > plan.stop_loss:
            return True
        if plan.direction == TradeDirection.SHORT and ltp < plan.stop_loss:
            return True
        return False
