"""Wait for price markers before entry — no immediate market orders."""

from __future__ import annotations

import logging

from analysis.technical import momentum_confirmation
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
            try:
                if self._conditions_met(plan):
                    log.info(
                        "Entry markers fulfilled for %s — LTP in zone [%.2f, %.2f]",
                        plan.symbol,
                        plan.entry_price_low,
                        plan.entry_price_high,
                    )
                    plan.status = PlanStatus.PENDING
                    ready.append(plan)
            except Exception as e:
                log.warning("Entry check failed for %s: %s", plan.symbol, e)
        return ready

    def _quote_symbol(self, plan: TradePlan) -> str:
        """Entry markers are set on underlying/index price, not option premium."""
        return self.cfg.trading_symbol

    def _is_scalp(self, plan: TradePlan) -> bool:
        return plan.meta.get("setup_type") == "momentum_scalp"

    def _live_momentum_confirmed(self, plan: TradePlan) -> bool:
        candles = self.broker.get_ohlcv(self._quote_symbol(plan), "5m", 20)
        ok, reasons = momentum_confirmation(
            candles,
            plan.direction,
            min_confirmations=self.cfg.scalp_min_confirmations,
        )
        if not ok:
            log.debug(
                "Scalp entry waiting — momentum not confirmed for %s (got %s)",
                plan.symbol,
                reasons,
            )
        return ok

    def _conditions_met(self, plan: TradePlan) -> bool:
        if plan.entry_price_low is None or plan.entry_price_high is None:
            return False
        quote = self.broker.get_quote(self._quote_symbol(plan))
        ltp = quote.ltp
        lo = min(plan.entry_price_low, plan.entry_price_high)
        hi = max(plan.entry_price_low, plan.entry_price_high)
        in_zone = lo <= ltp <= hi
        if not in_zone:
            return False
        if plan.direction == TradeDirection.LONG and ltp <= plan.stop_loss:
            return False
        if plan.direction == TradeDirection.SHORT and ltp >= plan.stop_loss:
            return False
        if self._is_scalp(plan) and not self._live_momentum_confirmed(plan):
            return False
        return True
