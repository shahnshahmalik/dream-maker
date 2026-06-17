"""Wait for price markers before entry — no immediate market orders."""

from __future__ import annotations

import logging

from analysis.technical import momentum_confirmation
from config import Config
from models.trade_plan import PlanStatus, TradeDirection, TradePlan
from providers.base import BrokerProvider

log = logging.getLogger("dream_maker.entry_watcher")

# Symbols where get_quote returns synthetic data — use OHLCV fallback
_SYNTHETIC_QUOTE_SYMBOLS = {"DIXON", "KFINTECH", "JUBLFOOD", "HPCL",
                             "INDUSTOWER", "EXIDEIND", "ITC", "TATASTEEL"}


def _is_stock_option(symbol: str) -> bool:
    """Check if symbol is a stock option (vs index option)."""
    import re
    base = re.sub(r"\d{2}[A-Z]{3}.*", "", symbol.upper())
    base = re.sub(r"\d+(CE|PE).*$", "", base)
    return base in _SYNTHETIC_QUOTE_SYMBOLS


class EntryWatcher:
    def __init__(self, broker: BrokerProvider, cfg: Config):
        self.broker = broker
        self.cfg = cfg
        self._prev_ltp: dict[str, float] = {}  # plan_id → previous LTP for crossed-through detection

    def _get_ltp(self, plan: TradePlan) -> float:
        """Get live price, falling back to OHLCV when quote is synthetic."""
        quote = self.broker.get_quote(plan.symbol)
        if _is_stock_option(plan.symbol) and quote.ltp < 500:
            # Dhan returns synthetic ~100 for stock options — use candle close
            try:
                candles = self.broker.get_ohlcv(plan.symbol, "5m", 1)
                if candles:
                    return candles[0].close
            except Exception:
                pass
        return quote.ltp

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
        return plan.meta.get("setup_type") == "stacked_sweep"

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
        ltp = self._get_ltp(plan)
        lo = min(plan.entry_price_low, plan.entry_price_high)
        hi = max(plan.entry_price_low, plan.entry_price_high)
        in_zone = lo <= ltp <= hi

        # Crossed-through detection: if LTP moved THROUGH the zone between
        # polls, treat it as a valid entry trigger. Without this, a price
        # that crosses the zone in the ~60s between poll ticks is missed.
        crossed = False
        if not in_zone and plan.plan_id in self._prev_ltp:
            prev = self._prev_ltp[plan.plan_id]
            was_below = prev < lo
            was_above = prev > hi
            is_below = ltp < lo
            is_above = ltp > hi
            crossed = (was_below and is_above) or (was_above and is_below)

        self._prev_ltp[plan.plan_id] = ltp

        # Discount entry: for LONG, LTP below zone = cheaper = better R:R.
        # For SHORT, LTP above zone = more premium collected = better R:R.
        discount = False
        if not in_zone and not crossed:
            if plan.direction == TradeDirection.LONG and ltp < lo:
                discount = True
            elif plan.direction == TradeDirection.SHORT and ltp > hi:
                discount = True

        if not in_zone and not crossed and not discount:
            return False
        if in_zone:
            pass  # normal zone entry
        elif crossed:
            log.info(
                "Crossed-through entry for %s — LTP %.2f went through zone [%.2f, %.2f]",
                plan.symbol, ltp, lo, hi,
            )
        elif discount:
            log.info(
                "Discount entry for %s — LTP %.2f below zone [%.2f, %.2f] (premium cheaper than entry zone)",
                plan.symbol, ltp, lo, hi,
            )
            # Recalculate SL/TP for actual entry price (zone-based levels are wrong at discount)
            zone_mid = (lo + hi) / 2
            if zone_mid > 0:
                sl_pct = abs(zone_mid - plan.stop_loss) / zone_mid
                tp1_pct = abs(plan.take_profit_1 - zone_mid) / zone_mid
                tp2_pct = abs(plan.take_profit_2 - zone_mid) / zone_mid if plan.take_profit_2 else 0
                if plan.direction == TradeDirection.LONG:
                    plan.stop_loss = round(ltp * (1 - sl_pct), 2)
                    plan.take_profit_1 = round(ltp * (1 + tp1_pct), 2)
                    if tp2_pct:
                        plan.take_profit_2 = round(ltp * (1 + tp2_pct), 2)
                else:
                    plan.stop_loss = round(ltp * (1 + sl_pct), 2)
                    plan.take_profit_1 = round(ltp * (1 - tp1_pct), 2)
                    if tp2_pct:
                        plan.take_profit_2 = round(ltp * (1 - tp2_pct), 2)
                # Update entry price and zone for P&L tracking + bracket validation
                plan.entry_price = ltp
                plan.entry_zone = str(ltp)
                plan.entry_price_low = ltp * 0.98
                plan.entry_price_high = ltp * 1.02
                log.info(
                    "Discount SL/TP adjusted: SL=%.2f TP1=%.2f TP2=%.2f (from LTP %.2f)",
                    plan.stop_loss, plan.take_profit_1, plan.take_profit_2, ltp,
                )
        if plan.direction == TradeDirection.LONG and ltp <= plan.stop_loss:
            return False
        if plan.direction == TradeDirection.SHORT and ltp >= plan.stop_loss:
            return False
        if self._is_scalp(plan) and not self._live_momentum_confirmed(plan):
            return False
        return True
