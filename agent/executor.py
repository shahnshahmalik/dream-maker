"""Trade execution — order placement, brackets, closes."""

from __future__ import annotations

import logging

from config import Config
from audit.trade_logger import TradeLogger
from models.trade_plan import EntryType, PlanStatus, TradeDirection, TradePlan, validate_bracket
from providers.base import BrokerProvider
from risk.limits import LimitsGuard
from utils.market_hours import is_market_open, is_square_off_time, is_within_trade_window

log = logging.getLogger("dream_maker.executor")


class TradeExecutor:
    MAX_PLACEMENT_ATTEMPTS = 3
    PLACEMENT_RETRY_SECONDS = 120

    def __init__(
        self,
        broker: BrokerProvider,
        cfg: Config,
        trade_logger: TradeLogger,
        limits: LimitsGuard,
        balance_manager=None,  # BalanceManager — optional for backward compat
        session_tracker=None,  # SessionTracker — optional, for post-loss sizing
    ):
        self.broker = broker
        self.cfg = cfg
        self.trade_logger = trade_logger
        self.limits = limits
        self._balance = balance_manager
        self._session = session_tracker

    def execute_plan(self, plan: TradePlan, open_count: int) -> TradePlan:
        if plan.status != PlanStatus.PENDING:
            return plan

        # Retry cooldown after a failed placement — avoid hammering the
        # broker (and duplicating entries) on every engine cycle.
        import time as _time
        next_attempt = float(plan.meta.get("_next_attempt_at") or 0)
        if next_attempt and _time.time() < next_attempt:
            return plan

        log.info("EXEC: %s direction=%s entry=%.2f sl=%.2f tp=%.2f open=%d",
                 plan.symbol, plan.direction.value, plan.entry_target(),
                 plan.stop_loss, plan.take_profit_1, open_count)

        from utils.symbols import normalize_symbol, same_underlying, same_underlying_strike

        # Allow: the configured symbol itself, a same-strike CE/PE flip, or an
        # option contract the strike selector derived from the configured
        # underlying (e.g. TRADING_SYMBOL=NIFTY50IDX → NIFTY26JUN24500CE).
        if (
            normalize_symbol(plan.symbol) != self.cfg.trading_symbol
            and not same_underlying_strike(plan.symbol, self.cfg.trading_symbol)
            and not same_underlying(plan.symbol, self.cfg.trading_symbol)
        ):
            log.warning(
                "Rejecting plan for %s — only TRADING_SYMBOL=%s (or contracts on the same underlying) is allowed",
                plan.symbol, self.cfg.trading_symbol,
            )
            plan.status = PlanStatus.INVALIDATED
            return plan

        if not is_market_open(self.cfg.trading_hours_ist, holidays=self.cfg.market_holidays):
            log.info("Outside trading hours — skipping %s", plan.symbol)
            return plan

        if not is_within_trade_window(
            self.cfg.trade_window_start_ist,
            self.cfg.trade_window_end_ist,
            holidays=self.cfg.market_holidays,
        ):
            log.info(
                "Outside trade window (%s–%s IST) — skipping %s",
                self.cfg.trade_window_start_ist,
                self.cfg.trade_window_end_ist,
                plan.symbol,
            )
            return plan

        ok, reason = self.limits.can_open_trade(open_count)
        if not ok:
            log.warning("Cannot open %s: %s", plan.symbol, reason)
            plan.status = PlanStatus.INVALIDATED
            return plan

        funds = self.broker.get_funds()
        if funds.available < plan.risk_amount:
            log.warning("Insufficient funds for %s", plan.symbol)
            plan.status = PlanStatus.INVALIDATED
            return plan

        # Balance-aware side: low-balance accounts → BUY only (CE/PE flip handled upstream)
        if self._balance is not None:
            side = self._balance.get_side(plan.direction, funds.available)
        else:
            side = "BUY" if plan.direction == TradeDirection.LONG else "SELL"

        # MARKET orders have no limit price — fill at best available
        if plan.entry_type == EntryType.MARKET:
            entry_price = None
        else:
            entry_price = float(plan.entry_zone)

        # ── BUY_ONLY bracket flip: CE-computed SL/TP need mirroring for PE ──
        # When BUY_ONLY flips SHORT CE → BUY PE, the SL/TP levels were computed
        # for the CE direction (SL above entry, TP below). A LONG PE needs them
        # reversed: SL below entry, TP above. This must happen BEFORE bracket
        # validation and order placement — the trailing service runs too late.
        entry_ref = entry_price or plan.entry_target()
        if (plan.direction == TradeDirection.SHORT
                and plan.symbol.upper().endswith("PE")
                and entry_ref > 0):
            sl = plan.stop_loss
            tp1 = plan.take_profit_1
            tp2 = plan.take_profit_2
            sl_dist = abs(entry_ref - sl)
            plan.stop_loss = entry_ref - sl_dist
            plan.take_profit_1 = entry_ref + abs(entry_ref - tp1)
            plan.take_profit_2 = entry_ref + abs(entry_ref - tp2)
            side = "BUY"  # BUY_ONLY forces buy side for PE positions
            log.info(
                "BUY_ONLY bracket flip: SHORT CE→LONG PE. "
                "SL %.2f→%.2f TP1 %.2f→%.2f TP2 %.2f→%.2f",
                sl, plan.stop_loss, tp1, plan.take_profit_1, tp2, plan.take_profit_2,
            )

        # ── Post-loss size reduction (lot-aware) ──
        if self._session is not None:
            mult = self._session.size_multiplier
            if mult < 1.0:
                lot = int(plan.meta.get("lot_size") or 1)
                lots = max(1, plan.position_size // max(1, lot))
                reduced_qty = max(1, int(lots * mult)) * max(1, lot)
                if reduced_qty < plan.position_size:
                    log.info(
                        "Post-loss size reduction for %s: %d → %d units (×%.1f)",
                        plan.symbol, plan.position_size, reduced_qty, mult,
                    )
                    plan.position_size = reduced_qty

        bracket_ok, bracket_reason = validate_bracket(plan, float(plan.entry_zone))
        if not bracket_ok:
            log.error("Bracket validation failed for %s: %s — entry blocked (never naked)", plan.symbol, bracket_reason)
            plan.status = PlanStatus.INVALIDATED
            self.trade_logger.log(
                "ORDER", plan.symbol, self.broker.name,
                f"Entry blocked — {bracket_reason}",
                {"success": False, "never_naked": True},
            )
            return plan

        result = self.broker.place_order(
            symbol=plan.symbol,
            side=side,
            qty=plan.position_size,
            order_type=plan.entry_type.value if plan.entry_type.value != "STOP_ENTRY" else "STOP_LOSS",
            price=entry_price,
            sl=plan.stop_loss,
            tp=plan.take_profit_1,
        )

        if not result.success:
            self.limits.record_api_error()
            attempts = int(plan.meta.get("_placement_attempts") or 0) + 1
            plan.meta["_placement_attempts"] = attempts
            if attempts >= self.MAX_PLACEMENT_ATTEMPTS:
                plan.status = PlanStatus.INVALIDATED
                log.error(
                    "Order placement failed %d times for %s — invalidating plan",
                    attempts, plan.symbol,
                )
            else:
                plan.meta["_next_attempt_at"] = _time.time() + self.PLACEMENT_RETRY_SECONDS
                log.warning(
                    "Order placement failed for %s (attempt %d/%d) — retrying in %ds",
                    plan.symbol, attempts, self.MAX_PLACEMENT_ATTEMPTS, self.PLACEMENT_RETRY_SECONDS,
                )
            self.trade_logger.log(
                "ORDER", plan.symbol, self.broker.name, result.message,
                {"success": False, "attempts": attempts},
            )
            return plan

        if not result.sl_order_id or not result.tp_order_id:
            log.error("Naked entry blocked — SL/TP not confirmed for %s", plan.symbol)
            if result.order_id:
                self.broker.cancel_order(result.order_id)
                flatten = "SELL" if side == "BUY" else "BUY"
                self.broker.place_order(plan.symbol, flatten, plan.position_size, "MARKET")
            plan.status = PlanStatus.INVALIDATED
            self.trade_logger.log(
                "ORDER", plan.symbol, self.broker.name,
                "Entry reversed — bracket SL/TP missing (never naked)",
                {"success": False, "order_id": result.order_id},
            )
            return plan

        self.limits.record_api_success()
        plan.meta.pop("_placement_attempts", None)
        plan.meta.pop("_next_attempt_at", None)
        plan.status = PlanStatus.ACTIVE
        plan.order_id = result.order_id
        plan.sl_order_id = result.sl_order_id
        plan.tp_order_id = result.tp_order_id
        plan.entry_price = result.fill_price or entry_price
        self.limits.record_trade_opened()

        self.trade_logger.log(
            "ORDER", plan.symbol, self.broker.name,
            f"Bracketed entry {side} qty={plan.position_size} @ {plan.entry_price}",
            {
                "order_id": result.order_id,
                "sl_order_id": result.sl_order_id,
                "tp_order_id": result.tp_order_id,
                "simulation": self.cfg.simulation_mode,
                "success": True,
                "plan": plan.to_dict(),
            },
        )

        return plan

    def modify_sl_breakeven(self, plan: TradePlan) -> None:
        if not plan.sl_order_id or not plan.entry_price:
            return
        result = self.broker.modify_order(plan.sl_order_id, sl=plan.entry_price)
        plan.stop_loss = plan.entry_price
        self.trade_logger.log(
            "MODIFY", plan.symbol, self.broker.name,
            f"SL moved to breakeven {plan.entry_price}",
            {"sl_order_id": plan.sl_order_id, "success": result.success},
        )

    def apply_trail(self, plan: TradePlan, update) -> None:
        """Push trailed SL/TP levels to the broker bracket orders."""
        from risk.trailing import TrailUpdate

        if not isinstance(update, TrailUpdate):
            return

        sl_ok = tp_ok = True
        if update.new_sl is not None and plan.sl_order_id:
            result = self.broker.modify_order(plan.sl_order_id, sl=update.new_sl)
            sl_ok = result.success
            if sl_ok:
                plan.stop_loss = update.new_sl

        if update.new_tp is not None and plan.tp_order_id:
            result = self.broker.modify_order(plan.tp_order_id, tp=update.new_tp)
            tp_ok = result.success
            if tp_ok:
                plan.take_profit_1 = update.new_tp

        if update.new_tp2 is not None:
            plan.take_profit_2 = update.new_tp2

        if update.tp1_milestone:
            plan.tp1_hit = True

        if update.new_sl is not None or update.new_tp is not None:
            self.trade_logger.log(
                "MODIFY", plan.symbol, self.broker.name,
                update.reason or "Trailing SL/TP updated",
                {
                    "new_sl": update.new_sl,
                    "new_tp": update.new_tp,
                    "new_tp2": update.new_tp2,
                    "sl_ok": sl_ok,
                    "tp_ok": tp_ok,
                    "plan": plan.to_dict(),
                },
            )
            log.info(
                "Trailed %s: SL=%s TP=%s (%s)",
                plan.symbol, plan.stop_loss, plan.take_profit_1, update.reason,
            )

    def partial_exit(self, plan: TradePlan, pct: float) -> None:
        qty = max(1, int(plan.position_size * pct / 100))

        # Account for BUY_ONLY flip — same logic as close_plan
        if self._balance is not None:
            is_buy_only_pe = (
                plan.symbol.upper().endswith("PE")
                and plan.direction == TradeDirection.SHORT
            )
            if is_buy_only_pe:
                side = "SELL"
            elif plan.direction == TradeDirection.LONG:
                side = "SELL"
            else:
                side = "BUY"
        else:
            side = "SELL" if plan.direction == TradeDirection.LONG else "BUY"

        result = self.broker.place_order(plan.symbol, side, qty, "MARKET")
        plan.position_size -= qty
        self.trade_logger.log(
            "CLOSE", plan.symbol, self.broker.name,
            f"Partial exit {pct}% ({qty} units)",
            {"success": result.success},
        )

    def close_plan(self, plan: TradePlan, reason: str = "manual close") -> None:
        if plan.position_size <= 0:
            plan.status = PlanStatus.CLOSED
            return

        # Determine close side — account for BUY_ONLY CE/PE flips.
        # When BUY_ONLY + SHORT bias → we BUY a PE (LONG the option).
        # Closing means SELL-ing the PE, regardless of plan.direction.
        if self._balance is not None:
            # BUY_ONLY flip: SHORT+PE → actual position is LONG the option → close with SELL
            is_buy_only_pe = (
                plan.symbol.upper().endswith("PE")
                and plan.direction == TradeDirection.SHORT
            )
            if is_buy_only_pe:
                side = "SELL"  # close LONG PE position
            elif plan.direction == TradeDirection.LONG:
                side = "SELL"
            else:
                side = "BUY"
        else:
            side = "SELL" if plan.direction == TradeDirection.LONG else "BUY"

        # Cancel open bracket orders (best-effort — don't crash if broker fails)
        for oid, label in [
            (plan.order_id, "entry"),
            (plan.sl_order_id, "SL"),
            (plan.tp_order_id, "TP"),
        ]:
            if oid:
                try:
                    self.broker.cancel_order(oid)
                except Exception as e:
                    log.warning("cancel_order(%s) failed for %s: %s", label, plan.symbol, e)

        # Place closing MARKET order
        try:
            result = self.broker.place_order(plan.symbol, side, plan.position_size, "MARKET")
        except Exception as e:
            log.error("Close order failed for %s: %s", plan.symbol, e)
            result = type("FakeResult", (), {"success": False, "message": str(e), "fill_price": None})()

        # ── Capture realized exit price for P&L tracking ──
        # Prefer the close-order fill price; fall back to the live quote.
        exit_px = float(getattr(result, "fill_price", None) or 0.0)
        if exit_px <= 0:
            try:
                exit_px = self.broker.get_quote(plan.symbol).ltp
            except Exception:
                exit_px = 0.0
        if exit_px > 0 and plan.entry_price and plan.entry_price > 0:
            # The actual position direction follows the close side: we SELL
            # to close LONG positions (incl. BUY_ONLY flipped PE plans).
            position_long = side == "SELL"
            pnl = (exit_px - plan.entry_price) * plan.position_size
            if not position_long:
                pnl = -pnl
            plan.meta["exit_price"] = exit_px
            plan.meta["realized_pnl"] = pnl
            log.info(
                "Realized P&L for %s: entry=%.2f exit=%.2f qty=%d → ₹%.0f",
                plan.symbol, plan.entry_price, exit_px, plan.position_size, pnl,
            )

        plan.status = PlanStatus.CLOSED
        self.trade_logger.log(
            "CLOSE", plan.symbol, self.broker.name, reason,
            {"success": result.success, "qty": plan.position_size, "plan": plan.to_dict()},
        )

    def square_off_intraday(self, plans: list[TradePlan]) -> None:
        if not is_square_off_time(self.cfg.intraday_square_off):
            return
        for plan in plans:
            if plan.is_intraday and plan.status == PlanStatus.ACTIVE:
                log.warning("Intraday square-off: closing %s", plan.symbol)
                self.close_plan(plan, reason="Intraday square-off 15:15 IST")
