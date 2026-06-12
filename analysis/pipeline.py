"""5-step market analysis pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from analysis.fundamental import check_fundamental
from analysis.macro import MacroContext, classify_macro
from analysis.technical import SetupType, TechnicalContext, analyze_technical, momentum_confirmation
from config import Config
from audit.trade_logger import TradeLogger
from models.trade_plan import EntryType, PlanStatus, TradeDirection, TradePlan, validate_plan
from providers.base import BrokerProvider
from risk.manager import RiskManager
from risk.strike_selector import select_strike_and_size
from utils.symbols import underlying_base

log = logging.getLogger("dream_maker.pipeline")


@dataclass
class PipelineResult:
    symbol: str
    macro: MacroContext
    technical: TechnicalContext | None
    fundamental_summary: str
    plan: TradePlan | None
    rejected_reason: str = ""


class AnalysisPipeline:
    def __init__(
        self,
        broker: BrokerProvider,
        cfg: Config,
        trade_logger: TradeLogger,
        risk: RiskManager,
    ):
        self.broker = broker
        self.cfg = cfg
        self.trade_logger = trade_logger
        self.risk = risk

    def run(self, symbol: str) -> PipelineResult:
        macro = classify_macro()
        self.trade_logger.log(
            "PLAN", symbol, self.broker.name,
            f"Step 1 Macro: {macro.environment.value}",
            {"headlines": macro.headlines[:3], "summary": macro.summary},
        )

        htf_symbol = symbol
        is_option = symbol.upper().endswith("CE") or symbol.upper().endswith("PE")
        if is_option:
            # Extract underlying index from option contract (e.g., "NIFTY" from "NIFTY26JUN23550CE")
            import re as _re
            _base = symbol.upper().replace(" ", "")
            _base = _re.sub(r"\d{2}[A-Z]{3}.*", "", _base)  # strip date+strike: 26JUN23550CE
            _base = _re.sub(r"\d+(CE|PE).*$", "", _base)    # strip strike: 23550CE
            htf_symbol = _base or underlying_base(symbol)
            if htf_symbol != symbol:
                log.info("Option detected — using %s for HTF analysis", htf_symbol)

        htf = self.broker.get_ohlcv(htf_symbol, "1d", 250)
        # For options, use the underlying index for LTF analysis — option candles
        # are too volatile (₹140→₹2 in hours). Only use option LTP for execution.
        ltf_symbol = htf_symbol if is_option else symbol
        ltf = self.broker.get_ohlcv(ltf_symbol, "15m", 100)
        if is_option and len(htf) < 20:
            log.info("HTF using option underlying index (%d candles)", len(htf))
        tech = analyze_technical(htf, ltf, min_rr=self.cfg.min_rr_ratio)
        setup_label = tech.setup_type.value if tech else "none"
        self.trade_logger.log(
            "PLAN", symbol, self.broker.name,
            f"Step 2-3 HTF/LTF ({setup_label}): {tech.htf_trend.value if tech else 'none'}",
            {
                "ltf_aligned": tech.ltf_aligned if tech else False,
                "strength": tech.signal_strength if tech else 0,
                "setup_type": setup_label,
                "confirmations": tech.confirmations if tech else [],
            },
        )

        fund = check_fundamental(symbol)
        self.trade_logger.log(
            "PLAN", symbol, self.broker.name,
            f"Step 4 Fundamental: {fund.summary}",
            {"is_weak": fund.is_weak, "flags": fund.news_flags},
        )

        if tech is None:
            return PipelineResult(symbol, macro, None, fund.summary, None, "No strong technical setup — paused")

        is_scalp = tech.setup_type == SetupType.MOMENTUM_SCALP
        min_strength = self.cfg.scalp_min_signal_strength if is_scalp else self.cfg.min_signal_strength
        required_rr = self.cfg.scalp_min_rr_ratio if is_scalp else self.cfg.min_rr_ratio

        if tech.signal_strength < min_strength:
            return PipelineResult(
                symbol, macro, tech, fund.summary, None,
                f"Signal strength {tech.signal_strength:.2f} below {min_strength} — paused",
            )

        if fund.is_weak:
            return PipelineResult(symbol, macro, tech, fund.summary, None, "Fundamentally weak — paused")

        if macro.opposing_long and tech.direction == TradeDirection.LONG and not is_scalp:
            return PipelineResult(symbol, macro, tech, fund.summary, None, "Macro opposes LONG — paused")
        if macro.opposing_short and tech.direction == TradeDirection.SHORT and not is_scalp:
            return PipelineResult(symbol, macro, tech, fund.summary, None, "Macro opposes SHORT — paused")

        funds = self.broker.get_funds()
        quote = self.broker.get_quote(symbol)
        # For options, use underlying index close as spot for premium estimation,
        # NOT the option LTP (which can be ₹14 for a far-OTM, skewing margin 100x).
        spot_for_pricing = float(htf[-1].close) if is_option else (quote.ltp or tech.entry)
        strike = select_strike_and_size(
            symbol,
            tech.direction,
            tech.entry,
            tech.stop_loss,
            available_funds=funds.available,
            risk_pct=self.cfg.risk_pct_per_trade,
            ltp=spot_for_pricing,
        )
        if not strike.affordable or strike.qty <= 0:
            return PipelineResult(symbol, macro, tech, fund.summary, None, strike.reason)

        sizing = self.risk.compute_size(
            funds.total, tech.entry, tech.stop_loss, tech.tp1,
            lot_size=max(1, strike.qty),
        )
        qty = min(strike.qty, sizing.qty) if sizing.qty > 0 else strike.qty

        self.trade_logger.log(
            "PLAN", symbol, self.broker.name,
            f"Step 5 Projection R:R={tech.rr_ratio:.2f} strike={strike.strike}",
            {"entry": tech.entry, "sl": tech.stop_loss, "tp1": tech.tp1, "qty": qty, "margin": strike.margin_required},
        )

        if qty <= 0 or tech.rr_ratio < required_rr:
            reason = sizing.reason if sizing.qty <= 0 else f"R:R {tech.rr_ratio:.2f} below {required_rr}"
            return PipelineResult(symbol, macro, tech, fund.summary, None, reason)

        tol = self.cfg.entry_zone_tolerance_pct / 100.0
        if is_scalp:
            tol = min(tol, self.cfg.scalp_max_sl_pct / 100.0 * 0.5)

        # ── Option premium scaling: tech levels are in NIFTY points (underlying).
        # Convert to option premium equivalents so entry_watcher matches correctly.
        option_ltp = quote.ltp
        if is_option and option_ltp > 0 and tech.entry > 0:
            nifty_entry = tech.entry
            sl_pct = abs(nifty_entry - tech.stop_loss) / nifty_entry
            tp1_pct = abs(tech.tp1 - nifty_entry) / nifty_entry
            tp2_pct = abs(tech.tp2 - nifty_entry) / nifty_entry if tech.tp2 else 0
            if tech.direction == TradeDirection.LONG:
                opt_sl = option_ltp * (1 - sl_pct)
                opt_tp1 = option_ltp * (1 + tp1_pct)
                opt_tp2 = option_ltp * (1 + tp2_pct) if tp2_pct else 0
            else:
                opt_sl = option_ltp * (1 + sl_pct)
                opt_tp1 = option_ltp * (1 - tp1_pct)
                opt_tp2 = option_ltp * (1 - tp2_pct) if tp2_pct else 0
            opt_entry = option_ltp
            log.info(
                "Option scaling: NIFTY entry=%.0f→opt=%.2f SL=%.0f→%.2f TP=%.0f→%.2f",
                nifty_entry, opt_entry, tech.stop_loss, opt_sl, tech.tp1, opt_tp1,
            )
        else:
            opt_entry = tech.entry
            opt_sl = tech.stop_loss
            opt_tp1 = tech.take_profit_1
            opt_tp2 = tech.take_profit_2

        plan = TradePlan(
            symbol=strike.tradable_symbol,
            direction=tech.direction,
            timeframe="15m",
            bias_source=tech.bias_source,
            entry_zone=f"{opt_entry:.2f}",
            entry_type=EntryType.LIMIT,
            stop_loss=opt_sl,
            stop_loss_reason=f"HTF {'support' if tech.direction == TradeDirection.LONG else 'resistance'}",
            take_profit_1=opt_tp1,
            tp1_exit_pct=50.0,
            take_profit_2=opt_tp2,
            tp2_exit_pct=50.0,
            risk_amount=sizing.risk_amount,
            position_size=qty,
            rr_ratio=tech.rr_ratio,
            status=PlanStatus.WAITING_ENTRY,
            entry_price_low=opt_entry * (1 - tol),
            entry_price_high=opt_entry * (1 + tol),
            strike_price=strike.strike,
            signal_strength=tech.signal_strength,
            macro_env=macro.environment.value,
            rationale=f"{tech.bias_source}; {fund.summary}; {strike.reason}",
            is_intraday=True,
            meta={
                "setup_type": tech.setup_type.value,
                "confirmations": tech.confirmations,
            },
        )
        # ── BUY_ONLY bracket flip: SHORT CE → BUY PE needs mirrored SL/TP ──
        # select_strike_and_size flips CE→PE but keeps SHORT direction.
        # A LONG PE needs SL below entry, TP above, and LONG direction for validation.
        if (plan.direction == TradeDirection.SHORT
                and plan.symbol.upper().endswith("PE")):
            entry = plan.entry_target()
            plan.stop_loss = entry - abs(entry - plan.stop_loss)
            plan.take_profit_1 = entry + abs(entry - plan.take_profit_1)
            plan.take_profit_2 = entry + abs(entry - plan.take_profit_2)
            plan.direction = TradeDirection.LONG
        ok, reason = validate_plan(
            plan,
            min_rr=self.cfg.min_rr_ratio,
            min_rr_scalp=self.cfg.scalp_min_rr_ratio,
        )
        if not ok:
            return PipelineResult(symbol, macro, tech, fund.summary, None, reason)

        return PipelineResult(symbol, macro, tech, fund.summary, plan)
