"""Symbol scanner — quality-over-quantity setup selection for precision scalping.

Instead of a cascade (try swing → momentum → range), we score ALL potential
setups and only trade the best ones. Hard limits:
- Max trades per day (default 4)
- Must exceed quality threshold (default 0.70)
- Only trade if session permits (profit target not hit, loss guard not tripped)
"""

from __future__ import annotations

import logging

from analysis.day_classifier import DayGate
from analysis.pipeline import AnalysisPipeline, PipelineResult
from analysis.technical import (
    SetupType,
    check_bb_confirmation,
    check_orb_confirmation,
    check_pullback_to_ema,
    check_volume_surge,
    score_setup,
)
from config import Config
from models.trade_plan import TradePlan
from notifications.service import NotificationService
from providers.base import BrokerProvider
from risk.session_tracker import SessionTracker
from utils.symbols import normalize_symbol

log = logging.getLogger("dream_maker.scanner")


class WatchlistScanner:
    def __init__(
        self,
        pipeline: AnalysisPipeline,
        cfg: Config,
        broker: BrokerProvider,
        session: SessionTracker | None = None,
        notifier: NotificationService | None = None,
    ):
        self.pipeline = pipeline
        self.cfg = cfg
        self.broker = broker
        self.session = session
        self.notifier = notifier
        self._best_score_today: float = 0.0
        self._best_symbol_today: str = ""
        self._day_gate = DayGate(
            broker=broker,
            enabled=getattr(cfg, "day_gate_enabled", True),
        )
        self._day_classified_today: bool = False  # fire day_classified only once per day
        self._day_classified_date: object = None   # tracks which date was classified

    def can_scan(self) -> tuple[bool, str]:
        """Check if we're allowed to generate new trade plans."""
        if self.session is None:
            return True, ""

        if not self.session.can_trade:
            return False, self.session.stop_reason

        return True, ""

    def scan(self) -> list[PipelineResult]:
        """Scan for setups. Returns at most 1 result (the best setup per cycle).

        Quality gates applied in order:
        0. Day-type gate (Range/Inside Day → skip; directional lock applied)
        1. Session check (trade cap, profit target, loss guard)
        2. Technical analysis (stacked sweep)
        3. Direction check against day-type lock
        4. Entry precision checks (pullback to EMA, volume surge)
        5. Setup scoring (must exceed quality threshold)
        6. Best-setup tracking (only trade if it's the best seen today)
        """
        # ── Gate 0: Day-type gate ──
        day_class = self._day_gate.classify_today()

        # Notify day classification once per calendar day
        from datetime import date as _date
        _today = _date.today()
        if self._day_classified_date != _today and self.notifier:
            _BREAKOUT = {"trend_up", "trend_down", "gap_up_trend", "gap_down_trend", "gap_down_rally"}
            strategy_name = "bb_orb_breakout" if day_class.day_type.value in _BREAKOUT else "stacked_sweep"
            allowed_str = (
                day_class.allowed_directions[0].value
                if day_class.allowed_directions and len(day_class.allowed_directions) == 1
                else ("none" if day_class.is_blocked else "any")
            )
            if day_class.is_blocked:
                self.notifier.on_indicator_event("day_blocked", self.cfg.trading_symbol, {
                    "reason": day_class.reason,
                })
            else:
                self.notifier.on_indicator_event("day_classified", self.cfg.trading_symbol, {
                    "day_type": day_class.day_type.value,
                    "allowed_direction": allowed_str,
                    "or_range_pct": round(day_class.or_range_pct * 100, 2),
                    "gap_pct": round(day_class.gap_pct * 100, 2),
                    "strategy": strategy_name,
                })
                self.notifier.on_indicator_event("strategy_routed", self.cfg.trading_symbol, {
                    "strategy": strategy_name,
                    "day_type": day_class.day_type.value,
                    "direction": allowed_str,
                })
            self._day_classified_today = True
            self._day_classified_date = _today

        if day_class.is_blocked:
            log.info("Day gate blocked: %s — %s", day_class.day_type.value, day_class.reason)
            return []

        # ── Gate 1: Session ──
        ok, reason = self.can_scan()
        if not ok:
            log.info("Scanner blocked: %s", reason)
            return []

        symbol = self.cfg.trading_symbol
        results: list[PipelineResult] = []

        # Inject day context into cfg so pipeline can pass it to analyze_technical.
        # Using transient attributes — no persistence, reset each scan cycle.
        self.cfg._day_type = day_class.day_type.value
        self.cfg._allowed_direction = (
            day_class.allowed_directions[0]
            if day_class.allowed_directions and len(day_class.allowed_directions) == 1
            else None
        )
        log.info(
            "Day gate: %s → strategy=%s direction=%s",
            day_class.day_type.value,
            "bb_orb_breakout" if self.cfg._day_type in {"trend_up", "trend_down", "gap_up_trend", "gap_down_trend", "gap_down_rally"} else "stacked_sweep",
            self.cfg._allowed_direction.value if self.cfg._allowed_direction else "any",
        )

        try:
            result = self.pipeline.run(symbol)
        except Exception as e:
            log.exception("Scan failed for %s: %s", symbol, e)
            return []

        if result.plan is None:
            log.info("No plan for %s: %s", symbol, result.rejected_reason)
            return []

        plan = result.plan
        setup_type = plan.meta.get("setup_type", "swing")

        # ── Gate 2.5: Direction locked by day type (sweep only) ──
        # BB_ORB_BREAKOUT already uses allowed_direction internally — no re-check needed.
        from analysis.technical import SetupType
        is_sweep = plan.meta.get("setup_type", "") == SetupType.STACKED_SWEEP.value
        if is_sweep and not day_class.allows_any_direction and not day_class.allows(plan.direction):
            log.info(
                "Direction gate blocked: %s wants %s but day_type=%s only allows %s",
                plan.symbol,
                plan.direction.value,
                day_class.day_type.value,
                [d.value for d in (day_class.allowed_directions or [])],
            )
            return [PipelineResult(
                symbol, result.macro, result.technical, result.fundamental_summary, None,
                f"Day gate: {day_class.day_type.value} does not allow {plan.direction.value}",
            )]

        # ── Gate 3: Entry precision checks ──
        # Fetch LTF candles for pullback/volume checks
        ltf_candles = self.broker.get_ohlcv(symbol, "15m", 100)
        pullback_ok, pullback_reason = True, "pullback check disabled"
        volume_ok, volume_reason = True, "volume check disabled"

        if self.cfg.entry_require_pullback:
            # For options, pass the entry price so the check can adapt tolerance
            opt_premium = 0.0
            if plan.symbol.upper().endswith(("CE", "PE")):
                opt_premium = plan.entry_target()
            pullback_ok, pullback_reason = check_pullback_to_ema(
                ltf_candles, plan.direction, option_premium=opt_premium,
            )

        if self.cfg.entry_require_volume_surge:
            # Penny options (<₹50) have naturally low volume — skip surge check
            if opt_premium > 0 and opt_premium < 50:
                volume_ok, volume_reason = True, 'volume waived for penny option'
            else:
                volume_ok, volume_reason = check_volume_surge(ltf_candles)

        # ── Gate 3: Setup quality score ──
        quality_score = score_setup(
            result.technical,
            pullback_ok=pullback_ok,
            volume_ok=volume_ok,
        )

        log.info(
            "Setup quality: %s %s score=%.2f (pullback=%s vol=%s) — threshold=%.2f",
            symbol, setup_type, quality_score,
            pullback_reason, volume_reason,
            self.cfg.entry_quality_threshold,
        )

        if quality_score < self.cfg.entry_quality_threshold:
            reason_parts = [f"Quality score {quality_score:.2f} < {self.cfg.entry_quality_threshold}"]
            if not pullback_ok:
                reason_parts.append(pullback_reason)
            if not volume_ok:
                reason_parts.append(volume_reason)
            rejected_reason = "; ".join(reason_parts)
            result = PipelineResult(
                symbol, result.macro, result.technical, result.fundamental_summary, None,
                rejected_reason,
            )
            log.info("Setup rejected: %s", result.rejected_reason)
            if self.notifier:
                self.notifier.on_indicator_event("signal_rejected", symbol, {
                    "reason": rejected_reason,
                    "quality_score": quality_score,
                })
            return [result]

        # ── Gate 4: Best-setup tracking ──
        if self.session is not None:
            self.session.track_setup_score(symbol, quality_score)
        self._best_score_today = max(self._best_score_today, quality_score)

        # Store quality score in plan meta for downstream use
        plan.meta["quality_score"] = quality_score
        plan.meta["pullback_reason"] = pullback_reason
        plan.meta["volume_reason"] = volume_reason

        results.append(result)
        log.info(
            "Plan APPROVED: %s | bias=%s | exec=%s | R:R=%.2f | score=%.2f | pullback=%s | vol=%s",
            plan.symbol,
            plan.bias_source,
            plan.direction.value,
            plan.rr_ratio,
            quality_score,
            pullback_reason,
            volume_reason,
        )

        # Notify signal approval with setup-specific details
        if self.notifier:
            tech = result.technical
            _setup_type = plan.meta.get("setup_type", "")
            if _setup_type == SetupType.BB_ORB_BREAKOUT.value:
                self.notifier.on_indicator_event("bb_orb_signal", symbol, {
                    "direction": plan.direction.value,
                    "entry": tech.entry if tech else 0.0,
                    "stop_loss": tech.stop_loss if tech else 0.0,
                    "tp1": tech.tp1 if tech else 0.0,
                    "bb_reason": "bb_confirmed" if tech and "bb_confirmed" in (tech.confirmations or []) else "—",
                    "orb_reason": "orb_confirmed" if tech and "orb_confirmed" in (tech.confirmations or []) else "—",
                })
            else:
                self.notifier.on_indicator_event("sweep_signal", symbol, {
                    "direction": plan.direction.value,
                    "levels": [c for c in (tech.confirmations or []) if c not in ("daily_up", "daily_down")] if tech else [],
                    "strength": tech.signal_strength if tech else 0.0,
                    "entry": tech.entry if tech else 0.0,
                    "stop_loss": tech.stop_loss if tech else 0.0,
                    "tp1": tech.tp1 if tech else 0.0,
                })
            self.notifier.on_indicator_event("signal_fired", symbol, {
                "strategy": _setup_type,
                "direction": plan.direction.value,
                "quality_score": quality_score,
            })

        return results

    def pending_plans(self, results: list[PipelineResult]) -> list[TradePlan]:
        return [r.plan for r in results if r.plan is not None]

    @staticmethod
    def matches_trading_symbol(cfg: Config, symbol: str) -> bool:
        return normalize_symbol(symbol) == cfg.trading_symbol
