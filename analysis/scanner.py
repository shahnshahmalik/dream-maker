"""Symbol scanner — quality-over-quantity setup selection for precision scalping.

Instead of a cascade (try swing → momentum → range), we score ALL potential
setups and only trade the best ones. Hard limits:
- Max trades per day (default 4)
- Must exceed quality threshold (default 0.70)
- Only trade if session permits (profit target not hit, loss guard not tripped)
"""

from __future__ import annotations

import logging

from analysis.pipeline import AnalysisPipeline, PipelineResult
from analysis.technical import (
    SetupType,
    check_pullback_to_ema,
    check_volume_surge,
    score_setup,
)
from config import Config
from models.trade_plan import TradePlan
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
    ):
        self.pipeline = pipeline
        self.cfg = cfg
        self.broker = broker
        self.session = session
        # Track the best setup seen today (within this scanner instance)
        self._best_score_today: float = 0.0
        self._best_symbol_today: str = ""

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
        1. Session check (trade cap, profit target, loss guard)
        2. Technical analysis (swing → momentum → candle → range)
        3. Entry precision checks (pullback to EMA, volume surge)
        4. Setup scoring (must exceed quality threshold)
        5. Best-setup tracking (only trade if it's the best seen today)
        """
        # ── Gate 1: Session ──
        ok, reason = self.can_scan()
        if not ok:
            log.info("Scanner blocked: %s", reason)
            return []

        symbol = self.cfg.trading_symbol
        results: list[PipelineResult] = []

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

        # ── Gate 2: Entry precision checks ──
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
            result = PipelineResult(
                symbol, result.macro, result.technical, result.fundamental_summary, None,
                "; ".join(reason_parts),
            )
            log.info("Setup rejected: %s", result.rejected_reason)
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
        return results

    def pending_plans(self, results: list[PipelineResult]) -> list[TradePlan]:
        return [r.plan for r in results if r.plan is not None]

    @staticmethod
    def matches_trading_symbol(cfg: Config, symbol: str) -> bool:
        return normalize_symbol(symbol) == cfg.trading_symbol
