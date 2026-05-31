"""Symbol scanner — single F&O symbol from env."""

from __future__ import annotations

import logging

from analysis.pipeline import AnalysisPipeline, PipelineResult
from config import Config
from models.trade_plan import TradePlan
from providers.base import BrokerProvider
from utils.symbols import normalize_symbol

log = logging.getLogger("dream_maker.scanner")


class WatchlistScanner:
    def __init__(self, pipeline: AnalysisPipeline, cfg: Config, broker: BrokerProvider):
        self.pipeline = pipeline
        self.cfg = cfg
        self.broker = broker

    def scan(self) -> list[PipelineResult]:
        symbol = self.cfg.trading_symbol
        results: list[PipelineResult] = []
        try:
            result = self.pipeline.run(symbol)
            results.append(result)
            if result.plan:
                log.info(
                    "Plan generated for %s: %s %s R:R=%.2f",
                    symbol,
                    result.plan.direction.value,
                    result.plan.bias_source,
                    result.plan.rr_ratio,
                )
            else:
                log.info("No plan for %s: %s", symbol, result.rejected_reason)
        except Exception as e:
            log.exception("Scan failed for %s: %s", symbol, e)
        return results

    def pending_plans(self, results: list[PipelineResult]) -> list[TradePlan]:
        return [r.plan for r in results if r.plan is not None]

    @staticmethod
    def matches_trading_symbol(cfg: Config, symbol: str) -> bool:
        return normalize_symbol(symbol) == cfg.trading_symbol
