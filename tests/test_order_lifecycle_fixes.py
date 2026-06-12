"""Regression tests for the place-then-cancel order churn fixes.

Covers:
- Premium-scale conversion when the strike selector picks an option from an
  index trading symbol (the root cause of index-scale/negative bracket prices)
- Rejection of implausible SL/TP distances and spot-scale option quotes
- Dhan place_order per-leg failure handling (honest partial results)
- Executor naked-entry guard and placement retry cooldown (single attempt
  per cycle, no duplicate entries)
"""

from __future__ import annotations

import time
from dataclasses import replace
from datetime import datetime, timezone

import pytest

import agent.executor as executor_mod
import analysis.pipeline as pipeline_mod
from agent.executor import TradeExecutor
from analysis.macro import MacroContext, MacroEnvironment
from analysis.fundamental import FundamentalContext
from analysis.pipeline import AnalysisPipeline
from analysis.technical import SetupType, TechnicalContext, Trend
from audit.trade_logger import TradeLogger
from config import load_config
from models.orders import OHLCV, Funds, OrderResult, Quote
from models.trade_plan import EntryType, PlanStatus, TradeDirection, TradePlan
from providers.base import BrokerProvider
from risk.limits import LimitsGuard
from risk.manager import RiskManager


# ------------------------------------------------------------------ #
# Fakes
# ------------------------------------------------------------------ #
class FakeBroker(BrokerProvider):
    name = "fake"

    def __init__(self, *, spot: float = 24500.0, option_premium: float = 180.0):
        self.spot = spot
        self.option_premium = option_premium
        self.orders: list[dict] = []
        self.cancelled: list[str] = []
        self.place_results: list[OrderResult] = []

    def get_funds(self) -> Funds:
        return Funds(available=100_000.0, invested=0.0, total=100_000.0)

    def get_positions(self):
        return []

    def get_orders(self):
        return []

    def place_order(self, symbol, side, qty, order_type, price=None, sl=None, tp=None) -> OrderResult:
        self.orders.append(
            {"symbol": symbol, "side": side, "qty": qty, "order_type": order_type,
             "price": price, "sl": sl, "tp": tp}
        )
        if self.place_results:
            return self.place_results.pop(0)
        return OrderResult(
            success=True, order_id=f"E{len(self.orders)}", message="ok",
            fill_price=price, sl_order_id="S1", tp_order_id="T1",
        )

    def modify_order(self, order_id, sl=None, tp=None, qty=None) -> OrderResult:
        return OrderResult(success=True, order_id=order_id, message="ok")

    def cancel_order(self, order_id) -> bool:
        self.cancelled.append(order_id)
        return True

    def get_quote(self, symbol) -> Quote:
        upper = symbol.upper()
        ltp = self.option_premium if upper.endswith(("CE", "PE")) else self.spot
        return Quote(symbol=symbol, ltp=ltp, bid=ltp, ask=ltp, volume=0)

    def get_ohlcv(self, symbol, timeframe, limit):
        now = datetime.now(timezone.utc)
        return [
            OHLCV(timestamp=now, open=self.spot, high=self.spot * 1.001,
                  low=self.spot * 0.999, close=self.spot, volume=1000)
            for _ in range(min(limit, 50))
        ]


def _cfg(monkeypatch):
    monkeypatch.setenv("TRADING_SYMBOL", "NIFTY50IDX")
    monkeypatch.setenv("ACTIVE_BROKER", "dhan")
    monkeypatch.setenv("MIN_RR_RATIO", "2.0")
    monkeypatch.setenv("SIMULATION_MODE", "true")
    return load_config()


def _tech(direction=TradeDirection.LONG, entry=24500.0, sl=24400.0,
          tp1=24800.0, tp2=24950.0) -> TechnicalContext:
    return TechnicalContext(
        htf_trend=Trend.UPTREND if direction == TradeDirection.LONG else Trend.DOWNTREND,
        ltf_trend=Trend.UPTREND if direction == TradeDirection.LONG else Trend.DOWNTREND,
        above_200ema=direction == TradeDirection.LONG,
        support=sl,
        resistance=tp1,
        ltf_aligned=True,
        entry=entry,
        stop_loss=sl,
        tp1=tp1,
        tp2=tp2,
        rr_ratio=3.0,
        direction=direction,
        bias_source="test",
        signal_strength=0.9,
        setup_type=SetupType.SWING,
    )


def _patch_pipeline_steps(monkeypatch, tech: TechnicalContext):
    monkeypatch.setattr(
        pipeline_mod, "classify_macro",
        lambda *a, **k: MacroContext(MacroEnvironment.NEUTRAL, [], "Macro: NEUTRAL"),
    )
    monkeypatch.setattr(
        pipeline_mod, "check_fundamental",
        lambda symbol: FundamentalContext(symbol, "Index", [], False, "ok"),
    )
    monkeypatch.setattr(pipeline_mod, "analyze_technical", lambda *a, **k: tech)


def _pipeline(broker, cfg, tmp_path) -> AnalysisPipeline:
    logger = TradeLogger(tmp_path / "trade_log.jsonl")
    risk = RiskManager(cfg.risk_pct_per_trade, cfg.min_rr_ratio)
    return AnalysisPipeline(broker, cfg, logger, risk)


# ------------------------------------------------------------------ #
# Pipeline: premium-scale conversion
# ------------------------------------------------------------------ #
def test_index_symbol_plan_uses_option_premium_scale(monkeypatch, tmp_path):
    """NIFTY50IDX config → strike selector picks an option contract; the plan
    must carry premium-scale entry/SL/TP, not index points."""
    cfg = _cfg(monkeypatch)
    broker = FakeBroker(spot=24500.0, option_premium=180.0)
    _patch_pipeline_steps(monkeypatch, _tech())
    pipe = _pipeline(broker, cfg, tmp_path)

    result = pipe.run("NIFTY50IDX")

    assert result.plan is not None, f"rejected: {result.rejected_reason}"
    plan = result.plan
    assert plan.symbol.upper().endswith("CE")
    assert float(plan.entry_zone) == pytest.approx(180.0)
    # Premium scale: all bracket levels far below index scale
    assert 0 < plan.stop_loss < 180.0
    assert 180.0 < plan.take_profit_1 < plan.take_profit_2 < 1000.0


def test_pipeline_rejects_absurd_sl_tp_distances(monkeypatch, tmp_path):
    """Garbage technical levels (negative TP — seen in real trade logs) must
    be rejected instead of producing negative bracket prices."""
    cfg = _cfg(monkeypatch)
    broker = FakeBroker(spot=24500.0, option_premium=180.0)
    _patch_pipeline_steps(
        monkeypatch,
        _tech(direction=TradeDirection.SHORT, entry=557.0, sl=769.93,
              tp1=-81.44, tp2=-100.0),
    )
    pipe = _pipeline(broker, cfg, tmp_path)

    result = pipe.run("NIFTY50IDX")

    assert result.plan is None
    assert "Implausible SL/TP" in result.rejected_reason


def test_pipeline_rejects_spot_scale_option_quote(monkeypatch, tmp_path):
    """When the option quote falls back to the underlying spot (Dhan quirk),
    the plan must be paused rather than priced at index scale."""
    cfg = _cfg(monkeypatch)
    broker = FakeBroker(spot=24500.0, option_premium=24500.0)  # quote == spot
    _patch_pipeline_steps(monkeypatch, _tech())
    pipe = _pipeline(broker, cfg, tmp_path)

    result = pipe.run("NIFTY50IDX")

    assert result.plan is None
    assert "implausible" in result.rejected_reason.lower()


# ------------------------------------------------------------------ #
# Executor: naked-entry guard and retry cooldown
# ------------------------------------------------------------------ #
def _plan(**overrides) -> TradePlan:
    defaults = dict(
        symbol="NIFTY50IDX",
        direction=TradeDirection.LONG,
        timeframe="15m",
        bias_source="test",
        entry_zone="180",
        entry_type=EntryType.LIMIT,
        stop_loss=170.0,
        stop_loss_reason="test",
        take_profit_1=200.0,
        tp1_exit_pct=50.0,
        take_profit_2=210.0,
        tp2_exit_pct=50.0,
        risk_amount=1000.0,
        position_size=65,
        rr_ratio=2.0,
        status=PlanStatus.PENDING,
        meta={"lot_size": 65},
    )
    defaults.update(overrides)
    return TradePlan(**defaults)


def _executor(broker, cfg, tmp_path, monkeypatch) -> TradeExecutor:
    monkeypatch.setattr(executor_mod, "is_market_open", lambda *a, **k: True)
    monkeypatch.setattr(executor_mod, "is_within_trade_window", lambda *a, **k: True)
    logger = TradeLogger(tmp_path / "trade_log.jsonl")
    limits = LimitsGuard(3, 5.0, 0, [])
    return TradeExecutor(broker, cfg, logger, limits)


def test_naked_entry_is_cancelled_and_flattened(monkeypatch, tmp_path):
    """Entry filled but bracket legs missing → cancel entry + market flatten,
    plan invalidated (never naked, never retried)."""
    cfg = _cfg(monkeypatch)
    broker = FakeBroker()
    broker.place_results = [
        OrderResult(success=True, order_id="E1", message="ok",
                    sl_order_id=None, tp_order_id=None),
        OrderResult(success=True, order_id="F1", message="ok"),  # flatten
    ]
    execu = _executor(broker, cfg, tmp_path, monkeypatch)
    plan = _plan()

    execu.execute_plan(plan, open_count=0)

    assert plan.status == PlanStatus.INVALIDATED
    assert broker.cancelled == ["E1"]
    assert len(broker.orders) == 2
    flatten = broker.orders[1]
    assert flatten["side"] == "SELL"
    assert flatten["order_type"] == "MARKET"


def test_failed_placement_uses_cooldown_not_immediate_retry(monkeypatch, tmp_path):
    """A failed placement must not be retried in the same cycle; after the
    max attempts the plan is invalidated instead of looping forever."""
    cfg = _cfg(monkeypatch)
    broker = FakeBroker()
    broker.place_results = [
        OrderResult(success=False, order_id=None, message="rejected")
        for _ in range(10)
    ]
    execu = _executor(broker, cfg, tmp_path, monkeypatch)
    plan = _plan()

    execu.execute_plan(plan, open_count=0)
    assert plan.status == PlanStatus.PENDING
    assert plan.meta["_placement_attempts"] == 1
    assert len(broker.orders) == 1

    # Same cycle / cooldown active → no new broker call
    execu.execute_plan(plan, open_count=0)
    assert len(broker.orders) == 1

    # Cooldown elapsed → one more attempt
    plan.meta["_next_attempt_at"] = time.time() - 1
    execu.execute_plan(plan, open_count=0)
    assert plan.meta["_placement_attempts"] == 2
    assert len(broker.orders) == 2

    # Final attempt → invalidated
    plan.meta["_next_attempt_at"] = time.time() - 1
    execu.execute_plan(plan, open_count=0)
    assert plan.meta["_placement_attempts"] == 3
    assert plan.status == PlanStatus.INVALIDATED

    # Invalidated plans are never executed again
    execu.execute_plan(plan, open_count=0)
    assert len(broker.orders) == 3


def test_successful_close_records_realized_pnl(monkeypatch, tmp_path):
    """close_plan must capture the realized exit price and P&L in plan meta."""
    cfg = _cfg(monkeypatch)
    broker = FakeBroker(option_premium=195.0)
    execu = _executor(broker, cfg, tmp_path, monkeypatch)
    plan = _plan(symbol="NIFTY26JUN24500CE", status=PlanStatus.ACTIVE,
                 entry_price=180.0, order_id="E1", sl_order_id="S1", tp_order_id="T1")

    broker.place_results = [
        OrderResult(success=True, order_id="C1", message="ok", fill_price=195.0),
    ]
    execu.close_plan(plan, reason="test close")

    assert plan.status == PlanStatus.CLOSED
    assert plan.meta["exit_price"] == pytest.approx(195.0)
    assert plan.meta["realized_pnl"] == pytest.approx((195.0 - 180.0) * 65)


def test_derived_option_contract_is_accepted_by_executor(monkeypatch, tmp_path):
    """A contract the strike selector derived from the configured index
    (NIFTY50IDX → NIFTY26JUN24500CE) must execute, not be rejected."""
    cfg = _cfg(monkeypatch)
    broker = FakeBroker()
    execu = _executor(broker, cfg, tmp_path, monkeypatch)
    plan = _plan(symbol="NIFTY26JUN24500CE")

    execu.execute_plan(plan, open_count=0)

    assert plan.status == PlanStatus.ACTIVE
    assert len(broker.orders) == 1


def test_foreign_underlying_is_still_rejected(monkeypatch, tmp_path):
    cfg = _cfg(monkeypatch)
    broker = FakeBroker()
    execu = _executor(broker, cfg, tmp_path, monkeypatch)
    plan = _plan(symbol="BANKNIFTY26JUN54000CE")

    execu.execute_plan(plan, open_count=0)

    assert plan.status == PlanStatus.INVALIDATED
    assert not broker.orders


def test_canonical_underlying():
    from utils.symbols import canonical_underlying, same_underlying

    assert canonical_underlying("NIFTY50IDX") == "NIFTY"
    assert canonical_underlying("NIFTY26JUN24500CE") == "NIFTY"
    assert canonical_underlying("NIFTY24500CE") == "NIFTY"
    assert canonical_underlying("BANKNIFTY26JUN54000PE") == "BANKNIFTY"
    assert same_underlying("NIFTY50IDX", "NIFTY26JUN24500CE")
    assert not same_underlying("NIFTY50IDX", "BANKNIFTY26JUN54000CE")


# ------------------------------------------------------------------ #
# Dhan provider: per-leg failures and honest simulation
# ------------------------------------------------------------------ #
class FakeHttp:
    """First post succeeds (entry), every later post raises (bracket legs)."""

    def __init__(self, entry_response=None, fail_after: int = 1):
        self.entry_response = {"orderId": "E1"} if entry_response is None else entry_response
        self.fail_after = fail_after
        self.post_calls = 0

    def post(self, path, json=None):
        self.post_calls += 1
        if self.post_calls > self.fail_after:
            raise RuntimeError("leg rejected")
        return self.entry_response

    def close(self):
        pass


def test_dhan_leg_failure_returns_partial_result(monkeypatch):
    """Entry placed but SL/TP legs fail → success=True with the entry id and
    missing leg ids, so the executor can roll back instead of duplicating."""
    from providers.dhan import DhanProvider

    cfg = replace(_cfg(monkeypatch), simulation_mode=False,
                  dhan_access_token="t", dhan_client_id="c")
    provider = DhanProvider(cfg)
    fake = FakeHttp()
    provider._http = fake

    result = provider.place_order(
        "NIFTY26JUN24500CE", "BUY", 65, "LIMIT", price=180.0, sl=170.0, tp=200.0,
    )

    assert result.success is True
    assert result.order_id == "E1"
    assert result.sl_order_id is None
    assert result.tp_order_id is None
    assert "bracket leg" in result.message.lower()
    # 1 entry + 2 SL attempts (retry) + 2 TP attempts (retry)
    assert fake.post_calls == 5


def test_dhan_api_error_is_not_simulated_success(monkeypatch):
    """An API error in simulation mode must surface as a failed order,
    never a fabricated success."""
    from providers.dhan import DhanProvider

    cfg = _cfg(monkeypatch)  # simulation_mode=True
    provider = DhanProvider(cfg)
    provider._http = FakeHttp(fail_after=0)  # every call raises

    result = provider.place_order(
        "NIFTY26JUN24500CE", "BUY", 65, "LIMIT", price=180.0, sl=170.0, tp=200.0,
    )

    assert result.success is False
    assert result.order_id is None


def test_dhan_empty_order_id_is_failure_in_live_mode(monkeypatch):
    """A live order response without an orderId cannot be tracked — failure."""
    from providers.dhan import DhanProvider

    cfg = replace(_cfg(monkeypatch), simulation_mode=False,
                  dhan_access_token="t", dhan_client_id="c")
    provider = DhanProvider(cfg)
    provider._http = FakeHttp(entry_response={}, fail_after=99)

    result = provider.place_order("NIFTY26JUN24500CE", "BUY", 65, "MARKET")

    assert result.success is False
    assert result.order_id is None
