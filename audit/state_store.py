"""Persist plans and recover state after crash."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from models.trade_plan import PlanStatus, TradePlan

log = logging.getLogger("dream_maker.state")


class StateStore:
    def __init__(self, state_dir: Path, trade_log_path: Path):
        self.state_dir = state_dir
        self.trade_log_path = trade_log_path
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.plans_path = self.state_dir / "active_plans.json"
        self.heartbeat_path = self.state_dir / "heartbeat.json"

    def save_plans(self, plans: list[TradePlan]) -> None:
        persist = [
            p.to_dict()
            for p in plans
            if p.status in {PlanStatus.WAITING_ENTRY, PlanStatus.ACTIVE, PlanStatus.PENDING}
        ]
        self.plans_path.write_text(json.dumps(persist, indent=2), encoding="utf-8")

    def load_plans(self) -> list[TradePlan]:
        if not self.plans_path.exists():
            return []
        try:
            raw = json.loads(self.plans_path.read_text(encoding="utf-8"))
            return [TradePlan.from_dict(item) for item in raw]
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            log.warning("Could not load active_plans.json: %s", e)
            return []

    def load_from_trade_log(self, trading_symbol: str) -> TradePlan | None:
        if not self.trade_log_path.exists():
            return None
        last_order: dict[str, Any] | None = None
        for line in self.trade_log_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("action") != "ORDER":
                continue
            details = entry.get("details") or {}
            if not details.get("success", True):
                continue
            if entry.get("symbol", "").upper() != trading_symbol.upper():
                continue
            if details.get("plan"):
                last_order = entry
        if not last_order:
            return None
        plan_data = (last_order.get("details") or {}).get("plan")
        if not plan_data:
            return None
        try:
            plan = TradePlan.from_dict(plan_data)
            if plan.status != PlanStatus.CLOSED:
                plan.status = PlanStatus.ACTIVE
            return plan
        except (KeyError, ValueError) as e:
            log.warning("Could not reconstruct plan from trade log: %s", e)
            return None

    def write_heartbeat(
        self,
        *,
        loop_count: int,
        active_plans: int,
        status: str = "running",
        market: dict[str, object] | None = None,
    ) -> None:
        from datetime import datetime, timezone

        payload: dict[str, object] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "loop_count": loop_count,
            "active_plans": active_plans,
            "pid": __import__("os").getpid(),
        }
        if market:
            payload["market"] = market
        self.heartbeat_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def is_stale(self, max_age_seconds: int = 180) -> bool:
        if not self.heartbeat_path.exists():
            return True
        from datetime import datetime, timezone

        try:
            data = json.loads(self.heartbeat_path.read_text(encoding="utf-8"))
            ts = datetime.fromisoformat(data["timestamp"])
            age = (datetime.now(timezone.utc) - ts).total_seconds()
            return age > max_age_seconds
        except (json.JSONDecodeError, KeyError, ValueError):
            return True
