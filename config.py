"""Central configuration loaded from config.yaml and environment."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv

from utils.symbols import require_fno_symbol

_ROOT = Path(__file__).resolve().parent
load_dotenv(_ROOT / ".env")


def _bool(v: str | None, default: bool = False) -> bool:
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def _load_yaml() -> dict:
    path = _ROOT / "config.yaml"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_trading_symbol(y: dict) -> str:
    raw = os.getenv("TRADING_SYMBOL", "").strip()
    if not raw:
        raise ValueError(
            "TRADING_SYMBOL must be set in .env (e.g. TRADING_SYMBOL=NIFTY50IDX or NIFTY25JUNFUT). "
            "Only F&O-eligible symbols are allowed."
        )
    return require_fno_symbol(raw)


@dataclass(frozen=True)
class Config:
    active_broker: str
    active_llm: str
    monitor_interval: int
    max_open_trades: int
    risk_pct_per_trade: float
    daily_loss_limit: float
    simulation_mode: bool
    trading_hours_ist: str
    min_rr_ratio: float
    trading_symbol: str
    event_blackout_minutes: int
    scheduled_events: list[str]
    intraday_square_off: str
    min_signal_strength: float
    entry_zone_tolerance_pct: float
    ai_review_cooldown_minutes: int
    ai_analysis_cooldown_minutes: int
    fno_margin_rate: float
    state_dir: Path

    dhan_access_token: str
    groww_session_token: str
    openai_api_key: str
    openai_base_url: str
    openai_model: str
    anthropic_api_key: str
    anthropic_model: str

    trade_log_path: Path = field(default_factory=lambda: _ROOT / "trade_log.jsonl")


def load_config() -> Config:
    y = _load_yaml()
    trading_symbol = _load_trading_symbol(y)
    active_broker = os.getenv("ACTIVE_BROKER", y.get("active_broker", "dhan")).lower()
    if active_broker == "groww":
        raise ValueError(
            "Groww does not support F&O. Set ACTIVE_BROKER=dhan for F&O-only trading."
        )

    return Config(
        active_broker=active_broker,
        active_llm=os.getenv("ACTIVE_LLM", y.get("active_llm", "openai_compat")).lower(),
        monitor_interval=int(os.getenv("MONITOR_INTERVAL", y.get("monitor_interval", 60))),
        max_open_trades=int(os.getenv("MAX_OPEN_TRADES", y.get("max_open_trades", 3))),
        risk_pct_per_trade=float(os.getenv("RISK_PCT_PER_TRADE", y.get("risk_pct_per_trade", 1.5))),
        daily_loss_limit=float(os.getenv("DAILY_LOSS_LIMIT", y.get("daily_loss_limit", 5.0))),
        simulation_mode=_bool(os.getenv("SIMULATION_MODE"), default=y.get("simulation_mode", True)),
        trading_hours_ist=os.getenv("TRADING_HOURS_IST", y.get("trading_hours_ist", "09:15-15:15")),
        min_rr_ratio=float(os.getenv("MIN_RR_RATIO", y.get("min_rr_ratio", 3.0))),
        trading_symbol=trading_symbol,
        event_blackout_minutes=int(os.getenv("EVENT_BLACKOUT_MINUTES", y.get("event_blackout_minutes", 15))),
        scheduled_events=list(y.get("scheduled_events", [])),
        intraday_square_off=os.getenv("INTRADAY_SQUARE_OFF", y.get("intraday_square_off", "15:15")),
        min_signal_strength=float(os.getenv("MIN_SIGNAL_STRENGTH", y.get("min_signal_strength", 0.65))),
        entry_zone_tolerance_pct=float(os.getenv("ENTRY_ZONE_TOLERANCE_PCT", y.get("entry_zone_tolerance_pct", 0.15))),
        ai_review_cooldown_minutes=int(os.getenv("AI_REVIEW_COOLDOWN_MIN", y.get("ai_review_cooldown_minutes", 30))),
        ai_analysis_cooldown_minutes=int(os.getenv("AI_ANALYSIS_COOLDOWN_MIN", y.get("ai_analysis_cooldown_minutes", 60))),
        fno_margin_rate=float(os.getenv("FNO_MARGIN_RATE", y.get("fno_margin_rate", 0.12))),
        state_dir=Path(os.getenv("STATE_DIR", str(_ROOT / "state"))),
        dhan_access_token=os.getenv("DHAN_ACCESS_TOKEN", ""),
        groww_session_token=os.getenv("GROWW_SESSION_TOKEN", ""),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        anthropic_model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022"),
    )
