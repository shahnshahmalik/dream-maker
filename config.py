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


def _parse_csv(env_val: str | None, default: list[str]) -> list[str]:
    if env_val is None:
        return list(default)
    return [part.strip() for part in env_val.split(",") if part.strip()]


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
    trade_window_start_ist: str  # earliest new-entry time (e.g. "09:30")
    trade_window_end_ist: str    # latest new-entry / auto-square-off time (e.g. "15:15")
    market_holidays: list[str]
    market_closed_poll_interval: int
    stop_at_market_close: bool
    wait_for_market_open: bool
    min_rr_ratio: float
    auto_symbol_picker: bool
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
    trail_enabled: bool
    trail_activate_pct: float
    trail_sl_distance_pct: float
    trail_tp_reward_pct: float
    trail_breakeven_progress_pct: float
    trail_breakeven_buffer_pct: float
    scalp_enabled: bool
    scalp_min_rr_ratio: float
    scalp_min_signal_strength: float
    scalp_max_sl_pct: float
    scalp_min_confirmations: int
    active_strategy: str  # "all" or a specific strategy name e.g. "three_candle"
    scalp_trail_activate_pct: float
    scalp_trail_sl_distance_pct: float
    scalp_trail_breakeven_progress_pct: float

    balance_min_sell: float
    balance_sell_buffer: float

    entry_stale_pct: float
    entry_stale_cycles: int

    # ── Precision scalping (low-balance option buying) ──
    max_trades_per_day: int          # hard cap on total trades/day (1-4 for scalping)
    daily_profit_target_pct: float   # stop trading when daily P&L hits this % of capital
    daily_loss_limit_inr: float      # stop trading when daily loss exceeds this ₹ amount
    entry_quality_threshold: float   # minimum setup score to enter (0.65-0.75)
    entry_require_pullback: bool     # require price pullback to EMA before entry
    entry_require_volume_surge: bool # require volume > 1.2x average for entry
    day_gate_enabled: bool           # block trades on Range/Inside days, lock direction
    require_bb_confirmation: bool    # BB(20,2) breakout must confirm sweep direction
    require_orb_confirmation: bool   # 15-min ORB must be broken in sweep direction

    dhan_access_token: str
    dhan_client_id: str
    openai_api_key: str
    openai_base_url: str
    openai_model: str
    anthropic_api_key: str
    anthropic_model: str
    deepseek_api_key: str
    deepseek_base_url: str
    deepseek_model: str

    telegram_enabled: bool
    telegram_bot_token: str
    telegram_chat_id: str
    telegram_notify_trades: bool
    telegram_notify_ai: bool

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
        trade_window_start_ist=os.getenv("TRADE_WINDOW_START_IST", y.get("trade_window_start_ist", "09:30")),
        trade_window_end_ist=os.getenv("TRADE_WINDOW_END_IST", y.get("trade_window_end_ist", "15:15")),
        market_holidays=_parse_csv(os.getenv("MARKET_HOLIDAYS"), y.get("market_holidays", [])),
        market_closed_poll_interval=int(
            os.getenv("MARKET_CLOSED_POLL_INTERVAL", y.get("market_closed_poll_interval", 300))
        ),
        stop_at_market_close=_bool(
            os.getenv("STOP_AT_MARKET_CLOSE"), default=y.get("stop_at_market_close", False)
        ),
        wait_for_market_open=_bool(
            os.getenv("WAIT_FOR_MARKET_OPEN"), default=y.get("wait_for_market_open", True)
        ),
        min_rr_ratio=float(os.getenv("MIN_RR_RATIO", y.get("min_rr_ratio", 2.0))),
        auto_symbol_picker=_bool(os.getenv("AUTO_SYMBOL_PICKER"), default=False),
        trading_symbol=trading_symbol,
        event_blackout_minutes=int(os.getenv("EVENT_BLACKOUT_MINUTES", y.get("event_blackout_minutes", 15))),
        scheduled_events=list(y.get("scheduled_events", [])),
        intraday_square_off=os.getenv("INTRADAY_SQUARE_OFF", y.get("intraday_square_off", "15:15")),
        min_signal_strength=float(os.getenv("MIN_SIGNAL_STRENGTH", y.get("min_signal_strength", 0.65))),
        entry_zone_tolerance_pct=float(os.getenv("ENTRY_ZONE_TOLERANCE_PCT", y.get("entry_zone_tolerance_pct", 1.5))),
        ai_review_cooldown_minutes=int(os.getenv("AI_REVIEW_COOLDOWN_MIN", y.get("ai_review_cooldown_minutes", 30))),
        ai_analysis_cooldown_minutes=int(os.getenv("AI_ANALYSIS_COOLDOWN_MIN", y.get("ai_analysis_cooldown_minutes", 60))),
        fno_margin_rate=float(os.getenv("FNO_MARGIN_RATE", y.get("fno_margin_rate", 0.12))),
        state_dir=Path(os.getenv("STATE_DIR", str(_ROOT / "state"))),
        trail_enabled=_bool(os.getenv("TRAIL_ENABLED"), default=y.get("trail_enabled", True)),
        trail_activate_pct=float(os.getenv("TRAIL_ACTIVATE_PCT", y.get("trail_activate_pct", 0.5))),
        trail_sl_distance_pct=float(os.getenv("TRAIL_SL_DISTANCE_PCT", y.get("trail_sl_distance_pct", 0.35))),
        trail_tp_reward_pct=float(os.getenv("TRAIL_TP_REWARD_PCT", y.get("trail_tp_reward_pct", 1.0))),
        trail_breakeven_progress_pct=float(
            os.getenv("TRAIL_BREAKEVEN_PROGRESS_PCT", y.get("trail_breakeven_progress_pct", 20))
        ),
        trail_breakeven_buffer_pct=float(os.getenv("TRAIL_BREAKEVEN_BUFFER_PCT", y.get("trail_breakeven_buffer_pct", 0.05))),
        scalp_enabled=_bool(os.getenv("SCALP_ENABLED"), default=y.get("scalp_enabled", True)),
        scalp_min_rr_ratio=float(os.getenv("SCALP_MIN_RR_RATIO", y.get("scalp_min_rr_ratio", 1.2))),
        scalp_min_signal_strength=float(
            os.getenv("SCALP_MIN_SIGNAL_STRENGTH", y.get("scalp_min_signal_strength", 0.55))
        ),
        scalp_max_sl_pct=float(os.getenv("SCALP_MAX_SL_PCT", y.get("scalp_max_sl_pct", 0.35))),
        scalp_min_confirmations=int(os.getenv("SCALP_MIN_CONFIRMATIONS", y.get("scalp_min_confirmations", 1))),
        active_strategy=str(os.getenv("ACTIVE_STRATEGY", y.get("active_strategy", "all"))),
        scalp_trail_activate_pct=float(
            os.getenv("SCALP_TRAIL_ACTIVATE_PCT", y.get("scalp_trail_activate_pct", 0.25))
        ),
        scalp_trail_sl_distance_pct=float(
            os.getenv("SCALP_TRAIL_SL_DISTANCE_PCT", y.get("scalp_trail_sl_distance_pct", 0.2))
        ),
        scalp_trail_breakeven_progress_pct=float(
            os.getenv("SCALP_TRAIL_BREAKEVEN_PROGRESS_PCT", y.get("scalp_trail_breakeven_progress_pct", 15))
        ),
        dhan_access_token=os.getenv("DHAN_ACCESS_TOKEN", ""),
        dhan_client_id=os.getenv("DHAN_CLIENT_ID", ""),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        anthropic_model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022"),
        deepseek_api_key=os.getenv("DEEPSEEK_API_KEY", ""),
        deepseek_base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        deepseek_model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        telegram_enabled=_bool(
            os.getenv("TELEGRAM_ENABLED"),
            default=bool(os.getenv("TELEGRAM_BOT_TOKEN", "").strip() and os.getenv("TELEGRAM_CHAT_ID", "").strip()),
        ),
        telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
        telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID", ""),
        telegram_notify_trades=_bool(
            os.getenv("TELEGRAM_NOTIFY_TRADES"), default=y.get("telegram_notify_trades", True)
        ),
        telegram_notify_ai=_bool(
            os.getenv("TELEGRAM_NOTIFY_AI"), default=y.get("telegram_notify_ai", True)
        ),
        trade_log_path=Path(os.getenv("TRADE_LOG_PATH", str(_ROOT / "trade_log.jsonl"))),
        balance_min_sell=float(os.getenv("BALANCE_MIN_SELL", y.get("balance_min_sell", 100_000))),
        balance_sell_buffer=float(os.getenv("BALANCE_SELL_BUFFER", y.get("balance_sell_buffer", 1.5))),
        entry_stale_pct=float(os.getenv("ENTRY_STALE_PCT", y.get("entry_stale_pct", 5.0))),
        entry_stale_cycles=int(os.getenv("ENTRY_STALE_CYCLES", y.get("entry_stale_cycles", 8))),
        # ── Precision scalping ──
        max_trades_per_day=int(os.getenv("MAX_TRADES_PER_DAY", y.get("max_trades_per_day", 8))),
        daily_profit_target_pct=float(os.getenv("DAILY_PROFIT_TARGET_PCT", y.get("daily_profit_target_pct", 3.0))),
        daily_loss_limit_inr=float(os.getenv("DAILY_LOSS_LIMIT_INR", y.get("daily_loss_limit_inr", 1000))),
        entry_quality_threshold=float(os.getenv("ENTRY_QUALITY_THRESHOLD", y.get("entry_quality_threshold", 0.70))),
        entry_require_pullback=_bool(os.getenv("ENTRY_REQUIRE_PULLBACK"), default=y.get("entry_require_pullback", True)),
        entry_require_volume_surge=_bool(os.getenv("ENTRY_REQUIRE_VOLUME_SURGE"), default=y.get("entry_require_volume_surge", True)),
        day_gate_enabled=_bool(os.getenv("DAY_GATE_ENABLED"), default=y.get("day_gate_enabled", True)),
        require_bb_confirmation=_bool(os.getenv("REQUIRE_BB_CONFIRMATION"), default=y.get("require_bb_confirmation", False)),
        require_orb_confirmation=_bool(os.getenv("REQUIRE_ORB_CONFIRMATION"), default=y.get("require_orb_confirmation", False)),
    )
