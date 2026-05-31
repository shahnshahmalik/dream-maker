# Dream Maker

Autonomous AI trading agent for Indian markets. Supports Dhan (primary) and Groww (equity-only) via a broker provider pattern, with LLM-agnostic AI review on position divergence.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
# Edit .env with DHAN_ACCESS_TOKEN etc.

python main.py --scan-only --once   # analysis only, one cycle
python main.py --once               # simulation mode, one cycle
python main.py                      # continuous monitoring loop
```

## Configuration

- [`config.yaml`](config.yaml) — defaults (risk limits, intervals); **symbol comes from `.env`**
- [`.env`](.env.example) — secrets and provider overrides

| Variable | Description |
|----------|-------------|
| `TRADING_SYMBOL` | **Required.** Single F&O symbol only (e.g. `NIFTY50IDX`, `NIFTY25JUNFUT`, `BANKNIFTY`) |
| `ACTIVE_BROKER` | `dhan` only (F&O requires Dhan) |
| `ACTIVE_LLM` | `openai_compat`, `anthropic`, or `none` |
| `SIMULATION_MODE` | `true` uses Dhan paper orders |
| `DHAN_ACCESS_TOKEN` | Dhan API token |
| `GROWW_SESSION_TOKEN` | Groww session token (experimental) |

## Trade flow

1. **Scan** — technical + macro analysis; requires signal strength ≥ 0.65 and R:R ≥ **1:3**
2. **AI setup (once)** — refines entry/SL/TP markers; cooldown between AI analysis calls
3. **WAITING_ENTRY** — watches price until LTP enters the marker zone (no immediate orders)
4. **Bracketed entry** — entry only if SL + TP orders are confirmed (never naked)
5. **Monitor** — rule-based checks each loop; **trail SL+TP** when moving as expected; AI review only on divergence + cooldown
6. **Crash recovery** — resumes from `state/active_plans.json` and last `ORDER` in `trade_log.jsonl`
7. **Heartbeat** — `state/heartbeat.json` updated each loop (detect crashes)

## Configuration highlights

| Variable | Default | Purpose |
|----------|---------|---------|
| `TRADING_SYMBOL` | required | Single F&O symbol |
| `MIN_RR_RATIO` | 3.0 | Minimum risk:reward |
| `MIN_SIGNAL_STRENGTH` | 0.65 | Pause if setup is weak |
| `AI_REVIEW_COOLDOWN_MIN` | 30 | Min minutes between AI position reviews |
| `AI_ANALYSIS_COOLDOWN_MIN` | 60 | Min minutes between AI setup calls |
| `ENTRY_ZONE_TOLERANCE_PCT` | 0.15 | Entry marker band around target price |

## Architecture

- **Broker providers** — `providers/dhan.py`, `providers/groww.py`
- **LLM providers** — `llm/openai_compat.py`, `llm/anthropic.py`, rule-based fallback
- **Analysis pipeline** — 5-step macro → HTF/LTF → fundamental → projection
- **Audit log** — append-only `trade_log.jsonl` in [`audit/trade_logger.py`](audit/trade_logger.py)

## Safety

- Default is simulation mode (`SIMULATION_MODE=true`)
- Live trading requires `--live` and typing `LIVE` at the prompt
- Max 3 open trades, 1.5% risk per trade, 5% daily loss halt

**Not financial advice.** Systematic execution engine only.
