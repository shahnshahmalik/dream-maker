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
python main.py                      # 24x7 daemon — trades 09:15–15:15 IST, pauses when closed
```

### Docker

```bash
cp .env.example .env   # add API keys
docker compose up -d --build
docker compose logs -f dream-maker
```

One-off cycle:

```bash
docker compose run --rm dream-maker python main.py --once
```

State and audit log persist in the `dream-maker-data` volume. Container timezone is `Asia/Kolkata`.

**Single instance:** the app writes `state/dream-maker.lock` on startup. A second start exits immediately instead of duplicating trades.

**External watchdog (cron/systemd):** use `scripts/ensure_running.sh` — it uses `flock` and checks the lock file before starting. Do **not** run both Docker `restart: unless-stopped` and a separate restart script without the lock checks.

## Configuration

- [`config.yaml`](config.yaml) — defaults (risk limits, intervals); **symbol comes from `.env`**
- [`.env`](.env.example) — secrets and provider overrides

| Variable | Description |
|----------|-------------|
| `TRADING_SYMBOL` | **Required.** Single F&O symbol only (e.g. `NIFTY50IDX`, `NIFTY25JUNFUT`, `BANKNIFTY`) |
| `ACTIVE_BROKER` | `dhan` only (F&O requires Dhan) |
| `ACTIVE_LLM` | `openai_compat`, `deepseek`, `anthropic`, or `none` |
| `DEEPSEEK_API_KEY` | DeepSeek API key (when `ACTIVE_LLM=deepseek`) |
| `SIMULATION_MODE` | `true` uses Dhan paper orders |
| `DHAN_ACCESS_TOKEN` | Dhan API token |
| `DHAN_CLIENT_ID` | Dhan client ID (required for LTP/market feed APIs) |
| `GROWW_SESSION_TOKEN` | Groww session token (experimental) |

## Trade flow

1. **Scan** — swing (HTF+LTF, R:R ≥ 3) or **momentum scalp** (LTF burst + ≥2 confirmations, R:R ≥ 1.2)
2. **AI setup (once)** — refines entry/SL/TP markers; cooldown between AI analysis calls
3. **WAITING_ENTRY** — LTP in marker zone; scalps also require live 5m momentum confirmation
4. **Bracketed entry** — entry only if SL + TP orders are confirmed (never naked); invalid brackets blocked pre-order
5. **Monitor** — rule-based checks each loop; **trail SL+TP** when moving as expected (tighter trail for scalps); AI review only on divergence + cooldown
6. **Crash recovery** — resumes from `state/active_plans.json` and last `ORDER` in `trade_log.jsonl`
7. **Heartbeat** — `state/heartbeat.json` updated each loop (detect crashes)

## Configuration highlights

| Variable | Default | Purpose |
|----------|---------|---------|
| `TRADING_SYMBOL` | required | Single F&O symbol |
| `MIN_RR_RATIO` | 3.0 | Minimum risk:reward (swing setups) |
| `MIN_SIGNAL_STRENGTH` | 0.65 | Pause if swing setup is weak |
| `SCALP_ENABLED` | true | Allow LTF momentum scalps when swing gates fail |
| `SCALP_MIN_RR_RATIO` | 1.2 | Minimum R:R for momentum scalps |
| `SCALP_MIN_SIGNAL_STRENGTH` | 0.55 | Signal floor for scalps |
| `SCALP_MIN_CONFIRMATIONS` | 2 | Required momentum signals (EMA, volume, breakout, etc.) |
| `SCALP_MAX_SL_PCT` | 0.35 | Max stop distance (% of price) for scalps |
| `TRAIL_ENABLED` | true | Trail SL+TP when trade moves favorably |
| `SCALP_TRAIL_*` | see `.env.example` | Tighter trailing for scalp trades |
| `AI_REVIEW_COOLDOWN_MIN` | 30 | Min minutes between AI position reviews |
| `AI_ANALYSIS_COOLDOWN_MIN` | 60 | Min minutes between AI setup calls |
| `ENTRY_ZONE_TOLERANCE_PCT` | 0.15 | Entry marker band around target price |
| `TRADING_HOURS_IST` | 09:15-15:15 | NSE session window (IST) |
| `MARKET_CLOSED_POLL_INTERVAL` | 300 | Heartbeat interval while trading is paused (seconds) |
| `MARKET_HOLIDAYS` | (empty) | Comma-separated `YYYY-MM-DD` NSE holidays |
| `STOP_AT_MARKET_CLOSE` | false | Exit the process at session end (optional) |
| `WAIT_FOR_MARKET_OPEN` | true | Wait for open when started early or on weekends |

### 24x7 schedule (default)

The process **keeps running** around the clock. **Trading** only happens during NSE hours (default 09:15–15:15 IST):

1. **Closed** (weekend, holiday, before/after session) — heartbeat only, no scans or orders
2. **Open** — full scan → entry → monitor loop
3. **At session end** — square-off, cancel pending plans, then pause until next open

Set `STOP_AT_MARKET_CLOSE=true` only if you want the process to exit after 15:15 instead of idling overnight.

### Telegram alerts

Add to `.env`:

```env
TELEGRAM_BOT_TOKEN=123456:ABC...   # from @BotFather
TELEGRAM_CHAT_ID=your_chat_id      # from @userinfobot or getUpdates
TELEGRAM_ENABLED=true              # auto-on when token + chat id are set
TELEGRAM_NOTIFY_TRADES=true        # entries, closes, trail updates
TELEGRAM_NOTIFY_AI=true            # AI levels, pauses, position insights
```

You will receive messages for bracket entries, closes, trail updates, AI level setup, and AI position reviews.

## Architecture

- **Broker providers** — `providers/dhan.py`, `providers/groww.py`
- **LLM providers** — `llm/openai_compat.py`, `llm/deepseek.py`, `llm/anthropic.py`, rule-based fallback
- **Analysis pipeline** — 5-step macro → HTF/LTF → fundamental → projection
- **Audit log** — append-only `trade_log.jsonl` in [`audit/trade_logger.py`](audit/trade_logger.py)

## Safety

- Default is simulation mode (`SIMULATION_MODE=true`)
- Live trading requires `--live` and typing `LIVE` at the prompt
- Max 3 open trades, 1.5% risk per trade, 5% daily loss halt

**Not financial advice.** Systematic execution engine only.
