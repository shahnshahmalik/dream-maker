# Dream-Maker Trading Engine — Issues & Fixes Log

Audit date: 2026-06-16  
Audited by: Claude Code  
Log coverage: trade_log.jsonl + /tmp/dream-maker-engine.log + config.yaml + .env

---

## Issue 1 — App not running on June 15–16 (trading days missed)

**Severity:** Critical  
**Detected:** 2026-06-16  
**Evidence:** Last heartbeat timestamp `2026-06-12T13:19:37+00:00` (18:49 IST). Zero log entries for June 15 or June 16. PID 30067 from heartbeat is no longer alive.

**Root cause:** Process died or was killed after the June 12 after-close cycle. `stop_at_market_close: false` was set to keep the engine alive over the weekend, but it did not survive.

**Fix:**
- Restart the engine before market open.
- Run it as a managed process (e.g., `systemd`, `supervisor`, or a `nohup` wrapper with auto-restart) so it survives crashes and reboots.
- The `guardian.py` file in the project root appears to be a watchdog — use it.

---

## Issue 2 — Simulation mode ON, no real orders ever placed

**Severity:** Critical  
**Detected:** 2026-06-16  
**Evidence:**
- `config.yaml` line: `simulation_mode: true`
- `.env` line: `SIMULATION_MODE=true`
- Runtime log (2026-06-12 18:49): `broker=dhan ... simulation=True`

**Root cause:** Both the YAML config and .env have simulation mode enabled. The executor skips the real Dhan order call entirely when this flag is set. Every plan that reaches execution is silently paper-traded.

**Fix:**
```
# .env
SIMULATION_MODE=false

# config.yaml
simulation_mode: false
```
Only one of these needs to change (`.env` takes precedence), but both should be consistent.

---

## Issue 3 — Dhan access token expired (DH-901)

**Severity:** Critical  
**Detected:** 2026-06-12T13:49:23 IST  
**Evidence:** `/tmp/dream-maker-engine.log`:
```
get_funds failed (401: {"errorType":"Invalid_Authentication","errorCode":"DH-901",
"errorMessage":"Client ID or user generated access token is invalid or expired."})
get_positions failed: 401: ... same error
```

**Root cause:** The Dhan access token in `.env` (`DHAN_ACCESS_TOKEN`) had expired by June 12. All broker API calls — funds, positions, OHLCV charts, live quotes — were failing silently. The engine fell back to simulation defaults (₹1,00,000 fake balance) and continued running as if connected.

**Impact cascade:** Expired token → Dhan OHLCV fails → Yahoo fallback (yfinance not installed) → synthetic data → bad SL/TP → plan rejected. See Issues 4 and 5.

**Fix:**
- Generate a new access token from the Dhan developer portal.
- Update `DHAN_ACCESS_TOKEN` in `.env`.
- Dhan tokens typically expire daily; consider automating token refresh or using a long-lived token if available.

---

## Issue 4 — yfinance not installed, Yahoo fallback silently broken

**Severity:** High  
**Detected:** 2026-06-16  
**Evidence:**
```
$ python3 -c "import yfinance"
ModuleNotFoundError: No module named 'yfinance'
```
`providers/yahoo_finance.py` line 15: `yf = None` when import fails, causing `fetch_ohlcv()` to always return `[]`.

**Root cause:** The `yfinance` package is listed as an optional dependency but is not installed. When Dhan chart API fails (e.g., expired token), the code tries Yahoo Finance as a fallback and then falls through to synthetic fake candles.

**Fix:**
```bash
pip install yfinance
# or add to pyproject.toml dependencies:
# yfinance>=0.2.0
```

---

## Issue 5 — Synthetic OHLCV fallback produces invalid SL/TP

**Severity:** High  
**Detected:** 2026-05-31 (log timestamps 10:33–10:42 UTC)  
**Evidence:** `trade_log.jsonl` Step 5 projections:
```json
{"entry": 557.09, "sl": 769.93, "tp1": -81.44, "qty": 25, "margin": 500.0}
{"entry": 467.23, "sl": 645.75, "tp1": -68.31, "qty": 25, "margin": 500.0}
```
`tp1` is negative — immediately rejected by `validate_bracket`: *"SL and TP levels must be positive"*.

**Root cause:** When both Dhan and Yahoo Finance fail, `_synthetic_ohlcv()` in `providers/dhan.py` generates fake candles based on a hardcoded base price (24500 for NIFTY). The small synthetic values (557, 467) suggest the security ID for `NIFTY50IDX` resolved to a non-numeric fallback string, triggering the Yahoo path directly and returning garbage data. The `analyze_technical()` function ran on this data and produced a negative TP because the R:R math pushed `tp1` below zero.

**Fix:**
- Installing yfinance (Issue 4) removes the need for synthetic data in most cases.
- As a safeguard, add a pre-flight check in the pipeline:
  ```python
  if any(c.close <= 0 for c in htf) or len(htf) < 20:
      return PipelineResult(symbol, macro, None, fund.summary, None, "Bad OHLCV data — skipping")
  ```
- Also add a sanity check in `analyze_technical` to reject results where `tp1 <= 0`.

---

## Issue 6 — Option LTP fallback returns spot price, fails plausibility check

**Severity:** High  
**Detected:** 2026-06-02T09:01 UTC (14:31 IST)  
**Evidence:** `trade_log.jsonl` Step 5:
```json
{"entry": 26079.58, "sl": 28263.64, "tp1": 19527.43, "qty": 25, "margin": 9187.5}
```
No subsequent PLAN or ENTER log — plan was silently dropped after Step 5.

**Root cause:** After Step 5, `pipeline.py` fetches the real-time option LTP for the selected strike (e.g., `NIFTY26JUN24500PE`). With a degraded/expired token, `get_quote()` calls `_fallback_quote()` which returns `ltp = 24500.0` (hardcoded NIFTY spot). The plausibility check then fires:
```python
max_plausible = spot_for_pricing * 0.25  # = 6125
if option_ltp >= max_plausible:          # 24500 >= 6125 → True → plan dropped
```
This rejection is only written to the Python logger, **not to `trade_log.jsonl`**, making it invisible in the audit trail.

**Fix:**
- Fix the underlying token expiry (Issue 3) — a live LTP will be well below `max_plausible`.
- Add a `trade_logger.log()` call in `pipeline.py` at the implausibility rejection point so it appears in `trade_log.jsonl`.
- `_fallback_quote()` should return `ltp=0` instead of `ltp=24500` so it fails fast and visibly rather than producing a misleading number.

---

## Issue 7 — Entry zone tolerance 0.15% (100× too tight)

**Severity:** Medium  
**Detected:** 2026-06-16  
**Evidence:**
- `.env`: `ENTRY_ZONE_TOLERANCE_PCT=0.15`
- `config.yaml`: `entry_zone_tolerance_pct: 1.5`
- Pipeline: `tol = self.cfg.entry_zone_tolerance_pct / 100.0` → `tol = 0.0015`

**Root cause:** The `.env` value overrides `config.yaml`. `0.15 / 100 = 0.0015` = 0.15% tolerance. For a NIFTY option trading at ₹200, the entry zone is ₹199.70–₹200.30 — a 60 paise window on an instrument that moves ₹5–₹20 per tick. The `entry_watcher` would almost never trigger.

The intended value was `1.5` (1.5%), giving a ₹3 window at ₹200 premium — much more reasonable.

**Fix:**
```
# .env
ENTRY_ZONE_TOLERANCE_PCT=1.5
```

---

## Summary Table

| # | Issue | Severity | First seen | Status |
|---|-------|----------|------------|--------|
| 1 | App process died, not running June 15–16 | Critical | 2026-06-12 | Open |
| 2 | `SIMULATION_MODE=true` — no real orders ever | Critical | all runs | Open |
| 3 | Dhan access token expired (DH-901) | Critical | 2026-06-12 | Open |
| 4 | yfinance not installed, Yahoo fallback broken | High | all runs | Open |
| 5 | Synthetic OHLCV produces negative TP, plan rejected | High | 2026-05-31 | Open |
| 6 | Option LTP fallback = spot price, fails plausibility check | High | 2026-06-02 | Open |
| 7 | Entry zone tolerance 0.15% (should be 1.5%) | Medium | all runs | Open |

## Minimum changes to go live

```bash
# 1. Refresh Dhan token in .env
#    DHAN_ACCESS_TOKEN=<new token from Dhan portal>

# 2. Turn off simulation
#    SIMULATION_MODE=false

# 3. Fix tolerance
#    ENTRY_ZONE_TOLERANCE_PCT=1.5

# 4. Install yfinance
pip install yfinance

# 5. Restart with a watchdog
python guardian.py   # or nohup python main.py &
```
