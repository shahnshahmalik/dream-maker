"""
Non-Expiry Day Scalper v1 — 5-minute precision.

Timeframe choice rationale:
  5-min: on non-expiry days, NIFTY moves develop over multiple candles.
         1m is too noisy without expiry-day gamma. 5m filters chop and
         gives cleaner ORB breaks and EMA signals.
  15-min for HTF trend context (true dual-timeframe, fetched separately).

Strategy (non-expiry days):
  - ORB: first 30-min (9:15–9:44) defines the range. Clean break = momentum trade.
  - Momentum: EMA9/EMA21 cross + HH/HL or LH/LL structure.
  - VWAP: rejection bounce with volume.
  - Any 2 of these signals required (higher bar than expiry day).

  Volume: enforced at 1.5× 10-bar average — not just logged.
  No trades in dead zone 12:00–13:00 IST.
  Theta kill at 15:00 IST (non-expiry premium decays slower).

Risk:
  TP = +30%, SL = -15% (2:1 R:R)
  Max 3 trades. ATM option on nearest upcoming (non-today) weekly expiry.
"""

from __future__ import annotations

import os, sys, time, logging, sqlite3, requests
from datetime import datetime, date
from zoneinfo import ZoneInfo
from dotenv import load_dotenv

sys.path.insert(0, "/home/ubuntu/projects/dream-maker")
load_dotenv("/home/ubuntu/projects/dream-maker/.env", override=True)

from utils.market_data import get_india_vix, vix_allows_entry, get_option_chain_context, max_pain_bias


def _headers() -> dict:
    """Build headers fresh each call — reads directly from .env file."""
    token, cid = "", ""
    try:
        with open("/home/ubuntu/projects/dream-maker/.env") as f:
            for line in f:
                line = line.strip()
                if line.startswith("DHAN_ACCESS_TOKEN="):
                    token = line.split("=", 1)[1].strip()
                elif line.startswith("DHAN_CLIENT_ID="):
                    cid = line.split("=", 1)[1].strip()
    except Exception:
        pass
    return {"access-token": token, "client-id": cid, "Content-Type": "application/json"}


IST     = ZoneInfo("Asia/Kolkata")
DB_PATH = "/home/ubuntu/projects/dream-maker/data/scrip_master.db"

TP_PCT       = 0.30
SL_PCT       = 0.15
MAX_TRADES         = 3
EXCEPTIONAL_SCORE  = 6  # bull or bear score ≥ this → allow 4th trade
DAILY_TARGET       = 2000      # ₹ target per day
CAPITAL_TARGET     = 1000      # ₹ target per trade — exit when unrealised PnL ≥ this
MAX_LOTS     = 1          # maximum lots per trade (1 lot = lot_size contracts)
THETA_KILL   = (15, 0)    # 15:00 IST — non-expiry, can hold longer than expiry day
DEAD_START   = (12, 0)
DEAD_END     = (13, 0)
ORB_END      = (9, 45)    # ORB window: 9:15–9:44 (6 x 5m candles), entries start 9:45
LOOP_SECS    = 60         # aligned to 5m candle rhythm
API_DELAY    = 2.0
VOL_MULT     = 1.5        # enforced entry gate — not just logged
MIN_SIGNAL   = 2          # need 2 confirming signals (vs 1 on expiry)
LOOKBACK_5M  = 30         # 2.5 hours of 5m history
LOOKBACK_15M = 20         # ~5 hours of 15m history for HTF context

TRAIL_ACTIVATE_PCT      = 0.15   # start trailing once +15% gained
TRAIL_BREAKEVEN_BUFFER  = 8.0    # SL locks this many rupees above entry
HIGH_MOMENTUM_THRESHOLD = 3      # skip capital-target exit when score >= this
MOMENTUM_REFRESH_TICKS  = 3      # refresh momentum every N monitor ticks

log = logging.getLogger("non_expiry_scalper")
log.setLevel(logging.INFO)

class _ISTFormatter(logging.Formatter):
    """Emit log timestamps in IST instead of the system (UTC) clock."""
    _tz = ZoneInfo("Asia/Kolkata")

    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        ct = datetime.fromtimestamp(record.created, tz=self._tz)
        return ct.strftime(datefmt or "%H:%M:%S")

fmt = _ISTFormatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")
for _h in [
    logging.FileHandler("/home/ubuntu/projects/dream-maker/logs/non_expiry_scalper.log"),
    logging.StreamHandler(),
]:
    _h.setFormatter(fmt)
    if not any(type(x) == type(_h) for x in log.handlers):
        log.addHandler(_h)

state: dict = {
    "trades": 0, "active": False,
    "symbol": None, "sid": None,
    "entry": None, "tp": None, "sl": None, "qty": None,
    "peak_ltp": None,
    "capital_profit_target": None,
    "last_momentum_score": 0,
    "monitor_tick": 0,
    "or_high": None, "or_low": None, "or_set": False,
    "last_dir": None, "daily_pnl": 0.0,
    "cooldown_until": 0,
}


# ── API ────────────────────────────────────────────────────────────────────────

def get_spot() -> float:
    try:
        r = requests.post("https://api.dhan.co/v2/marketfeed/ltp", headers=_headers(),
            json={"IDX_I": [13]}, timeout=6)
        return r.json().get("data", {}).get("IDX_I", {}).get("13", {}).get("last_price", 0.0)
    except Exception:
        return 0.0


def get_candles(interval: str = "5", limit: int = 30) -> list[dict]:
    """Fetch intraday NIFTY candles. interval: '1', '5', '15'."""
    try:
        today = date.today().strftime("%Y-%m-%d")
        r = requests.post("https://api.dhan.co/v2/charts/intraday", headers=_headers(),
            json={"securityId": "13", "exchangeSegment": "IDX_I", "instrument": "INDEX",
                  "interval": interval, "oi": False, "fromDate": today, "toDate": today},
            timeout=10)
        d = r.json()
        out = []
        for i in range(len(d.get("timestamp", []))):
            ist_s = int(d["timestamp"][i]) + 19800
            hh = (ist_s % 86400) // 3600
            mm = (ist_s % 3600) // 60
            out.append({"hh": hh, "mm": mm,
                "time": f"{hh:02d}:{mm:02d}",
                "open": d["open"][i], "high": d["high"][i],
                "low": d["low"][i], "close": d["close"][i], "volume": d["volume"][i]})
        return out[-limit:] if len(out) > limit else out
    except Exception:
        return []


def get_option_ltp(sid: int, retries: int = 2) -> float:
    for attempt in range(retries):
        try:
            r = requests.post("https://api.dhan.co/v2/marketfeed/ltp", headers=_headers(),
                json={"NSE_FNO": [sid]}, timeout=8)
            if r.status_code == 429:
                wait = 6 * (attempt + 1)
                log.warning("Rate limited — sleeping %ds (attempt %d)", wait, attempt + 1)
                time.sleep(wait)
                continue
            seg = r.json().get("data", {}).get("NSE_FNO", {})
            price = float(seg.get(str(sid), {}).get("last_price", 0.0))
            if price > 0:
                return price
            log.warning("LTP=0 for sid=%d (attempt %d) — raw: %s", sid, attempt + 1, str(seg)[:80])
            time.sleep(3)
        except Exception as e:
            log.warning("LTP exception attempt %d: %s", attempt + 1, e)
            time.sleep(3)
    return 0.0


def get_balance() -> float:
    try:
        r = requests.get("https://api.dhan.co/v2/fundlimit", headers=_headers(), timeout=6)
        return float(r.json().get("availabelBalance", 0))
    except Exception:
        return 0.0


def resolve_option(spot: float, opt_type: str) -> tuple[str, int, int] | None:
    """Resolve ATM option on the nearest upcoming weekly expiry (strictly after today)."""
    try:
        atm = round(spot / 50) * 50
        today = date.today().strftime("%Y-%m-%d")
        conn = sqlite3.connect(DB_PATH)
        cur  = conn.cursor()
        # Use > today (not >=) — on a non-expiry day we never want same-day expiry
        cur.execute("""
            SELECT trading_symbol, security_id, lot_size FROM scrip_master
            WHERE symbol_name='NIFTY' AND option_type=?
              AND date(expiry_date) > date(?)
              AND strike_price BETWEEN ? AND ?
            ORDER BY date(expiry_date) ASC, ABS(strike_price - ?) ASC LIMIT 1
        """, (opt_type, today, atm - 100, atm + 100, atm))
        row = cur.fetchone()
        conn.close()
        return (row[0], int(row[1]), int(row[2])) if row else None
    except Exception:
        return None


def place_order(sid: int, qty: int, side: str) -> str | None:
    try:
        cid = os.getenv("DHAN_CLIENT_ID", "").strip() or _headers()["client-id"]
        r = requests.post("https://api.dhan.co/v2/orders", headers=_headers(), json={
            "dhanClientId": cid, "transactionType": side,
            "exchangeSegment": "NSE_FNO", "productType": "INTRADAY",
            "orderType": "MARKET", "validity": "DAY",
            "securityId": str(sid), "quantity": qty,
            "price": 0, "triggerPrice": 0, "disclosedQuantity": 0,
            "afterMarketOrder": False, "boProfitValue": 0, "boStopLossValue": 0,
        }, timeout=10)
        if r.status_code == 200:
            oid = r.json().get("orderId")
            log.info("ORDER %s sid=%d qty=%d → orderId=%s", side, sid, qty, oid)
            return oid
        log.error("ORDER FAILED %d: %s", r.status_code, r.text[:150])
        return None
    except Exception as e:
        log.error("ORDER EXCEPTION: %s", e)
        return None


# ── Indicators ─────────────────────────────────────────────────────────────────

def ema(values: list[float], p: int) -> list[float]:
    if len(values) < p:
        return [0.0] * len(values)
    k = 2 / (p + 1)
    e = [sum(values[:p]) / p]
    for v in values[p:]:
        e.append(v * k + e[-1] * (1 - k))
    return [e[0]] * (len(values) - len(e)) + e


def vwap(candles: list[dict]) -> float:
    tv = sum(((c["high"] + c["low"] + c["close"]) / 3) * c["volume"] for c in candles)
    v  = sum(c["volume"] for c in candles)
    return tv / v if v else 0.0


# ── Time helpers ───────────────────────────────────────────────────────────────

def ist_now() -> datetime:
    return datetime.now(IST)

def hm() -> tuple[int, int]:
    n = ist_now()
    return n.hour, n.minute

def in_dead_zone() -> bool:
    return DEAD_START <= hm() < DEAD_END

def past_theta_kill() -> bool:
    return hm() >= THETA_KILL

def past_market_close() -> bool:
    return hm() >= (15, 30)

def orb_window_closed() -> bool:
    """ORB window is 9:15–9:44. Entries only allowed from 9:45 onward."""
    return hm() >= ORB_END

def in_trade_window() -> bool:
    return orb_window_closed() and not past_theta_kill() and not in_dead_zone()


# ── Signal engine ──────────────────────────────────────────────────────────────

def score_signal(
    candles_5m: list[dict],
    candles_15m: list[dict],
    spot: float,
) -> tuple[int, int, str]:
    """Returns (bull_score, bear_score, reason_string).

    Volume is an entry gate on non-expiry — signal scoring is skipped entirely
    if the current bar does not meet the volume threshold.
    """
    if len(candles_5m) < 12:
        return 0, 0, "too_few_5m"

    closes  = [c["close"]  for c in candles_5m]
    highs   = [c["high"]   for c in candles_5m]
    lows    = [c["low"]    for c in candles_5m]
    opens   = [c["open"]   for c in candles_5m]
    volumes = [c["volume"] for c in candles_5m]

    avg_vol = sum(volumes[-10:]) / min(10, len(volumes))
    has_vol = volumes[-1] >= avg_vol * VOL_MULT

    if not has_vol:
        reason = f"low_vol={volumes[-1]:.0f}<{avg_vol * VOL_MULT:.0f}"
        log.info("VOLUME GATE: %s — no entry", reason)
        return 0, 0, reason

    e9  = ema(closes, 9)
    e21 = ema(closes, 21)
    vw  = vwap(candles_5m)

    last_c = closes[-1]
    last_o = opens[-1]
    last_h = highs[-1]
    last_l = lows[-1]
    prev_c = closes[-2]
    bull_body = last_c > last_o
    bear_body = last_c < last_o

    bull, bear = 0, 0
    reasons: list[str] = []

    # ── 1. EMA9/EMA21 cross or alignment on 5m ───────────────────────────────
    if e9[-1] > e21[-1] and e9[-2] <= e21[-2]:
        bull += 2
        reasons.append("EMA_cross_bull")
    elif e9[-1] > e21[-1] and last_c > e9[-1]:
        bull += 1
        reasons.append("EMA_align_bull")

    if e9[-1] < e21[-1] and e9[-2] >= e21[-2]:
        bear += 2
        reasons.append("EMA_cross_bear")
    elif e9[-1] < e21[-1] and last_c < e9[-1]:
        bear += 1
        reasons.append("EMA_align_bear")

    # ── 2. ORB breakout ───────────────────────────────────────────────────────
    if state["or_set"]:
        if last_c > state["or_high"] and bull_body and prev_c <= state["or_high"]:
            bull += 2
            reasons.append(f"ORB_bull_break>{state['or_high']:.0f}")
        elif last_c > state["or_high"] and bull_body:
            bull += 1
            reasons.append("ORB_above")

        if last_c < state["or_low"] and bear_body and prev_c >= state["or_low"]:
            bear += 2
            reasons.append(f"ORB_bear_break<{state['or_low']:.0f}")
        elif last_c < state["or_low"] and bear_body:
            bear += 1
            reasons.append("ORB_below")

    # ── 3. VWAP rejection ─────────────────────────────────────────────────────
    if vw > 0:
        vwap_touch_bull = abs(last_l - vw) / vw < 0.0015 and last_c > vw and bull_body
        vwap_touch_bear = abs(last_h - vw) / vw < 0.0015 and last_c < vw and bear_body
        if vwap_touch_bull:
            bull += 1
            reasons.append(f"VWAP_reject_bull@{vw:.0f}")
        if vwap_touch_bear:
            bear += 1
            reasons.append(f"VWAP_reject_bear@{vw:.0f}")

    # ── 4. HH/HL or LH/LL momentum structure on last 5 bars ──────────────────
    last5h = highs[-5:]
    last5l = lows[-5:]
    hh_hl = (all(last5h[i] >= last5h[i-1] for i in range(1, 5))
              and all(last5l[i] >= last5l[i-1] for i in range(1, 5)))
    lh_ll = (all(last5h[i] <= last5h[i-1] for i in range(1, 5))
              and all(last5l[i] <= last5l[i-1] for i in range(1, 5)))

    if hh_hl:
        bull += 1
        reasons.append("HH_HL_struct")
    if lh_ll:
        bear += 1
        reasons.append("LH_LL_struct")

    # ── 5. HTF 15m bias (true 15m candles, fetched separately) ───────────────
    if len(candles_15m) >= 5:
        htf_closes = [c["close"] for c in candles_15m]
        htf_e9  = ema(htf_closes, min(9, len(htf_closes)))
        htf_e21 = ema(htf_closes, min(21, len(htf_closes)))
        if htf_e9[-1] > htf_e21[-1]:
            bull += 1
            reasons.append("HTF_bull")
        elif htf_e9[-1] < htf_e21[-1]:
            bear += 1
            reasons.append("HTF_bear")

    log.info("SCORE → bull=%d bear=%d | %s | vol=%.0f/thr=%.0f✓",
             bull, bear, " | ".join(reasons) or "no_signals",
             volumes[-1], avg_vol * VOL_MULT)
    return bull, bear, " | ".join(reasons)


def restore_state_from_broker() -> None:
    """On startup, check open NIFTY positions and restore state so TP/SL monitoring continues."""
    try:
        r = requests.get("https://api.dhan.co/v2/positions", headers=_headers(), timeout=10)
        positions = r.json()
        if not isinstance(positions, list):
            return
        for p in positions:
            if p.get("positionType") != "LONG" or p.get("netQty", 0) <= 0:
                continue
            sym = p.get("tradingSymbol", "")
            if "NIFTY" not in sym:
                continue
            sid   = int(p["securityId"])
            qty   = int(p["netQty"])
            entry = float(p["buyAvg"])
            state.update({
                "active": True, "symbol": sym, "sid": sid,
                "entry": entry,
                "tp": round(entry * (1 + TP_PCT), 2),
                "sl": round(entry * (1 - SL_PCT), 2),
                "qty": qty, "trades": 1,
                "peak_ltp": entry,
                "capital_profit_target": float(CAPITAL_TARGET),
                "last_momentum_score": 0,
                "monitor_tick": 0,
            })
            log.info("RESTORED position %s qty=%d entry=%.2f TP=%.2f SL=%.2f",
                     sym, qty, entry, state["tp"], state["sl"])
            return  # only restore one — multiple open positions signal a bigger problem
    except Exception as e:
        log.warning("Could not restore state from broker: %s", e)


# ── Main loop ──────────────────────────────────────────────────────────────────

def run() -> None:
    # NIFTY 50 weekly expiry is Tuesday (weekday=1) since Sep 1, 2025.
    # Non-expiry scalper must NOT run on Tuesdays — that's expiry scalper's day.
    today = ist_now()
    if today.weekday() == 1:
        log.info("Today is Tuesday (expiry day). Non-expiry scalper does not run on expiry day. Exiting.")
        return

    log.info("=" * 60)
    log.info("NON-EXPIRY SCALPER v1 — 5m entries | TP=+%.0f%% SL=-%.0f%% | MaxTrades=%d (+1 exceptional ≥%d)",
             TP_PCT * 100, SL_PCT * 100, MAX_TRADES, EXCEPTIONAL_SCORE)
    log.info("Target ₹%d/day | Capital target ₹%d/trade",
             DAILY_TARGET, CAPITAL_TARGET)
    log.info("Theta kill %02d:%02d | Dead zone %02d:%02d–%02d:%02d | ORB ends %02d:%02d",
             *THETA_KILL, *DEAD_START, *DEAD_END, *ORB_END)
    log.info("=" * 60)
    restore_state_from_broker()

    while True:
        try:
            now = ist_now()

            if past_market_close():
                log.info("Market closed. Done.")
                break

            # ── Theta kill — force exit any open position ──────────────────────
            if past_theta_kill():
                if state["active"]:
                    ltp = get_option_ltp(state["sid"])
                    log.info("THETA KILL %s — force exit @ ₹%.2f", now.strftime("%H:%M"), ltp)
                    oid = place_order(state["sid"], state["qty"], "SELL")
                    if oid:
                        pnl = (ltp - state["entry"]) * state["qty"]
                        state["daily_pnl"] += pnl
                        log.info("THETA EXIT — PnL=₹%.0f | daily=₹%.0f", pnl, state["daily_pnl"])
                        state["active"] = False
                time.sleep(60)
                continue

            log.info("── %s | trade=%d/%d | active=%s | pnl=₹%.0f ──",
                     now.strftime("%H:%M:%S"), state["trades"], MAX_TRADES,
                     state["active"], state["daily_pnl"])

            # ── Monitor active trade ───────────────────────────────────────────
            if state["active"]:
                ltp = get_option_ltp(state["sid"])
                if ltp <= 0:
                    log.warning("LTP=0 after retries — will retry next tick")
                    time.sleep(LOOP_SECS)
                    continue

                state["monitor_tick"] += 1

                if state["monitor_tick"] % MOMENTUM_REFRESH_TICKS == 0:
                    candles_m   = get_candles("5",  LOOKBACK_5M)
                    candles_htf = get_candles("15", LOOKBACK_15M)
                    if len(candles_m) >= 12:
                        bull_m, bear_m, _ = score_signal(candles_m, candles_htf, ltp)
                        state["last_momentum_score"] = max(bull_m, bear_m)
                        log.info("MOMENTUM refresh score=%d (bull=%d bear=%d)",
                                 state["last_momentum_score"], bull_m, bear_m)

                if state["peak_ltp"] is None or ltp > state["peak_ltp"]:
                    state["peak_ltp"] = ltp

                peak_gain_pct = (state["peak_ltp"] - state["entry"]) / state["entry"]
                if peak_gain_pct >= TRAIL_ACTIVATE_PCT:
                    trail_sl = round(state["entry"] + TRAIL_BREAKEVEN_BUFFER, 2)
                    if trail_sl > state["sl"]:
                        log.info("TRAIL SL raised %.2f → %.2f (breakeven+₹%.0f, peak=%.2f +%.1f%%)",
                                 state["sl"], trail_sl, TRAIL_BREAKEVEN_BUFFER,
                                 state["peak_ltp"], peak_gain_pct * 100)
                        state["sl"] = trail_sl

                pnl_pct = (ltp - state["entry"]) / state["entry"]
                pnl_rs  = (ltp - state["entry"]) * state["qty"]
                log.info("MONITOR %s LTP=%.2f PnL=%.1f%% (₹%.0f) TP=%.2f SL=%.2f peak=%.2f",
                         state["symbol"], ltp, pnl_pct * 100, pnl_rs,
                         state["tp"], state["sl"], state["peak_ltp"])

                hit_tp = ltp >= state["tp"]
                hit_sl = ltp <= state["sl"]

                hit_capital_target = (
                    state["capital_profit_target"] is not None
                    and pnl_rs >= state["capital_profit_target"]
                )
                high_momentum = state["last_momentum_score"] >= HIGH_MOMENTUM_THRESHOLD

                if hit_capital_target and not high_momentum and not hit_tp:
                    log.info("CAPITAL TARGET ₹%.0f reached (score=%d < %d) — booking @ ₹%.2f",
                             state["capital_profit_target"], state["last_momentum_score"],
                             HIGH_MOMENTUM_THRESHOLD, ltp)
                    oid = place_order(state["sid"], state["qty"], "SELL")
                    if oid:
                        state["daily_pnl"] += pnl_rs
                        state["active"]     = False
                        state["cooldown_until"] = time.time() + 180
                        log.info("CAPITAL EXIT entry=%.2f exit=%.2f PnL=₹%.0f | daily=₹%.0f",
                                 state["entry"], ltp, pnl_rs, state["daily_pnl"])
                    time.sleep(LOOP_SECS)
                    continue

                if hit_capital_target and high_momentum:
                    log.info("CAPITAL TARGET reached but momentum=%d — letting it run",
                             state["last_momentum_score"])

                if hit_tp or hit_sl:
                    tag = "TP ✅" if hit_tp else "SL ❌"
                    log.info("EXIT %s @ ₹%.2f", tag, ltp)
                    oid = place_order(state["sid"], state["qty"], "SELL")
                    if oid:
                        pnl = (ltp - state["entry"]) * state["qty"]
                        state["daily_pnl"] += pnl
                        state["active"]     = False
                        state["cooldown_until"] = time.time() + 180
                        log.info("CLOSED entry=%.2f exit=%.2f PnL=₹%.0f | daily=₹%.0f",
                                 state["entry"], ltp, pnl, state["daily_pnl"])
                time.sleep(LOOP_SECS)
                continue

            # ── Fetch data (only when hunting for new entries) ─────────────────
            candles_5m  = get_candles("5",  LOOKBACK_5M)
            time.sleep(API_DELAY)
            candles_15m = get_candles("15", LOOKBACK_15M)
            time.sleep(API_DELAY)
            spot = get_spot()

            if spot <= 0 or len(candles_5m) < 5:
                log.warning("Bad data spot=%.2f 5m=%d", spot, len(candles_5m))
                time.sleep(LOOP_SECS)
                continue

            # ── Build ORB from 9:15–9:44 using 5m candles ────────────────────
            if not state["or_set"]:
                or_c = [c for c in candles_5m if c["hh"] == 9 and 15 <= c["mm"] <= 44]
                if len(or_c) >= 5:
                    state["or_high"] = max(c["high"] for c in or_c)
                    state["or_low"]  = min(c["low"]  for c in or_c)
                    state["or_set"]  = True
                    log.info("ORB locked: H=%.2f L=%.2f range=%.2f pts",
                             state["or_high"], state["or_low"],
                             state["or_high"] - state["or_low"])

            last = candles_5m[-1]
            log.info("Spot=%.2f | 5m %s C=%.2f | 5m_bars=%d 15m_bars=%d",
                     spot, last["time"], last["close"], len(candles_5m), len(candles_15m))

            if in_dead_zone():
                log.info("DEAD ZONE — no trades")
                time.sleep(60)
                continue

            if not in_trade_window():
                log.info("Outside window (ORB building or pre-market)")
                time.sleep(LOOP_SECS)
                continue

            if state["trades"] >= MAX_TRADES:
                if state["trades"] >= MAX_TRADES + 1:
                    log.info("Max trades hit (4/4)")
                    time.sleep(120)
                    continue
                # trades == 3: allow flow-through — gate on exceptional score below

            if time.time() < state["cooldown_until"]:
                log.info("Cooldown %.0fs remaining", state["cooldown_until"] - time.time())
                time.sleep(LOOP_SECS)
                continue

            # ── Signal ────────────────────────────────────────────────────────
            bull, bear, reason = score_signal(candles_5m, candles_15m, spot)

            direction = None
            if bull >= MIN_SIGNAL and bull > bear:
                direction = "CE"
            elif bear >= MIN_SIGNAL and bear > bull:
                direction = "PE"

            if direction is None:
                time.sleep(LOOP_SECS)
                continue

            log.info("SIGNAL %s — %s", direction, reason)

            # ── Exceptional 4th-trade gate: allow only on very strong signals ─
            if state["trades"] == MAX_TRADES:
                max_score = max(bull, bear)
                if max_score < EXCEPTIONAL_SCORE:
                    log.info("Max trades (3/3) — score %d < %d exceptional threshold. Denied.",
                             max_score, EXCEPTIONAL_SCORE)
                    time.sleep(120)
                    continue
                log.info("EXCEPTIONAL MARKET — score=%d ≥ %d. Taking 4th trade.",
                         max_score, EXCEPTIONAL_SCORE)

            # ── India VIX gate ────────────────────────────────────────────────
            vix = get_india_vix()
            vix_ok, vix_reason = vix_allows_entry(vix)
            if not vix_ok:
                log.warning("VIX GATE: %s — skipping entry", vix_reason)
                time.sleep(LOOP_SECS)
                continue
            log.info("VIX: %s", vix_reason)

            # ── Max Pain + OI wall context ────────────────────────────────────
            # Fetch option chain for current expiry — used for directional bias
            # and to confirm we're not entering into a wall
            try:
                expiry_str = date.today().strftime("%d-%b-%Y")
                oc = get_option_chain_context("NIFTY", spot, expiry_str)
                if oc is not None:
                    mp_bias = max_pain_bias(spot, oc.max_pain)
                    log.info(
                        "OI: max_pain=%.0f CE_wall=%.0f PE_wall=%.0f PCR=%.2f bias=%s mp_bias=%s",
                        oc.max_pain, oc.ce_wall, oc.pe_wall, oc.pcr, oc.bias, mp_bias,
                    )
                    # Block CE entry if price is at or above CE wall (resistance)
                    if direction == "CE" and spot >= oc.ce_wall - 25:
                        log.warning(
                            "OI WALL GATE: spot=%.0f near CE_wall=%.0f — skipping CE entry",
                            spot, oc.ce_wall,
                        )
                        time.sleep(LOOP_SECS)
                        continue
                    # Block PE entry if price is at or below PE wall (support)
                    if direction == "PE" and spot <= oc.pe_wall + 25:
                        log.warning(
                            "OI WALL GATE: spot=%.0f near PE_wall=%.0f — skipping PE entry",
                            spot, oc.pe_wall,
                        )
                        time.sleep(LOOP_SECS)
                        continue
                    # Max pain alignment boost: log when direction aligns with max pain pull
                    if (direction == "CE" and mp_bias == "bullish") or \
                       (direction == "PE" and mp_bias == "bearish"):
                        log.info("MAX PAIN CONFLUENCE: direction=%s mp_bias=%s — high conviction", direction, mp_bias)
            except Exception as exc:
                log.warning("OI context fetch failed (non-fatal): %s", exc)

            # ── Resolve option + affordability check ──────────────────────────
            result = resolve_option(spot, direction)
            if result is None:
                log.warning("Cannot resolve ATM %s", direction)
                time.sleep(LOOP_SECS)
                continue

            sym, sid, lot = result
            lot = int(result[2]) * MAX_LOTS  # enforce MAX_LOTS cap (1 lot)
            time.sleep(API_DELAY)
            ltp = get_option_ltp(sid)
            if ltp <= 0:
                log.warning("LTP=0 for %s", sym)
                time.sleep(LOOP_SECS)
                continue

            balance = get_balance()
            cost    = ltp * lot * 1.2   # 20% buffer for slippage
            if cost > balance:
                log.warning("Cannot afford %s ₹%.0f > bal ₹%.0f", sym, cost, balance)
                time.sleep(LOOP_SECS)
                continue

            # ── Enter ─────────────────────────────────────────────────────────
            log.info("ENTER BUY %s @ ₹%.2f lot=%d cost≈₹%.0f bal=₹%.0f",
                     sym, ltp, lot, ltp * lot, balance)
            oid = place_order(sid, lot, "BUY")
            if oid is None:
                time.sleep(LOOP_SECS)
                continue

            state.update({
                "active": True, "symbol": sym, "sid": sid,
                "entry": ltp, "tp": round(ltp * (1 + TP_PCT), 2),
                "sl": round(ltp * (1 - SL_PCT), 2),
                "qty": lot, "last_dir": direction,
                "peak_ltp": ltp,
                "capital_profit_target": float(CAPITAL_TARGET),
                "last_momentum_score": max(bull, bear),
                "monitor_tick": 0,
            })
            state["trades"] += 1

            log.info("ACTIVE %s entry=%.2f TP=%.2f SL=%.2f trade=%d/%d (4th if score≥%d)",
                     sym, ltp, state["tp"], state["sl"], state["trades"], MAX_TRADES, EXCEPTIONAL_SCORE)

        except KeyboardInterrupt:
            log.info("Interrupted")
            break
        except Exception as exc:
            log.exception("Tick error: %s", exc)

        time.sleep(LOOP_SECS)

    log.info("Final PnL=₹%.0f trades=%d", state["daily_pnl"], state["trades"])
    log.info("=== DONE ===")


if __name__ == "__main__":
    run()
