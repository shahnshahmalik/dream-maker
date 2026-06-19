"""
Expiry Day Scalper v2 — 1-minute precision.

Timeframe choice rationale:
  1-min: expiry day moves happen in 2-3 candles. 5-min candles are too slow
         — by the time a 5m signal forms, half the premium move is gone.
         1-min catches the setup at birth, not the obituary.
  15-min for HTF trend context only (dual-timeframe approach).

Strategy (expiry-specific):
  - ORB: first 15-min (9:15–9:30) defines the range. Clean break = momentum trade.
  - Momentum continuation: EMA9 > EMA21 on 1m + 3 consecutive HH/HL (bull)
    or LH/LL (bear) on 1m confirms direction.
  - VWAP: price bouncing off VWAP with a rejection wick + volume.
  - Any 2 of 3 signals = entry.

  Volume: must be > 1.5× 10-bar average on signal candle.
  No trades in dead zone 12:00–13:00 IST.
  Theta kill at 14:15 IST (expiry day premium collapses faster).

Risk:
  TP = +60%, SL = -20% (3:1 R:R)
  Max 4 trades. ATM options only.
"""

from __future__ import annotations

import os, sys, time, logging, sqlite3, requests
from datetime import datetime, date
from zoneinfo import ZoneInfo
from dotenv import load_dotenv

sys.path.insert(0, "/home/ubuntu/projects/dream-maker")
load_dotenv("/home/ubuntu/projects/dream-maker/.env", override=True)

# Read token directly from file to avoid env inheritance issues
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
IST          = ZoneInfo("Asia/Kolkata")
DB_PATH      = "/home/ubuntu/projects/dream-maker/data/scrip_master.db"

TP_PCT       = 0.60
SL_PCT       = 0.20
MAX_TRADES   = 4
THETA_KILL   = (14, 15)   # 14:15 IST — earlier on expiry day
DEAD_START   = (12, 0)
DEAD_END     = (13, 0)
LOOP_SECS    = 30         # 30s loop — Dhan allows ~1 req/sec, we make 3-4 per tick
API_DELAY    = 2.0        # seconds between consecutive API calls
VOL_MULT     = 1.05       # expiry day — 1m bars have lower vol than opening spike
MIN_SIGNAL   = 1          # single strong signal is enough on expiry day
LOOKBACK_1M  = 30         # how many 1-min bars to fetch

log = logging.getLogger("scalper")
log.setLevel(logging.INFO)
fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")
for h in [logging.FileHandler("/home/ubuntu/projects/dream-maker/logs/expiry_scalper.log"), logging.StreamHandler()]:
    h.setFormatter(fmt)
    if not any(type(x) == type(h) for x in log.handlers):
        log.addHandler(h)

state = {
    "trades": 0, "active": False,
    "symbol": None, "sid": None,
    "entry": None, "tp": None, "sl": None, "qty": None,
    "peak_ltp": None,       # highest LTP seen since entry — for trailing SL
    "or_high": None, "or_low": None, "or_set": False,
    "last_dir": None, "daily_pnl": 0.0,
    "cooldown_until": 0,
}

TRAIL_ACTIVATE_PCT = 0.15   # start trailing once +15% gained
TRAIL_LOCK_PCT     = 0.50   # SL trails at 50% of peak gain (locks half)


# ── API ───────────────────────────────────────────────────────────────────────

def get_spot() -> float:
    try:
        r = requests.post("https://api.dhan.co/v2/marketfeed/ltp", headers=_headers(),
            json={"IDX_I": [13]}, timeout=6)
        return r.json().get("data", {}).get("IDX_I", {}).get("13", {}).get("last_price", 0.0)
    except Exception:
        return 0.0


def get_candles(interval: str = "1", limit: int = 60) -> list[dict]:
    """Fetch intraday candles. interval: '1','5','15'."""
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
            log.warning("LTP returned 0 for sid=%d (attempt %d) — raw: %s", sid, attempt + 1, str(seg)[:80])
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
    try:
        atm = round(spot / 50) * 50
        today = date.today().strftime("%Y-%m-%d")
        conn = sqlite3.connect(DB_PATH)
        cur  = conn.cursor()
        cur.execute("""
            SELECT trading_symbol, security_id, lot_size FROM scrip_master
            WHERE symbol_name='NIFTY' AND option_type=?
              AND date(expiry_date) >= date(?)
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


# ── Indicators ────────────────────────────────────────────────────────────────

def ema(values: list[float], p: int) -> list[float]:
    if len(values) < p:
        return [0.0] * len(values)
    k = 2 / (p + 1)
    e = [sum(values[:p]) / p]
    for v in values[p:]:
        e.append(v * k + e[-1] * (1 - k))
    # pad front to match length
    return [e[0]] * (len(values) - len(e)) + e


def vwap(candles: list[dict]) -> float:
    tv = sum(((c["high"] + c["low"] + c["close"]) / 3) * c["volume"] for c in candles)
    v  = sum(c["volume"] for c in candles)
    return tv / v if v else 0.0


# ── Time helpers ──────────────────────────────────────────────────────────────

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

def in_trade_window() -> bool:
    return hm() >= (9, 30) and not past_theta_kill() and not in_dead_zone()


# ── Signal engine ─────────────────────────────────────────────────────────────

def score_signal(candles_1m: list[dict], candles_15m: list[dict], spot: float) -> tuple[int, int, str]:
    """Returns (bull_score, bear_score, reason_string)."""
    if len(candles_1m) < 12:
        return 0, 0, "too_few_1m"

    closes  = [c["close"]  for c in candles_1m]
    highs   = [c["high"]   for c in candles_1m]
    lows    = [c["low"]    for c in candles_1m]
    opens   = [c["open"]   for c in candles_1m]
    volumes = [c["volume"] for c in candles_1m]

    # Volume is informational only on expiry day — intraday bars often dry up.
    # Don't block entries on volume; just log it.
    avg_vol = sum(volumes[-10:]) / min(10, len(volumes))
    vol_note = f"vol={volumes[-1]:.0f}/avg={avg_vol:.0f}"

    e9  = ema(closes, 9)
    e21 = ema(closes, 21)
    vw  = vwap(candles_1m)

    last_c = closes[-1]
    last_o = opens[-1]
    last_h = highs[-1]
    last_l = lows[-1]
    prev_c = closes[-2]
    bull_body = last_c > last_o
    bear_body = last_c < last_o

    bull, bear = 0, 0
    reasons = []

    # ── 1. EMA9/EMA21 cross or alignment on 1m ───────────────────────────────
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
    hh_hl = all(last5h[i] >= last5h[i-1] for i in range(1, 5)) and all(last5l[i] >= last5l[i-1] for i in range(1, 5))
    lh_ll = all(last5h[i] <= last5h[i-1] for i in range(1, 5)) and all(last5l[i] <= last5l[i-1] for i in range(1, 5))

    if hh_hl:
        bull += 1
        reasons.append("HH_HL_struct")
    if lh_ll:
        bear += 1
        reasons.append("LH_LL_struct")

    # ── 5. HTF 15m bias (context only — half weight) ─────────────────────────
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

    log.info("SCORE → bull=%d bear=%d | %s | %s", bull, bear, " | ".join(reasons) or "no_signals", vol_note)
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
            sid  = int(p["securityId"])
            qty  = int(p["netQty"])
            entry = float(p["buyAvg"])
            state.update({
                "active": True, "symbol": sym, "sid": sid,
                "entry": entry,
                "tp": round(entry * (1 + TP_PCT), 2),
                "sl": round(entry * (1 - SL_PCT), 2),
                "qty": qty, "trades": 1,
                "peak_ltp": entry,  # conservative — will update on first LTP read
            })
            log.info("RESTORED position %s qty=%d entry=%.2f TP=%.2f SL=%.2f",
                     sym, qty, entry, state["tp"], state["sl"])
            return  # only restore one — if multiple open, there's a bigger problem
    except Exception as e:
        log.warning("Could not restore state from broker: %s", e)


# ── Main loop ─────────────────────────────────────────────────────────────────

def run():
    log.info("=" * 60)
    log.info("EXPIRY SCALPER v2 — 1m entries | TP=+%.0f%% SL=-%.0f%% | MaxTrades=%d",
             TP_PCT * 100, SL_PCT * 100, MAX_TRADES)
    log.info("Theta kill %02d:%02d | Dead zone %02d:%02d–%02d:%02d",
             *THETA_KILL, *DEAD_START, *DEAD_END)
    log.info("=" * 60)
    restore_state_from_broker()

    while True:
        try:
            now = ist_now()

            if past_market_close():
                log.info("Market closed. Done.")
                break

            # ── Theta kill ────────────────────────────────────────────────────
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

            # ── Monitor active trade (skip candle/spot fetch to avoid 429) ──────
            if state["active"]:
                ltp = get_option_ltp(state["sid"])
                if ltp <= 0:
                    log.warning("LTP=0 after retries — will retry next tick")
                    time.sleep(LOOP_SECS)
                    continue

                # Update peak
                if state["peak_ltp"] is None or ltp > state["peak_ltp"]:
                    state["peak_ltp"] = ltp

                # Trailing SL: activates at +15%, locks 50% of peak gain
                peak_gain_pct = (state["peak_ltp"] - state["entry"]) / state["entry"]
                if peak_gain_pct >= TRAIL_ACTIVATE_PCT:
                    trail_sl = round(state["entry"] + (state["peak_ltp"] - state["entry"]) * TRAIL_LOCK_PCT, 2)
                    if trail_sl > state["sl"]:
                        log.info("TRAIL SL raised %.2f → %.2f (peak=%.2f +%.1f%%)",
                                 state["sl"], trail_sl, state["peak_ltp"], peak_gain_pct * 100)
                        state["sl"] = trail_sl

                pnl_pct = (ltp - state["entry"]) / state["entry"]
                log.info("MONITOR %s LTP=%.2f PnL=%.1f%% TP=%.2f SL=%.2f peak=%.2f",
                         state["symbol"], ltp, pnl_pct * 100, state["tp"], state["sl"], state["peak_ltp"])

                hit_tp = ltp >= state["tp"]
                hit_sl = ltp <= state["sl"]

                if hit_tp or hit_sl:
                    tag = "TP ✅" if hit_tp else "SL ❌"
                    log.info("EXIT %s @ ₹%.2f", tag, ltp)
                    oid = place_order(state["sid"], state["qty"], "SELL")
                    if oid:
                        pnl = (ltp - state["entry"]) * state["qty"]
                        state["daily_pnl"] += pnl
                        state["active"]     = False
                        state["cooldown_until"] = time.time() + 120
                        log.info("CLOSED entry=%.2f exit=%.2f PnL=₹%.0f | daily=₹%.0f",
                                 state["entry"], ltp, pnl, state["daily_pnl"])
                time.sleep(LOOP_SECS)
                continue

            # ── Fetch data (only needed when looking for new entries) ─────────
            candles_1m  = get_candles("1",  LOOKBACK_1M)
            time.sleep(API_DELAY)
            spot = get_spot()
            candles_15m = candles_1m[-20:] if len(candles_1m) >= 20 else candles_1m

            if spot <= 0 or len(candles_1m) < 5:
                log.warning("Bad data spot=%.2f 1m=%d", spot, len(candles_1m))
                time.sleep(LOOP_SECS)
                continue

            # ── Build OR from 9:15–9:30 1m candles ───────────────────────────
            if not state["or_set"]:
                or_c = [c for c in candles_1m if c["hh"] == 9 and 15 <= c["mm"] <= 29]
                if len(or_c) >= 5:
                    state["or_high"] = max(c["high"] for c in or_c)
                    state["or_low"]  = min(c["low"]  for c in or_c)
                    state["or_set"]  = True
                    log.info("OR locked: H=%.2f L=%.2f range=%.2f pts",
                             state["or_high"], state["or_low"],
                             state["or_high"] - state["or_low"])

            last = candles_1m[-1]
            log.info("Spot=%.2f | 1m %s C=%.2f | 1m_bars=%d",
                     spot, last["time"], last["close"], len(candles_1m))

            # ── Dead zone / window checks ─────────────────────────────────────
            if in_dead_zone():
                log.info("DEAD ZONE — no trades")
                time.sleep(60)
                continue

            if not in_trade_window():
                log.info("Outside window")
                time.sleep(LOOP_SECS)
                continue

            if state["trades"] >= MAX_TRADES:
                log.info("Max trades hit")
                time.sleep(120)
                continue

            if time.time() < state["cooldown_until"]:
                log.info("Cooldown %.0fs remaining", state["cooldown_until"] - time.time())
                time.sleep(LOOP_SECS)
                continue

            # ── Signal ───────────────────────────────────────────────────────
            bull, bear, reason = score_signal(candles_1m, candles_15m, spot)

            direction = None
            if bull >= MIN_SIGNAL and bull > bear:
                direction = "CE"
            elif bear >= MIN_SIGNAL and bear > bull:
                direction = "PE"

            if direction is None:
                time.sleep(LOOP_SECS)
                continue

            log.info("SIGNAL %s — %s", direction, reason)

            # ── Resolve + check ───────────────────────────────────────────────
            result = resolve_option(spot, direction)
            if result is None:
                log.warning("Cannot resolve ATM %s", direction)
                time.sleep(LOOP_SECS)
                continue

            sym, sid, lot = result
            time.sleep(API_DELAY)
            ltp = get_option_ltp(sid)
            if ltp <= 0:
                log.warning("LTP=0 for %s", sym)
                time.sleep(LOOP_SECS)
                continue

            balance = get_balance()
            cost    = ltp * lot * 1.2
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
            })
            state["trades"] += 1

            log.info("ACTIVE %s entry=%.2f TP=%.2f SL=%.2f trade=%d/%d",
                     sym, ltp, state["tp"], state["sl"], state["trades"], MAX_TRADES)

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
