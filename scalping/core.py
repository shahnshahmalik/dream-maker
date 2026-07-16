"""Shared API helpers, indicators, trail math, and sniper signal scoring."""

from __future__ import annotations

import logging
import os
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import requests
from dotenv import load_dotenv

from scalping.config import SniperConfig
from scalping.logfmt import fmt_pct, fmt_rupees, paint, setup_colored_logging, C

ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / ".env"
DB_PATH = ROOT / "data" / "scrip_master.db"
LOG_DIR = ROOT / "logs"
IST = ZoneInfo("Asia/Kolkata")

load_dotenv(ENV_PATH, override=True)

log = logging.getLogger("sniper")


def setup_logging(log_filename: str) -> logging.Logger:
    return setup_colored_logging(LOG_DIR, log_filename)


# ── Auth / HTTP ───────────────────────────────────────────────────────────────

def _headers() -> dict:
    token, cid = "", ""
    try:
        with open(ENV_PATH) as f:
            for line in f:
                line = line.strip()
                if line.startswith("DHAN_ACCESS_TOKEN="):
                    token = line.split("=", 1)[1].strip()
                elif line.startswith("DHAN_CLIENT_ID="):
                    cid = line.split("=", 1)[1].strip()
    except OSError:
        pass
    return {"access-token": token, "client-id": cid, "Content-Type": "application/json"}


def get_spot() -> float:
    try:
        r = requests.post(
            "https://api.dhan.co/v2/marketfeed/ltp",
            headers=_headers(),
            json={"IDX_I": [13]},
            timeout=6,
        )
        return float(r.json().get("data", {}).get("IDX_I", {}).get("13", {}).get("last_price", 0.0))
    except Exception:
        return 0.0


def get_candles(interval: str, limit: int) -> list[dict]:
    try:
        today = date.today().strftime("%Y-%m-%d")
        r = requests.post(
            "https://api.dhan.co/v2/charts/intraday",
            headers=_headers(),
            json={
                "securityId": "13",
                "exchangeSegment": "IDX_I",
                "instrument": "INDEX",
                "interval": interval,
                "oi": False,
                "fromDate": today,
                "toDate": today,
            },
            timeout=10,
        )
        d = r.json()
        out: list[dict] = []
        for i in range(len(d.get("timestamp", []))):
            ist_s = int(d["timestamp"][i]) + 19800
            hh = (ist_s % 86400) // 3600
            mm = (ist_s % 3600) // 60
            out.append({
                "hh": hh, "mm": mm, "time": f"{hh:02d}:{mm:02d}",
                "open": d["open"][i], "high": d["high"][i],
                "low": d["low"][i], "close": d["close"][i],
                "volume": d["volume"][i],
            })
        return out[-limit:] if len(out) > limit else out
    except Exception:
        return []


def get_option_ltp(sid: int, retries: int = 2) -> float:
    for attempt in range(retries):
        try:
            r = requests.post(
                "https://api.dhan.co/v2/marketfeed/ltp",
                headers=_headers(),
                json={"NSE_FNO": [sid]},
                timeout=8,
            )
            if r.status_code == 429:
                time.sleep(6 * (attempt + 1))
                continue
            seg = r.json().get("data", {}).get("NSE_FNO", {})
            price = float(seg.get(str(sid), {}).get("last_price", 0.0))
            if price > 0:
                return price
            time.sleep(2)
        except Exception as e:
            log.warning("LTP exception attempt %d: %s", attempt + 1, e)
            time.sleep(2)
    return 0.0


def get_balance() -> float:
    try:
        r = requests.get("https://api.dhan.co/v2/fundlimit", headers=_headers(), timeout=6)
        return float(r.json().get("availabelBalance", 0))
    except Exception:
        return 0.0


def resolve_option(
    spot: float,
    opt_type: str,
    *,
    same_day_ok: bool,
) -> tuple[str, int, int, str] | None:
    """Return (symbol, security_id, lot_size, expiry_yyyy_mm_dd) for ATM NIFTY."""
    try:
        atm = round(spot / 50) * 50
        today = date.today().strftime("%Y-%m-%d")
        op = ">=" if same_day_ok else ">"
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute(
            f"""
            SELECT trading_symbol, security_id, lot_size, expiry_date FROM scrip_master
            WHERE symbol_name='NIFTY' AND option_type=?
              AND date(expiry_date) {op} date(?)
              AND strike_price BETWEEN ? AND ?
            ORDER BY date(expiry_date) ASC, ABS(strike_price - ?) ASC LIMIT 1
            """,
            (opt_type, today, atm - 100, atm + 100, atm),
        )
        row = cur.fetchone()
        conn.close()
        if not row:
            return None
        exp = str(row[3])[:10]
        return row[0], int(row[1]), int(row[2]), exp
    except Exception as e:
        log.warning("resolve_option failed: %s", e)
        return None


def place_order(sid: int, qty: int, side: str) -> str | None:
    try:
        cid = os.getenv("DHAN_CLIENT_ID", "").strip() or _headers()["client-id"]
        r = requests.post(
            "https://api.dhan.co/v2/orders",
            headers=_headers(),
            json={
                "dhanClientId": cid,
                "transactionType": side,
                "exchangeSegment": "NSE_FNO",
                "productType": "INTRADAY",
                "orderType": "MARKET",
                "validity": "DAY",
                "securityId": str(sid),
                "quantity": qty,
                "price": 0,
                "triggerPrice": 0,
                "disclosedQuantity": 0,
                "afterMarketOrder": False,
                "boProfitValue": 0,
                "boStopLossValue": 0,
            },
            timeout=10,
        )
        if r.status_code == 200:
            oid = r.json().get("orderId")
            log.info("ORDER %s sid=%d qty=%d → orderId=%s", side, sid, qty, oid)
            return str(oid) if oid else None
        log.error("ORDER FAILED %d: %s", r.status_code, r.text[:150])
        return None
    except Exception as e:
        log.error("ORDER EXCEPTION: %s", e)
        return None


def confirm_fill_price(order_id: str, sid: int, fallback_ltp: float, wait_secs: float = 4.0) -> float:
    """Poll order status for average fill; fall back to LTP if unavailable."""
    deadline = time.time() + wait_secs
    while time.time() < deadline:
        try:
            r = requests.get(
                f"https://api.dhan.co/v2/orders/{order_id}",
                headers=_headers(),
                timeout=6,
            )
            if r.status_code == 200:
                data = r.json()
                row = data if isinstance(data, dict) else {}
                if isinstance(data, list) and data:
                    row = data[0]
                avg = float(
                    row.get("averageTradedPrice")
                    or row.get("avgPrice")
                    or row.get("price")
                    or 0
                )
                status = str(row.get("orderStatus") or row.get("status") or "").upper()
                if avg > 0 and ("TRADED" in status or "FILLED" in status or "COMPLETE" in status):
                    log.info("FILL confirmed orderId=%s avg=%.2f status=%s", order_id, avg, status)
                    return avg
                if avg > 0:
                    return avg
        except Exception as e:
            log.debug("fill poll: %s", e)
        time.sleep(0.8)

    ltp = get_option_ltp(sid) or fallback_ltp
    log.warning("FILL unconfirmed orderId=%s — using LTP %.2f", order_id, ltp)
    return ltp


def fetch_open_nifty_long() -> dict | None:
    try:
        r = requests.get("https://api.dhan.co/v2/positions", headers=_headers(), timeout=10)
        positions = r.json()
        if not isinstance(positions, list):
            return None
        for p in positions:
            if p.get("positionType") != "LONG" or p.get("netQty", 0) <= 0:
                continue
            if "NIFTY" not in str(p.get("tradingSymbol", "")):
                continue
            return {
                "symbol": p.get("tradingSymbol", ""),
                "sid": int(p["securityId"]),
                "qty": int(p["netQty"]),
                "entry": float(p["buyAvg"]),
            }
    except Exception as e:
        log.warning("positions fetch failed: %s", e)
    return None


# ── Indicators ────────────────────────────────────────────────────────────────

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
    v = sum(c["volume"] for c in candles)
    return tv / v if v else 0.0


# ── Time ──────────────────────────────────────────────────────────────────────

def ist_now() -> datetime:
    return datetime.now(IST)


def hm() -> tuple[int, int]:
    n = ist_now()
    return n.hour, n.minute


def _hm_ge(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a >= b


def in_dead_zone(cfg: SniperConfig) -> bool:
    return cfg.dead_start <= hm() < cfg.dead_end


def past_theta_kill(cfg: SniperConfig) -> bool:
    return hm() >= cfg.theta_kill


def past_market_close() -> bool:
    return hm() >= (15, 30)


def in_trade_window(cfg: SniperConfig) -> bool:
    return _hm_ge(hm(), cfg.entries_from) and not past_theta_kill(cfg) and not in_dead_zone(cfg)


def expiry_to_nse_str(expiry_yyyy_mm_dd: str) -> str:
    """Convert YYYY-MM-DD → DD-MMM-YYYY for NSE option-chain API."""
    d = datetime.strptime(expiry_yyyy_mm_dd[:10], "%Y-%m-%d")
    return d.strftime("%d-%b-%Y")


# ── Trail (percent-based) ─────────────────────────────────────────────────────

def compute_trail_sl(entry: float, peak: float, current_sl: float, cfg: SniperConfig) -> float | None:
    """Return raised SL or None if no change.

    Stage 1: peak >= +trail_activate → SL to entry*(1+be_buffer)
    Stage 2: also trail peak*(1-trail_from_peak), taking the higher of the two
    Never sets SL above current peak (avoids instant stop-out).
    """
    if entry <= 0 or peak <= 0:
        return None
    peak_gain = (peak - entry) / entry
    if peak_gain < cfg.trail_activate_pct:
        return None

    be_sl = round(entry * (1.0 + cfg.trail_be_buffer_pct), 2)
    peak_sl = round(peak * (1.0 - cfg.trail_from_peak_pct), 2)
    candidate = max(be_sl, peak_sl)
    # Hard ceiling: never trail above last traded peak (would auto-fire)
    candidate = min(candidate, round(peak * 0.995, 2))
    if candidate > current_sl:
        return candidate
    return None


def capital_target_rupees(entry: float, qty: int, cfg: SniperConfig) -> float:
    cost = entry * qty
    raw = cost * cfg.capital_target_pct
    return round(min(cfg.capital_target_cap, max(cfg.capital_target_floor, raw)), 2)


# ── Signals ───────────────────────────────────────────────────────────────────

PRIMARY_BULL = ("EMA_cross_bull", "ORB_bull_break", "VWAP_reject_bull")
PRIMARY_BEAR = ("EMA_cross_bear", "ORB_bear_break", "VWAP_reject_bear")


@dataclass
class SignalResult:
    bull: int = 0
    bear: int = 0
    reasons: list[str] = field(default_factory=list)
    blocked: str | None = None
    htf_bias: str | None = None  # "bull" | "bear" | None

    @property
    def reason_str(self) -> str:
        if self.blocked:
            return self.blocked
        return " | ".join(self.reasons) or "no_signals"

    def has_primary(self, direction: str) -> bool:
        keys = PRIMARY_BULL if direction == "CE" else PRIMARY_BEAR
        return any(any(r.startswith(k) or r == k for k in keys) for r in self.reasons)


def score_signal(
    candles: list[dict],
    candles_15m: list[dict],
    *,
    or_high: float | None,
    or_low: float | None,
    or_set: bool,
    vol_mult: float,
) -> SignalResult:
    """Sniper score on last *closed* bar for volume; signal candle = last bar."""
    out = SignalResult()
    if len(candles) < 12:
        out.blocked = "too_few_bars"
        return out

    closes = [c["close"] for c in candles]
    highs = [c["high"] for c in candles]
    lows = [c["low"] for c in candles]
    opens = [c["open"] for c in candles]
    volumes = [c["volume"] for c in candles]

    # Volume gate on last *closed* candle ([-2]) vs prior closed bars — avoids
    # forming-bar partial volume always failing mid-candle.
    closed_vol = volumes[-2] if len(volumes) >= 2 else volumes[-1]
    baseline = volumes[-12:-2] if len(volumes) >= 12 else volumes[:-1]
    if not baseline:
        baseline = volumes[-10:]
    # Drop the single largest bar (open spike) from the average
    if len(baseline) >= 4:
        trimmed = sorted(baseline)[:-1]
        avg_vol = sum(trimmed) / len(trimmed)
    else:
        avg_vol = sum(baseline) / len(baseline)

    if closed_vol < avg_vol * vol_mult:
        out.blocked = f"low_vol={closed_vol:.0f}<{avg_vol * vol_mult:.0f}"
        return out

    e9 = ema(closes, 9)
    e21 = ema(closes, 21)
    vw = vwap(candles)

    last_c, last_o = closes[-1], opens[-1]
    last_h, last_l = highs[-1], lows[-1]
    prev_c = closes[-2]
    bull_body = last_c > last_o
    bear_body = last_c < last_o

    bull = bear = 0
    reasons: list[str] = []

    # 1. EMA cross (primary, +2) or alignment (+1)
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

    # 2. ORB — fresh break (+2) or hold (+1)
    if or_set and or_high is not None and or_low is not None:
        if last_c > or_high and bull_body and prev_c <= or_high:
            bull += 2
            reasons.append(f"ORB_bull_break>{or_high:.0f}")
        elif last_c > or_high and bull_body:
            bull += 1
            reasons.append("ORB_above")

        if last_c < or_low and bear_body and prev_c >= or_low:
            bear += 2
            reasons.append(f"ORB_bear_break<{or_low:.0f}")
        elif last_c < or_low and bear_body:
            bear += 1
            reasons.append("ORB_below")

    # 3. VWAP rejection (primary)
    if vw > 0:
        if abs(last_l - vw) / vw < 0.0015 and last_c > vw and bull_body:
            bull += 2
            reasons.append(f"VWAP_reject_bull@{vw:.0f}")
        if abs(last_h - vw) / vw < 0.0015 and last_c < vw and bear_body:
            bear += 2
            reasons.append(f"VWAP_reject_bear@{vw:.0f}")

    # 4. Structure on last 5 bars
    last5h, last5l = highs[-5:], lows[-5:]
    if (all(last5h[i] >= last5h[i - 1] for i in range(1, 5))
            and all(last5l[i] >= last5l[i - 1] for i in range(1, 5))):
        bull += 1
        reasons.append("HH_HL_struct")
    if (all(last5h[i] <= last5h[i - 1] for i in range(1, 5))
            and all(last5l[i] <= last5l[i - 1] for i in range(1, 5))):
        bear += 1
        reasons.append("LH_LL_struct")

    # 5. True 15m HTF bias
    htf_bias = None
    if len(candles_15m) >= 5:
        htf_closes = [c["close"] for c in candles_15m]
        span9 = min(9, len(htf_closes))
        span21 = min(21, len(htf_closes))
        htf_e9 = ema(htf_closes, span9)
        htf_e21 = ema(htf_closes, span21)
        if htf_e9[-1] > htf_e21[-1]:
            bull += 1
            reasons.append("HTF_bull")
            htf_bias = "bull"
        elif htf_e9[-1] < htf_e21[-1]:
            bear += 1
            reasons.append("HTF_bear")
            htf_bias = "bear"

    out.bull = bull
    out.bear = bear
    out.reasons = reasons
    out.htf_bias = htf_bias
    log.info(
        "SCORE  %s  %s  | %s | vol_closed=%.0f thr=%.0f",
        paint(f"bull={bull}", C.GREEN if bull >= bear else C.DIM),
        paint(f"bear={bear}", C.RED if bear >= bull else C.DIM),
        " | ".join(reasons) or "no_signals",
        closed_vol, avg_vol * vol_mult,
    )
    return out


def pick_direction(sig: SignalResult, cfg: SniperConfig) -> str | None:
    """Return 'CE' / 'PE' only when sniper confluence is met."""
    if sig.blocked:
        return None

    direction = None
    score = 0
    if sig.bull >= cfg.min_signal and sig.bull > sig.bear:
        direction, score = "CE", sig.bull
    elif sig.bear >= cfg.min_signal and sig.bear > sig.bull:
        direction, score = "PE", sig.bear
    else:
        return None

    if cfg.require_primary and not sig.has_primary(direction):
        log.info("SNIPER GATE: score=%d but no primary trigger — skip", score)
        return None

    if cfg.require_htf_align and sig.htf_bias:
        if direction == "CE" and sig.htf_bias == "bear" and score < cfg.min_signal + 2:
            log.info("SNIPER GATE: CE vs HTF_bear score=%d — skip", score)
            return None
        if direction == "PE" and sig.htf_bias == "bull" and score < cfg.min_signal + 2:
            log.info("SNIPER GATE: PE vs HTF_bull score=%d — skip", score)
            return None

    return direction


# ── Engine state ──────────────────────────────────────────────────────────────

@dataclass
class EngineState:
    trades: int = 0
    active: bool = False
    symbol: str | None = None
    sid: int | None = None
    entry: float | None = None
    tp: float | None = None
    sl: float | None = None
    qty: int | None = None
    peak_ltp: float | None = None
    capital_profit_target: float | None = None
    last_momentum_score: int = 0
    monitor_tick: int = 0
    or_high: float | None = None
    or_low: float | None = None
    or_set: bool = False
    last_dir: str | None = None
    daily_pnl: float = 0.0
    cooldown_until: float = 0.0
    consecutive_losses: int = 0
    halted: bool = False
    halt_reason: str = ""


class SniperEngine:
    """Main loop shared by expiry and non-expiry snipers."""

    def __init__(self, cfg: SniperConfig):
        self.cfg = cfg
        self.state = EngineState()
        setup_logging(cfg.log_filename)

    def calendar_allows(self) -> bool:
        today = ist_now()
        wd = today.weekday()
        if self.cfg.run_on_weekday is not None and wd != self.cfg.run_on_weekday:
            log.info(
                "Not scheduled day (today=%s). %s runs on weekday=%s only. Exiting.",
                today.strftime("%A"), self.cfg.name, self.cfg.run_on_weekday,
            )
            return False
        if self.cfg.skip_weekday is not None and wd == self.cfg.skip_weekday:
            log.info(
                "Skip day (today=%s). %s does not run on weekday=%s. Exiting.",
                today.strftime("%A"), self.cfg.name, self.cfg.skip_weekday,
            )
            return False
        return True

    def restore_position(self) -> None:
        pos = fetch_open_nifty_long()
        if not pos:
            return
        entry = pos["entry"]
        self.state.active = True
        self.state.symbol = pos["symbol"]
        self.state.sid = pos["sid"]
        self.state.qty = pos["qty"]
        self.state.entry = entry
        self.state.tp = round(entry * (1 + self.cfg.tp_pct), 2)
        self.state.sl = round(entry * (1 - self.cfg.sl_pct), 2)
        self.state.peak_ltp = entry
        self.state.capital_profit_target = capital_target_rupees(entry, pos["qty"], self.cfg)
        self.state.trades = max(1, self.state.trades)
        log.info(
            "RESTORED %s qty=%d entry=%.2f TP=%.2f SL=%.2f cap_tgt=Rs%.0f",
            pos["symbol"], pos["qty"], entry, self.state.tp, self.state.sl,
            self.state.capital_profit_target,
        )

    def _lock_orb(self, candles: list[dict]) -> None:
        if self.state.or_set:
            return
        sh, sm = self.cfg.orb_start
        eh, em = self.cfg.orb_end_mm
        or_c = [
            c for c in candles
            if (c["hh"], c["mm"]) >= (sh, sm) and (c["hh"], c["mm"]) <= (eh, em)
        ]
        # Need enough bars for a meaningful range
        min_bars = 5 if self.cfg.entry_interval == "1" else 4
        if len(or_c) < min_bars:
            return
        self.state.or_high = max(c["high"] for c in or_c)
        self.state.or_low = min(c["low"] for c in or_c)
        self.state.or_set = True
        log.info(
            "ORB locked: H=%.2f L=%.2f range=%.2f pts (%d bars)",
            self.state.or_high, self.state.or_low,
            self.state.or_high - self.state.or_low, len(or_c),
        )

    def _risk_halted(self) -> bool:
        if self.state.halted:
            return True
        if self.state.daily_pnl <= -self.cfg.daily_max_loss:
            self.state.halted = True
            self.state.halt_reason = f"daily_max_loss(Rs{self.state.daily_pnl:.0f})"
            log.warning("HALT - %s", self.state.halt_reason)
            return True
        if self.state.consecutive_losses >= self.cfg.max_consecutive_losses:
            self.state.halted = True
            self.state.halt_reason = f"consecutive_losses({self.state.consecutive_losses})"
            log.warning("HALT - %s", self.state.halt_reason)
            return True
        if self.state.daily_pnl >= self.cfg.daily_target:
            self.state.halted = True
            self.state.halt_reason = f"daily_target(Rs{self.state.daily_pnl:.0f})"
            log.info("HALT - %s", self.state.halt_reason)
            return True
        return False

    def _register_close(self, pnl: float, *, entry: float, exit_px: float) -> None:
        self.state.daily_pnl += pnl
        self.state.active = False
        if pnl < 0:
            self.state.consecutive_losses += 1
            self.state.cooldown_until = time.time() + self.cfg.cooldown_loss_secs
        else:
            self.state.consecutive_losses = 0
            self.state.cooldown_until = time.time() + self.cfg.cooldown_win_secs
        log.info(
            "CLOSED entry=%.2f exit=%.2f %s | daily %s | streak_loss=%d",
            entry, exit_px,
            fmt_rupees(pnl, bold=True),
            fmt_rupees(self.state.daily_pnl, bold=True),
            self.state.consecutive_losses,
        )

    def _exit_position(self, tag: str, ltp: float) -> bool:
        assert self.state.sid and self.state.qty and self.state.entry is not None
        tag_u = tag.upper()
        if "TP" in tag_u or "TRAIL" in tag_u or "CAPITAL" in tag_u:
            tag_paint = paint(f"EXIT {tag}", C.BOLD, C.GREEN)
        elif "SL" in tag_u:
            tag_paint = paint(f"EXIT {tag}", C.BOLD, C.RED)
        else:
            tag_paint = paint(f"EXIT {tag}", C.BOLD, C.YELLOW)
        log.info("%s @ Rs%.2f", tag_paint, ltp)
        oid = place_order(self.state.sid, self.state.qty, "SELL")
        if not oid:
            return False
        fill = confirm_fill_price(oid, self.state.sid, ltp)
        entry = self.state.entry
        pnl = (fill - entry) * self.state.qty
        self._register_close(pnl, entry=entry, exit_px=fill)
        return True

    def _monitor(self) -> None:
        assert self.state.sid and self.state.entry is not None
        ltp = get_option_ltp(self.state.sid)
        if ltp <= 0:
            log.warning("LTP=0 — retry next tick")
            return

        self.state.monitor_tick += 1
        if self.state.monitor_tick % self.cfg.momentum_refresh_ticks == 0:
            entry_bars = get_candles(self.cfg.entry_interval, self.cfg.lookback_entry)
            htf = get_candles("15", self.cfg.lookback_htf)
            if len(entry_bars) >= 12:
                sig = score_signal(
                    entry_bars, htf,
                    or_high=self.state.or_high, or_low=self.state.or_low,
                    or_set=self.state.or_set, vol_mult=self.cfg.vol_mult,
                )
                self.state.last_momentum_score = max(sig.bull, sig.bear)
                log.info("MOMENTUM score=%d", self.state.last_momentum_score)

        if self.state.peak_ltp is None or ltp > self.state.peak_ltp:
            self.state.peak_ltp = ltp

        new_sl = compute_trail_sl(
            self.state.entry, self.state.peak_ltp, self.state.sl or 0.0, self.cfg,
        )
        if new_sl is not None and self.state.sl is not None:
            log.info(
                "%s %.2f -> %s  (peak=%.2f %s)",
                paint("TRAIL SL", C.GREEN),
                self.state.sl,
                paint(f"{new_sl:.2f}", C.BOLD, C.GREEN),
                self.state.peak_ltp,
                fmt_pct((self.state.peak_ltp - self.state.entry) / self.state.entry),
            )
            self.state.sl = new_sl

        pnl_pct = (ltp - self.state.entry) / self.state.entry
        pnl_rs = (ltp - self.state.entry) * (self.state.qty or 0)
        log.info(
            "MONITOR %s  LTP=%.2f  %s (%s)  TP=%.2f  SL=%.2f  peak=%.2f",
            self.state.symbol, ltp,
            fmt_pct(pnl_pct, bold=True),
            fmt_rupees(pnl_rs, bold=True),
            self.state.tp, self.state.sl, self.state.peak_ltp,
        )

        hit_tp = self.state.tp is not None and ltp >= self.state.tp
        hit_sl = self.state.sl is not None and ltp <= self.state.sl
        hit_cap = (
            self.state.capital_profit_target is not None
            and pnl_rs >= self.state.capital_profit_target
        )
        high_mom = self.state.last_momentum_score >= self.cfg.high_momentum_score

        if hit_cap and not high_mom and not hit_tp:
            self._exit_position(
                f"CAPITAL Rs{self.state.capital_profit_target:.0f}",
                ltp,
            )
            return
        if hit_cap and high_mom:
            log.info("CAPITAL reached but momentum=%d — let run", self.state.last_momentum_score)

        if hit_tp:
            self._exit_position("TP", ltp)
        elif hit_sl:
            # Distinguish trail/BE vs hard SL for logs
            be_level = self.state.entry * (1 + self.cfg.trail_be_buffer_pct)
            tag = "TRAIL" if self.state.sl and self.state.sl >= be_level - 0.01 else "SL"
            self._exit_position(tag, ltp)

    def _oi_vix_ok(self, direction: str, spot: float, expiry_ymd: str) -> bool:
        if self.cfg.use_vix_gate:
            try:
                from utils.market_data import get_india_vix, vix_allows_entry
                vix = get_india_vix()
                ok, reason = vix_allows_entry(vix)
                if not ok:
                    log.warning("VIX GATE: %s", reason)
                    return False
                log.info("VIX: %s", reason)
            except Exception as e:
                log.warning("VIX check failed (allow): %s", e)

        if not self.cfg.use_oi_gate:
            return True
        try:
            from utils.market_data import get_option_chain_context, max_pain_bias
            exp_nse = expiry_to_nse_str(expiry_ymd)
            oc = get_option_chain_context("NIFTY", spot, exp_nse)
            if oc is None:
                return True
            mp = max_pain_bias(spot, oc.max_pain)
            log.info(
                "OI: max_pain=%.0f CE_wall=%.0f PE_wall=%.0f PCR=%.2f bias=%s mp=%s",
                oc.max_pain, oc.ce_wall, oc.pe_wall, oc.pcr, oc.bias, mp,
            )
            if direction == "CE" and spot >= oc.ce_wall - 25:
                log.warning("OI WALL: spot near CE_wall=%.0f — skip CE", oc.ce_wall)
                return False
            if direction == "PE" and spot <= oc.pe_wall + 25:
                log.warning("OI WALL: spot near PE_wall=%.0f — skip PE", oc.pe_wall)
                return False
            if (direction == "CE" and mp == "bullish") or (direction == "PE" and mp == "bearish"):
                log.info("MAX PAIN confluence direction=%s", direction)
        except Exception as e:
            log.warning("OI context failed (non-fatal): %s", e)
        return True

    def _try_entry(self, candles: list[dict], htf: list[dict], spot: float) -> None:
        if self._risk_halted():
            return
        if not in_trade_window(self.cfg):
            log.info("Outside window")
            return
        if time.time() < self.state.cooldown_until:
            log.info("Cooldown %.0fs", self.state.cooldown_until - time.time())
            return

        hard_cap = self.cfg.max_trades
        if self.cfg.exceptional_score is not None:
            hard_cap = self.cfg.max_trades + 1
        if self.state.trades >= hard_cap:
            log.info("Max trades %d/%d", self.state.trades, hard_cap)
            return

        sig = score_signal(
            candles, htf,
            or_high=self.state.or_high, or_low=self.state.or_low,
            or_set=self.state.or_set, vol_mult=self.cfg.vol_mult,
        )
        if sig.blocked:
            log.info("VOLUME/DATA GATE: %s", sig.blocked)
            return

        direction = pick_direction(sig, self.cfg)
        if direction is None:
            return

        score = max(sig.bull, sig.bear)
        if self.state.trades >= self.cfg.max_trades:
            if self.cfg.exceptional_score is None or score < self.cfg.exceptional_score:
                log.info(
                    "Max trades %d/%d — score %d < exceptional %s",
                    self.state.trades, self.cfg.max_trades, score, self.cfg.exceptional_score,
                )
                return
            log.info("EXCEPTIONAL score=%d — allowing extra trade", score)

        log.info(
            "%s %s - %s",
            paint(f"SIGNAL {direction}", C.BOLD, C.MAGENTA),
            paint(f"score={score}", C.CYAN),
            sig.reason_str,
        )

        resolved = resolve_option(spot, direction, same_day_ok=self.cfg.same_day_expiry_ok)
        if resolved is None:
            log.warning("Cannot resolve ATM %s", direction)
            return
        sym, sid, lot, expiry_ymd = resolved

        if not self._oi_vix_ok(direction, spot, expiry_ymd):
            return

        time.sleep(self.cfg.api_delay)
        ltp = get_option_ltp(sid)
        if ltp <= 0:
            log.warning("LTP=0 for %s", sym)
            return
        if ltp < self.cfg.min_premium or ltp > self.cfg.max_premium:
            log.warning(
                "PREMIUM GATE: %s LTP=%.2f outside [%.0f, %.0f]",
                sym, ltp, self.cfg.min_premium, self.cfg.max_premium,
            )
            return

        balance = get_balance()
        cost = ltp * lot * 1.15
        if cost > balance:
            log.warning("Cannot afford %s Rs%.0f > bal Rs%.0f", sym, cost, balance)
            return

        log.info(
            "%s %s @ Rs%.2f lot=%d cost~Rs%.0f bal=Rs%.0f",
            paint("ENTER BUY", C.BOLD, C.CYAN), sym, ltp, lot, ltp * lot, balance,
        )
        oid = place_order(sid, lot, "BUY")
        if oid is None:
            return

        fill = confirm_fill_price(oid, sid, ltp)
        cap_tgt = capital_target_rupees(fill, lot, self.cfg)
        self.state.active = True
        self.state.symbol = sym
        self.state.sid = sid
        self.state.entry = fill
        self.state.tp = round(fill * (1 + self.cfg.tp_pct), 2)
        self.state.sl = round(fill * (1 - self.cfg.sl_pct), 2)
        self.state.qty = lot
        self.state.last_dir = direction
        self.state.peak_ltp = fill
        self.state.capital_profit_target = cap_tgt
        self.state.last_momentum_score = score
        self.state.monitor_tick = 0
        self.state.trades += 1
        log.info(
            "%s %s  entry=%.2f  TP=%.2f  SL=%.2f  cap=%s  trade=%d/%d",
            paint("ACTIVE", C.BOLD, C.GREEN),
            sym, fill, self.state.tp, self.state.sl,
            fmt_rupees(cap_tgt, signed=False),
            self.state.trades, self.cfg.max_trades,
        )

    def run(self) -> None:
        if not self.calendar_allows():
            return

        log.info(paint("=" * 60, C.BOLD, C.BLUE))
        log.info(
            "%s %s -- %sm | TP=+%.0f%% SL=-%.0f%% | MaxTrades=%d | MinScore=%d",
            paint(self.cfg.name, C.BOLD, C.CYAN),
            self.cfg.version, self.cfg.entry_interval,
            self.cfg.tp_pct * 100, self.cfg.sl_pct * 100,
            self.cfg.max_trades, self.cfg.min_signal,
        )
        log.info(
            "Trail: activate +%.0f%% -> BE+%.0f%% then peak-%.0f%% | "
            "Daily %s / %s | Prem [%.0f, %.0f]",
            self.cfg.trail_activate_pct * 100,
            self.cfg.trail_be_buffer_pct * 100,
            self.cfg.trail_from_peak_pct * 100,
            paint(f"+Rs{self.cfg.daily_target:.0f}", C.GREEN),
            paint(f"-Rs{self.cfg.daily_max_loss:.0f}", C.RED),
            self.cfg.min_premium, self.cfg.max_premium,
        )
        log.info(
            "Theta %02d:%02d | Dead %02d:%02d-%02d:%02d | Entries from %02d:%02d",
            *self.cfg.theta_kill, *self.cfg.dead_start, *self.cfg.dead_end,
            *self.cfg.entries_from,
        )
        log.info(paint("=" * 60, C.BOLD, C.BLUE))
        self.restore_position()

        while True:
            try:
                now = ist_now()
                if past_market_close():
                    log.info("Market closed. Done.")
                    break

                if past_theta_kill(self.cfg):
                    if self.state.active:
                        ltp = get_option_ltp(self.state.sid)  # type: ignore[arg-type]
                        self._exit_position("THETA", ltp if ltp > 0 else self.state.entry or 0)
                    if self._risk_halted():
                        break
                    time.sleep(60)
                    continue

                log.info(
                    "-- %s | trade %d/%d | active=%s | daily %s --",
                    now.strftime("%H:%M:%S"), self.state.trades, self.cfg.max_trades,
                    paint("YES", C.BOLD, C.CYAN) if self.state.active else paint("no", C.DIM),
                    fmt_rupees(self.state.daily_pnl, bold=True),
                )

                if self.state.active:
                    self._monitor()
                    if self._risk_halted() and not self.state.active:
                        break
                    time.sleep(self.cfg.loop_secs)
                    continue

                if self._risk_halted():
                    log.info("Halted (%s) — idle until close", self.state.halt_reason)
                    time.sleep(120)
                    continue

                if in_dead_zone(self.cfg):
                    log.info("DEAD ZONE")
                    time.sleep(60)
                    continue

                candles = get_candles(self.cfg.entry_interval, self.cfg.lookback_entry)
                time.sleep(self.cfg.api_delay)
                htf = get_candles("15", self.cfg.lookback_htf)
                time.sleep(self.cfg.api_delay)
                spot = get_spot()

                if spot <= 0 or len(candles) < 5:
                    log.warning("Bad data spot=%.2f bars=%d", spot, len(candles))
                    time.sleep(self.cfg.loop_secs)
                    continue

                self._lock_orb(candles)
                last = candles[-1]
                log.info(
                    "Spot=%.2f | %sm %s C=%.2f | bars=%d htf=%d",
                    spot, self.cfg.entry_interval, last["time"], last["close"],
                    len(candles), len(htf),
                )

                self._try_entry(candles, htf, spot)

            except KeyboardInterrupt:
                log.info("Interrupted")
                break
            except Exception as exc:
                log.exception("Tick error: %s", exc)

            time.sleep(self.cfg.loop_secs)

        log.info(
            "Final %s  trades=%d",
            fmt_rupees(self.state.daily_pnl, bold=True),
            self.state.trades,
        )
        log.info(paint("=== DONE ===", C.BOLD, C.BLUE))
