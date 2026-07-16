"""Backtest the non-expiry sniper's live signal engine against real history.

Runs the ACTUAL production functions from scalping.core (score_signal,
pick_direction, ORB lock) bar-by-bar over data/NIFTY_5min_Jan_Jun_2026.csv —
not a reimplementation. This tests entry-signal quality only.

Money caveat: we have historical NIFTY *index* candles, not historical option
premium ticks, so trades are sized in index points using the same ATR-based
convention already established in scripts/backtest_indicators.py (TP = ATR14
x 2.0, SL = ATR14 x 1.0 — a 2:1 R:R), entering at the next bar's open after a
signal closes. This measures whether the sniper's entry logic identifies real
directional edge; it is NOT a simulation of rupee option PnL.

Run: python scripts/backtest_sniper.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scalping.config import NonExpirySniperConfig as cfg  # noqa: E402
from scalping.core import pick_direction, score_signal  # noqa: E402

CSV_PATH = ROOT / "data" / "NIFTY_5min_Jan_Jun_2026.csv"
ATR_PERIOD = 14
TP_MULT = 2.0
SL_MULT = 1.0


def to_candle_dicts(df: pd.DataFrame) -> list[dict]:
    out = []
    for row in df.itertuples(index=False):
        out.append({
            "hh": row.datetime.hour, "mm": row.datetime.minute,
            "time": f"{row.datetime.hour:02d}:{row.datetime.minute:02d}",
            "open": row.open, "high": row.high, "low": row.low,
            "close": row.close, "volume": row.volume,
        })
    return out


def resample_15m(candles: list[dict]) -> list[dict]:
    """Group sequential 5m bars into 15m OHLCV (3-bar buckets from 9:15)."""
    out = []
    for i in range(0, len(candles) - 2, 3):
        chunk = candles[i:i + 3]
        if len(chunk) < 3:
            break
        out.append({
            "hh": chunk[0]["hh"], "mm": chunk[0]["mm"], "time": chunk[0]["time"],
            "open": chunk[0]["open"], "high": max(c["high"] for c in chunk),
            "low": min(c["low"] for c in chunk), "close": chunk[-1]["close"],
            "volume": sum(c["volume"] for c in chunk),
        })
    return out


def atr14(candles: list[dict]) -> list[float]:
    trs = [0.0]
    for i in range(1, len(candles)):
        h, l, pc = candles[i]["high"], candles[i]["low"], candles[i - 1]["close"]
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    out = [float("nan")] * len(candles)
    for i in range(ATR_PERIOD, len(candles)):
        out[i] = sum(trs[i - ATR_PERIOD + 1: i + 1]) / ATR_PERIOD
    return out


@dataclass
class Trade:
    date: str
    direction: str
    entry_time: str
    entry: float
    exit: float
    exit_reason: str
    r_points: float
    score: int
    reason: str


@dataclass
class DayResult:
    date: str
    signals_seen: int = 0
    blocked_volume: int = 0
    blocked_no_direction: int = 0
    blocked_gate: int = 0
    trades: list[Trade] = field(default_factory=list)


def run_day(date_str: str, day_candles: list[dict]) -> DayResult:
    res = DayResult(date=date_str)
    n = len(day_candles)
    htf_full = resample_15m(day_candles)
    atr = atr14(day_candles)

    or_high = or_low = None
    or_set = False
    active = False
    direction = None
    entry = tp = sl = 0.0
    entry_time = ""
    trades = 0
    consecutive_losses = 0
    cooldown_until_idx = -1
    HARD_CAP = cfg.max_trades + (1 if cfg.exceptional_score is not None else 0)

    sh, sm = cfg.orb_start
    eh, em = cfg.orb_end_mm
    efh, efm = cfg.entries_from
    tkh, tkm = cfg.theta_kill
    dsh, dsm = cfg.dead_start
    deh, dem = cfg.dead_end

    for idx in range(n):
        bar = day_candles[idx]
        hhmm = (bar["hh"], bar["mm"])
        candles_so_far = day_candles[: idx + 1]

        if not or_set:
            or_c = [c for c in candles_so_far if (sh, sm) <= (c["hh"], c["mm"]) <= (eh, em)]
            if len(or_c) >= 4:
                or_high = max(c["high"] for c in or_c)
                or_low = min(c["low"] for c in or_c)
                or_set = True

        if active:
            if direction == "CE":
                hit_tp, hit_sl = bar["high"] >= tp, bar["low"] <= sl
            else:
                hit_tp, hit_sl = bar["low"] <= tp, bar["high"] >= sl
            force_exit = hhmm >= (tkh, tkm) or idx == n - 1

            if hit_sl or hit_tp or force_exit:
                if hit_sl:
                    exit_px, reason = sl, "SL"
                elif hit_tp:
                    exit_px, reason = tp, "TP"
                else:
                    exit_px, reason = bar["close"], "EOD/theta"
                r_pts = (exit_px - entry) if direction == "CE" else (entry - exit_px)
                trades_list_entry = Trade(
                    date=date_str, direction=direction, entry_time=entry_time,
                    entry=entry, exit=exit_px, exit_reason=reason,
                    r_points=r_pts, score=last_score, reason=last_reason,
                )
                res.trades.append(trades_list_entry)
                if r_pts < 0:
                    consecutive_losses += 1
                    cooldown_until_idx = idx + (cfg.cooldown_loss_secs // 300)
                else:
                    consecutive_losses = 0
                    cooldown_until_idx = idx + (cfg.cooldown_win_secs // 300)
                active = False
            continue

        if consecutive_losses >= cfg.max_consecutive_losses:
            continue
        if (dsh, dsm) <= hhmm < (deh, dem):
            continue
        if hhmm < (efh, efm) or hhmm >= (tkh, tkm):
            continue
        if idx < cooldown_until_idx:
            continue
        if trades >= HARD_CAP:
            continue
        if idx < 12 or np.isnan(atr[idx]):
            continue

        htf_so_far = [h for h in htf_full if (h["hh"], h["mm"]) <= (bar["hh"], bar["mm"])]
        sig = score_signal(
            candles_so_far, htf_so_far,
            or_high=or_high, or_low=or_low, or_set=or_set,
            vol_mult=cfg.vol_mult,
        )
        if sig.blocked:
            res.signals_seen += 1
            res.blocked_volume += 1
            continue

        dirn = pick_direction(sig, cfg)
        if dirn is None:
            if sig.bull or sig.bear:
                res.signals_seen += 1
                res.blocked_gate += 1
            continue

        score = max(sig.bull, sig.bear)
        if trades >= cfg.max_trades:
            if cfg.exceptional_score is None or score < cfg.exceptional_score:
                continue

        if idx + 1 >= n:
            continue
        next_bar = day_candles[idx + 1]
        entry = next_bar["open"]
        entry_time = next_bar["time"]
        a = atr[idx]
        if dirn == "CE":
            tp, sl = entry + a * TP_MULT, entry - a * SL_MULT
        else:
            tp, sl = entry - a * TP_MULT, entry + a * SL_MULT
        direction = dirn
        last_score, last_reason = score, sig.reason_str
        active = True
        trades += 1

    return res


def main() -> None:
    df = pd.read_csv(CSV_PATH, parse_dates=["datetime"])
    df["date"] = df["datetime"].dt.date
    df["dow"] = df["datetime"].dt.dayofweek

    all_results: list[DayResult] = []
    for d, day_df in df.groupby("date"):
        if day_df["dow"].iloc[0] == cfg.skip_weekday:
            continue
        day_df = day_df.sort_values("datetime").reset_index(drop=True)
        candles = to_candle_dicts(day_df)
        all_results.append(run_day(str(d), candles))

    trades = [t for r in all_results for t in r.trades]
    days_with_trade = sum(1 for r in all_results if r.trades)
    total_signals = sum(r.signals_seen for r in all_results)
    total_vol_blocked = sum(r.blocked_volume for r in all_results)
    total_gate_blocked = sum(r.blocked_gate for r in all_results)

    wins = [t for t in trades if t.r_points > 0]
    losses = [t for t in trades if t.r_points <= 0]

    print(f"Days scanned (non-Tuesday): {len(all_results)}")
    print(f"Days with >=1 trade: {days_with_trade}")
    print(f"Total trades: {len(trades)}")
    print(f"Candidate signals seen: {total_signals} (blocked_volume={total_vol_blocked}, blocked_gate={total_gate_blocked})")
    print(f"Wins: {len(wins)}  Losses: {len(losses)}")
    if trades:
        win_rate = len(wins) / len(trades) * 100
        total_r = sum(t.r_points for t in trades)
        avg_r = total_r / len(trades)
        avg_win = sum(t.r_points for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t.r_points for t in losses) / len(losses) if losses else 0
        print(f"Win rate: {win_rate:.1f}%")
        print(f"Total index points captured: {total_r:+.1f}")
        print(f"Avg points/trade: {avg_r:+.2f}")
        print(f"Avg win: {avg_win:+.2f} pts | Avg loss: {avg_loss:+.2f} pts")
        ce = [t for t in trades if t.direction == "CE"]
        pe = [t for t in trades if t.direction == "PE"]
        print(f"CE trades: {len(ce)} (win {sum(1 for t in ce if t.r_points>0)}/{len(ce)})")
        print(f"PE trades: {len(pe)} (win {sum(1 for t in pe if t.r_points>0)}/{len(pe)})")
        by_reason = {"TP": 0, "SL": 0, "EOD/theta": 0}
        for t in trades:
            by_reason[t.exit_reason] = by_reason.get(t.exit_reason, 0) + 1
        print(f"Exit reasons: {by_reason}")

        print("\nAll trades:")
        for t in trades:
            print(f"  {t.date} {t.entry_time} {t.direction} entry={t.entry:.1f} exit={t.exit:.1f} "
                  f"({t.exit_reason}) r={t.r_points:+.1f} score={t.score} [{t.reason}]")


if __name__ == "__main__":
    main()
