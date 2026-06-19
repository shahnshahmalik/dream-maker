"""Indicator backtest — NIFTY 5-min Jan–Jun 2026.

Tests 11 individual indicators, all pairwise AND combos, all triple AND combos,
then repeats the best results with the day-type filter applied.

Entry rules:
  - Signal fires on candle close → enter at NEXT candle open
  - TP = entry ± ATR(14) × 2.0  |  SL = entry ∓ ATR(14) × 1.0  (1:2 R:R)
  - Exit at close of bar that first hits TP or SL (worst case = EOD close)
  - No new entries after 14:30 IST
  - One trade at a time — no pyramiding

Day-type filter (derived from playbook analysis):
  - Range/Inside Day (OR range < 0.8%) → skip ALL signals
  - V-Reversal Bull / Trend-Up / Gap-Down-Rally → LONG signals only
  - V-Reversal Bear / Trend-Down / Gap-Down-Trend / Gap-Up-Trend → SHORT only

Run: python scripts/backtest_indicators.py
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# ── Constants ────────────────────────────────────────────────────────

CSV_PATH = Path(__file__).parent.parent / "data" / "NIFTY_5min_Jan_Jun_2026.csv"
ATR_PERIOD = 14
TP_MULT = 2.0
SL_MULT = 1.0
NO_ENTRY_AFTER = "14:30"
MIN_ATR = 5.0          # skip signals when market dead (NIFTY points)
MAX_SHOWN = 15         # top N to display per table

# Day-type filter thresholds
_GAP_THRESH = 0.003        # 0.3% = meaningful gap
_FULL_DAY_RANGE_THRESH = 0.008   # full day range < 0.8% → Range/Inside Day
_OR_BARS = 3               # 3 × 5m = 15-min OR


# ── Data loading + day metadata ──────────────────────────────────────

def load_data() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH, parse_dates=["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["time_str"] = df["time"].astype(str).str.strip()
    df["is_market_hour"] = df["time_str"] <= NO_ENTRY_AFTER
    return df


def build_day_meta(df: pd.DataFrame) -> pd.DataFrame:
    """Per-day: prev_close, full-day range, OR structure, allowed_direction.

    allowed_direction: +1=LONG only, -1=SHORT only, 0=blocked (Range day), 99=any
    Range/Inside Day: full day range (high-low) < 0.8% of prev_close.
    Direction lock: based on 15-min OR HH/HL vs LH/LL structure.
    """
    rows = []
    dates = sorted(df["date"].unique())
    date_to_close: dict = {}

    for d in dates:
        day = df[df["date"] == d].reset_index(drop=True)
        prev_date = dates[max(0, dates.index(d) - 1)]
        prev_close = date_to_close.get(prev_date, day["open"].iloc[0])

        # Full-day range (used for Range/Inside Day classification)
        day_high = day["high"].max()
        day_low = day["low"].min()
        day_range_pct = (day_high - day_low) / prev_close if prev_close else 0.0

        # Opening range structure (first 15 min = 3 bars)
        or_bars = day.iloc[:_OR_BARS]
        or_highs = or_bars["high"].values
        or_lows = or_bars["low"].values
        hh = all(or_highs[i] >= or_highs[i - 1] for i in range(1, len(or_highs)))
        hl = all(or_lows[i] >= or_lows[i - 1] for i in range(1, len(or_lows)))
        bullish_structure = hh or hl

        today_open = day["open"].iloc[0]
        gap_pct = (today_open - prev_close) / prev_close if prev_close else 0.0

        # Classify
        if day_range_pct < _FULL_DAY_RANGE_THRESH:
            allowed = 0   # Range/Inside → blocked, 25% DirWR
        elif abs(gap_pct) >= _GAP_THRESH:
            if gap_pct < 0:
                allowed = 1 if bullish_structure else -1  # gap down rally vs trend
            else:
                allowed = 1   # gap up → follow gap long
        else:
            allowed = 1 if bullish_structure else -1  # V-Reversal direction

        date_to_close[d] = day["close"].iloc[-1]
        rows.append({"date": d, "allowed_direction": allowed, "day_range_pct": day_range_pct})

    return pd.DataFrame(rows)


# ── ATR ──────────────────────────────────────────────────────────────

def compute_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    h, l, pc = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


# ── Signal generators ────────────────────────────────────────────────

def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def signal_ema_cross(df: pd.DataFrame, fast: int, slow: int) -> pd.Series:
    cross = _ema(df["close"], fast) - _ema(df["close"], slow)
    prev = cross.shift(1)
    sig = pd.Series(0, index=df.index)
    crossed = ((cross > 0) & (prev <= 0)) | ((cross < 0) & (prev >= 0))
    sig[crossed & (cross > 0)] = 1
    sig[crossed & (cross < 0)] = -1
    return sig


def signal_rsi(df: pd.DataFrame, period: int = 14,
               ob: float = 65.0, os: float = 35.0) -> pd.Series:
    delta = df["close"].diff()
    avg_g = delta.clip(lower=0).ewm(span=period, adjust=False).mean()
    avg_l = (-delta).clip(lower=0).ewm(span=period, adjust=False).mean()
    rsi = 100 - (100 / (1 + avg_g / avg_l.replace(0, np.nan)))
    prev = rsi.shift(1)
    sig = pd.Series(0, index=df.index)
    sig[(rsi > os) & (prev <= os)] = 1
    sig[(rsi < ob) & (prev >= ob)] = -1
    return sig


def signal_macd(df: pd.DataFrame,
                fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    macd = _ema(df["close"], fast) - _ema(df["close"], slow)
    sig_line = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - sig_line
    prev = hist.shift(1)
    sig = pd.Series(0, index=df.index)
    sig[(hist > 0) & (prev <= 0)] = 1
    sig[(hist < 0) & (prev >= 0)] = -1
    return sig


def signal_bb_breakout(df: pd.DataFrame,
                        period: int = 20, std: float = 2.0) -> pd.Series:
    mid = df["close"].rolling(period).mean()
    band = std * df["close"].rolling(period).std()
    upper, lower = mid + band, mid - band
    pc, pu, pl = df["close"].shift(1), upper.shift(1), lower.shift(1)
    sig = pd.Series(0, index=df.index)
    sig[(df["close"] > upper) & (pc <= pu)] = 1
    sig[(df["close"] < lower) & (pc >= pl)] = -1
    return sig


def signal_supertrend(df: pd.DataFrame,
                       period: int = 10, mult: float = 3.0) -> pd.Series:
    atr = compute_atr(df, period)
    mid = (df["high"] + df["low"]) / 2
    ub_raw = (mid + mult * atr).values
    lb_raw = (mid - mult * atr).values
    close = df["close"].values

    ub = ub_raw.copy()
    lb = lb_raw.copy()
    direction = np.ones(len(df), dtype=int)
    st = np.full(len(df), np.nan)

    for i in range(1, len(df)):
        lb[i] = lb_raw[i] if (lb_raw[i] > lb[i-1] or close[i-1] < lb[i-1]) else lb[i-1]
        ub[i] = ub_raw[i] if (ub_raw[i] < ub[i-1] or close[i-1] > ub[i-1]) else ub[i-1]
        if np.isnan(st[i-1]):
            st[i] = lb[i]; direction[i] = 1
        elif st[i-1] == ub[i-1]:
            st[i], direction[i] = (lb[i], 1) if close[i] > ub[i] else (ub[i], -1)
        else:
            st[i], direction[i] = (ub[i], -1) if close[i] < lb[i] else (lb[i], 1)

    d = pd.Series(direction, index=df.index)
    prev_d = d.shift(1)
    sig = pd.Series(0, index=df.index)
    sig[(d == 1) & (prev_d == -1)] = 1
    sig[(d == -1) & (prev_d == 1)] = -1
    return sig


def signal_vwap_cross(df: pd.DataFrame) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    cum_tpv = df.groupby("date", group_keys=False).apply(
        lambda g: (tp.loc[g.index] * g["volume"]).cumsum()
    )
    cum_v = df.groupby("date", group_keys=False)["volume"].cumsum()
    vwap = cum_tpv / cum_v
    above = (df["close"] > vwap).astype(bool)
    prev = above.shift(1).fillna(False).astype(bool)
    sig = pd.Series(0, index=df.index)
    sig[above & ~prev] = 1
    sig[~above & prev] = -1
    return sig


def signal_orb(df: pd.DataFrame, or_min: int = 15) -> pd.Series:
    n = or_min // 5
    or_h = df.groupby("date", group_keys=False).apply(
        lambda g: pd.Series(g["high"].iloc[:n].max(), index=g.index)
    ).rename("or_high")
    or_l = df.groupby("date", group_keys=False).apply(
        lambda g: pd.Series(g["low"].iloc[:n].min(), index=g.index)
    ).rename("or_low")
    df2 = df.copy()
    df2["or_high"] = or_h
    df2["or_low"] = or_l
    after_or = df2["time_str"] > "09:2"
    pc = df2["close"].shift(1)
    sig = pd.Series(0, index=df.index)
    sig[after_or & (df2["close"] > df2["or_high"]) & (pc <= df2["or_high"].shift(1).fillna(df2["or_high"]))] = 1
    sig[after_or & (df2["close"] < df2["or_low"]) & (pc >= df2["or_low"].shift(1).fillna(df2["or_low"]))] = -1
    return sig


def signal_stochastic(df: pd.DataFrame,
                       k: int = 14, d: int = 3,
                       ob: float = 80.0, os: float = 20.0) -> pd.Series:
    lo = df["low"].rolling(k).min()
    hi = df["high"].rolling(k).max()
    K = 100 * (df["close"] - lo) / (hi - lo).replace(0, np.nan)
    D = K.rolling(d).mean()
    pK, pD = K.shift(1), D.shift(1)
    sig = pd.Series(0, index=df.index)
    sig[(K > D) & (pK <= pD) & (K < ob)] = 1
    sig[(K < D) & (pK >= pD) & (K > os)] = -1
    return sig


def combine(*signals: pd.Series) -> pd.Series:
    """AND of N signals — all must agree on direction."""
    result = signals[0].copy()
    for s in signals[1:]:
        # LONG only if all are LONG, SHORT only if all are SHORT, else 0
        result = pd.Series(
            np.where((result == 1) & (s == 1), 1,
            np.where((result == -1) & (s == -1), -1, 0)),
            index=result.index,
        )
    return result


# ── Trade simulation ─────────────────────────────────────────────────

@dataclass
class Trade:
    direction: int
    entry: float
    tp: float
    sl: float
    entry_bar: int
    exit_price: float = 0.0
    exit_bar: int = 0
    pnl: float = 0.0
    hit_tp: bool = False
    hit_sl: bool = False
    timed_out: bool = False


@dataclass
class Result:
    name: str
    trades: list[Trade] = field(default_factory=list)

    @property
    def n(self) -> int: return len(self.trades)

    @property
    def wr(self) -> float:
        return len([t for t in self.trades if t.pnl > 0]) / self.n * 100 if self.n else 0.0

    @property
    def pf(self) -> float:
        gw = sum(t.pnl for t in self.trades if t.pnl > 0)
        gl = abs(sum(t.pnl for t in self.trades if t.pnl <= 0))
        return gw / gl if gl else (float("inf") if gw > 0 else 0.0)

    @property
    def total(self) -> float: return sum(t.pnl for t in self.trades)

    @property
    def expect(self) -> float:
        if not self.trades: return 0.0
        wr = self.wr / 100
        aw = sum(t.pnl for t in self.trades if t.pnl > 0) / max(len([t for t in self.trades if t.pnl > 0]), 1)
        al = abs(sum(t.pnl for t in self.trades if t.pnl <= 0) / max(len([t for t in self.trades if t.pnl <= 0]), 1))
        return wr * aw - (1 - wr) * al

    @property
    def maxdd(self) -> float:
        if not self.trades: return 0.0
        cum = np.cumsum([t.pnl for t in self.trades])
        return float((cum - np.maximum.accumulate(cum)).min())

    def row(self) -> dict:
        return {"Name": self.name, "N": self.n, "WR%": round(self.wr, 1),
                "PF": round(self.pf, 2), "Expect": round(self.expect, 1),
                "TotalPnL": round(self.total, 0), "MaxDD": round(self.maxdd, 0)}


def simulate(
    df: pd.DataFrame,
    signal: pd.Series,
    name: str,
    day_filter: Optional[pd.Series] = None,  # per-bar allowed_direction
) -> Result:
    result = Result(name=name)
    atr = df["atr"].values
    opens = df["open"].values
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    is_mkt = df["is_market_hour"].values
    dates = df["date"].values
    sig = signal.values
    allowed = day_filter.values if day_filter is not None else np.full(len(df), 99)

    in_trade = False
    active: Trade | None = None

    for i in range(1, len(df)):
        if in_trade and active is not None:
            # Force close at end of day
            if dates[i] != dates[active.entry_bar]:
                active.exit_price = closes[i - 1]
                active.exit_bar = i - 1
                active.pnl = (active.exit_price - active.entry) * active.direction
                active.timed_out = True
                result.trades.append(active)
                in_trade = False
                active = None
            else:
                hit_tp = (active.direction == 1 and highs[i] >= active.tp) or \
                         (active.direction == -1 and lows[i] <= active.tp)
                hit_sl = (active.direction == 1 and lows[i] <= active.sl) or \
                         (active.direction == -1 and highs[i] >= active.sl)
                if hit_tp or hit_sl:
                    if hit_sl:
                        active.exit_price = active.sl
                        active.hit_sl = True
                    else:
                        active.exit_price = active.tp
                        active.hit_tp = True
                    active.exit_bar = i
                    active.pnl = (active.exit_price - active.entry) * active.direction
                    result.trades.append(active)
                    in_trade = False
                    active = None
                continue

        if not in_trade and sig[i - 1] != 0 and is_mkt[i]:
            direction = int(sig[i - 1])
            bar_allowed = int(allowed[i])

            # Day-type filter: 0=blocked, 99=any, else must match direction
            if bar_allowed == 0:
                continue
            if bar_allowed != 99 and bar_allowed != direction:
                continue
            if atr[i - 1] < MIN_ATR:
                continue

            entry = opens[i]
            tp = entry + direction * TP_MULT * atr[i - 1]
            sl = entry - direction * SL_MULT * atr[i - 1]
            active = Trade(direction=direction, entry=entry, tp=tp, sl=sl, entry_bar=i)
            in_trade = True

    if in_trade and active is not None:
        active.exit_price = closes[-1]
        active.exit_bar = len(df) - 1
        active.pnl = (active.exit_price - active.entry) * active.direction
        active.timed_out = True
        result.trades.append(active)

    return result


# ── Indicator registry ───────────────────────────────────────────────

def build_indicators(df: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "EMA 9/21":    signal_ema_cross(df, 9, 21),
        "EMA 9/50":    signal_ema_cross(df, 9, 50),
        "EMA 20/50":   signal_ema_cross(df, 20, 50),
        "RSI 35/65":   signal_rsi(df, 14, ob=65, os=35),
        "RSI 30/70":   signal_rsi(df, 14, ob=70, os=30),
        "MACD":        signal_macd(df),
        "BB(20,2)":    signal_bb_breakout(df),
        "Supert(10,3)":signal_supertrend(df),
        "VWAP":        signal_vwap_cross(df),
        "ORB 15m":     signal_orb(df, 15),
        "Stoch(14,3)": signal_stochastic(df),
    }


# ── Reporting ────────────────────────────────────────────────────────

def print_table(results: list[Result], title: str, min_trades: int = 5) -> None:
    valid = [r for r in results if r.n >= min_trades]
    if not valid:
        print(f"\n{title}\n  (no results with ≥{min_trades} trades)\n")
        return

    valid.sort(key=lambda r: r.pf, reverse=True)
    top = valid[:MAX_SHOWN]

    name_w = max(max(len(r.name) for r in top), 30)
    hdr = (f"{'#':>3}  {'Indicator':<{name_w}}  {'N':>5}  {'WR%':>6}  "
           f"{'PF':>5}  {'Expect':>7}  {'TotalPnL':>9}  {'MaxDD':>8}")
    sep = "─" * len(hdr)

    print(f"\n{'═' * len(hdr)}")
    print(f"  {title}")
    print(f"{'═' * len(hdr)}")
    print(hdr)
    print(sep)

    for rank, r in enumerate(top, 1):
        pf_s = f"{r.pf:.2f}" if r.pf != float("inf") else " INF"
        print(f"{rank:>3}  {r.name:<{name_w}}  {r.n:>5}  {r.wr:>5.1f}%  "
              f"{pf_s:>5}  {r.expect:>+7.1f}  {int(r.total):>+9}  {int(r.maxdd):>8}")

    # Highlight standout
    best = top[0]
    print(f"\n  ★ Best: {best.name}  |  N={best.n}  WR={best.wr:.1f}%  "
          f"PF={best.pf:.2f}  Expect={best.expect:+.1f} pts/trade")


# ── Main ─────────────────────────────────────────────────────────────

from typing import Optional


def main() -> None:
    print("Loading data…", end=" ", flush=True)
    df = load_data()
    df["atr"] = compute_atr(df)
    n_days = df["date"].nunique()
    print(f"{len(df):,} bars | {n_days} trading days")

    print("Building day metadata…", end=" ", flush=True)
    day_meta = build_day_meta(df)
    df = df.merge(day_meta[["date", "allowed_direction"]], on="date", how="left")
    range_days = (day_meta["allowed_direction"] == 0).sum()
    print(f"{range_days} Range/Inside days filtered ({range_days/n_days*100:.1f}%)")

    print("Computing indicator signals…", end=" ", flush=True)
    indicators = build_indicators(df)
    print(f"{len(indicators)} indicators")

    allowed_col = df["allowed_direction"]  # per-bar series

    # ════════════════════════════════════════════════════
    # PASS 1 — No day filter
    # ════════════════════════════════════════════════════
    print("\n── Pass 1: No day-type filter ──")

    # Individual
    ind_raw = [simulate(df, sig, name) for name, sig in indicators.items()]
    print_table(ind_raw, "INDIVIDUAL  (no filter)", min_trades=5)

    # Pairs
    print("  Running pairs…", end=" ", flush=True)
    names = list(indicators.keys())
    pairs_raw = [
        simulate(df, combine(indicators[a], indicators[b]), f"{a} + {b}")
        for a, b in itertools.combinations(names, 2)
    ]
    print(f"{len(pairs_raw)} pairs")
    print_table(pairs_raw, "PAIRS  (no filter)", min_trades=5)

    # Triples
    print("  Running triples…", end=" ", flush=True)
    triples_raw = [
        simulate(df, combine(indicators[a], indicators[b], indicators[c]), f"{a} + {b} + {c}")
        for a, b, c in itertools.combinations(names, 3)
    ]
    print(f"{len(triples_raw)} triples")
    print_table(triples_raw, "TRIPLES  (no filter)", min_trades=5)

    # ════════════════════════════════════════════════════
    # PASS 2 — With day-type filter
    # ════════════════════════════════════════════════════
    print("\n── Pass 2: With day-type filter ──")

    # Individual
    ind_filt = [simulate(df, sig, name, day_filter=allowed_col) for name, sig in indicators.items()]
    print_table(ind_filt, "INDIVIDUAL  (day filter)", min_trades=3)

    # Pairs
    print("  Running pairs…", end=" ", flush=True)
    pairs_filt = [
        simulate(df, combine(indicators[a], indicators[b]), f"{a} + {b}", day_filter=allowed_col)
        for a, b in itertools.combinations(names, 2)
    ]
    print(f"{len(pairs_filt)} pairs")
    print_table(pairs_filt, "PAIRS  (day filter)", min_trades=3)

    # Triples
    print("  Running triples…", end=" ", flush=True)
    triples_filt = [
        simulate(df, combine(indicators[a], indicators[b], indicators[c]),
                 f"{a} + {b} + {c}", day_filter=allowed_col)
        for a, b, c in itertools.combinations(names, 3)
    ]
    print(f"{len(triples_filt)} triples")
    print_table(triples_filt, "TRIPLES  (day filter)", min_trades=3)

    # ════════════════════════════════════════════════════
    # Final summary
    # ════════════════════════════════════════════════════
    all_raw = ind_raw + pairs_raw + triples_raw
    all_filt = ind_filt + pairs_filt + triples_filt

    best_raw = max((r for r in all_raw if r.n >= 5), key=lambda r: r.pf, default=None)
    best_filt = max((r for r in all_filt if r.n >= 3), key=lambda r: r.pf, default=None)

    print(f"""
{'═' * 72}
  FINAL SUMMARY
{'═' * 72}

  WITHOUT day filter:
    Best  : {best_raw.name if best_raw else 'N/A'}
            N={best_raw.n if best_raw else 0}  WR={best_raw.wr:.1f}%  PF={best_raw.pf:.2f}  Expect={best_raw.expect:+.1f} pts

  WITH day filter:
    Best  : {best_filt.name if best_filt else 'N/A'}
            N={best_filt.n if best_filt else 0}  WR={best_filt.wr:.1f}%  PF={best_filt.pf:.2f}  Expect={best_filt.expect:+.1f} pts

  Setup : TP={TP_MULT}×ATR | SL={SL_MULT}×ATR | ATR({ATR_PERIOD}) | no entry after {NO_ENTRY_AFTER}
  Data  : Jan–Jun 2026 | {n_days} trading days | {len(df):,} 5-min bars
  PnL   : NIFTY index points (multiply by option delta ~0.4–0.8 for premium)
{'═' * 72}
""")


if __name__ == "__main__":
    main()
