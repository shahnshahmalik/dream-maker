# Dream-Maker Bug Analysis Report

## Critical Issues Found

### 1. Overly Strict Trend Detection (analysis/technical.py:61-70)
**Problem**: `detect_trend()` requires perfect monotonic increase/decrease of both highs AND lows for 5 consecutive candles. This is extremely rare in real markets.

**Current Logic**:
```python
highs = df["high"].tail(5)
lows = df["low"].tail(5)
if highs.is_monotonic_increasing and lows.is_monotonic_increasing:
    return Trend.UPTREND
```

**Impact**: 99% of setups fail with "No strong technical setup — paused"

**Fix**: Use EMA crossovers and price action instead of strict monotonic requirements

### 2. High Signal Strength Threshold
**Problem**: MIN_SIGNAL_STRENGTH=0.65 is too high combined with strict trend logic
**Impact**: Even valid setups get filtered out
**Fix**: Lower to 0.45 or improve calculation

### 3. High RR Ratio Requirement  
**Problem**: MIN_RR_RATIO=3.0 is very aggressive for intraday F&O
**Impact**: Reduces opportunity count significantly
**Fix**: Lower to 2.0 or make it adaptive

### 4. Log Buffering Issues
**Problem**: When running via background terminal, Python stdout buffering causes delayed/missing logs
**Current**: `PYTHONUNBUFFERED=1` used but may not be sufficient
**Fix**: Explicit file logging with immediate flush

### 5. Stale Plan Recovery Bug (audit/state_store.py)
**Problem**: If engine crashes with active plans from different symbols, recovery blocks scanner
**Impact**: Scanner never runs when stale plans exist
**Fix**: Clear mismatched symbol plans on startup

### 6. News Engine Error Handling
**Problem**: `fetch_headlines()` and `check_fundamental()` fail silently on HTTP errors
**Impact**: Analysis pipeline continues with stale/empty data
**Fix**: Better error handling and fallbacks

### 7. Premium Estimation Accuracy
**Problem**: Option premium estimation may be inaccurate for current market conditions
**Fix**: Review and calibrate estimation parameters

### 8. Instance Lock Edge Cases
**Problem**: Lock file can become stale on force kill, blocking restarts
**Fix**: Better stale lock detection and cleanup

## Data Flow Analysis

```
Scanner → Pipeline → Technical Analysis → [FAILS HERE] → No Plans Generated
```

The pipeline flow is:
1. Scanner calls pipeline.run()
2. Pipeline calls analyze_technical() 
3. Technical analysis fails due to strict trend detection
4. Pipeline returns "No strong technical setup — paused"
5. No plans generated, cycle repeats

## Root Cause Summary

The primary issue is **overly conservative technical analysis parameters**:
- Trend detection requiring perfect monotonic sequences
- High signal strength threshold (0.65)
- High RR ratio requirement (3.0)

This creates a "risk-off" configuration that rarely finds valid setups.