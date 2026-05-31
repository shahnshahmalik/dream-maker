"""Pytest configuration — default TRADING_SYMBOL for tests."""

from __future__ import annotations

import os

os.environ.setdefault("TRADING_SYMBOL", "NIFTY50IDX")
os.environ.setdefault("ACTIVE_BROKER", "dhan")
