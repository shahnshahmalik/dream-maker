"""Config loading tests."""

import os

import pytest

from config import load_config


def test_load_config_requires_trading_symbol(monkeypatch):
    monkeypatch.delenv("TRADING_SYMBOL", raising=False)
    with pytest.raises(ValueError, match="TRADING_SYMBOL must be set"):
        load_config()


def test_load_config_rejects_non_fno(monkeypatch):
    monkeypatch.setenv("TRADING_SYMBOL", "NOTAVALIDSTOCKXYZ")
    monkeypatch.setenv("ACTIVE_BROKER", "dhan")
    with pytest.raises(ValueError, match="not F&O eligible"):
        load_config()


def test_load_config_rejects_groww_for_fno(monkeypatch):
    monkeypatch.setenv("TRADING_SYMBOL", "NIFTY50IDX")
    monkeypatch.setenv("ACTIVE_BROKER", "groww")
    with pytest.raises(ValueError, match="Groww does not support F&O"):
        load_config()


def test_load_config_single_symbol(monkeypatch):
    monkeypatch.setenv("TRADING_SYMBOL", "NIFTY50IDX")
    monkeypatch.setenv("ACTIVE_BROKER", "dhan")
    cfg = load_config()
    assert cfg.trading_symbol == "NIFTY50IDX"
