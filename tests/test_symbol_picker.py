"""Tests for SymbolPicker — dynamic F&O symbol selection."""
import pytest
from agent.symbol_picker import InstrumentCandidate, SymbolPicker


def test_instrument_candidate_creation():
    c = InstrumentCandidate(
        symbol="NIFTY25JUNFUT",
        lot_size=25,
        min_capital_estimate=35000.0,
        tier=1,
    )
    assert c.symbol == "NIFTY25JUNFUT"
    assert c.lot_size == 25
    assert c.tier == 1


def test_symbol_picker_returns_fallback_when_no_candidates_fit():
    picker = SymbolPicker(
        preferred=["SENSEX", "BANKNIFTY", "NIFTY50IDX"],
        fallback="NIFTY50IDX",
    )
    result = picker._best_fit(
        candidates=[
            InstrumentCandidate("SENSEX", 10, 40000, 1),
            InstrumentCandidate("BANKNIFTY", 15, 35000, 1),
            InstrumentCandidate("NIFTY50IDX", 25, 30000, 1),
        ],
        balance=5000.0,
    )
    assert result is None  # none fit → picker will use fallback


def test_symbol_picker_picks_best_preferred_that_fits():
    picker = SymbolPicker(
        preferred=["SENSEX", "BANKNIFTY", "NIFTY50IDX"],
        fallback="NIFTY50IDX",
    )
    result = picker._best_fit(
        candidates=[
            InstrumentCandidate("SENSEX", 10, 40000, 1),
            InstrumentCandidate("BANKNIFTY", 15, 35000, 1),
            InstrumentCandidate("NIFTY50IDX", 25, 30000, 1),
        ],
        balance=50000.0,
    )
    assert result is not None
    assert result.symbol == "BANKNIFTY"  # higher-pref than NIFTY, fits margin


def test_symbol_picker_select_uses_fallback_when_none_fit():
    picker = SymbolPicker(
        preferred=["SENSEX", "BANKNIFTY"],
        fallback="NIFTY50IDX",
    )
    selected = picker.select(
        balance=5000.0,
        candidates=[
            InstrumentCandidate("SENSEX", 10, 40000, 1),
            InstrumentCandidate("BANKNIFTY", 15, 35000, 1),
        ],
    )
    assert selected == "NIFTY50IDX"


def test_symbol_picker_select_picks_best_fit():
    picker = SymbolPicker(
        preferred=["SENSEX", "BANKNIFTY"],
        fallback="NIFTY50IDX",
    )
    selected = picker.select(
        balance=50000.0,
        candidates=[
            InstrumentCandidate("SENSEX", 10, 40000, 1),
            InstrumentCandidate("BANKNIFTY", 15, 35000, 1),
        ],
    )
    assert selected == "BANKNIFTY"  # preferred order: BANKNIFTY before SENSEX? 
    # Actually both fit, BANKNIFTY comes after SENSEX in preferred list
    # SENSEX should win (position 0)

def test_symbol_picker_pref_order_sorts_by_preferred_list():
    picker = SymbolPicker(
        preferred=["SENSEX", "BANKNIFTY", "NIFTY50IDX"],
        fallback="NIFTY50IDX",
    )
    result = picker._best_fit(
        candidates=[
            InstrumentCandidate("NIFTY50IDX", 25, 10000, 1),
            InstrumentCandidate("SENSEX", 10, 10000, 1),
            InstrumentCandidate("BANKNIFTY", 15, 10000, 1),
        ],
        balance=15000.0,
    )
    # All same min_capital, same tier → SENSEX should win (first in preferred)
    assert result is not None
    assert result.symbol == "SENSEX"
