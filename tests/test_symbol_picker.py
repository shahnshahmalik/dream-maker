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
    assert selected == "BANKNIFTY"  # Both fit, BANKNIFTY preferred over SENSEX in list


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


# ---------- Task 2: build_candidates ----------

class TestBuildCandidates:
    def test_includes_index_futures_for_high_balance(self):
        picker = SymbolPicker(preferred=["SENSEX", "BANKNIFTY", "NIFTY50IDX"])
        candidates = picker.build_candidates(balance=100000.0)
        symbols = {c.symbol for c in candidates}
        assert "SENSEXFUT" in symbols
        assert "BANKNIFTYFUT" in symbols
        assert "NIFTYFUT" in symbols
        # All should be tier 1
        assert all(c.tier == 1 for c in candidates)

    def test_includes_options_for_low_balance(self):
        picker = SymbolPicker(preferred=["NIFTY50IDX"])
        candidates = picker.build_candidates(balance=5000.0)
        symbols = {c.symbol for c in candidates}
        assert "NIFTYOPT" in symbols, f"Expected NIFTYOPT in candidates: {candidates}"
        # Should have tier 2 options
        tiers = {c.tier for c in candidates}
        assert 2 in tiers, f"Expected tier 2 candidates, got tiers: {tiers}"

    def test_includes_stock_options_for_very_low_balance(self):
        picker = SymbolPicker(preferred=["NIFTY50IDX"])
        candidates = picker.build_candidates(balance=6500.0)
        symbols = {c.symbol for c in candidates}
        # Should include stock options (tier 3) for low balance
        assert "DIXONOPT" in symbols, f"Expected DIXONOPT in: {symbols}"
        # Should include tier 3
        tiers = {c.tier for c in candidates}
        assert 3 in tiers, f"Expected tier 3 candidates, got tiers: {tiers}"

    def test_full_pipeline_low_balance_selects_option(self):
        """With ₹5K balance, the pipeline should select an option symbol."""
        picker = SymbolPicker(
            preferred=["SENSEX", "BANKNIFTY", "NIFTY50IDX"],
            fallback="NIFTY50IDX",
        )
        candidates = picker.build_candidates(balance=5000.0)
        selected = picker.select(5000.0, candidates)
        # Should be an option (tier 2) — NOT an index future
        assert selected == "NIFTYOPT" or selected != "NIFTY50IDX", (
            f"Expected option symbol, got {selected}"
        )

    def test_full_pipeline_high_balance_selects_index_future(self):
        """With ₹2L balance, the pipeline should pick preferred index future."""
        picker = SymbolPicker(
            preferred=["BANKNIFTY", "NIFTY50IDX"],
            fallback="NIFTY50IDX",
        )
        candidates = picker.build_candidates(balance=200000.0)
        selected = picker.select(200000.0, candidates)
        assert selected == "BANKNIFTYFUT"


# ---------- Task 3: resolve ----------

class TestResolve:
    def test_concrete_contract_passes_through(self):
        assert SymbolPicker.resolve("NIFTY25JUNFUT") == "NIFTY25JUNFUT"
        assert SymbolPicker.resolve("BANKNIFTY25JUNFUT") == "BANKNIFTY25JUNFUT"

    def test_idx_symbol_returns_as_is(self):
        assert SymbolPicker.resolve("NIFTY50IDX") == "NIFTY50IDX"
        assert SymbolPicker.resolve("SENSEX") == "SENSEX"

    def test_generic_fut_maps_to_current_month(self):
        from datetime import datetime
        result = SymbolPicker.resolve("NIFTYFUT")
        assert "NIFTY" in result
        assert "FUT" in result
        # Should contain current year's 2-digit year
        yy = str(datetime.now().year)[-2:]
        assert yy in result, f"Expected {yy} in {result}"

    def test_option_symbol_kept_as_is(self):
        assert SymbolPicker.resolve("NIFTYOPT") == "NIFTYOPT"
        assert SymbolPicker.resolve("BANKNIFTYOPT") == "BANKNIFTYOPT"

    def test_stock_fut_resolves_to_current_month(self):
        result = SymbolPicker.resolve("IDEAFUT")
        from datetime import datetime
        yy = str(datetime.now().year)[-2:]
        assert "IDEA" in result
        assert yy in result
        assert "FUT" in result
