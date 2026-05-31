"""F&O symbol eligibility tests."""

import pytest

from utils.symbols import is_fno_eligible, normalize_symbol, require_fno_symbol


@pytest.mark.parametrize(
    "symbol",
    [
        "NIFTY50IDX",
        "NIFTY25JUNFUT",
        "BANKNIFTY",
        "RELIANCE",
        "RELIANCE24JUNFUT",
        "NIFTY24100CE",
    ],
)
def test_fno_eligible_symbols(symbol: str):
    assert is_fno_eligible(symbol)


@pytest.mark.parametrize("symbol", ["RANDOMSTOCK", "ABC", "GOLD"])
def test_non_fno_symbols_rejected(symbol: str):
    assert not is_fno_eligible(symbol)


def test_normalize_symbol():
    assert normalize_symbol(" nifty50idx ") == "NIFTY50IDX"


def test_require_fno_symbol_raises():
    with pytest.raises(ValueError, match="not F&O eligible"):
        require_fno_symbol("NOTAVALIDSTOCKXYZ")


def test_require_fno_symbol_accepts():
    assert require_fno_symbol("NIFTY50IDX") == "NIFTY50IDX"
