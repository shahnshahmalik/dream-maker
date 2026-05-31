"""Dhan instrument resolution tests."""

from providers.dhan_instruments import resolve_market_data_instrument


def test_nifty_index_mapping():
    inst = resolve_market_data_instrument("NIFTY50IDX")
    assert inst is not None
    assert inst.security_id == "13"
    assert inst.exchange_segment == "IDX_I"
    assert inst.instrument == "INDEX"


def test_banknifty_mapping():
    inst = resolve_market_data_instrument("BANKNIFTY")
    assert inst is not None
    assert inst.security_id == "25"


def test_normalize_nifty_alias():
    inst = resolve_market_data_instrument(" nifty50idx ")
    assert inst is not None
    assert inst.security_id == "13"
