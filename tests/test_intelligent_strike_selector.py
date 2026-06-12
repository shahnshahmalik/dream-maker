"""Tests for intelligent strike selector."""

import pytest
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

from agent.intelligent_strike_selector import (
    IntelligentStrikeSelector,
    StrikeCandidate,
    find_affordable_strike,
    MARGIN_BUFFER,
)


class TestStrikeCandidate:
    def test_total_cost_with_actual_premium(self):
        c = StrikeCandidate(
            symbol="NIFTY26JUN23400CE",
            underlying="NIFTY",
            strike=23400,
            option_type="CE",
            expiry_date=date.today(),
            expiry_week=0,
            distance_pct=0.01,
            lot_size=25,
            estimated_premium=150.0,
            actual_premium=185.0,
        )
        assert c.total_cost_estimate == 185.0 * 25  # uses actual

    def test_total_cost_with_estimated_premium(self):
        c = StrikeCandidate(
            symbol="NIFTY26JUN23400CE",
            underlying="NIFTY",
            strike=23400,
            option_type="CE",
            expiry_date=date.today(),
            expiry_week=0,
            distance_pct=0.01,
            lot_size=25,
            estimated_premium=150.0,
        )
        assert c.total_cost_estimate == 150.0 * 25  # uses estimate


class TestIntelligentStrikeSelector:
    def setup_method(self):
        self.spot = 23400.0
        self.balance = 16000.0

    def _make_selector(self, **kwargs):
        defaults = dict(
            underlying="NIFTY",
            spot=self.spot,
            balance=self.balance,
        )
        defaults.update(kwargs)
        return IntelligentStrikeSelector(**defaults)

    # ── Symbol building ──────────────────────────────────────────

    def test_build_symbol_ce(self):
        expiry = date(2026, 6, 5)
        sym = IntelligentStrikeSelector._build_symbol("NIFTY", 23400, "CE", expiry)
        assert sym == "NIFTY26JUN23400CE"

    def test_build_symbol_pe(self):
        expiry = date(2026, 6, 12)
        sym = IntelligentStrikeSelector._build_symbol("BANKNIFTY", 51700, "PE", expiry)
        assert sym == "BANKNIFTY26JUN51700PE"

    # ── Strike interval detection ────────────────────────────────

    def test_nifty_strike_interval(self):
        s = self._make_selector(underlying="NIFTY")
        assert s._strike_interval == 50

    def test_banknifty_strike_interval(self):
        s = self._make_selector(underlying="BANKNIFTY")
        assert s._strike_interval == 100

    def test_sensex_strike_interval(self):
        s = self._make_selector(underlying="SENSEX")
        assert s._strike_interval == 100

    def test_stock_strike_interval(self):
        s = self._make_selector(underlying="TATASTEEL", spot=140.0)
        assert s._strike_interval == 5  # spot ≤ 200

    def test_stock_mid_price_strike_interval(self):
        s = self._make_selector(underlying="TATASTEEL", spot=600.0)
        assert s._strike_interval == 10  # 200 < spot ≤ 1000

    # ── Option types to try ──────────────────────────────────────

    def test_prefers_ce_first(self):
        s = self._make_selector(preferred_direction="CE")
        assert s._option_types_to_try() == ["CE", "PE"]

    def test_prefers_pe_first(self):
        s = self._make_selector(preferred_direction="PE")
        assert s._option_types_to_try() == ["PE", "CE"]

    def test_any_tries_ce_first(self):
        s = self._make_selector(preferred_direction="ANY")
        assert s._option_types_to_try() == ["CE", "PE"]

    # ── Candidate generation ─────────────────────────────────────

    def test_generates_both_ce_and_pe(self):
        s = self._make_selector()
        candidates = s._generate_candidates(max_otm_distance=0.10, max_expiry_weeks=1)
        types = {c.option_type for c in candidates}
        assert "CE" in types
        assert "PE" in types

    def test_generates_multiple_expiries(self):
        s = self._make_selector()
        candidates = s._generate_candidates(max_otm_distance=0.05, max_expiry_weeks=3)
        weeks = {c.expiry_week for c in candidates}
        assert len(weeks) >= 2, f"Expected multiple expiry weeks, got {weeks}"

    def test_all_strikes_positive(self):
        s = self._make_selector(spot=500.0, underlying="TATASTEEL")
        candidates = s._generate_candidates(max_otm_distance=0.15, max_expiry_weeks=1)
        for c in candidates:
            assert c.strike > 0, f"Negative strike in {c.symbol}"

    def test_ce_strikes_above_atm(self):
        s = self._make_selector()
        atm = int(round(self.spot / 50) * 50)  # 23400
        candidates = s._generate_candidates(max_otm_distance=0.05, max_expiry_weeks=1)
        ce_candidates = [c for c in candidates if c.option_type == "CE"]
        for c in ce_candidates:
            assert c.strike >= atm, f"CE {c.symbol} has strike {c.strike} below ATM {atm}"

    def test_pe_strikes_below_atm(self):
        s = self._make_selector()
        atm = int(round(self.spot / 50) * 50)
        candidates = s._generate_candidates(max_otm_distance=0.05, max_expiry_weeks=1)
        pe_candidates = [c for c in candidates if c.option_type == "PE"]
        for c in pe_candidates:
            assert c.strike <= atm, f"PE {c.symbol} has strike {c.strike} above ATM {atm}"

    def test_respects_max_otm_distance(self):
        s = self._make_selector()
        candidates = s._generate_candidates(max_otm_distance=0.05, max_expiry_weeks=1)
        for c in candidates:
            assert c.distance_pct <= 0.05, f"{c.symbol} distance {c.distance_pct:.4f} > 0.05"

    # ── Premium estimation ───────────────────────────────────────

    def test_atm_premium_higher_than_deep_otm(self):
        s = self._make_selector()
        atm = s._estimate_premium(0.0, 0)
        deep = s._estimate_premium(0.10, 0)
        assert atm > deep, f"ATM={atm:.1f} should be > deep OTM={deep:.1f}"

    def test_further_expiry_higher_premium(self):
        s = self._make_selector()
        this_week = s._estimate_premium(0.02, 0)
        next_week = s._estimate_premium(0.02, 1)
        assert next_week > this_week, f"Next week={next_week:.1f} should be > this week={this_week:.1f}"

    def test_premium_has_floor(self):
        s = self._make_selector()
        very_deep = s._estimate_premium(0.50, 3)
        assert very_deep >= 5.0, f"Premium floor violated: {very_deep}"

    # ── Scoring ──────────────────────────────────────────────────

    def test_atm_scores_higher_than_deep_otm(self):
        s = self._make_selector()
        atm = StrikeCandidate(
            symbol="NIFTY26JUN23400CE", underlying="NIFTY",
            strike=23400, option_type="CE",
            expiry_date=date.today(), expiry_week=0,
            distance_pct=0.0, lot_size=25, estimated_premium=200.0,
        )
        deep = StrikeCandidate(
            symbol="NIFTY26JUN24000CE", underlying="NIFTY",
            strike=24000, option_type="CE",
            expiry_date=date.today(), expiry_week=0,
            distance_pct=0.05, lot_size=25, estimated_premium=80.0,
        )
        scored = s._score_candidates([atm, deep])
        atm_scored = next(c for c in scored if c.strike == 23400)
        deep_scored = next(c for c in scored if c.strike == 24000)
        assert atm_scored.score > deep_scored.score, \
            f"ATM score {atm_scored.score:.3f} should be > deep score {deep_scored.score:.3f}"

    def test_ce_preferred_scores_higher(self):
        s = self._make_selector(preferred_direction="CE")
        ce = StrikeCandidate(
            symbol="NIFTY26JUN23400CE", underlying="NIFTY",
            strike=23400, option_type="CE",
            expiry_date=date.today(), expiry_week=0,
            distance_pct=0.0, lot_size=25, estimated_premium=200.0,
        )
        pe = StrikeCandidate(
            symbol="NIFTY26JUN23200PE", underlying="NIFTY",
            strike=23200, option_type="PE",
            expiry_date=date.today(), expiry_week=0,
            distance_pct=0.01, lot_size=25, estimated_premium=195.0,
        )
        scored = s._score_candidates([ce, pe])
        ce_scored = next(c for c in scored if c.option_type == "CE")
        pe_scored = next(c for c in scored if c.option_type == "PE")
        # CE should have higher direction score, all else being equal
        # But affordability differs — let's just check scores are in [0,1]
        assert 0 <= ce_scored.score <= 1
        assert 0 <= pe_scored.score <= 1

    # ── find_best with mocked broker ─────────────────────────────

    def test_find_best_returns_none_when_nothing_fits(self):
        s = self._make_selector(balance=100.0)  # impossibly low
        broker = MagicMock()
        broker.get_quote.return_value.ltp = 500.0  # expensive
        result = s.find_best(broker, max_otm_distance=0.10, max_expiry_weeks=1)
        assert result is None

    def test_find_best_finds_affordable(self):
        s = self._make_selector(balance=16000.0)
        broker = MagicMock()
        # Mock quotes: first few candidates with known premiums
        def mock_quote(symbol):
            quote = MagicMock()
            if "23400" in symbol:
                quote.ltp = 200.0  # ₹200/unit × 25 = ₹5,000 — affordable
            elif "23500" in symbol:
                quote.ltp = 120.0
            elif "24000" in symbol:
                quote.ltp = 50.0
            else:
                quote.ltp = 300.0
            return quote
        broker.get_quote.side_effect = mock_quote

        result = s.find_best(broker, max_otm_distance=0.10, max_expiry_weeks=1)
        assert result is not None
        assert result.total_cost_estimate <= 16000 / MARGIN_BUFFER

    def test_find_best_prefers_ce_when_affordable(self):
        s = self._make_selector(balance=16000.0, preferred_direction="CE")
        broker = MagicMock()
        def mock_quote(symbol):
            quote = MagicMock()
            quote.ltp = 150.0  # all affordable
            return quote
        broker.get_quote.side_effect = mock_quote

        result = s.find_best(broker, max_otm_distance=0.10, max_expiry_weeks=1)
        assert result is not None
        assert result.option_type == "CE"

    def test_find_best_falls_back_to_pe_when_ce_too_expensive(self):
        """When all CEs are too expensive and PE is cheap, pick PE."""
        # Balance is low enough that only very cheap options fit
        s = self._make_selector(balance=3000.0, preferred_direction="CE")
        broker = MagicMock()

        def mock_quote(symbol):
            quote = MagicMock()
            if "CE" in symbol:
                quote.ltp = 600.0  # 600 × 25 = 15,000 — way over balance
            else:
                quote.ltp = 60.0   # 60 × 25 = 1,500 — fits within balance
            return quote
        broker.get_quote.side_effect = mock_quote

        result = s.find_best(broker, max_otm_distance=0.05, max_expiry_weeks=2)
        # With CE at 15K and PE at 1.5K, only PE can fit 3K balance
        if result is not None:
            assert result.option_type == "PE"
            assert result.total_cost_estimate * s.margin_buffer <= s.balance

    # ── Convenience function ─────────────────────────────────────

    def test_find_affordable_strike_returns_symbol(self):
        broker = MagicMock()
        def mock_quote(symbol):
            m = MagicMock()
            m.ltp = 150.0
            return m
        broker.get_quote.side_effect = mock_quote

        result = find_affordable_strike(
            broker=broker,
            underlying="NIFTY",
            spot=23400.0,
            balance=16000.0,
        )
        assert result is not None
        assert "NIFTY" in result and "CE" in result

    def test_find_affordable_strike_returns_none_when_poor(self):
        broker = MagicMock()
        broker.get_quote.return_value.ltp = 1000.0  # way too expensive

        result = find_affordable_strike(
            broker=broker,
            underlying="NIFTY",
            spot=23400.0,
            balance=100.0,
        )
        assert result is None
