"""Tests for processor.normalization."""

# Third party imports
import pytest

# Application imports
from idi_corporate_structure.normalization import (
    SEC_STATE_OF_INCORPORATION,
    normalize_parent_location,
    normalize_subsidiary_location,
)


class TestNormalizeSubsidiaryLocation:
    """Tests for normalize_subsidiary_location."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("", ""),
            ("   ", ""),
            ("Delaware", "Delaware"),
            ("Cayman Islands", "Cayman Islands"),
        ],
    )
    def test_passthrough(self, raw, expected):
        assert normalize_subsidiary_location(raw) == expected

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("Mexico(2)", "Mexico"),
            ("Mexico (2)", "Mexico"),
            ("Delaware(1)", "Delaware"),
            ("Delaware (12) ", "Delaware"),
        ],
    )
    def test_strips_trailing_footnote(self, raw, expected):
        assert normalize_subsidiary_location(raw) == expected

    @pytest.mark.parametrize(
        "raw",
        ["Unknown", "unknown", "UNKNOWN", "N/A", "n/a", "None", "Not Applicable", "--", "  none  "],
    )
    def test_blank_sentinels_become_empty(self, raw):
        assert normalize_subsidiary_location(raw) == ""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("PRC", "China"),
            ("People's Republic of China", "China"),
            ("p.r.c.", "China"),
            ("USA", "United States"),
            ("U.S.A.", "United States"),
            ("United States of America", "United States"),
            ("UK", "United Kingdom"),
            ("Hong Kong SAR", "Hong Kong"),
            ("UAE", "United Arab Emirates"),
            ("South Korea", "Korea, Republic of"),
        ],
    )
    def test_country_aliases(self, raw, expected):
        assert normalize_subsidiary_location(raw) == expected

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("State of Wisconsin", "Wisconsin"),
            ("state of delaware", "Delaware"),
            ("State of New York", "New York"),
        ],
    )
    def test_state_of_x_regex(self, raw, expected):
        assert normalize_subsidiary_location(raw) == expected

    def test_handles_none(self):
        assert normalize_subsidiary_location(None) == ""

    def test_combination_footnote_then_alias(self):
        assert normalize_subsidiary_location("PRC(1)") == "China"

    def test_trailing_punctuation_stripped(self):
        assert normalize_subsidiary_location("Delaware,") == "Delaware"


class TestNormalizeParentLocation:
    """Tests for normalize_parent_location."""

    @pytest.mark.parametrize(
        ("code", "expected"),
        [
            ("DE", "Delaware"),
            ("MD", "Maryland"),
            ("E9", "Cayman Islands"),
            ("D0", "Bermuda"),
            ("L2", "Ireland"),
            ("V8", "Switzerland"),
            ("1T", "Marshall Islands"),
            ("X1", "United States"),
            ("X0", "United Kingdom"),
            ("A6", "Ontario, Canada"),
        ],
    )
    def test_known_codes(self, code, expected):
        assert normalize_parent_location(code) == expected

    def test_unknown_code_passes_through(self):
        # Future SEC additions should not be silently dropped.
        assert normalize_parent_location("ZZ") == "ZZ"

    def test_blank(self):
        assert normalize_parent_location("") == ""
        assert normalize_parent_location("   ") == ""

    def test_none(self):
        assert normalize_parent_location(None) == ""

    def test_xx_unknown_collapses_to_blank(self):
        # SEC explicitly reserves XX for "Unknown" — treat as blank for parity.
        assert normalize_parent_location("XX") == ""


class TestSecCodeTable:
    """Sanity checks on the static SEC_STATE_OF_INCORPORATION mapping."""

    def test_size(self):
        # Sanity check: the SEC table should have ~310 entries. Allow a small
        # window so additive updates don't break the test.
        assert 280 <= len(SEC_STATE_OF_INCORPORATION) <= 400

    def test_us_states_present(self):
        for code in ("AL", "CA", "DE", "NY", "WY"):
            assert code in SEC_STATE_OF_INCORPORATION

    def test_common_offshore_jurisdictions_present(self):
        for code in ("E9", "D0", "L2", "V8"):
            assert code in SEC_STATE_OF_INCORPORATION
