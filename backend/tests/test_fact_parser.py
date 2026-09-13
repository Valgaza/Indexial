"""
Cell parsing tests. No database, no network, no API keys.

The cases that matter are the ambiguous ones: the policy is that an uncertain
cell must leave value_num NULL rather than guess, because a NULL is a visible
gap and a wrong number is an invisible lie.
"""

from decimal import Decimal

import pytest

from indexial.ingest.fact_parser import (
    parse_cell,
    profile_table,
    sanitize_key,
    sanitize_keys,
)


# ------------------------------------------------------------------ basics --

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("1234", Decimal("1234")),
        ("1,234", Decimal("1234")),
        ("1,234.56", Decimal("1234.56")),
        ("-42", Decimal("-42")),
        ("+42", Decimal("42")),
        ("0.5", Decimal("0.5")),
    ],
)
def test_plain_numbers(raw, expected):
    cell = parse_cell(raw)
    assert cell.value_type == "number"
    assert cell.value_num == expected


def test_indian_grouping():
    assert parse_cell("1,23,456").value_num == Decimal("123456")


def test_european_separators_when_column_says_so():
    assert parse_cell("1.234,56", european=True).value_num == Decimal("1234.56")


def test_last_separator_wins_without_a_column_hint():
    assert parse_cell("1.234,56").value_num == Decimal("1234.56")
    assert parse_cell("1,234.56").value_num == Decimal("1234.56")


def test_accounting_parentheses_are_negative():
    cell = parse_cell("(1,234)")
    assert cell.value_type == "number"
    assert cell.value_num == Decimal("-1234")


# ------------------------------------------------------- units and scaling --

def test_percent_is_stored_as_written():
    cell = parse_cell("45.2%")
    assert cell.value_num == Decimal("45.2")  # never silently /100
    assert cell.unit == "%"


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("1.2M", Decimal("1200000")),
        ("3bn", Decimal("3000000000")),
        ("5k", Decimal("5000")),
        ("2 crore", Decimal("20000000")),
        ("1.5 lakh", Decimal("150000")),
    ],
)
def test_magnitude_suffix_is_fully_scaled(raw, expected):
    """value_num holds base units so aggregation never multiplies."""
    cell = parse_cell(raw)
    assert cell.value_type == "number"
    assert cell.value_num == expected


def test_currency_prefix_becomes_unit():
    for raw, unit in [("₹1,200", "INR"), ("$99.50", "USD"), ("€10", "EUR")]:
        cell = parse_cell(raw)
        assert cell.value_type == "number", raw
        assert cell.unit == unit, raw


# ----------------------------------------------------------- the null rule --

@pytest.mark.parametrize("raw", ["", "-", "–", "N/A", "n.a.", "nil", "NULL"])
def test_nullish_is_empty_not_zero(raw):
    cell = parse_cell(raw)
    assert cell.value_type == "empty"
    assert cell.value_num is None


@pytest.mark.parametrize(
    "raw,note",
    [("12-15", "range"), (">1000", "inequality"), ("≥ 5", "inequality")],
)
def test_ambiguous_cells_refuse_to_guess(raw, note):
    cell = parse_cell(raw)
    assert cell.value_type == "ambiguous"
    assert cell.value_num is None
    assert cell.parse_note == note


def test_multi_value_takes_the_first_number_and_says_so():
    cell = parse_cell("1,234 (5.6%)")
    assert cell.value_num == Decimal("1234")
    assert cell.parse_note == "multi_value"


def test_approximate_is_kept_but_flagged():
    cell = parse_cell("~500")
    assert cell.value_num == Decimal("500")
    assert cell.parse_note == "approximate"


def test_ocr_noise_is_not_corrected():
    cell = parse_cell("l23")  # lowercase L, not a 1
    assert cell.value_type == "text"
    assert cell.value_num is None


def test_value_text_always_survives():
    for raw in ["1,234", "N/A", "12-15", "~500", "hello"]:
        assert parse_cell(raw).value_text == raw.strip()


# -------------------------------------------------------------- years/dates --

def test_year_is_distinct_from_number():
    """Otherwise SUM over a year column returns 4045 for 2022 + 2023."""
    cell = parse_cell("2023")
    assert cell.value_type == "year"
    assert cell.value_num == Decimal("2023")
    assert cell.value_date is not None and cell.value_date.year == 2023


def test_iso_date():
    cell = parse_cell("2024-01-31")
    assert cell.value_type == "date"
    assert cell.value_date.isoformat() == "2024-01-31"


# ---------------------------------------------------------------- profiling --

def test_notes_column_with_stray_numbers_stays_text():
    headers = ["Item", "Notes"]
    rows = [["a", "see p. 4"], ["b", "12"], ["c", "unchanged"], ["d", "7"], ["e", "n/a"]]
    profiles, parsed = profile_table(headers, rows)
    assert profiles[1].value_type == "text"
    # and the numeric readings are cleared, so the column is coherent
    assert all(c[1].value_num is None for c in parsed)


def test_header_unit_is_inherited_by_cells():
    headers = ["Segment", "Revenue (INR crore)"]
    rows = [["Consumer", "1204"], ["Enterprise", "3880"]]
    profiles, parsed = profile_table(headers, rows)
    assert profiles[1].unit == "INR"
    # 1204 crore, scaled into base units
    assert parsed[0][1].value_num == Decimal("1204") * Decimal("1e7")


def test_percent_header_inherited():
    profiles, _ = profile_table(["Region", "Share (%)"], [["N", "12"], ["S", "88"]])
    assert profiles[1].unit == "%"


def test_label_column_selection():
    headers = ["Segment", "2022", "2023"]
    rows = [["Consumer", "1", "2"], ["Enterprise", "3", "4"], ["Other", "5", "6"]]
    profiles, _ = profile_table(headers, rows)
    assert profiles[0].is_label
    assert not profiles[1].is_label


def test_ragged_rows_are_kept_not_dropped():
    """The old parser discarded any row whose width != header width."""
    headers = ["a", "b", "c"]
    rows = [["1", "2", "3"], ["4", "5"]]
    _, parsed = profile_table(headers, rows)
    assert len(parsed) == 2
    assert len(parsed[1]) == 2


def test_numeric_coverage_is_reported():
    headers = ["x"]
    rows = [["1"], ["2"], ["bad"], ["4"]]
    profiles, _ = profile_table(headers, rows)
    assert profiles[0].numeric_coverage == 0.75


# --------------------------------------------------------------------- keys --

def test_sanitize_key():
    assert sanitize_key("Total Revenue (₹)") == "total_revenue"
    assert sanitize_key("2024") == "2024"
    assert sanitize_key("") == "column"


def test_duplicate_headers_are_disambiguated():
    assert sanitize_keys(["Total", "Total", "Other"]) == ["total", "total_2", "other"]


# ------------------------------------------------- footnotes vs negatives --

def test_bare_parenthesised_number_is_a_negative_not_a_footnote():
    """Regression: the footnote stripper used to eat '(45)' whole."""
    cell = parse_cell("(45)")
    assert cell.value_type == "number"
    assert cell.value_num == Decimal("-45")


def test_trailing_paren_after_content_is_a_footnote():
    assert parse_cell("Revenue (1)").value_text == "Revenue (1)"
    assert parse_cell("Revenue (1)").value_type == "text"


def test_footnote_marker_on_a_number_is_stripped():
    for raw in ["1,234[1]", "1,234†", "1,234*"]:
        cell = parse_cell(raw)
        assert cell.value_num == Decimal("1234"), raw


def test_negative_with_currency_and_scale():
    cell = parse_cell("(1.5) crore")
    assert cell.value_num == Decimal("-1.5") * Decimal("1e7")
