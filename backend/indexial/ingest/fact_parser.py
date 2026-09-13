"""
Cell parsing: raw markdown table cell -> typed value.

Pure functions. No database, no network, no LLM. Typing 2,400 cells with a
language model would be slow, non-reproducible, and impossible to unit test;
this is deterministic and free.

The governing policy is: NEVER guess into a typed column. When a cell is
ambiguous, value_num stays NULL, value_type becomes 'ambiguous', and
parse_note records why. A NULL is a visible gap that the coverage counts in
query/sql_templates.py will report. A wrong number is an invisible lie.

Two passes:
  parse_cell    per cell, in isolation
  profile_table per column, reconciling the whole column at once

The second pass is the one that matters. Deciding "is this column numeric" or
"is 1.234 one-point-two-three-four or one thousand two hundred thirty four"
cell by cell produces an incoherent column; deciding it by vote does not.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Iterable, List, Optional, Sequence

__all__ = [
    "CellValue",
    "ColumnProfile",
    "parse_cell",
    "profile_table",
    "sanitize_key",
    "sanitize_keys",
]

# --------------------------------------------------------------- constants --

NULLISH = {
    "", "-", "--", "---", "–", "—", "n/a", "na", "n.a.", "n.a", "nil", "none",
    "null", "not available", "not applicable", ".", "..", "...",
}

# Prefix currency symbols and codes, longest first so "Rs." wins over "R".
CURRENCY_PREFIXES = [
    ("₹", "INR"), ("rs.", "INR"), ("rs", "INR"), ("inr", "INR"),
    ("$", "USD"), ("usd", "USD"),
    ("€", "EUR"), ("eur", "EUR"),
    ("£", "GBP"), ("gbp", "GBP"),
    ("¥", "JPY"), ("jpy", "JPY"),
]

# Magnitude suffixes. value_num always stores the FULLY SCALED number so that
# aggregation never depends on the model remembering to multiply.
SCALES = [
    ("trillion", Decimal("1e12")), ("tn", Decimal("1e12")),
    ("billion", Decimal("1e9")), ("bn", Decimal("1e9")),
    ("crore", Decimal("1e7")), ("cr", Decimal("1e7")),
    ("million", Decimal("1e6")), ("mn", Decimal("1e6")), ("m", Decimal("1e6")),
    ("lakh", Decimal("1e5")), ("lakhs", Decimal("1e5")), ("lac", Decimal("1e5")),
    ("thousand", Decimal("1e3")), ("k", Decimal("1e3")),
    ("bps", Decimal("0.0001")),
]

DATE_FORMATS = [
    "%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d",
    "%d %b %Y", "%d %B %Y", "%b %d, %Y", "%B %d, %Y",
    "%b %Y", "%B %Y", "%Y-%m",
]

# Footnote and formatting noise stripped before parsing.
#
# Bracketed and symbol markers are always safe to remove. A trailing "(12)" is
# NOT: on its own it is an accounting negative, not a footnote. It only counts
# as a footnote when real content precedes it, which _clean() checks.
_FOOTNOTE_RE = re.compile(r"(\[\d+\]|[†‡]+$)")
_TRAILING_PAREN_NOTE_RE = re.compile(r"\(\d+\)$")
_MARKDOWN_RE = re.compile(r"(\*\*|__|`|~~)")
_WHITESPACE_RE = re.compile(r"[\s   ​]+")
_NUMERIC_CORE_RE = re.compile(r"^[+-]?[\d.,]*\d[\d.,]*$")
_YEAR_RE = re.compile(r"^(19|20)\d{2}$")
_INEQUALITY_RE = re.compile(r"^[<>≤≥]=?\s*")
_APPROX_RE = re.compile(r"^[~≈]\s*")
_RANGE_RE = re.compile(r"^[+-]?[\d.,]+\s*(?:-|–|—|to)\s*[+-]?[\d.,]+$", re.IGNORECASE)

MIN_TYPE_AGREEMENT = 0.70   # below this a column stays text
EUROPEAN_VOTE = 0.60        # share of disambiguating cells needed to flip
LABEL_DISTINCT_RATIO = 0.90


# ------------------------------------------------------------------ models --

@dataclass(frozen=True)
class CellValue:
    """One parsed cell. value_text is always the raw original."""

    value_text: str
    value_num: Optional[Decimal] = None
    value_date: Optional[date] = None
    unit: Optional[str] = None
    scale_factor: Optional[Decimal] = None
    value_type: str = "text"  # number | year | date | text | empty | ambiguous
    parse_note: Optional[str] = None


@dataclass(frozen=True)
class ColumnProfile:
    """Reconciled facts about one column, after looking at all of its cells."""

    index: int
    name: str
    key: str
    value_type: str
    unit: Optional[str] = None
    scale_factor: Optional[Decimal] = None
    numeric_coverage: float = 0.0
    distinct_count: int = 0
    is_label: bool = False


# ------------------------------------------------------------------- keys --

def sanitize_key(name: str) -> str:
    """Normalize a header into the identifier the LLM will write in WHERE."""
    key = re.sub(r"[^a-z0-9]+", "_", (name or "").strip().lower()).strip("_")
    if not key:
        return "column"
    if key[0].isdigit():
        # Bare years are extremely common headers; keep them readable.
        return key if _YEAR_RE.match(key) else f"c_{key}"
    return key


def sanitize_keys(headers: Sequence[str]) -> List[str]:
    """Sanitize headers, disambiguating duplicates with _2, _3, ..."""
    out: List[str] = []
    seen: dict[str, int] = {}
    for h in headers:
        key = sanitize_key(h)
        seen[key] = seen.get(key, 0) + 1
        out.append(key if seen[key] == 1 else f"{key}_{seen[key]}")
    return out


# -------------------------------------------------------------- pass one ----

def _clean(raw: str) -> str:
    text = _MARKDOWN_RE.sub("", raw or "")
    text = _WHITESPACE_RE.sub(" ", text).strip()
    text = _FOOTNOTE_RE.sub("", text).strip()

    # "Revenue (1)" -> footnote. "(1,234)" or "(45)" -> accounting negative.
    # The difference is whether anything meaningful comes before it.
    match = _TRAILING_PAREN_NOTE_RE.search(text)
    if match and text[: match.start()].strip():
        text = text[: match.start()].strip()

    # A trailing '*' is a footnote only when it is not the whole cell.
    if text.endswith("*") and text.strip("*"):
        text = text.rstrip("*").strip()

    return text


def _peel_unit(text: str) -> tuple[str, Optional[str]]:
    """Strip a currency marker or trailing percent, returning (rest, unit)."""
    unit = None
    low = text.lower()
    for token, code in CURRENCY_PREFIXES:
        if low.startswith(token):
            unit = code
            text = text[len(token):].strip()
            break
    if text.endswith("%"):
        unit = "%"
        text = text[:-1].strip()
    return text, unit


def _peel_scale(text: str) -> tuple[str, Optional[Decimal], Optional[str]]:
    """Strip a magnitude suffix, returning (rest, factor, matched token)."""
    low = text.lower().replace(" ", "")
    for token, factor in SCALES:
        if low.endswith(token) and len(low) > len(token):
            core = low[: -len(token)].strip(" .")
            if core and _NUMERIC_CORE_RE.match(core):
                return core, factor, token
    return text, None, None


def _to_decimal(text: str, european: Optional[bool]) -> Optional[Decimal]:
    """
    Interpret grouping and decimal separators.

    With both separators present the LAST one is the decimal separator, which
    settles 1.234,56 against 1,234.56 without guessing. With only commas, a
    non-uniform group length (1,23,456) means Indian grouping rather than a
    decimal.
    """
    text = text.strip()
    if not text or not _NUMERIC_CORE_RE.match(text):
        return None

    has_dot, has_comma = "." in text, "," in text
    try:
        if has_dot and has_comma:
            if text.rfind(",") > text.rfind("."):
                return Decimal(text.replace(".", "").replace(",", "."))
            return Decimal(text.replace(",", ""))

        if has_comma:
            groups = text.split(",")
            tail = groups[-1]
            uniform_thousands = all(len(g) == 3 for g in groups[1:])
            if european and len(groups) == 2 and len(tail) != 3:
                return Decimal(text.replace(",", "."))
            if uniform_thousands or not all(g.isdigit() for g in groups[1:]):
                return Decimal(text.replace(",", ""))
            # 1,23,456 style Indian grouping
            return Decimal(text.replace(",", ""))

        if has_dot:
            parts = text.split(".")
            if european and len(parts) == 2 and len(parts[-1]) == 3:
                return Decimal(text.replace(".", ""))
            if len(parts) > 2:
                return Decimal(text.replace(".", ""))
            return Decimal(text)

        return Decimal(text)
    except (InvalidOperation, ValueError):
        return None


def _to_date(text: str) -> Optional[date]:
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    try:
        from dateutil.parser import parse as dateutil_parse

        return dateutil_parse(text, fuzzy=False).date()
    except Exception:  # noqa: BLE001 - any parse failure means "not a date"
        return None


def parse_cell(raw: str, european: Optional[bool] = None) -> CellValue:
    """
    Parse one cell in isolation. profile_table() reconciles the column after.
    """
    original = raw if raw is not None else ""
    text = _clean(original)

    if text.lower() in NULLISH:
        return CellValue(value_text=original.strip(), value_type="empty")

    if _RANGE_RE.match(text):
        return CellValue(original.strip(), value_type="ambiguous", parse_note="range")

    if _INEQUALITY_RE.match(text):
        return CellValue(original.strip(), value_type="ambiguous", parse_note="inequality")

    note = None
    body = text
    if _APPROX_RE.match(body):
        body = _APPROX_RE.sub("", body).strip()
        note = "approximate"

    # Accounting negatives. The parentheses may wrap the whole cell,
    # "(1,234)", or only its numeric part, "(1.5) crore" / "(45)%".
    negative = False
    if body.startswith("("):
        wrapped = re.match(r"^\(([^)]+)\)\s*(.*)$", body)
        if wrapped:
            inner, remainder = wrapped.group(1).strip(), wrapped.group(2).strip()
            if inner and not inner[0].isalpha():
                negative = True
                body = f"{inner} {remainder}".strip()

    body, unit = _peel_unit(body)
    body, scale, _ = _peel_scale(body)

    # Dates must be tried before the multi-value fallback below, which would
    # otherwise shorten "2024-01-31" to "2024" and type it as a year.
    if not _NUMERIC_CORE_RE.match(body):
        parsed_date = _to_date(text)
        if parsed_date is not None:
            return CellValue(
                value_text=original.strip(),
                value_date=parsed_date,
                value_type="date",
                parse_note=note,
            )

    # A cell like "1,234 (5.6%)" carries more than one number; take the first.
    if not _NUMERIC_CORE_RE.match(body):
        head = re.match(r"^([+-]?[\d.,]*\d[\d.,]*)\b", body)
        if head:
            body, note = head.group(1), note or "multi_value"

    if _YEAR_RE.match(body) and scale is None and unit is None:
        year = int(body)
        return CellValue(
            value_text=original.strip(),
            value_num=Decimal(year),
            value_date=date(year, 1, 1),
            value_type="year",
            parse_note=note,
        )

    number = _to_decimal(body, european)
    if number is not None:
        if negative:
            number = -number
        if scale is not None:
            number = number * scale
        return CellValue(
            value_text=original.strip(),
            value_num=number,
            unit=unit,
            scale_factor=scale,
            value_type="number",
            parse_note=note,
        )

    parsed_date = _to_date(text)
    if parsed_date is not None:
        return CellValue(
            value_text=original.strip(),
            value_date=parsed_date,
            value_type="date",
            parse_note=note,
        )

    return CellValue(value_text=original.strip(), value_type="text", parse_note=note)


# -------------------------------------------------------------- pass two ----

def _header_unit_and_scale(header: str) -> tuple[Optional[str], Optional[Decimal]]:
    """
    Read units out of a header such as 'Revenue (INR crore)' or 'Rate (%)'.

    Very common in real PDF tables and a large accuracy win: the cells
    themselves carry no marker, so without this the unit is simply lost.
    """
    match = re.search(r"[（(\[]([^)\]）]*)[）)\]]\s*$", header or "")
    if not match:
        return (("%", None) if "%" in (header or "") else (None, None))

    inside = match.group(1).strip().lower()
    unit: Optional[str] = None
    scale: Optional[Decimal] = None

    if "%" in inside or "percent" in inside:
        unit = "%"
    for token, code in CURRENCY_PREFIXES:
        if re.search(rf"\b{re.escape(token)}\b", inside) or token in ("₹", "$", "€", "£", "¥") and token in inside:
            unit = code
            break
    for token, factor in SCALES:
        if re.search(rf"\b{re.escape(token)}\b", inside):
            scale = factor
            break
    if unit is None and scale is None and inside and len(inside) <= 12 and inside.isalpha():
        unit = inside
    return unit, scale


def _european_vote(cells: Iterable[str]) -> bool:
    """Decide separator convention for a whole column rather than per cell."""
    european = western = 0
    for raw in cells:
        text = _clean(raw)
        if "." in text and "," in text:
            if text.rfind(",") > text.rfind("."):
                european += 1
            else:
                western += 1
        elif "," in text:
            groups = text.split(",")
            if len(groups) == 2 and len(groups[-1]) in (1, 2):
                european += 1
            elif all(len(g) == 3 for g in groups[1:]):
                western += 1
    total = european + western
    return total > 0 and european / total >= EUROPEAN_VOTE


def profile_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
) -> tuple[List[ColumnProfile], List[List[CellValue]]]:
    """
    Parse a whole table and reconcile each column.

    Returns (profiles, parsed_rows). parsed_rows is ragged-tolerant: a row with
    fewer cells than headers yields fewer CellValues rather than being dropped.
    """
    keys = sanitize_keys(headers)
    width = len(headers)
    columns: List[List[str]] = [[] for _ in range(width)]
    for row in rows:
        for i in range(min(width, len(row))):
            columns[i].append(row[i])

    profiles: List[ColumnProfile] = []
    europeans: List[bool] = []

    for i, raw_cells in enumerate(columns):
        european = _european_vote(raw_cells)
        europeans.append(european)

        parsed = [parse_cell(c, european) for c in raw_cells]
        non_empty = [p for p in parsed if p.value_type != "empty"]

        counts: dict[str, int] = {}
        for p in non_empty:
            counts[p.value_type] = counts.get(p.value_type, 0) + 1

        if non_empty:
            winner, hits = max(counts.items(), key=lambda kv: kv[1])
            agreement = hits / len(non_empty)
        else:
            winner, agreement = "empty", 0.0

        # A 'Notes' column with three stray numbers is not a numeric column.
        col_type = winner if agreement >= MIN_TYPE_AGREEMENT else "text"

        numeric_hits = sum(1 for p in parsed if p.value_num is not None)
        coverage = numeric_hits / len(parsed) if parsed else 0.0

        header_unit, header_scale = _header_unit_and_scale(headers[i] if i < len(headers) else "")
        cell_unit = next((p.unit for p in non_empty if p.unit), None)

        profiles.append(
            ColumnProfile(
                index=i,
                name=headers[i] if i < len(headers) else f"column_{i}",
                key=keys[i] if i < len(keys) else f"column_{i}",
                value_type=col_type,
                unit=cell_unit or header_unit,
                scale_factor=header_scale,
                numeric_coverage=coverage,
                distinct_count=len({c.strip() for c in raw_cells if c and c.strip()}),
            )
        )

    # Label column: first text column whose values are mostly distinct.
    label_index = 0
    row_total = len(rows)
    for prof in profiles:
        if prof.value_type == "text" and row_total and prof.distinct_count >= LABEL_DISTINCT_RATIO * row_total:
            label_index = prof.index
            break
    profiles = [replace(p, is_label=(p.index == label_index)) for p in profiles]

    # Re-parse rows with the reconciled column decisions applied.
    parsed_rows: List[List[CellValue]] = []
    for row in rows:
        parsed_row: List[CellValue] = []
        for i in range(min(width, len(row))):
            prof = profiles[i]
            cell = parse_cell(row[i], europeans[i])

            # Column voted text: drop typed values so the column is coherent.
            if prof.value_type == "text" and cell.value_type in ("number", "date", "year"):
                cell = CellValue(
                    value_text=cell.value_text,
                    value_type="text",
                    parse_note=cell.parse_note or "column_is_text",
                )
            # Inherit unit and scale declared in the header.
            elif cell.value_type in ("number", "year"):
                if cell.unit is None and prof.unit:
                    cell = replace(cell, unit=prof.unit)
                if cell.scale_factor is None and prof.scale_factor and cell.value_num is not None:
                    cell = replace(
                        cell,
                        value_num=cell.value_num * prof.scale_factor,
                        scale_factor=prof.scale_factor,
                    )
            parsed_row.append(cell)
        parsed_rows.append(parsed_row)

    return profiles, parsed_rows
