"""
Deterministic artifact generation.

Takes (columns, type_oids, rows) and returns a chart specification. Never
imports llm, db, flask or requests, and the user's question influences only the
artifact's title string - never the chart type, never the data. A test enforces
that by parsing this module's own imports.

That constraint is the whole point. Because nothing generative sits between the
verified rows and the renderer, a chart here cannot misrepresent the data: the
worst failure available is picking a duller chart than a human would.

Column types come from Postgres type OIDs rather than from sniffing values, so
an empty column or a column of NULLs is still correctly typed.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Literal, Optional, Sequence

from indexial.query.pivot import is_long_format, pivot_long

__all__ = [
    "SCHEMA_VERSION",
    "ColumnMeta",
    "build_artifact",
    "classify",
    "describe_columns",
    "json_safe",
    "slug_keys",
    "to_records",
]

SCHEMA_VERSION = 1

# ------------------------------------------------------------ type codes ----

NUMERIC_OIDS = frozenset({20, 21, 23, 700, 701, 1700})      # int8 int2 int4 float4 float8 numeric
TEMPORAL_OIDS = frozenset({1082, 1114, 1184})                # date timestamp timestamptz
TEXTUAL_OIDS = frozenset({25, 1043, 1042, 18, 19})           # text varchar bpchar char name
BOOLEAN_OIDS = frozenset({16})
CLOCK_OIDS = frozenset({1083, 1266})                         # time, timetz: not an axis
# 790 (money) is deliberately NOT numeric: psycopg2 returns it as a locale
# formatted string like '$1,234.00', which would plot as NaN.
OPAQUE_OIDS = frozenset({114, 3802, 17, 2950, 1186, 790})

ColumnKind = Literal["numeric", "temporal", "categorical", "boolean", "opaque"]
ColumnRole = Literal["measure", "dimension", "identifier", "empty_measure", "constant", "opaque"]

# Numeric columns whose name says they are an identifier, not a measure. The
# fact store makes this essential: document_id, table_index, row_index and
# page_number are all integers, and charting row_index is the single most
# likely wrong chart this system could draw.
IDENTIFIER_RE = re.compile(
    r"^(id|uuid|rank|n|row_index|col_index|page_number|table_index)$|(_id|_uuid|_index)$"
)

# ------------------------------------------------------------- thresholds ---

MAX_SERIES = 5              # --chart-1..5 in app/globals.css is the palette
MAX_BAR_CATEGORIES = 25     # ~680px bubble / 25 bands = 27px, the narrowest
MAX_HBAR_CATEGORIES = 60    # 60 * 28px is already two screens tall
MAX_LINE_POINTS = 500
MIN_CHART_ROWS = 2          # one point is a scalar, not a trend
MIN_SCATTER_ROWS = 10       # fewer invites fitting a line through noise
MAX_CHART_COLUMNS = 12
MAX_DATA_ROWS = 200
_MAX_EXACT = 2**53          # JS JSON.parse silently rounds integers above this

# Bookkeeping columns the aggregate templates return alongside the answer.
# They belong in the prose caveat, not on a chart axis: without excluding them
# a one-number aggregate arrives as four columns and classifies as a table.
METADATA_COLUMNS = frozenset({"parsed_cells", "total_cells", "unit"})


@dataclass(frozen=True)
class ColumnMeta:
    name: str
    key: str
    kind: ColumnKind
    role: ColumnRole
    type_oid: int
    null_count: int
    distinct_count: int
    decimals: int = 0

    @property
    def is_measure(self) -> bool:
        return self.role == "measure"


# ------------------------------------------------------------ conversion ----

def json_safe(value: Any, _depth: int = 0) -> Any:
    """
    Make a psycopg2 value safe for jsonify.

    None of these have bitten yet only because no row value has ever reached
    the response. Every one of them bites on the first SQL query that renders a
    chart: Decimal raises outright, float('nan') emits a bare NaN token that is
    invalid JSON and loses the whole response, and date becomes an RFC-822
    string that cannot be sorted.
    """
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        # A bigint id above 2^53 loses precision in the browser.
        return value if abs(value) < _MAX_EXACT else str(value)
    if isinstance(value, float):
        return value if value == value and value not in (float("inf"), float("-inf")) else None
    if isinstance(value, Decimal):
        if not value.is_finite():
            return None
        as_float = float(value)
        return as_float if abs(as_float) < _MAX_EXACT else str(value)
    if isinstance(value, (datetime, date, time)):
        # ISO strings, deliberately. An epoch timestamp parsed by JS Date and
        # rendered in a western timezone shows the previous day.
        return value.isoformat()
    if isinstance(value, timedelta):
        return value.total_seconds()
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, (bytes, memoryview)):
        return f"<{len(bytes(value))} bytes>"
    if isinstance(value, (list, tuple)):
        return ["<nested>"] if _depth >= 6 else [json_safe(v, _depth + 1) for v in value]
    if isinstance(value, dict):
        return "<nested>" if _depth >= 6 else {
            str(k): json_safe(v, _depth + 1) for k, v in value.items()
        }
    return str(value)


def slug_keys(columns: Sequence[str]) -> List[str]:
    """
    Recharts-safe dataKeys, de-duplicated.

    'Total Revenue (₹)' is not usable as a dataKey, and a self-join can return
    two columns both named 'id'.
    """
    out: List[str] = []
    seen: Dict[str, int] = {}
    for name in columns:
        slug = re.sub(r"[^a-z0-9]+", "_", (name or "").lower()).strip("_") or "col"
        if slug[0].isdigit():
            slug = f"c_{slug}"
        seen[slug] = seen.get(slug, 0) + 1
        out.append(slug if seen[slug] == 1 else f"{slug}_{seen[slug]}")
    return out


# ------------------------------------------------------------- describing ---

def _kind_for(oid: int) -> ColumnKind:
    if oid in NUMERIC_OIDS:
        return "numeric"
    if oid in TEMPORAL_OIDS:
        return "temporal"
    if oid in BOOLEAN_OIDS:
        return "boolean"
    if oid in TEXTUAL_OIDS or oid in CLOCK_OIDS:
        return "categorical"
    return "opaque"


def describe_columns(
    columns: Sequence[str],
    type_oids: Sequence[int],
    rows: Sequence[Sequence[Any]],
) -> List[ColumnMeta]:
    """Type and role every column. Types come from OIDs; roles from counting."""
    keys = slug_keys(columns)
    metas: List[ColumnMeta] = []
    row_count = len(rows)

    for i, name in enumerate(columns):
        oid = type_oids[i] if i < len(type_oids) else 0
        kind = _kind_for(oid)

        values = [r[i] for r in rows if i < len(r)]
        nulls = sum(1 for v in values if v is None)
        distinct = len({str(v) for v in values if v is not None})

        decimals = 0
        if kind == "numeric":
            for v in values:
                if isinstance(v, Decimal):
                    decimals = max(decimals, max(0, -v.as_tuple().exponent))
                elif isinstance(v, float) and v == v:
                    decimals = max(decimals, min(6, len(f"{v!r}".partition(".")[2])))

        role: ColumnRole = "dimension"
        if kind == "numeric":
            if IDENTIFIER_RE.search((name or "").lower()):
                role = "identifier"
            elif row_count and nulls == row_count:
                # Long-format results carry value_num, value_text and
                # value_date side by side; two of the three are always NULL.
                role = "empty_measure"
            elif distinct == 1 and row_count > 2:
                role = "constant"
            else:
                role = "measure"
        elif kind == "opaque":
            role = "opaque"

        metas.append(
            ColumnMeta(
                name=name,
                key=keys[i],
                kind=kind,
                role=role,
                type_oid=oid,
                null_count=nulls,
                distinct_count=distinct,
                decimals=min(decimals, 4),
            )
        )
    return metas


def to_records(
    metas: Sequence[ColumnMeta],
    rows: Sequence[Sequence[Any]],
    limit: int = MAX_DATA_ROWS,
) -> List[Dict[str, Any]]:
    """Rows as JSON-safe dicts keyed by slug."""
    return [
        {m.key: json_safe(row[i]) for i, m in enumerate(metas) if i < len(row)}
        for row in rows[:limit]
    ]


# ------------------------------------------------------------ classifying ---

def _field(meta: ColumnMeta, color_index: Optional[int] = None) -> Dict[str, Any]:
    field: Dict[str, Any] = {
        "key": meta.key,
        "label": meta.name,
        "kind": meta.kind,
        "align": "right" if meta.kind == "numeric" else "left",
    }
    if meta.kind == "numeric":
        field["decimals"] = meta.decimals
    if color_index is not None:
        field["color_index"] = color_index
    return field


def _sorted_by(records: List[Dict[str, Any]], key: str, descending: bool) -> List[Dict[str, Any]]:
    def sort_key(record):
        value = record.get(key)
        return (value is None, value if value is not None else 0)

    try:
        return sorted(records, key=sort_key, reverse=descending)
    except TypeError:
        return records


def classify(
    metas: Sequence[ColumnMeta],
    records: List[Dict[str, Any]],
    *,
    unordered_truncation: bool = False,
) -> Dict[str, Any]:
    """
    Pick a chart from the shape of the result set.

    Ordered; first match wins. `table` is the universal fallback, so there is
    no branch that produces nothing.
    """
    notes: List[str] = []
    rows = len(records)
    cols = len(metas)

    measures = [m for m in metas if m.role == "measure"]
    temporals = [m for m in metas if m.kind == "temporal"]
    categoricals = [m for m in metas if m.kind in ("categorical", "boolean")]

    def table(rule: str) -> Dict[str, Any]:
        return {"type": "table", "kind": "table", "encoding": None, "rule": rule, "notes": notes}

    if rows == 0:
        return {"type": "empty", "kind": "empty", "encoding": None, "rule": "R1_empty", "notes": notes}

    if rows == 1 and cols == 1:
        only = metas[0]
        value = records[0].get(only.key)
        if only.kind == "numeric" or (isinstance(value, str) and len(value) <= 200):
            return {
                "type": "scalar",
                "kind": "scalar",
                "encoding": None,
                "rule": "R2_scalar",
                "notes": notes,
            }
        return table("R2_scalar_too_long")

    if cols > MAX_CHART_COLUMNS:
        notes.append(f"{cols} columns is too many to chart legibly.")
        return table("R3_too_wide")

    if unordered_truncation:
        # An arbitrary N rows of an unknown total. Charting it would assert a
        # ranking the query never established.
        notes.append("Result hit the row limit with no ORDER BY, so it is an arbitrary subset.")
        return table("R5_unordered_truncation")

    # Trend: one temporal axis, up to MAX_SERIES measures.
    if len(temporals) == 1 and measures and rows >= MIN_CHART_ROWS:
        if len(measures) > MAX_SERIES:
            notes.append(f"{len(measures)} series exceeds the {MAX_SERIES}-colour palette.")
            return table("R7_series_cap")
        if rows <= MAX_LINE_POINTS:
            return {
                "type": "line",
                "kind": "chart",
                "encoding": {
                    "x": _field(temporals[0]),
                    "y": [_field(m, i + 1) for i, m in enumerate(measures)],
                    "series": None,
                },
                "rule": "R6_temporal_line",
                "notes": notes,
                "sort": (temporals[0].key, False),
            }
        notes.append(f"{rows} points exceeds the {MAX_LINE_POINTS}-point chart cap.")
        return table("R6_too_many_points")

    # Category vs measure(s).
    if len(categoricals) == 1 and measures and rows >= MIN_CHART_ROWS:
        category = categoricals[0]

        if len(measures) == 1:
            measure = measures[0]
            if rows <= MAX_BAR_CATEGORIES:
                return {
                    "type": "bar",
                    "kind": "chart",
                    "encoding": {"x": _field(category), "y": [_field(measure, 1)], "series": None},
                    "rule": "R8_bar",
                    "notes": notes,
                    "sort": (measure.key, True),
                }
            if rows <= MAX_HBAR_CATEGORIES:
                return {
                    "type": "bar_horizontal",
                    "kind": "chart",
                    "encoding": {"x": _field(category), "y": [_field(measure, 1)], "series": None},
                    "rule": "R9_bar_horizontal",
                    "notes": notes,
                    "sort": (measure.key, True),
                }
            notes.append(f"{rows} categories is past the point a bar chart stays readable.")
            return table("R9_too_many_categories")

        if len(measures) <= MAX_SERIES and rows <= 15:
            return {
                "type": "bar_grouped",
                "kind": "chart",
                "encoding": {
                    "x": _field(category),
                    "y": [_field(m, i + 1) for i, m in enumerate(measures)],
                    "series": None,
                },
                "rule": "R10_bar_grouped",
                "notes": notes,
            }
        return table("R10_grouped_too_large")

    # Two measures, nothing to put on a category axis.
    if not categoricals and not temporals and len(measures) == 2 and rows >= MIN_SCATTER_ROWS:
        return {
            "type": "scatter",
            "kind": "chart",
            "encoding": {"x": _field(measures[0]), "y": [_field(measures[1], 1)], "series": None},
            "rule": "R12_scatter",
            "notes": notes,
        }

    if not measures:
        return table("R13_no_measures")

    return table("R14_default")


# ----------------------------------------------------------------- build ----

_LIMIT_RE = re.compile(r"\bLIMIT\s+(\d+)", re.IGNORECASE)
_ORDER_RE = re.compile(r"\bORDER\s+BY\b", re.IGNORECASE)


def _truncation_flags(sql: Optional[str], row_count: int) -> tuple[bool, bool]:
    """
    (unordered_truncation, partial).

    `SELECT a, b FROM t LIMIT 100` returning exactly 100 rows is an arbitrary
    100 of an unknown total, and a bar chart of it is a lie about ranking. The
    same query with an ORDER BY is a genuine top-100: chartable, but the reader
    has to be told more rows may exist.
    """
    if not sql:
        return False, False
    limits = _LIMIT_RE.findall(sql)
    if not limits:
        return False, False
    hit = row_count >= int(limits[-1])
    ordered = bool(_ORDER_RE.search(sql))
    return hit and not ordered, hit and ordered


def build_artifact(
    exec_result: Dict[str, Any],
    *,
    sql: Optional[str] = None,
    tables_used: Optional[List[str]] = None,
    sources: Optional[List[Dict[str, Any]]] = None,
    title: str = "",
    llm_merged: bool = False,
) -> List[Dict[str, Any]]:
    """
    Single entry point. Returns [] on failure, else a one-element list.

    A list so a HYBRID answer can later carry both a chart and a table without
    a breaking change, and so [] is the natural failure value.
    """
    if not exec_result or not exec_result.get("success"):
        return []

    columns = exec_result.get("columns") or []
    rows = exec_result.get("rows") or []
    row_count = exec_result.get("row_count", len(rows))
    executed = exec_result.get("sql_executed") or sql

    if not columns:
        return []

    type_oids = list(exec_result.get("type_oids", []))
    extra_notes: List[str] = []

    # Long-format results are cell rows, not a table. Widen them first, or the
    # reader gets row_index/col_index/value_text as the "chart".
    #
    # This is value-preserving by construction: out[row][column] = value, no
    # aggregation, so nothing can be silently combined.
    if is_long_format(columns):
        headers, wide, wide_oids = pivot_long(columns, rows)
        if wide and headers:
            columns, rows, type_oids = headers, wide, wide_oids
            row_count = len(wide)
            extra_notes.append("Widened from cell-level rows.")

    all_metas = describe_columns(columns, type_oids, rows)

    # Hide bookkeeping columns from the chart. The unit is kept aside and put
    # back on the artifact, because it belongs to the number rather than being
    # a column of its own.
    metas = [m for m in all_metas if m.key not in METADATA_COLUMNS] or all_metas
    unit = None
    if "unit" in {m.key for m in all_metas} and rows:
        unit_index = next(i for i, m in enumerate(all_metas) if m.key == "unit")
        unit = json_safe(rows[0][unit_index]) if unit_index < len(rows[0]) else None

    records = to_records(metas, rows, limit=MAX_DATA_ROWS)

    unordered, partial = _truncation_flags(executed, row_count)
    decision = classify(metas, records, unordered_truncation=unordered)

    sort = decision.get("sort")
    if sort:
        records = _sorted_by(records, sort[0], sort[1])

    artifact: Dict[str, Any] = {
        "id": "art_0",
        "schema_version": SCHEMA_VERSION,
        "kind": decision["kind"],
        "type": decision["type"],
        "title": (title or "").strip()[:120],
        "data": records,
        "encoding": decision["encoding"] or {"x": None, "y": [], "series": None},
        "columns": [_field(m) for m in metas],
        "value": None,
        "unit": unit,
        "provenance": {
            "sql": executed,
            "tables_used": [str(t) for t in (tables_used or [])],
            "row_count": row_count,
            "data_rows": len(records),
            "truncated": len(records) < row_count,
            "partial": partial,
            "source_documents": [
                {"source_file": s.get("source_file", ""), "score": s.get("score")}
                for s in (sources or [])
            ],
            "answer_is_llm_merged": llm_merged,
        },
        "notes": extra_notes + list(decision.get("notes") or []),
        "classifier": {"rule": decision["rule"], "version": SCHEMA_VERSION},
    }

    if decision["kind"] == "scalar" and records:
        first_key = metas[0].key
        artifact["value"] = records[0].get(first_key)

    if partial:
        artifact["notes"].append("Result hit the query limit; more rows may exist.")

    return [artifact]
