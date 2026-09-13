"""
SQL templates and the slot-filling contract.

Why templates rather than asking the model for SQL:

A small model is markedly better at structured extraction than at SQL
composition, and long-format storage makes the SQL harder, not easier. Asking
for a two-series comparison directly means asking for repeated
`FILTER (WHERE ...)` expressions, because Postgres will not let the delta
expression reference the aliases beside it. That is the query users ask for
most and the one the model is worst at.

So the model returns SLOTS - a small JSON object naming a pattern and its
parameters - and Python renders parameterised SQL from the template. The SQL
is ours; only the values come from the model, and they are bound, never
interpolated.

Second reason, equally important: templates control the SHAPE of the result.
The artifact classifier in query/artifacts.py can only draw a chart from a
crosstab-shaped result set, so comparison and trend patterns emit one column
per series rather than long rows. Left to free-form SQL the result shape is
unpredictable and most answers degrade to a plain table.

Every aggregate carries coverage counts. A column that parsed at 60% numeric
coverage silently holds NULLs, and SUM over it returns a confident, wrong,
too-small total. parsed_cells/total_cells is what lets the answer say "summed
28 of 40 cells" instead of quietly under-reporting.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Tuple

__all__ = ["FACT_SCHEMA_PROMPT", "PATTERNS", "render", "PatternError"]


class PatternError(ValueError):
    """The model's slots did not describe a runnable query."""


FACT_SCHEMA_PROMPT = """You answer questions about tables extracted from documents.
All table data lives in ONE table whose schema never changes:

document_facts(
  document_id uuid, table_id uuid, table_index int,
  row_index int, col_index int, page_number int,
  column_name text,   -- the header exactly as printed
  column_key  text,   -- normalised header: '2023', 'segment', 'net_revenue'
  row_label   text,   -- the row's label, copied onto EVERY cell of that row
  value_text  text,   -- the cell as printed; always present
  value_num   numeric,-- the number in base units, NULL if not numeric
  value_date  date, unit text, value_type text
)

One ROW of the original document table becomes MANY rows here, all sharing
(table_id, row_index). To find a row, filter on row_label. To find a column,
filter on column_key.

Choose ONE pattern and return ONLY its JSON slots.

cell_lookup     one value        {"pattern":"cell_lookup","table_id":..,"column_key":..,"row_filter":..}
scalar_aggregate one number      {"pattern":"scalar_aggregate","table_id":..,"agg":"SUM|AVG|MIN|MAX|COUNT","column_key":..,"row_filter":null}
row_filter      matching rows    {"pattern":"row_filter","table_id":..,"row_filter":..}
column_series   label vs value   {"pattern":"column_series","table_id":..,"column_key":..,"order":"desc|asc|label"}
comparison      2+ columns       {"pattern":"comparison","table_id":..,"column_keys":["2022","2023"]}
list_table      whole table      {"pattern":"list_table","table_id":..}

Rules:
- table_id is REQUIRED and must be copied verbatim from the catalogue.
- row_filter is a substring matched case-insensitively against row_label. Use
  null for "all rows".
- column_key must be copied verbatim from the catalogue's column list.
- Never invent a column_key or a table_id.
"""

_AGGS = {"SUM", "AVG", "MIN", "MAX", "COUNT"}
_MAX_ROWS = 500


def _table_id(slots: Dict[str, Any]) -> str:
    table_id = slots.get("table_id")
    if not table_id or not isinstance(table_id, str):
        raise PatternError("table_id is required")
    return table_id


def _like(value: Any) -> Any:
    return f"%{value}%" if value else None


# --------------------------------------------------------------- patterns --

def _cell_lookup(slots: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """One value. The model gets this right most of the time even unaided."""
    sql = """
        SELECT row_label, column_name, value_text, value_num, unit, page_number
        FROM document_facts
        WHERE table_id = %(table_id)s
          AND column_key = %(column_key)s
          AND (%(row_filter)s IS NULL OR row_label ILIKE %(row_like)s)
        ORDER BY row_index
        LIMIT 25
    """
    rf = slots.get("row_filter")
    return sql, {
        "table_id": _table_id(slots),
        "column_key": slots.get("column_key"),
        "row_filter": rf,
        "row_like": _like(rf),
    }


def _scalar_aggregate(slots: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """
    One number, plus how much of the column it was actually computed from.

    value_type <> 'year' matters: without it, "total of the year column"
    returns 4045 for two rows of 2022 and 2023. The model reliably forgets
    this, which is exactly why it is in the template and not in the prompt.
    """
    agg = str(slots.get("agg", "SUM")).upper()
    if agg not in _AGGS:
        raise PatternError(f"unsupported aggregate {agg!r}")

    sql = f"""
        SELECT {agg}(value_num)                                AS result,
               min(unit)                                       AS unit,
               count(*) FILTER (WHERE value_num IS NOT NULL)    AS parsed_cells,
               count(*)                                        AS total_cells
        FROM document_facts
        WHERE table_id = %(table_id)s
          AND column_key = %(column_key)s
          AND value_type <> 'year'
          AND (%(row_filter)s IS NULL OR row_label ILIKE %(row_like)s)
    """
    rf = slots.get("row_filter")
    return sql, {
        "table_id": _table_id(slots),
        "column_key": slots.get("column_key"),
        "row_filter": rf,
        "row_like": _like(rf),
    }


def _row_filter(slots: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """
    Whole rows matching a label, returned wide.

    The naive version of this puts the predicate on the outer table and gets
    back only the matching cells rather than the matching rows. The semi-join
    on row_index is the part the model gets wrong, so it is templated.
    """
    sql = """
        WITH matching AS (
            SELECT DISTINCT row_index
            FROM document_facts
            WHERE table_id = %(table_id)s
              AND row_label ILIKE %(row_like)s
            LIMIT %(max_rows)s
        )
        SELECT f.row_index, f.col_index, f.column_name, f.column_key,
               f.value_text, f.value_num, f.unit, f.page_number
        FROM document_facts f
        JOIN matching m ON m.row_index = f.row_index
        WHERE f.table_id = %(table_id)s
        ORDER BY f.row_index, f.col_index
    """
    rf = slots.get("row_filter")
    if not rf:
        raise PatternError("row_filter is required for row_filter")
    return sql, {"table_id": _table_id(slots), "row_like": _like(rf), "max_rows": _MAX_ROWS}


def _column_series(slots: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """
    label -> value. Two columns, chart-ready: the bar chart case.
    """
    order = str(slots.get("order", "desc")).lower()
    order_sql = {
        "asc": "value ASC NULLS LAST",
        "label": "label ASC",
    }.get(order, "value DESC NULLS LAST")

    sql = f"""
        SELECT row_label                      AS label,
               max(value_num)                 AS value,
               min(unit)                      AS unit
        FROM document_facts
        WHERE table_id = %(table_id)s
          AND column_key = %(column_key)s
          AND value_num IS NOT NULL
          AND row_label IS NOT NULL
        GROUP BY row_label
        ORDER BY {order_sql}
        LIMIT %(max_rows)s
    """
    return sql, {
        "table_id": _table_id(slots),
        "column_key": slots.get("column_key"),
        "max_rows": _MAX_ROWS,
    }


def _comparison(slots: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """
    Several columns side by side, one column per series.

    This is the pattern users ask for most and the one free-form generation
    fails at hardest, because the repeated FILTER expressions cannot be
    replaced by aliases. It is also the one that must come back crosstab-shaped
    for the artifact classifier to draw a grouped bar or line chart, so the
    shape is fixed here rather than left to the model.
    """
    keys = slots.get("column_keys") or []
    if not isinstance(keys, list) or not 1 <= len(keys) <= 5:
        raise PatternError("column_keys must be a list of 1-5 column keys")

    params: Dict[str, Any] = {"table_id": _table_id(slots), "max_rows": _MAX_ROWS}
    projections = []
    for i, key in enumerate(keys):
        params[f"k{i}"] = key
        # The alias is the column key itself, so a chart legend reads "2023"
        # rather than "series_1". An alias cannot be a bound parameter, so it
        # is sanitised to [a-z0-9_] and prefixed when it starts with a digit -
        # the value is still bound separately in the FILTER clause.
        alias = re.sub(r"[^a-z0-9_]", "_", str(key).lower()) or f"series_{i}"
        if alias[0].isdigit():
            alias = f"y_{alias}"
        projections.append(
            f'max(value_num) FILTER (WHERE column_key = %(k{i})s) AS "{alias}"'
        )

    sql = f"""
        SELECT row_label AS label,
               {', '.join(projections)}
        FROM document_facts
        WHERE table_id = %(table_id)s
          AND column_key = ANY(%(all_keys)s)
          AND row_label IS NOT NULL
        GROUP BY row_label
        ORDER BY row_label
        LIMIT %(max_rows)s
    """
    params["all_keys"] = keys
    return sql, params


def _list_table(slots: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """
    The whole table, long. query/pivot.py widens it in Python.

    Deliberately not pivoted in SQL: a wide projection needs one FILTER clause
    per column, which means knowing the columns at render time. Python already
    has them from the registry, so it does the work instead.
    """
    sql = """
        SELECT row_index, col_index, column_name, column_key, row_label,
               value_text, value_num, unit, page_number
        FROM document_facts
        WHERE table_id = %(table_id)s
          AND row_index < %(max_rows)s
        ORDER BY row_index, col_index
    """
    return sql, {"table_id": _table_id(slots), "max_rows": _MAX_ROWS}


PATTERNS: Dict[str, Callable[[Dict[str, Any]], Tuple[str, Dict[str, Any]]]] = {
    "cell_lookup": _cell_lookup,
    "scalar_aggregate": _scalar_aggregate,
    "row_filter": _row_filter,
    "column_series": _column_series,
    "comparison": _comparison,
    "list_table": _list_table,
}


def render(slots: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """
    Turn model-supplied slots into (sql, params).

    Raises PatternError when the slots do not describe a runnable query, which
    the caller turns into a retry or a fallback rather than a failed answer.
    """
    if not isinstance(slots, dict):
        raise PatternError("slots must be an object")
    name = slots.get("pattern")
    if name not in PATTERNS:
        raise PatternError(f"unknown pattern {name!r}")
    return PATTERNS[name](slots)
