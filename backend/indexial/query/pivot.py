"""
Long-format rows -> wide result set, in Python.

This is the escape hatch that makes long-format storage tolerable for display.
The alternative - a materialised wide view per table - would reintroduce
exactly the dynamic DDL machinery the fact store exists to delete.

Pure functions: no database, no LLM.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = ["pivot_long", "is_long_format"]

# Column names produced by sql_templates._list_table.
_ROW = "row_index"
_COL_KEY = "column_key"
_COL_NAME = "column_name"
_TEXT = "value_text"
_NUM = "value_num"

# Postgres OIDs the widened columns are reported as, so the artifact
# classifier types them exactly as it would a crosstab from SQL.
NUMERIC_OID = 1700  # numeric
TEXT_OID = 25  # text


def is_long_format(columns: Sequence[str]) -> bool:
    """Whether a result set looks like raw cell rows rather than a crosstab."""
    lowered = {c.lower() for c in columns}
    return _ROW in lowered and _COL_KEY in lowered and (_TEXT in lowered or _NUM in lowered)


def pivot_long(
    columns: Sequence[str],
    rows: Sequence[Sequence[Any]],
    column_keys: Optional[Sequence[str]] = None,
    column_names: Optional[Sequence[str]] = None,
) -> Tuple[List[str], List[List[Any]], List[int]]:
    """
    Widen cell rows into one row per row_index.

    Args:
        columns: result-set column names
        rows: result-set rows
        column_keys: the table's column keys, in original order. When omitted,
                     order of first appearance is used, which is correct for a
                     query that ordered by col_index.
        column_names: display headers matching column_keys.

    Returns:
        (headers, wide_rows, type_oids)

    Prefers value_num over value_text so numeric columns stay numeric: the
    artifact classifier types columns from the result set, and a numeric column
    arriving as strings would be classified categorical and never charted.

    The returned type_oids are derived from WHICH source column supplied each
    value - value_num or value_text - not from inspecting the text, so this
    stays a lookup rather than a guess.
    """
    index = {name.lower(): i for i, name in enumerate(columns)}
    if not is_long_format(columns):
        return list(columns), [list(r) for r in rows], []

    i_row = index[_ROW]
    i_key = index[_COL_KEY]
    i_text = index.get(_TEXT)
    i_num = index.get(_NUM)
    i_name = index.get(_COL_NAME)

    grid: Dict[Any, Dict[str, Any]] = {}
    seen_keys: List[str] = []
    label_by_key: Dict[str, str] = {}
    numeric_keys: set = set()

    for row in rows:
        key = row[i_key]
        if key not in label_by_key:
            seen_keys.append(key)
            label_by_key[key] = row[i_name] if i_name is not None else key

        value = None
        if i_num is not None and row[i_num] is not None:
            value = row[i_num]
            numeric_keys.add(key)
        elif i_text is not None:
            value = row[i_text]

        grid.setdefault(row[i_row], {})[key] = value

    keys = list(column_keys) if column_keys else seen_keys
    headers = list(column_names) if column_names else [label_by_key.get(k, k) for k in keys]

    wide = [[grid[r].get(k) for k in keys] for r in sorted(grid)]
    oids = [NUMERIC_OID if k in numeric_keys else TEXT_OID for k in keys]
    return headers, wide, oids
