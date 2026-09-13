"""
SQL validation on a parsed AST.

Replaces three regex checks, each of which was independently bypassable:

  (a) The table whitelist accepted any name merely starting with "tbl_",
      so an attacker-chosen relation passed by picking its own name.
  (b) Multi-statement detection counted semicolons and rejected only
      `> 1`, so `SELECT 1; DROP TABLE documents` - exactly one semicolon -
      went straight through.
  (c) The keyword blocklist scanned the whole uppercased query including
      string literals and identifiers, so a column named "update" was
      rejected while `/**/` comment tricks were not.

Parsing fixes all three structurally rather than patching each with a better
regex: `exp.Update` is a statement node, `exp.Column(this="update")` is a
column, and a parser cannot be fooled by quoting or comments.
"""

from __future__ import annotations

import logging
import re
from typing import Optional, Tuple

import sqlglot
from sqlglot import exp

logger = logging.getLogger(__name__)

__all__ = ["validate_sql", "enforce_limit", "ALLOWED_RELATIONS"]

# With a fixed schema the whitelist is a constant instead of a database
# lookup - simpler, and strictly safer than the old prefix test.
ALLOWED_RELATIONS = frozenset({"document_facts", "table_registry", "documents"})

# Functions that read the filesystem, reach the network, or change session
# state. None of these are DML, so a statement-type check alone misses them.
FORBIDDEN_FUNCTIONS = frozenset({
    "pg_read_file", "pg_read_binary_file", "pg_ls_dir", "pg_stat_file",
    "pg_sleep", "lo_import", "lo_export", "dblink", "dblink_exec",
    "query_to_xml", "set_config", "pg_terminate_backend", "pg_cancel_backend",
})

WRITE_NODES = (
    exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create, exp.Alter,
    exp.Command, exp.Grant, exp.Merge, exp.Set, exp.Transaction,
    exp.Commit, exp.Rollback,
)

READ_ROOTS = (exp.Select, exp.Union, exp.Except, exp.Intersect, exp.Subquery, exp.With)

# A query with no scope predicate scans every fact of every document. In the
# old per-table world the FROM clause was the scope; long format has none.
SCOPE_COLUMNS = {"table_id", "document_id"}


def validate_sql(sql: str, max_rows: int = 1000) -> Tuple[bool, Optional[str]]:
    """
    Check a generated SELECT before it is executed.

    Returns (ok, error_message).
    """
    if not sql or not sql.strip():
        return False, "Empty SQL query"

    # (b) Statement count, structurally. Comments are consumed by the parser
    # rather than smuggled past a regex.
    try:
        statements = [s for s in sqlglot.parse(sql, read="postgres") if s is not None]
    except Exception as exc:  # noqa: BLE001 - sqlglot raises several types
        return False, f"Could not parse SQL: {str(exc)[:150]}"

    if len(statements) != 1:
        return False, f"Exactly one statement allowed, found {len(statements)}"

    root = statements[0]

    if not isinstance(root, READ_ROOTS):
        return False, f"Only SELECT queries are allowed, got {type(root).__name__}"

    # (c) Operations by node type, so a column named "update" is fine and a
    # quoted literal containing "DROP" is fine, while a real DROP is not.
    for node in root.walk():
        if isinstance(node, WRITE_NODES):
            return False, f"Forbidden operation: {type(node).__name__.upper()}"
        if isinstance(node, (exp.Anonymous, exp.Func)):
            # An unrecognised function parses as Anonymous, whose sql_name() is
            # the literal string "ANONYMOUS"; the real name lives in `this`.
            # Every function of interest here is one sqlglot does not know.
            raw = node.this if isinstance(getattr(node, "this", None), str) else node.sql_name()
            if (raw or "").lower() in FORBIDDEN_FUNCTIONS:
                return False, f"Forbidden function: {raw}"

    # (a) Whitelist by exact membership. No prefix escape hatch, and a
    # non-public schema is refused outright - which also removes the old
    # INFORMATION_SCHEMA exemption that leaked other tables' names.
    cte_names = {cte.alias_or_name.lower() for cte in root.find_all(exp.CTE)}
    for table in root.find_all(exp.Table):
        name = (table.name or "").lower()
        if name in cte_names:
            continue
        if table.db and table.db.lower() not in ("public", ""):
            return False, f"Schema '{table.db}' is not allowed"
        if name not in ALLOWED_RELATIONS:
            return False, (
                f"Relation '{table.name}' is not allowed. "
                f"Queryable: {', '.join(sorted(ALLOWED_RELATIONS))}"
            )

    # Mandatory scope predicate.
    columns = {(c.name or "").lower() for c in root.find_all(exp.Column)}
    if not (columns & SCOPE_COLUMNS):
        return False, "Query must filter on table_id or document_id"

    return True, None


def enforce_limit(sql: str, max_rows: int) -> str:
    """
    Cap the OUTERMOST limit only.

    The previous implementation ran `re.sub(r"LIMIT\\s+\\d+", ...)` over the
    whole string, which rewrote limits inside subqueries too and silently
    changed what the query meant.
    """
    try:
        parsed = sqlglot.parse_one(sql, read="postgres")
    except Exception:  # noqa: BLE001 - fall back to the string form
        return sql if re.search(r"\bLIMIT\b", sql, re.I) else f"{sql.rstrip().rstrip(';')} LIMIT {max_rows}"

    existing = parsed.args.get("limit")
    if existing is not None:
        try:
            current = int(existing.expression.this)
            if current <= max_rows:
                return parsed.sql(dialect="postgres")
        except (AttributeError, TypeError, ValueError):
            pass

    return parsed.limit(max_rows).sql(dialect="postgres")
