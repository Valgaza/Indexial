"""
NL-to-SQL over the fact store.

Three pieces:

  FactCatalog       bounded, searchable description of what tables exist
  SQLGenerator      question -> SQL, by slot filling or free-form generation
  SafeSQLExecutor   validation, limits, timeout, typed results

The prompt is now a constant. It used to interpolate the whole table registry
untruncated, so prompt size - and with it SQL accuracy - degraded with every
document uploaded. With one fixed schema only a small catalogue slice varies,
and that slice is capped at three tables. This matters more than it looks on
the free tier: 8000 tokens per minute is the ceiling and a HYBRID query makes
four model calls.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from indexial.core import config
from indexial.core.db import get_connection
from indexial.providers.llm import GroqLLM
from indexial.query import sql_templates
from indexial.query.sql_validator import enforce_limit, validate_sql

logger = logging.getLogger(__name__)


class FactCatalog:
    """
    What tables exist, and which are relevant to a question.

    Returns at most `limit` tables at roughly 120 tokens each, so the prompt is
    the same size whether the corpus holds one document or a hundred.
    Candidates are ranked by full-text match over the facts themselves plus the
    table description, so the model picks from real column keys and row labels
    instead of inventing them.
    """

    def __init__(self, limit: int = 3, sample_labels: int = 5):
        self.limit = limit
        self.sample_labels = sample_labels

    def candidates(
        self,
        query: str,
        document_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Rank tables against a question; falls back to most recent."""
        sql = """
            WITH scored AS (
                SELECT r.table_id, r.document_id, r.table_index,
                       r.original_filename, r.semantic_description,
                       r.headers, r.column_keys, r.label_column_key,
                       r.row_count, r.page_start, r.page_end, r.created_at,
                       COALESCE((
                           SELECT count(*) FROM document_facts f
                           WHERE f.table_id = r.table_id
                             AND f.fts @@ plainto_tsquery('english', %(q)s)
                       ), 0) AS hits
                FROM table_registry r
                WHERE (%(doc_ids)s::text[] IS NULL
                       OR r.document_id::text = ANY(%(doc_ids)s::text[]))
            )
            SELECT * FROM scored
            ORDER BY hits DESC, created_at DESC
            LIMIT %(limit)s
        """
        params = {"q": query or "", "doc_ids": document_ids, "limit": self.limit}

        try:
            with get_connection(readonly=True) as conn:
                cur = conn.cursor()
                cur.execute(sql, params)
                names = [d[0] for d in cur.description]
                tables = [dict(zip(names, row)) for row in cur.fetchall()]

                for table in tables:
                    cur.execute(
                        """
                        SELECT DISTINCT row_label FROM document_facts
                        WHERE table_id = %s AND row_label IS NOT NULL
                        ORDER BY row_label LIMIT %s
                        """,
                        (table["table_id"], self.sample_labels),
                    )
                    table["sample_labels"] = [r[0] for r in cur.fetchall()]
                cur.close()
            return tables
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Catalogue lookup failed: {exc}")
            return []

    @staticmethod
    def render(tables: List[Dict[str, Any]]) -> str:
        """Compact catalogue text for the prompt."""
        if not tables:
            return "No tables have been extracted yet."

        blocks = []
        for table in tables:
            keys = table.get("column_keys") or []
            labels = table.get("sample_labels") or []
            pages = ""
            if table.get("page_start"):
                pages = (
                    f", page {table['page_start']}"
                    if table["page_start"] == table.get("page_end")
                    else f", pages {table['page_start']}-{table.get('page_end')}"
                )
            blocks.append(
                f"table_id: {table['table_id']}\n"
                f"  about: {table.get('semantic_description') or 'unlabelled table'}\n"
                f"  from: {table.get('original_filename') or 'unknown'}{pages}, "
                f"{table.get('row_count', 0)} rows\n"
                f"  column_key options: {', '.join(keys) if keys else 'unknown'}\n"
                f"  example row_labels: {', '.join(labels) if labels else 'none'}"
            )
        return "\n\n".join(blocks)

    # -------------------------------------------------- compat helpers ------

    def get_table_list(self, document_ids: Optional[List[str]] = None) -> List[str]:
        """Table ids currently in the registry."""
        sql = "SELECT table_id FROM table_registry"
        params: Tuple = ()
        if document_ids:
            sql += " WHERE document_id::text = ANY(%s)"
            params = (document_ids,)
        try:
            with get_connection(readonly=True) as conn:
                cur = conn.cursor()
                cur.execute(sql, params)
                out = [str(r[0]) for r in cur.fetchall()]
                cur.close()
            return out
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Table list failed: {exc}")
            return []

    def get_registry_context(self, document_ids: Optional[List[str]] = None) -> str:
        """Whole-catalogue text. Prefer candidates() on the hot path."""
        return self.render(self.candidates("", document_ids))


# Name used by app.py and router.py before the rewrite.
TableRegistryReader = FactCatalog


class SQLGenerator:
    """
    Question -> SQL.

    Primary path is slot filling: the model returns a small JSON object naming
    a pattern and its parameters, and Python renders parameterised SQL from a
    template. Free-form generation is available behind SQL_STRATEGY and runs
    through the same validator.

    Both paths exist deliberately. The accuracy figures that motivated slot
    filling are estimates for a model of this size rather than measurements of
    gpt-oss-20b, so `stats` records what actually happens and the default can
    be settled with data instead of argument.
    """

    def __init__(self, strategy: Optional[str] = None):
        self.llm = GroqLLM()
        self.catalog = FactCatalog()
        self.strategy = strategy or config.SQL_STRATEGY
        self.stats: Dict[str, int] = {
            "slot_attempts": 0,
            "slot_ok": 0,
            "freeform_attempts": 0,
            "freeform_ok": 0,
        }

    def generate(
        self,
        query: str,
        document_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Returns {sql, params, pattern, tables_used, catalog, explanation, error}."""
        tables = self.catalog.candidates(query, document_ids)
        if not tables:
            return {
                "sql": None,
                "error": "No tables have been extracted yet.",
                "explanation": "Upload a document containing tables first.",
                "catalog": [],
            }

        catalog_text = self.catalog.render(tables)
        table_ids = [str(t["table_id"]) for t in tables]

        if self.strategy == "freeform":
            result = self._generate_freeform(query, catalog_text)
            result.setdefault("tables_used", table_ids[:1])
        else:
            result = self._generate_slots(query, catalog_text, table_ids)

        result["catalog"] = tables
        return result

    # Kept for callers written against the previous name.
    def generate_sql(
        self, query: str, document_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        return self.generate(query, document_ids)

    # --------------------------------------------------------- slots -------

    def _generate_slots(
        self,
        query: str,
        catalog_text: str,
        table_ids: List[str],
    ) -> Dict[str, Any]:
        self.stats["slot_attempts"] += 1

        messages = [
            {"role": "system", "content": sql_templates.FACT_SCHEMA_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Available tables:\n\n{catalog_text}\n\n"
                    f"Question: {query}\n\nReturn only the JSON slots."
                ),
            },
        ]

        try:
            raw = self.llm.chat(
                messages,
                temperature=0,
                max_tokens=2000,
                response_format={"type": "json_object"},
            )
            slots = json.loads(raw)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Slot generation failed: {exc}")
            return {
                "sql": None,
                "error": str(exc),
                "explanation": "Could not interpret the question.",
            }

        # The model occasionally answers with a table_id that is not in the
        # catalogue. Pin it to the best candidate rather than fail outright.
        if slots.get("table_id") not in table_ids:
            slots["table_id"] = table_ids[0]

        try:
            sql, params = sql_templates.render(slots)
        except sql_templates.PatternError as exc:
            logger.warning(f"Unusable slots {slots}: {exc}")
            return {
                "sql": None,
                "error": str(exc),
                "pattern": slots.get("pattern"),
                "tables_used": [slots.get("table_id")],
                "explanation": "Could not map the question onto a known query shape.",
            }

        self.stats["slot_ok"] += 1
        logger.info(f"SQL pattern={slots.get('pattern')} table={str(slots['table_id'])[:8]}")
        return {
            "sql": sql,
            "params": params,
            "pattern": slots.get("pattern"),
            "slots": slots,
            "tables_used": [slots["table_id"]],
            "explanation": f"Matched the '{slots.get('pattern')}' query pattern.",
        }

    # ------------------------------------------------------ free-form -------

    def _generate_freeform(self, query: str, catalog_text: str) -> Dict[str, Any]:
        self.stats["freeform_attempts"] += 1

        system = (
            sql_templates.FACT_SCHEMA_PROMPT
            + "\n\nInstead of slots, write a single PostgreSQL SELECT.\n"
            "HARD RULES\n"
            "1. Always include table_id = '<uuid>' in the WHERE clause.\n"
            "2. Numbers live in value_num; add value_num IS NOT NULL.\n"
            "3. To place columns side by side use\n"
            "   agg(value_num) FILTER (WHERE column_key='x') with GROUP BY row_label.\n"
            "4. NEVER join document_facts to itself.\n"
            'Return JSON: {"sql": "...", "explanation": "..."}'
        )

        try:
            raw = self.llm.chat(
                [
                    {"role": "system", "content": system},
                    {
                        "role": "user",
                        "content": f"Available tables:\n\n{catalog_text}\n\nQuestion: {query}",
                    },
                ],
                temperature=0,
                max_tokens=2500,
                response_format={"type": "json_object"},
            )
            payload = json.loads(raw)
            sql = (payload.get("sql") or "").strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Free-form generation failed: {exc}")
            return {"sql": None, "error": str(exc), "explanation": "Could not generate SQL."}

        if not sql:
            return {"sql": None, "error": "empty SQL", "explanation": "Could not generate SQL."}

        self.stats["freeform_ok"] += 1
        return {
            "sql": sql,
            "params": None,
            "pattern": "custom",
            "explanation": payload.get("explanation", ""),
        }

    def repair(self, sql: str, error: str, query: str) -> Optional[str]:
        """
        One self-repair attempt after an execution failure.

        There was no retry at all before: a single malformed query became a
        failed answer. Capped at one attempt, because the budget is 8000 tokens
        per minute.
        """
        try:
            raw = self.llm.chat(
                [
                    {
                        "role": "system",
                        "content": (
                            "Fix the PostgreSQL query, keeping the same intent. It must "
                            "remain a single SELECT over document_facts and keep its "
                            'table_id filter.\nReturn JSON: {"sql": "..."}'
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Question: {query}\n\nSQL:\n{sql}\n\nPostgres error:\n{error}",
                    },
                ],
                temperature=0,
                max_tokens=2500,
                response_format={"type": "json_object"},
            )
            return (json.loads(raw).get("sql") or "").strip() or None
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"SQL repair failed: {exc}")
            return None


class SafeSQLExecutor:
    """
    Runs generated SQL under validation, a row cap and a statement timeout.

    Layers, in order:
      1. AST validation      query/sql_validator.py
      2. Outermost LIMIT     top-level query only
      3. Read-only session   core/db.py
      4. statement_timeout   bounded execution
    """

    def __init__(self, max_rows: Optional[int] = None):
        self.max_rows = max_rows or config.SQL_MAX_ROWS
        self.timeout = config.SQL_STATEMENT_TIMEOUT

    def validate_sql(self, sql: str) -> Tuple[bool, Optional[str]]:
        """Kept as a method for callers written against the old class."""
        return validate_sql(sql, self.max_rows)

    def execute(
        self,
        sql: str,
        params: Optional[Dict[str, Any]] = None,
        max_rows: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Returns {success, columns, type_oids, rows, records, row_count,
                 sql_executed, error}.

        type_oids is new and load-bearing: the artifact classifier types
        columns from Postgres' own type codes rather than guessing from values.
        records is the same data as dicts, used by both the pivot and the JSON
        response.
        """
        cap = max_rows or self.max_rows
        empty = {
            "success": False,
            "columns": [],
            "type_oids": [],
            "rows": [],
            "records": [],
            "row_count": 0,
            "sql_executed": sql,
        }

        ok, error = validate_sql(sql, cap)
        if not ok:
            logger.warning(f"SQL rejected: {error}")
            return {**empty, "error": error}

        try:
            limited = enforce_limit(sql, cap)
        except Exception:  # noqa: BLE001
            limited = sql

        try:
            with get_connection(readonly=True) as conn:
                cur = conn.cursor()
                cur.execute(f"SET statement_timeout = '{self.timeout}'")
                cur.execute(limited, params or {})

                description = cur.description or []
                columns = [d[0] for d in description]
                type_oids = [d[1] for d in description]
                rows = cur.fetchall()
                cur.close()

            return {
                "success": True,
                "columns": columns,
                "type_oids": type_oids,
                "rows": rows,
                "records": [dict(zip(columns, row)) for row in rows],
                "row_count": len(rows),
                "sql_executed": limited,
            }

        except Exception as exc:  # noqa: BLE001
            logger.error(f"SQL execution failed: {exc}")
            return {**empty, "sql_executed": limited, "error": str(exc)}

    # ------------------------------------------------------- formatting ----

    def format_results(
        self,
        results: Dict[str, Any],
        original_query: str,
        slots: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Turn a result set into prose.

        Two fixes over the previous version. It called str() on raw psycopg2
        values, so the model saw Decimal('120.50') and sometimes echoed that
        syntax into the answer; it now sees the same sanitised values the chart
        does. And it is told how many rows it is actually looking at, because
        it sees ten while the chart shows up to two hundred - without that it
        will confidently describe a maximum it never saw.
        """
        if not results.get("success"):
            return f"Query failed: {results.get('error', 'Unknown error')}"
        if results["row_count"] == 0:
            return "No results found for your query."

        from indexial.query import artifacts

        columns = results["columns"]
        metas = artifacts.describe_columns(
            columns, results.get("type_oids", []), results["rows"]
        )
        records = artifacts.to_records(metas, results["rows"], limit=10)

        aggregate = self._aggregate_summary(records, slots)
        if aggregate:
            # An aggregate arrives as result/unit/parsed_cells/total_cells.
            # Handed over as a bare table the model reads those bookkeeping
            # columns as noise and answers that it has no information, despite
            # holding the number. Stated plainly it answers the question.
            context = aggregate
        else:
            # The templates project generic names - label, value, series_n -
            # because they have to work for any table. Handed over as-is the
            # model cannot tell that "value" means "Q3 2024" and refuses to
            # answer, while the chart beside it shows the right numbers. Naming
            # the columns after the slots that produced them fixes that.
            display = self._display_columns(columns, slots)
            header = " | ".join(display)
            lines = [f"Found {results['row_count']} result(s).", "", header, "-" * len(header)]
            for record in records:
                lines.append(" | ".join("" if v is None else str(v) for v in record.values()))

            if results["row_count"] > 10:
                lines.append(
                    f"\n(Showing the first 10 of {results['row_count']} rows. Do not describe "
                    "maxima, minima or rankings as if you had seen every row.)"
                )

            note = self._coverage_note(records)
            if note:
                lines.append(note)
            context = "\n".join(lines)

        return GroqLLM().generate_answer(
            original_query, context, max_tokens=600, temperature=0.3
        )

    @staticmethod
    def _display_columns(
        columns: List[str],
        slots: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """Rename a template's generic projections after the slots used."""
        if not slots:
            return list(columns)

        column_key = slots.get("column_key")
        keys = slots.get("column_keys") or []
        out = []
        for name in columns:
            if name == "label":
                out.append("Row")
            elif name == "value" and column_key:
                out.append(str(column_key))
            elif name.startswith("y_") and name[2:] in [str(k) for k in keys]:
                out.append(name[2:])
            elif name in [str(k) for k in keys]:
                out.append(name)
            else:
                out.append(name)
        return out

    @staticmethod
    def _aggregate_summary(
        records: List[Dict[str, Any]],
        slots: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Render a single-row aggregate as a sentence the model can use.

        The slots matter. Given only "the computed value is 6,907" the model
        cannot tell whether that answers the question and replies that it has
        no information - while the chart beside it shows the right number.
        Naming the operation and the column removes the ambiguity.
        """
        if len(records) != 1 or "result" not in records[0]:
            return None

        row = records[0]
        value, unit = row.get("result"), row.get("unit")

        what = "The requested value"
        if slots:
            agg = str(slots.get("agg", "")).upper()
            column = slots.get("column_key")
            verb = {
                "SUM": "total", "AVG": "average", "MIN": "minimum",
                "MAX": "maximum", "COUNT": "count",
            }.get(agg, "value")
            if column:
                what = f"The {verb} of the '{column}' column"
                if slots.get("row_filter"):
                    what += f" for rows matching '{slots['row_filter']}'"

        if value is None:
            return f"{what} could not be computed: no cell in that column could be read as a number."

        rendered = f"{value:,}" if isinstance(value, (int, float)) else str(value)
        if unit == "%":
            rendered += "%"
        elif unit:
            rendered = f"{rendered} {unit}"

        text = (
            f"{what} is {rendered}.\n\n"
            "This figure was computed directly from the extracted table data and is correct. "
            "Report it as the answer."
        )

        parsed, total = row.get("parsed_cells"), row.get("total_cells")
        if isinstance(parsed, int) and isinstance(total, int) and total and parsed < total:
            text += (
                f"\n\nCaveat: it covers {parsed} of {total} cells; {total - parsed} could not "
                "be read as numbers, so the figure is incomplete. Say so in the answer."
            )
        return text

    @staticmethod
    def _coverage_note(records: List[Dict[str, Any]]) -> str:
        """
        Surface partial parsing.

        A column that parsed at 60% numeric coverage holds NULLs, and SUM over
        it returns a confident, too-small total. The aggregate templates return
        parsed_cells/total_cells precisely so the answer can say so rather than
        quietly under-report.
        """
        if not records:
            return ""
        first = records[0]
        parsed, total = first.get("parsed_cells"), first.get("total_cells")
        if isinstance(parsed, int) and isinstance(total, int) and total and parsed < total:
            return (
                f"\nIMPORTANT: this figure covers {parsed} of {total} cells; "
                f"{total - parsed} could not be read as numbers. Say so in the answer."
            )
        return ""
