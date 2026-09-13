"""
Fact storage: ExtractedTable -> document_facts rows.

Replaces the old ingestion path, which asked Groq for a CREATE TABLE statement
per extracted table and executed that string unparameterised. Nothing here is
model-authored: the schema is fixed, the cell typing is deterministic
(ingest/fact_parser.py), and every value is a bound parameter.

The LLM is still used for one thing, and only one: a short natural-language
description of each table, so the catalogue can be searched semantically. That
is a label, not data, and a bad one degrades retrieval rather than corrupting
a number.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Dict, List, Optional, Sequence

from psycopg2.extras import execute_values

from indexial.core.db import get_connection
from indexial.ingest.fact_parser import CellValue, ColumnProfile, profile_table
from indexial.ingest.table_parser import ExtractedTable

logger = logging.getLogger(__name__)

# Cell-level storage multiplies row counts by the column count, so a guard is
# needed that the old per-table model did not need.
MAX_FACTS_PER_TABLE = 200_000


class FactStore:
    """Writes extracted tables into document_facts and table_registry."""

    def __init__(self, describe: bool = True):
        """
        Args:
            describe: ask the LLM for a semantic description per table.
                      Disable for bulk/offline ingest to stay inside the
                      free-tier token budget.
        """
        self.describe = describe
        self._describer = None

    # ------------------------------------------------------------ describe --

    def _semantic_description(self, table: ExtractedTable) -> str:
        """Short description for catalogue search. Falls back to headers."""
        fallback = (
            f"Table with {', '.join(table.headers[:3])}"
            + ("..." if len(table.headers) > 3 else "")
        )
        if not self.describe:
            return fallback
        try:
            if self._describer is None:
                from indexial.providers.llm import GroqSchemaGenerator

                self._describer = GroqSchemaGenerator()
            return self._describer.generate_semantic_description(
                table.headers, table.rows[:2]
            ) or fallback
        except Exception as exc:  # noqa: BLE001 - a label is never worth failing ingest
            logger.warning(f"Semantic description unavailable, using headers: {exc}")
            return fallback

    # -------------------------------------------------------------- ingest --

    def ingest_table(
        self,
        table: ExtractedTable,
        document_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Store one table as cell-level facts.

        Returns a summary dict, or None when the table has nothing to store.
        """
        if not table.headers or not table.rows:
            logger.warning(f"Skipping empty table {table.table_index}")
            return None

        profiles, parsed_rows = profile_table(table.headers, table.rows)
        label_profile = next((p for p in profiles if p.is_label), profiles[0] if profiles else None)
        label_index = label_profile.index if label_profile else 0

        estimated = sum(len(r) for r in parsed_rows)
        if estimated > MAX_FACTS_PER_TABLE:
            logger.warning(
                f"Table {table.table_index} would produce {estimated} facts; "
                f"truncating to {MAX_FACTS_PER_TABLE}"
            )

        table_id = str(uuid.uuid4())
        rows_to_insert: List[tuple] = []

        for row_index, parsed_row in enumerate(parsed_rows):
            # The row's label, copied onto every cell of the row. This is what
            # makes a flat WHERE possible instead of a self-join.
            row_label = None
            if label_index < len(parsed_row):
                row_label = (parsed_row[label_index].value_text or "").strip() or None

            page_number = table.page_of(row_index)
            ragged = len(parsed_row) != len(table.headers)

            for col_index, cell in enumerate(parsed_row):
                if len(rows_to_insert) >= MAX_FACTS_PER_TABLE:
                    break
                profile = profiles[col_index]
                note = cell.parse_note
                if ragged:
                    note = f"{note},ragged_row" if note else "ragged_row"

                rows_to_insert.append(
                    (
                        document_id,
                        table_id,
                        table.table_index,
                        row_index,
                        col_index,
                        page_number,
                        profile.name,
                        profile.key,
                        row_label,
                        cell.value_text,
                        cell.value_num,
                        cell.value_date,
                        cell.unit,
                        cell.scale_factor,
                        cell.value_type,
                        note,
                    )
                )

        if not rows_to_insert:
            logger.warning(f"Table {table.table_index} produced no facts")
            return None

        description = self._semantic_description(table)
        column_keys = [p.key for p in profiles]
        column_profiles = [
            {
                "key": p.key,
                "name": p.name,
                "type": p.value_type,
                "unit": p.unit,
                "scale": str(p.scale_factor) if p.scale_factor is not None else None,
                "numeric_coverage": round(p.numeric_coverage, 3),
                "distinct": p.distinct_count,
                "is_label": p.is_label,
            }
            for p in profiles
        ]

        try:
            with get_connection() as conn:
                cur = conn.cursor()

                cur.execute(
                    """
                    INSERT INTO table_registry (
                        table_id, document_id, table_index, original_filename,
                        semantic_description, headers, column_keys, column_profiles,
                        label_column_key, row_count, fact_count, page_start, page_end,
                        raw_markdown
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (document_id, table_index) DO UPDATE SET
                        semantic_description = EXCLUDED.semantic_description,
                        headers              = EXCLUDED.headers,
                        column_keys          = EXCLUDED.column_keys,
                        column_profiles      = EXCLUDED.column_profiles,
                        label_column_key     = EXCLUDED.label_column_key,
                        row_count            = EXCLUDED.row_count,
                        fact_count           = EXCLUDED.fact_count,
                        page_start           = EXCLUDED.page_start,
                        page_end             = EXCLUDED.page_end,
                        raw_markdown         = EXCLUDED.raw_markdown
                    RETURNING table_id
                    """,
                    (
                        table_id,
                        document_id,
                        table.table_index,
                        table.source_file,
                        description,
                        json.dumps(table.headers),
                        json.dumps(column_keys),
                        json.dumps(column_profiles),
                        column_keys[label_index] if label_index < len(column_keys) else None,
                        len(parsed_rows),
                        len(rows_to_insert),
                        table.page_start,
                        table.page_end,
                        table.raw_markdown,
                    ),
                )

                # An ON CONFLICT re-ingest keeps the existing table_id, so use
                # whatever came back rather than the one generated above.
                actual_table_id = str(cur.fetchone()[0])
                if actual_table_id != table_id:
                    cur.execute("DELETE FROM document_facts WHERE table_id = %s", (actual_table_id,))
                    rows_to_insert = [(r[0], actual_table_id, *r[2:]) for r in rows_to_insert]

                execute_values(
                    cur,
                    """
                    INSERT INTO document_facts (
                        document_id, table_id, table_index, row_index, col_index,
                        page_number, column_name, column_key, row_label,
                        value_text, value_num, value_date, unit, scale_factor,
                        value_type, parse_note
                    ) VALUES %s
                    ON CONFLICT (table_id, row_index, col_index) DO NOTHING
                    """,
                    rows_to_insert,
                    page_size=1000,
                )

                conn.commit()
                cur.close()

        except Exception as exc:  # noqa: BLE001
            logger.error(f"Failed to ingest table {table.table_index}: {exc}")
            return None

        logger.info(
            f"Ingested table {table.table_index}: {len(parsed_rows)} rows x "
            f"{len(table.headers)} cols = {len(rows_to_insert)} facts "
            f"(pages {table.page_start}-{table.page_end})"
        )

        return {
            "table_id": actual_table_id,
            "table_index": table.table_index,
            "description": description,
            "headers": table.headers,
            "column_keys": column_keys,
            "row_count": len(parsed_rows),
            "fact_count": len(rows_to_insert),
            "page_start": table.page_start,
            "page_end": table.page_end,
        }

    def ingest_all(
        self,
        tables: Sequence[ExtractedTable],
        document_id: str,
    ) -> List[Dict[str, Any]]:
        """Ingest every table, skipping the ones that fail."""
        results = []
        for table in tables:
            result = self.ingest_table(table, document_id)
            if result:
                results.append(result)

        total_facts = sum(r["fact_count"] for r in results)
        logger.info(
            f"Ingested {len(results)}/{len(tables)} tables "
            f"({total_facts} facts) for document {document_id}"
        )
        return results
