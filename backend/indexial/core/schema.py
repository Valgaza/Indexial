"""
Database schema.

Every CREATE statement in the project lives here, so the shape of the database
can be read in one place instead of being reconstructed from three modules'
lazy `CREATE TABLE IF NOT EXISTS` calls.

The design decision that drives this file: table data is stored in ONE
long-format table, `document_facts`, one row per cell. Previously each
extracted table got a bespoke `CREATE TABLE` written by the LLM, which meant
the SQL-generation prompt grew with every document uploaded, cross-document
questions required joining two schemas the model had invented minutes earlier,
and no row index or page number was persisted anywhere.

A fixed schema costs something real: aggregation now needs a pivot rather than
`SELECT SUM(revenue)`. That is paid for in query/sql_templates.py, where the
never-changing schema makes canned query patterns possible.
"""

from __future__ import annotations

import logging

from indexial.core.db import get_connection

logger = logging.getLogger(__name__)

EXTENSIONS_DDL = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS pgcrypto;
"""

DOCUMENTS_DDL = """
CREATE TABLE IF NOT EXISTS documents (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filename      TEXT NOT NULL,
    file_hash     TEXT UNIQUE NOT NULL,
    status        TEXT DEFAULT 'pending',
    page_count    INTEGER,
    table_count   INTEGER DEFAULT 0,
    chunk_count   INTEGER DEFAULT 0,
    fact_count    INTEGER DEFAULT 0,
    error_message TEXT,
    markdown_path TEXT,
    uploaded_at   TIMESTAMPTZ DEFAULT NOW(),
    completed_at  TIMESTAMPTZ,
    metadata      JSONB DEFAULT '{}'
);
"""

# Metadata-only catalogue. It no longer points at a physical table; the rows
# live in document_facts. /api/tables still serves the five fields the frontend
# reads, with physical_table_name computed on the way out.
TABLE_REGISTRY_DDL = """
CREATE TABLE IF NOT EXISTS table_registry (
    table_id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id          UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    table_index          SMALLINT NOT NULL,
    original_filename    TEXT,
    semantic_description TEXT,
    headers              JSONB NOT NULL DEFAULT '[]'::jsonb,
    column_keys          JSONB NOT NULL DEFAULT '[]'::jsonb,
    column_profiles      JSONB NOT NULL DEFAULT '[]'::jsonb,
    label_column_key     TEXT,
    row_count            INTEGER NOT NULL DEFAULT 0,
    fact_count           INTEGER NOT NULL DEFAULT 0,
    page_start           SMALLINT,
    page_end             SMALLINT,
    raw_markdown         TEXT,
    created_at           TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT table_registry_doc_idx_uniq UNIQUE (document_id, table_index)
);

CREATE INDEX IF NOT EXISTS table_registry_document_idx
    ON table_registry (document_id);
"""

# One row per cell.
#
# row_label is the load-bearing column: the row's label value is copied onto
# every cell of that row, which turns "total 2023 revenue" from a correlated
# self-join into a flat WHERE. Without it a small model cannot write a useful
# aggregate against this shape at all.
#
# value_type distinguishes 'year' from 'number' so templates can exclude year
# columns from SUM; otherwise "total of the year column" cheerfully returns
# 4045 for two rows of 2022 and 2023.
DOCUMENT_FACTS_DDL = """
CREATE TABLE IF NOT EXISTS document_facts (
    fact_id      BIGSERIAL PRIMARY KEY,

    document_id  UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    table_id     UUID NOT NULL REFERENCES table_registry(table_id) ON DELETE CASCADE,
    table_index  SMALLINT NOT NULL,
    row_index    INTEGER  NOT NULL,
    col_index    SMALLINT NOT NULL,
    page_number  SMALLINT,

    column_name  TEXT NOT NULL,
    column_key   TEXT NOT NULL,
    row_label    TEXT,

    value_text   TEXT,
    value_num    NUMERIC,
    value_date   DATE,
    unit         TEXT,
    scale_factor NUMERIC,
    value_type   TEXT NOT NULL DEFAULT 'text',
    parse_note   TEXT,

    created_at   TIMESTAMPTZ DEFAULT NOW(),

    CONSTRAINT document_facts_cell_uniq UNIQUE (table_id, row_index, col_index),
    CONSTRAINT document_facts_type_ck CHECK (
        value_type IN ('number', 'year', 'date', 'text', 'empty', 'ambiguous')
    )
);

ALTER TABLE document_facts
    ADD COLUMN IF NOT EXISTS fts tsvector
    GENERATED ALWAYS AS (
        to_tsvector('english',
            coalesce(row_label, '') || ' ' ||
            coalesce(column_name, '') || ' ' ||
            coalesce(value_text, ''))
    ) STORED;

CREATE INDEX IF NOT EXISTS document_facts_row_idx
    ON document_facts (table_id, row_index, col_index);
CREATE INDEX IF NOT EXISTS document_facts_doc_idx
    ON document_facts (document_id);
CREATE INDEX IF NOT EXISTS document_facts_colkey_idx
    ON document_facts (table_id, column_key);
CREATE INDEX IF NOT EXISTS document_facts_num_idx
    ON document_facts (table_id, column_key, value_num)
    WHERE value_num IS NOT NULL;
CREATE INDEX IF NOT EXISTS document_facts_fts_idx
    ON document_facts USING GIN (fts);
CREATE INDEX IF NOT EXISTS document_facts_label_trgm_idx
    ON document_facts USING GIN (row_label gin_trgm_ops);
"""


def _chunks_ddl(table_name: str, dimensions: int) -> str:
    """
    Vector store DDL.

    HNSW rather than the previous ivfflat(lists=100): ivfflat needs to be built
    against representative data to be useful, and this corpus is wiped and
    rebuilt constantly, so its lists were always badly calibrated. HNSW has no
    training step. Requires pgvector >= 0.5.0 (0.8.2 is installed).
    """
    return f"""
CREATE TABLE IF NOT EXISTS {table_name} (
    id              TEXT PRIMARY KEY,
    document_id     UUID REFERENCES documents(id) ON DELETE CASCADE,
    source_file     TEXT,
    chunk_id        INTEGER,
    content         TEXT,
    start_offset    INTEGER,
    end_offset      INTEGER,
    heading_context TEXT,
    section         TEXT,
    page_number     SMALLINT,
    processed_date  TIMESTAMPTZ,
    embedding       vector({dimensions}),
    metadata        JSONB
);

CREATE INDEX IF NOT EXISTS {table_name}_embedding_idx
    ON {table_name} USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS {table_name}_document_id_idx
    ON {table_name} (document_id);
"""


# Legacy teardown. Every extracted table used to become its own
# `tbl_<uuid8>_t<n>_extracted` relation created from LLM-authored DDL. Those
# have no place in the new model, and the old table_registry is structurally
# incompatible (SERIAL id and a NOT NULL physical_table_name).
LEGACY_SWEEP = r"""
DO $$
DECLARE
    r RECORD;
BEGIN
    FOR r IN
        SELECT tablename FROM pg_tables
        WHERE schemaname = 'public' AND tablename LIKE 'tbl\_%\_extracted'
    LOOP
        EXECUTE format('DROP TABLE IF EXISTS %I CASCADE', r.tablename);
    END LOOP;

    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = 'table_registry'
          AND column_name = 'physical_table_name'
    ) THEN
        DROP TABLE IF EXISTS document_facts CASCADE;
        DROP TABLE IF EXISTS table_registry CASCADE;
    END IF;
END $$;
"""

# Order matters: documents, then table_registry, then the tables holding
# foreign keys into them.
_STEPS = [
    ("extensions", EXTENSIONS_DDL),
    ("legacy sweep", LEGACY_SWEEP),
    ("documents", DOCUMENTS_DDL),
    ("table_registry", TABLE_REGISTRY_DDL),
    ("document_facts", DOCUMENT_FACTS_DDL),
]

# Everything the reset wipes, in dependency order.
TRUNCATE_TABLES = ["document_facts", "document_chunks", "table_registry", "documents"]


def ensure_schema(vector_table: str | None = None, dimensions: int | None = None) -> None:
    """
    Create the full schema. Idempotent; safe to call on every startup.
    """
    from indexial.core import config

    vector_table = vector_table or config.VECTOR_TABLE_NAME
    dimensions = dimensions or config.EMBEDDING_DIMENSIONS

    steps = _STEPS + [("document_chunks", _chunks_ddl(vector_table, dimensions))]

    with get_connection() as conn:
        cur = conn.cursor()
        for name, ddl in steps:
            try:
                cur.execute(ddl)
            except Exception:
                conn.rollback()
                logger.error("Schema step failed: %s", name)
                raise
        conn.commit()
        cur.close()

    logger.info("Schema verified (%d steps)", len(steps))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ensure_schema()
    print("schema OK")
