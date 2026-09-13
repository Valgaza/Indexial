"""
Unified Ingestion Pipeline

Orchestrates the full document processing flow:
PDF -> Markdown -> Tables + Chunks -> Supabase

Tracks document status in a `documents` table for dedup and monitoring.
"""

import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Sequence
from datetime import datetime


from indexial.core import config
from indexial.core.db import get_connection
from indexial.core.schema import ensure_schema
from indexial.ingest.extractor import MistralOCRExtractor
from indexial.ingest.fact_store import FactStore
from indexial.ingest.table_parser import MarkdownTableExtractor
from indexial.ingest.chunker import DocumentProcessor


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def excise_tables(
    markdown: str,
    tables: Sequence[Any],
    ingested: Sequence[Dict[str, Any]],
    sample_rows: int = 2,
) -> str:
    """
    Replace table bodies in the text handed to the chunker with short stubs.

    Tables used to be embedded as chunks *and* stored as SQL rows. The
    duplication actively hurt retrieval: pipe-delimited numeric soup embeds
    poorly, so every table's vector ends up near every other table's, and those
    near-duplicates crowd real prose out of a top-5 search.

    Deleting them outright would be worse. The RAG route cannot reach facts,
    and the heuristic router will sometimes send a table question to RAG, which
    would then answer "I couldn't find any relevant information". A stub keeps
    three things: prose chunks stay contiguous, the semantic description is
    still embedded so the right neighbourhood is still retrievable, and the
    chunk carries table_id so a RAG hit can be upgraded into a fact query.
    """
    if not tables:
        return markdown

    by_index = {t["table_index"]: t for t in ingested}
    out = markdown

    for table in tables:
        meta = by_index.get(table.table_index)
        if not meta:
            continue

        pages = (
            f" | page {table.page_start}"
            if table.page_start and table.page_start == table.page_end
            else f" | pages {table.page_start}-{table.page_end}"
            if table.page_start
            else ""
        )
        preview = "\n".join(
            " | ".join(str(c) for c in row) for row in table.rows[:sample_rows]
        )
        stub = (
            f"[TABLE {table.table_index} — {meta['description']}{pages} "
            f"| table_id={meta['table_id']}]\n"
            f"columns: {', '.join(table.headers)} | {len(table.rows)} rows\n"
            f"{preview}\n"
            + ("...\n" if len(table.rows) > sample_rows else "")
        )

        # raw_fragments holds the exact source span per page. The first becomes
        # the stub; any continuation fragments are dropped.
        replacement = stub
        for fragment in table.raw_fragments:
            fragment = fragment.strip()
            if fragment and fragment in out:
                out = out.replace(fragment, replacement, 1)
                replacement = ""

    return out


class IngestionPipeline:
    """
    Unified pipeline that processes a PDF through all stages:
    1. Register document (dedup via file hash)
    2. Extract markdown via Mistral OCR
    3. Parse & ingest tables into Supabase
    4. Chunk & embed text into pgvector

    Tracks status in the `documents` table at each stage.
    """

    def __init__(
        self,
        output_dir: Optional[str] = None,
        chunk_size: int = 512,
        overlap: float = 0.1,
    ):
        self.output_dir = Path(output_dir) if output_dir else config.OUTPUT_DIR
        self.markdown_dir = self.output_dir / "markdown"
        self.markdown_dir.mkdir(parents=True, exist_ok=True)

        self.chunk_size = chunk_size
        self.overlap = overlap

        # Ensure documents table exists
        self._ensure_documents_table()

    def _ensure_documents_table(self):
        """
        Create the whole schema if needed.

        Delegates to core.schema, which owns every CREATE statement in the
        project. This used to carry its own copy of the documents DDL, which
        meant the real shape of the database was spread across three modules.
        """
        ensure_schema()

    @staticmethod
    def _compute_file_hash(file_path: str) -> str:
        """Compute SHA256 hash of a file for deduplication."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _register_document(self, pdf_path: Path, file_hash: str) -> Optional[str]:
        """
        Register a document in the database. Returns the document ID.
        Returns None if the document was already processed (dedup).
        """
        with get_connection() as conn:
            cur = conn.cursor()

            # Check if already processed
            cur.execute(
                "SELECT id, status FROM documents WHERE file_hash = %s",
                (file_hash,),
            )
            existing = cur.fetchone()

            if existing:
                doc_id, status = existing
                if status == "completed":
                    logger.info(f"Document already processed: {pdf_path.name} (id={doc_id})")
                    cur.close()
                    return None
                elif status == "failed":
                    # Allow re-processing of failed documents
                    cur.execute(
                        "UPDATE documents SET status = 'pending', error_message = NULL WHERE id = %s",
                        (str(doc_id),),
                    )
                    conn.commit()
                    cur.close()
                    logger.info(f"Re-processing failed document: {pdf_path.name} (id={doc_id})")
                    return str(doc_id)
                else:
                    # In progress - skip
                    logger.info(f"Document already in progress: {pdf_path.name} (status={status})")
                    cur.close()
                    return None

            # Register new document
            cur.execute(
                """
                INSERT INTO documents (filename, file_hash, status)
                VALUES (%s, %s, 'pending')
                RETURNING id
                """,
                (pdf_path.name, file_hash),
            )
            doc_id = str(cur.fetchone()[0])
            conn.commit()
            cur.close()

            logger.info(f"Registered new document: {pdf_path.name} (id={doc_id})")
            return doc_id

    def _update_status(self, doc_id: str, status: str, **kwargs):
        """Update document status and optional fields."""
        set_parts = ["status = %s"]
        values = [status]

        for key, value in kwargs.items():
            set_parts.append(f"{key} = %s")
            values.append(value)

        if status == "completed":
            set_parts.append("completed_at = NOW()")

        values.append(doc_id)

        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                f"UPDATE documents SET {', '.join(set_parts)} WHERE id = %s::uuid",
                values,
            )
            conn.commit()
            cur.close()

    def process_pdf(self, pdf_path: str, force: bool = False) -> Dict[str, Any]:
        """
        Full pipeline: PDF -> Markdown -> Tables + Chunks -> Supabase.

        Args:
            pdf_path: Path to the PDF file
            force: If True, reprocess even if already completed

        Returns:
            Dict with processing results:
            {
                "doc_id": str,
                "filename": str,
                "status": str,
                "tables_created": List[str],
                "chunk_count": int,
                "skipped": bool,
            }
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        result = {
            "filename": pdf_path.name,
            "status": "pending",
            "doc_id": None,
            "tables_created": [],
            "chunk_count": 0,
            "fact_count": 0,
            "skipped": False,
        }

        # Step 1: Compute hash and register
        file_hash = self._compute_file_hash(str(pdf_path))

        if force:
            # Delete existing record to allow reprocessing
            with get_connection() as conn:
                cur = conn.cursor()
                cur.execute("DELETE FROM documents WHERE file_hash = %s", (file_hash,))
                conn.commit()
                cur.close()

        doc_id = self._register_document(pdf_path, file_hash)
        if doc_id is None:
            # doc_id is seeded as None above so this early return still carries
            # the key; app.py reads result["doc_id"] unconditionally and used
            # to raise KeyError on any duplicate upload.
            result["status"] = "skipped"
            result["skipped"] = True
            print(f"Skipped: {pdf_path.name} (already processed)")
            return result

        result["doc_id"] = doc_id

        try:
            # Step 2: Extract markdown via Mistral OCR, page by page
            self._update_status(doc_id, "extracting")
            print(f"\n[1/3] Extracting markdown from {pdf_path.name}...")

            extractor = MistralOCRExtractor()
            pages = extractor.extract_pdf_pages(str(pdf_path))
            markdown_text = extractor.pages_to_markdown(pages)

            md_output_path = self.markdown_dir / f"{pdf_path.stem}.md"
            md_output_path.parent.mkdir(parents=True, exist_ok=True)
            md_output_path.write_text(markdown_text, encoding="utf-8")

            # Exact, rather than inferred by counting a '---' separator that a
            # horizontal rule in the document body would also match.
            page_count = len(pages)
            self._update_status(
                doc_id, "extracting",
                page_count=page_count,
                markdown_path=str(md_output_path),
            )

            # Step 3: Parse & ingest tables as cell-level facts
            self._update_status(doc_id, "parsing_tables")
            print(f"[2/3] Extracting and ingesting tables...")

            table_extractor = MarkdownTableExtractor()
            tables = table_extractor.extract_tables_from_pages(pages, pdf_path.name)

            ingested = []
            if tables:
                logger.info(f"Found {len(tables)} tables, ingesting...")
                ingested = FactStore().ingest_all(tables, document_id=doc_id)
                result["tables_created"] = [t["table_id"] for t in ingested]
                result["fact_count"] = sum(t["fact_count"] for t in ingested)

            self._update_status(
                doc_id, "parsing_tables",
                table_count=len(ingested),
                fact_count=result["fact_count"],
            )

            # Step 4: Chunk & embed text, with table bodies replaced by stubs
            self._update_status(doc_id, "chunking")
            print(f"[3/3] Chunking and embedding text...")

            chunk_source = excise_tables(markdown_text, tables, ingested)
            chunked_path = self.markdown_dir / f"{pdf_path.stem}.chunked.md"
            chunked_path.write_text(chunk_source, encoding="utf-8")

            processor = DocumentProcessor(
                chunk_size=self.chunk_size,
                overlap=self.overlap,
                output_dir=str(self.output_dir),
                embed_with_context=True,
            )
            chunks = processor.process_markdown_file(
                input_path=str(chunked_path),
                save_chunks=True,
                store_to_qdrant=True,
                document_id=doc_id,
            )
            result["chunk_count"] = len(chunks)

            # Step 5: Mark completed
            self._update_status(
                doc_id, "completed",
                chunk_count=len(chunks),
            )
            result["status"] = "completed"

            print(f"\nCompleted: {pdf_path.name}")
            print(f"  Document ID: {doc_id}")
            print(f"  Pages: {page_count}")
            print(f"  Tables created: {len(tables_created)}")
            print(f"  Chunks embedded: {len(chunks)}")

            return result

        except Exception as e:
            logger.exception(f"Pipeline failed for {pdf_path.name}")
            self._update_status(doc_id, "failed", error_message=str(e)[:500])
            result["status"] = "failed"
            result["error"] = str(e)
            print(f"\nFailed: {pdf_path.name} - {e}")
            return result

    def get_document_status(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a document by ID."""
        with get_connection(readonly=True) as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT id, filename, file_hash, status, page_count,
                       table_count, chunk_count, error_message,
                       uploaded_at, completed_at
                FROM documents WHERE id = %s::uuid
                """,
                (doc_id,),
            )
            row = cur.fetchone()
            cur.close()

            if not row:
                return None

            return {
                "id": str(row[0]),
                "filename": row[1],
                "file_hash": row[2],
                "status": row[3],
                "page_count": row[4],
                "table_count": row[5],
                "chunk_count": row[6],
                "error_message": row[7],
                "uploaded_at": str(row[8]) if row[8] else None,
                "completed_at": str(row[9]) if row[9] else None,
            }

    def list_documents(self) -> list:
        """List all documents with their status."""
        with get_connection(readonly=True) as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT id, filename, status, page_count, table_count,
                       chunk_count, uploaded_at, completed_at
                FROM documents
                ORDER BY uploaded_at DESC
                """
            )
            rows = cur.fetchall()
            cur.close()

            return [
                {
                    "id": str(row[0]),
                    "filename": row[1],
                    "status": row[2],
                    "page_count": row[3],
                    "table_count": row[4],
                    "chunk_count": row[5],
                    "uploaded_at": str(row[6]) if row[6] else None,
                    "completed_at": str(row[7]) if row[7] else None,
                }
                for row in rows
            ]


def main():
    """CLI entry point for processing PDFs."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: uv run python pipeline.py <pdf_path> [--force]")
        print("       uv run python pipeline.py --list")
        print("       uv run python pipeline.py --all [--force]")
        sys.exit(1)

    pipeline = IngestionPipeline()

    if sys.argv[1] == "--list":
        docs = pipeline.list_documents()
        if not docs:
            print("No documents found.")
        else:
            print(f"\n{'ID':<38} {'Filename':<25} {'Status':<15} {'Pages':>5} {'Tables':>6} {'Chunks':>6}")
            print("-" * 100)
            for doc in docs:
                print(
                    f"{doc['id']:<38} {doc['filename']:<25} {doc['status']:<15} "
                    f"{doc['page_count'] or '-':>5} {doc['table_count'] or '-':>6} "
                    f"{doc['chunk_count'] or '-':>6}"
                )
        return

    force = "--force" in sys.argv

    if sys.argv[1] == "--all":
        # Process all PDFs in Docs/
        docs_dir = config.UPLOAD_DIR
        if not docs_dir.exists():
            print("Docs/ directory not found")
            sys.exit(1)

        pdf_files = list(docs_dir.glob("*.pdf"))
        if not pdf_files:
            print("No PDF files found in Docs/")
            sys.exit(1)

        print(f"Found {len(pdf_files)} PDF file(s)")
        print("=" * 60)

        for pdf_file in pdf_files:
            print(f"\n{'=' * 60}")
            print(f"Processing: {pdf_file.name}")
            print(f"{'=' * 60}")
            pipeline.process_pdf(str(pdf_file), force=force)
    else:
        # Process single PDF
        pdf_path = sys.argv[1]
        print("=" * 60)
        print("Indexial - Document Ingestion Pipeline")
        print("=" * 60)
        pipeline.process_pdf(pdf_path, force=force)


if __name__ == "__main__":
    main()
