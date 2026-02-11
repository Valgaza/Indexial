"""
Unified Ingestion Pipeline

Orchestrates the full document processing flow:
PDF -> Markdown -> Tables + Chunks -> Supabase

Tracks document status in a `documents` table for dedup and monitoring.
"""

import hashlib
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

from dotenv import load_dotenv

from db import get_connection
from extractor import MistralOCRExtractor
from table_parser import MarkdownTableExtractor, MarkdownTableStitcher, TableIngestionPipeline
from chunker import DocumentProcessor

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        output_dir: str = "output",
        chunk_size: int = 512,
        overlap: float = 0.1,
    ):
        self.output_dir = Path(output_dir)
        self.markdown_dir = self.output_dir / "markdown"
        self.markdown_dir.mkdir(parents=True, exist_ok=True)

        self.chunk_size = chunk_size
        self.overlap = overlap

        # Ensure documents table exists
        self._ensure_documents_table()

    def _ensure_documents_table(self):
        """Create the documents tracking table if it doesn't exist."""
        sql = """
        CREATE TABLE IF NOT EXISTS documents (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            filename TEXT NOT NULL,
            file_hash TEXT UNIQUE NOT NULL,
            status TEXT DEFAULT 'pending',
            page_count INTEGER,
            table_count INTEGER DEFAULT 0,
            chunk_count INTEGER DEFAULT 0,
            error_message TEXT,
            markdown_path TEXT,
            uploaded_at TIMESTAMP DEFAULT NOW(),
            completed_at TIMESTAMP,
            metadata JSONB DEFAULT '{}'
        );
        """
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute(sql)
            conn.commit()
            cur.close()
        logger.info("Documents table verified/created")

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
            "tables_created": [],
            "chunk_count": 0,
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
            result["status"] = "skipped"
            result["skipped"] = True
            print(f"Skipped: {pdf_path.name} (already processed)")
            return result

        result["doc_id"] = doc_id

        try:
            # Step 2: Extract markdown via Mistral OCR
            self._update_status(doc_id, "extracting")
            print(f"\n[1/3] Extracting markdown from {pdf_path.name}...")

            extractor = MistralOCRExtractor()
            md_output_path = self.markdown_dir / f"{pdf_path.stem}.md"
            markdown_text = extractor.extract_pdf_to_markdown(
                str(pdf_path), str(md_output_path)
            )

            # Count pages from the markdown (separated by ---)
            page_count = markdown_text.count("\n\n---\n\n") + 1
            self._update_status(
                doc_id, "extracting",
                page_count=page_count,
                markdown_path=str(md_output_path),
            )

            # Step 3: Parse & ingest tables
            self._update_status(doc_id, "parsing_tables")
            print(f"[2/3] Extracting and ingesting tables...")

            table_extractor = MarkdownTableExtractor()
            tables = table_extractor.extract_tables(markdown_text, pdf_path.name)

            tables_created = []
            if tables:
                logger.info(f"Found {len(tables)} tables, ingesting...")
                pipeline = TableIngestionPipeline()
                tables_created = pipeline.ingest_all_tables(tables, document_id=doc_id)
                result["tables_created"] = tables_created

            self._update_status(doc_id, "parsing_tables", table_count=len(tables_created))

            # Step 4: Chunk & embed text
            self._update_status(doc_id, "chunking")
            print(f"[3/3] Chunking and embedding text...")

            processor = DocumentProcessor(
                chunk_size=self.chunk_size,
                overlap=self.overlap,
                output_dir=str(self.output_dir),
                embed_with_context=True,
            )
            chunks = processor.process_markdown_file(
                input_path=str(md_output_path),
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
        docs_dir = Path("Docs")
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
