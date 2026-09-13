"""
Indexial REST API

Flask REST API for the Indexial RAG + SQL system.
Provides endpoints for document upload, querying, and session management.
"""

import shutil
import logging
import threading
import time
from typing import Optional

from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from psycopg2 import sql as psql

from indexial.core import config
from indexial.core.db import get_connection
from indexial.core.schema import TRUNCATE_TABLES
from indexial.ingest.pipeline import IngestionPipeline
from indexial.memory import MemoryManager
from indexial.query.router import QueryOrchestrator
from indexial.query.sql_engine import TableRegistryReader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration. Every path is absolute and derived from the repository root,
# so the server behaves identically started from backend/ or from the root.
config.require()
config.ensure_dirs()

UPLOAD_FOLDER = config.UPLOAD_DIR
ALLOWED_EXTENSIONS = {"pdf"}

app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = config.MAX_UPLOAD_BYTES

# Initialize global components
memory_manager = MemoryManager()
query_orchestrator = QueryOrchestrator(memory=memory_manager)
ingestion_pipeline = IngestionPipeline()
table_registry = TableRegistryReader()

# Reset mechanism state
_reset_lock = threading.Lock()
_resetting = False
_last_activity_time = time.time()
INACTIVITY_TIMEOUT_SECONDS = config.INACTIVITY_TIMEOUT_SECONDS


# =================== Helper Functions =================== #

def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def get_document_by_id(doc_id: str) -> Optional[dict]:
    """Fetch document from database by UUID."""
    try:
        with get_connection(readonly=True) as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT id, filename, file_hash, status, page_count,
                       table_count, chunk_count, error_message,
                       uploaded_at, completed_at
                FROM documents
                WHERE id::text = %s
                """,
                (doc_id,)
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
                "uploaded_at": row[8].isoformat() if row[8] else None,
                "completed_at": row[9].isoformat() if row[9] else None,
            }
    except Exception as e:
        logger.error(f"Error fetching document {doc_id}: {e}")
        return None


# =================== Middleware =================== #

@app.before_request
def track_activity():
    """Track API activity and block requests during reset."""
    global _last_activity_time
    _last_activity_time = time.time()
    if _resetting and request.endpoint not in ('health', 'reset_database'):
        return jsonify({"error": "System is resetting. Please try again."}), 503


# =================== Reset Function =================== #

def perform_reset():
    """
    Core database reset. Clears all data and returns system to initial state.

    Steps:
    1. Drop all dynamic tbl_*_extracted tables
    2. Truncate documents, document_chunks, table_registry
    3. Clear filesystem artifacts (uploads, output)
    4. Clear in-memory session state
    """
    global _resetting

    with _reset_lock:
        if _resetting:
            return {"status": "already_resetting"}
        _resetting = True

    try:
        summary = {
            "tables_dropped": [],
            "tables_truncated": [],
            "files_removed": 0,
            "sessions_cleared": 0,
            "errors": [],
        }

        # One TRUNCATE across a fixed set of tables.
        #
        # This used to read physical_table_name out of the registry and DROP
        # each dynamic table in its own connection. That had two failure modes
        # the fixed schema removes: a table whose registry row was lost could
        # never be dropped, and a mid-loop error left the database half reset.
        try:
            with get_connection() as conn:
                cur = conn.cursor()
                cur.execute(
                    f"TRUNCATE TABLE {', '.join(TRUNCATE_TABLES)} RESTART IDENTITY CASCADE"
                )
                conn.commit()
                cur.close()
            summary["tables_truncated"] = list(TRUNCATE_TABLES)
            logger.info(f"Truncated: {', '.join(TRUNCATE_TABLES)}")
        except Exception as e:
            logger.error(f"Database truncation failed: {e}")
            summary["errors"].append(f"TRUNCATE: {str(e)}")

        # Sweep any relation left behind by the pre-fact-store design.
        try:
            with get_connection() as conn:
                cur = conn.cursor()
                cur.execute(
                    r"""SELECT tablename FROM pg_tables
                        WHERE schemaname='public' AND tablename LIKE 'tbl\_%\_extracted'"""
                )
                for (name,) in cur.fetchall():
                    cur.execute(
                        psql.SQL("DROP TABLE IF EXISTS {} CASCADE").format(psql.Identifier(name))
                    )
                    summary["tables_dropped"].append(name)
                conn.commit()
                cur.close()
        except Exception as e:
            logger.warning(f"Legacy table sweep failed: {e}")

        # Step 4: Clear filesystem artifacts
        dirs_to_clear = [
            UPLOAD_FOLDER,
            config.MARKDOWN_DIR,
            config.CHUNKS_DIR,
        ]

        files_removed = 0
        for dir_path in dirs_to_clear:
            if dir_path.exists():
                for item in dir_path.iterdir():
                    try:
                        if item.is_file():
                            item.unlink()
                            files_removed += 1
                        elif item.is_dir():
                            shutil.rmtree(item)
                            files_removed += 1
                    except Exception as e:
                        logger.error(f"Failed to remove {item}: {e}")
                        summary["errors"].append(f"FS {item}: {str(e)}")

        summary["files_removed"] = files_removed

        # Step 5: Clear in-memory session state
        sessions_cleared = memory_manager.clear_all()
        summary["sessions_cleared"] = sessions_cleared

        logger.info(f"Reset complete: {summary}")
        return summary

    finally:
        with _reset_lock:
            _resetting = False


# =================== API Endpoints =================== #

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    try:
        # Test database connection
        with get_connection(readonly=True) as conn:
            cur = conn.cursor()
            cur.execute("SELECT 1")
            cur.close()

        return jsonify({
            "status": "healthy",
            "service": "indexial-api",
            "database": "connected"
        }), 200
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500


@app.route("/api/documents/upload", methods=["POST"])
def upload_document():
    """
    Upload a PDF document for processing.

    Expected: multipart/form-data with 'file' field
    Optional: 'force' parameter to reprocess existing files
    """
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Only PDF files are allowed"}), 400

    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = UPLOAD_FOLDER / filename
        file.save(str(filepath))

        logger.info(f"Saved uploaded file: {filepath}")

        # Process through ingestion pipeline
        force = request.form.get("force", "false").lower() == "true"
        result = ingestion_pipeline.process_pdf(str(filepath), force=force)

        skipped = result.get("skipped", False)
        return jsonify({
            "message": (
                "Document already processed"
                if skipped
                else "Document uploaded and processed"
            ),
            # .get(), not ["doc_id"]. The skip path returns no document id, so
            # re-uploading a duplicate PDF used to raise KeyError and 500.
            "document_id": result.get("doc_id"),
            "filename": filename,
            "status": result.get("status"),
            "skipped": skipped,
            "table_count": len(result.get("tables_created", [])),
            "fact_count": result.get("fact_count", 0),
            "chunk_count": result.get("chunk_count", 0),
        }), 201

    except Exception as e:
        logger.error(f"Upload failed: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/documents", methods=["GET"])
def list_documents():
    """
    List all documents with their processing status.

    Query params:
    - status: Filter by status (pending, extracting, parsing_tables, chunking, completed, failed)
    - limit: Max results (default 50)
    - offset: Pagination offset (default 0)
    """
    try:
        status_filter = request.args.get("status")
        limit = int(request.args.get("limit", 50))
        offset = int(request.args.get("offset", 0))

        with get_connection(readonly=True) as conn:
            cur = conn.cursor()

            if status_filter:
                query = """
                    SELECT id, filename, file_hash, status, page_count,
                           table_count, chunk_count, uploaded_at, completed_at
                    FROM documents
                    WHERE status = %s
                    ORDER BY uploaded_at DESC
                    LIMIT %s OFFSET %s
                """
                cur.execute(query, (status_filter, limit, offset))
            else:
                query = """
                    SELECT id, filename, file_hash, status, page_count,
                           table_count, chunk_count, uploaded_at, completed_at
                    FROM documents
                    ORDER BY uploaded_at DESC
                    LIMIT %s OFFSET %s
                """
                cur.execute(query, (limit, offset))

            rows = cur.fetchall()

            # Get total count
            count_query = "SELECT COUNT(*) FROM documents"
            if status_filter:
                count_query += " WHERE status = %s"
                cur.execute(count_query, (status_filter,))
            else:
                cur.execute(count_query)

            total = cur.fetchone()[0]
            cur.close()

            documents = []
            for row in rows:
                documents.append({
                    "id": str(row[0]),
                    "filename": row[1],
                    "file_hash": row[2],
                    "status": row[3],
                    "page_count": row[4],
                    "table_count": row[5],
                    "chunk_count": row[6],
                    "uploaded_at": row[7].isoformat() if row[7] else None,
                    "completed_at": row[8].isoformat() if row[8] else None,
                })

            return jsonify({
                "documents": documents,
                "total": total,
                "limit": limit,
                "offset": offset
            }), 200

    except Exception as e:
        logger.error(f"List documents failed: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/documents/<doc_id>", methods=["GET"])
def get_document(doc_id: str):
    """Get detailed information about a specific document."""
    document = get_document_by_id(doc_id)

    if not document:
        return jsonify({"error": "Document not found"}), 404

    return jsonify(document), 200


@app.route("/api/query", methods=["POST"])
def query():
    """
    Main query endpoint - processes natural language queries.

    Request body:
    {
        "query": "string (required)",
        "session_id": "string (optional)",
        "document_ids": ["uuid1", "uuid2"] (optional),
        "force_route": "SQL|RAG|HYBRID" (optional, for testing)
    }

    Response:
    {
        "answer": "string",
        "route": "SQL|RAG|HYBRID",
        "original_query": "string",
        "rewritten_query": "string (if rewritten)",
        "sql": "string (if SQL route)",
        "sql_explanation": "string (if SQL route)",
        "tables_used": ["table1"] (if SQL route),
        "sources": [{score, content, ...}] (if RAG route)
    }
    """
    try:
        data = request.get_json()

        if not data or "query" not in data:
            return jsonify({"error": "Missing 'query' field"}), 400

        user_query = data["query"]
        session_id = data.get("session_id")
        document_ids = data.get("document_ids")
        force_route = data.get("force_route")

        # Validate force_route if provided
        if force_route and force_route not in ["SQL", "RAG", "HYBRID"]:
            return jsonify({"error": "Invalid force_route. Must be SQL, RAG, or HYBRID"}), 400

        logger.info(f"Query: {user_query} (session={session_id})")

        # Process query
        result = query_orchestrator.process_query(
            query=user_query,
            session_id=session_id,
            document_ids=document_ids,
            force_route=force_route
        )

        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Query failed: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/sessions/<session_id>/clear", methods=["POST"])
def clear_session(session_id: str):
    """Clear conversation history for a session."""
    try:
        memory_manager.clear(session_id)
        return jsonify({
            "message": f"Session {session_id} cleared",
            "session_id": session_id
        }), 200
    except Exception as e:
        logger.error(f"Clear session failed: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/reset", methods=["POST"])
def reset_database():
    """Reset the entire database to its initial state."""
    try:
        summary = perform_reset()

        if summary.get("status") == "already_resetting":
            return jsonify({
                "message": "Reset already in progress",
                "status": "in_progress"
            }), 409

        return jsonify({
            "message": "Database reset complete",
            "status": "success",
            "summary": summary
        }), 200

    except Exception as e:
        logger.error(f"Reset failed: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/tables", methods=["GET"])
def list_tables():
    """
    List all tables in the registry (for debugging/transparency).

    Query params:
    - document_ids: Comma-separated UUIDs to filter by
    """
    try:
        document_ids_str = request.args.get("document_ids")
        document_ids = document_ids_str.split(",") if document_ids_str else None

        # physical_table_name no longer names a relation - table rows live in
        # document_facts. It is computed here rather than stored, so the column
        # cannot drift into lying about reality, while the frontend keeps the
        # same string shape and the same React key.
        query = """
            SELECT table_id,
                   'tbl_' || left(document_id::text, 8) || '_t' || table_index
                       AS physical_table_name,
                   document_id, semantic_description, headers, column_keys,
                   row_count, fact_count, original_filename,
                   page_start, page_end, created_at
            FROM table_registry
            WHERE (%(doc_ids)s::text[] IS NULL
                   OR document_id::text = ANY(%(doc_ids)s::text[]))
            ORDER BY created_at DESC
        """

        with get_connection(readonly=True) as conn:
            cur = conn.cursor()
            cur.execute(query, {"doc_ids": document_ids})
            rows = cur.fetchall()
            cur.close()

        tables = [
            {
                "id": str(row[0]),
                "table_id": str(row[0]),
                "physical_table_name": row[1],
                "source_doc_uuid": str(row[2]),
                "semantic_description": row[3],
                # headers and row_count are hard invariants: tables-view calls
                # .map() and .toLocaleString() on them unguarded.
                "headers": row[4] or [],
                "column_keys": row[5] or [],
                "row_count": row[6] or 0,
                "fact_count": row[7] or 0,
                "original_filename": row[8],
                "page_start": row[9],
                "page_end": row[10],
                "created_at": row[11].isoformat() if row[11] else None,
            }
            for row in rows
        ]

        return jsonify({"tables": tables, "total": len(tables)}), 200

    except Exception as e:
        logger.error(f"List tables failed: {e}")
        return jsonify({"error": str(e)}), 500


# =================== Error Handlers =================== #

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500


# =================== Inactivity Timer =================== #

def _inactivity_checker():
    """Background thread that triggers reset after inactivity timeout."""
    global _last_activity_time
    while True:
        time.sleep(60)  # Check every 60 seconds
        elapsed = time.time() - _last_activity_time
        if elapsed >= INACTIVITY_TIMEOUT_SECONDS:
            logger.info(f"Inactivity timeout reached ({elapsed:.0f}s). Triggering reset.")
            try:
                perform_reset()
                _last_activity_time = time.time()
            except Exception as e:
                logger.error(f"Inactivity reset failed: {e}")


# =================== Main =================== #


def main() -> None:
    """Entry point: `python -m indexial.api.app`."""
    import os

    port = config.API_PORT
    debug = config.FLASK_DEBUG

    # Start inactivity timer (avoid double-start with Flask reloader)
    if not debug or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        _inactivity_thread = threading.Thread(target=_inactivity_checker, daemon=True)
        _inactivity_thread.start()
        logger.info(f"Inactivity timer started (timeout={INACTIVITY_TIMEOUT_SECONDS}s)")

    logger.info(f"Starting Indexial API on port {port}")
    app.run(host="0.0.0.0", port=port, debug=debug)


if __name__ == "__main__":
    main()
