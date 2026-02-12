"""
Indexial REST API

Flask REST API for the Indexial RAG + SQL system.
Provides endpoints for document upload, querying, and session management.
"""

import os
import logging
from pathlib import Path
from typing import Optional

from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

from pipeline import IngestionPipeline
from router import QueryOrchestrator
from memory import MemoryManager
from db import get_connection
from sql_engine import TableRegistryReader

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = Path(os.getenv("UPLOAD_FOLDER", "uploads"))
UPLOAD_FOLDER.mkdir(exist_ok=True)
ALLOWED_EXTENSIONS = {"pdf"}

app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50MB max file size

# Initialize global components
memory_manager = MemoryManager()
query_orchestrator = QueryOrchestrator(memory=memory_manager)
ingestion_pipeline = IngestionPipeline()
table_registry = TableRegistryReader()


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

        return jsonify({
            "message": "Document uploaded and queued for processing",
            "document_id": result["doc_id"],
            "filename": filename,
            "status": result["status"],
            "skipped": result.get("skipped", False)
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

        with get_connection(readonly=True) as conn:
            cur = conn.cursor()

            if document_ids:
                placeholders = ",".join(["%s"] * len(document_ids))
                query = f"""
                    SELECT id, physical_table_name, source_doc_uuid,
                           semantic_description, headers, row_count,
                           original_filename, created_at
                    FROM table_registry
                    WHERE source_doc_uuid::text IN ({placeholders})
                    ORDER BY created_at DESC
                """
                cur.execute(query, document_ids)
            else:
                query = """
                    SELECT id, physical_table_name, source_doc_uuid,
                           semantic_description, headers, row_count,
                           original_filename, created_at
                    FROM table_registry
                    ORDER BY created_at DESC
                """
                cur.execute(query)

            rows = cur.fetchall()
            cur.close()

            tables = []
            for row in rows:
                tables.append({
                    "id": row[0],
                    "physical_table_name": row[1],
                    "source_doc_uuid": str(row[2]),
                    "semantic_description": row[3],
                    "headers": row[4],  # Already parsed JSONB
                    "row_count": row[5],
                    "original_filename": row[6],
                    "created_at": row[7].isoformat() if row[7] else None
                })

            return jsonify({
                "tables": tables,
                "total": len(tables)
            }), 200

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


# =================== Main =================== #

if __name__ == "__main__":
    port = int(os.getenv("API_PORT", 8000))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"

    logger.info(f"Starting Indexial API on port {port}")
    app.run(host="0.0.0.0", port=port, debug=debug)
