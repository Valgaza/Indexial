"""
Central configuration.

Every environment variable the project reads is resolved here, once, so that
call sites take a constant instead of calling os.getenv with a local default.

Two things this module exists to prevent:

1. CWD-relative output paths. Every directory constant below is absolute,
   derived from the repository root, so the app behaves identically whether it
   is started from the repo root or from backend/.
2. Credentials that fail deep inside a client constructor. require() and
   check() surface configuration problems up front with readable messages.
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------- paths ----

# config.py lives at <root>/backend/indexial/core/config.py
PROJECT_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = PROJECT_ROOT / "backend"

load_dotenv(PROJECT_ROOT / ".env")

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR") or (PROJECT_ROOT / "output")).resolve()
UPLOAD_DIR = Path(os.getenv("UPLOAD_FOLDER") or (PROJECT_ROOT / "uploads")).resolve()
MARKDOWN_DIR = OUTPUT_DIR / "markdown"
CHUNKS_DIR = OUTPUT_DIR / "chunks"


def ensure_dirs() -> None:
    """Create the writable directories the pipeline depends on."""
    for d in (UPLOAD_DIR, MARKDOWN_DIR, CHUNKS_DIR):
        d.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------- database ----

# Query parameters that are meaningful to ORMs but rejected by libpq/psycopg2.
_NON_LIBPQ_PARAMS = {"pgbouncer", "schema", "connection_limit", "pool_timeout"}


def _clean_dsn(dsn: str) -> str:
    """
    Strip query parameters libpq does not understand.

    Supabase hands out a Prisma-flavoured URL carrying ?pgbouncer=true, which
    psycopg2 rejects outright with 'invalid URI query parameter'.
    """
    parts = urlsplit(dsn)
    if not parts.query:
        return dsn
    kept = [(k, v) for k, v in parse_qsl(parts.query) if k.lower() not in _NON_LIBPQ_PARAMS]
    return urlunsplit(parts._replace(query=urlencode(kept)))


def get_db_url() -> str:
    """
    Resolve the Postgres DSN.

    DIRECT_URL (session mode, :5432) is preferred over DATABASE_URL
    (transaction pooler, :6543). The code relies on session-scoped state that
    pgbouncer's transaction pooling does not preserve:

      - conn.set_session(readonly=True)  in core/db.py
      - SET statement_timeout            in query/sql_engine.py

    SUPABASE_DB_URL is accepted for backwards compatibility with older setups.
    """
    dsn = os.getenv("DIRECT_URL") or os.getenv("DATABASE_URL") or os.getenv("SUPABASE_DB_URL")
    if not dsn:
        raise RuntimeError(
            "No database URL configured. Set DIRECT_URL (preferred, session mode "
            "on port 5432) in the .env at the repository root."
        )
    return _clean_dsn(dsn)


# ------------------------------------------------------------ providers ----

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = os.getenv("GROQ_API_URL", "https://api.groq.com/openai/v1/chat/completions")
# llama-3.1-8b-instant was retired from Groq; no Llama model remains available.
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

# Free-tier ceiling, measured from x-ratelimit-limit-tokens. Identical for
# gpt-oss-20b and gpt-oss-120b, so the larger model costs latency and output
# tokens rather than headroom. Drives backoff in providers/llm.py.
GROQ_TOKENS_PER_MINUTE = int(os.getenv("GROQ_TOKENS_PER_MINUTE", "8000"))
GROQ_MAX_RETRIES = int(os.getenv("GROQ_MAX_RETRIES", "4"))

JINA_API_KEY = os.getenv("JINA_API_KEY")
JINA_API_URL = os.getenv("JINA_API_URL", "https://api.jina.ai/v1/embeddings")
JINA_MODEL = os.getenv("JINA_MODEL", "jina-embeddings-v3")
JINA_TASK = os.getenv("JINA_TASK", "text-matching")
JINA_BATCH_SIZE = int(os.getenv("JINA_BATCH_SIZE", "32"))

# The README documents MISTRAL_API_KEY; the code has always read MISTRAL_OCR.
# Accept both so neither spelling is a silent failure.
MISTRAL_API_KEY = os.getenv("MISTRAL_OCR") or os.getenv("MISTRAL_API_KEY")
MISTRAL_OCR_URL = os.getenv("MISTRAL_OCR_URL", "https://api.mistral.ai/v1/ocr")
MISTRAL_OCR_MODEL = os.getenv("MISTRAL_OCR_MODEL", "mistral-ocr-latest")

# ------------------------------------------------------------- storage ----

EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "1024"))
VECTOR_TABLE_NAME = os.getenv("VECTOR_TABLE_NAME", "document_chunks")

# -------------------------------------------------------------- engine ----

# "slots"    — LLM fills a template slot set; Python renders parameterised SQL.
# "freeform" — LLM writes SQL directly, validated by the AST checker.
# Both paths are built; this selects the primary. See query/sql_templates.py.
SQL_STRATEGY = os.getenv("SQL_STRATEGY", "slots")
SQL_MAX_ROWS = int(os.getenv("SQL_MAX_ROWS", "1000"))
SQL_STATEMENT_TIMEOUT = os.getenv("SQL_STATEMENT_TIMEOUT", "10s")

# --------------------------------------------------------------- memory ----

MEMORY_MAX_TURNS = int(os.getenv("MEMORY_MAX_TURNS", "6"))
MEMORY_MAX_CHARS = int(os.getenv("MEMORY_MAX_CHARS", "2000"))

# ------------------------------------------------------------------ api ----

API_PORT = int(os.getenv("API_PORT", "8000"))
FLASK_DEBUG = os.getenv("FLASK_DEBUG", "false").lower() == "true"
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(50 * 1024 * 1024)))
INACTIVITY_TIMEOUT_SECONDS = int(os.getenv("INACTIVITY_TIMEOUT_SECONDS", str(2 * 60 * 60)))


# ----------------------------------------------------------- validation ----

_REQUIRED = {
    "database": lambda: get_db_url(),
    "GROQ_API_KEY": lambda: GROQ_API_KEY,
    "JINA_API_KEY": lambda: JINA_API_KEY,
    "MISTRAL_OCR": lambda: MISTRAL_API_KEY,
}


def require() -> None:
    """Raise with a single readable message if anything essential is missing."""
    missing = []
    for name, getter in _REQUIRED.items():
        try:
            if not getter():
                missing.append(name)
        except RuntimeError:
            missing.append(name)
    if missing:
        raise RuntimeError(
            "Missing required configuration: "
            + ", ".join(missing)
            + f".\nExpected in {PROJECT_ROOT / '.env'} — see .env.example."
        )


def check() -> bool:
    """
    Probe every provider and print a pass/fail table.

    Exists because two of the five credentials in this project were broken in
    ways no code path reported clearly: an unencoded '@' in the database
    password, and a retired Groq model returning 404 rather than 401.

    Run with:  python -m indexial.core.config
    """
    import requests

    ok = True

    def report(name: str, passed: bool, detail: str) -> None:
        nonlocal ok
        ok = ok and passed
        print(f"  {'PASS' if passed else 'FAIL'}  {name:<24} {detail}")

    print(f"\nIndexial configuration check\n  root: {PROJECT_ROOT}\n")

    try:
        import psycopg2

        with psycopg2.connect(get_db_url(), connect_timeout=15) as conn:
            cur = conn.cursor()
            cur.execute("select current_user, current_database()")
            user, db = cur.fetchone()
            cur.execute("select extname from pg_extension where extname in ('vector','pg_trgm')")
            exts = sorted(r[0] for r in cur.fetchall())
            report("database", True, f"{user}@{db}")
            report(
                "extensions",
                {"vector", "pg_trgm"}.issubset(set(exts)),
                ", ".join(exts) or "none installed",
            )
    except Exception as exc:  # noqa: BLE001 - surfacing any failure is the point
        report("database", False, str(exc).strip().splitlines()[0][:120])

    try:
        r = requests.post(
            GROQ_API_URL,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={
                "model": GROQ_MODEL,
                "messages": [{"role": "user", "content": "ok"}],
                "max_tokens": 4,
            },
            timeout=30,
        )
        detail = GROQ_MODEL if r.ok else f"{r.status_code} {r.json().get('error', {}).get('message', '')[:80]}"
        report("groq", r.ok, detail)
    except Exception as exc:  # noqa: BLE001
        report("groq", False, str(exc)[:120])

    try:
        r = requests.post(
            JINA_API_URL,
            headers={"Authorization": f"Bearer {JINA_API_KEY}"},
            json={
                "model": JINA_MODEL,
                "task": JINA_TASK,
                "dimensions": EMBEDDING_DIMENSIONS,
                "input": ["check"],
            },
            timeout=30,
        )
        dims = len(r.json()["data"][0]["embedding"]) if r.ok else 0
        report("jina", r.ok and dims == EMBEDDING_DIMENSIONS, f"{dims} dims")
    except Exception as exc:  # noqa: BLE001
        report("jina", False, str(exc)[:120])

    try:
        r = requests.get(
            "https://api.mistral.ai/v1/models",
            headers={"Authorization": f"Bearer {MISTRAL_API_KEY}"},
            timeout=30,
        )
        ids = [m["id"] for m in r.json().get("data", [])] if r.ok else []
        report("mistral", MISTRAL_OCR_MODEL in ids, MISTRAL_OCR_MODEL)
    except Exception as exc:  # noqa: BLE001
        report("mistral", False, str(exc)[:120])

    print()
    return ok


if __name__ == "__main__":
    import sys

    sys.exit(0 if check() else 1)
