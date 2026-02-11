"""
Shared Database Module

Provides context-managed PostgreSQL connections for all modules.
Centralizes connection logic to eliminate duplication across
table_parser.py, chunker.py, and retrieval.py.
"""

import os
import logging
from contextlib import contextmanager

import psycopg2
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


def get_db_url() -> str:
    """Get database URL from environment."""
    db_url = os.getenv("SUPABASE_DB_URL")
    if not db_url:
        raise ValueError("SUPABASE_DB_URL environment variable is required")
    return db_url


@contextmanager
def get_connection(readonly=False):
    """
    Context-managed database connection.

    Args:
        readonly: If True, sets the transaction to READ ONLY.
                  Critical for safe NL-to-SQL execution.

    Usage:
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT 1")
            conn.commit()
    """
    conn = psycopg2.connect(get_db_url())
    if readonly:
        conn.set_session(readonly=True)
    try:
        yield conn
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


@contextmanager
def get_cursor(readonly=False, commit=False):
    """
    Context-managed cursor with automatic connection cleanup.

    Args:
        readonly: If True, sets the connection to READ ONLY.
        commit: If True, auto-commits on successful exit.

    Usage:
        with get_cursor(commit=True) as (cur, conn):
            cur.execute("INSERT INTO ...")
    """
    with get_connection(readonly=readonly) as conn:
        cur = conn.cursor()
        try:
            yield cur, conn
            if commit:
                conn.commit()
        finally:
            cur.close()
