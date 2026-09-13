"""
Shared Database Module

Provides context-managed PostgreSQL connections for all modules.

The DSN is resolved once in core.config, which prefers DIRECT_URL (session
mode) because both settings used here — readonly and statement_timeout — are
session-scoped and are not preserved by a transaction pooler.
"""

import logging
from contextlib import contextmanager

import psycopg2

from indexial.core.config import get_db_url

logger = logging.getLogger(__name__)

__all__ = ["get_db_url", "get_connection", "get_cursor"]


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
