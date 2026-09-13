"""
SQL validator tests. No database, no network, no API keys.

Each of the first three groups pins one hole that the previous regex-based
validator left open.
"""

import pytest

from indexial.query.sql_validator import enforce_limit, validate_sql

OK = "SELECT value_num FROM document_facts WHERE table_id = 'x'"


def ok(sql):
    passed, _ = validate_sql(sql)
    return passed


def why(sql):
    _, error = validate_sql(sql)
    return error or ""


# ----------------------------------------------------- hole (a): whitelist --

def test_arbitrary_relation_is_rejected():
    assert not ok("SELECT * FROM secrets WHERE table_id = 'x'")


def test_tbl_prefix_no_longer_grants_access():
    """The old check accepted any name starting with 'tbl_'."""
    assert not ok("SELECT * FROM tbl_evil_extracted WHERE table_id = 'x'")


def test_information_schema_is_rejected():
    """It used to be explicitly exempted, which leaked other tables' names."""
    assert not ok("SELECT table_name FROM information_schema.tables WHERE table_id='x'")


def test_non_public_schema_is_rejected():
    assert not ok("SELECT * FROM pg_catalog.pg_tables WHERE table_id = 'x'")


def test_allowed_relations_pass():
    assert ok(OK)
    assert ok("SELECT headers FROM table_registry WHERE document_id = 'd'")


def test_cte_names_are_not_mistaken_for_tables():
    assert ok(
        "WITH matching AS (SELECT row_index FROM document_facts WHERE table_id='x') "
        "SELECT * FROM matching"
    )


# ------------------------------------------------ hole (b): multi-statement --

def test_injected_second_statement_is_rejected():
    """One semicolon total, so the old `.count(';') > 1` check let it pass."""
    assert not ok("SELECT 1 FROM document_facts WHERE table_id='x'; DROP TABLE documents")


def test_trailing_semicolon_is_fine():
    assert ok(OK + ";")


def test_comment_hidden_statement_is_rejected():
    assert not ok(
        "SELECT value_num FROM document_facts WHERE table_id='x' /* */; DELETE FROM documents"
    )


# ---------------------------------------------------- hole (c): false hits --

def test_column_named_update_is_allowed():
    """The old blocklist uppercased the whole query and matched the word."""
    assert ok("SELECT update FROM document_facts WHERE table_id = 'x'")


def test_literal_containing_a_keyword_is_allowed():
    assert ok("SELECT value_text FROM document_facts WHERE table_id='x' AND row_label = 'DELETE'")


def test_column_named_delete_is_allowed():
    assert ok("SELECT value_num AS delete FROM document_facts WHERE table_id = 'x'")


# --------------------------------------------------------- write operations --

@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM documents WHERE table_id='x'",
        "UPDATE document_facts SET value_num = 0 WHERE table_id='x'",
        "INSERT INTO documents (filename) VALUES ('x')",
        "DROP TABLE documents",
        "ALTER TABLE documents ADD COLUMN x int",
        "CREATE TABLE evil (id int)",
        "TRUNCATE documents",
    ],
)
def test_writes_are_rejected(sql):
    assert not ok(sql)


def test_dangerous_functions_are_rejected():
    assert not ok("SELECT pg_read_file('/etc/passwd') FROM document_facts WHERE table_id='x'")
    assert not ok("SELECT pg_sleep(10) FROM document_facts WHERE table_id='x'")


# ------------------------------------------------------------------- scope --

def test_unscoped_query_is_rejected():
    """Long format has no implicit scope; without a predicate this scans all."""
    assert not ok("SELECT value_num FROM document_facts")
    assert "table_id" in why("SELECT value_num FROM document_facts")


def test_document_id_also_counts_as_scope():
    assert ok("SELECT value_num FROM document_facts WHERE document_id = 'd'")


# ------------------------------------------------------------------ basics --

def test_empty_and_garbage():
    assert not ok("")
    assert not ok("   ")
    assert not ok("this is not sql at all !!!")


# ------------------------------------------------------------------ limits --

def test_limit_is_added_when_missing():
    assert "LIMIT 100" in enforce_limit(OK, 100).upper()


def test_smaller_existing_limit_is_kept():
    assert "LIMIT 10" in enforce_limit(OK + " LIMIT 10", 100).upper()


def test_larger_limit_is_capped():
    out = enforce_limit(OK + " LIMIT 9999", 100).upper()
    assert "LIMIT 100" in out and "9999" not in out


def test_subquery_limit_is_not_rewritten():
    """The regex version rewrote every LIMIT, silently changing the result."""
    sql = (
        "SELECT * FROM (SELECT row_index FROM document_facts "
        "WHERE table_id='x' LIMIT 5) t"
    )
    out = enforce_limit(sql, 100)
    assert "LIMIT 5" in out.upper()
    assert out.upper().rstrip().endswith("LIMIT 100")
