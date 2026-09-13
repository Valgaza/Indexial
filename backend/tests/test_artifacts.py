"""
Artifact classifier tests. No database, no network, no API keys.

The tests that matter are the purity guards and the refusals. Everything else
is regression coverage.
"""

import ast
import inspect
import json
import uuid
from datetime import date, datetime, timezone
from decimal import Decimal

import pytest

from indexial.query import artifacts
from indexial.query.artifacts import build_artifact, describe_columns, json_safe, slug_keys

INT4, INT8, TEXT, NUM, DATE, F8, BOOL, UUIDO, MONEY = 23, 20, 25, 1700, 1082, 701, 16, 2950, 790


def ex(columns, oids, rows, sql="SELECT x FROM document_facts WHERE table_id='t'"):
    return {
        "success": True,
        "columns": columns,
        "type_oids": oids,
        "rows": rows,
        "records": [],
        "row_count": len(rows),
        "sql_executed": sql,
    }


def kind_of(result):
    arts = build_artifact(result, title="t")
    return arts[0]["type"] if arts else None


# ------------------------------------------------------------- purity ------

def test_artifacts_module_imports_nothing_generative():
    """The core invariant: no LLM, no database, no HTTP in this module."""
    source = inspect.getsource(artifacts)
    banned = {"llm", "db", "flask", "requests", "groq", "psycopg2"}
    found = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module.split(".")[0])
            if "." in node.module:
                found |= set(node.module.split("."))
    assert not (found & banned), f"artifacts.py must not import {found & banned}"


def test_classify_cannot_see_the_question():
    """Intent must never influence chart choice."""
    params = set(inspect.signature(artifacts.classify).parameters)
    assert not (params & {"query", "question", "prompt"})


# ---------------------------------------------------------- column types ---

def test_oid_typing():
    metas = describe_columns(
        ["a", "b", "c", "d"], [INT4, TEXT, DATE, MONEY],
        [[1, "x", date(2024, 1, 1), "$5.00"]],
    )
    assert [m.kind for m in metas] == ["numeric", "categorical", "temporal", "opaque"]


def test_unknown_oid_is_opaque():
    assert describe_columns(["x"], [9999], [[1]])[0].kind == "opaque"


def test_identifier_columns_are_demoted():
    """row_index as a measure is the most likely wrong chart in this system."""
    for name in ["row_index", "document_id", "page_number", "table_index", "user_id"]:
        meta = describe_columns([name], [INT4], [[1], [2]])[0]
        assert meta.role == "identifier", name


def test_all_null_numeric_is_demoted():
    """Long-format results carry value_num/value_text/value_date side by side."""
    meta = describe_columns(["value_num"], [NUM], [[None], [None]])[0]
    assert meta.role == "empty_measure"


def test_constant_numeric_is_demoted():
    meta = describe_columns(["x"], [INT4], [[5], [5], [5], [5]])[0]
    assert meta.role == "constant"


# ------------------------------------------------------- decision table ----

def test_empty_result():
    assert kind_of(ex([], [], [])) is None or kind_of(ex(["a"], [INT4], [])) == "empty"


def test_single_scalar():
    assert kind_of(ex(["total"], [NUM], [[Decimal("42")]])) == "scalar"


def test_temporal_series_is_a_line():
    rows = [[date(2024, m, 1), Decimal(m * 10)] for m in range(1, 13)]
    assert kind_of(ex(["month", "revenue"], [DATE, NUM], rows)) == "line"


def test_line_is_sorted_by_the_axis_server_side():
    rows = [[date(2024, m, 1), Decimal(m)] for m in (5, 1, 12, 3)]
    art = build_artifact(ex(["month", "v"], [DATE, NUM], rows), title="t")[0]
    months = [r["month"] for r in art["data"]]
    assert months == sorted(months)


def test_too_many_series_falls_back_to_table():
    rows = [[date(2024, 1, 1)] + [Decimal(i) for i in range(6)]]
    cols = ["d"] + [f"m{i}" for i in range(6)]
    oids = [DATE] + [NUM] * 6
    assert kind_of(ex(cols, oids, rows * 3)) == "table"


@pytest.mark.parametrize(
    "n,expected",
    [(5, "bar"), (40, "bar_horizontal"), (200, "table")],
)
def test_category_measure_by_row_count(n, expected):
    rows = [[f"c{i}", Decimal(i)] for i in range(n)]
    assert kind_of(ex(["label", "value"], [TEXT, NUM], rows)) == expected


def test_grouped_bar():
    rows = [[f"c{i}", Decimal(i), Decimal(i * 2), Decimal(i * 3)] for i in range(8)]
    assert kind_of(ex(["seg", "a", "b", "c"], [TEXT, NUM, NUM, NUM], rows)) == "bar_grouped"


def test_scatter_needs_enough_points():
    many = [[Decimal(i), Decimal(i * 2)] for i in range(50)]
    few = [[Decimal(i), Decimal(i * 2)] for i in range(4)]
    assert kind_of(ex(["x", "y"], [F8, F8], many)) == "scatter"
    assert kind_of(ex(["x", "y"], [F8, F8], few)) == "table"


def test_all_text_is_a_table():
    rows = [["a", "b", "c"], ["d", "e", "f"]]
    assert kind_of(ex(["p", "q", "r"], [TEXT, TEXT, TEXT], rows)) == "table"


def test_too_many_columns_is_a_table():
    cols = [f"c{i}" for i in range(15)]
    rows = [[Decimal(i) for i in range(15)] for _ in range(3)]
    assert kind_of(ex(cols, [NUM] * 15, rows)) == "table"


# --------------------------------------------------------- truncation ------

def test_unordered_truncation_refuses_to_chart():
    """An arbitrary 100 of an unknown N must not become a ranked bar chart."""
    rows = [[f"c{i}", Decimal(i)] for i in range(100)]
    sql = "SELECT label, value FROM document_facts WHERE table_id='t' LIMIT 100"
    art = build_artifact(ex(["label", "value"], [TEXT, NUM], rows, sql), title="t")[0]
    assert art["kind"] == "table"
    assert art["classifier"]["rule"] == "R5_unordered_truncation"


def test_ordered_truncation_may_chart_but_is_flagged():
    rows = [[f"c{i}", Decimal(i)] for i in range(20)]
    sql = ("SELECT label, value FROM document_facts WHERE table_id='t' "
           "ORDER BY value DESC LIMIT 20")
    art = build_artifact(ex(["label", "value"], [TEXT, NUM], rows, sql), title="t")[0]
    assert art["kind"] == "chart"
    assert art["provenance"]["partial"] is True


# ------------------------------------------------------- serialisation -----

def test_json_safe_conversions():
    assert json_safe(Decimal("1.50")) == 1.5
    assert json_safe(Decimal("NaN")) is None
    assert json_safe(float("inf")) is None
    assert json_safe(float("nan")) is None
    assert json_safe(date(2024, 1, 31)) == "2024-01-31"
    assert json_safe(datetime(2024, 1, 31, 9, 30, tzinfo=timezone.utc)).startswith("2024-01-31T09:30")
    assert isinstance(json_safe(uuid.uuid4()), str)
    assert json_safe(b"abcd") == "<4 bytes>"
    assert json_safe(2**60) == str(2**60)     # JS would round this
    assert json_safe(42) == 42


def test_every_artifact_is_json_serialisable():
    """Single highest-value assertion: catches every hazard at once."""
    fixtures = [
        ex(["total"], [NUM], [[Decimal("42.50")]]),
        ex(["d", "v"], [DATE, NUM], [[date(2024, m, 1), Decimal(m)] for m in range(1, 6)]),
        ex(["l", "v"], [TEXT, NUM], [[f"c{i}", Decimal(i)] for i in range(5)]),
        ex(["id", "raw"], [INT8, UUIDO], [[2**60, uuid.uuid4()]]),
        ex(["x"], [F8], [[float("nan")], [float("inf")], [1.5]]),
    ]
    for fixture in fixtures:
        for art in build_artifact(fixture, title="t"):
            json.dumps(art)  # must not raise


def test_charted_values_are_never_strings():
    rows = [[f"c{i}", Decimal(f"{i}.25")] for i in range(5)]
    art = build_artifact(ex(["label", "value"], [TEXT, NUM], rows), title="t")[0]
    for y in art["encoding"]["y"]:
        for record in art["data"]:
            assert isinstance(record[y["key"]], (int, float, type(None)))


# --------------------------------------------------------------- slugs -----

def test_slug_keys_disambiguate():
    assert slug_keys(["Total Revenue (₹)", "id", "id", "2024"]) == [
        "total_revenue", "id", "id_2", "c_2024",
    ]


# --------------------------------------------------------- determinism -----

def test_classification_is_deterministic():
    rows = [[f"c{i}", Decimal(i)] for i in range(6)]
    a = build_artifact(ex(["l", "v"], [TEXT, NUM], rows), title="t")[0]
    b = build_artifact(ex(["l", "v"], [TEXT, NUM], rows), title="t")[0]
    assert a == b


def test_failed_execution_produces_no_artifact():
    assert build_artifact({"success": False, "error": "boom"}) == []
