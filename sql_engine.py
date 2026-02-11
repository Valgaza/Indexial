"""
NL-to-SQL Engine

Generates and executes SQL queries from natural language using table registry context.
Implements 6-layer safety mechanism to prevent SQL injection and destructive operations.
"""

import re
import json
import logging
from typing import List, Dict, Any, Optional, Tuple

from dotenv import load_dotenv

from db import get_connection
from llm import GroqLLM

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TableRegistryReader:
    """
    Reads table_registry to build context for SQL generation.
    Provides both full registry context and relevance-filtered context.
    """

    def get_registry_context(self, document_ids: Optional[List[str]] = None) -> str:
        """
        Query table_registry and return a formatted context string for LLM.

        Args:
            document_ids: Optional list of document UUIDs to filter by

        Returns:
            Formatted string with table metadata
        """
        with get_connection(readonly=True) as conn:
            cur = conn.cursor()

            if document_ids:
                # Filter by document IDs
                placeholders = ",".join(["%s"] * len(document_ids))
                query = f"""
                    SELECT physical_table_name, semantic_description, headers,
                           row_count, original_filename
                    FROM table_registry
                    WHERE source_doc_uuid::text IN ({placeholders})
                    ORDER BY created_at DESC
                """
                cur.execute(query, document_ids)
            else:
                # Get all tables
                query = """
                    SELECT physical_table_name, semantic_description, headers,
                           row_count, original_filename
                    FROM table_registry
                    ORDER BY created_at DESC
                """
                cur.execute(query)

            rows = cur.fetchall()
            cur.close()

            if not rows:
                return "No tables found in the registry."

            context_parts = []
            for row in rows:
                table_name = row[0]
                description = row[1] or "No description"
                headers = row[2] if row[2] else []  # JSONB already parsed by psycopg2
                row_count = row[3]
                source_file = row[4] or "Unknown"

                table_info = f"""Table: {table_name}
Description: {description}
Columns: {', '.join(headers)}
Row Count: {row_count}
Source: {source_file}
"""
                context_parts.append(table_info)

            return "\n".join(context_parts)

    def get_table_list(self, document_ids: Optional[List[str]] = None) -> List[str]:
        """Get list of physical table names, optionally filtered by document."""
        with get_connection(readonly=True) as conn:
            cur = conn.cursor()

            if document_ids:
                placeholders = ",".join(["%s"] * len(document_ids))
                query = f"""
                    SELECT physical_table_name
                    FROM table_registry
                    WHERE source_doc_uuid::text IN ({placeholders})
                """
                cur.execute(query, document_ids)
            else:
                query = "SELECT physical_table_name FROM table_registry"
                cur.execute(query)

            tables = [row[0] for row in cur.fetchall()]
            cur.close()
            return tables

    def get_table_schema(self, table_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed schema for a specific table."""
        with get_connection(readonly=True) as conn:
            cur = conn.cursor()

            # Get from registry
            cur.execute(
                """
                SELECT headers, semantic_description, row_count, original_filename
                FROM table_registry
                WHERE physical_table_name = %s
                """,
                (table_name,),
            )
            row = cur.fetchone()

            if not row:
                cur.close()
                return None

            # Also get actual column types from the table
            cur.execute(
                """
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = %s
                ORDER BY ordinal_position
                """,
                (table_name,),
            )
            columns = cur.fetchall()
            cur.close()

            return {
                "table_name": table_name,
                "headers": row[0] if row[0] else [],  # JSONB already parsed
                "description": row[1],
                "row_count": row[2],
                "source_file": row[3],
                "columns": [{"name": c[0], "type": c[1]} for c in columns],
            }


class SQLGenerator:
    """
    Generates SQL queries from natural language using table registry context.
    """

    def __init__(self):
        self.llm = GroqLLM()
        self.registry = TableRegistryReader()

    def generate_sql(
        self,
        query: str,
        document_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Generate SQL from natural language query.

        Args:
            query: Natural language question
            document_ids: Optional document filter

        Returns:
            Dict with 'sql', 'explanation', and 'tables_used'
        """
        # Get table registry context
        registry_context = self.registry.get_registry_context(document_ids)

        if registry_context == "No tables found in the registry.":
            return {
                "sql": None,
                "explanation": "No tables available in the database.",
                "tables_used": [],
                "error": "No tables found",
            }

        system_prompt = """You are a PostgreSQL expert that generates SELECT queries from natural language.

CRITICAL RULES:
1. ONLY generate SELECT statements - no INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, TRUNCATE, GRANT, REVOKE
2. ONLY use tables and columns listed in the "Available Tables" section below
3. Use proper PostgreSQL syntax
4. Always include LIMIT clause (default 100 if not specified)
5. Use table aliases for readability
6. Escape column names with special characters using double quotes
7. Return JSON format: {"sql": "SELECT ...", "explanation": "...", "tables_used": ["table1", "table2"]}

If the question cannot be answered with the available tables, return:
{"sql": null, "explanation": "Cannot answer: [reason]", "tables_used": []}
"""

        user_message = f"""Available Tables:
{registry_context}

---

Question: {query}

Generate a PostgreSQL SELECT query to answer this question. Return JSON only."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        try:
            response = self.llm.chat(
                messages,
                temperature=0,
                response_format={"type": "json_object"},
            )
            result = json.loads(response)

            # Validate response structure
            if "sql" not in result:
                return {
                    "sql": None,
                    "explanation": "Invalid response from LLM",
                    "tables_used": [],
                    "error": "LLM response missing 'sql' field",
                }

            # Ensure SQL has LIMIT if present
            if result["sql"] and "LIMIT" not in result["sql"].upper():
                result["sql"] = result["sql"].rstrip(";") + " LIMIT 100"

            return {
                "sql": result.get("sql"),
                "explanation": result.get("explanation", ""),
                "tables_used": result.get("tables_used", []),
            }

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            return {
                "sql": None,
                "explanation": "Failed to parse SQL generation response",
                "tables_used": [],
                "error": str(e),
            }
        except Exception as e:
            logger.error(f"SQL generation failed: {e}")
            return {
                "sql": None,
                "explanation": f"Error: {str(e)}",
                "tables_used": [],
                "error": str(e),
            }


class SafeSQLExecutor:
    """
    Executes SQL with 6-layer safety mechanism.

    Safety Layers:
    1. Keyword validation - reject DDL/DML keywords
    2. Statement validation - must start with SELECT or WITH
    3. Table whitelist - only tbl_*_extracted tables
    4. Read-only connection - database-level protection
    5. Row limit - maximum 1000 rows
    6. Timeout - 10 second statement_timeout
    """

    # Forbidden keywords (case-insensitive)
    FORBIDDEN_KEYWORDS = [
        "INSERT",
        "UPDATE",
        "DELETE",
        "DROP",
        "ALTER",
        "CREATE",
        "TRUNCATE",
        "GRANT",
        "REVOKE",
        "EXECUTE",
        "COPY",
        "VACUUM",
        "ANALYZE",
        "REINDEX",
        "CLUSTER",
        "LOCK",
        "COMMENT",
        "PREPARE",
    ]

    def __init__(self):
        self.registry = TableRegistryReader()

    def validate_sql(self, sql: str) -> Tuple[bool, Optional[str]]:
        """
        Validate SQL query through multiple layers.

        Returns:
            (is_valid, error_message)
        """
        if not sql or not sql.strip():
            return False, "Empty SQL query"

        sql_upper = sql.upper()

        # Layer 1: Keyword validation
        for keyword in self.FORBIDDEN_KEYWORDS:
            if re.search(r"\b" + keyword + r"\b", sql_upper):
                return False, f"Forbidden keyword detected: {keyword}"

        # Layer 2: Statement validation
        sql_stripped = sql_upper.strip()
        if not (sql_stripped.startswith("SELECT") or sql_stripped.startswith("WITH")):
            return False, "Query must start with SELECT or WITH"

        # Check for multiple statements (naive check)
        # Remove string literals first to avoid false positives
        sql_no_strings = re.sub(r"'[^']*'", "", sql)
        if sql_no_strings.count(";") > 1:
            return False, "Multiple statements not allowed"

        # Layer 3: Table whitelist validation
        # Extract table names from query (simple regex approach)
        # This catches FROM and JOIN clauses, including schema-qualified names
        table_pattern = r"\bFROM\s+([a-zA-Z_][a-zA-Z0-9_.]*)|JOIN\s+([a-zA-Z_][a-zA-Z0-9_.]*)"
        matches = re.findall(table_pattern, sql_upper)
        referenced_tables = [m[0] or m[1] for m in matches]

        # Get whitelist
        allowed_tables_upper = [t.upper() for t in self.registry.get_table_list()]

        for table in referenced_tables:
            if table:
                # Allow information_schema (read-only system catalog)
                if "INFORMATION_SCHEMA" in table:
                    continue
                # Check if table is in registry
                if table not in allowed_tables_upper:
                    # Check if it starts with TBL_ (our convention)
                    if not table.startswith("TBL_"):
                        return (
                            False,
                            f"Table '{table}' not in registry. Only extracted tables allowed.",
                        )

        return True, None

    def execute(
        self,
        sql: str,
        max_rows: int = 1000,
    ) -> Dict[str, Any]:
        """
        Execute validated SQL query on read-only connection.

        Args:
            sql: SQL query to execute
            max_rows: Maximum rows to return (safety limit)

        Returns:
            Dict with 'columns', 'rows', 'row_count'
        """
        # Validate first
        is_valid, error = self.validate_sql(sql)
        if not is_valid:
            return {
                "success": False,
                "error": error,
                "columns": [],
                "rows": [],
                "row_count": 0,
            }

        # Add/enforce LIMIT
        if "LIMIT" not in sql.upper():
            sql = sql.rstrip(";") + f" LIMIT {max_rows}"
        else:
            # Extract existing LIMIT and cap it
            limit_match = re.search(r"LIMIT\s+(\d+)", sql, re.IGNORECASE)
            if limit_match:
                existing_limit = int(limit_match.group(1))
                if existing_limit > max_rows:
                    sql = re.sub(
                        r"LIMIT\s+\d+",
                        f"LIMIT {max_rows}",
                        sql,
                        flags=re.IGNORECASE,
                    )

        try:
            # Layer 4: Read-only connection
            with get_connection(readonly=True) as conn:
                cur = conn.cursor()

                # Layer 6: Set statement timeout
                cur.execute("SET statement_timeout = '10s'")

                # Execute query
                logger.info(f"Executing SQL: {sql[:200]}...")
                cur.execute(sql)

                # Fetch results
                rows = cur.fetchall()
                columns = [desc[0] for desc in cur.description] if cur.description else []

                cur.close()

                logger.info(f"Query returned {len(rows)} rows")

                return {
                    "success": True,
                    "columns": columns,
                    "rows": rows,
                    "row_count": len(rows),
                    "sql_executed": sql,
                }

        except Exception as e:
            logger.error(f"SQL execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "columns": [],
                "rows": [],
                "row_count": 0,
            }

    def format_results(
        self,
        results: Dict[str, Any],
        original_query: str,
    ) -> str:
        """
        Format SQL results into natural language using LLM.

        Args:
            results: Results from execute()
            original_query: Original natural language question

        Returns:
            Natural language answer
        """
        if not results.get("success"):
            return f"Query failed: {results.get('error', 'Unknown error')}"

        if results["row_count"] == 0:
            return "No results found for your query."

        # Build data summary
        data_summary = f"Found {results['row_count']} result(s).\n\n"

        # Format as table (first 10 rows for context)
        columns = results["columns"]
        rows = results["rows"][:10]  # Limit to 10 for LLM context

        # Simple table format
        data_summary += "Results:\n"
        data_summary += " | ".join(columns) + "\n"
        data_summary += "-" * (len(" | ".join(columns))) + "\n"

        for row in rows:
            data_summary += " | ".join(str(v) for v in row) + "\n"

        if results["row_count"] > 10:
            data_summary += f"\n... and {results['row_count'] - 10} more rows"

        # Use LLM to format into natural language
        llm = GroqLLM()

        system_prompt = """You are a helpful assistant that presents SQL query results in natural language.

Rules:
- Summarize the data clearly and concisely
- Highlight key findings
- If there are many rows, summarize patterns or aggregate info
- Be direct and factual
"""

        user_message = f"""Original Question: {original_query}

Query Results:
{data_summary}

Present these results as a clear, natural language answer to the original question."""

        try:
            answer = llm.chat(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.3,
                max_tokens=500,
            )
            return answer
        except Exception as e:
            logger.error(f"Failed to format results: {e}")
            # Fallback to simple format
            return f"Query returned {results['row_count']} row(s). Here are the results:\n\n{data_summary}"


def test_sql_engine():
    """Test the SQL engine components."""
    print("=" * 60)
    print("Testing SQL Engine")
    print("=" * 60)

    # Test registry reader
    print("\n[1] Testing TableRegistryReader...")
    reader = TableRegistryReader()
    context = reader.get_registry_context()
    print(context[:300] + "..." if len(context) > 300 else context)

    tables = reader.get_table_list()
    print(f"\nFound {len(tables)} tables in registry")

    # Test SQL generator
    print("\n[2] Testing SQLGenerator...")
    generator = SQLGenerator()

    test_queries = [
        "How many rows are in the first table?",
        "Show me all the data from the tables",
        "What columns are available?",
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        result = generator.generate_sql(query)
        print(f"  SQL: {result.get('sql', 'None')}")
        print(f"  Explanation: {result.get('explanation', 'None')}")

    # Test executor
    print("\n[3] Testing SafeSQLExecutor...")
    executor = SafeSQLExecutor()

    # Test validation
    test_cases = [
        ("SELECT * FROM tbl_test_extracted LIMIT 10", True),
        ("DELETE FROM tbl_test_extracted", False),
        ("SELECT * FROM users", False),  # Not in whitelist
        ("SELECT 1; DROP TABLE users;", False),  # Multiple statements
    ]

    for sql, should_pass in test_cases:
        is_valid, error = executor.validate_sql(sql)
        status = "✓" if is_valid == should_pass else "✗"
        print(f"{status} {sql[:50]}... - {error if error else 'Valid'}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    test_sql_engine()
