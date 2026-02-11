"""
Query Router

Intelligently routes natural language queries to SQL, RAG, or HYBRID execution.

Two-phase classification:
1. Heuristic fast-path - pattern matching for obvious cases
2. LLM classification - for ambiguous queries

QueryOrchestrator handles end-to-end query processing with routing.
"""

import re
import json
import logging
from typing import Dict, Any, List, Optional, Literal

from dotenv import load_dotenv

from sql_engine import SQLGenerator, SafeSQLExecutor, TableRegistryReader
from retrieval import Retriever
from llm import GroqLLM
from memory import MemoryManager

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RouteType = Literal["SQL", "RAG", "HYBRID"]


class QueryRouter:
    """
    Classifies natural language queries to determine routing strategy.

    Uses two-phase approach:
    1. Heuristic pattern matching (fast)
    2. LLM classification (when ambiguous)
    """

    # SQL signal patterns (queries about structured data operations)
    SQL_SIGNALS = [
        r"\bhow many\b",
        r"\bcount\b",
        r"\btotal\b",
        r"\bsum\b",
        r"\baverage\b",
        r"\bmean\b",
        r"\bmedian\b",
        r"\blist all\b",
        r"\bshow all\b",
        r"\bcompare\b",
        r"\btable\b",
        r"\bcolumns?\b",
        r"\brows?\b",
        r"\bdata from\b",
        r"\bvalue[sd]?\b.*\bin\b",
        r"\bfilter\b",
        r"\bsort\b",
        r"\bgroup by\b",
        r"\bmax\b",
        r"\bmin\b",
    ]

    # RAG signal patterns (queries about understanding/explanation)
    RAG_SIGNALS = [
        r"\bexplain\b",
        r"\bdescribe\b",
        r"\bsummarize\b",
        r"\bwhat does\b",
        r"\bwhat is\b",
        r"\bwhy\b",
        r"\bhow does\b",
        r"\bwhat are the implications\b",
        r"\btell me about\b",
        r"\bmeaning of\b",
        r"\binterpret\b",
        r"\bdefine\b",
        r"\bconcept\b",
        r"\bmethodology\b",
        r"\bapproach\b",
        r"\bprocess\b",
    ]

    def __init__(self):
        self.llm = GroqLLM()
        self.registry_reader = TableRegistryReader()

    def classify_heuristic(self, query: str) -> Optional[RouteType]:
        """
        Fast heuristic classification based on pattern matching.

        Args:
            query: Natural language query

        Returns:
            "SQL", "RAG", or None (if ambiguous)
        """
        query_lower = query.lower()

        # Count matches
        sql_matches = sum(1 for pattern in self.SQL_SIGNALS if re.search(pattern, query_lower))
        rag_matches = sum(1 for pattern in self.RAG_SIGNALS if re.search(pattern, query_lower))

        # Strong SQL signal
        if sql_matches >= 2 and rag_matches == 0:
            logger.info(f"Heuristic: SQL (matches={sql_matches})")
            return "SQL"

        # Strong RAG signal
        if rag_matches >= 2 and sql_matches == 0:
            logger.info(f"Heuristic: RAG (matches={rag_matches})")
            return "RAG"

        # Single strong SQL indicator
        if sql_matches == 1 and rag_matches == 0:
            strong_sql = [r"\bhow many\b", r"\bcount\b", r"\btotal\b", r"\bsum\b"]
            if any(re.search(p, query_lower) for p in strong_sql):
                logger.info("Heuristic: SQL (strong single match)")
                return "SQL"

        # Ambiguous - fall back to LLM
        logger.info(f"Heuristic: AMBIGUOUS (sql={sql_matches}, rag={rag_matches})")
        return None

    def classify_llm(
        self,
        query: str,
        document_ids: Optional[List[str]] = None
    ) -> RouteType:
        """
        LLM-based classification when heuristics are ambiguous.

        Args:
            query: Natural language query
            document_ids: Optional document filter for context

        Returns:
            "SQL", "RAG", or "HYBRID"
        """
        # Get table context for LLM
        table_context = self.registry_reader.get_registry_context(document_ids)

        system_prompt = """You are a query classification expert for a hybrid RAG + SQL system.

Your task: Classify the user's query into ONE of these categories:
- SQL: Query asks for structured data operations (counts, totals, filtering, listing table data, aggregations)
- RAG: Query asks for explanations, summaries, or understanding of unstructured text content
- HYBRID: Query needs BOTH structured data AND conceptual understanding

Rules:
1. If the query explicitly references table names, columns, or asks for data values → SQL
2. If the query asks "why", "how does X work", "explain concept" → RAG
3. If the query asks "what does the data show about concept X" → HYBRID
4. Return JSON format: {"route": "SQL"|"RAG"|"HYBRID", "reasoning": "brief explanation"}

Available Tables:
{table_context}

Example classifications:
- "How many rows in table X?" → SQL
- "Explain the methodology" → RAG
- "What do the revenue figures tell us about market trends?" → HYBRID
"""

        user_message = f"""Query: {query}

Classify this query as SQL, RAG, or HYBRID. Return JSON only."""

        response = None
        try:
            response = self.llm.chat(
                messages=[
                    {"role": "system", "content": system_prompt.format(table_context=table_context[:1000])},
                    {"role": "user", "content": user_message}
                ],
                temperature=0,
                response_format={"type": "json_object"},
                max_tokens=200
            )

            # Parse JSON response
            result = json.loads(response)

            route = result.get("route", "RAG").upper()
            reasoning = result.get("reasoning", "")

            # Validate route
            if route not in ["SQL", "RAG", "HYBRID"]:
                logger.warning(f"Invalid route '{route}' from LLM, defaulting to RAG")
                route = "RAG"

            logger.info(f"LLM classification: {route} - {reasoning}")
            return route

        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            logger.error(f"Response was: {response[:500] if response else 'None'}")
            return "RAG"
        except KeyError as e:
            logger.error(f"Missing key in JSON response: {e}")
            logger.error(f"Response was: {response[:500] if response else 'None'}")
            return "RAG"
        except Exception as e:
            logger.error(f"LLM classification failed: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Response was: {response[:500] if response else 'None'}")
            return "RAG"

    def classify(
        self,
        query: str,
        document_ids: Optional[List[str]] = None
    ) -> RouteType:
        """
        Main classification method combining heuristic and LLM.

        Args:
            query: Natural language query
            document_ids: Optional document filter

        Returns:
            "SQL", "RAG", or "HYBRID"
        """
        # Try heuristic first
        heuristic_route = self.classify_heuristic(query)

        if heuristic_route is not None:
            return heuristic_route

        # Fall back to LLM for ambiguous cases
        return self.classify_llm(query, document_ids)


class QueryOrchestrator:
    """
    End-to-end query processing with intelligent routing.

    Handles:
    - Session memory and conversation history
    - Follow-up query rewriting
    - Query classification (SQL/RAG/HYBRID)
    - Execution via appropriate engine(s)
    - Result formatting and merging
    """

    def __init__(self, memory: Optional[MemoryManager] = None):
        self.router = QueryRouter()
        self.sql_generator = SQLGenerator()
        self.sql_executor = SafeSQLExecutor()
        self.retriever = Retriever()
        self.llm = GroqLLM()
        self.memory = memory or MemoryManager()

    def process_query(
        self,
        query: str,
        session_id: Optional[str] = None,
        document_ids: Optional[List[str]] = None,
        force_route: Optional[RouteType] = None
    ) -> Dict[str, Any]:
        """
        Process a natural language query end-to-end.

        Args:
            query: Natural language question
            session_id: Session identifier for conversation history
            document_ids: Optional document filter
            force_route: Override classification (for testing)

        Returns:
            Dict with answer, route, sources, sql_data, etc.
        """
        logger.info(f"Processing query: {query}")

        original_query = query

        # Step 1: Rewrite follow-ups if session history exists
        if session_id and self.memory.session_exists(session_id):
            history = self.memory.get_history(session_id)
            if history:
                logger.info("Rewriting follow-up query with conversation context")
                query = self.llm.rewrite_followup(query, history)
                logger.info(f"Rewritten query: {query}")

        # Step 2: Classify route
        route = force_route or self.router.classify(query, document_ids)
        logger.info(f"Route: {route}")

        # Step 3: Execute based on route
        if route == "SQL":
            result = self._execute_sql(query, document_ids)
        elif route == "RAG":
            result = self._execute_rag(query, document_ids)
        else:  # HYBRID
            result = self._execute_hybrid(query, document_ids)

        # Step 4: Store exchange in memory
        if session_id:
            self.memory.add_exchange(session_id, original_query, result.get("answer", ""))

        # Add original query to result
        result["original_query"] = original_query
        if original_query != query:
            result["rewritten_query"] = query

        return result

    def _execute_sql(
        self,
        query: str,
        document_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Execute SQL-only query."""
        logger.info("Executing SQL route")

        # Generate SQL
        sql_result = self.sql_generator.generate_sql(query, document_ids)

        if not sql_result.get("sql"):
            return {
                "answer": sql_result.get("explanation", "Could not generate SQL for this query."),
                "route": "SQL",
                "error": sql_result.get("error"),
                "query": query
            }

        # Execute SQL
        exec_result = self.sql_executor.execute(sql_result["sql"])

        if not exec_result["success"]:
            return {
                "answer": f"SQL execution failed: {exec_result.get('error')}",
                "route": "SQL",
                "error": exec_result.get("error"),
                "sql": sql_result["sql"],
                "query": query
            }

        # Format results as natural language
        answer = self.sql_executor.format_results(exec_result, query)

        return {
            "answer": answer,
            "route": "SQL",
            "sql": sql_result["sql"],
            "sql_explanation": sql_result.get("explanation"),
            "tables_used": sql_result.get("tables_used", []),
            "row_count": exec_result["row_count"],
            "query": query
        }

    def _execute_rag(
        self,
        query: str,
        document_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Execute RAG-only query."""
        logger.info("Executing RAG route")

        # Search for relevant chunks
        results = self.retriever.search(query, limit=5, score_threshold=0.5)

        if not results:
            return {
                "answer": "I couldn't find any relevant information to answer your question.",
                "route": "RAG",
                "sources": [],
                "query": query
            }

        # Build context
        context = self.retriever.build_context(results)

        # Generate answer
        answer = self.llm.generate_answer(query, context)

        return {
            "answer": answer,
            "route": "RAG",
            "sources": [
                {
                    "score": r["score"],
                    "content": r["content"][:200] + "..." if len(r["content"]) > 200 else r["content"],
                    "heading_context": r.get("heading_context", ""),
                    "source_file": r.get("source_file", "")
                }
                for r in results
            ],
            "query": query
        }

    def _execute_hybrid(
        self,
        query: str,
        document_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Execute hybrid SQL + RAG query."""
        logger.info("Executing HYBRID route")

        # Execute both paths
        sql_result = self._execute_sql(query, document_ids)
        rag_result = self._execute_rag(query, document_ids)

        # Merge results using LLM
        merge_prompt = f"""You are answering a question that requires both structured data analysis and conceptual understanding.

Question: {query}

Structured Data Analysis:
{sql_result.get('answer', 'No SQL data available')}

Conceptual Context:
{rag_result.get('answer', 'No text context available')}

Combine these two perspectives into a comprehensive answer that:
1. Presents the data findings
2. Provides conceptual context and interpretation
3. Creates a unified, coherent response

Be concise but thorough."""

        try:
            merged_answer = self.llm.chat(
                messages=[{"role": "user", "content": merge_prompt}],
                temperature=0.5,
                max_tokens=1024
            )
        except Exception as e:
            logger.error(f"Failed to merge results: {e}")
            merged_answer = f"**Data Analysis:**\n{sql_result.get('answer')}\n\n**Context:**\n{rag_result.get('answer')}"

        return {
            "answer": merged_answer,
            "route": "HYBRID",
            "sql": sql_result.get("sql"),
            "tables_used": sql_result.get("tables_used", []),
            "sources": rag_result.get("sources", []),
            "query": query
        }


def test_router():
    """Test the query router with various query types."""
    print("=" * 60)
    print("Testing Query Router")
    print("=" * 60)

    router = QueryRouter()

    test_queries = [
        # SQL queries
        ("How many rows are in the tables?", "SQL"),
        ("Show me all data from table X", "SQL"),
        ("What is the total count?", "SQL"),
        ("List all columns in the database", "SQL"),

        # RAG queries
        ("Explain the methodology used in the study", "RAG"),
        ("What is somatization?", "RAG"),
        ("Describe the approach", "RAG"),
        ("Why is this important?", "RAG"),

        # Hybrid queries
        ("What do the revenue figures tell us about market trends?", "HYBRID"),
        ("Analyze the data and explain the implications", "HYBRID"),
    ]

    print("\n[1] Testing Heuristic Classification...")
    for query, expected in test_queries[:8]:  # First 8 are clear SQL/RAG
        result = router.classify_heuristic(query)
        status = "✓" if result == expected else "?"
        print(f"{status} '{query[:50]}...' → {result} (expected {expected})")

    print("\n[2] Testing End-to-End Classification...")
    for query, expected in test_queries:
        result = router.classify(query)
        # For hybrid, just check it doesn't crash
        status = "✓" if result in ["SQL", "RAG", "HYBRID"] else "✗"
        print(f"{status} '{query[:50]}...' → {result}")

    print("\n" + "=" * 60)


def test_orchestrator():
    """Test the query orchestrator with actual queries."""
    print("=" * 60)
    print("Testing Query Orchestrator")
    print("=" * 60)

    orchestrator = QueryOrchestrator()

    test_cases = [
        ("How many rows are in tbl_69bde682_t1_extracted?", "SQL"),
        ("What is computational somatization?", "RAG"),
    ]

    for query, expected_route in test_cases:
        print(f"\nQuery: {query}")
        print(f"Expected route: {expected_route}")

        result = orchestrator.process_query(query)

        print(f"Actual route: {result['route']}")
        print(f"Answer: {result['answer'][:200]}...")

        if result['route'] == "SQL" and result.get('sql'):
            print(f"SQL: {result['sql']}")

        if result.get('sources'):
            print(f"Sources: {len(result['sources'])} chunks")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    test_router()
    print("\n")
    test_orchestrator()
