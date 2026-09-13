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


from indexial.query import artifacts
from indexial.query.sql_engine import SQLGenerator, SafeSQLExecutor, TableRegistryReader
from indexial.query.retrieval import Retriever
from indexial.providers.llm import GroqLLM
from indexial.memory import MemoryManager


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

    # Visualisation wording. Deliberately NOT added to SQL_SIGNALS: chart
    # choice is data-driven, but *reaching* the data is intent-driven, and
    # "plot revenue over time" matches none of the 20 SQL signals. These break
    # a one-weak-signal tie and never force a route on their own.
    VIZ_SIGNALS = [
        r"\bchart\b",
        r"\bplot\b",
        r"\bgraph\b",
        r"\bvisuali[sz]e\b",
        r"\btrend\b",
        r"\bover time\b",
        r"\bbreakdown\b",
        r"\bdistribution\b",
        r"\bby (month|quarter|year|region|category|segment|department)\b",
        r"\btop \d+\b",
    ]

    # Phrases that mean "a picture printed in the document", not "draw me one".
    # Without this veto, "explain the graph in Figure 3" would be sent to the
    # SQL engine and come back empty.
    VIZ_DOC_REFS = [
        r"\bfigure\s*\d",
        r"\b(figure|graph|chart|table)\s+(on|in)\s+(page|section)\b",
        r"\bthe (graph|chart|figure) (shows|shown|above|below)\b",
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

        # Visualisation wording as a tie-breaker only. A viz word alone never
        # forces SQL, and a reference to a figure printed in the document vetoes.
        viz_matches = sum(1 for p in self.VIZ_SIGNALS if re.search(p, query_lower))
        refers_to_printed_figure = any(re.search(p, query_lower) for p in self.VIZ_DOC_REFS)

        if viz_matches and sql_matches >= 1 and rag_matches == 0 and not refers_to_printed_figure:
            logger.info(f"Heuristic: SQL (viz tie-break, viz={viz_matches}, sql={sql_matches})")
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
__TABLE_CONTEXT__

Example classifications:
- "How many rows in table X?" → SQL
- "Show revenue by segment" → SQL
- "Explain the methodology" → RAG
- "What do the revenue figures tell us about market trends?" → HYBRID
"""

        user_message = f"""Query: {query}

Classify this query as SQL, RAG, or HYBRID. Return JSON only."""

        response = None
        try:
            response = self.llm.chat(
                messages=[
                    # A plain replace, not .format(). The prompt contains a
                    # literal JSON example - {"route": "SQL"|...} - and
                    # str.format reads those braces as a field, raising
                    # KeyError('"route"'). Classification then silently fell
                    # back to RAG for every ambiguous query, which is how a
                    # question like "show revenue by segment" reached the text
                    # retriever instead of the SQL engine.
                    {
                        "role": "system",
                        "content": system_prompt.replace(
                            "__TABLE_CONTEXT__", table_context[:1000]
                        ),
                    },
                    {"role": "user", "content": user_message}
                ],
                temperature=0,
                response_format={"type": "json_object"},
                max_tokens=1500
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

        # `artifacts` is guaranteed on every route so a client never has to
        # test for the key. One line here covers all six early-return branches
        # across the three _execute_* methods.
        result.setdefault("artifacts", [])
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

        sql_result = self.sql_generator.generate(query, document_ids)

        if not sql_result.get("sql"):
            return {
                "answer": sql_result.get("explanation", "Could not generate SQL for this query."),
                "route": "SQL",
                "error": sql_result.get("error"),
                "artifacts": [],
                "query": query,
            }

        params = sql_result.get("params")
        exec_result = self.sql_executor.execute(sql_result["sql"], params)

        # One self-repair attempt. Free-form SQL only: a template that failed
        # is a bug here, not something the model can fix.
        if not exec_result["success"] and sql_result.get("pattern") == "custom":
            repaired = self.sql_generator.repair(
                sql_result["sql"], exec_result.get("error", ""), query
            )
            if repaired:
                logger.info("Retrying with repaired SQL")
                exec_result = self.sql_executor.execute(repaired)
                if exec_result["success"]:
                    sql_result["sql"] = repaired

        if not exec_result["success"]:
            return {
                "answer": f"SQL execution failed: {exec_result.get('error')}",
                "route": "SQL",
                "error": exec_result.get("error"),
                "sql": sql_result["sql"],
                "artifacts": [],
                "query": query,
            }

        # Build the artifact BEFORE the prose, so both describe the same
        # sanitised values rather than the chart saying 4.53 and the text 4.5.
        arts = artifacts.build_artifact(
            exec_result,
            sql=exec_result.get("sql_executed"),
            tables_used=sql_result.get("tables_used", []),
            title=query,
        )

        answer = self.sql_executor.format_results(
            exec_result, query, slots=sql_result.get("slots")
        )

        return {
            "answer": answer,
            "route": "SQL",
            "sql": sql_result["sql"],
            "sql_executed": exec_result.get("sql_executed"),
            "sql_explanation": sql_result.get("explanation"),
            "sql_pattern": sql_result.get("pattern"),
            "tables_used": sql_result.get("tables_used", []),
            "row_count": exec_result["row_count"],
            "artifacts": arts,
            "query": query,
        }

    def _execute_rag(
        self,
        query: str,
        document_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Execute RAG-only query."""
        logger.info("Executing RAG route")

        # document_ids is now honoured. It used to be accepted and ignored, so
        # "ask this document" silently searched the whole corpus.
        results = self.retriever.search(
            query, limit=5, score_threshold=0.5, document_ids=document_ids
        )

        # Retry once without the threshold before giving up. A 0.49 match used
        # to produce a flat "I couldn't find any relevant information", which
        # is a worse answer than a hedged one.
        if not results:
            results = self.retriever.search(
                query, limit=3, score_threshold=None, document_ids=document_ids
            )
            if results:
                logger.info("No result cleared the threshold; answering from weaker matches")

        if not results:
            return {
                "answer": "I couldn't find any relevant information to answer your question.",
                "route": "RAG",
                "sources": [],
                "artifacts": [],
                "query": query,
            }

        context = self.retriever.build_context(results)
        answer = self.llm.generate_answer(query, context)

        return {
            "answer": answer,
            "route": "RAG",
            # chunk_id, offsets and page are kept now: the retriever already
            # returned them and dropping them left citations unlocatable.
            "sources": [
                {
                    "score": r["score"],
                    "content": r["content"][:200] + "..." if len(r["content"]) > 200 else r["content"],
                    "heading_context": r.get("heading_context", ""),
                    "source_file": r.get("source_file", ""),
                    "chunk_id": r.get("chunk_id"),
                    "section": r.get("section"),
                    "start_offset": r.get("start_offset"),
                    "end_offset": r.get("end_offset"),
                    "page_number": r.get("page_number"),
                }
                for r in results
            ],
            "artifacts": [],
            "query": query,
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

        # Carry the SQL leg's artifact across, marked as merged prose.
        #
        # On this route the answer text is a model merge of two other model
        # outputs, three generative hops from the rows, while the artifact is
        # zero. The flag lets the UI say which surface is verified - the whole
        # point of building the classifier deterministically.
        arts = [dict(a) for a in sql_result.get("artifacts", [])]
        for art in arts:
            art["provenance"] = {
                **art["provenance"],
                "answer_is_llm_merged": True,
                "source_documents": [
                    {"source_file": s.get("source_file", ""), "score": s.get("score")}
                    for s in rag_result.get("sources", [])
                ],
            }

        return {
            "answer": merged_answer,
            "route": "HYBRID",
            "sql": sql_result.get("sql"),
            "sql_explanation": sql_result.get("sql_explanation"),
            # Both legs' failures used to be dropped entirely, so a hybrid
            # answer whose SQL leg failed reported nothing at all.
            "sql_error": sql_result.get("error"),
            "tables_used": sql_result.get("tables_used", []),
            "row_count": sql_result.get("row_count"),
            "sources": rag_result.get("sources", []),
            "artifacts": arts,
            "query": query,
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
