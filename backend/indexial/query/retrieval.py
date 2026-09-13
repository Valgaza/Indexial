"""
Retrieval and Answering Module

This module handles:
- Semantic search in Supabase PostgreSQL (pgvector)
- Answer generation using Groq LLM (Llama 3.1)
"""

import os
import logging
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv

from db import get_connection
from embeddings import JinaEmbeddingClient
from llm import GroqLLM

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


# =================== Retriever =================== #

class Retriever:
    """
    Retrieves relevant chunks from Supabase (pgvector) and generates answers using Groq.
    """

    def __init__(self, table_name: Optional[str] = None):
        self.table_name = table_name or os.getenv("VECTOR_TABLE_NAME", "document_chunks")

        # Initialize clients
        self.embedder = JinaEmbeddingClient()
        self.llm = GroqLLM()

        logger.info(f"Initialized Retriever with table={self.table_name}")

    def search(
        self,
        query: str,
        limit: int = 5,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks in Supabase using cosine similarity.

        Args:
            query: Search query
            limit: Maximum number of results
            score_threshold: Minimum similarity score

        Returns:
            List of matching chunks with scores
        """
        # Generate query embedding
        query_vector = self.embedder.embed(query)
        if not query_vector:
            logger.error("Failed to generate query embedding")
            return []

        with get_connection(readonly=True) as conn:
            cur = conn.cursor()
            try:
                if score_threshold is not None:
                    sql = f"""
                        SELECT id, source_file, chunk_id, content, start_offset, end_offset,
                               heading_context, section, processed_date, metadata,
                               1 - (embedding <=> %s::vector) as similarity
                        FROM {self.table_name}
                        WHERE 1 - (embedding <=> %s::vector) >= %s
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s
                    """
                    cur.execute(sql, (query_vector, query_vector, score_threshold, query_vector, limit))
                else:
                    sql = f"""
                        SELECT id, source_file, chunk_id, content, start_offset, end_offset,
                               heading_context, section, processed_date, metadata,
                               1 - (embedding <=> %s::vector) as similarity
                        FROM {self.table_name}
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s
                    """
                    cur.execute(sql, (query_vector, query_vector, limit))

                rows = cur.fetchall()

                results = []
                for row in rows:
                    results.append({
                        "id": row[0],
                        "source_file": row[1],
                        "chunk_id": row[2],
                        "content": row[3],
                        "start_offset": row[4],
                        "end_offset": row[5],
                        "heading_context": row[6],
                        "section": row[7],
                        "processed_date": row[8],
                        "metadata": row[9],
                        "score": row[10]
                    })

                return results

            except Exception as e:
                logger.error(f"Supabase search failed: {e}")
                return []
            finally:
                cur.close()

    def build_context(self, results: List[Dict[str, Any]]) -> str:
        """Build context string from search results."""
        if not results:
            return "No relevant context found."

        context_parts = []
        for i, result in enumerate(results, 1):
            header = f"[Source {i}]"
            if result.get("heading_context"):
                header += f" {result['heading_context']}"

            context_parts.append(f"{header}\n{result['content']}")

        return "\n\n---\n\n".join(context_parts)

    def answer(
        self,
        query: str,
        limit: int = 5,
        score_threshold: Optional[float] = 0.5,
        return_sources: bool = True,
    ) -> Dict[str, Any]:
        """
        Search for relevant chunks and generate an answer.

        Args:
            query: User's question
            limit: Number of chunks to retrieve
            score_threshold: Minimum similarity score
            return_sources: Whether to include source chunks in response

        Returns:
            Dictionary with answer and optionally sources
        """
        # Step 1: Search for relevant chunks
        results = self.search(query, limit=limit, score_threshold=score_threshold)

        if not results:
            return {
                "answer": "I couldn't find any relevant information to answer your question.",
                "query": query,
                "sources": [],
            }

        # Step 2: Build context from results
        context = self.build_context(results)

        # Step 3: Generate answer using Groq
        answer = self.llm.generate_answer(query, context)

        response = {
            "answer": answer,
            "query": query,
        }

        if return_sources:
            response["sources"] = [
                {
                    "score": r["score"],
                    "content": r["content"],
                    "heading_context": r.get("heading_context", ""),
                    "source_file": r.get("source_file", ""),
                }
                for r in results
            ]

        return response


# =================== Interactive CLI =================== #

def interactive_mode():
    """Run an interactive Q&A session."""
    print("=" * 60)
    print("RAG Q&A System with Supabase + Groq (Llama 3.1)")
    print("=" * 60)
    print("\nType your questions and press Enter.")
    print("Type 'quit' or 'exit' to stop.\n")

    try:
        retriever = Retriever()
    except ValueError as e:
        print(f"Configuration Error: {e}")
        print("Make sure all required environment variables are set.")
        return

    while True:
        try:
            query = input("\nQuestion: ").strip()

            if not query:
                continue

            if query.lower() in ["quit", "exit", "q"]:
                print("\nGoodbye!")
                break

            print("\nSearching and generating answer...")

            result = retriever.answer(query, limit=3, score_threshold=0.5)

            print("\n" + "=" * 60)
            print("Answer:")
            print("=" * 60)
            print(result["answer"])

            if result.get("sources"):
                print("\n" + "-" * 60)
                print(f"Sources Used ({len(result['sources'])} chunks):")
                print("-" * 60)
                for i, source in enumerate(result["sources"], 1):
                    print(f"\n[{i}] Score: {source['score']:.4f}")
                    if source.get("heading_context"):
                        print(f"    Section: {source['heading_context']}")
                    if source.get("source_file"):
                        print(f"    File: {source['source_file']}")

        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")


def test_supabase_connection():
    """Test Supabase connection and check vector table."""
    print("\n" + "=" * 50)
    print("Testing Supabase Connection")
    print("=" * 50)

    table_name = os.getenv("VECTOR_TABLE_NAME", "document_chunks")

    try:
        with get_connection(readonly=True) as conn:
            cur = conn.cursor()

            cur.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cur.fetchone()[0]

            print(f"   Connection successful!")
            print(f"   Table '{table_name}' has {count} chunks")

            cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
            version = cur.fetchone()
            if version:
                print(f"   pgvector extension version: {version[0]}")

            cur.close()

        print("Supabase connection successful!")
        return True
    except Exception as e:
        print(f"Supabase test failed!")
        print(f"   Error type: {type(e).__name__}")
        print(f"   Error message: {str(e)}")

        if "password" in str(e).lower() or "authentication" in str(e).lower():
            print("\n   Possible fixes:")
            print("      - Check if SUPABASE_DB_URL contains correct password")
            print("      - Verify database credentials in Supabase dashboard")
        elif "does not exist" in str(e).lower():
            print("\n   Possible fixes:")
            print(f"      - Table '{table_name}' may not exist yet")
            print("      - Run chunker.py first to create the table")

        return None


def main():
    """Main entry point."""
    import sys

    if len(sys.argv) > 1:
        # Single query mode
        query = " ".join(sys.argv[1:])

        try:
            retriever = Retriever()
            result = retriever.answer(query, limit=3)

            print("\n" + "=" * 60)
            print("Answer:")
            print("=" * 60)
            print(result["answer"])

            if result.get("sources"):
                print(f"\nSources Used ({len(result['sources'])} chunks):")
                for i, source in enumerate(result["sources"], 1):
                    print(f"  [{i}] {source.get('heading_context', 'N/A')} (score: {source['score']:.4f})")

        except Exception as e:
            print(f"Error: {e}")
    else:
        # Interactive mode
        interactive_mode()


if __name__ == "__main__":
    main()
