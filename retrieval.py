"""
Retrieval and Answering Module

This module handles:
- Semantic search in Qdrant vector database
- Answer generation using Groq LLM (Llama 3.1)
"""

import os
import logging
from typing import List, Dict, Any, Optional

import requests
from dotenv import load_dotenv
from qdrant_client import QdrantClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


# =================== Jina Embeddings Client =================== #

class JinaEmbeddingClient:
    """Jina AI embeddings client for generating query vectors."""
    
    def __init__(self):
        self.api_key = os.getenv("JINA_API_KEY")
        if not self.api_key:
            raise ValueError("JINA_API_KEY environment variable is required")
        
        self.api_url = os.getenv("JINA_API_URL", "https://api.jina.ai/v1/embeddings")
        self.model = os.getenv("JINA_MODEL", "jina-embeddings-v3")
        self.dimensions = int(os.getenv("EMBEDDING_DIMENSIONS", "1024"))
        self.task = os.getenv("JINA_TASK", "text-matching")
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
    
    def embed(self, text: str) -> List[float]:
        """Generate embedding vector for query text."""
        payload = {
            "model": self.model,
            "task": self.task,
            "dimensions": self.dimensions,
            "input": [text[:12000]]  # Truncate if too long
        }
        
        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            data = response.json()
            return data["data"][0]["embedding"]
        except Exception as e:
            logger.error(f"Jina embedding failed: {e}")
            return []


# =================== Groq LLM Client =================== #

class GroqLLMClient:
    """Groq LLM client for generating answers using Llama 3.1."""
    
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY environment variable is required")
        
        self.api_url = os.getenv("GROQ_API_URL", "https://api.groq.com/openai/v1/chat/completions")
        self.model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        logger.info(f"Initialized GroqLLMClient with model={self.model}")
    
    def generate_answer(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        """
        Generate an answer based on the query and retrieved context.
        
        Args:
            query: User's question
            context: Retrieved context from vector search
            system_prompt: Optional custom system prompt
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (0-1)
        
        Returns:
            Generated answer string
        """
        if system_prompt is None:
            system_prompt = """You are a helpful assistant that answers questions based on the provided context in short.
            
Rules:
- Answer ONLY based on the provided context
- If the context doesn't contain enough information, say so
- Be concise but thorough
- Cite specific parts of the context when relevant
- If the question is unclear, ask for clarification"""
        
        user_message = f"""Context:
{context}

---

Question: {query}

Please answer the question based on the context provided above."""
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        
        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            logger.error(f"Groq API request failed: {e}")
            return f"Error generating answer: {e}"
        except (KeyError, IndexError) as e:
            logger.error(f"Unexpected response format from Groq: {e}")
            return "Error: Unexpected response format"


# =================== Retriever =================== #

class Retriever:
    """
    Retrieves relevant chunks from Qdrant and generates answers using Groq.
    """
    
    def __init__(self, collection_name: Optional[str] = None):
        # Qdrant setup
        qdrant_endpoint = os.getenv("QDRANT_CLUSTER_ENDPOINT")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        self.collection_name = collection_name or os.getenv("QDRANT_COLLECTION_NAME", "documents_collection")
        
        if not qdrant_endpoint or not qdrant_api_key:
            raise ValueError("QDRANT_CLUSTER_ENDPOINT and QDRANT_API_KEY must be set")
        
        self.qdrant_client = QdrantClient(
            url=qdrant_endpoint,
            api_key=qdrant_api_key,
        )
        
        # Initialize clients
        self.embedder = JinaEmbeddingClient()
        self.llm = GroqLLMClient()
        
        logger.info(f"Initialized Retriever with collection={self.collection_name}")
    
    def search(
        self,
        query: str,
        limit: int = 5,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks in Qdrant.
        
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
        
        try:
            results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold,
            )
            
            return [
                {
                    "id": hit.id,
                    "score": hit.score,
                    "content": hit.payload.get("content", ""),
                    "source_file": hit.payload.get("source_file", ""),
                    "heading_context": hit.payload.get("heading_context", ""),
                    "section": hit.payload.get("section", ""),
                    "chunk_id": hit.payload.get("chunk_id", 0),
                }
                for hit in results
            ]
        except Exception as e:
            logger.error(f"Qdrant search failed: {e}")
            return []
    
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
        
        # ...existing code...

        if return_sources:
            response["sources"] = [
                {
                    "score": r["score"],
                    "content": r["content"],  # Full content instead of preview
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
    print("RAG Q&A System with Groq (Llama 3.1)")
    print("=" * 60)
    print("\nType your questions and press Enter.")
    print("Type 'quit' or 'exit' to stop.\n")
    
    try:
        retriever = Retriever()
    except ValueError as e:
        print(f"❌ Configuration Error: {e}")
        print("Make sure all required environment variables are set.")
        return
    
    while True:
        try:
            query = input("\n🔍 Question: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ["quit", "exit", "q"]:
                print("\nGoodbye! 👋")
                break
            
            print("\n⏳ Searching and generating answer...")
            
            result = retriever.answer(query, limit=3, score_threshold=0.5)
            
            print("\n" + "=" * 60)
            print("📝 Answer:")
            print("=" * 60)
            print(result["answer"])
            
            if result.get("sources"):
                print("\n" + "-" * 60)
                print(f"📚 Sources Used ({len(result['sources'])} chunks):")
                print("-" * 60)
                for i, source in enumerate(result["sources"], 1):
                    print(f"\n[{i}] Score: {source['score']:.4f}")
                    if source.get("heading_context"):
                        print(f"    Section: {source['heading_context']}")
                    if source.get("source_file"):
                        print(f"    File: {source['source_file']}")
            
        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye! 👋")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

# ...existing code...

def test_qdrant_connection():
    """Test Qdrant connection and list collections."""
    print("\n" + "=" * 50)
    print("🧪 Testing Qdrant Connection")
    print("=" * 50)
    
    endpoint = os.getenv("QDRANT_CLUSTER_ENDPOINT")
    api_key = os.getenv("QDRANT_API_KEY")
    
    # Debug: Print what we're reading
    print(f"   Endpoint from .env: {endpoint}")
    print(f"   API Key present: {bool(api_key)}")
    print(f"   API Key length: {len(api_key) if api_key else 0}")
    
    if not endpoint or not api_key:
        print("❌ QDRANT_CLUSTER_ENDPOINT or QDRANT_API_KEY not set")
        return False
    
    print(f"   Attempting connection to: {endpoint}")
    
    try:
        client = QdrantClient(url=endpoint, api_key=api_key)
        
        # Try to ping the server
        print("   Testing connection...")
        collections = client.get_collections()
        
        print(f"   ✓ Connection successful!")
        print(f"   Collections found: {len(collections.collections)}")
        for col in collections.collections:
            print(f"     - {col.name} ({col.points_count} points)")
        
        print("✅ Qdrant connection successful!")
        return client
    except Exception as e:
        print(f"❌ Qdrant test failed!")
        print(f"   Error type: {type(e).__name__}")
        print(f"   Error message: {str(e)}")
        
        # Additional debugging
        if "401" in str(e) or "Unauthorized" in str(e):
            print("\n   💡 Possible fixes:")
            print("      - Check if QDRANT_API_KEY is correct")
            print("      - Regenerate API key in Qdrant Cloud dashboard")
        elif "timeout" in str(e).lower():
            print("\n   💡 Possible fixes:")
            print("      - Check your internet connection")
            print("      - Verify cluster is running in Qdrant Cloud")
        elif "404" in str(e) or "not found" in str(e).lower():
            print("\n   💡 Possible fixes:")
            print("      - Check if QDRANT_CLUSTER_ENDPOINT URL is correct")
            print("      - Ensure cluster exists in Qdrant Cloud")
        
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
            print("📝 Answer:")
            print("=" * 60)
            print(result["answer"])
            
            if result.get("sources"):
                print(f"\n📚 Sources Used ({len(result['sources'])} chunks):")
                for i, source in enumerate(result["sources"], 1):
                    print(f"  [{i}] {source.get('heading_context', 'N/A')} (score: {source['score']:.4f})")
                    
        except Exception as e:
            print(f"❌ Error: {e}")
    else:
        # Interactive mode
        interactive_mode()


if __name__ == "__main__":
    main()
