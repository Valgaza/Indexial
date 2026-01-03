from flask import Flask, request, jsonify
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
import requests
import json
import os
import threading
from collections import defaultdict, deque
from dotenv import load_dotenv
from llm_client import LLMClient  # <— note: plural to match filename

load_dotenv()

app = Flask(__name__)

# -------------------------- Simple Session Memory --------------------------

class MemoryManager:
    """
    Thread-safe per-session conversational buffer memory.
    Stores the last N turns (user+assistant) as [{role, content}, ...].
    """
    def __init__(self, max_turns: int = None, max_msg_chars: int = None):
        self.max_turns = int(os.getenv("MEMORY_MAX_TURNS", "6")) if max_turns is None else max_turns
        self.max_msg_chars = int(os.getenv("MEMORY_MAX_CHARS", "2000")) if max_msg_chars is None else max_msg_chars
        self._store: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.max_turns * 2))
        self._lock = threading.Lock()

    def add(self, session_id: str, role: str, content: str) -> None:
        if not session_id:
            return
        content = (content or "")
        content = content if len(content) <= self.max_msg_chars else content[: self.max_msg_chars]
        with self._lock:
            self._store[session_id].append({"role": role, "content": content})

    def add_exchange(self, session_id: str, user_msg: str, assistant_msg: str) -> None:
        self.add(session_id, "user", user_msg)
        self.add(session_id, "assistant", assistant_msg)

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        with self._lock:
            return list(self._store.get(session_id, deque()))

    def clear(self, session_id: str) -> None:
        with self._lock:
            if session_id in self._store:
                del self._store[session_id]


memory = MemoryManager()

# -------------------------- Jina Embedding (Open Source) --------------------------

class JinaEmbeddingClient:
    """Jina AI embeddings client for generating 1024-dimensional vectors."""
    
    def __init__(self):
        load_dotenv()
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
        """Generate embedding vector for the given text."""
        payload = {
            "model": self.model,
            "task": self.task,
            "dimensions": self.dimensions,
            "input": text
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
            
            # Extract embedding from response
            embedding = data["data"][0]["embedding"]
            
            if len(embedding) != self.dimensions:
                raise ValueError(f"Expected {self.dimensions} dimensions, got {len(embedding)}")
            
            return embedding
        
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Jina API request failed: {e}")
        except (KeyError, IndexError) as e:
            raise ValueError(f"Unexpected response format from Jina API: {e}")
    
    def embed_query(self, text: str) -> List[float]:
        """Alias for embed() to maintain compatibility with existing code."""
        return self.embed(text)


# -------------------------- RAG Orchestrator --------------------------

class DocumentRAG:
    def __init__(self):
        self.embedding_model = JinaEmbeddingClient()
        self.qdrant_client = QdrantClient(host="localhost", port=6333)
        self.llm_client = LLMClient()

    def generate_query_embedding(self, query: str) -> List[float]:
        return self.embedding_model.embed_query(query)

    def build_filter(
        self,
        custom_filters: Optional[Dict[str, str]] = None
    ) -> Optional[Filter]:
        """Build Qdrant filter from custom key-value pairs if provided."""
        if not custom_filters:
            return None
        
        conditions = []
        for key, value in custom_filters.items():
            conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
        
        return Filter(must=conditions) if conditions else None

    def search_documents(
        self,
        query: str,
        collection_name: str,
        limit: int = 5,
        score_threshold: float = 0.5,
        custom_filters: Optional[Dict[str, str]] = None
    ) -> List[Dict[str, Any]]:
        """Search documents using query embedding with optional custom filters."""
        query_embedding = self.generate_query_embedding(query)
        if not query_embedding:
            return []

        search_filter = self.build_filter(custom_filters)

        search_results = self.qdrant_client.query_points(
            collection_name=collection_name,
            query=query_embedding,
            limit=limit,
            score_threshold=score_threshold,
            query_filter=search_filter
        ).points

        results = []
        for result in search_results:
            results.append({
                "id": result.id,
                "score": result.score,
                "file_name": result.payload.get("file_name"),
                "file_path": result.payload.get("file_path"),
                "content_preview": (result.payload.get("content", "")[:500] + "..."),
                "heading": result.payload.get("heading"),
                "subheading": result.payload.get("subheading"),
            })

        return results

    def generate_rag_response(
        self,
        query: str,
        collection_name: str,
        limit: int = 3,
        score_threshold: float = 0.7,
        custom_filters: Optional[Dict[str, str]] = None,
        memory_msgs: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """Generate RAG response with generic document context."""

        search_results = self.search_documents(
            query, collection_name, limit, score_threshold, custom_filters
        )

        context_parts = []
        sources = []

        for result in search_results:
            full_result = self.qdrant_client.retrieve(
                collection_name=collection_name,
                ids=[result["id"]]
            )

            if full_result:
                payload = full_result[0].payload
                content = payload.get("content", "")

                # Build metadata string from available fields
                metadata_info = []
                metadata_info.append(f"File: {payload.get('file_name', 'Unknown')}")
                if payload.get('heading'):
                    metadata_info.append(f"Heading: {payload.get('heading')}")
                if payload.get('subheading'):
                    metadata_info.append(f"Subheading: {payload.get('subheading')}")

                metadata_str = " | ".join(metadata_info)
                context_parts.append(f"Document Metadata: {metadata_str}\n\nDocument Content:\n{content}")

                sources.append({
                    "file_name": result["file_name"],
                    "heading": result.get("heading"),
                })

        context = "\n\n---\n\n".join(context_parts) if context_parts else "No context retrieved."
        answer = self.llm_client.generate_response(query, context, memory=memory_msgs)

        return {
            "answer": answer,
            "sources": sources,
            "query": query
        }


# Initialize RAG system
rag_system = DocumentRAG()

# -------------------------- Routes --------------------------

@app.route("/")
def root():
    return jsonify({"message": "Document RAG API with Jina embeddings is running"})

@app.route("/memory/clear", methods=["POST"])
def clear_memory():
    data = request.get_json() or {}
    session_id = data.get("session_id") or request.headers.get("X-Session-Id") or request.remote_addr
    memory.clear(session_id)
    return jsonify({"ok": True, "message": f"Memory cleared for session: {session_id}"}), 200

@app.route("/rag", methods=["POST"])
def rag_query():
    data = request.get_json() or {}
    user_query = data.get("user_query")
    collection_name = data.get("collection_name")
    session_id = data.get("session_id") or request.headers.get("X-Session-Id") or request.remote_addr

    if not user_query:
        return jsonify({"error": "user_query is required"}), 400
    if not collection_name:
        return jsonify({"error": "collection_name is required"}), 400

    # 1) Pull recent memory for this session
    memory_msgs = memory.get_history(session_id)

    # 2) Rewrite follow-up into stand-alone query (memory-aware)
    standalone_query = rag_system.llm_client.rewrite_followup(user_query, memory_msgs)
    if standalone_query != user_query:
        print(f"[Rewriter] '{user_query}' -> '{standalone_query}'")

    # 3) Route with memory
    route = rag_system.llm_client.classify_query(standalone_query, memory=memory_msgs)
    print(f"Routing decision: {route}")

    # 4) Execute branch
    if route == "SQL":
        sql_response = rag_system.llm_client.run_sql_query(standalone_query, memory=memory_msgs)
        # Save turn to memory
        memory.add_exchange(session_id, user_query, sql_response)
        return jsonify({
            "answer": sql_response,
            "query": standalone_query,
            "route": "SQL",
            "sources": []
        })
    else:
        rag_response = rag_system.generate_rag_response(
            query=standalone_query,
            collection_name=collection_name,
            limit=3,
            score_threshold=0.3,
            memory_msgs=memory_msgs
        )
        # Save turn to memory
        memory.add_exchange(session_id, user_query, rag_response.get("answer", ""))
        # Add route to response
        rag_response["route"] = "RAG"
        return jsonify(rag_response)


if __name__ == "__main__":
    # For local dev
    app.run(host="0.0.0.0", port=8000, debug=True)
