
# from flask import Flask, request, jsonify
# from typing import List, Dict, Any, Optional
# from qdrant_client import QdrantClient
# from qdrant_client.models import Filter, FieldCondition, MatchValue
# import boto3
# import json
# import os
# from dotenv import load_dotenv
# from llm_client import LLMClient

# load_dotenv()

# app = Flask(__name__)

# class BedrockEmbeddingClient:
#     def __init__(self):
#         """Initialize AWS Bedrock Embedding Client"""
#         AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
#         AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
#         ROLE_ARN = os.getenv("BEDROCK_ROLE_ARN")
#         REGION = os.getenv("AWS_REGION", "us-east-1")

#         # Step 1: Assume role
#         sts_client = boto3.client(
#             "sts",
#             region_name=REGION,
#             aws_access_key_id=AWS_ACCESS_KEY_ID,
#             aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
#             verify=False
#         )

#         assumed_role = sts_client.assume_role(
#             RoleArn=ROLE_ARN,
#             RoleSessionName="BedrockSession"
#         )
#         creds = assumed_role["Credentials"]

#         # Step 2: Bedrock runtime client
#         self.bedrock_client = boto3.client(
#             "bedrock-runtime",
#             region_name=REGION,
#             aws_access_key_id=creds["AccessKeyId"],
#             aws_secret_access_key=creds["SecretAccessKey"],
#             aws_session_token=creds["SessionToken"],
#             verify=False
#         )

#     def embed_query(self, text: str) -> List[float]:
#         """Generate embeddings using Amazon Titan Embeddings"""
#         response = self.bedrock_client.invoke_model(
#             modelId="amazon.titan-embed-text-v2:0",
#             contentType="application/json",
#             accept="application/json",
#             body=json.dumps({"inputText": text})
#         )
#         result = json.loads(response["body"].read())
#         return result["embedding"]


# class DocumentRAG:
#     def __init__(self):
#         self.embedding_model = BedrockEmbeddingClient()
#         self.qdrant_client = QdrantClient(host="localhost", port=6333)
#         self.llm_client = LLMClient()
    
#     def generate_query_embedding(self, query: str) -> List[float]:
#         """Generate embedding for search query using Bedrock"""
#         return self.embedding_model.embed_query(query)
    
#     def build_filter(self, project_id: str = None, project_no: str = None, 
#                     project_manager_name: str = None) -> Optional[Filter]:
#         """Build Qdrant filter based on metadata"""
#         conditions = []
        
#         if project_id:
#             conditions.append(
#                 FieldCondition(key="project_id", match=MatchValue(value=project_id))
#             )
        
#         if project_no:
#             conditions.append(
#                 FieldCondition(key="project_no", match=MatchValue(value=project_no))
#             )
        
#         if project_manager_name:
#             conditions.append(
#                 FieldCondition(key="project_manager_name", match=MatchValue(value=project_manager_name))
#             )
        
#         if conditions:
#             return Filter(must=conditions)
        
#         return None
    
#     def search_documents(self, query: str, collection_name: str, limit: int = 5, score_threshold: float = 0.5,
#                         project_id: str = None, project_no: str = None, 
#                         project_manager_name: str = None) -> List[Dict[str, Any]]:
#         """Search for relevant documents using vector similarity"""
#         query_embedding = self.generate_query_embedding(query)
#         if not query_embedding:
#             return []
        
#         search_filter = self.build_filter(project_id, project_no, project_manager_name)
        
#         search_results = self.qdrant_client.query_points(
#             collection_name=collection_name,
#             query=query_embedding,
#             limit=limit,
#             score_threshold=score_threshold,
#             query_filter=search_filter
#         ).points
        
#         results = []
#         for result in search_results:
#             results.append({
#                 "id": result.id,
#                 "score": result.score,
#                 "file_name": result.payload.get("file_name"),
#                 "file_path": result.payload.get("file_path"),
#                 "content_preview": result.payload.get("content", "")[:500] + "...",
#                 "project_id": result.payload.get("project_id"),
#                 "project_no": result.payload.get("project_no"),
#                 "project_manager_name": result.payload.get("project_manager_name"),
#                 "docs_summit_date": result.payload.get("docs_summit_date"),
#                 "start_date": result.payload.get("start_date"),
#                 "end_date": result.payload.get("end_date"),
#                 "currency_code": result.payload.get("currency_code")
#             })
        
#         return results
    
#     def generate_rag_response(self, query: str, collection_name: str, limit: int = 3, score_threshold: float = 0.7,
#                              project_id: str = None, project_no: str = None, 
#                              project_manager_name: str = None) -> Dict[str, Any]:
#         """Generate RAG response using retrieved documents"""
#         search_results = self.search_documents(
#             query, collection_name, limit, score_threshold, project_id, project_no, project_manager_name
#         )
        
#         context_parts = []
#         sources = []
        
#         for result in search_results:
#             full_result = self.qdrant_client.retrieve(
#                 collection_name=collection_name,
#                 ids=[result["id"]]
#             )
            
#             if full_result:
#                 payload = full_result[0].payload
#                 content = payload.get("content", "")
                
#                 metadata_info = []
#                 metadata_info.append(f"File: {payload.get('file_name', 'Unknown')}")
#                 if payload.get('project_id'):
#                     metadata_info.append(f"Project ID: {payload.get('project_id')}")
#                 if payload.get('project_no'):
#                     metadata_info.append(f"Project Number: {payload.get('project_no')}")
#                 if payload.get('project_manager_name'):
#                     metadata_info.append(f"Project Manager: {payload.get('project_manager_name')}")
#                 if payload.get('start_date'):
#                     metadata_info.append(f"Start Date: {payload.get('start_date')}")
#                 if payload.get('end_date'):
#                     metadata_info.append(f"End Date: {payload.get('end_date')}")
#                 if payload.get('currency_code'):
#                     metadata_info.append(f"Currency: {payload.get('currency_code')}")
#                 if payload.get('docs_summit_date'):
#                     metadata_info.append(f"Document Date: {payload.get('docs_summit_date')}")
                
#                 metadata_str = " | ".join(metadata_info)
#                 context_parts.append(f"Document Metadata: {metadata_str}\n\nDocument Content:\n{content}")
                
#                 sources.append({
#                     "file_name": result["file_name"],
#                     "score": result["score"],
#                     "project_id": result["project_id"],
#                     "project_no": result["project_no"]
#                 })
        
#         context = "\n\n---\n\n".join(context_parts)
#         answer = self.llm_client.generate_response(query, context)
        
#         return {
#             "answer": answer,
#             "sources": sources,
#             "query": query
#         }


# # Initialize RAG system
# rag_system = DocumentRAG()

# @app.route("/")
# def root():
#     return jsonify({"message": "Document RAG API with Bedrock embeddings is running"})

# @app.route("/rag", methods=["POST"])
# def rag_query():
#     data = request.get_json()
#     user_query = data.get("user_query")
#     collection_name = data.get("collection_name")

#     if not user_query:
#         return jsonify({"error": "user_query is required"}), 400
#     if not collection_name:
#         return jsonify({"error": "collection_name is required"}), 400

#     route = rag_system.llm_client.classify_query(user_query)
#     print(f"Routing decision: {route}")

#     if route == "SQL":
#         sql_response = rag_system.llm_client.run_sql_query(user_query)
#         return jsonify({
#             "answer": sql_response,
#             "query": user_query,
#             "sources": []
#         })
#     else:
#         rag_response = rag_system.generate_rag_response(
#             query=user_query,
#             collection_name=collection_name,
#             limit=3,
#             score_threshold=0.3
#         )
#         return jsonify(rag_response)


# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=8000, debug=True)

from flask import Flask, request, jsonify
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
import boto3
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

# -------------------------- Bedrock Embedding --------------------------

class BedrockEmbeddingClient:
    def __init__(self):
        """Initialize AWS Bedrock Embedding Client"""
        AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
        AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
        ROLE_ARN = os.getenv("BEDROCK_ROLE_ARN")
        REGION = os.getenv("AWS_REGION", "us-east-1")

        # Step 1: Assume role
        sts_client = boto3.client(
            "sts",
            region_name=REGION,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            verify=False
        )

        assumed_role = sts_client.assume_role(
            RoleArn=ROLE_ARN,
            RoleSessionName="BedrockSession"
        )
        creds = assumed_role["Credentials"]

        # Step 2: Bedrock runtime client
        self.bedrock_client = boto3.client(
            "bedrock-runtime",
            region_name=REGION,
            aws_access_key_id=creds["AccessKeyId"],
            aws_secret_access_key=creds["SecretAccessKey"],
            aws_session_token=creds["SessionToken"],
            verify=False
        )

    def embed_query(self, text: str) -> List[float]:
        """Generate embeddings using Amazon Titan Embeddings"""
        response = self.bedrock_client.invoke_model(
            modelId="amazon.titan-embed-text-v2:0",
            contentType="application/json",
            accept="application/json",
            body=json.dumps({"inputText": text})
        )
        result = json.loads(response["body"].read())
        return result["embedding"]


# -------------------------- RAG Orchestrator --------------------------

class DocumentRAG:
    def __init__(self):
        self.embedding_model = BedrockEmbeddingClient()
        self.qdrant_client = QdrantClient(host="localhost", port=6333)
        self.llm_client = LLMClient()

    def generate_query_embedding(self, query: str) -> List[float]:
        return self.embedding_model.embed_query(query)

    def build_filter(
        self,
        project_id: str = None,
        project_no: str = None,
        project_manager_name: str = None
    ) -> Optional[Filter]:
        conditions = []

        if project_id:
            conditions.append(FieldCondition(key="project_id", match=MatchValue(value=project_id)))

        if project_no:
            conditions.append(FieldCondition(key="project_no", match=MatchValue(value=project_no)))

        if project_manager_name:
            conditions.append(FieldCondition(key="project_manager_name", match=MatchValue(value=project_manager_name)))

        return Filter(must=conditions) if conditions else None

    def search_documents(
        self,
        query: str,
        collection_name: str,
        limit: int = 5,
        score_threshold: float = 0.5,
        project_id: str = None,
        project_no: str = None,
        project_manager_name: str = None
    ) -> List[Dict[str, Any]]:
        query_embedding = self.generate_query_embedding(query)
        if not query_embedding:
            return []

        search_filter = self.build_filter(project_id, project_no, project_manager_name)

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
                "project_id": result.payload.get("project_id"),
                "project_no": result.payload.get("project_no"),
                "project_manager_name": result.payload.get("project_manager_name"),
                "docs_summit_date": result.payload.get("docs_summit_date"),
                "start_date": result.payload.get("start_date"),
                "end_date": result.payload.get("end_date"),
                "currency_code": result.payload.get("currency_code")
            })

        return results

    def generate_rag_response(
        self,
        query: str,
        collection_name: str,
        limit: int = 3,
        score_threshold: float = 0.7,
        project_id: str = None,
        project_no: str = None,
        project_manager_name: str = None,
        memory_msgs: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:

        search_results = self.search_documents(
            query, collection_name, limit, score_threshold, project_id, project_no, project_manager_name
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

                metadata_info = []
                metadata_info.append(f"File: {payload.get('file_name', 'Unknown')}")
                if payload.get('project_id'):
                    metadata_info.append(f"Project ID: {payload.get('project_id')}")
                if payload.get('project_no'):
                    metadata_info.append(f"Project Number: {payload.get('project_no')}")
                if payload.get('project_manager_name'):
                    metadata_info.append(f"Project Manager: {payload.get('project_manager_name')}")
                if payload.get('start_date'):
                    metadata_info.append(f"Start Date: {payload.get('start_date')}")
                if payload.get('end_date'):
                    metadata_info.append(f"End Date: {payload.get('end_date')}")
                if payload.get('currency_code'):
                    metadata_info.append(f"Currency: {payload.get('currency_code')}")
                if payload.get('docs_summit_date'):
                    metadata_info.append(f"Document Date: {payload.get('docs_summit_date')}")

                metadata_str = " | ".join(metadata_info)
                context_parts.append(f"Document Metadata: {metadata_str}\n\nDocument Content:\n{content}")

                sources.append({
                    "file_name": result["file_name"],
                    # "score": result["score"],
                    "project_id": result["project_id"],
                    "project_no": result["project_no"]
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
    return jsonify({"message": "Document RAG API with Bedrock embeddings is running"})

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
