"""
Unified Groq LLM Module

Consolidates all Groq API interactions:
- GroqClient: Low-level API wrapper
- GroqLLM: Answer generation, query classification, follow-up rewriting
- GroqSchemaGenerator: SQL schema and description generation for table_parser
"""

import os
import re
import json
import logging
from typing import List, Dict, Optional

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class GroqClient:
    """Low-level Groq API wrapper for chat completions."""

    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY environment variable is required")

        self.api_url = os.getenv(
            "GROQ_API_URL", "https://api.groq.com/openai/v1/chat/completions"
        )
        self.model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        logger.info(f"Initialized GroqClient with model={self.model}")

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        response_format: Optional[Dict] = None,
        timeout: int = 60,
    ) -> str:
        """
        Send a chat completion request to Groq API.

        Args:
            messages: List of {role, content} message dicts
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens in response
            response_format: Optional format constraint (e.g. {"type": "json_object"})
            timeout: Request timeout in seconds

        Returns:
            Response content string
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if response_format:
            payload["response_format"] = response_format

        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            logger.error(f"Groq API request failed: {e}")
            raise
        except (KeyError, IndexError) as e:
            logger.error(f"Unexpected response format from Groq: {e}")
            raise


class GroqLLM(GroqClient):
    """Higher-level LLM operations: answer generation, classification, rewriting."""

    def generate_answer(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        """
        Generate an answer based on query and retrieved context.

        Args:
            query: User's question
            context: Retrieved context from vector search or SQL results
            system_prompt: Optional custom system prompt
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature

        Returns:
            Generated answer string
        """
        if system_prompt is None:
            system_prompt = (
                "You are a helpful assistant that answers questions based on "
                "the provided context in short.\n\n"
                "Rules:\n"
                "- Answer ONLY based on the provided context\n"
                "- If the context doesn't contain enough information, say so\n"
                "- Be concise but thorough\n"
                "- Cite specific parts of the context when relevant\n"
                "- If the question is unclear, ask for clarification"
            )

        user_message = (
            f"Context:\n{context}\n\n---\n\n"
            f"Question: {query}\n\n"
            f"Please answer the question based on the context provided above."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        try:
            return self.chat(
                messages, temperature=temperature, max_tokens=max_tokens
            )
        except Exception as e:
            return f"Error generating answer: {e}"


class GroqSchemaGenerator(GroqClient):
    """SQL schema and description generation for table ingestion."""

    def generate_schema(
        self,
        headers: List[str],
        sample_rows: List[List[str]],
        table_name: str,
    ) -> str:
        """
        Generate CREATE TABLE statement using Groq LLM.

        Args:
            headers: Column headers
            sample_rows: Sample data rows for type inference
            table_name: Target table name

        Returns:
            CREATE TABLE SQL statement
        """
        prompt = f"""You are a PostgreSQL Expert. Generate a CREATE TABLE statement.

Rules:
1. Table Name: {table_name}
2. Add 'id' SERIAL PRIMARY KEY as the FIRST column.
3. Analyze these headers: {headers}
4. Analyze these sample rows: {sample_rows[:3]}
5. Infer appropriate PostgreSQL types (TEXT, INTEGER, NUMERIC, BOOLEAN, DATE, etc.)
6. Sanitize column names: lowercase, underscores for spaces, remove special characters.
7. Output JSON ONLY with format: {{"sql": "CREATE TABLE..."}}

Example output:
{{"sql": "CREATE TABLE {table_name} (id SERIAL PRIMARY KEY, column_name TEXT, another_col INTEGER);"}}
"""

        try:
            content = self.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                response_format={"type": "json_object"},
            )
            result = json.loads(content)
            return result.get("sql", "")
        except Exception as e:
            logger.error(f"Schema generation failed: {e}")
            return self._fallback_schema(headers, table_name)

    def _fallback_schema(self, headers: List[str], table_name: str) -> str:
        """Generate fallback schema with all TEXT columns."""
        sanitized = [self._sanitize_column_name(h) for h in headers]
        columns = ["id SERIAL PRIMARY KEY"]
        columns.extend([f"{col} TEXT" for col in sanitized])
        return f"CREATE TABLE {table_name} ({', '.join(columns)});"

    def _sanitize_column_name(self, name: str) -> str:
        """Sanitize column name for PostgreSQL."""
        sanitized = re.sub(r"[^a-zA-Z0-9]", "_", name.lower())
        sanitized = re.sub(r"_+", "_", sanitized)
        sanitized = sanitized.strip("_")
        if sanitized and sanitized[0].isdigit():
            sanitized = "col_" + sanitized
        return sanitized or "column"

    def generate_semantic_description(
        self,
        headers: List[str],
        sample_rows: List[List[str]],
        max_length: int = 100,
    ) -> str:
        """
        Generate a concise semantic description of the table.

        Args:
            headers: Column headers
            sample_rows: Sample data rows
            max_length: Maximum characters for description

        Returns:
            Short semantic description string
        """
        prompt = f"""Analyze this table and provide a concise 1-sentence description (max {max_length} chars).

        Headers: {headers}
        Sample: {sample_rows[:2]}

        Output JSON: {{"description": "your description here"}}"""

        try:
            content = self.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=100,
                response_format={"type": "json_object"},
            )
            result = json.loads(content)
            description = result.get("description", "")[:max_length]
            return description if description else self._fallback_description(headers)
        except Exception as e:
            logger.warning(f"Semantic description generation failed: {e}")
            return self._fallback_description(headers)

    def _fallback_description(self, headers: List[str]) -> str:
        """Generate fallback description from headers."""
        if len(headers) <= 3:
            return f"Table with {', '.join(headers)} data"
        return f"Table with {len(headers)} columns including {', '.join(headers[:2])}..."
