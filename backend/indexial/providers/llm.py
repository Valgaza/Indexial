"""
Unified Groq LLM Module

Consolidates all Groq API interactions:
- GroqClient: Low-level API wrapper
- GroqLLM: Answer generation, query classification, follow-up rewriting
- GroqSchemaGenerator: SQL schema and description generation for table_parser
"""

import re
import json
import time
import random
import logging
from typing import List, Dict, Optional

import requests

from indexial.core import config

logger = logging.getLogger(__name__)


class GroqRateLimit(RuntimeError):
    """Raised when Groq's rate limit could not be waited out."""


class GroqClient:
    """Low-level Groq API wrapper for chat completions."""

    def __init__(self, model: Optional[str] = None):
        self.api_key = config.GROQ_API_KEY
        if not self.api_key:
            raise ValueError("GROQ_API_KEY is required (set it in the repo-root .env)")

        self.api_url = config.GROQ_API_URL
        self.model = model or config.GROQ_MODEL
        self.max_retries = config.GROQ_MAX_RETRIES

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        logger.info(f"Initialized GroqClient with model={self.model}")

    @staticmethod
    def _retry_after(response: requests.Response, attempt: int) -> float:
        """
        How long to wait before retrying.

        Groq sends Retry-After on 429. When it is absent, back off
        exponentially with jitter so concurrent callers do not resynchronise.
        """
        header = response.headers.get("retry-after") if response is not None else None
        if header:
            try:
                return min(float(header), 60.0)
            except ValueError:
                pass
        return min(2**attempt + random.uniform(0, 1), 60.0)

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

        Retries on 429 and 5xx. The free tier allows 8000 tokens per minute
        across every model, and a single HYBRID query makes four calls, so
        rate limiting is a normal operating condition here rather than an edge
        case — an unhandled 429 would surface to the user as a failed query.

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

        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=timeout,
                )

                if response.status_code == 429 or response.status_code >= 500:
                    wait = self._retry_after(response, attempt)
                    if attempt == self.max_retries - 1:
                        break
                    logger.warning(
                        f"Groq {response.status_code}; retrying in {wait:.1f}s "
                        f"(attempt {attempt + 1}/{self.max_retries})"
                    )
                    time.sleep(wait)
                    continue

                # gpt-oss is a reasoning model: it spends tokens thinking
                # before it emits content. When max_tokens is too small the
                # budget is gone before any JSON appears and Groq rejects the
                # empty generation with json_validate_failed. Retry once with
                # a bigger budget rather than surfacing it as a dead end.
                if response.status_code == 400 and attempt < self.max_retries - 1:
                    try:
                        code = response.json().get("error", {}).get("code")
                    except ValueError:
                        code = None
                    if code == "json_validate_failed":
                        payload["max_tokens"] = min(payload["max_tokens"] * 4, 8000)
                        logger.warning(
                            "Groq truncated its JSON; retrying with "
                            f"max_tokens={payload['max_tokens']}"
                        )
                        continue

                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]

            except requests.exceptions.Timeout as e:
                last_error = e
                if attempt == self.max_retries - 1:
                    break
                time.sleep(self._retry_after(None, attempt))
            except requests.exceptions.RequestException as e:
                # 4xx other than 429 will not succeed on retry. Log the body:
                # Groq puts the actual reason there, and without it a 400 is
                # indistinguishable from any other failure.
                detail = ""
                if e.response is not None:
                    try:
                        detail = json.dumps(e.response.json().get("error", {}))[:400]
                    except ValueError:
                        detail = e.response.text[:400]
                logger.error(f"Groq API request failed: {e} | {detail}")
                raise
            except (KeyError, IndexError) as e:
                logger.error(f"Unexpected response format from Groq: {e}")
                raise

        raise GroqRateLimit(
            f"Groq unavailable after {self.max_retries} attempts"
            + (f": {last_error}" if last_error else " (rate limited)")
        )


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

    def rewrite_followup(
        self,
        query: str,
        history: List[Dict[str, str]],
        max_tokens: int = 200,
    ) -> str:
        """
        Rewrite follow-up queries with context from conversation history.

        Resolves pronouns and references like:
        - "What about X?" -> "What is X in the context of [previous topic]?"
        - "Tell me more" -> "Tell me more about [previous topic]"
        - "How does it work?" -> "How does [previous subject] work?"

        Args:
            query: Current user query (may have pronouns/references)
            history: Conversation history [{role, content}, ...]
            max_tokens: Maximum tokens in response

        Returns:
            Rewritten standalone query
        """
        # If no history or query seems standalone, return as-is
        if not history or len(history) < 2:
            return query

        # Heuristic: Check if query has pronouns or short interrogatives
        standalone_signals = [
            r"\bwhat is\b",
            r"\bwho is\b",
            r"\bhow many\b",
            r"\bshow me\b",
            r"\blist\b",
            r"\bexplain\b.*\b(the|a)\b",
        ]

        followup_signals = [
            r"\bit\b",
            r"\bthat\b",
            r"\bthis\b",
            r"\bthey\b",
            r"\bthem\b",
            r"\btell me more\b",
            r"\bwhat about\b",
            r"\band\b",
            r"^(how|why|when|where)",
        ]

        query_lower = query.lower()

        # If query has strong standalone signals, don't rewrite
        import re
        has_standalone = any(re.search(p, query_lower) for p in standalone_signals)
        has_followup = any(re.search(p, query_lower) for p in followup_signals)

        if has_standalone and not has_followup:
            return query

        # Build conversation context (last few exchanges)
        context_parts = []
        for msg in history[-4:]:  # Last 2 exchanges
            role = msg.get("role", "").capitalize()
            content = msg.get("content", "")[:200]  # Truncate
            context_parts.append(f"{role}: {content}")

        conversation_context = "\n".join(context_parts)

        system_prompt = """You are a query rewriting assistant. Your task is to rewrite follow-up questions into standalone queries.

Rules:
1. Replace pronouns (it, that, this, they) with specific references from conversation history
2. Add necessary context from previous questions/answers
3. Keep the query concise and natural
4. If the query is already standalone, return it unchanged
5. Return ONLY the rewritten query, nothing else

Examples:
- "What about profit?" + [context: revenue discussed] -> "What is the profit for the same period?"
- "Tell me more" + [context: AI definition] -> "Tell me more about artificial intelligence"
- "How does it work?" + [context: neural networks] -> "How do neural networks work?"
"""

        user_message = f"""Conversation history:
{conversation_context}

Current query: {query}

Rewrite this query to be standalone and clear. If already standalone, return it unchanged."""

        try:
            rewritten = self.chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.3,
                max_tokens=max_tokens,
            )

            # Clean up the response (remove quotes, extra text)
            rewritten = rewritten.strip().strip('"').strip("'")

            # If rewritten is too long or seems wrong, return original
            if len(rewritten) > len(query) * 3 or len(rewritten) < 3:
                return query

            return rewritten

        except Exception as e:
            logger.error(f"Follow-up rewrite failed: {e}")
            return query


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
