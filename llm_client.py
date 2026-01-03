import os
import re
import json
import requests
import psycopg2
from psycopg2 import errors as pg_errors
from typing import Dict, Any, Optional, Tuple, List
from dotenv import load_dotenv

load_dotenv()


class LLMClient:

    # -------------------------- Init & Groq Setup --------------------------

    def __init__(self):
        # --- Groq API configuration ---
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable is required")
        
        self.groq_url = os.getenv("GROQ_API_URL", "https://api.groq.com/openai/v1/chat/completions")
        self.groq_model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

        # --- Database connection ---
        self.conn = psycopg2.connect(
            dbname=os.getenv("PGDATABASE", "postgres"),
            user=os.getenv("PGUSER", "postgres"),
            password=os.getenv("PGPASSWORD", "Vivek@7566"),
            host=os.getenv("PGHOST", "localhost"),
            port=int(os.getenv("PGPORT", "5432")),
        )
        self.conn.autocommit = True

        # Table name (override with env if needed)
        self.table_name = os.getenv("PGTABLE", "coupa_contracts")

        # Live schema
        live_cols = self._load_table_schema(self.table_name)
        self.columns = [c.lower() for c in live_cols]
        self.columns_set = set(self.columns)

        # Synonyms that only map to actually-existing columns
        self.synonyms = self._build_synonyms()

        # Memory limits for prompt injection
        self.memory_turns = int(os.getenv("MEMORY_MAX_TURNS", "6"))
        self.memory_chars = int(os.getenv("MEMORY_MAX_CHARS", "2000"))

    # -------------------------- Groq Helpers --------------------------

    def _groq_chat(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = 500,
        temperature: float = 0.1,
    ) -> str:
        """
        Call Groq API with OpenAI-style messages (supports 'system'/'user'/'assistant').
        Returns assistant text.
        """
        # Groq uses standard OpenAI format - messages can be passed directly
        payload = {
            "model": self.groq_model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        
        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json",
        }
        
        try:
            response = requests.post(
                self.groq_url,
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
        
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Groq API request failed: {e}")
        except (KeyError, IndexError) as e:
            raise ValueError(f"Unexpected response format from Groq API: {e}")

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        if not text:
            return text
        fenced = re.match(r"^\s*```(?:\w+)?\s*(.*?)\s*```\s*$", text, re.DOTALL)
        if fenced:
            return fenced.group(1).strip()
        return re.sub(r"```(?:\w+)?\s*|\s*```", "", text).strip()

    # -------------------------- Utils & Schema --------------------------

    def _load_table_schema(self, table_name: str) -> List[str]:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = %s
                ORDER BY ordinal_position;
                """,
                (table_name,),
            )
            cols = [r[0] for r in cur.fetchall()]
        return cols

    @staticmethod
    def _norm(s: str) -> str:
        return re.sub(r"\s+", " ", s.strip().lower())

    def _build_synonyms(self) -> Dict[str, str]:
        raw = {
            # project manager
            "project manager": "project_manager_name",
            "manager": "project_manager_name",
            "pm name": "project_manager_name",

            # ids / numbers
            "project id": "project_id",
            "id": "project_id",
            "project number": "project_no",
            "project no": "project_no",
            "number": "project_no",

            # dates
            "start date": "start_date",
            "end date": "end_date",

            # money / amounts
            "total cost": "total_cost",
            "contract value": "total_cost",
            "sow amount": "total_cost",
            "fees": "total_cost",
            "fee amount": "total_cost",
            "amount": "total_cost",
            "cost": "total_cost",

            # currency
            "currency": "currency_code",
            "currency code": "currency_code",
        }

        valid: Dict[str, str] = {}
        for k, v in raw.items():
            if v in self.columns_set:
                valid[self._norm(k)] = v
        return valid

    def _detect_referenced_columns(self, text: str) -> Tuple[set, set]:
        q = self._norm(text)

        found_cols = set()
        unknown_mentions = set()

        # 1) Multi-word phrases / synonyms
        for phrase, col in self.synonyms.items():
            if re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", q):
                found_cols.add(col)

        # 2) Raw identifier-style mentions like "project_id", "total_cost"
        ident_tokens = set(re.findall(r"\b[a-z_][a-z0-9_]*\b", q))
        underscore_like = {t for t in ident_tokens if "_" in t}
        for token in underscore_like:
            if token in self.columns_set:
                found_cols.add(token)
            else:
                unknown_mentions.add(token)

        return found_cols, unknown_mentions

    # -------------------------- Memory Helpers --------------------------

    def _format_memory_for_prompt(
        self, memory: Optional[List[Dict[str, str]]]
    ) -> str:
        """
        Flatten recent memory (user/assistant turns) into a compact text block
        to help resolve pronouns/follow-ups without blowing context.
        """
        if not memory:
            return ""
        # Keep the last self.memory_turns*2 messages, trim each to memory_chars
        recent = memory[-self.memory_turns * 2 :]
        lines = []
        for m in recent:
            role = m.get("role", "user")
            content = str(m.get("content", ""))[: self.memory_chars]
            # Only keep non-empty text
            if content.strip():
                lines.append(f"{role.title()}: {content}")
        return "\n".join(lines)

    def rewrite_followup(self, query: str, memory: Optional[List[Dict[str, str]]]) -> str:
        """
        Turn a follow-up question into a self-contained query using recent memory.
        Keeps it short and faithful; no extra assumptions.
        """
        try:
            if not memory:
                return query

            mem_text = self._format_memory_for_prompt(memory)
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You rewrite follow-up questions into a single, self-contained question.\n"
                        "- Keep meaning intact.\n"
                        "- Include needed entities from history.\n"
                        "- Do NOT add new information.\n"
                        "- Return only the rewritten question."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Conversation (recent):\n{mem_text}\n\n"
                        f"Follow-up from user:\n{query}\n\n"
                        "Rewrite now:"
                    ),
                },
            ]
            out = self._groq_chat(messages, max_tokens=120, temperature=0.0)
            return self._strip_code_fences(out) or query
        except Exception:
            return query

    # -------------------------- Routing (SQL vs RAG) --------------------------

    def classify_query(self, query: str, memory: Optional[List[Dict[str, str]]] = None) -> str:
        """
        Hybrid routing with memory awareness:
          1) Fast heuristics for obvious RAG
          2) Detect referenced fields from schema+synonyms
          3) Ask LLM for SQL vs RAG (include compact history)
          4) If LLM says SQL but query references any unknown field -> force RAG
        """
        # Heuristics for obvious RAG (explanations/definitions)
        rag_keywords = [
            "explain", "meaning", "summarize", "summary", "describe",
            "how to", "definition", "assumptions", "scope"
        ]
        q_norm = self._norm(query)
        if any(k in q_norm for k in rag_keywords):
            return "RAG"

        # Detect column mentions
        _, unknown_mentions = self._detect_referenced_columns(query)

        # LLM vote with brief memory context
        mem_text = self._format_memory_for_prompt(memory)
        user_prompt = f"""
You are a query router.
If the query is about counts, lists, filters, or structured fields in the table, classify as SQL.
If it's about explanations, summaries, or unstructured documents, classify as RAG.

Table: {self.table_name}
Available columns: {", ".join(sorted(self.columns))}

Recent conversation (short):
{mem_text or "[none]"}

User query:
{query}

Return only one word: SQL or RAG
""".strip()

        try:
            text = self._groq_chat(
                messages=[
                    {"role": "system", "content": "Respond with exactly 'SQL' or 'RAG'."},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=5,
                temperature=0.0,
            ).strip().upper().replace(".", "")
            llm_vote = "SQL" if text == "SQL" else "RAG"
        except Exception as e:
            print(f"Error classifying query: {e}")
            llm_vote = "RAG"

        if llm_vote == "SQL" and unknown_mentions:
            print(f"[Router] Unknown field(s) mentioned: {sorted(unknown_mentions)} -> RAG")
            return "RAG"

        return llm_vote

    # -------------------------- NL -> SQL + Execute --------------------------

    def run_sql_query(self, query: str, memory: Optional[List[Dict[str, str]]] = None) -> str:
        """
        Convert natural language to SQL and run against Postgres DB.
        Memory is used as soft context (e.g., pronouns, implicit filters),
        but we still constrain to explicit schema.
        """
        mem_text = self._format_memory_for_prompt(memory)
        sql_prompt = f"""
Convert the following natural language query into a single SQL statement.

Conversation context (recent, may contain implied filters):
{mem_text or "[none]"}

Rules:
- Use only this table: {self.table_name}
- Valid columns are: {", ".join(sorted(self.columns))}
- If no specific column is referenced, assume COUNT(*) or reasonable aggregates.
- NEVER use columns not listed above.
- Return only the SQL.

Query: {query}
""".strip()

        try:
            sql_text = self._groq_chat(
                messages=[
                    {"role": "system", "content": "You are an expert SQL generator."},
                    {"role": "user", "content": sql_prompt},
                ],
                max_tokens=300,
                temperature=0.0,
            )

            sql_query = self._strip_code_fences(sql_text)

            # Keep only the first semicolon-terminated statement if multiple appear
            semicolon_idx = sql_query.find(";")
            if semicolon_idx != -1:
                sql_query = sql_query[: semicolon_idx + 1].strip()

            print(f"Final SQL: {sql_query}")

            # with self.conn.cursor() as cur:
            #     cur.execute(sql_query)
            #     if cur.description:  # SELECT
            #         rows = cur.fetchall()
            #         colnames = [d[0] for d in cur.description]
            #         result = [dict(zip(colnames, row)) for row in rows]
            #     else:  # UPDATE/INSERT/DELETE
            #         result = f"Query executed successfully: {cur.rowcount} rows affected."

            # return json.dumps(result, indent=2, default=str)

            with self.conn.cursor() as cur:
                cur.execute(sql_query)
                if cur.description:  # SELECT
                    rows = cur.fetchall()
                    colnames = [d[0] for d in cur.description]
                    
                    # Format results in a clean, human-readable way
                    if len(rows) == 0:
                        result = "No results found."
                    elif len(rows) == 1 and len(colnames) == 1:
                        # Single value result (like COUNT)
                        result = f"Result: {rows[0][0]}"
                    elif len(rows) == 1:
                        # Single row with multiple columns
                        result = "\n".join([f"{col}: {val}" for col, val in zip(colnames, rows[0])])
                    else:
                        # Multiple rows - format as numbered list (cleaner than JSON)
                        lines = [f"Found {len(rows)} result(s):\n"]
                        for i, row in enumerate(rows, 1):
                            row_dict = dict(zip(colnames, row))
                            row_items = ", ".join([f"{k}: {v}" for k, v in row_dict.items()])
                            lines.append(f"{i}. {row_items}")
                        result = "\n".join(lines)
                else:  # UPDATE/INSERT/DELETE
                    result = f"Query executed successfully: {cur.rowcount} rows affected."

            return result
        
        except pg_errors.UndefinedColumn as e:
            print(f"[SQL] UndefinedColumn -> fallback to RAG: {e}")
            return self.generate_response(
                query,
                context="The SQL path failed due to an unknown column; answering via documents instead.",
                memory=memory,
            )
        except Exception as e:
            return f"Error generating/executing SQL: {e}"

    # -------------------------- RAG-style Answering --------------------------

    def generate_response(
        self, query: str, context: str, memory: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Generate response based on query + retrieved context + recent memory (for continuity).
        """
        mem_text = self._format_memory_for_prompt(memory)
        prompt = f"""
You are an expert assistant. Use only the information provided in the CONTEXT below to answer the QUESTION.
Use the RECENT CHAT HISTORY only to resolve references/pronouns; do not invent facts beyond CONTEXT.

RECENT CHAT HISTORY:
{mem_text or "[none]"}

CONTEXT:
{context}

QUESTION:
{query}

Instructions:
- Greet user if user query is a greeting.
- For specific questions, provide a clear, comprehensive answer.
- If context is insufficient, say: "The context does not provide enough information."
- Do not add external knowledge beyond CONTEXT.
""".strip()

        try:
            return self._groq_chat(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant for answering based on documents.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1000,
                temperature=0.3,
            )
        except Exception as e:
            return f"Error generating response: {e}"

    # -------------------------- Entry Point --------------------------

    def handle_query(
        self, query: str, context: Optional[str] = None, memory: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Route query with memory-aware processing.
        """
        route = self.classify_query(query, memory=memory)
        print(f"Routing decision: {route}")

        if route == "SQL":
            return self.run_sql_query(query, memory=memory)
        else:
            if context is None:
                context = "No context provided."
            return self.generate_response(query, context, memory=memory)