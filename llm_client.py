import os
import re
import json
import boto3
import psycopg2
from psycopg2 import errors as pg_errors
from typing import Dict, Any, Optional, Tuple, List
from dotenv import load_dotenv

load_dotenv()


class LLMClient:
    """
    LLM helper for:
      - Query routing (SQL vs RAG) with schema-aware validation and synonyms
      - NL -> SQL generation and execution (Postgres)
      - RAG-style response generation
      - Document-type classification (SOW / CHANGE_REQUEST / AMENDMENT / UNKNOWN)
      - Uses conversational buffer memory supplied in-call (list of {role, content})
    """

    # -------------------------- Init & Bedrock Setup --------------------------

    def __init__(self):
        # --- Bedrock credentials / client ---
        region = os.getenv("AWS_REGION", "us-east-1")
        role_arn = os.getenv("BEDROCK_ROLE_ARN")

        session_kwargs: Dict[str, Any] = {"region_name": region}

        # Use base keys if provided (optional; otherwise, let default AWS chain resolve)
        base_access = os.getenv("AWS_ACCESS_KEY_ID")
        base_secret = os.getenv("AWS_SECRET_ACCESS_KEY")
        base_token = os.getenv("AWS_SESSION_TOKEN")
        if base_access and base_secret:
            session_kwargs.update(
                {
                    "aws_access_key_id": base_access,
                    "aws_secret_access_key": base_secret,
                    **({"aws_session_token": base_token} if base_token else {}),
                }
            )

        if role_arn:
            # Assume a Bedrock-enabled role if specified
            sts_client = boto3.client("sts", **session_kwargs)
            assumed = sts_client.assume_role(
                RoleArn=role_arn, RoleSessionName="BedrockSession"
            )["Credentials"]
            self.bedrock_client = boto3.client(
                "bedrock-runtime",
                region_name=region,
                aws_access_key_id=assumed["AccessKeyId"],
                aws_secret_access_key=assumed["SecretAccessKey"],
                aws_session_token=assumed["SessionToken"],
                # Set verify=False if your corp proxy breaks SSL; otherwise prefer True
                verify=False,
            )
        else:
            # Fall back to default creds (env/EC2/Role/etc.)
            self.bedrock_client = boto3.client("bedrock-runtime", **session_kwargs)

        # Default Bedrock model
        self.model_id = os.getenv(
            "BEDROCK_MODEL_ID", "anthropic.claude-3-sonnet-20240229-v1:0"
        )

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

    # -------------------------- Bedrock Helpers --------------------------

    def _bedrock_chat(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = 500,
        temperature: float = 0.1,
    ) -> str:
        """
        Pass OpenAI-style 'messages' directly (supports 'system'/'user'/'assistant').
        Returns assistant text.
        """
        system_parts: List[str] = []
        convo: List[Dict[str, Any]] = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                system_parts.append(str(content))
            elif role in ("user", "assistant"):
                convo.append(
                    {"role": role, "content": [{"type": "text", "text": str(content)}]}
                )

        body: Dict[str, Any] = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": convo,
        }
        if system_parts:
            body["system"] = "\n\n".join(system_parts)

        response = self.bedrock_client.invoke_model(
            modelId=self.model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body),
        )
        payload = json.loads(response["body"].read())
        return payload["content"][0]["text"].strip()

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
            out = self._bedrock_chat(messages, max_tokens=120, temperature=0.0)
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
            text = self._bedrock_chat(
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
            sql_text = self._bedrock_chat(
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
            return self._bedrock_chat(
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

    # -------------------------- Doc-type Classification --------------------------

    def classify_document_type(self, text: str) -> Tuple[str, float, List[str]]:
        snippet = text[:24000]

        user_prompt = f"""
You are a legal-doc classifier. Read the document and decide if it is:
- "SOW" (Statement of Work: services/deliverables, Statement of Work, fees, term, assumptions, signatories)
- "CHANGE_REQUEST" (CR: changes to scope/cost/timing of an existing SOW; fields like Revised SOW Amount, Timing Impact)
- "AMENDMENT" (amendment to an agreement/SOW; language like "This Amendment is made by and between", "amends", "effective date", "amended contract value")
- "UNKNOWN" (if ambiguous)

Return ONLY a strict JSON object with keys:
{{
  "doc_type": "SOW" | "CHANGE_REQUEST" | "AMENDMENT" | "UNKNOWN",
  "confidence": 0.0-1.0,
  "reasons": ["short bullets"]
}}

Document (truncated):
\"\"\"{snippet}\"\"\"
""".strip()

        try:
            raw = self._bedrock_chat(
                messages=[
                    {
                        "role": "system",
                        "content": "You classify contract documents. Always answer with valid JSON.",
                    },
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=220,
                temperature=0.0,
            )

            raw = self._strip_code_fences(raw)
            data = json.loads(raw)

            doc_type = str(data.get("doc_type", "UNKNOWN")).upper()
            conf = float(data.get("confidence", 0.0))
            reasons = data.get("reasons", [])

            if doc_type not in {"SOW", "CHANGE_REQUEST", "AMENDMENT", "UNKNOWN"}:
                doc_type = "UNKNOWN"
            if not isinstance(reasons, list):
                reasons = [str(reasons)]
            return (doc_type, conf, reasons)

        except Exception as e:
            print(f"Error classify_document_type: {e}")
            return ("UNKNOWN", 0.0, [])


# import os
# import re
# import json
# import boto3
# import psycopg2
# from psycopg2 import errors as pg_errors
# from typing import Dict, Any, Optional, Tuple, List
# from dotenv import load_dotenv

# load_dotenv()


# class LLMClient:
#     """
#     LLM helper for:
#       - Query routing (SQL vs RAG) with schema-aware validation and synonyms
#       - NL -> SQL generation and execution (Postgres)
#       - RAG-style response generation
#       - Document-type classification (SOW / CHANGE_REQUEST / AMENDMENT / UNKNOWN)
#     """

#     # -------------------------- Init & Bedrock Setup --------------------------

#     def __init__(self):
#         # --- Bedrock credentials / client ---
#         region = os.getenv("AWS_REGION", "us-east-1")
#         role_arn = os.getenv("BEDROCK_ROLE_ARN")

#         session_kwargs: Dict[str, Any] = {"region_name": region}

#         # Use base keys if provided (optional; otherwise, let default AWS chain resolve)
#         base_access = os.getenv("AWS_ACCESS_KEY_ID")
#         base_secret = os.getenv("AWS_SECRET_ACCESS_KEY")
#         base_token = os.getenv("AWS_SESSION_TOKEN")
#         if base_access and base_secret:
#             session_kwargs.update(
#                 {
#                     "aws_access_key_id": base_access,
#                     "aws_secret_access_key": base_secret,
#                     **({"aws_session_token": base_token} if base_token else {}),
#                 }
#             )

#         if role_arn:
#             # Assume a Bedrock-enabled role if specified
#             sts_client = boto3.client("sts", **session_kwargs)
#             assumed = sts_client.assume_role(
#                 RoleArn=role_arn, RoleSessionName="BedrockSession"
#             )["Credentials"]
#             self.bedrock_client = boto3.client(
#                 "bedrock-runtime",
#                 region_name=region,
#                 aws_access_key_id=assumed["AccessKeyId"],
#                 aws_secret_access_key=assumed["SecretAccessKey"],
#                 aws_session_token=assumed["SessionToken"],
#                 # Set verify=False if your corp proxy breaks SSL; otherwise prefer True
#                 verify=False,
#             )
#         else:
#             # Fall back to default creds (env/EC2/Role/etc.)
#             self.bedrock_client = boto3.client("bedrock-runtime", **session_kwargs)

#         # Default Bedrock model
#         self.model_id = os.getenv(
#             "BEDROCK_MODEL_ID", "anthropic.claude-3-sonnet-20240229-v1:0"
#         )

#         # --- Database connection ---
#         self.conn = psycopg2.connect(
#             dbname=os.getenv("PGDATABASE", "postgres"),
#             user=os.getenv("PGUSER", "postgres"),
#             password=os.getenv("PGPASSWORD", "Vivek@7566"),
#             host=os.getenv("PGHOST", "localhost"),
#             port=int(os.getenv("PGPORT", "5432")),
#         )
#         self.conn.autocommit = True

#         # Table name (override with env if needed)
#         self.table_name = os.getenv("PGTABLE", "coupa_contracts")

#         # Live schema
#         live_cols = self._load_table_schema(self.table_name)
#         self.columns = [c.lower() for c in live_cols]
#         self.columns_set = set(self.columns)

#         # Synonyms that only map to actually-existing columns
#         self.synonyms = self._build_synonyms()

#     # -------------------------- Bedrock Helpers --------------------------

#     def _bedrock_chat(
#         self,
#         messages: List[Dict[str, str]],
#         max_tokens: int = 500,
#         temperature: float = 0.1,
#     ) -> str:
#         """
#         OpenAI-style messages -> Anthropic messages (Bedrock).
#         Supports "system", "user", "assistant".
#         Returns assistant text.
#         """
#         # Extract/concatenate any system messages into one system string
#         system_parts: List[str] = []
#         convo: List[Dict[str, Any]] = []
#         for m in messages:
#             role = m.get("role", "user")
#             content = m.get("content", "")
#             if role == "system":
#                 system_parts.append(content)
#             elif role in ("user", "assistant"):
#                 convo.append(
#                     {"role": role, "content": [{"type": "text", "text": content}]}
#                 )

#         body: Dict[str, Any] = {
#             "anthropic_version": "bedrock-2023-05-31",
#             "max_tokens": max_tokens,
#             "temperature": temperature,
#             "messages": convo,
#         }
#         if system_parts:
#             body["system"] = "\n\n".join(system_parts)

#         response = self.bedrock_client.invoke_model(
#             modelId=self.model_id,
#             contentType="application/json",
#             accept="application/json",
#             body=json.dumps(body),
#         )
#         payload = json.loads(response["body"].read())
#         # Anthropic content shape: {"content":[{"type":"text","text":"..."}], ...}
#         return payload["content"][0]["text"].strip()

#     @staticmethod
#     def _strip_code_fences(text: str) -> str:
#         """
#         Remove ```...``` fences with optional language hints.
#         """
#         if not text:
#             return text
#         fenced = re.match(r"^\s*```(?:\w+)?\s*(.*?)\s*```\s*$", text, re.DOTALL)
#         if fenced:
#             return fenced.group(1).strip()
#         return re.sub(r"```(?:\w+)?\s*|\s*```", "", text).strip()

#     # -------------------------- Utils & Schema --------------------------

#     def _load_table_schema(self, table_name: str) -> List[str]:
#         """
#         Read live column names from Postgres for robust validation.
#         """
#         with self.conn.cursor() as cur:
#             cur.execute(
#                 """
#                 SELECT column_name
#                 FROM information_schema.columns
#                 WHERE table_name = %s
#                 ORDER BY ordinal_position;
#                 """,
#                 (table_name,),
#             )
#             cols = [r[0] for r in cur.fetchall()]
#         return cols

#     @staticmethod
#     def _norm(s: str) -> str:
#         return re.sub(r"\s+", " ", s.strip().lower())

#     def _build_synonyms(self) -> Dict[str, str]:
#         """
#         Map common natural phrases to actual DB columns.
#         Only keep targets that truly exist in the live schema.
#         """
#         raw = {
#             # project manager
#             "project manager": "project_manager_name",
#             "manager": "project_manager_name",
#             "pm name": "project_manager_name",

#             # ids / numbers
#             "project id": "project_id",
#             "id": "project_id",
#             "project number": "project_no",
#             "project no": "project_no",
#             "number": "project_no",

#             # dates
#             "start date": "start_date",
#             "end date": "end_date",

#             # money / amounts
#             "total cost": "total_cost",
#             "contract value": "total_cost",
#             "sow amount": "total_cost",
#             "fees": "total_cost",
#             "fee amount": "total_cost",
#             "amount": "total_cost",
#             "cost": "total_cost",

#             # currency
#             "currency": "currency_code",
#             "currency code": "currency_code",
#         }

#         valid: Dict[str, str] = {}
#         for k, v in raw.items():
#             if v in self.columns_set:
#                 valid[self._norm(k)] = v
#         return valid

#     def _detect_referenced_columns(self, text: str) -> Tuple[set, set]:
#         """
#         Return (known_cols, unknown_mentions) based on:
#           - exact column tokens in query (with underscores)
#           - multi-word synonyms mapped to columns
#         """
#         q = self._norm(text)

#         found_cols = set()
#         unknown_mentions = set()

#         # 1) Multi-word phrases / synonyms
#         for phrase, col in self.synonyms.items():
#             if re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", q):
#                 found_cols.add(col)

#         # 2) Raw identifier-style mentions like "project_id", "total_cost"
#         ident_tokens = set(re.findall(r"\b[a-z_][a-z0-9_]*\b", q))
#         underscore_like = {t for t in ident_tokens if "_" in t}
#         for token in underscore_like:
#             if token in self.columns_set:
#                 found_cols.add(token)
#             else:
#                 # looks like a field but not actually present
#                 unknown_mentions.add(token)

#         return found_cols, unknown_mentions

#     # -------------------------- Routing (SQL vs RAG) --------------------------

#     def classify_query(self, query: str) -> str:
#         """
#         Hybrid routing:
#           1) Fast heuristic for obvious RAG queries
#           2) Detect referenced fields from schema+synonyms
#           3) Ask LLM for SQL vs RAG (hint)
#           4) If LLM says SQL but query references any unknown field -> force RAG
#         """
#         # Heuristics for obvious RAG (explanations/definitions)
#         rag_keywords = [
#             "explain", "meaning", "summarize", "summary", "describe",
#             "what is", "how to", "definition", "assumptions", "scope"
#         ]
#         q_norm = self._norm(query)
#         if any(k in q_norm for k in rag_keywords):
#             return "RAG"

#         # Detect column mentions
#         known_cols, unknown_mentions = self._detect_referenced_columns(query)

#         # LLM vote (uses actual schema list for context)
#         user_prompt = f"""
# You are a query router.
# - If the query is about counts, lists, filters, or structured fields in a table, classify as SQL.
# - If it's about explanations, summaries, or unstructured documents, classify as RAG.

# Table: {self.table_name}
# Available columns: {", ".join(sorted(self.columns))}

# Query: {query}

# Return only one word: SQL or RAG
# """.strip()

#         try:
#             text = self._bedrock_chat(
#                 messages=[
#                     {"role": "system", "content": "Respond with exactly 'SQL' or 'RAG'."},
#                     {"role": "user", "content": user_prompt},
#                 ],
#                 max_tokens=5,
#                 temperature=0.0,
#             ).strip().upper().replace(".", "")
#             llm_vote = "SQL" if text == "SQL" else "RAG"
#         except Exception as e:
#             print(f"Error classifying query: {e}")
#             llm_vote = "RAG"

#         # If LLM says SQL but query references unknown fields -> shift to RAG
#         if llm_vote == "SQL" and unknown_mentions:
#             print(f"[Router] Unknown field(s) mentioned: {sorted(unknown_mentions)} -> RAG")
#             return "RAG"

#         # Otherwise, respect the LLM vote
#         return llm_vote

#     # -------------------------- NL -> SQL + Execute --------------------------

#     def run_sql_query(self, query: str) -> str:
#         """
#         Convert natural language to SQL and run against Postgres DB.
#         Uses schema-aware prompt; falls back to RAG if an UndefinedColumn occurs.
#         """
#         sql_prompt = f"""
# Convert the following natural language query into a single SQL statement.

# Rules:
# - Use only this table: {self.table_name}
# - Valid columns are: {", ".join(sorted(self.columns))}
# - If no specific column is referenced, assume COUNT(*) or reasonable aggregates.
# - NEVER use columns not listed above.
# - Return only the SQL.

# Query: {query}
# """.strip()

#         try:
#             sql_text = self._bedrock_chat(
#                 messages=[
#                     {"role": "system", "content": "You are an expert SQL generator."},
#                     {"role": "user", "content": sql_prompt},
#                 ],
#                 max_tokens=300,
#                 temperature=0.0,
#             )

#             sql_query = self._strip_code_fences(sql_text)

#             # Keep only the first semicolon-terminated statement if multiple appear
#             semicolon_idx = sql_query.find(";")
#             if semicolon_idx != -1:
#                 sql_query = sql_query[: semicolon_idx + 1].strip()

#             print(f"Final SQL: {sql_query}")

#             with self.conn.cursor() as cur:
#                 cur.execute(sql_query)
#                 if cur.description:  # SELECT
#                     rows = cur.fetchall()
#                     colnames = [d[0] for d in cur.description]
#                     result = [dict(zip(colnames, row)) for row in rows]
#                 else:  # UPDATE/INSERT/DELETE
#                     result = f"Query executed successfully: {cur.rowcount} rows affected."

#             return json.dumps(result, indent=2, default=str)

#         except pg_errors.UndefinedColumn as e:
#             # Last-resort safety: route to RAG if model still used a bad column
#             print(f"[SQL] UndefinedColumn -> fallback to RAG: {e}")
#             return self.generate_response(
#                 query,
#                 context="The SQL path failed due to an unknown column; answering via documents instead."
#             )
#         except Exception as e:
#             return f"Error generating/executing SQL: {e}"

#     # -------------------------- RAG-style Answering --------------------------

#     def generate_response(self, query: str, context: str) -> str:
#         """
#         Generate response based on query and retrieved context using Bedrock Claude
#         """
#         prompt = f"""
# You are an expert assistant. Use only the information provided in the context below to answer the question.

# Context:
# {context}

# Question:
# {query}

# Instructions:
# - Greet user if user query is a greeting.
# - For specific questions, provide a clear, comprehensive answer.
# - If context is insufficient, say: "The context does not provide enough information."
# - Do not add external knowledge.
# """.strip()

#         try:
#             return self._bedrock_chat(
#                 messages=[
#                     {
#                         "role": "system",
#                         "content": "You are a helpful assistant for answering based on documents.",
#                     },
#                     {"role": "user", "content": prompt},
#                 ],
#                 max_tokens=1000,
#                 temperature=0.3,
#             )
#         except Exception as e:
#             return f"Error generating response: {e}"

#     # -------------------------- Entry Point --------------------------

#     def handle_query(self, query: str, context: Optional[str] = None) -> str:
#         """
#         Route query: decide SQL vs RAG, with safety fallbacks.
#         """
#         route = self.classify_query(query)
#         print(f"Routing decision: {route}")

#         if route == "SQL":
#             out = self.run_sql_query(query)
#             return out
#         else:
#             if context is None:
#                 context = "No context provided."
#             return self.generate_response(query, context)

#     # -------------------------- Doc-type Classification --------------------------

#     def classify_document_type(self, text: str) -> Tuple[str, float, List[str]]:
#         """
#         Classify a document as one of: SOW, CHANGE_REQUEST, AMENDMENT, UNKNOWN.
#         Returns: (doc_type_str, confidence_float, reasons_list)
#         """
#         snippet = text[:24000]

#         user_prompt = f"""
# You are a legal-doc classifier. Read the document and decide if it is:
# - "SOW" (Statement of Work: services/deliverables, Statement of Work, fees, term, assumptions, signatories)
# - "CHANGE_REQUEST" (CR: changes to scope/cost/timing of an existing SOW; fields like Revised SOW Amount, Timing Impact)
# - "AMENDMENT" (amendment to an agreement/SOW; language like "This Amendment is made by and between", "amends", "effective date", "amended contract value")
# - "UNKNOWN" (if ambiguous)

# Return ONLY a strict JSON object with keys:
# {{
#   "doc_type": "SOW" | "CHANGE_REQUEST" | "AMENDMENT" | "UNKNOWN",
#   "confidence": 0.0-1.0,
#   "reasons": ["short bullets"]
# }}

# Document (truncated):
# \"\"\"{snippet}\"\"\"
# """.strip()

#         try:
#             raw = self._bedrock_chat(
#                 messages=[
#                     {
#                         "role": "system",
#                         "content": "You classify contract documents. Always answer with valid JSON.",
#                     },
#                     {"role": "user", "content": user_prompt},
#                 ],
#                 max_tokens=220,
#                 temperature=0.0,
#             )

#             raw = self._strip_code_fences(raw)
#             data = json.loads(raw)

#             doc_type = str(data.get("doc_type", "UNKNOWN")).upper()
#             conf = float(data.get("confidence", 0.0))
#             reasons = data.get("reasons", [])

#             if doc_type not in {"SOW", "CHANGE_REQUEST", "AMENDMENT", "UNKNOWN"}:
#                 doc_type = "UNKNOWN"
#             if not isinstance(reasons, list):
#                 reasons = [str(reasons)]
#             return (doc_type, conf, reasons)

#         except Exception as e:
#             print(f"Error classify_document_type: {e}")
#             return ("UNKNOWN", 0.0, [])

# import os
# import re
# import json
# import boto3
# import psycopg2
# from typing import Dict, Any, Optional, Tuple, List
# from dotenv import load_dotenv

# load_dotenv()


# class LLMClient:
#     """
#     LLM helper for:
#       - Query routing (SQL vs RAG)
#       - NL -> SQL generation and execution (Postgres)
#       - RAG-style response generation
#       - Document-type classification (SOW / CHANGE_REQUEST / AMENDMENT / UNKNOWN)
#     """

#     # -------------------------- Init & Bedrock Setup --------------------------

#     def __init__(self):
#         # --- Bedrock credentials / client ---
#         region = os.getenv("AWS_REGION", "us-east-1")
#         role_arn = os.getenv("BEDROCK_ROLE_ARN")

#         session_kwargs: Dict[str, Any] = {"region_name": region}

#         # Use base keys if provided (optional; otherwise, let default AWS chain resolve)
#         base_access = os.getenv("AWS_ACCESS_KEY_ID")
#         base_secret = os.getenv("AWS_SECRET_ACCESS_KEY")
#         base_token = os.getenv("AWS_SESSION_TOKEN")
#         if base_access and base_secret:
#             session_kwargs.update(
#                 {
#                     "aws_access_key_id": base_access,
#                     "aws_secret_access_key": base_secret,
#                     **({"aws_session_token": base_token} if base_token else {}),
#                 }
#             )

#         if role_arn:
#             # Assume a Bedrock-enabled role if specified
#             sts_client = boto3.client("sts", **session_kwargs)
#             assumed = sts_client.assume_role(
#                 RoleArn=role_arn, RoleSessionName="BedrockSession"
#             )["Credentials"]
#             self.bedrock_client = boto3.client(
#                 "bedrock-runtime",
#                 region_name=region,
#                 aws_access_key_id=assumed["AccessKeyId"],
#                 aws_secret_access_key=assumed["SecretAccessKey"],
#                 aws_session_token=assumed["SessionToken"],
#                 verify=False,
#             )
#         else:
#             # Fall back to default creds (env/EC2/Role/etc.)
#             self.bedrock_client = boto3.client("bedrock-runtime", **session_kwargs)

#         # Default Bedrock model (override with BEDROCK_MODEL_ID if you like)
#         # Good choices: "anthropic.claude-3-5-sonnet-20240620-v1:0" or "anthropic.claude-3-sonnet-20240229-v1:0"
#         self.model_id = os.getenv(
#             "BEDROCK_MODEL_ID", "anthropic.claude-3-sonnet-20240229-v1:0"
#         )

#         # --- Database connection ---
#         self.conn = psycopg2.connect(
#             dbname=os.getenv("PGDATABASE", "postgres"),
#             user=os.getenv("PGUSER", "postgres"),
#             password=os.getenv("PGPASSWORD", "Vivek@7566"),
#             host=os.getenv("PGHOST", "localhost"),
#             port=int(os.getenv("PGPORT", "5432")),
#         )
#         self.conn.autocommit = True

#     # -------------------------- Bedrock Helpers --------------------------

#     def _bedrock_chat(
#         self,
#         messages: List[Dict[str, str]],
#         max_tokens: int = 500,
#         temperature: float = 0.1,
#     ) -> str:
#         """
#         OpenAI-style messages -> Anthropic messages (Bedrock)
#         Supports "system", "user", "assistant".
#         Returns assistant text.
#         """
#         # Extract/concatenate any system messages into one system string
#         system_parts: List[str] = []
#         convo: List[Dict[str, Any]] = []
#         for m in messages:
#             role = m.get("role", "user")
#             content = m.get("content", "")
#             if role == "system":
#                 system_parts.append(content)
#             elif role in ("user", "assistant"):
#                 convo.append(
#                     {"role": role, "content": [{"type": "text", "text": content}]}
#                 )

#         body: Dict[str, Any] = {
#             "anthropic_version": "bedrock-2023-05-31",
#             "max_tokens": max_tokens,
#             "temperature": temperature,
#             "messages": convo,
#         }
#         if system_parts:
#             body["system"] = "\n\n".join(system_parts)

#         response = self.bedrock_client.invoke_model(
#             modelId=self.model_id,
#             contentType="application/json",
#             accept="application/json",
#             body=json.dumps(body),
#         )
#         payload = json.loads(response["body"].read())
#         # Anthropic content shape: {"content":[{"type":"text","text":"..."}], ...}
#         return payload["content"][0]["text"].strip()

#     @staticmethod
#     def _strip_code_fences(text: str) -> str:
#         """
#         Remove ```...``` fences with optional language hints.
#         """
#         if not text:
#             return text
#         # Strip leading and trailing fenced blocks if present
#         fenced = re.match(r"^\s*```(?:\w+)?\s*(.*?)\s*```\s*$", text, re.DOTALL)
#         if fenced:
#             return fenced.group(1).strip()
#         # Also remove stray triple-backticks inside
#         return re.sub(r"```(?:\w+)?\s*|\s*```", "", text).strip()

#     # -------------------------- Routing (SQL vs RAG) --------------------------

#     def classify_query(self, query: str) -> str:
#         """
#         Classify whether query should go to SQL or RAG.
#         Returns: "SQL" or "RAG"
#         """
#         user_prompt = f"""
# You are a query router.
# - If the query is about numbers, counts, lists, filters, or structured fields (like customers,total cost, orders, project_id, dates, etc.), classify as SQL.
# - If the query is about explanations, summaries, meaning, or unstructured documents, classify as RAG.

# Query: {query}

# Return only one word: SQL or RAG
# """.strip()

#         try:
#             text = self._bedrock_chat(
#                 messages=[
#                     {
#                         "role": "system",
#                         "content": "You are a classifier. Reply only with SQL or RAG.",
#                     },
#                     {"role": "user", "content": user_prompt},
#                 ],
#                 max_tokens=5,
#                 temperature=0.0,
#             )
#             return text.strip().upper().replace(".", "")
#         except Exception as e:
#             print(f"Error classifying query: {e}")
#             return "RAG"  # conservative fallback

#     # -------------------------- NL -> SQL + Execute --------------------------

#     def run_sql_query(self, query: str) -> str:
#         """
#         Convert natural language to SQL and run against Postgres DB
#         """
#         sql_prompt = f"""
# Convert the following natural language query into an SQL statement:
# Query: {query}

# Table name: coupa_contracts
# Columns: project_manager_name,total cost, project_id, project_no, start_date, end_date, currency_code

# Return only the SQL statement without explanation.
# """.strip()

#         try:
#             sql_text = self._bedrock_chat(
#                 messages=[
#                     {"role": "system", "content": "You are an expert SQL generator."},
#                     {"role": "user", "content": sql_prompt},
#                 ],
#                 max_tokens=300,
#                 temperature=0.0,
#             )

#             sql_query = self._strip_code_fences(sql_text)

#             # Guard against model echoing prose
#             # Keep only the first semicolon-terminated statement if multiple appear
#             semicolon_idx = sql_query.find(";")
#             if semicolon_idx != -1:
#                 sql_query = sql_query[: semicolon_idx + 1].strip()

#             print(f"Final SQL: {sql_query}")

#             with self.conn.cursor() as cur:
#                 cur.execute(sql_query)
#                 if cur.description:  # SELECT
#                     rows = cur.fetchall()
#                     colnames = [d[0] for d in cur.description]
#                     result = [dict(zip(colnames, row)) for row in rows]
#                 else:  # UPDATE/INSERT/DELETE
#                     result = f"Query executed successfully: {cur.rowcount} rows affected."

#             return json.dumps(result, indent=2, default=str)

#         except Exception as e:
#             return f"Error generating/executing SQL: {e}"

#     # -------------------------- RAG-style Answering --------------------------

#     def generate_response(self, query: str, context: str) -> str:
#         """
#         Generate response based on query and retrieved context using Bedrock Claude
#         """
#         prompt = f"""
# You are an expert assistant. Use only the information provided in the context below to answer the question.

# Context:
# {context}

# Question:
# {query}

# Instructions:
# - Greet user if user query is a greeting.
# - For specific questions, provide a clear, comprehensive answer.
# - If context is insufficient, say: "The context does not provide enough information."
# - Do not add external knowledge.
# """.strip()

#         try:
#             return self._bedrock_chat(
#                 messages=[
#                     {
#                         "role": "system",
#                         "content": "You are a helpful assistant for answering based on documents.",
#                     },
#                     {"role": "user", "content": prompt},
#                 ],
#                 max_tokens=1000,
#                 temperature=0.3,
#             )
#         except Exception as e:
#             return f"Error generating response: {e}"

#     def handle_query(self, query: str, context: Optional[str] = None) -> str:
#         """
#         Route query: decide SQL vs RAG
#         """
#         route = self.classify_query(query)
#         print(f"Routing decision: {route}")

#         if route == "SQL":
#             return self.run_sql_query(query)
#         else:
#             if context is None:
#                 context = "No context provided."
#             return self.generate_response(query, context)

#     # -------------------------- Doc-type Classification --------------------------

#     def classify_document_type(self, text: str) -> Tuple[str, float, List[str]]:
#         """
#         Classify a document as one of: SOW, CHANGE_REQUEST, AMENDMENT, UNKNOWN.
#         Returns: (doc_type_str, confidence_float, reasons_list)
#         """
#         snippet = text[:24000]

#         user_prompt = f"""
# You are a legal-doc classifier. Read the document and decide if it is:
# - "SOW" (Statement of Work: services/deliverables, Statement of Work,fees, term, assumptions, signatories)
# - "CHANGE_REQUEST" (CR: changes to scope/cost/timing of an existing SOW; fields like Revised SOW Amount, Timing Impact)
# - "AMENDMENT" (amendment to an agreement/SOW; language like "This Amendment is made by and between", "amends", "effective date", "amended contract value")
# - "UNKNOWN" (if ambiguous)

# Return ONLY a strict JSON object with keys:
# {{
#   "doc_type": "SOW" | "CHANGE_REQUEST" | "AMENDMENT" | "UNKNOWN",
#   "confidence": 0.0-1.0,
#   "reasons": ["short bullets"]
# }}

# Document (truncated):
# \"\"\"{snippet}\"\"\"
# """.strip()

#         try:
#             raw = self._bedrock_chat(
#                 messages=[
#                     {
#                         "role": "system",
#                         "content": "You classify contract documents. Always answer with valid JSON.",
#                     },
#                     {"role": "user", "content": user_prompt},
#                 ],
#                 max_tokens=220,
#                 temperature=0.0,
#             )

#             raw = self._strip_code_fences(raw)
#             data = json.loads(raw)

#             doc_type = str(data.get("doc_type", "UNKNOWN")).upper()
#             conf = float(data.get("confidence", 0.0))
#             reasons = data.get("reasons", [])

#             if doc_type not in {"SOW", "CHANGE_REQUEST", "AMENDMENT", "UNKNOWN"}:
#                 doc_type = "UNKNOWN"
#             if not isinstance(reasons, list):
#                 reasons = [str(reasons)]
#             return (doc_type, conf, reasons)

#         except Exception as e:
#             print(f"Error classify_document_type: {e}")
#             return ("UNKNOWN", 0.0, [])
