import os
import re
import json
import uuid
import logging
from enum import Enum
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import requests
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Optional: use PyPDF2 if available to build proper headings/subheadings
try:
    from PyPDF2 import PdfReader  # type: ignore
    PDF_AVAILABLE = True
except Exception:
    PDF_AVAILABLE = False

# If you already have your own LLMClient for classification, you can still use it
try:
    from llm_client import LLMClient  # optional, for doc-type classification only
except Exception:  # pragma: no cover
    LLMClient = None  # type: ignore

# ============================= Utils (unchanged except markdown clean) ============================= #

class DocType(str, Enum):
    SOW = "SOW"
    CHANGE_REQUEST = "CHANGE_REQUEST"
    AMENDMENT = "AMENDMENT"
    UNKNOWN = "UNKNOWN"


def normalize_ws(s: str) -> str:
    s = s.replace("\u00A0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def clean_markdown_noise(s: str) -> str:
    # Keep headings! Only remove page markers and collapse spacing.
    s = re.sub(r"<!--\s*Page\s*\d+\s*-->\s*", "\n", s)
    return s


def try_parse_date(s: str) -> Optional[str]:
    if not s:
        return None
    s = s.strip().strip(".").strip(",")
    s = re.sub(r"(\d{1,2})(st|nd|rd|th)\b", r"\1", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s)
    s = s.replace(" - ", "-").replace(" – ", "-").replace(" — ", "-")
    fmts = [
        "%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%m/%d/%Y",
        "%d %b %Y", "%d %B %Y", "%b %d, %Y", "%B %d, %Y",
        "%d.%m.%Y", "%d-%b-%Y", "%d-%B-%Y"
    ]
    for f in fmts:
        try:
            return datetime.strptime(s, f).strftime("%Y-%m-%d")
        except Exception:
            pass
    m = re.match(r"(\d{1,2})[-/\s](Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*[-/\s](\d{4})",
                 s, flags=re.IGNORECASE)
    if m:
        try:
            return datetime.strptime(f"{m.group(1)} {m.group(2)} {m.group(3)}", "%d %b %Y").strftime("%Y-%m-%d")
        except Exception:
            pass
    m2 = re.search(r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s+\d{1,2},\s+\d{4}", s,
                   flags=re.IGNORECASE)
    if m2:
        try:
            return datetime.strptime(m2.group(0), "%b %d, %Y").strftime("%Y-%m-%d")
        except Exception:
            pass
    m3 = re.search(r"\d{1,2}\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}",
                   s, flags=re.IGNORECASE)
    if m3:
        try:
            return datetime.strptime(m3.group(0), "%d %B %Y").strftime("%Y-%m-%d")
        except Exception:
            pass
    return s


def flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def extract_headings_from_markdown(md: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Find first '# ' (heading) and first '## ' (subheading) lines in markdown.
    Returns the full markdown lines (e.g., '# Title', '## Section') or None.
    """
    heading_md = None
    subheading_md = None
    for line in md.splitlines():
        if heading_md is None and line.strip().startswith("# "):
            heading_md = line.strip()
        elif subheading_md is None and line.strip().startswith("## "):
            subheading_md = line.strip()
        if heading_md and subheading_md:
            break
    return heading_md, subheading_md


# =================== Jina Embeddings (Open Source) =================== #

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


# ======================= NEW: LLM JSON Field Extractor ======================== #

class LLMFieldExtractor:
    """Call an LLM on Bedrock to extract contract fields as strict JSON.

    This extractor does **no regex mining** of fields. We only:
      1) Build a doc-type specific JSON schema (keys match your old output keys).
      2) Ask the LLM to return **only** a JSON object matching that schema.
      3) Post-process dates via `try_parse_date` and normalize amounts.
    """

    def __init__(self):
        load_dotenv()
        self.region = os.getenv("BEDROCK_REGION", "us-east-1")
        self.role_arn = os.getenv("BEDROCK_ROLE_ARN")
        self.model_id = os.getenv("BEDROCK_LLM_MODEL_ID", "anthropic.claude-3-sonnet-20240229-v1:0")

        if self.role_arn:
            sts_client = boto3.client(
                "sts",
                region_name=self.region,
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID") or None,
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY") or None,
                aws_session_token=os.getenv("AWS_SESSION_TOKEN") or None,
            )
            assumed = sts_client.assume_role(
                RoleArn=self.role_arn,
                RoleSessionName=f"BedrockLLMExtract-{uuid.uuid4().hex[:8]}"
            )
            creds = assumed["Credentials"]
            self.client = boto3.client(
                "bedrock-runtime",
                region_name=self.region,
                aws_access_key_id=creds["AccessKeyId"],
                aws_secret_access_key=creds["SecretAccessKey"],
                aws_session_token=creds["SessionToken"],
            )
        else:
            self.client = boto3.client("bedrock-runtime", region_name=self.region)

    # ---- schema helpers ---- #

    @staticmethod
    def _base_schema() -> Dict[str, Any]:
        return {
            "SOW Name": None,
            "Supplier Name": None,
            "Supplier Geography": None,
            "EXECUTIVE SUMMARY": None,
            "Services": None,
            "Deliverables": None,
            "SOW Effective Date": None,
            "SOW Start Date": None,
            "SOW End Date": None,
            "effect until": None,  # kept for legacy compat
            "Fee Amount": None,
            "Fee Basis - Fixed/TnM": None,
            "Invoices": None,
            "GEHC Project Name": None,
            "GEHC PM Phone": None,
            "GEHC PM Email": None,
            "Supplier PM": None,
            "Supplier PM Phone": None,
            "Supplier Email": None,
            "General Assumption": None,
            "Out of Scope": None,
            "SLA": None,
            "Location": None,
            "GEHC Signed Party": None,
            "Supplier Signed Party": None,
        }

    @staticmethod
    def _cr_schema() -> Dict[str, Any]:
        return {
            "SOW Name": None,
            "Change Request Start Date": None,
            "Change Request End Date": None,
            "Supplier Name": None,
            "Supplier Geography": None,
            "Date Submitted": None,
            "Change To: Delivery/Cost": None,
            "Revised SOW Amount": None,
            "Original SOW Amount": None,
            "Invoices": None,
            "Timing Impact": None,
            "Date Response Delivered": None,
            "GEHC Signed Party": None,
            "Supplier Signed Party": None,
        }

    @staticmethod
    def _amend_schema() -> Dict[str, Any]:
        return {
            "Ammendment Date": None,
            "Parties": None,
            "Effective Date": None,
            "Total Cost": None,
            "Payment Schedule": None,
            "Supplier Geography": None,
            "Supplier Name": None,
        }

    @staticmethod
    def _norm_amount(x: Optional[str]) -> Optional[str]:
        if not x:
            return x
        v = str(x).strip()
        v = re.sub(r"[,$₹€£]", "", v)
        v = re.sub(r"\s{2,}", " ", v)
        return v

    @staticmethod
    def _select_schema(doc_type: DocType) -> Dict[str, Any]:
        if doc_type == DocType.SOW:
            return LLMFieldExtractor._base_schema()
        if doc_type == DocType.CHANGE_REQUEST:
            return LLMFieldExtractor._cr_schema()
        if doc_type == DocType.AMENDMENT:
            return LLMFieldExtractor._amend_schema()
        # default: try SOW-like
        return LLMFieldExtractor._base_schema()

    def _build_prompt(self, text: str, doc_type: DocType, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Build Anthropic messages-style payload for Bedrock with a single user turn
        and system guidance to avoid role alternation errors.
        """
        schema_keys = list(schema.keys())
        guidance = (
            "You are an expert contracts analyst. Extract the requested fields from the document strictly as JSON. "
            "If a value is truly absent, return null. Do not invent values. Dates can appear in many formats; "
            "return them as found (we will normalize later). Do not include any extra keys."
        )
        fields_list = "\n- " + "\n- ".join(schema_keys)
        user_msg = (
            f"Document type: {doc_type.value}.\n\n"
            f"Return ONLY a JSON object with exactly these keys (values can be string or null):\n{fields_list}\n\n"
            f"Document follows below between <doc> tags.\n<doc>\n{text[:150000]}\n</doc>"
        )
        return {
            "anthropic_version": "bedrock-2023-05-31",
            "system": guidance,
            "max_tokens": 4096,
            "temperature": 0,
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": user_msg}]}
            ],
        }

    @staticmethod
    def _extract_json_from_text(txt: str) -> Dict[str, Any]:
        """Be forgiving if the model wraps JSON with prose; try to snip the JSON blob."""
        try:
            return json.loads(txt)
        except Exception:
            pass
        # Find first { and last } and try again
        start = txt.find("{")
        end = txt.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = txt[start:end+1]
            try:
                return json.loads(snippet)
            except Exception:
                pass
        raise ValueError("LLM did not return valid JSON")

    def extract_fields(self, text: str, doc_type: DocType) -> Dict[str, Any]:
        schema = self._select_schema(doc_type)
        payload = self._build_prompt(text, doc_type, schema)
        resp = self.client.invoke_model(
            modelId=self.model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(payload),
        )
        raw = resp["body"].read().decode("utf-8")
        data = json.loads(raw)
        # Anthropic on Bedrock returns {"content":[{"type":"text","text":"..."}], ...}
        try:
            model_text = "".join([p.get("text", "") for p in data.get("content", [])]) or data.get("output_text", "")
        except Exception:
            model_text = data.get("output_text", "")
        parsed = self._extract_json_from_text(model_text)

        # ensure all required keys exist
        out: Dict[str, Any] = {k: parsed.get(k) for k in schema.keys()}

        # normalize dates and amounts
        for k in list(out.keys()):
            v = out[k]
            if v is None:
                continue
            if "date" in k.lower() or k.lower().endswith(" end date") or k.lower().endswith(" start date") or k.lower() == "effect until":
                out[k] = try_parse_date(str(v))
            if any(tok in k.lower() for tok in ["amount", "cost", "value", "total"]):
                out[k] = self._norm_amount(str(v))

        # maintain legacy mirror for SOW end date under "effect until"
        if doc_type == DocType.SOW and out.get("SOW End Date") and not out.get("effect until"):
            out["effect until"] = out.get("SOW End Date")

        return out


# ================== Tiny Heuristic Doc-Type Detector (fallback) ================== #

class HeuristicTypeDetector:
    @staticmethod
    def detect_type(text: str) -> DocType:
        t = text.lower()
        if re.search(r"\bchange\s+request\b", t) or re.search(r"\bcr\s*#?:?\b", t):
            return DocType.CHANGE_REQUEST
        if re.search(r"\bstatement\s+of\s+work\b", t) or re.search(r"\bsow\b", t):
            return DocType.SOW
        if re.search(r"\bamendment\b", t) or re.search(r"\bammendment\b", t):
            return DocType.AMENDMENT
        return DocType.UNKNOWN


# ========================= Structure-Preserving Parser (RESTORED) ========================= #

class StructurePreservingDocumentParser:
    """
    Advanced document parser that preserves and creates markdown structure.
    - Adds a top-level '# {Title}' (from filename).
    - Detects numbered/lettered sections as headings '##', '###', etc.
    - Converts bullet/numbered lists.
    - Inserts '<!-- Page N -->' markers (later collapsed by clean_markdown_noise()).
    """

    def __init__(self):
        # Patterns copied from your previous implementation
        self.title_patterns = [
            re.compile(r'^([A-Z][A-Z\s&-]+[A-Z])$', re.MULTILINE),  # ALL CAPS titles
            re.compile(r'^([A-Z][^.!?]*[A-Z])$', re.MULTILINE),      # Title case
        ]
        self.section_patterns = [
            re.compile(r'^(\d+(?:\.\d+)*)\s*[:\-\.]?\s*(.+)$', re.MULTILINE),  # 1.2.3 Section Name
            re.compile(r'^([A-Z])\.\s*(.+)$', re.MULTILINE),                   # A. Section Name
            re.compile(r'^\(([a-z]+)\)\s*(.+)$', re.MULTILINE),                # (a) Subsection
        ]
        self.list_patterns = [
            re.compile(r'^\s*[\-\*\+]\s+(.+)$', re.MULTILINE),                 # - List item
            re.compile(r'^\s*\d+\.\s+(.+)$', re.MULTILINE),                    # 1. Numbered list
        ]

    def parse_pdf_to_markdown(self, file_path: str) -> str:
        """
        Parse PDF and convert to structured markdown with # / ## headings.
        Falls back to PyPDFLoader if PyPDF2 is unavailable or fails.
        """
        try:
            if not PDF_AVAILABLE:
                raise ImportError("PyPDF2 not available")

            reader = PdfReader(file_path)
            markdown_content: List[str] = []

            # Top-level heading from filename
            file_name = os.path.splitext(os.path.basename(file_path))[0]
            title = file_name.replace('_', ' ').replace('-', ' ').strip()
            title = re.sub(r'\s+', ' ', title).title()
            markdown_content.append(f"# {title}\n")

            page_num = 0
            for page in reader.pages:
                page_num += 1
                try:
                    text = page.extract_text() or ""
                except Exception:
                    text = ""

                if not text.strip():
                    continue

                # Keep page marker (cleaner will just collapse spacing)
                markdown_content.append(f"\n<!-- Page {page_num} -->\n")

                lines = text.split('\n')
                processed_lines = self._structure_text_lines(lines)
                markdown_content.extend(processed_lines)

            return '\n'.join(markdown_content)

        except Exception as e:
            logging.warning(f"Failed to parse PDF structure for {file_path}: {e}")
            # Fallback to regular loading and still add a '# Title'
            try:
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                file_name = os.path.splitext(os.path.basename(file_path))[0]
                title = file_name.replace('_', ' ').replace('-', ' ').strip()
                title = re.sub(r'\s+', ' ', title).title()
                body = '\n'.join([normalize_ws(d.page_content) for d in docs])
                return f"# {title}\n\n{body}"
            except Exception as e2:
                logging.error(f"PDF fallback failed for {file_path}: {e2}")
                return ""

    def _structure_text_lines(self, lines: List[str]) -> List[str]:
        structured_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            structured_lines.append(self._apply_structure_detection(line))
        return structured_lines

    def _apply_structure_detection(self, text: str) -> str:
        # Section patterns → heading levels
        for pattern in self.section_patterns:
            match = pattern.match(text)
            if match:
                if len(match.groups()) >= 2:
                    section_num = match.group(1)
                    section_title = match.group(2).strip()

                    # Determine heading level based on section number depth
                    if pattern.pattern.startswith('^([A-Z])\\.'):
                        depth = 2  # Lettered sections as H2
                        label = section_num
                    elif pattern.pattern.startswith('^\\(([a-z]+)\\)'):
                        depth = 3  # Parenthesized lowercase as H3
                        label = f"({section_num})"
                    else:
                        if '.' in section_num:
                            depth = section_num.count('.') + 2
                            depth = min(depth, 6)
                        else:
                            depth = 2
                        label = section_num

                    return f"{'#' * depth} {label} {section_title}"

        # Lists
        for pattern in self.list_patterns:
            match = pattern.match(text)
            if match:
                return f"- {match.group(1)}"

        # Inline titles (ALL CAPS or Title Case) → subheading
        for pattern in self.title_patterns:
            if pattern.match(text) and 10 < len(text) < 100:
                return f"## {text}"

        # Paragraph
        return text


# ============================== Main Processor ============================== #

class DocumentProcessor:
    def __init__(self, output_folder: str = "output"):
        load_dotenv()
        self.qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        self.qdrant_port = int(os.getenv("QDRANT_PORT", 6333))
        self.collection_name = os.getenv("QDRANT_COLLECTION_NAME", "documents_collection")
        self.client = QdrantClient(host=self.qdrant_host, port=self.qdrant_port)

        self.embedder = JinaEmbeddingClient()
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=350, length_function=len)

        self.llm_classifier = LLMClient() if LLMClient else None
        self.field_extractor = LLMFieldExtractor()
        self.heuristic = HeuristicTypeDetector()
        self.document_parser = StructurePreservingDocumentParser()

        self.output_folder = output_folder
        os.makedirs(output_folder, exist_ok=True)
        self.markdown_folder = os.path.join(output_folder, "markdown")
        self.chunks_folder = os.path.join(output_folder, "chunks")
        os.makedirs(self.markdown_folder, exist_ok=True)
        os.makedirs(self.chunks_folder, exist_ok=True)

        self._ensure_collection_exists()

    def _ensure_collection_exists(self):
        try:
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            vector_size = int(os.getenv("EMBEDDING_DIMENSIONS", "1024"))  # Jina v3 = 1024 dims
            if self.collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
                )
                print(f"Created collection: {self.collection_name} (size={vector_size})")
            else:
                print(f"Collection {self.collection_name} already exists")
        except Exception as e:
            print(f"Error setting up Qdrant collection: {e}")

    # ---- Extraction flow with LLM ---- #

    def pdf_to_markdown(self, pdf_path: str) -> str:
        markdown_content = self.document_parser.parse_pdf_to_markdown(pdf_path)
        self._save_markdown(pdf_path, markdown_content)
        return markdown_content

    def create_document_chunk(self, file_path: str) -> Optional[Dict[str, Any]]:
        try:
            markdown_content = self.pdf_to_markdown(file_path)
            if not markdown_content:
                return None

            # Preserve headings (don't strip them in clean_markdown_noise)
            cleaned = normalize_ws(clean_markdown_noise(markdown_content))

            # Extract heading/subheading from the markdown we just produced
            heading_md, subheading_md = extract_headings_from_markdown(markdown_content)

            # (A) LLM classification (if available)
            llm_doc_type_str, llm_conf, llm_reasons = ("UNKNOWN", 0.0, [])
            if self.llm_classifier:
                try:
                    llm_doc_type_str, llm_conf, llm_reasons = self.llm_classifier.classify_document_type(cleaned)
                except Exception as e:
                    logging.warning(f"LLM doc-type classification failed: {e}")

            # (B) Heuristic detection fallback
            heur_type = self.heuristic.detect_type(cleaned)

            # (C) Decide final type
            final_doc_type = heur_type
            try:
                if llm_doc_type_str != "UNKNOWN" and llm_conf >= 0.55:
                    final_doc_type = DocType[llm_doc_type_str]
                elif heur_type == DocType.UNKNOWN and llm_doc_type_str in {"SOW", "CHANGE_REQUEST", "AMENDMENT"}:
                    final_doc_type = DocType[llm_doc_type_str]
            except Exception:
                pass

            # (D) **LLM JSON extraction**
            fields = self.field_extractor.extract_fields(cleaned, final_doc_type)

            metadata = {
                "doc_type": final_doc_type.value,
                "doc_type_llm": llm_doc_type_str,
                "doc_type_llm_confidence": llm_conf,
                "doc_type_llm_reasons": llm_reasons,
                "doc_type_heuristic": heur_type.value,
                "extracted_fields": fields,
                # Save headings in metadata as well
                "heading_md": heading_md,
                "subheading_md": subheading_md,
            }

            chunk = {
                "id": str(uuid.uuid4()),
                "file_path": file_path,
                "file_name": os.path.basename(file_path),
                "content": markdown_content,   # contains '#', '##' structure
                "content_type": "document",
                "processed_date": datetime.now().isoformat(),
                "metadata": metadata,
            }
            return chunk
        except Exception as e:
            print(f"Error creating document chunk: {e}")
            return None

    def generate_embedding(self, text: str) -> List[float]:
        try:
            max_length = 8000
            if len(text) > max_length:
                text = text[:max_length]
            return self.embedder.embed(text)
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return []

    def _save_markdown(self, file_path: str, markdown_content: str):
        try:
            file_name = os.path.basename(file_path)
            base_name = os.path.splitext(file_name)[0]
            markdown_file = os.path.join(self.markdown_folder, f"{base_name}.md")
            with open(markdown_file, "w", encoding="utf-8") as f:
                f.write(markdown_content)
            print(f"Saved markdown: {markdown_file}")
        except Exception as e:
            print(f"Error saving markdown for {file_path}: {e}")

    def _save_chunk_json(self, file_path: str, chunk: Dict[str, Any]):
        try:
            file_name = os.path.basename(file_path)
            base_name = os.path.splitext(file_name)[0]
            chunk_file = os.path.join(self.chunks_folder, f"{base_name}_chunk.json")
            with open(chunk_file, "w", encoding="utf-8") as f:
                json.dump(chunk, f, indent=2, ensure_ascii=False)

            chunk_summary = dict(chunk)
            content = chunk["content"] or ""
            chunk_summary["content_preview"] = (content[:500] + "...") if len(content) > 500 else content
            chunk_summary["content_length"] = len(content)
            summary_file = os.path.join(self.chunks_folder, f"{base_name}_summary.json")
            with open(summary_file, "w", encoding="utf-8") as f:
                json.dump(chunk_summary, f, indent=2, ensure_ascii=False)

            print(f"Saved chunk JSON: {chunk_file}")
            print(f"Saved chunk summary: {summary_file}")
        except Exception as e:
            print(f"Error saving chunk JSON for {file_path}: {e}")

    def store_to_qdrant(self, chunk: Dict[str, Any]) -> bool:
        try:
            embedding = self.generate_embedding(chunk["content"])
            if not embedding:
                print("Failed to generate embedding")
                return False

            # Include heading/subheading in payload
            payload = {
                "file_path": chunk["file_path"],
                "file_name": chunk["file_name"],
                "content": chunk["content"],  # markdown with '#', '##'
                "content_type": chunk["content_type"],
                "processed_date": chunk["processed_date"],
                "doc_type": chunk["metadata"].get("doc_type"),
                "extracted_fields": chunk["metadata"].get("extracted_fields", {}),
                "heading_md": chunk["metadata"].get("heading_md"),
                "subheading_md": chunk["metadata"].get("subheading_md"),
            }
            flat = flatten_dict({"extracted": payload["extracted_fields"]})
            payload.update(flat)

            point = PointStruct(id=chunk["id"], vector=embedding, payload=payload)
            self.client.upsert(collection_name=self.collection_name, points=[point])
            print(f"Successfully stored document: {chunk['file_name']}")
            return True
        except Exception as e:
            print(f"Error storing to Qdrant: {e}")
            return False

    def process_document(self, file_path: str):
        print(f"Processing document: {file_path}")
        chunk = self.create_document_chunk(file_path)
        if chunk:
            self._save_chunk_json(file_path, chunk)
            ok = self.store_to_qdrant(chunk)
            if ok:
                print(f"Successfully processed: {file_path}")
            else:
                print(f"Failed to store to Qdrant: {file_path}")
        else:
            print(f"Failed to create chunk for: {file_path}")

    def process_documents_directory(self, directory_path: str) -> None:
        if not os.path.exists(directory_path):
            print(f"Directory not found: {directory_path}")
            return
        pdf_files = [f for f in os.listdir(directory_path) if f.lower().endswith('.pdf')]
        if not pdf_files:
            print(f"No PDF files found in: {directory_path}")
            return
        print(f"Found {len(pdf_files)} PDF files to process")
        for pdf_file in pdf_files:
            file_path = os.path.join(directory_path, pdf_file)
            self.process_document(file_path)
            print("-" * 50)


# ------------------------------ Main (example) ------------------------------ #

def main():
    output_folder = os.getenv("OUTPUT_DIR", r"C:\\Users\\CSS\\Desktop\\Workspace\\RAG_Coupa_chatbot\\output2")
    docs_directory = os.getenv("DOCS_DIR", r"C:\\Users\\CSS\\Desktop\\Workspace\\RAG_Coupa_chatbot\\Docs")
    processor = DocumentProcessor(output_folder=output_folder)
    processor.process_documents_directory(docs_directory)
    print(f"\n📁 Output saved to: {output_folder}")
    print(f"  - Markdown files: {processor.markdown_folder}")
    print(f"  - Chunk JSON files: {processor.chunks_folder}")


if __name__ == "__main__":
    main()
