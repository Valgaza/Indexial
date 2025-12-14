
# # import os
# # import re
# # import json
# # import logging
# # import uuid
# # from enum import Enum
# # from datetime import datetime
# # from typing import List, Dict, Any, Optional, Tuple
# # from dotenv import load_dotenv
# # # Core dependencies
# # from langchain.text_splitter import RecursiveCharacterTextSplitter
# # from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
# # import boto3  # for Bedrock client
# # from qdrant_client import QdrantClient
# # from qdrant_client.models import Distance, VectorParams, PointStruct
# # from llm_client import LLMClient

# # # ------------------------------ Helpers: Types & Utilities ------------------------------

# # class DocType(str, Enum):
# #     SOW = "SOW"
# #     CHANGE_REQUEST = "CHANGE_REQUEST"
# #     AMENDMENT = "AMENDMENT"
# #     UNKNOWN = "UNKNOWN"


# # def normalize_ws(s: str) -> str:
# #     """Collapse weird whitespace; keep newlines but strip trailing spaces.
# #     ex: "input=Hello\u00A0World!    This   is\ta test.\n\n\n\nNew section.
# #     output='Hello World! This is a test.\n\nNew section.'""""
# #     s = s.replace('\u00A0', ' ') #Replace non-breaking spaces with normal spaces
# #     s = re.sub(r'[ \t]+', ' ', s)
# #     # Remove many blank lines
# #     s = re.sub(r'\n{3,}', '\n\n', s)
# #     return s.strip()


# # def clean_markdown_noise(s: str) -> str:
# #     """Remove markdown markers we added so regex over raw text is easier."""
# #     # Remove <!-- Page x --> markers
# #     s = re.sub(r'<!--\s*Page\s*\d+\s*-->\s*', '\n', s)
# #     # Remove markdown heading(###) hashes but keep the text
# #     s = re.sub(r'^[ \t]*#{1,6}[ \t]+', '', s, flags=re.MULTILINE)
# #     return s


# # def try_parse_date(s: str) -> Optional[str]:
# #     """Try to normalize common/narrative date formats to YYYY-MM-DD. Return original if unknown.
# #     print(try_parse_date("2024-09-27"))       # ISO format
# #     # -> '2024-09-27'

# #     print(try_parse_date("27-09-2024"))       # DD-MM-YYYY
# #     # -> '2024-09-27'

# #     print(try_parse_date("09/27/2024"))       # MM/DD/YYYY
# #     # -> '2024-09-27'

# #     print(try_parse_date("27th September 2024"))  # With ordinal suffix
# #     # -> '2024-09-27'

# #     print(try_parse_date("Jan 1, 2024"))
# #     # -> '2024-01-01'

# #     print(try_parse_date("1-Jan-2024"))
# #     # -> '2024-01-01'

# #     print(try_parse_date("random string"))
# #     # -> 'random string'  (fallback)
# #     """
# #     if not s:
# #         return None
# #     s = s.strip().strip('.').strip(',')
# #     # Remove ordinal suffixes like 1st, 2nd, 3rd, 21st
# #     s = re.sub(r'(\d{1,2})(st|nd|rd|th)\b', r'\1', s, flags=re.IGNORECASE)
# #     # Normalize multiple spaces/hyphens
# #     s = re.sub(r'\s+', ' ', s)
# #     s = s.replace(' - ', '-').replace(' – ', '-').replace(' — ', '-')

# #     fmts = [
# #         "%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%m/%d/%Y",
# #         "%d %b %Y", "%d %B %Y", "%b %d, %Y", "%B %d, %Y",
# #         "%d.%m.%Y", "%d-%b-%Y", "%d-%B-%Y"
# #     ]
# #     for f in fmts:
# #         try:
# #             return datetime.strptime(s, f).strftime("%Y-%m-%d")
# #         except Exception:
# #             pass
# #     # Loose “DD-Mon-YYYY” variants like 01-Jan-2024
# #     m = re.match(r'(\d{1,2})[-/\s](Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*[-/\s](\d{4})', s, flags=re.IGNORECASE)
# #     if m:
# #         try:
# #             return datetime.strptime(f"{m.group(1)} {m.group(2)} {m.group(3)}", "%d %b %Y").strftime("%Y-%m-%d")
# #         except Exception:
# #             pass
# #     # “Month DD, YYYY”
# #     m2 = re.search(r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s+\d{1,2},\s+\d{4}', s, flags=re.IGNORECASE)
# #     if m2:
# #         try:
# #             return datetime.strptime(m2.group(0), "%b %d, %Y").strftime("%Y-%m-%d")
# #         except Exception:
# #             pass
# #     m3 = re.search(r'\d{1,2}\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}', s, flags=re.IGNORECASE)
# #     if m3:
# #         try:
# #             return datetime.strptime(m3.group(0), "%d %B %Y").strftime("%Y-%m-%d")
# #         except Exception:
# #             pass
# #     return s  # fallback: return as-is


# # def first_nonempty(*vals) -> Optional[str]:
# #     for v in vals:
# #         if v and str(v).strip():
# #             return str(v).strip()
# #     return None


# # def flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
# #     """This function flattens a nested dictionary into a single-level dictionary, 
# #         where nested keys are joined using a separator (default ".")."""
# #     items = []
# #     for k, v in d.items():
# #         new_key = f"{parent_key}{sep}{k}" if parent_key else k
# #         if isinstance(v, dict):
# #             items.extend(flatten_dict(v, new_key, sep=sep).items())
# #         else:
# #             items.append((new_key, v))
# #     return dict(items)


# # # ------------------------------ Bedrock Embedding Client ------------------------------

# # class BedrockEmbeddingClient:
# #     """
# #     Minimal Bedrock embedder using boto3. Supports optional STS assume-role.
# #     Default model: amazon.titan-embed-text-v2:0 (1024-dim).
# #     """
# #     def __init__(self):
# #         load_dotenv()

# #         self.region = os.getenv("BEDROCK_REGION", "us-east-1")
# #         self.role_arn = os.getenv("BEDROCK_ROLE_ARN")  # optional
# #         self.model_id = os.getenv("BEDROCK_EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0")

# #         # If role provided, assume it; else rely on ambient creds / env / profile
# #         if self.role_arn:
# #             # Optional: env keys; if not set, boto3 will use default/instance profile
# #             sts_client = boto3.client(
# #                 "sts",
# #                 region_name=self.region,
# #                 aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID") or None,
# #                 aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY") or None,
# #                 aws_session_token=os.getenv("AWS_SESSION_TOKEN") or None,
# #             )
# #             assumed = sts_client.assume_role(
# #                 RoleArn=self.role_arn,
# #                 RoleSessionName=f"BedrockEmbeddingsSession-{uuid.uuid4().hex[:8]}"
# #             )
# #             creds = assumed["Credentials"]
# #             self.client = boto3.client(
# #                 "bedrock-runtime",
# #                 region_name=self.region,
# #                 aws_access_key_id=creds["AccessKeyId"],
# #                 aws_secret_access_key=creds["SecretAccessKey"],
# #                 aws_session_token=creds["SessionToken"],
# #                 verify=False,
                
# #             )
# #         else:
# #             # Use default credentials chain (env, config/credentials, role on EC2/ECS, SSO, etc.)
# #             self.client = boto3.client("bedrock-runtime", region_name=self.region)

# #     def embed(self, text: str) -> List[float]:
# #         """
# #         Generate a single embedding for `text`.
# #         Titan v2 request uses {"inputText": "..."} and returns {"embedding": [float,...]}.
# #         """
# #         payload = {"inputText": text}
# #         resp = self.client.invoke_model(
# #             modelId=self.model_id,
# #             contentType="application/json",
# #             accept="application/json",
# #             body=json.dumps(payload),
# #         )
# #         body = resp["body"].read()
# #         parsed = json.loads(body)
# #         # Titan v2 returns "embedding"
# #         vec = parsed.get("embedding") or parsed.get("embeddings") or []
# #         # Some providers return {"embeddings": {"values":[...]}}; normalize if needed
# #         if isinstance(vec, dict) and "values" in vec:
# #             vec = vec["values"]
# #         if not isinstance(vec, list):
# #             raise ValueError(f"Unexpected embedding format from Bedrock: {type(vec)}")
# #         return [float(x) for x in vec]


# # # ------------------------------ Structure-Preserving Parser ------------------------------

# # class StructurePreservingDocumentParser:
# #     """Advanced document parser that preserves and creates markdown structure
# #     input: 
# #         GENERAL TERMS AND CONDITIONS

# #         1. Introduction
# #         This agreement sets forth the terms.

# #         1.1 Scope
# #         The agreement applies to...

# #         A. Special Provisions
# #         These apply only in some cases.

# #         - Parties must comply
# #         - Terms are binding

# #         output:
# #         ## GENERAL TERMS AND CONDITIONS

# #         ## 1 Introduction
# #         This agreement sets forth the terms.

# #         ### 1.1 Scope
# #         The agreement applies to...

# #         ## A. Special Provisions
# #         These apply only in some cases.

# #         - Parties must comply
# #         - Terms are binding


# # """

# #     def __init__(self):
# #         self.title_patterns = [
# #             re.compile(r'^([A-Z][A-Z\s&/\-]+[A-Z])$', re.MULTILINE),  # ALL CAPS titles
# #             re.compile(r'^([A-Z][^.!?]*[A-Z])$', re.MULTILINE),       # Title case-ish
# #         ]
# #         self.section_patterns = [
# #             re.compile(r'^(\d+(?:\.\d+)*)\s*[:\-\.]?\s*(.+)$', re.MULTILINE),  # 1.2.3 Section Name
# #             re.compile(r'^([A-Z])\.\s*(.+)$', re.MULTILINE),                   # A. Section Name
# #             re.compile(r'^\(([a-z]+)\)\s*(.+)$', re.MULTILINE),                # (a) Subsection
# #         ]
# #         self.list_patterns = [
# #             re.compile(r'^\s*[\-\*\+]\s+(.+)$', re.MULTILINE),                 # - List item
# #             re.compile(r'^\s*\d+\.\s+(.+)$', re.MULTILINE),                    # 1. Numbered list
# #         ]

# #     def parse_pdf_to_markdown(self, file_path: str) -> str:
# #         """Parse PDF and convert to structured markdown"""
# #         try:
# #             if not PDF_AVAILABLE or PdfReader is None:
# #                 raise ImportError("PyPDF2 not available")

# #             reader = PdfReader(file_path)
# #             markdown_content: List[str] = []

# #             # Extract title from filename or first page
# #             file_name = os.path.splitext(os.path.basename(file_path))[0]
# #             title = file_name.replace('_', ' ').replace('-', ' ').title()
# #             markdown_content.append(f"# {title}\n")

# #             page_num = 0
# #             for page in reader.pages:
# #                 page_num += 1
# #                 text = page.extract_text() or ""
# #                 text = text.replace("\x00", " ")
# #                 text = normalize_ws(text)
# #                 if not text.strip():
# #                     continue

# #                 # Add page marker
# #                 markdown_content.append(f"\n<!-- Page {page_num} -->\n")

# #                 # Process text line by line to preserve structure
# #                 lines = text.split('\n')
# #                 processed_lines = self._structure_text_lines(lines)
# #                 markdown_content.extend(processed_lines)

# #             return '\n'.join(markdown_content)

# #         except Exception as e:
# #             logging.warning(f"Failed to parse PDF structure for {file_path}: {e}")
# #             # Fallback to regular loading
# #             try:
# #                 loader = PyPDFLoader(file_path)
# #                 docs = loader.load()
# #                 return '\n'.join([normalize_ws(doc.page_content) for doc in docs])
# #             except Exception as e2:
# #                 logging.error(f"Fallback PDF load failed for {file_path}: {e2}")
# #                 return ""

# #     def _structure_text_lines(self, lines: List[str]) -> List[str]:
# #         """Apply structure detection to text lines"""
# #         structured_lines = []
# #         for line in lines:
# #             line = line.strip()
# #             if not line:
# #                 continue
# #             structured_line = self._apply_structure_detection(line)
# #             structured_lines.append(structured_line)
# #         return structured_lines

# #     def _apply_structure_detection(self, text: str) -> str:
# #         """Apply structure detection patterns to text"""

# #         # Check for section patterns
# #         for pattern in self.section_patterns:
# #             match = pattern.match(text)
# #             if match and len(match.groups()) >= 2:
# #                 section_num = match.group(1)
# #                 section_title = match.group(2).strip()
# #                 # Determine heading level based on section number depth
# #                 if '.' in section_num:
# #                     depth = section_num.count('.') + 2
# #                     depth = min(depth, 6)  # Max heading level in markdown
# #                 else:
# #                     depth = 2
# #                 return f"{'#' * depth} {section_num} {section_title}"

# #         # Check for list patterns
# #         for pattern in self.list_patterns:
# #             match = pattern.match(text)
# #             if match:
# #                 return f"- {match.group(1)}"

# #         # Check for title patterns (ALL CAPS or Title Case)
# #         for pattern in self.title_patterns:
# #             if pattern.match(text) and 10 < len(text) < 120:
# #                 return f"## {text}"

# #         # Regular paragraph
# #         return text


# # # ------------------------------ Classifier & Field Extraction ------------------------------

# # class ContractClassifierExtractor:
# #     """
# #     Detect document type and extract fields based on robust regex & heading parsing.
# #     """

# #     # Label aliases per field (case-insensitive)
# #     SOW_ALIASES = {
# #         # Names / parties
# #         "SOW Name": [
# #             r"Statement\s+of\s+Work\s+Name", r"SOW\s*Name", r"Project\s*Name", r"GEHC\s*Project\s*Name"
# #         ],
# #         "Supplier Name": [
# #             r"Supplier\s*Name", r"Vendor\s*Name", r"Party\s*B\s*Name"
# #         ],
# #         "Supplier Geography": [r"Supplier\s*Geograph\w*", r"Supplier\s*Location", r"Supplier\s*Country"],

# #         # Dates (plus narrative fallbacks handled in helpers)
# #         "SOW Effective Date": [r"SOW\s*Effective\s*Date", r"Effective\s*Date"],
# #         "SOW Start Date": [r"SOW\s*Start\s*Date", r"Start\s*Date\s*of\s*SOW"],
# #         "SOW End Date":   [r"SOW\s*End\s*Date", r"End\s*date(?:\s*of\s*SOW)?"],
# #         "effect until": [
# #             r"Effective\s*until", r"End\s*Date", r"Expiry\s*Date", r"Expiration\s*Date", r"Term\s*End(?:s)?\s*On"
# #         ],

# #         # Money
# #         "Fee Amount": [
# #             r"Fee\s*Amount", r"Total\s*Fee", r"Total\s*Cost", r"Contract\s*Value", r"Contract\s*Amount"
# #         ],
# #         "Fee Basis - Fixed/TnM": [
# #             r"Fee\s*Basis", r"Pricing\s*Model", r"Payment\s*Basis",
# #             r"Fixed\s*Price|Time\s*and\s*Materials|T[&\s]*M"
# #         ],

# #         # Contacts / PMs
# #         "GEHC Project Name": [r"GEHC\s*Project\s*Name"],
# #         "GEHC PM Phone":  [r"GEHC\s*PM\s*Phone", r"GEHC\s*Project\s*Manager\s*Phone"],
# #         "GEHC PM Email":  [r"GEHC\s*PM\s*Email", r"GEHC\s*Project\s*Manager\s*Email"],
# #         "Supplier PM":    [r"Supplier\s*PM", r"Supplier\s*Project\s*Manager", r"Supplier\s*Contact\s*Name", r"TCS\s*PM"],
# #         "Supplier PM Phone": [r"Supplier\s*PM\s*Phone", r"Supplier\s*Contact\s*Phone", r"TCS\s*PM\s*Phone"],
# #         "Supplier Email": [r"Supplier\s*Email", r"Supplier\s*PM\s*Email", r"Supplier\s*Contact\s*Email", r"TCS\s*PM\s*Email"],

# #         # Sections / misc
# #         "Invoices": [r"Invoices?", r"Invoicing\s*Instructions"],
# #         "Location": [r"Location", r"Service\s*Location", r"Place\s*of\s*Performance"],

# #         # Signatures
# #         "GEHC Signed Party":    [r"GEHC\s*Signed\s*Party", r"For\s+GEHC", r"Signed\s*for\s+GEHC"],
# #         "Supplier Signed Party":[r"Supplier\s*Signed\s*Party", r"For\s+Supplier", r"Signed\s*for\s+Supplier", r"For\s+TCS"],
# #     }

# #     # Section-style fields whose content can be multiple lines/paragraphs
# #     SOW_SECTION_FIELDS = [
# #         "EXECUTIVE SUMMARY", "Services", "Deliverables", "General Assumption",
# #         "Out of Scope", "SLA", "Invoices", "FEES", "INVOICES", "LOCATION"
# #     ]

# #     CR_ALIASES = {
# #         "SOW Name": [r"SOW\s*Name", r"Project\s*Name"],
# #         "Change Request Start Date": [r"Change\s*Request\s*Start\s*Date", r"CR\s*Start\s*Date", r"Start\s*Date"],
# #         "Change Request End Date": [r"Change\s*Request\s*End\s*Date", r"CR\s*End\s*Date", r"End\s*Date"],
# #         "Supplier Name": [r"Supplier\s*Name", r"Vendor\s*Name", r"TCS"],
# #         "Supplier Geography": [r"Supplier\s*Geograph\w*", r"Supplier\s*Location", r"Supplier\s*Country"],
# #         "Date Submitted": [r"Date\s*Submitted", r"Submission\s*Date"],
# #         "Change To: Delivery/Cost": [r"Change\s*To[:\s-]*(?:Delivery|Cost|Delivery\s*/\s*Cost)?", r"Change\s*Description", r"Scope\s*Change"],
# #         "Revised SOW Amount": [r"Revised\s*SOW\s*Amount", r"Revised\s*Contract\s*Value", r"New\s*Total\s*Amount"],
# #         "Original SOW Amount": [r"Original\s*SOW\s*Amount", r"Original\s*Contract\s*Value"],
# #         "Invoices": [r"Invoices?", r"Invoicing\s*Instructions"],
# #         "Timing Impact": [r"Timing\s*Impact", r"Schedule\s*Impact", r"Timeline\s*Impact"],
# #         "Date Response Delivered": [r"Date\s*Response\s*Delivered", r"Response\s*Date"],
# #         "GEHC Signed Party": [r"GEHC\s*Signed\s*Party", r"For\s+GEHC", r"Signed\s*for\s+GEHC"],
# #         "Supplier Signed Party": [r"Supplier\s*Signed\s*Party", r"For\s+Supplier", r"Signed\s*for\s*Supplier", r"For\s+TCS"],
# #     }

# #     AM_ALIASES = {
# #         "Ammendment Date": [r"Amendment\s*Date", r"Ammendment\s*Date"],
# #         "Parties": [r"Parties", r"Between", r"This\s*Amendment\s*is\s*made\s*by\s*and\s*between"],
# #         "Effective Date": [r"Effective\s*Date", r"Commencement\s*Date"],
# #         "Total Cost": [r"Total\s*Cost", r"Total\s*Amount", r"Amended\s*Contract\s*Value"],
# #         "Payment Schedule": [r"Payment\s*Schedule", r"Milestone\s*Payments?", r"Payment\s*Terms?"],
# #         "Supplier Geography": [r"Supplier\s*Geograph\w*", r"Supplier\s*Location", r"Supplier\s*Country"],
# #         "Supplier Name": [r"Supplier\s*Name", r"Vendor\s*Name", r"Party\s*B\s*Name", r"TCS"],
# #     }

# #     # Heading detection: support A./B./C. prefixes, optional trailing dot, case-insensitive
# #     HEADING_LINE = re.compile(
# #         r'^(?:#{1,6}\s*)?(?:[A-Z]{1,4}\s*[.)]\s*)?([A-Z][A-Z0-9 &/\-,]+?)\s*\.?\s*$',
# #         flags=re.MULTILINE
# #     )

# #     def detect_type(self, text: str) -> DocType:
# #         t = text.lower()
# #         if re.search(r'\bchange\s+request\b', t) or re.search(r'\bcr\s*#:?\b', t):
# #             return DocType.CHANGE_REQUEST
# #         if re.search(r'\bstatement\s+of\s+work\b', t) or re.search(r'\bsow\b', t):
# #             return DocType.SOW
# #         if re.search(r'\bamendment\b', t) or re.search(r'\bammendment\b', t):
# #             return DocType.AMENDMENT
# #         # More heuristics
# #         if re.search(r'\bchange\s*request\s*start\s*date\b', t):
# #             return DocType.CHANGE_REQUEST
# #         if re.search(r'\beffective\s*date\b', t) and re.search(r'\bamendment\b', t):
# #             return DocType.AMENDMENT
# #         return DocType.UNKNOWN

# #     # ---------- generic KV extraction ----------

# #     def _extract_kv(self, text: str, label_patterns: List[str], multiline: bool = False) -> Optional[str]:
# #         """
# #         Find value for any of the candidate labels. Accept "Label: value", "Label - value",
# #         en/em dash, or "Label" on one line and value on the next line.
# #         """
# #         for lp in label_patterns:
# #             # direct "Label: value" (various dashes)
# #             pattern_inline = re.compile(rf'(?im)^\s*{lp}\s*[:\-–—]\s*(.+)$')
# #             m = pattern_inline.search(text)
# #             if m:
# #                 return m.group(1).strip()

# #             # "Label" newline "value"
# #             pattern_nextline = re.compile(rf'(?im)^\s*{lp}\s*\.?\s*$\s*\n\s*(.+)$')
# #             m2 = pattern_nextline.search(text)
# #             if m2:
# #                 return m2.group(1).strip()

# #             # inline anywhere (less strict)
# #             pattern_any = re.compile(rf'(?i){lp}\s*[:\-–—]\s*([^\n]+)')
# #             m3 = pattern_any.search(text)
# #             if m3:
# #                 return m3.group(1).strip()
# #         return None

# #     def _extract_date(self, text: str, label_patterns: List[str]) -> Optional[str]:
# #         val = self._extract_kv(text, label_patterns)
# #         if val:
# #             return try_parse_date(val)

# #         # Narrative fallbacks: "dated <DATE> (“SOW Effective Date”)"
# #         m = re.search(r'(?is)\bdated\s+([A-Za-z0-9,\- ]{5,40})\s*\(\s*[“"]?SOW\s*Effective\s*Date', text)
# #         if m:
# #             return try_parse_date(m.group(1).strip())
# #         return None

# #     # ---------- section extraction ----------

# #     def _extract_section(self, text: str, section_name: str, max_chars: int = 6000) -> Optional[str]:
# #         """
# #         Extract section content after a heading like "A. EXECUTIVE SUMMARY" or uppercase line.
# #         Stop at next heading-like line. Fallback: take a few paragraphs after the matched label.
# #         """
# #         # 1) Prefer markdown/uppercase heading with optional A./B. prefixes
# #         head_regex = re.compile(
# #             rf'(?im)^(?:[A-Z]{{1,4}}\s*[.)]\s*)?{re.escape(section_name)}\s*:?\s*\.?\s*$'
# #         )
# #         m = head_regex.search(text)
# #         if m:
# #             start = m.end()
# #             # find next heading after 'start'
# #             nxt = None
# #             for m2 in self.HEADING_LINE.finditer(text, pos=start):
# #                 nxt = m2.start()
# #                 break
# #             chunk = text[start:nxt] if nxt else text[start:]
# #             return normalize_ws(chunk)[:max_chars].strip() or None

# #         # 2) Fallback: look for label inline and take next ~60 lines (tables)
# #         inline = re.compile(rf'(?i){re.escape(section_name)}\s*:?\s*$', re.MULTILINE)
# #         m2 = inline.search(text)
# #         if m2:
# #             start = m2.end()
# #             tail = text[start:]
# #             lines = tail.splitlines()
# #             grabbed = []
# #             for ln in lines[:60]:
# #                 if self.HEADING_LINE.match(ln.strip()):
# #                     break
# #                 grabbed.append(ln)
# #             out = normalize_ws('\n'.join(grabbed))[:max_chars].strip()
# #             return out or None
# #         return None

# #     def _extract_many_sections(self, text: str, names: List[str]) -> Dict[str, Optional[str]]:
# #         return {name: self._extract_section(text, name) for name in names}

# #     # ---------- narrative & specialized helpers ----------

# #     def _extract_effect_until(self, text: str) -> Optional[str]:
# #         """Capture narrative end date: 'shall remain in effect until <DATE>'"""
# #         m = re.search(r'(?i)\b(?:shall\s+)?remain(?:s)?\s+in\s+effect\s+until\s+([A-Za-z0-9,\- ]{5,40})', text)
# #         if m:
# #             return try_parse_date(m.group(1).strip())
# #         return None

# #     def _extract_fee_amount(self, text: str) -> Optional[str]:
# #         """Find fee amount from FEES section or narrative phrases (Fixed Fee/Amount of/Total)."""
# #         fees_block = self._extract_section(text, "FEES")
# #         src = fees_block or text
# #         pat = re.compile(
# #             r'(?is)\b(?:Fixed\s+(?:Fee|Price)\b[^$]{0,80}?|Amount(?:\s+of)?\b[^$]{0,40}?|Total\b[^$]{0,40}?)(?:USD|US\$|€|£|₹|\$)?\s*\$?\s*([0-9][\d,]*(?:\.\d{2})?)'
# #         )
# #         m = pat.search(src)
# #         return m.group(1).replace(',', '') if m else None

# #     def _extract_title_project_name(self, text: str) -> Optional[str]:
# #         """Fallback SOW name from document title/paragraphs."""
# #         # STATEMENT OF WORK for <Name>
# #         m1 = re.search(r'(?im)^\s*STATEMENT\s+OF\s+WORK\s+for\s+(.+?)\s*$', text)
# #         if m1:
# #             return m1.group(1).strip()
# #         # “This Statement of Work (“SOW”) for “<Name>” …”
# #         m2 = re.search(r'(?is)Statement\s+of\s+Work\s*\(.*?\)\s*for\s*[“"]([^”"]+)[”"]', text)
# #         if m2:
# #             return m2.group(1).strip()
# #         return None

# #     def _extract_parties_from_opening(self, text: str) -> Dict[str, Optional[str]]:
# #         """Capture GEHC and Supplier legal names from opening 'made by and between' clause."""
# #         out = {"GEHC": None, "Supplier": None}
# #         m = re.search(
# #             r'(?is)made\s+by\s+and\s+between\s+([A-Z][A-Za-z0-9 .,&\-()]+?)\s*\(.*?(?:GEHC|Customer).*?\)\s*and\s+([A-Z][A-Za-z0-9 .,&\-()]+?)\s*\(.*?(?:TCS|SP|Supplier).*?\)',
# #             text
# #         )
# #         if m:
# #             out["GEHC"] = m.group(1).strip()
# #             out["Supplier"] = m.group(2).strip()
# #         return out

# #     def _extract_signature_printed_names(self, text: str) -> Dict[str, Optional[str]]:
# #         """
# #         Capture DocuSign-style signature blocks:
# #         'GE Medical Systems Limited ... Printed Name: <name>' , 'Tata Consultancy Services Limited ... Printed Name: <name>'
# #         """
# #         res = {"GEHC Signed Party": None, "Supplier Signed Party": None}
# #         tail = text[-4000:] if len(text) > 4000 else text

# #         def printed_for(company_pat: str) -> Optional[str]:
# #             comp = re.search(company_pat, tail, flags=re.IGNORECASE)
# #             if not comp:
# #                 return None
# #             seg = tail[comp.end():comp.end()+1200]
# #             m = re.search(r'(?i)Printed\s+Name\s*[:\-–—]\s*([A-Z][A-Za-z .,\-]+)', seg)
# #             if m:
# #                 return m.group(1).strip()
# #             return None

# #         gehc_name = printed_for(r'GE\s*(?:PRECISION\s+HEALTHCARE|Medical\s+Systems\s+Limited|Healthcare)')
# #         supplier_name = printed_for(r'Tata\s+Consultancy\s+Services\s+Limited|TCS')

# #         if gehc_name:
# #             res["GEHC Signed Party"] = gehc_name
# #         if supplier_name:
# #             res["Supplier Signed Party"] = supplier_name
# #         return res

# #     def _extract_locations_from_section(self, text: str) -> Optional[str]:
# #         """Collect bullet/row items from LOCATION/M. LOCATION section."""
# #         loc = self._extract_section(text, "LOCATION")
# #         if not loc:
# #             loc = self._extract_section(text, "M. LOCATION")
# #         if not loc:
# #             return None
# #         items = re.findall(r'(?m)^\s*(?:\d+\.|[-•\u2022])?\s*([A-Z][A-Za-z0-9 .,&\-()]+)$', loc)
# #         if items:
# #             # unique while preserving order
# #             return ", ".join(dict.fromkeys([normalize_ws(i) for i in items]))
# #         return normalize_ws(loc)

# #     def _extract_signed_party(self, text: str, for_who: str) -> Optional[str]:
# #         """
# #         Try to extract a name after phrases like "Signed for GEHC" or "For Supplier".
# #         """
# #         pats = [
# #             rf'(?is)Signed\s*(?:By|for)\s*{for_who}[:\s\-]*([A-Z][A-Za-z .,\-]+)',
# #             rf'(?is)For\s+{for_who}\s*[:\-]*\s*([A-Z][A-Za-z .,\-]+)',
# #         ]
# #         for p in pats:
# #             m = re.search(p, text, flags=re.IGNORECASE)
# #             if m:
# #                 val = m.group(1).strip()
# #                 # stop at linebreak/role
# #                 val = re.split(r'[\n\r]|Title\s*:|Designation\s*:', val)[0].strip()
# #                 return val
# #         return None

# #     # ---------- SOW extraction ----------

# #     def extract_sow(self, text: str) -> Dict[str, Any]:
# #         a = self.SOW_ALIASES
# #         sections = self._extract_many_sections(text, self.SOW_SECTION_FIELDS)

# #         # Fee Basis can be "Fixed" or "T&M" hidden in narrative; try a generic detection too
# #         fee_basis = self._extract_kv(text, a["Fee Basis - Fixed/TnM"])
# #         if not fee_basis:
# #             if re.search(r'\bFixed\s*Price\b', text, flags=re.IGNORECASE):
# #                 fee_basis = "Fixed Price"
# #             elif re.search(r'\bT\s*&?\s*M\b|\bTime\s*and\s*Materials\b', text, flags=re.IGNORECASE):
# #                 fee_basis = "Time & Materials"

# #         # Parties (opening clause)
# #         parties = self._extract_parties_from_opening(text)

# #         # SOW name with fallbacks
# #         sow_name = self._extract_kv(text, a["SOW Name"]) or self._extract_title_project_name(text)

# #         # Dates
# #         sow_effective = self._extract_date(text, a["SOW Effective Date"])
# #         sow_start = self._extract_date(text, a["SOW Start Date"])
# #         sow_end = self._extract_date(text, a["SOW End Date"]) or self._extract_effect_until(text)

# #         # Fees
# #         fee_amount = self._extract_kv(text, a["Fee Amount"]) or self._extract_fee_amount(text)

# #         # Managers / contacts
# #         gehc_pm_phone = self._extract_kv(text, a["GEHC PM Phone"])
# #         gehc_pm_email = self._extract_kv(text, a["GEHC PM Email"])
# #         supplier_pm = self._extract_kv(text, a["Supplier PM"])
# #         supplier_pm_phone = self._extract_kv(text, a["Supplier PM Phone"])
# #         supplier_email = self._extract_kv(text, a["Supplier Email"])

# #         # Locations
# #         location_val = self._extract_kv(text, a["Location"]) or self._extract_locations_from_section(text)

# #         # Signatures – prefer DocuSign block, fallback to "Signed for ..." phrases
# #         sigs = self._extract_signature_printed_names(text)
# #         gehc_signed = sigs.get("GEHC Signed Party") or self._extract_kv(text, a["GEHC Signed Party"]) or self._extract_signed_party(text, "GEHC")
# #         supplier_signed = sigs.get("Supplier Signed Party") or self._extract_kv(text, a["Supplier Signed Party"]) or self._extract_signed_party(text, "Supplier") or self._extract_signed_party(text, "TCS")

# #         out = {
# #             "SOW Name": sow_name,
# #             "Supplier Name": self._extract_kv(text, a["Supplier Name"]) or parties.get("Supplier"),
# #             "Supplier Geography": self._extract_kv(text, a["Supplier Geography"]),
# #             "EXECUTIVE SUMMARY": sections.get("EXECUTIVE SUMMARY"),
# #             "Services": sections.get("Services"),
# #             "Deliverables": sections.get("Deliverables"),
# #             "SOW Effective Date": sow_effective,
# #             "SOW Start Date": sow_start,
# #             "SOW End Date": sow_end,
# #             "effect until": sow_end,  # preserve legacy key
# #             "Fee Amount": fee_amount,
# #             "Fee Basis - Fixed/TnM": fee_basis,
# #             "Invoices": sections.get("INVOICES") or sections.get("Invoices") or self._extract_kv(text, a["Invoices"]),
# #             "GEHC Project Name": self._extract_kv(text, a["GEHC Project Name"]) or sow_name,
# #             "GEHC PM Phone": gehc_pm_phone,
# #             "GEHC PM Email": gehc_pm_email,
# #             "Supplier PM": supplier_pm,
# #             "Supplier PM Phone": supplier_pm_phone,
# #             "Supplier Email": supplier_email,
# #             "General Assumption": sections.get("General Assumption"),
# #             "Out of Scope": sections.get("Out of Scope"),
# #             "SLA": sections.get("SLA"),
# #             "Location": location_val,
# #             "GEHC Signed Party": gehc_signed,
# #             "Supplier Signed Party": supplier_signed,
# #         }
# #         # Light normalization for amounts (strip currency symbols but keep original if parse fails)
# #         def norm_amount(x: Optional[str]) -> Optional[str]:
# #             if not x:
# #                 return x
# #             v = x.strip()
# #             v = re.sub(r'[,$₹€£]', '', v)
# #             v = re.sub(r'\s{2,}', ' ', v)
# #             return v

# #         for k in ["Fee Amount", "Revised SOW Amount", "Original SOW Amount", "Total Cost"]:
# #             if k in out:
# #                 out[k] = norm_amount(out.get(k))
# #         return out

# #     # ---------- Change Request extraction ----------

# #     def extract_change_request(self, text: str) -> Dict[str, Any]:
# #         a = self.CR_ALIASES
# #         out = {
# #             "SOW Name": self._extract_kv(text, a["SOW Name"]),
# #             "Change Request Start Date": self._extract_date(text, a["Change Request Start Date"]),
# #             "Change Request End Date": self._extract_date(text, a["Change Request End Date"]),
# #             "Supplier Name": self._extract_kv(text, a["Supplier Name"]),
# #             "Supplier Geography": self._extract_kv(text, a["Supplier Geography"]),
# #             "Date Submitted": self._extract_date(text, a["Date Submitted"]),
# #             "Change To: Delivery/Cost": self._extract_kv(text, a["Change To: Delivery/Cost"]),
# #             "Revised SOW Amount": self._extract_kv(text, a["Revised SOW Amount"]),
# #             "Original SOW Amount": self._extract_kv(text, a["Original SOW Amount"]),
# #             "Invoices": self._extract_kv(text, a["Invoices"]),
# #             "Timing Impact": self._extract_kv(text, a["Timing Impact"]),
# #             "Date Response Delivered": self._extract_date(text, a["Date Response Delivered"]),
# #             "GEHC Signed Party": self._extract_kv(text, a["GEHC Signed Party"]) or self._extract_signed_party(text, "GEHC"),
# #             "Supplier Signed Party": self._extract_kv(text, a["Supplier Signed Party"]) or self._extract_signed_party(text, "Supplier"),
# #         }
# #         return out

# #     # ---------- Amendment extraction ----------

# #     def extract_amendment(self, text: str) -> Dict[str, Any]:
# #         a = self.AM_ALIASES
# #         out = {
# #             "Ammendment Date": self._extract_date(text, a["Ammendment Date"]),
# #             "Parties": self._extract_kv(text, a["Parties"]),
# #             "Effective Date": self._extract_date(text, a["Effective Date"]),
# #             "Total Cost": self._extract_kv(text, a["Total Cost"]),
# #             "Payment Schedule": self._extract_kv(text, a["Payment Schedule"]),
# #             "Supplier Geography": self._extract_kv(text, a["Supplier Geography"]),
# #             "Supplier Name": self._extract_kv(text, a["Supplier Name"]),
# #         }
# #         return out

# #     # ---------- Unified helpers ----------

# #     def classify_and_extract(self, markdown_text: str) -> Tuple[DocType, Dict[str, Any]]:
# #         # Clean / normalize text for extraction
# #         raw = normalize_ws(clean_markdown_noise(markdown_text))

# #         doc_type = self.detect_type(raw)
# #         fields: Dict[str, Any] = {}

# #         if doc_type == DocType.SOW:
# #             fields = self.extract_sow(raw)
# #         elif doc_type == DocType.CHANGE_REQUEST:
# #             fields = self.extract_change_request(raw)
# #         elif doc_type == DocType.AMENDMENT:
# #             fields = self.extract_amendment(raw)
# #         else:
# #             # Unknown – try best-effort guess: prioritize SOW-like fields
# #             fields = self.extract_sow(raw)

# #         # Light normalization for amounts (strip currency symbols but keep original if parse fails)
# #         def norm_amount(x: Optional[str]) -> Optional[str]:
# #             if not x:
# #                 return x
# #             v = x.strip()
# #             v = re.sub(r'[,$₹€£]', '', v)
# #             v = re.sub(r'\s{2,}', ' ', v)
# #             return v

# #         for k in ["Fee Amount", "Revised SOW Amount", "Original SOW Amount", "Total Cost"]:
# #             if k in fields:
# #                 fields[k] = norm_amount(fields.get(k))

# #         return doc_type, fields

# #     def extract_by_type(self, markdown_text: str, doc_type: DocType) -> Dict[str, Any]:
# #         """
# #         Extract using a known/selected doc_type without re-detecting.
# #         """
# #         raw = normalize_ws(clean_markdown_noise(markdown_text))
# #         if doc_type == DocType.SOW:
# #             return self.extract_sow(raw)
# #         elif doc_type == DocType.CHANGE_REQUEST:
# #             return self.extract_change_request(raw)
# #         elif doc_type == DocType.AMENDMENT:
# #             return self.extract_amendment(raw)
# #         else:
# #             return self.extract_sow(raw)  # best-effort


# # # ------------------------------ Document Processor ------------------------------

# # class DocumentProcessor:
# #     def __init__(self, output_folder: str = "output"):
# #         load_dotenv()

# #         # Initialize Qdrant client
# #         self.qdrant_host = os.getenv("QDRANT_HOST", "localhost")
# #         self.qdrant_port = int(os.getenv("QDRANT_PORT", 6333))
# #         self.collection_name = os.getenv("QDRANT_COLLECTION_NAME", "documents_collection")

# #         self.client = QdrantClient(host=self.qdrant_host, port=self.qdrant_port)

# #         # Initialize Bedrock embedding client (replaces HuggingFaceEmbeddings)
# #         self.embedder = BedrockEmbeddingClient()

# #         # Initialize text splitter (kept for potential future chunking)
# #         self.text_splitter = RecursiveCharacterTextSplitter(
# #             chunk_size=1000,
# #             chunk_overlap=350,
# #             length_function=len,
# #         )

# #         # LLM client
# #         self.llm_client = LLMClient()

# #         # Parser and classifier/extractor
# #         self.document_parser = StructurePreservingDocumentParser()
# #         self.contract_extractor = ContractClassifierExtractor()

# #         # Output folders
# #         self.output_folder = output_folder
# #         os.makedirs(output_folder, exist_ok=True)
# #         self.markdown_folder = os.path.join(output_folder, "markdown")
# #         self.chunks_folder = os.path.join(output_folder, "chunks")
# #         os.makedirs(self.markdown_folder, exist_ok=True)
# #         os.makedirs(self.chunks_folder, exist_ok=True)

# #         # Ensure collection exists
# #         self._ensure_collection_exists()

# #     def _ensure_collection_exists(self):
# #         """Initialize Qdrant collection if it doesn't exist"""
# #         try:
# #             collections = self.client.get_collections()
# #             collection_names = [col.name for col in collections.collections]

# #             # Titan v2 = 1024 dims
# #             vector_size = int(os.getenv("QDRANT_VECTOR_SIZE", "1024"))

# #             if self.collection_name not in collection_names:
# #                 self.client.create_collection(
# #                     collection_name=self.collection_name,
# #                     vectors_config=VectorParams(
# #                         size=vector_size,
# #                         distance=Distance.COSINE
# #                     )
# #                 )
# #                 print(f"Created collection: {self.collection_name} (size={vector_size})")
# #             else:
# #                 print(f"Collection {self.collection_name} already exists")
# #         except Exception as e:
# #             print(f"Error setting up Qdrant collection: {e}")

# #     def extract_text_from_pdf(self, file_path: str) -> str:
# #         """Extract structured text from PDF using advanced parser"""
# #         try:
# #             # Use structure-preserving parser
# #             structured_text = self.document_parser.parse_pdf_to_markdown(file_path)
# #             return structured_text
# #         except Exception as e:
# #             print(f"Error extracting text from {file_path}: {e}")
# #             # Fallback to basic PyPDF loader
# #             try:
# #                 loader = PyPDFLoader(file_path)
# #                 docs = loader.load()
# #                 return '\n'.join([normalize_ws(doc.page_content) for doc in docs])
# #             except Exception as fallback_error:
# #                 print(f"Fallback extraction also failed: {fallback_error}")
# #                 return ""

# #     def pdf_to_markdown(self, pdf_path: str) -> str:
# #         """Convert PDF to structured markdown using advanced parser"""
# #         markdown_content = self.document_parser.parse_pdf_to_markdown(pdf_path)
# #         # Save markdown to output folder
# #         self._save_markdown(pdf_path, markdown_content)
# #         return markdown_content

# #     def create_document_chunk(self, file_path: str) -> Optional[Dict[str, Any]]:
# #         """Create a single chunk (full doc) and extract typed metadata"""
# #         try:
# #             markdown_content = self.pdf_to_markdown(file_path)
# #             if not markdown_content:
# #                 return None

# #             print("Detecting document type & extracting fields...")

# #             # Cleaned text (for heuristics)
# #             cleaned = normalize_ws(clean_markdown_noise(markdown_content))

# #             # (A) LLM classification
# #             llm_doc_type_str, llm_conf, llm_reasons = ("UNKNOWN", 0.0, [])
# #             try:
# #                 llm_doc_type_str, llm_conf, llm_reasons = self.llm_client.classify_document_type(markdown_content)
# #             except Exception as e:
# #                 logging.warning(f"LLM doc-type classification failed: {e}")

# #             # (B) Heuristic detection (existing)
# #             heur_type = self.contract_extractor.detect_type(cleaned)

# #             # (C) Final decision rule:
# #             # - Prefer LLM when confident enough and not UNKNOWN
# #             # - Otherwise fall back to heuristic
# #             final_doc_type = heur_type
# #             try:
# #                 if llm_doc_type_str != "UNKNOWN" and llm_conf >= 0.55:
# #                     final_doc_type = DocType[llm_doc_type_str]
# #                 elif heur_type == DocType.UNKNOWN and llm_doc_type_str in {"SOW", "CHANGE_REQUEST", "AMENDMENT"}:
# #                     final_doc_type = DocType[llm_doc_type_str]
# #             except Exception as e:
# #                 logging.warning(f"DocType resolution fallback used: {e}")

# #             # (D) Extract with chosen type
# #             fields = self.contract_extractor.extract_by_type(markdown_content, final_doc_type)

# #             # Build metadata
# #             metadata = {
# #                 "doc_type": final_doc_type.value,
# #                 "doc_type_llm": llm_doc_type_str,
# #                 "doc_type_llm_confidence": llm_conf,
# #                 "doc_type_llm_reasons": llm_reasons,
# #                 "doc_type_heuristic": heur_type.value,
# #                 "extracted_fields": fields
# #             }

# #             chunk = {
# #                 "id": str(uuid.uuid4()),
# #                 "file_path": file_path,
# #                 "file_name": os.path.basename(file_path),
# #                 "content": markdown_content,
# #                 "content_type": "document",
# #                 "processed_date": datetime.now().isoformat(),
# #                 "metadata": metadata
# #             }
# #             return chunk

# #         except Exception as e:
# #             print(f"Error creating document chunk: {e}")
# #             return None

# #     def generate_embedding(self, text: str) -> List[float]:
# #         """Generate embedding for text using AWS Bedrock Titan Embeddings"""
# #         try:
# #             max_length = 8000  # keep conservative; Titan handles long input, but no need to overdo
# #             if len(text) > max_length:
# #                 text = text[:max_length]
# #             embedding = self.embedder.embed(text)
# #             return embedding
# #         except Exception as e:
# #             print(f"Error generating embedding: {e}")
# #             return []

# #     def _save_markdown(self, file_path: str, markdown_content: str):
# #         """Save markdown content to output folder"""
# #         try:
# #             file_name = os.path.basename(file_path)
# #             base_name = os.path.splitext(file_name)[0]
# #             markdown_file = os.path.join(self.markdown_folder, f"{base_name}.md")

# #             with open(markdown_file, 'w', encoding='utf-8') as f:
# #                 f.write(markdown_content)

# #             print(f"Saved markdown: {markdown_file}")

# #         except Exception as e:
# #             print(f"Error saving markdown for {file_path}: {e}")

# #     def _save_chunk_json(self, file_path: str, chunk: Dict[str, Any]):
# #         """Save chunk JSON to output folder"""
# #         try:
# #             file_name = os.path.basename(file_path)
# #             base_name = os.path.splitext(file_name)[0]

# #             # Save full chunk
# #             chunk_file = os.path.join(self.chunks_folder, f"{base_name}_chunk.json")
# #             with open(chunk_file, 'w', encoding='utf-8') as f:
# #                 json.dump(chunk, f, indent=2, ensure_ascii=False)

# #             # Create summary version with content preview
# #             chunk_summary = dict(chunk)  # shallow copy is fine
# #             content = chunk["content"] or ""
# #             chunk_summary["content_preview"] = (content[:500] + "...") if len(content) > 500 else content
# #             chunk_summary["content_length"] = len(content)

# #             # Save summary version
# #             summary_file = os.path.join(self.chunks_folder, f"{base_name}_summary.json")
# #             with open(summary_file, 'w', encoding='utf-8') as f:
# #                 json.dump(chunk_summary, f, indent=2, ensure_ascii=False)

# #             print(f"Saved chunk JSON: {chunk_file}")
# #             print(f"Saved chunk summary: {summary_file}")

# #         except Exception as e:
# #             print(f"Error saving chunk JSON for {file_path}: {e}")

# #     def store_to_qdrant(self, chunk: Dict[str, Any]) -> bool:
# #         """Store document chunk with embedding to Qdrant"""
# #         try:
# #             embedding = self.generate_embedding(chunk["content"])
# #             if not embedding:
# #                 print("Failed to generate embedding")
# #                 return False

# #             # Prepare payload
# #             payload = {
# #                 "file_path": chunk["file_path"],
# #                 "file_name": chunk["file_name"],
# #                 "content": chunk["content"],
# #                 "content_type": chunk["content_type"],
# #                 "processed_date": chunk["processed_date"],
# #                 "doc_type": chunk["metadata"].get("doc_type"),
# #                 "extracted_fields": chunk["metadata"].get("extracted_fields", {}),
# #             }

# #             # Also add flattened fields for easy filtering/query
# #             flat = flatten_dict({"extracted": payload["extracted_fields"]})
# #             payload.update(flat)

# #             point = PointStruct(
# #                 id=chunk["id"],
# #                 vector=embedding,
# #                 payload=payload
# #             )

# #             self.client.upsert(
# #                 collection_name=self.collection_name,
# #                 points=[point]
# #             )

# #             print(f"Successfully stored document: {chunk['file_name']}")
# #             return True

# #         except Exception as e:
# #             print(f"Error storing to Qdrant: {e}")
# #             return False

# #     def process_document(self, file_path: str):
# #         """Process a single document"""
# #         print(f"Processing document: {file_path}")

# #         # Create document chunk
# #         chunk = self.create_document_chunk(file_path)

# #         if chunk:
# #             # Save chunk JSON to output folder (only once here)
# #             self._save_chunk_json(file_path, chunk)

# #             # Store to Qdrant
# #             success = self.store_to_qdrant(chunk)

# #             if success:
# #                 print(f"Successfully processed: {file_path}")
# #             else:
# #                 print(f"Failed to store to Qdrant: {file_path}")
# #         else:
# #             print(f"Failed to create chunk for: {file_path}")

# #     def process_documents_directory(self, directory_path: str) -> None:
# #         """Process all PDF documents in a directory"""
# #         if not os.path.exists(directory_path):
# #             print(f"Directory not found: {directory_path}")
# #             return

# #         pdf_files = [f for f in os.listdir(directory_path) if f.lower().endswith('.pdf')]

# #         if not pdf_files:
# #             print(f"No PDF files found in: {directory_path}")
# #             return

# #         print(f"Found {len(pdf_files)} PDF files to process")

# #         for pdf_file in pdf_files:
# #             file_path = os.path.join(directory_path, pdf_file)
# #             self.process_document(file_path)
# #             print("-" * 50)


# # # ------------------------------ Main ------------------------------

# # def main():
# #     """Main function to process documents"""
# #     # Initialize processor with output folder
# #     output_folder = r"C:\Users\CSS\Desktop\Workspace\RAG_Coupa_chatbot\output2"
# #     processor = DocumentProcessor(output_folder=output_folder)

# #     # Process documents in the docs directory
# #     docs_directory = r"C:\Users\CSS\Desktop\Workspace\RAG_Coupa_chatbot\Docs"
# #     processor.process_documents_directory(docs_directory)

# #     print(f"\n📁 Output saved to: {output_folder}")
# #     print(f"  - Markdown files: {processor.markdown_folder}")
# #     print(f"  - Chunk JSON files: {processor.chunks_folder}")


# # if __name__ == "__main__":
# #     main()
# """
# Refactor: Replace regex-based field extraction with LLM JSON extraction (AWS Bedrock)

# What changed?
# - Added `LLMFieldExtractor` that prompts an LLM to extract a strict JSON object for SOW/CR/Amendment.
# - `DocumentProcessor.create_document_chunk()` now calls LLM-based extraction instead of regex patterns.
# - Kept a tiny `HeuristicTypeDetector` (regex-light) only for doc-type detection fallback.
# - Dates are normalized post‑LLM via `try_parse_date()`.

# Dependencies:
# - boto3 configured for Bedrock (env vars or instance role)
# - Set `BEDROCK_LLM_MODEL_ID` (e.g., "anthropic.claude-3-sonnet-20240229-v1:0")
# - Optional: `BEDROCK_REGION`, `BEDROCK_ROLE_ARN`

# This file is a drop‑in replacement for your previous module: it preserves existing
# class names where possible and clearly marks the LLM changes.
# """

# import os
# import re
# import json
# import uuid
# import logging
# from enum import Enum
# from datetime import datetime
# from typing import Any, Dict, List, Optional, Tuple

# from dotenv import load_dotenv
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import PyPDFLoader
# import boto3
# from qdrant_client import QdrantClient
# from qdrant_client.models import Distance, VectorParams, PointStruct

# # If you already have your own LLMClient for classification, you can still use it
# try:
#     from llm_client import LLMClient  # optional, for doc-type classification only
# except Exception:  # pragma: no cover
#     LLMClient = None  # type: ignore

# # ============================= Utils (unchanged) ============================= #

# class DocType(str, Enum):
#     SOW = "SOW"
#     CHANGE_REQUEST = "CHANGE_REQUEST"
#     AMENDMENT = "AMENDMENT"
#     UNKNOWN = "UNKNOWN"


# def normalize_ws(s: str) -> str:
#     s = s.replace("\u00A0", " ")
#     s = re.sub(r"[ \t]+", " ", s)
#     s = re.sub(r"\n{3,}", "\n\n", s)
#     return s.strip()


# def clean_markdown_noise(s: str) -> str:
#     s = re.sub(r"<!--\s*Page\s*\d+\s*-->\s*", "\n", s)
#     s = re.sub(r"^[ \t]*#{1,6}[ \t]+", "", s, flags=re.MULTILINE)
#     return s


# def try_parse_date(s: str) -> Optional[str]:
#     if not s:
#         return None
#     s = s.strip().strip(".").strip(",")
#     s = re.sub(r"(\d{1,2})(st|nd|rd|th)\b", r"\1", s, flags=re.IGNORECASE)
#     s = re.sub(r"\s+", " ", s)
#     s = s.replace(" - ", "-").replace(" – ", "-").replace(" — ", "-")
#     fmts = [
#         "%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%m/%d/%Y",
#         "%d %b %Y", "%d %B %Y", "%b %d, %Y", "%B %d, %Y",
#         "%d.%m.%Y", "%d-%b-%Y", "%d-%B-%Y"
#     ]
#     for f in fmts:
#         try:
#             return datetime.strptime(s, f).strftime("%Y-%m-%d")
#         except Exception:
#             pass
#     m = re.match(r"(\d{1,2})[-/\s](Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*[-/\s](\d{4})",
#                  s, flags=re.IGNORECASE)
#     if m:
#         try:
#             return datetime.strptime(f"{m.group(1)} {m.group(2)} {m.group(3)}", "%d %b %Y").strftime("%Y-%m-%d")
#         except Exception:
#             pass
#     m2 = re.search(r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s+\d{1,2},\s+\d{4}", s,
#                    flags=re.IGNORECASE)
#     if m2:
#         try:
#             return datetime.strptime(m2.group(0), "%b %d, %Y").strftime("%Y-%m-%d")
#         except Exception:
#             pass
#     m3 = re.search(r"\d{1,2}\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}",
#                    s, flags=re.IGNORECASE)
#     if m3:
#         try:
#             return datetime.strptime(m3.group(0), "%d %B %Y").strftime("%Y-%m-%d")
#         except Exception:
#             pass
#     return s


# def flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
#     items = []
#     for k, v in d.items():
#         new_key = f"{parent_key}{sep}{k}" if parent_key else k
#         if isinstance(v, dict):
#             items.extend(flatten_dict(v, new_key, sep=sep).items())
#         else:
#             items.append((new_key, v))
#     return dict(items)


# # =================== Bedrock Embeddings (unchanged from yours) =================== #

# class BedrockEmbeddingClient:
#     def __init__(self):
#         load_dotenv()
#         self.region = os.getenv("BEDROCK_REGION", "us-east-1")
#         self.role_arn = os.getenv("BEDROCK_ROLE_ARN")
#         self.model_id = os.getenv("BEDROCK_EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0")

#         if self.role_arn:
#             sts_client = boto3.client(
#                 "sts",
#                 region_name=self.region,
#                 aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID") or None,
#                 aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY") or None,
#                 aws_session_token=os.getenv("AWS_SESSION_TOKEN") or None,
#             )
#             assumed = sts_client.assume_role(
#                 RoleArn=self.role_arn,
#                 RoleSessionName=f"BedrockEmbeddingsSession-{uuid.uuid4().hex[:8]}"
#             )
#             creds = assumed["Credentials"]
#             self.client = boto3.client(
#                 "bedrock-runtime",
#                 region_name=self.region,
#                 aws_access_key_id=creds["AccessKeyId"],
#                 aws_secret_access_key=creds["SecretAccessKey"],
#                 aws_session_token=creds["SessionToken"],
                
#             )
#         else:
#             self.client = boto3.client("bedrock-runtime", region_name=self.region)

#     def embed(self, text: str) -> List[float]:
#         payload = {"inputText": text}
#         resp = self.client.invoke_model(
#             modelId=self.model_id,
#             contentType="application/json",
#             accept="application/json",
#             body=json.dumps(payload),
#         )
#         body = resp["body"].read()
#         parsed = json.loads(body)
#         vec = parsed.get("embedding") or parsed.get("embeddings") or []
#         if isinstance(vec, dict) and "values" in vec:
#             vec = vec["values"]
#         if not isinstance(vec, list):
#             raise ValueError(f"Unexpected embedding format from Bedrock: {type(vec)}")
#         return [float(x) for x in vec]


# # ======================= NEW: LLM JSON Field Extractor ======================== #

# class LLMFieldExtractor:
#     """Call an LLM on Bedrock to extract contract fields as strict JSON.

#     This extractor does **no regex mining** of fields. We only:
#       1) Build a doc-type specific JSON schema (keys match your old output keys).
#       2) Ask the LLM to return **only** a JSON object matching that schema.
#       3) Post-process dates via `try_parse_date` and normalize amounts.
#     """

#     def __init__(self):
#         load_dotenv()
#         self.region = os.getenv("BEDROCK_REGION", "us-east-1")
#         self.role_arn = os.getenv("BEDROCK_ROLE_ARN")
#         self.model_id = os.getenv("BEDROCK_LLM_MODEL_ID", "anthropic.claude-3-sonnet-20240229-v1:0")

#         if self.role_arn:
#             sts_client = boto3.client(
#                 "sts",
#                 region_name=self.region,
#                 aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID") or None,
#                 aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY") or None,
#                 aws_session_token=os.getenv("AWS_SESSION_TOKEN") or None,
#             )
#             assumed = sts_client.assume_role(
#                 RoleArn=self.role_arn,
#                 RoleSessionName=f"BedrockLLMExtract-{uuid.uuid4().hex[:8]}"
#             )
#             creds = assumed["Credentials"]
#             self.client = boto3.client(
#                 "bedrock-runtime",
#                 region_name=self.region,
#                 aws_access_key_id=creds["AccessKeyId"],
#                 aws_secret_access_key=creds["SecretAccessKey"],
#                 aws_session_token=creds["SessionToken"],
                
#             )
#         else:
#             self.client = boto3.client("bedrock-runtime", region_name=self.region)

#     # ---- schema helpers ---- #

#     @staticmethod
#     def _base_schema() -> Dict[str, Any]:
#         return {
#             "SOW Name": None,
#             "Supplier Name": None,
#             "Supplier Geography": None,
#             "EXECUTIVE SUMMARY": None,
#             "Services": None,
#             "Deliverables": None,
#             "SOW Effective Date": None,
#             "SOW Start Date": None,
#             "SOW End Date": None,
#             "effect until": None,  # kept for legacy compat
#             "Fee Amount": None,
#             "Fee Basis - Fixed/TnM": None,
#             "Invoices": None,
#             "GEHC Project Name": None,
#             "GEHC PM Phone": None,
#             "GEHC PM Email": None,
#             "Supplier PM": None,
#             "Supplier PM Phone": None,
#             "Supplier Email": None,
#             "General Assumption": None,
#             "Out of Scope": None,
#             "SLA": None,
#             "Location": None,
#             "GEHC Signed Party": None,
#             "Supplier Signed Party": None,
#         }

#     @staticmethod
#     def _cr_schema() -> Dict[str, Any]:
#         return {
#             "SOW Name": None,
#             "Change Request Start Date": None,
#             "Change Request End Date": None,
#             "Supplier Name": None,
#             "Supplier Geography": None,
#             "Date Submitted": None,
#             "Change To: Delivery/Cost": None,
#             "Revised SOW Amount": None,
#             "Original SOW Amount": None,
#             "Invoices": None,
#             "Timing Impact": None,
#             "Date Response Delivered": None,
#             "GEHC Signed Party": None,
#             "Supplier Signed Party": None,
#         }

#     @staticmethod
#     def _amend_schema() -> Dict[str, Any]:
#         return {
#             "Ammendment Date": None,
#             "Parties": None,
#             "Effective Date": None,
#             "Total Cost": None,
#             "Payment Schedule": None,
#             "Supplier Geography": None,
#             "Supplier Name": None,
#         }

#     @staticmethod
#     def _norm_amount(x: Optional[str]) -> Optional[str]:
#         if not x:
#             return x
#         v = str(x).strip()
#         v = re.sub(r"[,$₹€£]", "", v)
#         v = re.sub(r"\s{2,}", " ", v)
#         return v

#     @staticmethod
#     def _select_schema(doc_type: DocType) -> Dict[str, Any]:
#         if doc_type == DocType.SOW:
#             return LLMFieldExtractor._base_schema()
#         if doc_type == DocType.CHANGE_REQUEST:
#             return LLMFieldExtractor._cr_schema()
#         if doc_type == DocType.AMENDMENT:
#             return LLMFieldExtractor._amend_schema()
#         # default: try SOW-like
#         return LLMFieldExtractor._base_schema()

#     def _build_prompt(self, text: str, doc_type: DocType, schema: Dict[str, Any]) -> Dict[str, Any]:
#         """Build Anthropic messages-style payload for Bedrock with a single user turn
#         and system guidance to avoid role alternation errors.
#         """
#         schema_keys = list(schema.keys())
#         guidance = (
#             "You are an expert contracts analyst. Extract the requested fields from the document strictly as JSON. "
#             "If a value is truly absent, return null. Do not invent values. Dates can appear in many formats; "
#             "return them as found (we will normalize later). Do not include any extra keys."
#         )
#         fields_list = "\n- " + "\n- ".join(schema_keys)
#         user_msg = (
#             f"Document type: {doc_type.value}.\n\n"
#             f"Return ONLY a JSON object with exactly these keys (values can be string or null):\n{fields_list}\n\n"
#             f"Document follows below between <doc> tags.\n<doc>\n{text[:150000]}\n</doc>"
#         )
#         return {
#             "anthropic_version": "bedrock-2023-05-31",
#             "system": guidance,
#             "max_tokens": 4096,
#             "temperature": 0,
#             "messages": [
#                 {"role": "user", "content": [{"type": "text", "text": user_msg}]}
#             ],
#         }

#     @staticmethod
#     def _extract_json_from_text(txt: str) -> Dict[str, Any]:
#         """Be forgiving if the model wraps JSON with prose; try to snip the JSON blob."""
#         try:
#             return json.loads(txt)
#         except Exception:
#             pass
#         # Find first { and last } and try again
#         start = txt.find("{")
#         end = txt.rfind("}")
#         if start != -1 and end != -1 and end > start:
#             snippet = txt[start:end+1]
#             try:
#                 return json.loads(snippet)
#             except Exception:
#                 pass
#         raise ValueError("LLM did not return valid JSON")

#     def extract_fields(self, text: str, doc_type: DocType) -> Dict[str, Any]:
#         schema = self._select_schema(doc_type)
#         payload = self._build_prompt(text, doc_type, schema)
#         resp = self.client.invoke_model(
#             modelId=self.model_id,
#             contentType="application/json",
#             accept="application/json",
#             body=json.dumps(payload),
#         )
#         raw = resp["body"].read().decode("utf-8")
#         data = json.loads(raw)
#         # Anthropic on Bedrock returns {"content":[{"type":"text","text":"..."}], ...}
#         try:
#             model_text = "".join([p.get("text", "") for p in data.get("content", [])]) or data.get("output_text", "")
#         except Exception:
#             model_text = data.get("output_text", "")
#         parsed = self._extract_json_from_text(model_text)

#         # ensure all required keys exist
#         out: Dict[str, Any] = {k: parsed.get(k) for k in schema.keys()}

#         # normalize dates and amounts
#         for k in list(out.keys()):
#             v = out[k]
#             if v is None:
#                 continue
#             if "date" in k.lower() or k.lower().endswith(" end date") or k.lower().endswith(" start date") or k.lower() == "effect until":
#                 out[k] = try_parse_date(str(v))
#             if any(tok in k.lower() for tok in ["amount", "cost", "value", "total"]):
#                 out[k] = self._norm_amount(str(v))

#         # maintain legacy mirror for SOW end date under "effect until"
#         if doc_type == DocType.SOW and out.get("SOW End Date") and not out.get("effect until"):
#             out["effect until"] = out.get("SOW End Date")

#         return out


# # ================== Tiny Heuristic Doc-Type Detector (fallback) ================== #

# class HeuristicTypeDetector:
#     @staticmethod
#     def detect_type(text: str) -> DocType:
#         t = text.lower()
#         if re.search(r"\bchange\s+request\b", t) or re.search(r"\bcr\s*#?:?\b", t):
#             return DocType.CHANGE_REQUEST
#         if re.search(r"\bstatement\s+of\s+work\b", t) or re.search(r"\bsow\b", t):
#             return DocType.SOW
#         if re.search(r"\bamendment\b", t) or re.search(r"\bammendment\b", t):
#             return DocType.AMENDMENT
#         return DocType.UNKNOWN


# # ========================= Structure-Preserving Parser ========================= #

# # NOTE: Keep your existing advanced parser; we reuse it as-is. For brevity, the
# # following uses a minimal version that falls back to PyPDFLoader. If you already
# # have a richer parser (titles/sections), plug it in here unchanged.

# class StructurePreservingDocumentParser:
#     def parse_pdf_to_markdown(self, file_path: str) -> str:
#         try:
#             loader = PyPDFLoader(file_path)
#             docs = loader.load()
#             text = "\n".join([normalize_ws(d.page_content) for d in docs])
#             return text
#         except Exception as e:
#             logging.error(f"PDF load failed for {file_path}: {e}")
#             return ""


# # ============================== Main Processor ============================== #

# class DocumentProcessor:
#     def __init__(self, output_folder: str = "output"):
#         load_dotenv()
#         self.qdrant_host = os.getenv("QDRANT_HOST", "localhost")
#         self.qdrant_port = int(os.getenv("QDRANT_PORT", 6333))
#         self.collection_name = os.getenv("QDRANT_COLLECTION_NAME", "documents_collection")
#         self.client = QdrantClient(host=self.qdrant_host, port=self.qdrant_port)

#         self.embedder = BedrockEmbeddingClient()
#         self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=350, length_function=len)

#         self.llm_classifier = LLMClient() if LLMClient else None
#         self.field_extractor = LLMFieldExtractor()
#         self.heuristic = HeuristicTypeDetector()
#         self.document_parser = StructurePreservingDocumentParser()

#         self.output_folder = output_folder
#         os.makedirs(output_folder, exist_ok=True)
#         self.markdown_folder = os.path.join(output_folder, "markdown")
#         self.chunks_folder = os.path.join(output_folder, "chunks")
#         os.makedirs(self.markdown_folder, exist_ok=True)
#         os.makedirs(self.chunks_folder, exist_ok=True)

#         self._ensure_collection_exists()

#     def _ensure_collection_exists(self):
#         try:
#             collections = self.client.get_collections()
#             collection_names = [col.name for col in collections.collections]
#             vector_size = int(os.getenv("QDRANT_VECTOR_SIZE", "1024"))  # Titan v2 = 1024 dims
#             if self.collection_name not in collection_names:
#                 self.client.create_collection(
#                     collection_name=self.collection_name,
#                     vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
#                 )
#                 print(f"Created collection: {self.collection_name} (size={vector_size})")
#             else:
#                 print(f"Collection {self.collection_name} already exists")
#         except Exception as e:
#             print(f"Error setting up Qdrant collection: {e}")

#     # ---- Extraction flow with LLM ---- #

#     def pdf_to_markdown(self, pdf_path: str) -> str:
#         markdown_content = self.document_parser.parse_pdf_to_markdown(pdf_path)
#         self._save_markdown(pdf_path, markdown_content)
#         return markdown_content

#     def create_document_chunk(self, file_path: str) -> Optional[Dict[str, Any]]:
#         try:
#             markdown_content = self.pdf_to_markdown(file_path)
#             if not markdown_content:
#                 return None

#             cleaned = normalize_ws(clean_markdown_noise(markdown_content))

#             # (A) LLM classification (if available)
#             llm_doc_type_str, llm_conf, llm_reasons = ("UNKNOWN", 0.0, [])
#             if self.llm_classifier:
#                 try:
#                     llm_doc_type_str, llm_conf, llm_reasons = self.llm_classifier.classify_document_type(cleaned)
#                 except Exception as e:
#                     logging.warning(f"LLM doc-type classification failed: {e}")

#             # (B) Heuristic detection fallback
#             heur_type = self.heuristic.detect_type(cleaned)

#             # (C) Decide final type
#             final_doc_type = heur_type
#             try:
#                 if llm_doc_type_str != "UNKNOWN" and llm_conf >= 0.55:
#                     final_doc_type = DocType[llm_doc_type_str]
#                 elif heur_type == DocType.UNKNOWN and llm_doc_type_str in {"SOW", "CHANGE_REQUEST", "AMENDMENT"}:
#                     final_doc_type = DocType[llm_doc_type_str]
#             except Exception:
#                 pass

#             # (D) **LLM JSON extraction**
#             fields = self.field_extractor.extract_fields(cleaned, final_doc_type)

#             metadata = {
#                 "doc_type": final_doc_type.value,
#                 "doc_type_llm": llm_doc_type_str,
#                 "doc_type_llm_confidence": llm_conf,
#                 "doc_type_llm_reasons": llm_reasons,
#                 "doc_type_heuristic": heur_type.value,
#                 "extracted_fields": fields,
#             }

#             chunk = {
#                 "id": str(uuid.uuid4()),
#                 "file_path": file_path,
#                 "file_name": os.path.basename(file_path),
#                 "content": markdown_content,
#                 "content_type": "document",
#                 "processed_date": datetime.now().isoformat(),
#                 "metadata": metadata,
#             }
#             return chunk
#         except Exception as e:
#             print(f"Error creating document chunk: {e}")
#             return None

#     def generate_embedding(self, text: str) -> List[float]:
#         try:
#             max_length = 8000
#             if len(text) > max_length:
#                 text = text[:max_length]
#             return self.embedder.embed(text)
#         except Exception as e:
#             print(f"Error generating embedding: {e}")
#             return []

#     def _save_markdown(self, file_path: str, markdown_content: str):
#         try:
#             file_name = os.path.basename(file_path)
#             base_name = os.path.splitext(file_name)[0]
#             markdown_file = os.path.join(self.markdown_folder, f"{base_name}.md")
#             with open(markdown_file, "w", encoding="utf-8") as f:
#                 f.write(markdown_content)
#             print(f"Saved markdown: {markdown_file}")
#         except Exception as e:
#             print(f"Error saving markdown for {file_path}: {e}")

#     def _save_chunk_json(self, file_path: str, chunk: Dict[str, Any]):
#         try:
#             file_name = os.path.basename(file_path)
#             base_name = os.path.splitext(file_name)[0]
#             chunk_file = os.path.join(self.chunks_folder, f"{base_name}_chunk.json")
#             with open(chunk_file, "w", encoding="utf-8") as f:
#                 json.dump(chunk, f, indent=2, ensure_ascii=False)

#             chunk_summary = dict(chunk)
#             content = chunk["content"] or ""
#             chunk_summary["content_preview"] = (content[:500] + "...") if len(content) > 500 else content
#             chunk_summary["content_length"] = len(content)
#             summary_file = os.path.join(self.chunks_folder, f"{base_name}_summary.json")
#             with open(summary_file, "w", encoding="utf-8") as f:
#                 json.dump(chunk_summary, f, indent=2, ensure_ascii=False)

#             print(f"Saved chunk JSON: {chunk_file}")
#             print(f"Saved chunk summary: {summary_file}")
#         except Exception as e:
#             print(f"Error saving chunk JSON for {file_path}: {e}")

#     def store_to_qdrant(self, chunk: Dict[str, Any]) -> bool:
#         try:
#             embedding = self.generate_embedding(chunk["content"])
#             if not embedding:
#                 print("Failed to generate embedding")
#                 return False

#             payload = {
#                 "file_path": chunk["file_path"],
#                 "file_name": chunk["file_name"],
#                 "content": chunk["content"],
#                 "content_type": chunk["content_type"],
#                 "processed_date": chunk["processed_date"],
#                 "doc_type": chunk["metadata"].get("doc_type"),
#                 "extracted_fields": chunk["metadata"].get("extracted_fields", {}),
#             }
#             flat = flatten_dict({"extracted": payload["extracted_fields"]})
#             payload.update(flat)

#             point = PointStruct(id=chunk["id"], vector=embedding, payload=payload)
#             self.client.upsert(collection_name=self.collection_name, points=[point])
#             print(f"Successfully stored document: {chunk['file_name']}")
#             return True
#         except Exception as e:
#             print(f"Error storing to Qdrant: {e}")
#             return False

#     def process_document(self, file_path: str):
#         print(f"Processing document: {file_path}")
#         chunk = self.create_document_chunk(file_path)
#         if chunk:
#             self._save_chunk_json(file_path, chunk)
#             ok = self.store_to_qdrant(chunk)
#             if ok:
#                 print(f"Successfully processed: {file_path}")
#             else:
#                 print(f"Failed to store to Qdrant: {file_path}")
#         else:
#             print(f"Failed to create chunk for: {file_path}")

#     def process_documents_directory(self, directory_path: str) -> None:
#         if not os.path.exists(directory_path):
#             print(f"Directory not found: {directory_path}")
#             return
#         pdf_files = [f for f in os.listdir(directory_path) if f.lower().endswith('.pdf')]
#         if not pdf_files:
#             print(f"No PDF files found in: {directory_path}")
#             return
#         print(f"Found {len(pdf_files)} PDF files to process")
#         for pdf_file in pdf_files:
#             file_path = os.path.join(directory_path, pdf_file)
#             self.process_document(file_path)
#             print("-" * 50)


# # ------------------------------ Main (example) ------------------------------ #

# def main():
#     output_folder = os.getenv("OUTPUT_DIR", r"C:\\Users\\CSS\\Desktop\\Workspace\\RAG_Coupa_chatbot\\output2")
#     docs_directory = os.getenv("DOCS_DIR", r"C:\\Users\\CSS\\Desktop\\Workspace\\RAG_Coupa_chatbot\\Docs")
#     processor = DocumentProcessor(output_folder=output_folder)
#     processor.process_documents_directory(docs_directory)
#     print(f"\n📁 Output saved to: {output_folder}")
#     print(f"  - Markdown files: {processor.markdown_folder}")
#     print(f"  - Chunk JSON files: {processor.chunks_folder}")


# if __name__ == "__main__":
#     main()
"""
Refactor: Replace regex-based field extraction with LLM JSON extraction (AWS Bedrock)

What changed?
- Added `LLMFieldExtractor` that prompts an LLM to extract a strict JSON object for SOW/CR/Amendment.
- `DocumentProcessor.create_document_chunk()` now calls LLM-based extraction instead of regex patterns.
- Kept a tiny `HeuristicTypeDetector` (regex-light) only for doc-type detection fallback.
- Dates are normalized post-LLM via `try_parse_date()`.

Dependencies:
- boto3 configured for Bedrock (env vars or instance role)
- Set `BEDROCK_LLM_MODEL_ID` (e.g., "anthropic.claude-3-sonnet-20240229-v1:0")
- Optional: `BEDROCK_REGION`, `BEDROCK_ROLE_ARN`

This file is a drop-in replacement for your previous module: it preserves existing
class names where possible and clearly marks the LLM changes.
"""

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
import boto3
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


# =================== Bedrock Embeddings (unchanged from yours) =================== #

class BedrockEmbeddingClient:
    def __init__(self):
        load_dotenv()
        self.region = os.getenv("BEDROCK_REGION", "us-east-1")
        self.role_arn = os.getenv("BEDROCK_ROLE_ARN")
        self.model_id = os.getenv("BEDROCK_EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0")

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
                RoleSessionName=f"BedrockEmbeddingsSession-{uuid.uuid4().hex[:8]}"
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

    def embed(self, text: str) -> List[float]:
        payload = {"inputText": text}
        resp = self.client.invoke_model(
            modelId=self.model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(payload),
        )
        body = resp["body"].read()
        parsed = json.loads(body)
        vec = parsed.get("embedding") or parsed.get("embeddings") or []
        if isinstance(vec, dict) and "values" in vec:
            vec = vec["values"]
        if not isinstance(vec, list):
            raise ValueError(f"Unexpected embedding format from Bedrock: {type(vec)}")
        return [float(x) for x in vec]


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

        self.embedder = BedrockEmbeddingClient()
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
            vector_size = int(os.getenv("QDRANT_VECTOR_SIZE", "1024"))  # Titan v2 = 1024 dims
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
