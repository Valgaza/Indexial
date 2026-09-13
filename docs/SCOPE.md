# SCOPE — Krushitatva / NDAP Conversational Analytics Platform

> A self-hosted, citation-grounded **agentic RAG platform** that turns India's
> public statistical datasets (NDAP / NFHS / PLFS / NSS / TB / Census) into a
> multilingual, audit-ready chat experience. Built as a POC for a government
> data programme (NITI Aayog's NDAP), later rebranded for a new client
> ("Krushitatva").
>
> This document is the engineering scope of the whole solution — architecture,
> stack, and the hard problems solved — written so it can be mined for résumé
> bullets. A **résumé bullet bank** is at the bottom.

---

## 1. One-paragraph summary

A retrieval-augmented, multi-agent question-answering system over Indian
government statistical data. A **LangGraph** state machine routes each query
through a deterministic pipeline (translate → classify intent → plan → route to
1..N specialist agents → verify → synthesise → enforce citations → translate
back), where **structured numbers always come from SQL and the LLM only writes
prose on top of already-verified, cited claims**. Retrieval is **hybrid
(vector KNN + full-text) fused with reciprocal-rank fusion and re-scored by a
cross-encoder reranker**, served from **Postgres + pgvector**. Every model in
the stack — the chat/synthesis LLM (Gemma 3 27B), the embedding model (BGE-M3),
and the reranker (bge-reranker-v2-m3) — is **self-hosted on vLLM** with no
external/commercial API, to satisfy data-residency and no-external-LLM
compliance constraints. The whole app + data tier ships as a single
**Docker Compose** topology that runs identically on a laptop and on the GPU
server, with a separate **GCP/GKE Terraform + Helm + ArgoCD** production
blueprint.

---

## 2. The problem it solves

Government statistical portals are rich but unusable by non-analysts: data is
spread across surveys, vintages, geographic levels (national / state / district
/ city), and hundreds of dense PDF appendix tables. The platform lets a policy
user ask a plain-language question — in English **or 20+ Indic languages, by
text or voice** — and get back a **structured, cited, confidence-scored answer**
(cards / tables / charts + editorial prose), with a full **traceable audit
trail** of which agents ran, which tools they called, and which source rows and
document pages every number came from.

The central engineering constraint: **never fabricate a number, never
hard-refuse when partial data exists, and never present interpretation as a
data finding.** Much of the architecture exists to enforce exactly that.

---

## 3. High-level architecture

Three tiers; the first two are fully implemented and ship as one Compose file,
the third is an infrastructure-as-code production blueprint.

```
┌──────────────────────────────────────────────────────────────────────┐
│  APP TIER  (Docker Compose — runs identically local & on-server)       │
│                                                                        │
│   Browser ──► web (Next.js 14)  ──►  orchestrator (FastAPI+LangGraph)  │
│                                          │  hosts MCP servers as        │
│                                          │  in-process stdio subprocs   │
│                                   ┌──────┼─────────────┐                │
│                              mcp-sqlite mcp-bhashini mcp-pgvector       │
│                              (structured (lang detect/  (vector KNN +   │
│                               SQL+FTS5)   translate)     tsvector FTS)  │
└──────────────────────────────────────────────────────────────────────┘
            │                                        │
┌───────────┴─────────────┐          ┌───────────────┴───────────────────┐
│  DATA TIER               │          │  MODEL TIER (native vLLM, systemd, │
│  Postgres + pgvector     │          │  GPU server — 8× NVIDIA L4)        │
│   documents,             │          │   Gemma 3 27B  :8000  (TP=4, GPU0-3)│
│   document_chunks(        │◄────────┤   BGE-M3       :8080  (embed, GPU7) │
│     embedding vec(1024),  │  embed/  │   bge-reranker :8081  (rerank,GPU6)│
│     ts tsvector)          │  rerank  └────────────────────────────────────┘
│   ndap_facts (structured) │
│  + hourly incremental     │
│    ingest sidecar         │
└───────────────────────────┘

PRODUCTION BLUEPRINT (IaC, designed): GCP asia-south1/2 · GKE · Cloud SQL
(pgvector+PostGIS) · APISIX gateway · LiteLLM router · Vault · Prometheus/
Grafana/Loki · Wazuh SIEM · ArgoCD GitOps · Kata Containers microVM sandbox
```

---

## 4. The query pipeline (LangGraph state machine)

Every request runs through a **deterministic graph** — same input state in,
same output state out (modulo timestamps) — which is what makes runs replayable
and auditable.

```
START
 → translate_in        detect language; translate non-English → English
 → classify_intent     LLM normalises colloquial query → canonical terms,
                       then a deterministic rule parser extracts intent
 → plan                pick which of the 9 specialist agents to run
 → route               run the chosen agent(s) (each is an MCP client)
 → verify              Agent 8 (Hallucination Detection) re-checks every number
 → synthesise          LLM writes prose ON TOP of already-verified claims
 → enforce_citations   downgrade any unsourced claim to UNVERIFIED
 → translate_out       translate answer + claims back to user's language
 → END
```

Key design decision: **semantic flexibility lives only on the front end.** The
LLM rewrites *"why do babies die in MP"* → *"why is the neonatal mortality rate
high in Madhya Pradesh"*, but parameter extraction and all numeric work stay
deterministic and SQL-backed. The canonical output unit is a **Claim**, which
always carries citations, a confidence band, and an Agent-8 verification
verdict. The UI renders claims as cards/tables/charts and the synthesised prose
as editorial on top.

Graceful degradation is wired through the whole graph: if the LLM endpoint is
down, classification/synthesis fall back to a deterministic rule chain and a
templated response; if translation fails, it passes through in English.

---

## 5. Retrieval — hybrid, fused, reranked

Implemented in `mcp-pgvector`:

1. **Embed** the query with self-hosted BGE-M3 (1024-dim).
2. **First-stage recall** — cosine **KNN over an HNSW index** merged with
   **`tsvector` full-text** keyword candidates, de-duplicated by chunk, fused
   with **reciprocal-rank fusion (RRF)** into a candidate pool.
3. **Second-stage precision** — a **cross-encoder reranker (bge-reranker-v2-m3)**
   re-scores every `(query, chunk)` pair; the top-k go to the agent.

Multiple layers of graceful degradation: no reranker configured → vector top-k
by cosine; reranker unreachable → first-stage order; embed endpoint down → the
agent falls back to pure full-text search. A single env flag (`PGVECTOR_DSN`)
switches the entire retrieval backend.

A documented root-cause investigation (`FINDINGS.md`) drove targeted upgrades:
**row-level table extraction** (turning dense PDF appendix tables into discrete,
rankable rows in a structured `ndap_facts` table), **acronym/synonym query
expansion**, **query decomposition** (splitting a broad question into atomic
sub-questions, each with its own coverage verdict: ANSWERED / NOT_FOUND /
PARTIAL), and a **synthesis critic** that recomputes aggregates from structured
rows and separates FACTS from INTERPRETATION — so the system answers what it
can and flags what's genuinely absent rather than emitting a blanket
"no data".

---

## 6. The agents (specialist + routing)

The orchestrator is the **10th, routing-only agent**; it dispatches to nine
specialist agents, each an **MCP client**:

| # | Agent | What it does |
|---|---|---|
| 1 | data-discovery | Find relevant datasets for a question |
| 2 | nl-query / analytics | Natural-language → SQL analytics over structured facts |
| 3 | cross-dataset-insight (composite) | Join/intersect 2+ indicators across geographies |
| 4 | predictive-scenario / forecast | Trend projection and scenarios |
| 5 | anomaly / early-warning | Outlier and anomaly detection |
| 6 | auto-report | Assemble multi-claim narrative reports |
| 7 | data-quality | Completeness / freshness checks |
| 8 | hallucination-detect | Re-verify every numeric claim against source |
| 9 | delta-monitor | Detect changes/refreshes in source data |

The **composite agent** is a highlight: driven by a **capability map** that
encodes, per dimension, what data actually exists and at which geographic level.
It joins at the requested level when possible, **degrades to the finest common
level** (city/district → state) when not, and gives every dimension that still
can't join a best-effort claim — "never silence, never a fabricated join."
Join quality, vintage caveats, and degradation notes all ride along as claims.

---

## 7. Multilingual + voice front door

`mcp-bhashini` is a hot-swappable translation/STT layer. The default backend
translates Indic languages (Hindi, Gujarati, Tamil, Bengali, …) through the
**self-hosted Gemma endpoint** (no extra models to mount); alternative backends
include Argos Translate, **IndicTrans2** (AI4Bharat), and **Vosk** for
speech-to-text — all selectable by env flag, with a script-based language
detector always on as ground truth. Designed for a production cutover to
**Bhashini ULCA / Sarvam** speech models. This means the entire English
pipeline (intent + 9 agents) is reused unchanged for any language: translate in,
answer in English, translate out.

---

## 8. Ingestion & vectorization (hourly, incremental, idempotent)

A standalone **ingest sidecar** (reusing the orchestrator image) runs a forever
loop:

- Parses source PDFs/Excel/CSV with **PyMuPDF (fitz)** including
  `find_tables()`, with a **Tesseract OCR fallback** for scanned pages.
- Bespoke + generic **table extractors** (AQI, NSS price lists, generic
  `find_tables()` markdown, plus structured backfills for stunting / TB / LFPR)
  emit both free-text chunks and structured `ndap_facts` rows.
- **Incremental rule:** a new `(document_id, chunk_index)` inserts with a NULL
  embedding (embedded next pass); an existing row is re-embedded **only if its
  text changed**. Unchanged rows are never re-embedded — each cycle processes
  only genuinely new/changed data. Embeddings are derived data → safe to wipe
  and rebuild.

Self-hosted **BGE-M3** does the embedding via an OpenAI-compatible `/v1/embeddings`
endpoint. The whole thing is idempotent and survives a fresh, empty Postgres
(it creates the database + schema on first run).

---

## 9. The web app

**Next.js 14 (App Router) · React 18 · TypeScript strict · Tailwind ·
shadcn-style primitives on Radix.** Far beyond a basic chat box:

- **Streaming chat** over WebSocket with a live **pipeline / agent-trace
  stream** showing each graph node executing in real time.
- **Claim cards** with **confidence pills** (HIGH / MEDIUM / LOW / UNVERIFIED),
  **clickable citations**, and an **in-app citation preview** that renders the
  referenced document section and jumps to the source page.
- **Charts & tables**: a chart kit / catalogue, comparison tables, trend
  visualizers, cross-dataset merger, dashboards.
- **Export** (report generation), copy buttons, suggestion/prebuilt-insight
  cards, transcript history.
- An **admin console**: dataset upload tab (drops PDFs/Excel into the ingest
  corpus), users, activity, and a **run-detail drawer / X-ray view** exposing
  the full traceability of any past run.
- **Voice input** (audio capture → transcribe endpoint).
- **WCAG 2.1 AA / GIGW** accessibility baselines (aria-labels, keyboard nav,
  responsive mobile-first breakpoints).

Auth, session history, and an orchestrator-replay/audit API back the admin
surfaces.

---

## 10. Infrastructure & deployment

**Local + on-server (implemented):** one `docker-compose.yml` brings up web,
orchestrator (with in-process MCP servers), the ingest sidecar, and an optional
bundled pgvector container — on a fixed-subnet private bridge network with
health-check gating and `host.docker.internal` reaching the native vLLM
processes. Postgres can be the bundled container *or* an external/managed
instance, switched by one DSN.

**Model tier (implemented):** Gemma 3 27B, BGE-M3, and the reranker run as
**native vLLM processes managed by systemd** on an 8× L4 GPU server — Gemma
tensor-parallel across 4 GPUs, the two pooling models on one GPU each.

**Production blueprint (designed as IaC):**
- **Terraform** modules for GCP (`asia-south1`/`asia-south2` only — India data
  residency), GKE Standard with Workload Identity Federation, A3/A2/L40S/L4 GPU
  node pools, Cloud SQL Postgres 16 with pgvector + PostGIS, customer-managed
  KMS, private VPC peering (no public DB IPs).
- **Helm** umbrella chart + per-service subcharts, wired for **ArgoCD GitOps**.
- **APISIX** API gateway, **LiteLLM** router (no external LLM backend),
  **Vault** secrets, **Prometheus / Grafana / Loki** observability, **Wazuh**
  SIEM, **Velero** backup, and a **Kata Containers (KVM microVM)** runtime class
  for an isolated Python execution sandbox.

---

## 11. Compliance & trust engineering

This was a government programme, so trustworthiness is a first-class feature,
not an afterthought:

- **No external LLM / embedding API** — everything self-hosted on vLLM.
- **Data residency** — all cloud resources pinned to Indian regions; CMEK
  encryption; no public Cloud SQL IPs.
- **Citation + confidence + Unverified enforcement** — a dedicated graph node
  downgrades any claim lacking a deterministic source to `UNVERIFIED`
  regardless of model confidence.
- **Hallucination detection** — a separate agent re-checks every number against
  its source before synthesis.
- **Deterministic replay & audit** — orchestrator state persisted per
  `(session_id, request_id)`; runs are replayable and every agent/tool call is
  recorded for a CERT-In-style audit trail.
- **MCP as an auditable surface** — every tool every agent can reach is declared
  with a JSON schema, so the entire capability surface is enumerable.

---

## 12. Tech stack at a glance

| Layer | Technology |
|---|---|
| **Orchestration** | Python 3.11, FastAPI, **LangGraph** state machine, async |
| **Agents / tools** | **Model Context Protocol (MCP)**, JSON-RPC over stdio (HTTP+SSE in prod) |
| **LLM** | **Gemma 3 27B** on **vLLM** (tensor-parallel), OpenAI-compatible API |
| **Embeddings** | **BGE-M3** (1024-dim) on vLLM, self-hosted |
| **Reranking** | **bge-reranker-v2-m3** cross-encoder on vLLM |
| **Vector store** | **Postgres + pgvector**, HNSW cosine index, `tsvector` GIN FTS |
| **Retrieval** | Hybrid KNN + FTS, **reciprocal-rank fusion**, cross-encoder rerank |
| **Structured data** | Postgres `ndap_facts`, SQLite + FTS5 (Phase-1 backend) |
| **Doc ingestion** | PyMuPDF (fitz) + `find_tables()`, Tesseract OCR, incremental embed |
| **Frontend** | **Next.js 14**, React 18, TypeScript, Tailwind, Radix/shadcn, WebSocket streaming |
| **Multilingual** | mcp-bhashini: Gemma-LLM / IndicTrans2 / Argos translate, Vosk STT |
| **Packaging** | Docker, Docker Compose, multi-stage non-root images |
| **Prod IaC** | Terraform (GCP/GKE/Cloud SQL), Helm, ArgoCD, APISIX, LiteLLM, Vault |
| **Observability/Sec** | Prometheus, Grafana, Loki, Wazuh SIEM, OpenTelemetry, Kata Containers |
| **Quality** | pytest unit + real-subprocess integration tests, OpenAPI 3.x contract, eval suites |

---

## 13. Résumé bullet bank

Pick and tailor; quantify where you can verify the number.

- Built a **self-hosted agentic RAG platform** over Indian government
  statistical datasets, orchestrating **10 specialist AI agents** through a
  deterministic **LangGraph** state machine (FastAPI, Python 3.11) that
  guarantees every numeric answer is SQL-derived, cited, and confidence-scored.
- Designed a **hybrid retrieval pipeline** — vector KNN (pgvector/HNSW) + full
  text (tsvector) fused via **reciprocal-rank fusion** and re-scored by a
  **cross-encoder reranker** — with multi-level graceful degradation when any
  model endpoint is unavailable.
- Stood up a fully **self-hosted model tier on vLLM** (Gemma 3 27B
  tensor-parallel for chat/synthesis, BGE-M3 for 1024-dim embeddings,
  bge-reranker-v2-m3 for reranking) across an 8× NVIDIA L4 GPU server — **zero
  external LLM API** to meet government data-residency / compliance rules.
- Implemented **citation + confidence + hallucination-detection enforcement**: a
  dedicated agent re-verifies every number against source and a graph node
  downgrades unsourced claims to `UNVERIFIED`, plus a synthesis critic that
  recomputes aggregates and separates fact from interpretation.
- Engineered an **idempotent, incremental ingestion pipeline** (PyMuPDF +
  `find_tables()` + Tesseract OCR + bespoke table extractors) that re-embeds
  only changed chunks each hourly cycle and back-fills dense PDF tables into a
  queryable structured store.
- Used **Model Context Protocol (MCP)** to expose every data/tool capability as
  a uniform, hot-swappable, auditable JSON-RPC surface (vector store, structured
  SQL, multilingual translation), enabling backend swaps without touching agents.
- Built a multilingual, voice-enabled front door supporting **20+ Indic
  languages** by translating in/out around a single English pipeline
  (Gemma-LLM / IndicTrans2 / Vosk backends).
- Delivered a production **Next.js 14 + TypeScript** UI with WebSocket-streamed
  agent traces, interactive claim cards with in-app citation preview,
  chart/table rendering, report export, an admin console with full run
  traceability, and WCAG 2.1 AA accessibility.
- Authored the full **production infrastructure-as-code blueprint**: Terraform
  (GCP/GKE, Cloud SQL pgvector+PostGIS, Workload Identity, CMEK, India-only
  regions), Helm + **ArgoCD GitOps**, APISIX gateway, LiteLLM router, Vault,
  Prometheus/Grafana/Loki, Wazuh SIEM, and a KVM-isolated **Kata Containers**
  code sandbox.
- Packaged the entire app + data tier as a **single Docker Compose topology**
  that runs identically on a laptop and the GPU server, with health-check
  gating and an env-flag-switchable Postgres / retrieval backend.

---

> **Scope note:** the app tier, data tier, RAG/retrieval pipeline, agents,
> ingestion, multilingual layer, web UI, and the vLLM model tier are
> **implemented and demoable**. The GCP/GKE production stack (Terraform, Helm,
> ArgoCD, APISIX, Wazuh, etc.) is committed as **infrastructure-as-code /
> architecture blueprint** validated locally but not yet applied to a live
> cloud account. Represent it accordingly on a résumé (e.g. "designed and
> codified" vs "operated in production").
