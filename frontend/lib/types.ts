/**
 * Shared API types.
 *
 * Reconstructed from the component call sites after the original lib/ was lost
 * to an unanchored `lib/` pattern in the root .gitignore. Where a field is
 * marked required below it is because a component dereferences it unguarded.
 */

// ---------------------------------------------------------------- documents --

export type DocumentStatus =
  | "pending"
  | "extracting"
  | "parsing_tables"
  | "chunking"
  | "completed"
  | "failed"

export interface Document {
  id: string
  filename: string
  /** Widened to string: document-sidebar types statusConfig as Record<string, ...>. */
  status: DocumentStatus | string
  table_count: number
  chunk_count: number
  file_hash?: string
  page_count?: number | null
  uploaded_at?: string | null
  completed_at?: string | null
  error_message?: string | null
}

export type UploadStatus = "idle" | "uploading" | "success" | "error"

export interface TableInfo {
  /** Also used as the React key in tables-view. */
  physical_table_name: string
  semantic_description?: string | null
  /** Required: tables-view calls .map() on it unguarded. */
  headers: string[]
  /** Required: tables-view calls .toLocaleString() on it unguarded. */
  row_count: number
  original_filename: string
  id?: number | string
  table_id?: string
  source_doc_uuid?: string
  page_start?: number | null
  page_end?: number | null
  created_at?: string | null
}

// ---------------------------------------------------------------- artifacts --

export type ArtifactKind = "chart" | "table" | "scalar" | "empty"

export type ArtifactType =
  | "line"
  | "bar"
  | "bar_horizontal"
  | "bar_grouped"
  | "scatter"
  | "table"
  | "scalar"
  | "empty"
  /** Reserved, never emitted. Part-of-whole cannot be verified from data shape. */
  | "pie"

export type FieldKind = "numeric" | "temporal" | "categorical" | "boolean" | "opaque"

export interface ArtifactField {
  key: string
  label: string
  kind: FieldKind
  unit?: string | null
  decimals?: number
  /** 1-5, indexes the --chart-N CSS variables in app/globals.css. */
  color_index?: number
  align?: "left" | "right"
}

export interface ArtifactProvenance {
  sql: string | null
  tables_used: string[]
  /** Rows Postgres actually returned. */
  row_count: number
  /** Rows present in `data` (capped). */
  data_rows: number
  /** data_rows < row_count. */
  truncated: boolean
  /** Hit the query LIMIT with an ORDER BY: a genuine top-N, more may exist. */
  partial: boolean
  source_documents: Array<{ source_file: string; score?: number }>
  /** HYBRID only: the prose is an LLM merge, the artifact is not. */
  answer_is_llm_merged: boolean
}

export type DataCell = string | number | boolean | null

export interface Artifact {
  id: string
  schema_version: number
  kind: ArtifactKind
  type: ArtifactType
  title: string
  /** Array-of-objects, recharts-native. */
  data: Array<Record<string, DataCell>>
  encoding: {
    x: ArtifactField | null
    y: ArtifactField[]
    series: ArtifactField | null
  }
  columns: ArtifactField[]
  /** kind === "scalar" only. */
  value?: DataCell
  unit?: string | null
  provenance: ArtifactProvenance
  notes: string[]
  classifier: { rule: string; version: number; pivoted?: boolean }
}

// -------------------------------------------------------------------- query --

export type Route = "SQL" | "RAG" | "HYBRID"

export interface Source {
  score: number
  content: string
  heading_context?: string
  source_file?: string
  chunk_id?: number
  section?: string
  start_offset?: number
  end_offset?: number
  page_number?: number | null
}

/**
 * Response of POST /api/query.
 *
 * Only `answer`, `route`, `query`, `original_query` and `artifacts` are
 * guaranteed; the rest vary by route and by which branch the router took.
 */
export interface QueryResponse {
  answer: string
  route?: Route
  query?: string
  original_query?: string
  rewritten_query?: string
  sql?: string
  sql_executed?: string
  sql_explanation?: string
  sql_error?: string
  tables_used?: string[]
  row_count?: number
  sources?: Source[]
  artifacts?: Artifact[]
  error?: string
}

export interface ChatMessage {
  id: string
  role: "user" | "assistant"
  content: string
  /** Required: chat-interface calls .toLocaleTimeString() unguarded. */
  timestamp: Date
  route?: Route
  sql?: string
  sql_explanation?: string
  tables_used?: string[]
  rewritten_query?: string
  sources?: Source[]
  artifacts?: Artifact[]
}
