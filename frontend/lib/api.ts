/**
 * Backend API client.
 *
 * There is no Next.js rewrite/proxy - next.config.mjs has none. The Flask app
 * enables CORS instead, so these are plain cross-origin calls to :8000.
 *
 * Every function throws an Error on a non-OK response: both catch sites in the
 * components read `err.message`.
 */

import type { Document, QueryResponse, TableInfo } from "@/lib/types"

export const API_BASE = (
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"
).replace(/\/$/, "")

/** Pull the most useful message out of a failed response body. */
async function toError(res: Response, fallback: string): Promise<Error> {
  let detail = ""
  try {
    const body = await res.json()
    detail = body?.error || body?.message || ""
  } catch {
    detail = await res.text().catch(() => "")
  }
  return new Error(detail || `${fallback} (HTTP ${res.status})`)
}

async function getJSON<T>(path: string, fallback: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { Accept: "application/json" },
    cache: "no-store",
  })
  if (!res.ok) throw await toError(res, fallback)
  return (await res.json()) as T
}

// ----------------------------------------------------------------- querying --

export async function sendQuery(
  query: string,
  sessionId: string,
  options?: { documentIds?: string[]; forceRoute?: "SQL" | "RAG" | "HYBRID" },
): Promise<QueryResponse> {
  const res = await fetch(`${API_BASE}/api/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query,
      session_id: sessionId,
      ...(options?.documentIds ? { document_ids: options.documentIds } : {}),
      ...(options?.forceRoute ? { force_route: options.forceRoute } : {}),
    }),
  })
  if (!res.ok) throw await toError(res, "Query failed")

  const data = (await res.json()) as QueryResponse
  // `artifacts` is guaranteed by the orchestrator, but normalise defensively so
  // callers can always map over it.
  return { ...data, artifacts: data.artifacts ?? [] }
}

export async function clearSession(sessionId: string): Promise<void> {
  const res = await fetch(
    `${API_BASE}/api/sessions/${encodeURIComponent(sessionId)}/clear`,
    { method: "POST" },
  )
  if (!res.ok) throw await toError(res, "Could not clear session")
}

/**
 * Fire-and-forget reset on tab close. Synchronous by necessity: `beforeunload`
 * will not await a promise, which is why this uses sendBeacon rather than fetch.
 */
export function resetDatabaseBeacon(): void {
  try {
    const url = `${API_BASE}/api/reset`
    if (typeof navigator !== "undefined" && navigator.sendBeacon) {
      navigator.sendBeacon(url, new Blob([], { type: "application/json" }))
    } else {
      void fetch(url, { method: "POST", keepalive: true })
    }
  } catch {
    // Nothing useful to do while the page is unloading.
  }
}

// ---------------------------------------------------------------- documents --

export async function fetchDocuments(): Promise<Document[]> {
  const data = await getJSON<{ documents?: Document[] }>(
    "/api/documents",
    "Could not load documents",
  )
  return data.documents ?? []
}

export async function uploadDocument(
  file: File,
  options?: { force?: boolean },
): Promise<{ doc_id?: string; status?: string; skipped?: boolean }> {
  const form = new FormData()
  // Field name must be "file": app.py checks `if "file" not in request.files`.
  form.append("file", file)
  if (options?.force) form.append("force", "true")

  const res = await fetch(`${API_BASE}/api/documents/upload`, {
    method: "POST",
    body: form,
  })
  if (!res.ok) throw await toError(res, "Upload failed")
  return res.json()
}

// ------------------------------------------------------------------- tables --

export async function fetchTables(documentIds?: string[]): Promise<TableInfo[]> {
  const qs =
    documentIds && documentIds.length
      ? `?document_ids=${encodeURIComponent(documentIds.join(","))}`
      : ""
  const data = await getJSON<{ tables?: TableInfo[] }>(
    `/api/tables${qs}`,
    "Could not load tables",
  )
  return data.tables ?? []
}

// ------------------------------------------------------------------- health --

export async function fetchHealth(): Promise<{ status: string }> {
  return getJSON<{ status: string }>("/health", "Backend unreachable")
}
