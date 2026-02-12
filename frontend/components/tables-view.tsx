"use client"

import { useCallback, useEffect, useState } from "react"
import {
  Database,
  FileText,
  Hash,
  Columns3,
  RefreshCw,
  Table2,
} from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { ScrollArea } from "@/components/ui/scroll-area"
import type { TableInfo } from "@/lib/types"
import { fetchTables } from "@/lib/api"

export function TablesView() {
  const [tables, setTables] = useState<TableInfo[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const loadTables = useCallback(async () => {
    setIsLoading(true)
    setError(null)
    try {
      const data = await fetchTables()
      setTables(Array.isArray(data) ? data : [])
    } catch {
      setError("Could not load tables. Is the backend running?")
      setTables([])
    } finally {
      setIsLoading(false)
    }
  }, [])

  useEffect(() => {
    loadTables()
  }, [loadTables])

  return (
    <div className="flex h-full flex-col">
      {/* Header */}
      <div className="glass-strong flex items-center justify-between px-4 py-3 lg:px-6">
        <div className="flex items-center gap-3">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-[hsl(270,60%,58%)]/20">
            <Database className="h-4 w-4 text-[hsl(270,60%,68%)]" />
          </div>
          <div>
            <h1 className="text-sm font-semibold text-foreground">Tables</h1>
            <p className="text-[10px] text-muted-foreground">
              {tables.length} tables extracted
            </p>
          </div>
        </div>

        <button
          onClick={loadTables}
          disabled={isLoading}
          className="flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground disabled:pointer-events-none disabled:opacity-50"
        >
          <RefreshCw
            className={`h-3.5 w-3.5 ${isLoading ? "animate-spin" : ""}`}
          />
          <span className="hidden sm:inline">Refresh</span>
        </button>
      </div>

      {/* Table list */}
      <ScrollArea className="flex-1 px-4 py-4 lg:px-6">
        {error && (
          <div className="mb-4 rounded-lg bg-red-500/10 px-4 py-3 text-sm text-red-400">
            {error}
          </div>
        )}

        {isLoading ? (
          <div className="flex flex-col gap-3">
            {[1, 2, 3].map((i) => (
              <div
                key={i}
                className="glass-subtle animate-pulse rounded-xl p-4"
              >
                <div className="mb-3 h-4 w-1/3 rounded bg-secondary" />
                <div className="mb-2 h-3 w-2/3 rounded bg-secondary" />
                <div className="h-3 w-1/2 rounded bg-secondary" />
              </div>
            ))}
          </div>
        ) : tables.length === 0 ? (
          <div className="flex flex-col items-center gap-3 py-16 text-center">
            <div className="rounded-2xl bg-secondary p-4">
              <Table2 className="h-8 w-8 text-muted-foreground/40" />
            </div>
            <div>
              <h3 className="text-sm font-medium text-foreground">
                No tables yet
              </h3>
              <p className="mt-1 text-xs text-muted-foreground">
                Upload a document to extract tables automatically
              </p>
            </div>
          </div>
        ) : (
          <div className="flex flex-col gap-3">
            {tables.map((table) => (
              <div
                key={table.physical_table_name}
                className="glass-subtle group rounded-xl p-4 transition-colors hover:bg-[hsl(225,15%,14%)]"
              >
                {/* Table name and meta */}
                <div className="flex items-start justify-between gap-3">
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-2">
                      <Table2 className="h-4 w-4 shrink-0 text-[hsl(224,76%,56%)]" />
                      <h3 className="truncate font-mono text-sm font-semibold text-foreground">
                        {table.physical_table_name}
                      </h3>
                    </div>
                    {table.semantic_description && (
                      <p className="mt-1 text-xs leading-relaxed text-muted-foreground">
                        {table.semantic_description}
                      </p>
                    )}
                  </div>

                  <div className="flex shrink-0 items-center gap-2">
                    <Badge className="border border-[hsl(224,76%,56%)]/30 bg-[hsl(224,76%,56%)]/20 px-2 py-0 text-[10px] text-[hsl(224,76%,70%)]">
                      <Hash className="mr-0.5 h-2.5 w-2.5" />
                      {table.row_count.toLocaleString()} rows
                    </Badge>
                  </div>
                </div>

                {/* Headers */}
                <div className="mt-3">
                  <div className="mb-1.5 flex items-center gap-1 text-[10px] font-medium uppercase tracking-wider text-muted-foreground">
                    <Columns3 className="h-2.5 w-2.5" />
                    Columns
                  </div>
                  <div className="flex flex-wrap gap-1.5">
                    {table.headers.map((header) => (
                      <span
                        key={header}
                        className="rounded-md bg-secondary px-2 py-0.5 font-mono text-[11px] text-secondary-foreground"
                      >
                        {header}
                      </span>
                    ))}
                  </div>
                </div>

                {/* Source file */}
                <div className="mt-3 flex items-center gap-1.5 text-[10px] text-muted-foreground">
                  <FileText className="h-2.5 w-2.5" />
                  <span>Source: {table.original_filename}</span>
                </div>
              </div>
            ))}
          </div>
        )}
      </ScrollArea>
    </div>
  )
}
