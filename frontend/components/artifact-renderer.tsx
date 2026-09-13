"use client"

/**
 * Renders an artifact produced by the backend's deterministic classifier.
 *
 * The chart type and the data are decided server-side and arrive fixed. This
 * component chooses nothing about what the data means: it only draws what it
 * is handed. That is what keeps the guarantee intact end to end - no step
 * between the verified SQL rows and the pixels can reinterpret them.
 *
 * Built on components/ui/chart.tsx and components/ui/table.tsx, both of which
 * shipped with the project and had never been imported anywhere.
 */

import * as React from "react"
import {
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  Scatter,
  ScatterChart,
  XAxis,
  YAxis,
} from "recharts"
import { AlertTriangle, Database } from "lucide-react"

import {
  ChartContainer,
  ChartLegend,
  ChartLegendContent,
  ChartTooltip,
  ChartTooltipContent,
  type ChartConfig,
} from "@/components/ui/chart"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import type { Artifact, ArtifactField, DataCell } from "@/lib/types"

/**
 * Map each series onto one of the --chart-1..5 tokens in app/globals.css.
 *
 * ChartStyle (inside ui/chart.tsx) turns this into `--color-<key>` custom
 * properties, which the marks below reference as `var(--color-key)`. Those are
 * attribute values Tailwind never scans, so unlike a dynamically built
 * `fill-chart-${i}` class they cannot be purged from the production build.
 */
function buildChartConfig(series: ArtifactField[]): ChartConfig {
  return Object.fromEntries(
    series.map((field, index) => [
      field.key,
      {
        label: field.label,
        color: `hsl(var(--chart-${field.color_index ?? (index % 5) + 1}))`,
      },
    ]),
  ) satisfies ChartConfig
}

function formatValue(value: DataCell, field?: ArtifactField): string {
  if (value === null || value === undefined) return "—"
  if (typeof value === "number") {
    const formatted = new Intl.NumberFormat(undefined, {
      minimumFractionDigits: field?.decimals ?? 0,
      maximumFractionDigits: field?.decimals ?? 2,
    }).format(value)
    if (field?.unit === "%") return `${formatted}%`
    return field?.unit ? `${formatted} ${field.unit}` : formatted
  }
  return String(value)
}

/** Compact axis ticks: 1.2M rather than 1,200,000. */
function abbreviate(value: number): string {
  const abs = Math.abs(value)
  if (abs >= 1e9) return `${(value / 1e9).toFixed(1)}B`
  if (abs >= 1e6) return `${(value / 1e6).toFixed(1)}M`
  if (abs >= 1e3) return `${(value / 1e3).toFixed(1)}K`
  return String(Number(value.toFixed(2)))
}

/**
 * A malformed data/dataKey pair makes recharts throw, and an uncaught throw
 * inside a chat bubble unmounts the whole conversation. A chart bug must never
 * cost the user their transcript.
 */
class ChartBoundary extends React.Component<
  { children: React.ReactNode; fallback: React.ReactNode },
  { failed: boolean }
> {
  state = { failed: false }

  static getDerivedStateFromError() {
    return { failed: true }
  }

  render() {
    return this.state.failed ? this.props.fallback : this.props.children
  }
}

function DataTable({ artifact }: { artifact: Artifact }) {
  return (
    <div className="max-h-[360px] overflow-auto">
      <Table>
        <TableHeader>
          <TableRow>
            {artifact.columns.map((column) => (
              <TableHead
                key={column.key}
                className={column.align === "right" ? "text-right" : ""}
              >
                {column.label}
              </TableHead>
            ))}
          </TableRow>
        </TableHeader>
        <TableBody>
          {artifact.data.map((row, index) => (
            <TableRow key={index}>
              {artifact.columns.map((column) => (
                <TableCell
                  key={column.key}
                  className={
                    column.align === "right" ? "text-right tabular-nums" : ""
                  }
                >
                  {formatValue(row[column.key], column)}
                </TableCell>
              ))}
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  )
}

export function ArtifactRenderer({ artifact }: { artifact: Artifact }) {
  const { encoding, data, type, kind, provenance } = artifact
  const series = encoding?.y ?? []
  const config = React.useMemo(() => buildChartConfig(series), [series])
  const xKey = encoding?.x?.key ?? ""
  const primary = series[0]

  const chartHeight =
    type === "bar_horizontal" ? Math.min(data.length * 28 + 56, 900) : 260

  const renderChart = () => {
    switch (type) {
      case "line":
        return (
          <LineChart data={data} margin={{ left: 4, right: 12, top: 8 }}>
            <CartesianGrid vertical={false} strokeOpacity={0.15} />
            <XAxis
              dataKey={xKey}
              tickLine={false}
              axisLine={false}
              tickMargin={8}
              minTickGap={24}
              fontSize={11}
            />
            <YAxis
              tickLine={false}
              axisLine={false}
              width={52}
              fontSize={11}
              tickFormatter={abbreviate}
            />
            <ChartTooltip content={<ChartTooltipContent indicator="line" />} />
            {series.length > 1 && <ChartLegend content={<ChartLegendContent />} />}
            {series.map((field) => (
              <Line
                key={field.key}
                dataKey={field.key}
                type="monotone"
                dot={false}
                strokeWidth={2}
                // A gap in the data is a gap, not a straight line through it.
                connectNulls={false}
                stroke={`var(--color-${field.key})`}
                isAnimationActive={false}
              />
            ))}
          </LineChart>
        )

      case "bar":
      case "bar_grouped":
        return (
          <BarChart data={data} margin={{ left: 4, right: 12, top: 8 }}>
            <CartesianGrid vertical={false} strokeOpacity={0.15} />
            <XAxis
              dataKey={xKey}
              tickLine={false}
              axisLine={false}
              tickMargin={8}
              interval={0}
              fontSize={11}
              tickFormatter={(value) => String(value).slice(0, 14)}
            />
            <YAxis
              tickLine={false}
              axisLine={false}
              width={52}
              fontSize={11}
              tickFormatter={abbreviate}
            />
            <ChartTooltip content={<ChartTooltipContent />} />
            {series.length > 1 && <ChartLegend content={<ChartLegendContent />} />}
            {series.map((field) => (
              <Bar
                key={field.key}
                dataKey={field.key}
                radius={[4, 4, 0, 0]}
                fill={`var(--color-${field.key})`}
                isAnimationActive={false}
              />
            ))}
          </BarChart>
        )

      case "bar_horizontal":
        return (
          <BarChart data={data} layout="vertical" margin={{ left: 8, right: 20 }}>
            <CartesianGrid horizontal={false} strokeOpacity={0.15} />
            <XAxis type="number" tickLine={false} axisLine={false} fontSize={11} tickFormatter={abbreviate} />
            <YAxis
              type="category"
              dataKey={xKey}
              width={150}
              tickLine={false}
              axisLine={false}
              fontSize={11}
              tickFormatter={(value) => String(value).slice(0, 22)}
            />
            <ChartTooltip content={<ChartTooltipContent />} />
            {primary && (
              <Bar
                dataKey={primary.key}
                radius={[0, 4, 4, 0]}
                fill={`var(--color-${primary.key})`}
                isAnimationActive={false}
              />
            )}
          </BarChart>
        )

      case "scatter":
        return (
          <ScatterChart margin={{ left: 4, right: 12, top: 8 }}>
            <CartesianGrid strokeOpacity={0.15} />
            <XAxis
              type="number"
              dataKey={xKey}
              name={encoding?.x?.label}
              tickLine={false}
              fontSize={11}
              tickFormatter={abbreviate}
            />
            {primary && (
              <YAxis
                type="number"
                dataKey={primary.key}
                name={primary.label}
                width={52}
                fontSize={11}
                tickFormatter={abbreviate}
              />
            )}
            <ChartTooltip
              content={<ChartTooltipContent />}
              cursor={{ strokeDasharray: "3 3" }}
            />
            {primary && (
              <Scatter
                data={data}
                fill={`var(--color-${primary.key})`}
                isAnimationActive={false}
              />
            )}
          </ScatterChart>
        )

      default:
        return null
    }
  }

  const chart = renderChart()
  const hasCaveat =
    provenance.truncated || provenance.partial || artifact.notes.length > 0

  return (
    <figure className="glass-subtle mt-3 overflow-hidden rounded-xl border border-[hsl(var(--glass-border))]">
      <figcaption className="flex items-center justify-between gap-2 border-b border-[hsl(var(--glass-border))] px-3 py-1.5">
        <span className="flex min-w-0 items-center gap-1.5 text-[10px] font-medium text-muted-foreground">
          <Database className="h-3 w-3 shrink-0" aria-hidden />
          {/* On HYBRID the prose is an LLM merge of two model outputs while the
              artifact is computed straight from the rows. Worth saying which. */}
          <span className="truncate">
            {provenance.answer_is_llm_merged ? "Computed from SQL" : "From your data"}
          </span>
        </span>
        <span className="shrink-0 tabular-nums text-[10px] text-muted-foreground">
          {provenance.row_count.toLocaleString()}{" "}
          {provenance.row_count === 1 ? "row" : "rows"}
        </span>
      </figcaption>

      <div className="p-3">
        {kind === "chart" && chart && (
          <ChartBoundary fallback={<DataTable artifact={artifact} />}>
            <ChartContainer
              config={config}
              className="aspect-auto w-full"
              style={{ height: chartHeight }}
            >
              {chart}
            </ChartContainer>
          </ChartBoundary>
        )}

        {kind === "table" && <DataTable artifact={artifact} />}

        {kind === "scalar" && (
          <div className="py-1">
            <div className="text-2xl font-semibold tabular-nums text-foreground">
              {formatValue(
                artifact.value ?? null,
                artifact.unit
                  ? { ...artifact.columns[0], unit: artifact.unit }
                  : artifact.columns[0],
              )}
            </div>
            <div className="text-[10px] text-muted-foreground">
              {artifact.columns[0]?.label}
            </div>
          </div>
        )}

        {kind === "empty" && (
          <p className="py-2 text-xs text-muted-foreground">
            The query ran but matched no rows.
          </p>
        )}
      </div>

      {hasCaveat && (
        <div className="flex items-start gap-1.5 border-t border-[hsl(var(--glass-border))] px-3 py-1.5 text-[10px] text-muted-foreground">
          <AlertTriangle className="mt-px h-3 w-3 shrink-0" aria-hidden />
          <span>
            {provenance.truncated &&
              `Showing ${provenance.data_rows} of ${provenance.row_count} rows. `}
            {provenance.partial && "Result hit the query limit; more rows may exist. "}
            {artifact.notes.join(" ")}
          </span>
        </div>
      )}
    </figure>
  )
}
