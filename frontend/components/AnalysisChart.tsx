/**
 * AnalysisChart Component
 * Renders Python analysis results: scatter plots with an optional fitted
 * line/curve, small multiples for grouped analyses, and a bar chart of group
 * means for grouped descriptive statistics. A statistics table is always
 * shown so values never depend on reading the chart.
 */

import React from 'react'
import {
  ResponsiveContainer,
  ComposedChart,
  Scatter,
  Line,
  BarChart,
  Bar,
  ErrorBar,
  XAxis,
  YAxis,
  ZAxis,
  CartesianGrid,
  Tooltip,
} from 'recharts'
import {
  AnalysisAxis,
  AnalysisResult,
  AnalysisSeries,
} from '@/types/chat'
import { formatCoefficient, formatStat } from '@/utils/format'

const POINT_COLOR = '#2563EB'
const FIT_COLOR = '#111827'
const GRID_COLOR = '#E5E7EB'
const AXIS_TEXT = '#6B7280'

const FIT_LABELS: Record<string, string> = {
  scatter_with_line: 'Fitted linear regression',
  scatter_with_curve: 'Fitted quadratic curve',
}

const TITLES: Record<string, string> = {
  correlation: 'Correlation',
  linear_regression: 'Linear regression',
  quadratic_regression: 'Quadratic regression',
  descriptive_statistics: 'Descriptive statistics',
}


const formatP = (value: number | null | undefined) =>
  typeof value !== 'number' ? '—' : value < 0.001 ? '< 0.001' : value.toFixed(3)

const axisName = (axis?: AnalysisAxis | null) =>
  axis ? `${axis.code}${axis.unit ? ` (${axis.unit})` : ''}` : ''

// Domains come from the observed points only; a fitted curve that
// extrapolates beyond them (e.g. below zero yield) is clipped, not zoomed to.
const domainOf = (series: AnalysisSeries[], key: 'x' | 'y'): [number, number] => {
  const values = series.flatMap((s) =>
    s.points
      .map((point) => point[key])
      .filter((v): v is number => typeof v === 'number')
  )
  if (!values.length) {
    return [0, 1]
  }
  const min = Math.min(...values)
  const max = Math.max(...values)
  // Round outward to a 1/2/5 step so ticks land on readable values, and
  // never pad a non-negative variable below zero.
  const rough = (max - min || Math.abs(max) || 1) / 5
  const magnitude = 10 ** Math.floor(Math.log10(rough))
  const step =
    [1, 2, 2.5, 5, 10].map((m) => m * magnitude).find((s) => s >= rough) ?? rough
  const lower = Math.floor(min / step) * step
  return [min >= 0 ? Math.max(0, lower) : lower, Math.ceil(max / step) * step]
}

const cardStyle: React.CSSProperties = {
  width: '100%',
  marginTop: 16,
  padding: 16,
  border: '1px solid #D1D5DB',
  borderRadius: 12,
  background: '#FFFFFF',
  boxSizing: 'border-box',
}

const tickStyle = { fontSize: 11, fill: AXIS_TEXT }

interface ScatterPanelProps {
  series: AnalysisSeries
  xAxis?: AnalysisAxis | null
  yAxis?: AnalysisAxis | null
  xDomain: [number, number]
  yDomain: [number, number]
  height: number
  compact?: boolean
}

const ScatterPanel: React.FC<ScatterPanelProps> = ({
  series,
  xAxis,
  yAxis,
  xDomain,
  yDomain,
  height,
  compact = false,
}) => (
  <div style={{ width: '100%', height }}>
    <ResponsiveContainer width="100%" height="100%">
      <ComposedChart margin={{ top: 8, right: 12, bottom: compact ? 4 : 20, left: compact ? 0 : 12 }}>
        <CartesianGrid stroke={GRID_COLOR} strokeDasharray="3 3" />
        <XAxis
          type="number"
          dataKey="x"
          domain={xDomain}
          allowDataOverflow
          tick={tickStyle}
          tickFormatter={(v: number) => formatStat(v)}
          stroke={GRID_COLOR}
          label={
            compact
              ? undefined
              : { value: axisName(xAxis), position: 'insideBottom', offset: -12, fontSize: 12, fill: AXIS_TEXT }
          }
        />
        <YAxis
          type="number"
          dataKey="y"
          domain={yDomain}
          allowDataOverflow
          tick={tickStyle}
          tickFormatter={(v: number) => formatStat(v)}
          stroke={GRID_COLOR}
          width={compact ? 44 : 56}
          label={
            compact
              ? undefined
              : { value: axisName(yAxis), angle: -90, position: 'insideLeft', fontSize: 12, fill: AXIS_TEXT }
          }
        />
        <ZAxis range={[compact ? 12 : 20, compact ? 12 : 20]} />
        <Tooltip
          cursor={{ strokeDasharray: '3 3' }}
          labelFormatter={() => ''}
          labelStyle={{ display: 'none' }}
          formatter={(value: unknown, name: unknown) => [
            formatStat(Number(value)),
            name === 'x' ? xAxis?.code ?? 'x' : yAxis?.code ?? 'y',
          ]}
        />
        <Scatter
          data={series.points}
          fill={POINT_COLOR}
          fillOpacity={0.45}
          isAnimationActive={false}
        />
        {series.fit && series.fit.length > 1 && (
          <Line
            data={series.fit}
            dataKey="y"
            type="monotone"
            stroke={FIT_COLOR}
            strokeWidth={2}
            dot={false}
            activeDot={false}
            isAnimationActive={false}
            tooltipType="none"
          />
        )}
      </ComposedChart>
    </ResponsiveContainer>
  </div>
)

const StatsTable: React.FC<{ analysis: AnalysisResult }> = ({ analysis }) => {
  const isDescriptive = analysis.analysis_type === 'descriptive_statistics'
  const showR = analysis.analysis_type !== 'quadratic_regression'
  const showR2 = analysis.analysis_type !== 'correlation' && !isDescriptive
  const rows = analysis.groups.length
    ? analysis.groups.map((g) => ({ ...g, label: g.group ?? '' }))
    : [{ ...analysis, label: 'All records', status: 'ok' as const, message: null }]

  if (isDescriptive) {
    return null
  }

  const cell: React.CSSProperties = { padding: '4px 8px', borderBottom: '1px solid #E5E7EB', textAlign: 'right' }
  const head: React.CSSProperties = { ...cell, color: '#374151', fontWeight: 600 }

  return (
    <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12, marginTop: 12, color: '#111827' }}>
      <thead>
        <tr>
          <th style={{ ...head, textAlign: 'left' }}>{analysis.group_by ?? 'Subset'}</th>
          <th style={head}>n</th>
          {showR && <th style={head}>r</th>}
          {showR2 && <th style={head}>R²</th>}
          <th style={head}>p</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((row) => (
          <tr key={row.label}>
            <td style={{ ...cell, textAlign: 'left' }}>{row.label}</td>
            <td style={cell}>{row.sample_size.toLocaleString()}</td>
            {row.status === 'ok' ? (
              <>
                {showR && <td style={cell}>{formatCoefficient(row.correlation, 3)}</td>}
                {showR2 && <td style={cell}>{formatCoefficient(row.r_squared, 3)}</td>}
                <td style={cell}>{formatP(row.p_value)}</td>
              </>
            ) : (
              <td style={{ ...cell, textAlign: 'left', color: '#6B7280' }} colSpan={1 + Number(showR) + Number(showR2)}>
                Insufficient data
              </td>
            )}
          </tr>
        ))}
      </tbody>
    </table>
  )
}

export const AnalysisChart: React.FC<{ analysis: AnalysisResult }> = ({ analysis }) => {
  const data = analysis.chart_data
  if (analysis.status !== 'ok' || !analysis.chart_type || !data) {
    return null
  }

  const title = `${TITLES[analysis.analysis_type] ?? 'Analysis'}: ${
    data.y?.code ?? ''
  }${data.x ? ` vs ${data.x.code}` : ''}`

  if (analysis.chart_type === 'bar') {
    const bars = (data.bars ?? []).filter((b) => typeof b.mean === 'number')
    return (
      <div style={cardStyle}>
        <h3 style={{ margin: '0 0 12px 0', fontSize: 15 }}>
          Mean {axisName(data.y)} by {data.group_by}
        </h3>
        <div style={{ width: '100%', height: Math.max(220, bars.length * 44) }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={bars} layout="vertical" margin={{ top: 4, right: 24, left: 16, bottom: 4 }}>
              <CartesianGrid stroke={GRID_COLOR} strokeDasharray="3 3" horizontal={false} />
              <XAxis type="number" tick={tickStyle} stroke={GRID_COLOR} tickFormatter={(v: number) => formatStat(v)} />
              <YAxis type="category" dataKey="group" tick={tickStyle} width={80} stroke={GRID_COLOR} />
              <Tooltip
                formatter={(value: unknown) => [formatStat(Number(value)), 'Mean']}
              />
              <Bar dataKey="mean" fill={POINT_COLOR} radius={[0, 4, 4, 0]} barSize={18}>
                <ErrorBar dataKey="std" width={4} stroke={FIT_COLOR} direction="x" />
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
        <p style={{ margin: '8px 0 0', fontSize: 12, color: AXIS_TEXT }}>
          Whiskers show ±1 standard deviation.
        </p>
      </div>
    )
  }

  const series = data.series ?? []
  if (!series.length) {
    return null
  }

  // Shared domains keep small multiples directly comparable.
  const xDomain = domainOf(series, 'x')
  const yDomain = domainOf(series, 'y')
  const grouped = series.length > 1
  const fitLabel = FIT_LABELS[analysis.chart_type]

  return (
    <div style={cardStyle}>
      <h3 style={{ margin: 0, fontSize: 15 }}>{title}</h3>
      <p style={{ margin: '4px 0 12px', fontSize: 12, color: AXIS_TEXT }}>
        {grouped ? `x: ${axisName(data.x)} · y: ${axisName(data.y)} · ` : ''}
        {fitLabel ? `Line: ${fitLabel.toLowerCase()} · ` : ''}
        {data.sampled
          ? `Showing ${data.plotted_points?.toLocaleString()} of ${data.total_points?.toLocaleString()} records (statistics use all records)`
          : `${data.total_points?.toLocaleString()} records`}
      </p>

      {grouped ? (
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))',
            gap: 12,
          }}
        >
          {series.map((s) => (
            <div key={s.name}>
              <div style={{ fontSize: 12, fontWeight: 600, color: '#111827', marginBottom: 2 }}>
                {s.name}
                <span style={{ fontWeight: 400, color: AXIS_TEXT }}>
                  {' '}
                  {typeof s.correlation === 'number' ? ` · r = ${formatCoefficient(s.correlation, 2)}` : ''}
                  {typeof s.r_squared === 'number' ? ` · R² = ${formatCoefficient(s.r_squared, 2)}` : ''}
                </span>
              </div>
              <ScatterPanel
                series={s}
                xAxis={data.x}
                yAxis={data.y}
                xDomain={xDomain}
                yDomain={yDomain}
                height={190}
                compact
              />
            </div>
          ))}
        </div>
      ) : (
        <ScatterPanel
          series={series[0]}
          xAxis={data.x}
          yAxis={data.y}
          xDomain={xDomain}
          yDomain={yDomain}
          height={320}
        />
      )}

      <StatsTable analysis={analysis} />
    </div>
  )
}

export default AnalysisChart
