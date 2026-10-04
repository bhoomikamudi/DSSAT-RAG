import React from 'react'
import {
  ResponsiveContainer,
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
} from 'recharts'
import { Statistics } from '@/types/chat'
import { formatStat } from '@/utils/format'

interface BreakdownValue {
  group_value?: string | number
  year?: string | number
  value?: number
  avg_value?: number
  count?: number
}

interface TrendChartProps {
  statistics: Statistics
}

export const TrendChart: React.FC<TrendChartProps> = ({ statistics }) => {
  const breakdown = statistics.breakdown as
    | {
        group_by?: string
        values?: BreakdownValue[]
      }
    | null
    | undefined

  if (!breakdown?.group_by || !breakdown.values?.length) {
    return null
  }

  const isYearChart = ['year', 'simulation_year'].includes(breakdown.group_by)

  const data = breakdown.values
    .map((item) => ({
      label: String(item.group_value ?? item.year),
      value: Number(item.value ?? item.avg_value),
      count: item.count,
    }))
    .filter((item) => Number.isFinite(item.value))
    .sort((a, b) =>
      isYearChart
        ? Number(a.label) - Number(b.label)
        : b.value - a.value
    )

  if (!data.length) {
    return null
  }

  const title = isYearChart
    ? `Average ${statistics.metric || 'value'} by year`
    : `Average ${statistics.metric || 'value'} by ${breakdown.group_by}`

  return (
    <div
      style={{
        width: '100%',
        height: isYearChart ? 300 : Math.max(300, data.length * 60),
        marginTop: 16,
        padding: 16,
        border: '1px solid #D1D5DB',
        borderRadius: 12,
        background: '#FFFFFF',
      }}
    >
      <h3 style={{ margin: '0 0 12px 0' }}>{title}</h3>

      <ResponsiveContainer width="100%" height="85%">
        {isYearChart ? (
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="label" />
            <YAxis tickFormatter={(v: number) => formatStat(v)} />
            <Tooltip
              formatter={(value: unknown) => [
                formatStat(Number(value)),
                'Average',
              ]}
            />
            <Line
              type="monotone"
              dataKey="value"
              stroke="#2563EB"
              strokeWidth={3}
              dot={{ r: 3 }}
            />
          </LineChart>
        ) : (
          <BarChart
            data={data}
            layout="vertical"
            margin={{ top: 5, right: 20, left: 35, bottom: 5 }}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" tickFormatter={(v: number) => formatStat(v)} />
            <YAxis type="category" dataKey="label" width={70} />
            <Tooltip
              formatter={(value: unknown) => [
                formatStat(Number(value)),
                'Average',
              ]}
            />
            <Bar dataKey="value" fill="#2563EB" radius={[0, 5, 5, 0]} />
          </BarChart>
        )}
      </ResponsiveContainer>
    </div>
  )
}

export default TrendChart
