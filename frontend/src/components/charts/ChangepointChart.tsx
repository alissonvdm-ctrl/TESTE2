import ReactECharts from 'echarts-for-react'
import type { RegimeStats } from '../../types'

interface Props {
  regimes: RegimeStats[]
  title?: string
}

export default function ChangepointChart({ regimes, title = 'Regime Segments' }: Props) {
  if (regimes.length === 0) {
    return <p className="text-sm text-gray-500">No regimes detected.</p>
  }

  const option = {
    title: { text: title, textStyle: { fontSize: 14 } },
    tooltip: {
      trigger: 'item',
      formatter: (p: { name: string; value: number[] }) =>
        `Regime ${p.name}: rows ${p.value[0]}–${p.value[1]} (mean ${p.value[2].toFixed(2)})`,
    },
    xAxis: { type: 'value', name: 'Row Index' },
    yAxis: {
      type: 'category',
      data: regimes.map((r) => `R${r.regime_id}`),
    },
    series: [
      {
        type: 'bar',
        data: regimes.map((r) => ({
          name: String(r.regime_id),
          value: [r.start_index, r.end_index, r.mean],
        })),
        encode: { x: [0, 1], y: 'y' },
        itemStyle: { color: '#7c3aed' },
      },
    ],
    grid: { left: 60, right: 20, bottom: 40, top: 50 },
  }

  return <ReactECharts option={option} style={{ height: Math.max(200, regimes.length * 40) }} />
}
