import ReactECharts from 'echarts-for-react'

interface Props {
  frequencies: Record<string, number>
  title?: string
}

export default function FrequencyChart({ frequencies, title = 'Number Frequency' }: Props) {
  const numbers = Object.keys(frequencies).map(Number).sort((a, b) => a - b)
  const values = numbers.map((n) => +(frequencies[n] * 100).toFixed(2))

  const option = {
    title: { text: title, textStyle: { fontSize: 14 } },
    tooltip: {
      trigger: 'axis',
      formatter: (params: { name: string; value: number }[]) =>
        `Number ${params[0].name}: ${params[0].value}%`,
    },
    xAxis: {
      type: 'category',
      data: numbers,
      name: 'Number',
    },
    yAxis: {
      type: 'value',
      name: 'Frequency (%)',
    },
    series: [
      {
        type: 'bar',
        data: values,
        itemStyle: { color: '#2563eb' },
      },
    ],
    grid: { left: 50, right: 20, bottom: 40, top: 50 },
  }

  return <ReactECharts option={option} style={{ height: 300 }} />
}
