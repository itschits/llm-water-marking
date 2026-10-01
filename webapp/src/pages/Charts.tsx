import { useEffect, useState } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  LineChart, Line, ResponsiveContainer, ReferenceLine,
} from 'recharts'
import { getResults, type ResultsData } from '../api/client'
import { RefreshCw, BarChart2 } from 'lucide-react'

// ── Helpers ────────────────────────────────────────────────────────────

function buildHistogram(wScores: number[], uScores: number[], bins = 12) {
  const all = [...wScores, ...uScores]
  if (!all.length) return []
  const min = Math.floor(Math.min(...all))
  const max = Math.ceil(Math.max(...all)) + 1
  const size = (max - min) / bins

  return Array.from({ length: bins }, (_, i) => {
    const lo = min + i * size
    const hi = lo + size
    return {
      bin: lo.toFixed(1),
      Watermarked: wScores.filter(s => s >= lo && s < hi).length,
      'Not Watermarked': uScores.filter(s => s >= lo && s < hi).length,
    }
  })
}

function buildROC(wScores: number[], uScores: number[]) {
  const allScores = [...wScores, ...uScores].sort((a, b) => b - a)
  const unique = [...new Set(allScores)]
  const nPos = wScores.length
  const nNeg = uScores.length
  if (!nPos || !nNeg) return []

  const points = unique.map(thresh => {
    const tp = wScores.filter(s => s >= thresh).length
    const fp = uScores.filter(s => s >= thresh).length
    return { fpr: +(fp / nNeg).toFixed(3), tpr: +(tp / nPos).toFixed(3) }
  })
  return [{ fpr: 0, tpr: 0 }, ...points, { fpr: 1, tpr: 1 }]
}

// ── Tooltip styles ─────────────────────────────────────────────────────
const tooltipStyle = {
  backgroundColor: '#1e293b',
  border: '1px solid #334155',
  borderRadius: 8,
  fontSize: 12,
  color: '#f8fafc',
}

// ── Component ──────────────────────────────────────────────────────────

export default function Charts() {
  const [results, setResults] = useState<ResultsData | null>(null)
  const [loading, setLoading] = useState(true)

  const load = () => {
    setLoading(true)
    getResults()
      .then(({ available, data }) => { if (available && data) setResults(data) })
      .finally(() => setLoading(false))
  }

  useEffect(() => { load() }, [])

  if (loading) {
    return (
      <div className="flex items-center gap-3 text-slate-400 mt-8">
        <RefreshCw size={16} className="animate-spin" />
        <span className="text-sm">Loading chart data…</span>
      </div>
    )
  }

  if (!results) {
    return (
      <div className="space-y-4">
        <h2 className="text-2xl font-bold text-white">Charts</h2>
        <div className="bg-slate-800 rounded-xl p-10 text-center border border-slate-700 border-dashed space-y-3">
          <BarChart2 size={32} className="mx-auto text-slate-600" />
          <p className="text-slate-400">No results to plot yet.</p>
          <code className="text-indigo-400 text-sm bg-slate-900 px-4 py-2 rounded-lg block w-fit mx-auto">
            python main.py --quick
          </code>
        </div>
      </div>
    )
  }

  const sections: React.ReactNode[] = []

  Object.entries(results).forEach(([modelName, algos]) => {
    Object.entries(algos).forEach(([algoName, data]) => {
      const isSTA1 = algoName.includes('STA')
      const wColor = isSTA1 ? '#818cf8' : '#22d3ee'
      const uColor = '#f87171'
      const rocColor = isSTA1 ? '#a78bfa' : '#67e8f9'
      const threshold = data.metrics.threshold

      const histData = buildHistogram(data.watermarked_scores, data.unwatermarked_scores)
      const rocData = buildROC(data.watermarked_scores, data.unwatermarked_scores)
      const accentText = isSTA1 ? 'text-indigo-400' : 'text-cyan-400'

      sections.push(
        <section key={`${modelName}-${algoName}`} className="space-y-4">
          {/* Section header */}
          <div className="flex items-center gap-3">
            <h3 className="text-sm font-semibold text-slate-300">
              {modelName} — <span className={accentText}>{algoName}</span>
            </h3>
            <div className="flex-1 h-px bg-slate-700" />
            <span className="text-xs text-slate-500">
              AUC {data.metrics.auc_roc.toFixed(3)} · TPR {(data.metrics.tpr * 100).toFixed(0)}%
            </span>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {/* Score Distribution */}
            <div className="bg-slate-800 rounded-xl p-5 border border-slate-700">
              <p className="text-xs font-medium text-slate-400 mb-4">
                Score Distribution (z-score bins)
              </p>
              <ResponsiveContainer width="100%" height={210}>
                <BarChart data={histData} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="bin" tick={{ fontSize: 10, fill: '#64748b' }} />
                  <YAxis tick={{ fontSize: 10, fill: '#64748b' }} />
                  <Tooltip contentStyle={tooltipStyle} />
                  <Legend wrapperStyle={{ fontSize: 11, color: '#94a3b8' }} />
                  <ReferenceLine
                    x={threshold.toFixed(1)}
                    stroke="#facc15"
                    strokeDasharray="4 2"
                    label={{ value: 'threshold', fill: '#facc15', fontSize: 10 }}
                  />
                  <Bar dataKey="Watermarked" fill={wColor} radius={[3, 3, 0, 0]} />
                  <Bar dataKey="Not Watermarked" fill={uColor} radius={[3, 3, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* ROC Curve */}
            <div className="bg-slate-800 rounded-xl p-5 border border-slate-700">
              <p className="text-xs font-medium text-slate-400 mb-4">
                ROC Curve (AUC = {data.metrics.auc_roc.toFixed(3)})
              </p>
              <ResponsiveContainer width="100%" height={210}>
                <LineChart data={rocData} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis
                    dataKey="fpr"
                    type="number"
                    domain={[0, 1]}
                    tickFormatter={v => v.toFixed(1)}
                    tick={{ fontSize: 10, fill: '#64748b' }}
                    label={{ value: 'FPR', position: 'insideBottom', offset: -2, fill: '#64748b', fontSize: 11 }}
                  />
                  <YAxis
                    dataKey="tpr"
                    domain={[0, 1]}
                    tickFormatter={v => v.toFixed(1)}
                    tick={{ fontSize: 10, fill: '#64748b' }}
                  />
                  <Tooltip contentStyle={tooltipStyle} formatter={v => Number(v).toFixed(3)} />
                  {/* Random baseline */}
                  <Line
                    data={[{ fpr: 0, tpr: 0 }, { fpr: 1, tpr: 1 }]}
                    type="linear"
                    dataKey="tpr"
                    stroke="#475569"
                    strokeDasharray="5 3"
                    dot={false}
                    name="Random"
                  />
                  <Line
                    type="monotone"
                    dataKey="tpr"
                    stroke={rocColor}
                    strokeWidth={2}
                    dot={false}
                    name={algoName}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </section>
      )
    })
  })

  return (
    <div className="space-y-8">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-white">Charts</h2>
          <p className="text-slate-400 mt-1 text-sm">Score distributions and ROC curves</p>
        </div>
        <button
          onClick={load}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-700 hover:bg-slate-600 text-slate-300 text-sm transition-colors"
        >
          <RefreshCw size={14} /> Refresh
        </button>
      </div>

      {sections}
    </div>
  )
}
