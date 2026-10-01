import { useEffect, useState } from 'react'
import { getResults, getHealth, type ResultsData } from '../api/client'
import MetricCard from '../components/MetricCard'
import { Activity, FileJson, RefreshCw } from 'lucide-react'

export default function Dashboard() {
  const [results, setResults] = useState<ResultsData | null>(null)
  const [filename, setFilename] = useState<string | null>(null)
  const [apiOk, setApiOk] = useState<boolean | null>(null)
  const [loading, setLoading] = useState(true)

  const load = () => {
    setLoading(true)
    getHealth()
      .then(() => setApiOk(true))
      .catch(() => setApiOk(false))

    getResults()
      .then(({ available, data, filename: fn }) => {
        if (available && data) { setResults(data); setFilename(fn) }
      })
      .finally(() => setLoading(false))
  }

  useEffect(() => { load() }, [])

  const totalModels = results ? Object.keys(results).length : 0
  const allMetrics = results
    ? Object.values(results).flatMap(a => Object.values(a).map(d => d.metrics))
    : []
  const avgAUC = allMetrics.length
    ? (allMetrics.reduce((s, m) => s + m.auc_roc, 0) / allMetrics.length).toFixed(3)
    : '—'

  return (
    <div className="space-y-8">
      {/* ── Header ── */}
      <div className="flex items-start justify-between">
        <div>
          <h2 className="text-2xl font-bold text-white">Dashboard</h2>
          <p className="text-slate-400 mt-1 text-sm">
            Latest watermarking experiment results
          </p>
        </div>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 text-sm">
            <span
              className={`w-2 h-2 rounded-full ${
                apiOk === true ? 'bg-emerald-400 shadow-[0_0_6px_#34d399]'
                : apiOk === false ? 'bg-red-400'
                : 'bg-slate-500 animate-pulse'
              }`}
            />
            <span className="text-slate-400">
              {apiOk === true ? 'API online' : apiOk === false ? 'API offline' : 'Checking…'}
            </span>
          </div>
          <button
            onClick={load}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-700 hover:bg-slate-600 text-slate-300 text-sm transition-colors"
          >
            <RefreshCw size={14} /> Refresh
          </button>
        </div>
      </div>

      {/* ── Summary Stat Bar ── */}
      {results && (
        <div className="grid grid-cols-3 gap-4">
          {[
            { icon: Activity, label: 'Models evaluated', value: String(totalModels) },
            { icon: Activity, label: 'Algorithms', value: '2 (STA-1, SynthID)' },
            { icon: Activity, label: 'Avg AUC-ROC', value: avgAUC },
          ].map(({ icon: Icon, label, value }) => (
            <div key={label} className="bg-slate-800 rounded-xl p-4 border border-slate-700">
              <p className="text-xs text-slate-500 mb-1 flex items-center gap-1.5"><Icon size={13} />{label}</p>
              <p className="text-2xl font-bold text-white">{value}</p>
            </div>
          ))}
        </div>
      )}

      {/* ── States ── */}
      {loading && (
        <div className="flex items-center gap-3 text-slate-400">
          <RefreshCw size={16} className="animate-spin" />
          <span className="text-sm">Loading results…</span>
        </div>
      )}

      {!loading && !results && (
        <div className="bg-slate-800 rounded-xl p-10 text-center border border-slate-700 border-dashed space-y-3">
          <FileJson size={32} className="mx-auto text-slate-600" />
          <p className="text-slate-400">No experiment results found yet.</p>
          <p className="text-sm text-slate-500">Run the following command, then refresh:</p>
          <code className="block text-indigo-400 bg-slate-900 rounded-lg px-4 py-2 text-sm w-fit mx-auto">
            python main.py --quick
          </code>
        </div>
      )}

      {/* ── Per-model results ── */}
      {results && Object.entries(results).map(([modelName, algos]) => (
        <section key={modelName} className="space-y-4">
          <div className="flex items-center gap-3">
            <h3 className="text-base font-semibold text-white">{modelName}</h3>
            <div className="flex-1 h-px bg-slate-700" />
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {Object.entries(algos).map(([algoName, data]) => (
              <MetricCard key={algoName} algorithm={algoName} metrics={data.metrics} />
            ))}
          </div>
        </section>
      ))}

      {/* ── Source file ── */}
      {filename && (
        <p className="text-xs text-slate-600 flex items-center gap-1.5">
          <FileJson size={12} /> {filename}
        </p>
      )}
    </div>
  )
}
