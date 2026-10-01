import type { Metrics } from '../api/client'

interface Props {
  algorithm: string
  metrics: Metrics
}

function Stat({ label, value, good }: { label: string; value: string; good: boolean }) {
  return (
    <div className="bg-slate-900 rounded-lg p-3">
      <p className="text-xs text-slate-500 mb-1">{label}</p>
      <p className={`text-xl font-bold tabular-nums ${good ? 'text-emerald-400' : 'text-slate-300'}`}>
        {value}
      </p>
    </div>
  )
}

export default function MetricCard({ algorithm, metrics }: Props) {
  const isSTA1 = algorithm.includes('STA')
  const borderColor = isSTA1 ? 'border-indigo-500' : 'border-cyan-500'
  const labelColor = isSTA1 ? 'text-indigo-400' : 'text-cyan-400'
  const badge = isSTA1 ? 'bg-indigo-900/40 text-indigo-300' : 'bg-cyan-900/40 text-cyan-300'

  const pct = (n: number) => `${(n * 100).toFixed(1)}%`

  return (
    <div className={`bg-slate-800 rounded-xl border-l-4 ${borderColor} p-5 space-y-4`}>
      {/* Header */}
      <div className="flex items-center justify-between">
        <h4 className={`font-semibold text-base ${labelColor}`}>{algorithm}</h4>
        <span className={`text-xs font-medium px-2 py-0.5 rounded-full ${badge}`}>
          {metrics.n_watermarked} samples
        </span>
      </div>

      {/* Stats grid */}
      <div className="grid grid-cols-2 gap-2">
        <Stat label="TPR (Sensitivity)" value={pct(metrics.tpr)} good={metrics.tpr >= 0.5} />
        <Stat label="FPR" value={pct(metrics.fpr)} good={metrics.fpr <= 0.1} />
        <Stat label="Accuracy" value={pct(metrics.accuracy)} good={metrics.accuracy >= 0.7} />
        <Stat label="AUC-ROC" value={metrics.auc_roc.toFixed(3)} good={metrics.auc_roc >= 0.8} />
      </div>

      {/* Confusion counts */}
      <div className="flex gap-2 text-xs text-slate-500 pt-1 border-t border-slate-700">
        <span className="text-emerald-500">TP {metrics.tp}</span>
        <span>·</span>
        <span className="text-red-500">FP {metrics.fp}</span>
        <span>·</span>
        <span className="text-emerald-500">TN {metrics.tn}</span>
        <span>·</span>
        <span className="text-red-500">FN {metrics.fn}</span>
      </div>
    </div>
  )
}
