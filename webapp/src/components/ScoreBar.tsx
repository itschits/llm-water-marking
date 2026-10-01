interface Props {
  value: number     // current z-score (may be negative)
  threshold: number // detection threshold
  color: 'indigo' | 'cyan'
}

export default function ScoreBar({ value, threshold, color }: Props) {
  // Map z-score to a [0, 100] display range centred at 0
  const displayMax = Math.max(threshold * 2.5, Math.abs(value) * 1.3, threshold + 3)
  const clampedPos = Math.max(0, Math.min(value, displayMax))
  const fillPct = (clampedPos / displayMax) * 100
  const threshPct = (threshold / displayMax) * 100

  const fillColor = color === 'indigo' ? '#818cf8' : '#22d3ee'
  const bgFill = value >= threshold
    ? (color === 'indigo' ? '#4338ca' : '#0e7490')
    : '#1e293b'

  return (
    <div className="space-y-1">
      <div
        className="relative h-7 rounded-lg overflow-hidden"
        style={{ background: bgFill }}
      >
        {/* Fill bar */}
        <div
          className="absolute top-0 left-0 h-full rounded-lg transition-all duration-700"
          style={{ width: `${fillPct}%`, background: fillColor, opacity: 0.85 }}
        />

        {/* Threshold marker */}
        <div
          className="absolute top-0 bottom-0 w-0.5 bg-yellow-400/80"
          style={{ left: `${threshPct}%` }}
        />

        {/* Z-score label */}
        <span className="absolute inset-0 flex items-center justify-center text-xs font-mono font-semibold text-white/90">
          z = {value > 0 ? '+' : ''}{value.toFixed(2)}
        </span>
      </div>

      {/* Scale labels */}
      <div className="flex justify-between text-xs text-slate-600 px-0.5">
        <span>0</span>
        <span className="text-yellow-600">threshold {threshold}</span>
        <span>{displayMax.toFixed(0)}</span>
      </div>
    </div>
  )
}
