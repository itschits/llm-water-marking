import { useState } from 'react'
import { AlertCircle, CheckCircle2, LoaderCircle, Play, SlidersHorizontal } from 'lucide-react'
import { runExperiment, type GeneratedTextResult, type RunExperimentRequest, type RunExperimentResponse } from '../api/client'
import ScoreBar from '../components/ScoreBar'

const DEFAULT_PROMPT = 'Artificial intelligence is transforming the world because'

export default function RunExperiment() {
  const [form, setForm] = useState<RunExperimentRequest>({
    model: 'qwen-2.5-1.5b', algorithm: 'sta1', prompt: DEFAULT_PROMPT, max_new_tokens: 120, temperature: 1,
  })
  const [result, setResult] = useState<RunExperimentResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const run = async () => {
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      setResult(await runExperiment(form))
    } catch (err: unknown) {
      const detail = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail
      setError(detail ?? 'Could not reach the API. Start the backend and try again.')
    } finally {
      setLoading(false)
    }
  }

  const update = <K extends keyof RunExperimentRequest>(key: K, value: RunExperimentRequest[K]) =>
    setForm(current => ({ ...current, [key]: value }))

  return (
    <div className="space-y-6 max-w-5xl">
      <div>
        <h2 className="text-2xl font-bold text-white">Run experiment</h2>
        <p className="text-sm text-slate-400 mt-1">Generate a plain and a watermarked continuation, then compare their detection statistics.</p>
      </div>

      <section className="bg-slate-800 border border-slate-700 rounded-xl p-5 space-y-5">
        <div className="flex items-center gap-2 text-sm font-semibold text-white"><SlidersHorizontal size={16} className="text-indigo-400" /> Experiment settings</div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <label className="space-y-1.5 text-xs text-slate-400">Model
            <select value={form.model} onChange={e => update('model', e.target.value as RunExperimentRequest['model'])} disabled={loading} className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:ring-2 focus:ring-indigo-500">
              <option value="gpt2">GPT-2 (124M) — fast baseline</option>
              <option value="qwen-2.5-1.5b">Qwen 2.5 1.5B — smaller and faster</option>
              <option value="qwen-2.5-3b">Qwen 2.5 3B — public, more resource intensive</option>
            </select>
          </label>
          <label className="space-y-1.5 text-xs text-slate-400">Watermark algorithm
            <select value={form.algorithm} onChange={e => update('algorithm', e.target.value as RunExperimentRequest['algorithm'])} disabled={loading} className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:ring-2 focus:ring-indigo-500">
              <option value="sta1">STA-1 — rejection sampling</option>
              <option value="synthid">SynthID-Text — tournament sampling</option>
            </select>
          </label>
        </div>

        {form.model === 'qwen-2.5-1.5b' && <p className="text-xs text-amber-300 bg-amber-950/30 border border-amber-700/40 rounded-lg px-3 py-2">The first run downloads Qwen 2.5 1.5B. Generation speed depends on your hardware.</p>}
        {form.model === 'qwen-2.5-3b' && <p className="text-xs text-amber-300 bg-amber-950/30 border border-amber-700/40 rounded-lg px-3 py-2">Qwen 2.5 3B may require about 12 GB RAM on CPU or 8 GB+ VRAM on GPU. The first run downloads the model.</p>}

        <label className="block space-y-1.5 text-xs text-slate-400">Prompt
          <textarea value={form.prompt} onChange={e => update('prompt', e.target.value)} disabled={loading} maxLength={2000} className="w-full h-24 resize-y bg-slate-900 border border-slate-700 rounded-lg p-3 text-sm text-slate-200 focus:outline-none focus:ring-2 focus:ring-indigo-500" />
        </label>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <label className="space-y-1.5 text-xs text-slate-400">Maximum new tokens: <span className="text-white">{form.max_new_tokens}</span>
            <input type="range" min="10" max="256" step="10" value={form.max_new_tokens} disabled={loading} onChange={e => update('max_new_tokens', Number(e.target.value))} className="w-full accent-indigo-500" />
          </label>
          <label className="space-y-1.5 text-xs text-slate-400">Temperature: <span className="text-white">{form.temperature.toFixed(1)}</span>
            <input type="range" min="0.1" max="2" step="0.1" value={form.temperature} disabled={loading} onChange={e => update('temperature', Number(e.target.value))} className="w-full accent-indigo-500" />
          </label>
        </div>

        <button onClick={run} disabled={loading || !form.prompt.trim()} className="w-full py-3 rounded-lg bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed text-white font-semibold text-sm flex items-center justify-center gap-2">
          {loading ? <LoaderCircle size={17} className="animate-spin" /> : <Play size={17} />}
          {loading ? 'Loading model and generating…' : 'Run experiment'}
        </button>
      </section>

      {error && <div className="flex gap-3 text-sm text-red-300 bg-red-950/40 border border-red-800/50 rounded-xl p-4"><AlertCircle size={17} className="shrink-0" />{error}</div>}

      {result && <section className="space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div><h3 className="font-semibold text-white">{result.model} × {result.algorithm}</h3><p className="text-xs text-slate-500 font-mono mt-1">{result.model_id} · completed in {result.elapsed_seconds}s</p></div>
          <span className="text-xs bg-slate-800 border border-slate-700 rounded-full px-3 py-1.5 text-slate-300">Detection threshold: z ≥ {result.threshold}</span>
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <GenerationCard title="Plain continuation" item={result.plain} threshold={result.threshold} color="indigo" />
          <GenerationCard title="Watermarked continuation" item={result.watermarked} threshold={result.threshold} color="cyan" />
        </div>
      </section>}
    </div>
  )
}

function GenerationCard({ title, item, threshold, color }: { title: string; item: GeneratedTextResult; threshold: number; color: 'indigo' | 'cyan' }) {
  const detectedStyle = item.detected ? 'bg-emerald-950/60 text-emerald-300 border-emerald-700/50' : 'bg-slate-700 text-slate-300 border-slate-600'
  return <article className="bg-slate-800 border border-slate-700 rounded-xl p-5 space-y-4">
    <div className="flex justify-between gap-3"><div><h4 className="font-medium text-white">{title}</h4><p className="text-xs text-slate-500 mt-1">{item.token_count} generated tokens</p></div><span className={`h-fit text-xs font-semibold border rounded-full px-2.5 py-1 flex items-center gap-1 ${detectedStyle}`}>{item.detected && <CheckCircle2 size={13} />}{item.detected ? 'WATERMARKED' : 'NOT DETECTED'}</span></div>
    <p className="min-h-32 bg-slate-900/70 rounded-lg p-3 text-sm leading-6 text-slate-300 whitespace-pre-wrap">{item.text || 'The model ended the continuation immediately.'}</p>
    <ScoreBar value={item.z_score} threshold={threshold} color={color} />
    <div className="grid grid-cols-2 gap-3 text-xs"><Metric label={item.signal_label} value={item.signal_detail} /><Metric label="P-value" value={item.p_value.toExponential(2)} /></div>
  </article>
}

function Metric({ label, value }: { label: string; value: string }) {
  return <div className="bg-slate-900/60 rounded-lg p-2.5"><p className="text-slate-500">{label}</p><p className="text-slate-200 mt-1 font-mono leading-5">{value}</p></div>
}
