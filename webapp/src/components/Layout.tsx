import { NavLink, Outlet } from 'react-router-dom'
import { LayoutDashboard, Search, BarChart2, FlaskConical, Play } from 'lucide-react'

const nav = [
  { to: '/', icon: LayoutDashboard, label: 'Dashboard' },
  { to: '/run', icon: Play, label: 'Run experiment' },
  { to: '/detect', icon: Search, label: 'Detect' },
  { to: '/charts', icon: BarChart2, label: 'Charts' },
]

export default function Layout() {
  return (
    <div className="flex h-screen bg-slate-900 text-slate-100 overflow-hidden">
      {/* ── Sidebar ── */}
      <aside className="w-60 shrink-0 bg-slate-800 border-r border-slate-700 flex flex-col">
        {/* Logo */}
        <div className="px-5 py-5 border-b border-slate-700 flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-indigo-600 flex items-center justify-center">
            <FlaskConical size={16} className="text-white" />
          </div>
          <div>
            <p className="text-sm font-bold text-white leading-tight">Watermark Lab</p>
            <p className="text-xs text-slate-500">LLM Evaluation</p>
          </div>
        </div>

        {/* Nav */}
        <nav className="flex-1 p-3 space-y-0.5">
          {nav.map(({ to, icon: Icon, label }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              className={({ isActive }) =>
                `flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all ${
                  isActive
                    ? 'bg-indigo-600/90 text-white shadow-sm'
                    : 'text-slate-400 hover:bg-slate-700/60 hover:text-white'
                }`
              }
            >
              <Icon size={17} />
              {label}
            </NavLink>
          ))}
        </nav>

        {/* Footer */}
        <div className="p-4 border-t border-slate-700 space-y-1">
          <p className="text-xs text-slate-500 font-mono">STA-1 · SynthID-Text</p>
          <p className="text-xs text-slate-600">GPT-2 · OPT-1.3B</p>
        </div>
      </aside>

      {/* ── Main ── */}
      <main className="flex-1 overflow-y-auto">
        <div className="max-w-5xl mx-auto px-8 py-8">
          <Outlet />
        </div>
      </main>
    </div>
  )
}
