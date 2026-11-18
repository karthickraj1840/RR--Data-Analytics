import { Link } from 'react-router-dom'

export function Layout({ children }: { children: React.ReactNode }) {
  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 md:flex">
      <aside className="border-b border-slate-800 bg-slate-900/80 p-4 md:min-h-screen md:w-60 md:shrink-0 md:border-b-0 md:border-r">
        <Link to="/" className="block text-xl font-bold text-cyan-300">SupeRR Pressure Lab</Link>
        <nav aria-label="Primary navigation" className="mt-5 grid grid-cols-2 gap-2 text-sm sm:grid-cols-3 md:grid-cols-1">
          <Link to="/overview" className="rounded px-3 py-2 hover:bg-slate-800">Overview</Link>
          <Link to="/top10" className="rounded px-3 py-2 hover:bg-slate-800">Top 10</Link>
          <Link to="/data-health" className="rounded px-3 py-2 hover:bg-slate-800">Data Health</Link>
          <Link to="/pressure" className="rounded px-3 py-2 hover:bg-slate-800">Pressure Lab</Link>
          <Link to="/leaderboard" className="rounded px-3 py-2 hover:bg-slate-800">Leaderboard</Link>
          <Link to="/compare" className="rounded px-3 py-2 hover:bg-slate-800">Compare</Link>
          <Link to="/sensitivity" className="rounded px-3 py-2 hover:bg-slate-800">Sensitivity</Link>
          <Link to="/backtest" className="rounded px-3 py-2 hover:bg-slate-800">Backtest</Link>
          <Link to="/squad-fit" className="rounded px-3 py-2 hover:bg-slate-800">Squad Fit</Link>
          <Link to="/shortlist" className="rounded px-3 py-2 hover:bg-slate-800">Shortlist</Link>
          <Link to="/methodology" className="rounded px-3 py-2 hover:bg-slate-800">Methodology</Link>
        </nav>
        <p className="mt-5 text-xs text-slate-500">Open Player Profile from any player link.</p>
      </aside>
      <main className="min-w-0 flex-1 p-4 sm:p-6">{children}</main>
    </div>
  )
}
