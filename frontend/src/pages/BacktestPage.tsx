import { useQuery } from '@tanstack/react-query'
import { fetchJson } from '../api/client'
import { EmptyState, ErrorState, IntervalNote, LoadingState } from '../components/PageStates'
import type { BacktestMetric, BacktestReport } from '../types'

function percentage(value: number | null | undefined) {
  return value == null ? '—' : `${(value * 100).toFixed(1)}%`
}

function number(value: number | null | undefined) {
  return value == null ? '—' : value.toFixed(4)
}

function selectedPredictorName(value: BacktestReport['selected_predictor']) {
  if (typeof value === 'string') return value
  const name = value.name
  return typeof name === 'string' ? name : 'role_specific_blend'
}

export function BacktestPage() {
  const query = useQuery({
    queryKey: ['backtest-report'],
    queryFn: () => fetchJson<BacktestReport>('/api/backtest'),
  })

  if (query.isLoading) return <LoadingState label="Loading backtest results…" />
  if (query.error) return <ErrorState message={query.error.message} onRetry={() => void query.refetch()} />
  if (!query.data || Object.keys(query.data.folds).length === 0) return <EmptyState label="No backtest folds were returned." />

  const report = query.data
  const predictorName = selectedPredictorName(report.selected_predictor)
  const folds = Object.entries(report.folds)
  const roles = ['combined', 'batting', 'bowling']
  const rows = folds.flatMap(([foldName, fold]) => roles.map((role) => ({
    foldName,
    testYears: fold.test_years.join(', '),
    role,
    metrics: fold.predictors[predictorName]?.by_role[role] as BacktestMetric | undefined,
  })))

  return (
    <div className="space-y-6">
      <header className="space-y-2">
        <h1 className="text-3xl font-bold">Backtest</h1>
        <p className="text-sm text-slate-400">Selected predictor: {predictorName}. Metrics are shown by temporal fold and role.</p>
        <IntervalNote>The backtest endpoint does not include 80% intervals for fold metrics.</IntervalNote>
      </header>
      <aside className="rounded-xl border border-amber-800 bg-amber-950/40 p-4 text-sm text-amber-100">
        List precision and WPA uplift are retrospective on each fold’s realised outcomes; they are not prospective guarantees. Bowling results were weak.
      </aside>
      {rows.every((row) => !row.metrics) ? <EmptyState label="The selected predictor has no fold metrics." /> : (
        <div className="overflow-x-auto rounded-xl border border-slate-700">
          <table className="min-w-full divide-y divide-slate-700 text-left text-sm">
            <thead className="bg-slate-900 text-slate-300">
              <tr><th className="px-4 py-3">Fold</th><th className="px-4 py-3">Test years</th><th className="px-4 py-3">Role</th><th className="px-4 py-3">Spearman</th><th className="px-4 py-3">Precision@10 vs random</th><th className="px-4 py-3">Fold random baseline</th><th className="px-4 py-3">Predicted Top 10 uplift (WPA/ball)</th><th className="px-4 py-3">80% interval</th></tr>
            </thead>
            <tbody className="divide-y divide-slate-800">
              {rows.map((row) => (
                <tr key={`${row.foldName}-${row.role}`}>
                  <td className="px-4 py-3">{row.foldName.replaceAll('_', ' ')}</td>
                  <td className="px-4 py-3">{row.testYears}</td>
                  <td className="px-4 py-3">{row.role}</td>
                  <td className="px-4 py-3">{number(row.metrics?.spearman)}</td>
                  <td className="px-4 py-3">{percentage(row.metrics?.precision_at_10)}</td>
                  <td className="px-4 py-3">{percentage(row.metrics?.random_baseline_precision_at_10)}</td>
                  <td className="px-4 py-3">{number(row.metrics?.predicted_top10_wpa_uplift)}</td>
                  <td className="px-4 py-3 text-slate-500">Not supplied</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <p className="text-xs text-slate-500">Axes are not applicable; this view is a table. WPA uplift is the difference from the population mean, in win-probability added per ball.</p>
      <p className="text-xs text-slate-500">The expected random baseline is approximately 0.20 (20%); the exact fold-specific baseline is shown in the table.</p>
    </div>
  )
}
