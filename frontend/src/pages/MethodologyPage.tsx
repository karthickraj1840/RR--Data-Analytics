import { useQuery } from '@tanstack/react-query'
import { fetchJson } from '../api/client'
import { EmptyState, ErrorState, IntervalNote, LoadingState } from '../components/PageStates'
import type { AppConfig, BacktestReport, DataQuality } from '../types'

function nested(config: Record<string, unknown>, path: string[]) {
  let value: unknown = config
  for (const key of path) {
    if (value == null || typeof value !== 'object') return undefined
    value = (value as Record<string, unknown>)[key]
  }
  return value
}

function display(value: unknown) {
  if (value == null || value === '') return 'Not provided'
  if (Array.isArray(value)) return value.join('–')
  if (typeof value === 'number' && value > 0 && value < 1) return `${(value * 100).toFixed(1)}%`
  return String(value)
}

export function MethodologyPage() {
  const query = useQuery({
    queryKey: ['methodology'],
    queryFn: async () => {
      const [config, quality, backtest] = await Promise.all([
        fetchJson<AppConfig>('/api/config'),
        fetchJson<DataQuality>('/api/data-quality'),
        fetchJson<BacktestReport>('/api/backtest'),
      ])
      return { config, quality, backtest }
    },
  })

  if (query.isLoading) return <LoadingState label="Loading methodology and current configuration…" />
  if (query.error) return <ErrorState message={query.error.message} onRetry={() => void query.refetch()} />
  if (!query.data) return <EmptyState label="No methodology data was returned." />

  const { config, quality, backtest } = query.data
  if (Object.keys(config.config).length === 0 && Object.keys(quality.exclusions).length === 0) {
    return <EmptyState label="The API returned no configuration or exclusion detail for this methodology view." />
  }
  const thresholds: Array<[string, string[]]> = [
    ['Eligibility pressure-ball minimum', ['eligibility_min_pressure_balls']],
    ['Minimum IPL ball share', ['min_ipl_share']],
    ['All-rounder batting pressure balls', ['all_rounder_eligibility', 'min_batting_pressure_balls']],
    ['All-rounder bowling pressure balls', ['all_rounder_eligibility', 'min_bowling_pressure_balls']],
    ['High-leverage quantile', ['pressure', 'high_leverage_quantile']],
    ['Hard-chase required run rate', ['pressure', 'hard_chase_rrr']],
    ['Hard-chase balls remaining maximum', ['pressure', 'hard_chase_balls_left_max']],
    ['Death-overs range', ['pressure', 'death_overs']],
    ['Collapse wicket threshold', ['pressure', 'collapse_wickets']],
    ['Collapse over threshold', ['pressure', 'collapse_over']],
    ['Tight-finish balls remaining maximum', ['pressure', 'tight_finish_balls_left_max']],
    ['Tight-finish win-probability range', ['pressure', 'tight_finish_wp_min']],
  ]

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-3xl font-bold">Methodology</h1>
        <p className="mt-2 text-sm text-slate-400">Plain-language definitions and current API configuration.</p>
      </header>

      <section className="space-y-3 rounded-xl border border-slate-700 bg-slate-900 p-5">
        <h2 className="text-xl font-semibold">Definitions</h2>
        <dl className="space-y-3 text-sm">
          <div><dt className="font-semibold text-cyan-200">Pressure</dt><dd className="text-slate-300">A delivery flagged by the configured leverage, chase, death-over, collapse, or tight-finish rule. The composite definition combines the available pressure flags.</dd></div>
          <div><dt className="font-semibold text-cyan-200">WPA per ball</dt><dd className="text-slate-300">Change in the batting side’s win probability per legal ball, averaged over the player’s qualifying deliveries.</dd></div>
          <div><dt className="font-semibold text-cyan-200">Runs saved</dt><dd className="text-slate-300">Runs above or below the situation-adjusted expectation, expressed per ball in the player index.</dd></div>
          <div><dt className="font-semibold text-cyan-200">Consistency</dt><dd className="text-slate-300">The configured consistency component of the index; it contributes only its configured weight.</dd></div>
          <div><dt className="font-semibold text-cyan-200">80% interval</dt><dd className="text-slate-300">An uncertainty range containing the central 80% of the reported bootstrap estimate when that endpoint supplies one. A missing interval is not inferred.</dd></div>
          <div><dt className="font-semibold text-cyan-200">Calibration</dt><dd className="text-slate-300">Mean observed win frequency compared with mean predicted win probability within each probability bin.</dd></div>
        </dl>
      </section>

      <section className="space-y-3">
        <h2 className="text-xl font-semibold">Current thresholds</h2>
        <div className="overflow-x-auto rounded-xl border border-slate-700">
          <table className="min-w-full divide-y divide-slate-700 text-left text-sm">
            <thead className="bg-slate-900 text-slate-300"><tr><th className="px-4 py-3">Setting</th><th className="px-4 py-3">Value</th></tr></thead>
            <tbody className="divide-y divide-slate-800">
              {thresholds.map(([label, path]) => {
                const value = nested(config.config, path)
                const shownValue = path[path.length - 1] === 'tight_finish_wp_min'
                  ? `${display(value)}–${display(nested(config.config, ['pressure', 'tight_finish_wp_max']))}`
                  : display(value)
                return <tr key={label}><td className="px-4 py-3">{label}</td><td className="px-4 py-3">{shownValue}</td></tr>
              })}
            </tbody>
          </table>
        </div>
      </section>

      <section className="space-y-3 rounded-xl border border-slate-700 bg-slate-900 p-5">
        <h2 className="text-xl font-semibold">Assumptions</h2>
        <ul className="list-disc space-y-2 pl-5 text-sm text-slate-300">
          <li>Ball contributions are grouped by match for validation and match-level resampling.</li>
          <li>Only the configured eligible player and competition groups contribute to selection rankings.</li>
          <li>All-rounders retain their role and fill one assigned slot in the configured selection split.</li>
          <li>Data not present in the supplied aggregates or squad CSVs is shown as unavailable rather than guessed.</li>
        </ul>
      </section>

      <section className="space-y-3">
        <h2 className="text-xl font-semibold">Exclusions</h2>
        {Object.keys(quality.exclusions).length === 0 ? <EmptyState label="No exclusion breakdown was returned." /> : (
          <div className="overflow-x-auto rounded-xl border border-slate-700">
            <table className="min-w-full divide-y divide-slate-700 text-left text-sm">
              <thead className="bg-slate-900 text-slate-300"><tr><th className="px-4 py-3">Reason</th><th className="px-4 py-3">Rows dropped</th><th className="px-4 py-3">Balls dropped</th><th className="px-4 py-3">Matches dropped</th></tr></thead>
              <tbody className="divide-y divide-slate-800">
                {Object.entries(quality.exclusions).map(([reason, counts]) => (
                  <tr key={reason}><td className="px-4 py-3">{reason.replaceAll('_', ' ')}</td><td className="px-4 py-3">{counts.rows_dropped.toLocaleString()}</td><td className="px-4 py-3">{counts.balls_dropped.toLocaleString()}</td><td className="px-4 py-3">{counts.matches_dropped.toLocaleString()}</td></tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      <section className="space-y-3 rounded-xl border border-amber-800 bg-amber-950/40 p-5">
        <h2 className="text-xl font-semibold text-amber-100">Limitations</h2>
        <p className="text-sm text-amber-50">Backtest list precision and uplift are retrospective on each test fold. The backtest report identifies bowling results as weak; treat role-specific bowling rankings cautiously.</p>
        <p className="text-sm text-amber-50">Uncertainty intervals are endpoint-specific. Where an endpoint does not return an 80% interval, the UI labels it unavailable and does not manufacture one.</p>
        <p className="text-xs text-amber-200">Selected predictor: {display(typeof backtest.selected_predictor === 'object' ? backtest.selected_predictor.name : backtest.selected_predictor)} · Data version: {config.data_version}</p>
        <IntervalNote>Configuration values are exact settings, not estimates.</IntervalNote>
      </section>
    </div>
  )
}
