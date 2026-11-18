import { useQuery } from '@tanstack/react-query'
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { fetchJson } from '../api/client'
import { ChartPanel, EmptyState, ErrorState, IntervalNote, LoadingState } from '../components/PageStates'
import type { SensitivityReport, Top10Report } from '../types'

const robustThreshold = 0.75
const fragileThreshold = 0.5

export function SensitivityPage() {
  const query = useQuery({
    queryKey: ['sensitivity-report'],
    queryFn: async () => {
      const [sensitivity, top10] = await Promise.all([
        fetchJson<SensitivityReport>('/api/sensitivity'),
        fetchJson<Top10Report>('/api/top10'),
      ])
      return { sensitivity, top10 }
    },
  })

  if (query.isLoading) return <LoadingState label="Loading sensitivity analysis…" />
  if (query.error) return <ErrorState message={query.error.message} onRetry={() => void query.refetch()} />
  if (!query.data || query.data.sensitivity.players.length === 0) return <EmptyState label="No sensitivity results were returned." />

  const { sensitivity, top10 } = query.data
  const names = new Map([...top10.picks, ...top10.reserves].map((player) => [player.player_id, player.display_name]))
  const selected = top10.picks.map((pick) => {
    const result = sensitivity.players.find((player) => player.player_id === pick.player_id)
    if (!result) return { ...pick, sensitivity: null, classification: 'No sensitivity result' }
    const intervalWidth = result.rank_interval_80 ? result.rank_interval_80[1] - result.rank_interval_80[0] : null
    const robust = result.inclusion_probability >= robustThreshold
      && result.rank_stability >= robustThreshold
      && intervalWidth != null
      && intervalWidth <= 5
    const fragile = result.inclusion_probability < fragileThreshold
      || result.rank_stability < fragileThreshold
      || (intervalWidth != null && intervalWidth > 10)
    return {
      ...pick,
      sensitivity: result,
      classification: robust ? 'Robust' : fragile ? 'Fragile' : 'Mixed',
    }
  })
  const chartData = [...sensitivity.players]
    .sort((a, b) => b.inclusion_probability - a.inclusion_probability)
    .slice(0, 20)
    .map((player) => ({
      ...player,
      display_name: names.get(player.player_id) ?? `Player ID ${player.player_id}`,
    }))

  return (
    <div className="space-y-6">
      <header className="space-y-2">
        <h1 className="text-3xl font-bold">Sensitivity</h1>
        <p className="text-sm text-slate-400">
          Bootstrap unit: {sensitivity.bootstrap.unit ?? 'not provided'} · {sensitivity.bootstrap.resamples ?? '—'} resamples · seed {sensitivity.bootstrap.seed ?? '—'}
        </p>
        <IntervalNote>Rank bars include their API-provided 80% rank interval. Inclusion and stability intervals are not supplied.</IntervalNote>
      </header>

      <ChartPanel title="Top inclusion probabilities" note="Horizontal axis: probability of appearing in the Top 10 (%). Tooltip includes rank stability and 80% rank interval.">
        <div className="h-[560px]">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} layout="vertical" margin={{ left: 18, right: 24 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis type="number" domain={[0, 1]} tickFormatter={(value) => `${Math.round(Number(value) * 100)}%`} label={{ value: 'Top 10 inclusion probability (%)', position: 'insideBottom', offset: -2, fill: '#cbd5e1' }} tick={{ fill: '#94a3b8' }} />
              <YAxis type="category" dataKey="display_name" width={170} label={{ value: 'Player', angle: -90, position: 'insideLeft', fill: '#cbd5e1' }} tick={{ fill: '#cbd5e1', fontSize: 11 }} />
              <Tooltip formatter={(value, name, item) => {
                const row = item.payload as typeof chartData[number]
                const rankInterval = row.rank_interval_80 ? `${row.rank_interval_80[0]}–${row.rank_interval_80[1]}` : 'not supplied'
                return [`${(Number(value) * 100).toFixed(1)}%; stability ${(row.rank_stability * 100).toFixed(1)}%; 80% rank interval ${rankInterval}`, String(name)]
              }} />
              <Bar dataKey="inclusion_probability" name="Inclusion probability" fill="#22d3ee" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </ChartPanel>

      <section className="space-y-3">
        <h2 className="text-xl font-semibold">Model Top 10 stability</h2>
        <p className="text-sm text-slate-400">
          Robust: inclusion probability and rank stability at least {robustThreshold * 100}% with a rank interval width at most 5. Fragile: either probability below {fragileThreshold * 100}% or rank interval width over 10. Other picks are mixed.
        </p>
        {selected.length === 0 ? <EmptyState label="No model picks are available to classify." /> : (
          <div className="overflow-x-auto rounded-xl border border-slate-700">
            <table className="min-w-full divide-y divide-slate-700 text-left text-sm">
              <thead className="bg-slate-900 text-slate-300"><tr><th className="px-4 py-3">Player</th><th className="px-4 py-3">Inclusion probability</th><th className="px-4 py-3">Rank stability</th><th className="px-4 py-3">80% rank interval</th><th className="px-4 py-3">Assessment</th></tr></thead>
              <tbody className="divide-y divide-slate-800">
                {selected.map((pick) => (
                  <tr key={pick.player_id}>
                    <td className="px-4 py-3">{pick.display_name}</td>
                    <td className="px-4 py-3">{pick.sensitivity ? `${(pick.sensitivity.inclusion_probability * 100).toFixed(1)}%` : 'Not returned'}<span className="block text-xs text-slate-500">80% interval not supplied</span></td>
                    <td className="px-4 py-3">{pick.sensitivity ? `${(pick.sensitivity.rank_stability * 100).toFixed(1)}%` : 'Not returned'}<span className="block text-xs text-slate-500">80% interval not supplied</span></td>
                    <td className="px-4 py-3">{pick.sensitivity?.rank_interval_80?.join('–') ?? 'Not supplied'}</td>
                    <td className="px-4 py-3">{pick.classification}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  )
}
