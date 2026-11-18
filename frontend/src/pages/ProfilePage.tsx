import { useQuery } from '@tanstack/react-query'
import { Link, useParams } from 'react-router-dom'
import { fetchJson } from '../api/client'
import { EmptyState, ErrorState, IntervalNote, LoadingState } from '../components/PageStates'
import type { PlayerProfile, SimilarPlayer } from '../types'

function format(value: number | null | undefined, digits = 3) {
  return value == null ? '—' : value.toFixed(digits)
}

function Metric({ label, value, unit }: { label: string; value: string; unit?: string }) {
  return (
    <div className="rounded-xl border border-slate-700 bg-slate-900 p-4">
      <p className="text-sm text-slate-400">{label}{unit ? ` (${unit})` : ''}</p>
      <p className="mt-2 text-xl font-semibold">{value}</p>
      <p className="mt-1 text-xs text-slate-500">80% interval not supplied by the profile API</p>
    </div>
  )
}

export function ProfilePage() {
  const { playerId } = useParams()
  const query = useQuery({
    queryKey: ['player-profile', playerId],
    enabled: Boolean(playerId),
    queryFn: async () => {
      const [profile, similar] = await Promise.all([
        fetchJson<PlayerProfile>(`/api/players/${encodeURIComponent(playerId ?? '')}`),
        fetchJson<SimilarPlayer[]>(`/api/players/${encodeURIComponent(playerId ?? '')}/similar`),
      ])
      return { profile, similar }
    },
  })

  if (!playerId) return <ErrorState message="Player ID is missing from the URL." />
  if (query.isLoading) return <LoadingState label="Loading player profile…" />
  if (query.error) return <ErrorState message={query.error.message} onRetry={() => void query.refetch()} />
  if (!query.data) return <EmptyState label="No player profile was returned." />

  const { profile: player, similar } = query.data
  const metrics = player.pressure_vs_normal
  const phaseShares = player.phase_splits.batting_share
  const hasAggregates = player.pressure_balls > 0
    || player.stroke_aggregates.length > 0
    || player.season_trend.length > 0

  return (
    <div className="space-y-6">
      <header className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-3xl font-bold">{player.player_name}</h1>
          <p className="mt-1 text-slate-400">{player.role} · {player.pressure_balls.toLocaleString()} pressure balls</p>
        </div>
        <Link to="/leaderboard" className="rounded-lg border border-slate-700 px-4 py-2 text-sm text-cyan-200 hover:bg-slate-900">Back to leaderboard</Link>
      </header>
      {!hasAggregates && <EmptyState label="No performance aggregates are available for this player." />}
      <IntervalNote>Pressure, phase, stroke, and season aggregates do not include 80% intervals in this API response.</IntervalNote>

      <section className="space-y-3">
        <h2 className="text-xl font-semibold">Pressure vs normal</h2>
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          <Metric label="Batting pressure WPA / ball" value={format(metrics.batting_pressure_wpa_per_ball)} unit="WPA/ball" />
          <Metric label="Batting normal WPA / ball" value={format(metrics.batting_normal_wpa_per_ball)} unit="WPA/ball" />
          <Metric label="Batting pressure WPA delta" value={format(metrics.batting_pressure_vs_normal_delta)} unit="WPA/ball" />
          <Metric label="Batting pressure strike rate" value={format(metrics.batting_pressure_strike_rate, 1)} unit="runs/100 balls" />
          <Metric label="Bowling pressure economy" value={format(metrics.bowling_pressure_economy, 2)} unit="runs/over" />
          <Metric label="Bowling pressure economy delta" value={format(metrics.bowling_pressure_vs_normal_economy_delta, 2)} unit="runs/over" />
        </div>
      </section>

      <section className="grid gap-4 lg:grid-cols-2">
        <div className="space-y-3">
          <h2 className="text-xl font-semibold">Batting phase share</h2>
          {Object.keys(phaseShares).length === 0 ? <EmptyState label="No batting phase shares were returned." /> : (
            <div className="grid grid-cols-3 gap-3">
              {Object.entries(phaseShares).map(([phase, share]) => (
                <Metric key={phase} label={phase} value={share == null ? '—' : `${(share * 100).toFixed(1)}%`} unit="share" />
              ))}
            </div>
          )}
        </div>
        <div className="space-y-3">
          <h2 className="text-xl font-semibold">Bowling phase splits</h2>
          {Object.keys(player.phase_splits.bowling).length === 0 ? <EmptyState label="No bowling phase aggregates were returned." /> : (
            <div className="overflow-x-auto rounded-xl border border-slate-700">
              <table className="min-w-full divide-y divide-slate-700 text-left text-sm">
                <thead className="bg-slate-900 text-slate-300"><tr><th className="px-3 py-2">Phase</th><th className="px-3 py-2">Legal balls</th><th className="px-3 py-2">Economy (runs/over)</th><th className="px-3 py-2">Wickets / 24 balls</th><th className="px-3 py-2">80% intervals</th></tr></thead>
                <tbody className="divide-y divide-slate-800">
                  {Object.entries(player.phase_splits.bowling).map(([phase, values]) => (
                    <tr key={phase}>
                      <td className="px-3 py-2">{phase}</td>
                      <td className="px-3 py-2">{format(values.legal_balls, 0)}</td>
                      <td className="px-3 py-2">{format(values.economy, 2)}</td>
                      <td className="px-3 py-2">{format(values.wickets_per_24_balls, 2)}</td>
                      <td className="px-3 py-2 text-slate-500">Not supplied</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </section>

      <section className="space-y-3">
        <h2 className="text-xl font-semibold">Stroke aggregates</h2>
        {player.stroke_aggregates.length === 0 ? <EmptyState label="No stroke aggregates were returned." /> : (
          <AggregateTable
            headers={['Stroke', 'Legal balls', 'Runs', 'Runs above situation', 'WPA / ball', '80% interval']}
            rows={player.stroke_aggregates}
            columns={['stroke', 'legal_balls', 'runs', 'runs_above_situation', 'mean_wpa_per_ball']}
          />
        )}
      </section>

      <section className="space-y-3">
        <h2 className="text-xl font-semibold">Season trend</h2>
        {player.season_trend.length === 0 ? <EmptyState label="No season trend rows were returned." /> : (
          <AggregateTable
            headers={['Season', 'Competition', 'Role', 'Legal balls', 'Pressure balls', 'Runs', 'WPA / ball', '80% interval']}
            rows={player.season_trend}
            columns={['season', 'comp_group', 'role', 'legal_balls', 'pressure_balls', 'total_runs', 'mean_wpa_per_ball']}
          />
        )}
      </section>

      <section className="space-y-3">
        <h2 className="text-xl font-semibold">Similar players</h2>
        {similar.length === 0 ? <EmptyState label="No similar players are available." /> : (
          <>
            <p className="text-xs text-slate-400">Distance is a descriptive similarity measure; 80% interval is not supplied.</p>
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
              {similar.map((item) => (
                <Link key={item.similar_player_id} to={`/players/${item.similar_player_id}`} className="rounded-xl border border-slate-700 bg-slate-900 p-4 hover:border-cyan-700">
                  <span className="block font-medium text-cyan-200">{item.display_name}</span>
                  <span className="mt-1 block text-sm text-slate-400">{item.role} · distance {item.distance.toFixed(2)} (80% interval not supplied)</span>
                </Link>
              ))}
            </div>
          </>
        )}
      </section>
    </div>
  )
}

function AggregateTable({
  headers,
  rows,
  columns,
}: {
  headers: string[]
  rows: Array<Record<string, number | string | null>>
  columns: string[]
}) {
  return (
    <div className="overflow-x-auto rounded-xl border border-slate-700">
      <table className="min-w-full divide-y divide-slate-700 text-left text-sm">
        <thead className="bg-slate-900 text-slate-300"><tr>{headers.map((header) => <th className="px-3 py-2" key={header}>{header}</th>)}</tr></thead>
        <tbody className="divide-y divide-slate-800">
          {rows.map((row, index) => (
            <tr key={index}>
              {columns.map((column) => {
                const value = row[column]
                const formatted = typeof value === 'number' ? value.toFixed(3) : value ?? '—'
                return <td className="px-3 py-2" key={column}>{formatted}</td>
              })}
              <td className="px-3 py-2 text-slate-500">Not supplied</td>
            </tr>
          ))}
          {rows.length === 0 && <tr><td className="px-3 py-4 text-slate-400" colSpan={headers.length}>No aggregate rows available.</td></tr>}
        </tbody>
      </table>
    </div>
  )
}
