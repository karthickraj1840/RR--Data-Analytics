import { useState } from 'react'
import { Link } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { fetchJson } from '../api/client'
import { EmptyState, ErrorState, IntervalNote, LoadingState } from '../components/PageStates'
import type { Top10Entry, Top10Report } from '../types'
import { useShortlist } from '../store/useShortlist'

function PlayerCard({
  player,
  added,
  disabled,
  onAdd,
}: {
  player: Top10Entry
  added: boolean
  disabled: boolean
  onAdd: () => void
}) {
  return (
    <article className="rounded-xl border border-slate-700 bg-slate-900 p-4">
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="text-xs uppercase tracking-wide text-slate-400">
            Rank {player.rank} · {player.selection_slot} slot
          </p>
          <Link to={`/players/${player.player_id}`} className="mt-1 block font-semibold text-cyan-200 hover:underline">
            {player.display_name}
          </Link>
        </div>
        <span className="rounded-full border border-cyan-700 px-2 py-1 text-xs text-cyan-200">
          Grade {player.grade}
        </span>
      </div>
      <dl className="mt-4 grid grid-cols-2 gap-3 text-sm">
        <div><dt className="text-slate-400">True role</dt><dd>{player.role_label}</dd></div>
        <div><dt className="text-slate-400">Pressure balls</dt><dd>{player.pressure_balls.toLocaleString()} balls</dd></div>
        <div><dt className="text-slate-400">IPL share</dt><dd>{(player.ipl_share * 100).toFixed(1)}%</dd><dd className="text-xs text-slate-400">80% interval not supplied</dd></div>
        <div><dt className="text-slate-400">Selected WPA / ball</dt><dd>{player.selected_predictor_wpa_per_ball.toFixed(4)}</dd><dd className="text-xs text-slate-400">80% interval: {player.wpa_80_interval?.map((value) => value.toFixed(4)).join(' to ') ?? 'Not provided'}</dd></div>
        <div><dt className="text-slate-400">Bootstrap inclusion</dt><dd>{(player.bootstrap_inclusion_probability * 100).toFixed(1)}% · interval not provided</dd></div>
        <div><dt className="text-slate-400">Rank stability</dt><dd>{(player.rank_stability * 100).toFixed(1)}%</dd><dd className="text-xs text-slate-400">80% rank interval: {player.bootstrap_rank_interval_80?.join(' to ') ?? 'Not provided'}</dd></div>
      </dl>
      <button type="button" disabled={disabled || added} onClick={onAdd} className="mt-4 rounded-lg border border-cyan-700 px-3 py-2 text-sm text-cyan-200 disabled:cursor-not-allowed disabled:opacity-50">
        {added ? 'Added to shortlist' : 'Add to shortlist'}
      </button>
    </article>
  )
}

export function Top10Page() {
  const [actionError, setActionError] = useState<string | null>(null)
  const report = useQuery({
    queryKey: ['top10'],
    queryFn: () => fetchJson<Top10Report>('/api/top10'),
  })
  const shortlist = useShortlist()

  async function addPlayer(player: Top10Entry) {
    setActionError(null)
    try {
      await shortlist.add(player.player_id, player.display_name)
    } catch (error) {
      setActionError(error instanceof Error ? error.message : 'Unable to update shortlist.')
    }
  }

  if (report.isLoading) return <LoadingState label="Loading model selection…" />
  if (report.error) return <ErrorState message={report.error.message} onRetry={() => void report.refetch()} />
  if (!report.data || report.data.picks.length === 0) return <EmptyState label="The model did not return Top 10 picks." />

  return (
    <div className="space-y-6">
      <header className="space-y-2">
        <h1 className="text-3xl font-bold">Top 10 selection</h1>
        <p className="text-sm text-slate-400">
          {report.data.role_constraint.configured_split.batters} batter slots · {report.data.role_constraint.configured_split.bowlers} bowler slots
        </p>
        <p className="text-sm text-slate-300">{report.data.role_constraint.note}</p>
        <IntervalNote>WPA and rank intervals are shown when available; IPL-share and inclusion-probability intervals are not supplied.</IntervalNote>
      </header>
      {actionError && <ErrorState message={actionError} />}
      {shortlist.error && <ErrorState message={shortlist.error instanceof Error ? shortlist.error.message : 'Shortlist could not be loaded; adding players is disabled.'} onRetry={() => void shortlist.retry()} />}
      <section aria-label="Top 10 players" className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {report.data.picks.map((player) => (
          <PlayerCard
            key={player.player_id}
            player={player}
            added={shortlist.entries.some((entry) => entry.player_id === player.player_id)}
            disabled={shortlist.loading || Boolean(shortlist.error) || shortlist.saving || shortlist.entries.length >= 10}
            onAdd={() => void addPlayer(player)}
          />
        ))}
      </section>
      <section className="space-y-3">
        <h2 className="text-xl font-semibold">Reserves (ranks 11–20)</h2>
        {report.data.reserves.length === 0 ? <EmptyState label="No reserve players were returned." /> : (
          <div className="overflow-x-auto rounded-xl border border-slate-700">
            <table className="min-w-full divide-y divide-slate-700 text-left text-sm">
              <thead className="bg-slate-900 text-slate-300">
                <tr><th className="px-4 py-3">Rank</th><th className="px-4 py-3">Player</th><th className="px-4 py-3">True role</th><th className="px-4 py-3">Slot</th><th className="px-4 py-3">Grade</th><th className="px-4 py-3">Pressure balls</th><th className="px-4 py-3">Action</th></tr>
              </thead>
              <tbody className="divide-y divide-slate-800">
                {report.data.reserves.map((player) => {
                  const added = shortlist.entries.some((entry) => entry.player_id === player.player_id)
                  return (
                    <tr key={player.player_id}>
                      <td className="px-4 py-3">{player.rank}</td>
                      <td className="px-4 py-3"><Link className="text-cyan-200 hover:underline" to={`/players/${player.player_id}`}>{player.display_name}</Link></td>
                      <td className="px-4 py-3">{player.role_label}</td>
                      <td className="px-4 py-3">{player.selection_slot}</td>
                      <td className="px-4 py-3">{player.grade}</td>
                      <td className="px-4 py-3">{player.pressure_balls.toLocaleString()}</td>
                      <td className="px-4 py-3"><button type="button" disabled={shortlist.loading || Boolean(shortlist.error) || shortlist.saving || added || shortlist.entries.length >= 10} onClick={() => void addPlayer(player)} className="text-cyan-200 disabled:opacity-50">{added ? 'Added' : 'Add to shortlist'}</button></td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  )
}
