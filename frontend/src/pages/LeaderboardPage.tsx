import { useState } from 'react'
import { Link } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { fetchJson } from '../api/client'
import { EmptyState, ErrorState, IntervalNote, LoadingState } from '../components/PageStates'
import type { LeaderboardRequest, LeaderboardRow } from '../types'
import { useShortlist } from '../store/useShortlist'

const initialRequest: LeaderboardRequest = {
  role: 'combined',
  min_pressure_balls: 60,
  pressure_definition: 'composite',
}

function formatNumber(value: number | null | undefined, digits = 4) {
  return value == null ? '—' : value.toFixed(digits)
}

export function LeaderboardPage() {
  const [role, setRole] = useState<LeaderboardRequest['role'] | 'all'>('combined')
  const [compGroup, setCompGroup] = useState<LeaderboardRequest['comp_group'] | 'all'>('all')
  const [minPressureBalls, setMinPressureBalls] = useState(60)
  const [pressureDefinition, setPressureDefinition] = useState<LeaderboardRequest['pressure_definition']>('composite')
  const [request, setRequest] = useState(initialRequest)
  const [actionError, setActionError] = useState<string | null>(null)
  const rows = useQuery({
    queryKey: ['leaderboard', request],
    queryFn: () => fetchJson<LeaderboardRow[]>('/api/leaderboard', {
      method: 'POST',
      body: JSON.stringify(request),
    }),
  })
  const shortlist = useShortlist()

  function submit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault()
    const next: LeaderboardRequest = {
      min_pressure_balls: Math.max(0, Math.trunc(minPressureBalls)),
      pressure_definition: pressureDefinition,
    }
    if (role !== 'all') next.role = role
    if (compGroup !== 'all') next.comp_group = compGroup
    setRequest(next)
  }

  async function addPlayer(player: LeaderboardRow) {
    setActionError(null)
    try {
      await shortlist.add(player.player_id, player.player_name)
    } catch (error) {
      setActionError(error instanceof Error ? error.message : 'Unable to update shortlist.')
    }
  }

  return (
    <div className="space-y-6">
      <header className="space-y-2">
        <h1 className="text-3xl font-bold">Leaderboard</h1>
        <p className="text-sm text-slate-400">Ranked results from the processed aggregates; score intervals are not supplied by this endpoint.</p>
      </header>
      <form onSubmit={submit} className="grid gap-3 rounded-xl border border-slate-700 bg-slate-900 p-4 sm:grid-cols-2 lg:grid-cols-4">
        <label className="space-y-1 text-sm"><span className="block text-slate-300">Role</span><select value={role} onChange={(event) => setRole(event.target.value as LeaderboardRequest['role'] | 'all')} className="w-full rounded border border-slate-600 bg-slate-950 px-3 py-2"><option value="combined">Combined</option><option value="all">All roles</option><option value="batter">Batter</option><option value="bowler">Bowler</option><option value="all_rounder">All-rounder</option><option value="developing">Developing</option></select></label>
        <label className="space-y-1 text-sm"><span className="block text-slate-300">Competition group</span><select value={compGroup} onChange={(event) => setCompGroup(event.target.value as LeaderboardRequest['comp_group'] | 'all')} className="w-full rounded border border-slate-600 bg-slate-950 px-3 py-2"><option value="all">All groups</option><option value="IPL">IPL</option><option value="International">International</option><option value="Emerging">Emerging</option></select></label>
        <label className="space-y-1 text-sm"><span className="block text-slate-300">Minimum pressure balls</span><input type="number" min="0" value={minPressureBalls} onChange={(event) => setMinPressureBalls(Number(event.target.value))} className="w-full rounded border border-slate-600 bg-slate-950 px-3 py-2" /></label>
        <label className="space-y-1 text-sm"><span className="block text-slate-300">Pressure definition</span><select value={pressureDefinition} onChange={(event) => setPressureDefinition(event.target.value as LeaderboardRequest['pressure_definition'])} className="w-full rounded border border-slate-600 bg-slate-950 px-3 py-2"><option value="composite">Composite</option><option value="high_leverage">High leverage</option><option value="hard_chase">Hard chase</option><option value="death_overs">Death overs</option><option value="collapse">Collapse</option><option value="tight_finish">Tight finish</option></select></label>
        <button type="submit" className="rounded-lg bg-cyan-500 px-4 py-2 font-semibold text-slate-950 hover:bg-cyan-400 sm:col-span-2 lg:col-span-4">Apply filters</button>
      </form>

      {actionError && <ErrorState message={actionError} />}
      {shortlist.error && <ErrorState message={shortlist.error instanceof Error ? shortlist.error.message : 'Shortlist could not be loaded; adding players is disabled.'} onRetry={() => void shortlist.retry()} />}
      {rows.isLoading && <LoadingState label="Loading leaderboard…" />}
      {rows.error && <ErrorState message={rows.error.message} onRetry={() => void rows.refetch()} />}
      {rows.data && rows.data.length === 0 && <EmptyState label="No players match these filters." />}
      {rows.data && rows.data.length > 0 && (
        <>
          <p className="text-sm text-slate-400">Showing the first {Math.min(rows.data.length, 100)} of {rows.data.length} players.</p>
          <IntervalNote>WPA / ball has an 80% interval; the composite score and stability intervals are not returned.</IntervalNote>
          <div className="overflow-x-auto rounded-xl border border-slate-700">
            <table className="min-w-full divide-y divide-slate-700 text-left text-sm">
              <thead className="bg-slate-900 text-slate-300"><tr><th className="px-4 py-3">Rank</th><th className="px-4 py-3">Player</th><th className="px-4 py-3">Role</th><th className="px-4 py-3">Group</th><th className="px-4 py-3">Index score</th><th className="px-4 py-3">WPA / ball (80% interval)</th><th className="px-4 py-3">Pressure balls</th><th className="px-4 py-3">Rank stability</th><th className="px-4 py-3">Evidence status</th><th className="px-4 py-3">Action</th></tr></thead>
              <tbody className="divide-y divide-slate-800">
                {rows.data.slice(0, 100).map((player) => {
                  const added = shortlist.entries.some((entry) => entry.player_id === player.player_id)
                  return (
                    <tr key={player.player_id} className="hover:bg-slate-900">
                      <td className="px-4 py-3">{player.rank}</td>
                      <td className="px-4 py-3"><Link to={`/players/${player.player_id}`} className="text-cyan-200 hover:underline">{player.player_name}</Link></td>
                      <td className="px-4 py-3">{player.role}</td>
                      <td className="px-4 py-3">{player.comp_group ?? '—'}</td>
                      <td className="px-4 py-3">{formatNumber(player.score)}<span className="block text-xs text-slate-500">80% interval not supplied</span></td>
                      <td className="px-4 py-3">{formatNumber(player.wpa_per_ball)}<span className="block text-xs text-slate-400">[{player.wpa_80_interval.map((value) => formatNumber(value)).join(', ')}]</span></td>
                      <td className="px-4 py-3">{player.pressure_balls.toLocaleString()} balls</td>
                      <td className="px-4 py-3">{(player.rank_stability * 100).toFixed(1)}%<span className="block text-xs text-slate-500">interval not supplied</span></td>
                      <td className="px-4 py-3">{player.small_sample ? `Limited (<${player.small_sample_threshold} balls)` : 'Adequate'}</td>
                      <td className="px-4 py-3"><button type="button" disabled={shortlist.loading || Boolean(shortlist.error) || shortlist.saving || added || shortlist.entries.length >= 10} onClick={() => void addPlayer(player)} className="text-cyan-200 disabled:opacity-50">{added ? 'Added' : 'Add to shortlist'}</button></td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  )
}
