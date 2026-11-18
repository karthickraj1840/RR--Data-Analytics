import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { fetchJson } from '../api/client'
import { EmptyState, ErrorState, LoadingState } from '../components/PageStates'
import type { PlayerProfile } from '../types'
import { useAppStore, type ShortlistEntry } from '../store/useAppStore'
import { useShortlist } from '../store/useShortlist'

function EvidencePanel({ playerId }: { playerId: number }) {
  const query = useQuery({
    queryKey: ['player-profile', playerId],
    queryFn: () => fetchJson<PlayerProfile>(`/api/players/${playerId}`),
  })

  if (query.isLoading) return <LoadingState label="Loading aggregate evidence…" />
  if (query.error) return <ErrorState message={query.error.message} onRetry={() => void query.refetch()} />
  if (!query.data) return <EmptyState label="No aggregate evidence was returned." />

  const metrics = query.data.pressure_vs_normal
  const values: Array<[string, number | null, string]> = [
    ['Pressure balls', query.data.pressure_balls, 'balls'],
    ['Batting pressure WPA / ball', metrics.batting_pressure_wpa_per_ball, 'WPA/ball'],
    ['Batting pressure strike rate', metrics.batting_pressure_strike_rate, 'runs/100 balls'],
    ['Bowling pressure economy', metrics.bowling_pressure_economy, 'runs/over'],
  ]
  return (
    <dl className="grid gap-2 sm:grid-cols-2">
      {values.map(([label, value, unit]) => (
        <div key={label} className="rounded border border-slate-800 bg-slate-950 p-2">
          <dt className="text-xs text-slate-400">{label} ({unit})</dt>
          <dd className="mt-1 text-sm">{value == null ? 'Not provided' : typeof value === 'number' ? unit === 'balls' ? value.toLocaleString() : value.toFixed(3) : value}</dd>
          <dd className="text-xs text-slate-500">80% interval not supplied</dd>
        </div>
      ))}
    </dl>
  )
}

export function ShortlistPage() {
  const [actionError, setActionError] = useState<string | null>(null)
  const [draggedId, setDraggedId] = useState<number | null>(null)
  const [rationaleDirty, setRationaleDirty] = useState(false)
  const [startingFromTop10, setStartingFromTop10] = useState(false)
  const shortlist = useShortlist()
  const replaceShortlist = useAppStore((state) => state.replaceShortlist)
  const localEntries = useAppStore((state) => state.shortlist)

  async function save(next: ShortlistEntry[]): Promise<boolean> {
    setActionError(null)
    try {
      await shortlist.save(next)
      setRationaleDirty(false)
      return true
    } catch (error) {
      setActionError(error instanceof Error ? error.message : 'Unable to save the shortlist.')
      return false
    }
  }

  async function startFromTop10() {
    setActionError(null)
    setStartingFromTop10(true)
    try {
      await shortlist.addTop10()
    } catch (error) {
      setActionError(error instanceof Error ? error.message : 'Unable to load model picks.')
    } finally {
      setStartingFromTop10(false)
    }
  }

  async function exportShortlist() {
    setActionError(null)
    try {
      if (rationaleDirty && !await save(useAppStore.getState().shortlist)) return
      const response = await fetch('/api/export/shortlist', { method: 'POST' })
      if (!response.ok) {
        const detail = await response.text()
        throw new Error(detail || `Export failed (${response.status}).`)
      }
      const blob = await response.blob()
      const url = URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = 'shortlist-export.zip'
      link.click()
      window.setTimeout(() => URL.revokeObjectURL(url), 1000)
    } catch (error) {
      setActionError(error instanceof Error ? error.message : 'Unable to export the shortlist.')
    }
  }

  function reorder(fromId: number, toId: number) {
    const next = [...localEntries]
    const from = next.findIndex((entry) => entry.player_id === fromId)
    const to = next.findIndex((entry) => entry.player_id === toId)
    if (from < 0 || to < 0 || from === to) return
    const [moved] = next.splice(from, 1)
    next.splice(to, 0, moved)
    void save(next)
  }

  function moveBy(index: number, direction: -1 | 1) {
    const target = index + direction
    if (target < 0 || target >= localEntries.length) return
    reorder(localEntries[index].player_id, localEntries[target].player_id)
  }

  if (shortlist.loading) return <LoadingState label="Loading saved shortlist…" />
  if (shortlist.error) return <ErrorState message={shortlist.error instanceof Error ? shortlist.error.message : 'Saved shortlist could not be loaded.'} onRetry={() => void shortlist.retry()} />

  return (
    <div className="space-y-6">
      <header className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-3xl font-bold">Shortlist</h1>
          <p className="mt-2 text-sm text-slate-400">{localEntries.length} of 10 players. Rationale is user-written and starts blank for newly added players.</p>
        </div>
        <div className="flex flex-wrap gap-2">
          <button type="button" disabled={shortlist.saving || startingFromTop10 || localEntries.length >= 10} onClick={() => void startFromTop10()} className="rounded-lg border border-cyan-700 px-3 py-2 text-sm text-cyan-200 disabled:opacity-50">Start from Model Top 10</button>
          <button type="button" disabled={shortlist.saving} onClick={() => void exportShortlist()} className="rounded-lg bg-cyan-500 px-3 py-2 text-sm font-semibold text-slate-950 disabled:opacity-50">Export</button>
        </div>
      </header>
      {actionError && <ErrorState message={actionError} />}
      {shortlist.saving && <LoadingState label="Saving shortlist…" />}
      {startingFromTop10 && <LoadingState label="Adding model picks…" />}
      {localEntries.length === 0 ? (
        <EmptyState label="Your shortlist is empty. Start from the model picks or add players from Top 10 and Leaderboard." />
      ) : (
        <div className="space-y-4">
          {localEntries.map((entry, index) => (
            <article
              key={entry.player_id}
              draggable
              onDragStart={() => setDraggedId(entry.player_id)}
              onDragOver={(event) => event.preventDefault()}
              onDrop={() => {
                if (draggedId != null) reorder(draggedId, entry.player_id)
                setDraggedId(null)
              }}
              onDragEnd={() => setDraggedId(null)}
              className="grid gap-4 rounded-xl border border-slate-700 bg-slate-900 p-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.3fr)]"
            >
              <div className="space-y-3">
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <p className="text-xs uppercase tracking-wide text-slate-400">Position {index + 1} · Player ID {entry.player_id}</p>
                    <PlayerName playerId={entry.player_id} fallback={entry.display_name} />
                  </div>
                  <button type="button" onClick={() => void save(localEntries.filter((candidate) => candidate.player_id !== entry.player_id))} className="text-sm text-rose-300">Remove</button>
                </div>
                <label className="block space-y-1 text-sm">
                  <span className="text-slate-300">Your rationale</span>
                  <textarea
                    value={entry.rationale}
                    onChange={(event) => {
                      setRationaleDirty(true)
                      replaceShortlist(localEntries.map((candidate) => candidate.player_id === entry.player_id ? { ...candidate, rationale: event.target.value } : candidate))
                    }}
                    rows={3}
                    className="w-full rounded border border-slate-700 bg-slate-950 p-2"
                    aria-label={`Your rationale for player ${entry.player_id}`}
                  />
                </label>
                <div className="flex gap-2">
                  <button type="button" disabled={shortlist.saving} onClick={() => void save(useAppStore.getState().shortlist)} className="rounded border border-cyan-700 px-3 py-1 text-sm text-cyan-200 disabled:opacity-40">Save rationale</button>
                  <button type="button" disabled={index === 0 || shortlist.saving} onClick={() => moveBy(index, -1)} className="rounded border border-slate-700 px-3 py-1 text-sm disabled:opacity-40">Move up</button>
                  <button type="button" disabled={index === localEntries.length - 1 || shortlist.saving} onClick={() => moveBy(index, 1)} className="rounded border border-slate-700 px-3 py-1 text-sm disabled:opacity-40">Move down</button>
                  <span className="self-center text-xs text-slate-500">Drag to reorder</span>
                </div>
              </div>
              <div className="space-y-2">
                <h2 className="text-sm font-semibold text-slate-300">Evidence (aggregate statistics only)</h2>
                <EvidencePanel playerId={entry.player_id} />
              </div>
            </article>
          ))}
        </div>
      )}
    </div>
  )
}

function PlayerName({ playerId, fallback }: { playerId: number; fallback?: string }) {
  const query = useQuery({
    queryKey: ['player-profile', playerId],
    queryFn: () => fetchJson<PlayerProfile>(`/api/players/${playerId}`),
  })
  return (
    <Link to={`/players/${playerId}`} className="mt-1 block text-lg font-semibold text-cyan-200 hover:underline">
      {query.data?.player_name ?? fallback ?? `Player ID ${playerId}`}
    </Link>
  )
}
