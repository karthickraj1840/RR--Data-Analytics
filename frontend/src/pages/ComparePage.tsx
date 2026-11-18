import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  PolarAngleAxis,
  PolarGrid,
  PolarRadiusAxis,
  Radar,
  RadarChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { fetchJson } from '../api/client'
import { ChartPanel, EmptyState, ErrorState, IntervalNote, LoadingState } from '../components/PageStates'
import type { LeaderboardRequest, LeaderboardRow, PlayerProfile } from '../types'

const optionRequest: LeaderboardRequest = {
  role: 'combined',
  min_pressure_balls: 0,
  pressure_definition: 'composite',
}

type RadarMetric = {
  key: string
  label: string
  unit: string
  lowerIsBetter?: boolean
}

const radarMetrics: RadarMetric[] = [
  { key: 'batting_pressure_wpa_per_ball', label: 'Batting pressure WPA / ball', unit: 'WPA/ball' },
  { key: 'batting_pressure_strike_rate', label: 'Pressure strike rate', unit: 'runs/100 balls' },
  { key: 'bowling_pressure_economy', label: 'Bowling pressure economy', unit: 'runs/over', lowerIsBetter: true },
  { key: 'batting_pressure_vs_normal_delta', label: 'Pressure WPA delta', unit: 'WPA/ball' },
]

function metricValue(profile: PlayerProfile, key: string) {
  return profile.pressure_vs_normal[key] ?? null
}

function tooltipValue(value: unknown, label: string) {
  if (typeof value !== 'number' || !Number.isFinite(value)) return ['Not provided by API', label]
  return [`${value.toFixed(4)} · 80% interval not supplied`, label]
}

export function ComparePage() {
  const [selected, setSelected] = useState(['', ''])
  const [submittedIds, setSubmittedIds] = useState<number[]>([])
  const [selectionError, setSelectionError] = useState<string | null>(null)
  const options = useQuery({
    queryKey: ['compare-player-options'],
    queryFn: () => fetchJson<LeaderboardRow[]>('/api/leaderboard', {
      method: 'POST',
      body: JSON.stringify(optionRequest),
    }),
  })
  const comparison = useQuery({
    queryKey: ['compare', submittedIds],
    enabled: submittedIds.length >= 2,
    queryFn: () => fetchJson<PlayerProfile[]>('/api/compare', {
      method: 'POST',
      body: JSON.stringify({ player_ids: submittedIds }),
    }),
  })

  function submit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault()
    const ids = selected.filter(Boolean).map(Number)
    if (ids.length < 2 || ids.length > 4) {
      setSelectionError('Choose between two and four players.')
      return
    }
    if (ids.some((id) => !Number.isSafeInteger(id) || id <= 0)) {
      setSelectionError('Player IDs must be positive integers.')
      return
    }
    if (new Set(ids).size !== ids.length) {
      setSelectionError('Choose a player only once.')
      return
    }
    setSelectionError(null)
    setSubmittedIds(ids)
  }

  const profiles = comparison.data ?? []
  const radarData = radarMetrics.map((metric) => {
    const rawValues = profiles.map((profile) => metricValue(profile, metric.key))
    const finiteValues = rawValues.filter((value): value is number => value != null && Number.isFinite(value))
    const min = finiteValues.length ? Math.min(...finiteValues) : 0
    const max = finiteValues.length ? Math.max(...finiteValues) : 0
    const result: Record<string, string | number> = { metric: `${metric.label} (${metric.unit})` }
    profiles.forEach((profile, index) => {
      const raw = rawValues[index]
      const normalized = raw == null
        ? 0
        : max === min
          ? 50
          : ((metric.lowerIsBetter ? max - raw : raw - min) / (max - min)) * 100
      result[`${profile.player_name} (${profile.player_id})`] = normalized
    })
    return result
  })

  const barData = profiles.map((profile) => ({
    player: `${profile.player_name} (${profile.player_id})`,
    pressureWpa: profile.pressure_vs_normal.batting_pressure_wpa_per_ball,
    normalWpa: profile.pressure_vs_normal.batting_normal_wpa_per_ball,
    pressureStrikeRate: profile.pressure_vs_normal.batting_pressure_strike_rate,
    normalStrikeRate: profile.pressure_vs_normal.batting_normal_strike_rate,
  }))
  const hasRadarMetrics = radarMetrics.some((metric) => profiles.some((profile) => metricValue(profile, metric.key) != null))
  const hasWpaMetrics = barData.some((row) => row.pressureWpa != null || row.normalWpa != null)
  const hasStrikeRateMetrics = barData.some((row) => row.pressureStrikeRate != null || row.normalStrikeRate != null)

  return (
    <div className="space-y-6">
      <header className="space-y-2">
        <h1 className="text-3xl font-bold">Compare players</h1>
        <p className="text-sm text-slate-400">Choose two to four players. Charts use only aggregate fields returned by the compare endpoint.</p>
      </header>
      {options.isLoading && <LoadingState label="Loading player choices…" />}
      {options.error && <ErrorState message={options.error.message} onRetry={() => void options.refetch()} />}
      {options.data && options.data.length === 0 && <EmptyState label="No player choices were returned." />}
      {options.data && options.data.length > 0 && (
        <form onSubmit={submit} className="space-y-3 rounded-xl border border-slate-700 bg-slate-900 p-4">
          {selected.map((playerId, index) => (
            <div className="flex gap-2" key={index}>
              <label className="sr-only" htmlFor={`compare-player-${index}`}>Player {index + 1}</label>
              <select
                id={`compare-player-${index}`}
                value={playerId}
                onChange={(event) => setSelected((current) => current.map((value, row) => row === index ? event.target.value : value))}
                className="min-w-0 flex-1 rounded border border-slate-600 bg-slate-950 px-3 py-2"
              >
                <option value="">Choose player</option>
                {options.data.map((option) => (
                  <option value={option.player_id} key={option.player_id}>
                    {option.player_name} · {option.role} · ID {option.player_id}
                  </option>
                ))}
              </select>
              {selected.length > 2 && <button type="button" onClick={() => setSelected((current) => current.filter((_, row) => row !== index))} className="rounded border border-slate-700 px-3 text-sm">Remove</button>}
            </div>
          ))}
          <div className="flex flex-wrap gap-2">
            {selected.length < 4 && <button type="button" onClick={() => setSelected((current) => [...current, ''])} className="rounded border border-slate-700 px-3 py-2 text-sm">Add player</button>}
            <button type="submit" disabled={comparison.isFetching} className="rounded-lg bg-cyan-500 px-4 py-2 font-semibold text-slate-950 disabled:opacity-50">Compare</button>
          </div>
        </form>
      )}
      {!comparison.data && submittedIds.length === 0 && !options.isLoading && !options.error && (
        <EmptyState label="Choose two to four players to see their aggregate comparison." />
      )}
      {selectionError && <ErrorState message={selectionError} />}
      {comparison.isLoading && <LoadingState label="Comparing selected players…" />}
      {comparison.error && <ErrorState message={comparison.error.message} onRetry={() => void comparison.refetch()} />}
      {submittedIds.length >= 2 && comparison.data?.length === 0 && <EmptyState label="The compare endpoint returned no player aggregates." />}
      {profiles.length > 0 && (
        <div className="space-y-5">
          <IntervalNote>The compare endpoint provides no 80% intervals. Tooltips identify interval availability.</IntervalNote>
          <ChartPanel title="Pressure performance radar" note="Each axis is normalized within the chosen players to a relative 0–100 scale; labels include the underlying unit.">
            {!hasRadarMetrics ? <EmptyState label="No comparable radar metrics were returned." /> : <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <RadarChart data={radarData}>
                  <PolarGrid />
                  <PolarAngleAxis dataKey="metric" tick={{ fill: '#cbd5e1', fontSize: 11 }} />
                  <PolarRadiusAxis angle={30} domain={[0, 100]} unit=" relative points" tick={{ fill: '#94a3b8', fontSize: 10 }} />
                  {profiles.map((profile, index) => (
                    <Radar key={profile.player_id} name={`${profile.player_name} (${profile.player_id})`} dataKey={`${profile.player_name} (${profile.player_id})`} stroke={['#22d3ee', '#fbbf24', '#a78bfa', '#fb7185'][index]} fill={['#22d3ee', '#fbbf24', '#a78bfa', '#fb7185'][index]} fillOpacity={0.12} />
                  ))}
                  <Legend />
                  <Tooltip formatter={(value, name, item) => {
                    const player = profiles.find((candidate) => `${candidate.player_name} (${candidate.player_id})` === String(name))
                    const metric = radarMetrics.find((candidate) => `${candidate.label} (${candidate.unit})` === item.payload.metric)
                    const raw = player && metric ? metricValue(player, metric.key) : null
                    return [`${Number(value).toFixed(1)} relative points · raw ${raw == null ? 'not provided' : `${raw.toFixed(4)} ${metric?.unit}`} · 80% interval not supplied`, String(name)]
                  }} />
                </RadarChart>
              </ResponsiveContainer>
            </div>}
          </ChartPanel>
          <div className="grid gap-5 lg:grid-cols-2">
            <ChartPanel title="Pressure vs normal WPA / ball" note="Unit: win-probability added per legal ball. 80% intervals are not supplied by the compare endpoint.">
              {!hasWpaMetrics ? <EmptyState label="No batting WPA aggregates were returned." /> : <div className="h-72">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={barData} margin={{ left: 12, right: 12, bottom: 45 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                    <XAxis dataKey="player" angle={-20} textAnchor="end" interval={0} height={70} label={{ value: 'Player (name and ID)', position: 'insideBottom', offset: -38, fill: '#cbd5e1' }} tick={{ fill: '#cbd5e1', fontSize: 10 }} />
                    <YAxis label={{ value: 'WPA / legal ball', angle: -90, position: 'insideLeft', fill: '#cbd5e1' }} tick={{ fill: '#94a3b8' }} />
                    <Tooltip formatter={(value, name) => tooltipValue(value, String(name))} />
                    <Legend />
                    <Bar dataKey="pressureWpa" name="Pressure WPA / ball" fill="#22d3ee" />
                    <Bar dataKey="normalWpa" name="Normal WPA / ball" fill="#64748b" />
                  </BarChart>
                </ResponsiveContainer>
                </div>}
            </ChartPanel>
            <ChartPanel title="Pressure vs normal strike rate" note="Unit: runs per 100 balls. 80% intervals are not supplied by the compare endpoint.">
                {!hasStrikeRateMetrics ? <EmptyState label="No batting strike-rate aggregates were returned." /> : <div className="h-72">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={barData} margin={{ left: 12, right: 12, bottom: 45 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                    <XAxis dataKey="player" angle={-20} textAnchor="end" interval={0} height={70} label={{ value: 'Player (name and ID)', position: 'insideBottom', offset: -38, fill: '#cbd5e1' }} tick={{ fill: '#cbd5e1', fontSize: 10 }} />
                    <YAxis label={{ value: 'Runs / 100 balls', angle: -90, position: 'insideLeft', fill: '#cbd5e1' }} tick={{ fill: '#94a3b8' }} />
                    <Tooltip formatter={(value, name) => tooltipValue(value, String(name))} />
                    <Legend />
                    <Bar dataKey="pressureStrikeRate" name="Pressure strike rate" fill="#fbbf24" />
                    <Bar dataKey="normalStrikeRate" name="Normal strike rate" fill="#64748b" />
                  </BarChart>
                </ResponsiveContainer>
                </div>}
            </ChartPanel>
          </div>
        </div>
      )}
    </div>
  )
}
