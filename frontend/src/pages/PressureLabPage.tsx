import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { fetchJson } from '../api/client'
import { ChartPanel, EmptyState, ErrorState, IntervalNote, LoadingState } from '../components/PageStates'
import type { AppConfig, DataQuality, LeaderboardRequest, LeaderboardRow, WinProbabilityReport } from '../types'

const pressureDefinitions: LeaderboardRequest['pressure_definition'][] = [
  'composite',
  'high_leverage',
  'hard_chase',
  'death_overs',
  'collapse',
  'tight_finish',
]

type SliderDefinition = {
  path: string[]
  label: string
  min: number
  max: number
  step: number
  unit: string
  percent?: boolean
}

const sliders: SliderDefinition[] = [
  { path: ['eligibility_min_pressure_balls'], label: 'Eligibility pressure-ball minimum', min: 0, max: 300, step: 1, unit: 'balls' },
  { path: ['min_balls'], label: 'Minimum legal balls', min: 0, max: 300, step: 1, unit: 'legal balls' },
  { path: ['min_ipl_share'], label: 'Minimum IPL share', min: 0, max: 1, step: 0.01, unit: '%', percent: true },
  { path: ['all_rounder_eligibility', 'min_batting_pressure_balls'], label: 'All-rounder batting pressure minimum', min: 0, max: 300, step: 1, unit: 'balls' },
  { path: ['all_rounder_eligibility', 'min_bowling_pressure_balls'], label: 'All-rounder bowling pressure minimum', min: 0, max: 300, step: 1, unit: 'balls' },
  { path: ['pressure', 'high_leverage_quantile'], label: 'High-leverage quantile', min: 0.5, max: 1, step: 0.01, unit: 'quantile', percent: true },
  { path: ['pressure', 'hard_chase_rrr'], label: 'Hard-chase required run rate', min: 0, max: 20, step: 0.1, unit: 'runs/over' },
  { path: ['pressure', 'hard_chase_balls_left_max'], label: 'Hard-chase balls remaining maximum', min: 0, max: 120, step: 1, unit: 'balls' },
  { path: ['pressure', 'death_overs', '0'], label: 'Death-overs start', min: 1, max: 20, step: 1, unit: 'over' },
  { path: ['pressure', 'death_overs', '1'], label: 'Death-overs end', min: 1, max: 20, step: 1, unit: 'over' },
  { path: ['pressure', 'collapse_wickets'], label: 'Collapse wickets', min: 1, max: 10, step: 1, unit: 'wickets' },
  { path: ['pressure', 'collapse_over'], label: 'Collapse over threshold', min: 1, max: 20, step: 1, unit: 'overs' },
  { path: ['pressure', 'tight_finish_balls_left_max'], label: 'Tight-finish balls remaining maximum', min: 0, max: 60, step: 1, unit: 'balls' },
  { path: ['pressure', 'tight_finish_wp_min'], label: 'Tight-finish win-probability minimum', min: 0, max: 1, step: 0.01, unit: 'probability' },
  { path: ['pressure', 'tight_finish_wp_max'], label: 'Tight-finish win-probability maximum', min: 0, max: 1, step: 0.01, unit: 'probability' },
]

function configNumber(config: Record<string, unknown>, path: string[]) {
  let value: unknown = config
  for (const key of path) {
    if (value == null || typeof value !== 'object') return null
    value = (value as Record<string, unknown>)[key]
  }
  return typeof value === 'number' ? value : null
}

function overlapWithComposite(rows: Record<string, LeaderboardRow[]>) {
  const baseline = new Set((rows.composite ?? []).slice(0, 10).map((row) => row.player_id))
  return pressureDefinitions.map((definition) => {
    const ids = new Set((rows[definition] ?? []).slice(0, 10).map((row) => row.player_id))
    const common = [...baseline].filter((id) => ids.has(id)).length
    const union = new Set([...baseline, ...ids]).size
    return {
      definition: definition.replaceAll('_', ' '),
      overlap: common,
      jaccard: union ? common / union : null,
    }
  }).filter((row) => row.definition !== 'composite')
}

export function PressureLabPage() {
  const [overrides, setOverrides] = useState<Record<string, number>>({})
  const data = useQuery({
    queryKey: ['pressure-lab-data'],
    queryFn: async () => {
      const [config, quality, report] = await Promise.all([
        fetchJson<AppConfig>('/api/config'),
        fetchJson<DataQuality>('/api/data-quality'),
        fetchJson<WinProbabilityReport>('/api/win-probability/model-report'),
      ])
      return { config, quality, report }
    },
  })

  const minimumPressureBalls = overrides['eligibility_min_pressure_balls']
    ?? (data.data ? configNumber(data.data.config.config, ['eligibility_min_pressure_balls']) : null)
  const overlap = useQuery({
    queryKey: ['pressure-definition-overlap', minimumPressureBalls],
    enabled: minimumPressureBalls != null,
    queryFn: async () => {
      const entries = await Promise.all(pressureDefinitions.map(async (definition) => {
        const request: LeaderboardRequest = {
          role: 'combined',
          min_pressure_balls: Math.round(minimumPressureBalls ?? 0),
          pressure_definition: definition,
        }
        const rows = await fetchJson<LeaderboardRow[]>('/api/leaderboard', {
          method: 'POST',
          body: JSON.stringify(request),
        })
        return [definition, rows] as const
      }))
      return overlapWithComposite(Object.fromEntries(entries))
    },
  })

  if (data.isLoading) return <LoadingState label="Loading pressure configuration and model report…" />
  if (data.error) return <ErrorState message={data.error.message} onRetry={() => void data.refetch()} />
  if (!data.data) return <EmptyState label="No pressure-lab data was returned." />

  const { config, quality, report } = data.data
  const flagShares = Object.entries(quality.flag_shares.overall).map(([flag, share]) => ({
    flag: flag.replaceAll('_', ' '),
    share,
  }))
  const calibrationFor = (innings: string) => report.innings[innings]?.calibration_bins.map((bin) => ({
    ...bin,
    label: `Bin ${bin.bin}`,
  })) ?? []
  const sliderValues = (slider: SliderDefinition) => overrides[slider.path.join('.')]
    ?? configNumber(config.config, slider.path)
  const showShare = (value: number | null) => value == null ? '—' : `${(value * 100).toFixed(1)}%`

  return (
    <div className="space-y-6">
      <header className="space-y-2">
        <h1 className="text-3xl font-bold">Pressure Lab</h1>
        <p className="text-sm text-slate-400">Threshold sliders are local previews; changes are not persisted to backend configuration.</p>
        <IntervalNote>Calibration, flag-share, and ranking-overlap endpoints do not supply 80% intervals.</IntervalNote>
      </header>

      <section className="space-y-3 rounded-xl border border-slate-700 bg-slate-900 p-4">
        <h2 className="font-semibold">Configuration threshold preview</h2>
        <div className="grid gap-x-6 gap-y-4 sm:grid-cols-2">
          {sliders.map((slider) => {
            const value = sliderValues(slider)
            const key = slider.path.join('.')
            return (
              <label className="space-y-1 text-sm" key={key}>
                <span className="flex justify-between gap-2 text-slate-300">
                  <span>{slider.label}</span>
                  <span>{value == null ? 'Not provided' : slider.percent ? `${(value * 100).toFixed(0)}%` : `${value} ${slider.unit}`}</span>
                </span>
                {value != null && (
                  <input
                    type="range"
                    min={slider.min}
                    max={slider.max}
                    step={slider.step}
                    value={overrides[key] ?? value}
                    onChange={(event) => setOverrides((current) => ({ ...current, [key]: Number(event.target.value) }))}
                    className="w-full accent-cyan-400"
                  />
                )}
              </label>
            )
          })}
        </div>
        <p className="text-xs text-slate-500">The pressure-ball slider is applied to ranking-overlap requests. The current leaderboard API does not accept a minimum IPL share filter; that slider is preview-only.</p>
      </section>

      <ChartPanel title="Pressure-flag share of ball rows" note="Vertical axis: share of processed ball rows (%). The API returns marginal shares, not flag intersection counts.">
        {flagShares.length === 0 ? <EmptyState label="No pressure-flag shares were returned." /> : (
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={flagShares} margin={{ left: 10, right: 12, bottom: 36 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis dataKey="flag" angle={-20} textAnchor="end" interval={0} height={60} label={{ value: 'Pressure flag definition', position: 'insideBottom', offset: -24, fill: '#cbd5e1' }} tick={{ fill: '#cbd5e1', fontSize: 10 }} />
                <YAxis domain={[0, 1]} tickFormatter={(value) => `${Math.round(Number(value) * 100)}%`} label={{ value: 'Ball-row share (%)', angle: -90, position: 'insideLeft', fill: '#cbd5e1' }} tick={{ fill: '#94a3b8' }} />
                <Tooltip formatter={(value) => [`${(Number(value) * 100).toFixed(2)}% · 80% interval not supplied`, 'Share']} />
                <Bar dataKey="share" name="Share of ball rows" fill="#22d3ee" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
      </ChartPanel>

      <ChartPanel title="Top-10 player overlap by pressure definition" note="Overlap is the count of players shared with the composite-definition Top 10. Jaccard values are shown in the table; this is not ball-level flag overlap.">
        {overlap.isLoading && <LoadingState label="Loading ranking overlap…" />}
        {overlap.error && <ErrorState message={overlap.error.message} onRetry={() => void overlap.refetch()} />}
        {minimumPressureBalls == null && <EmptyState label="A pressure-ball minimum is not available in the configuration, so overlap cannot be computed." />}
        {overlap.data && overlap.data.length === 0 && <EmptyState label="No definition overlap results were returned." />}
        {overlap.data && overlap.data.length > 0 && (
          <>
            <div className="h-72">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={overlap.data} layout="vertical" margin={{ left: 22, right: 16 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis type="number" domain={[0, 10]} label={{ value: 'Shared players in Top 10 (players)', position: 'insideBottom', offset: -2, fill: '#cbd5e1' }} tick={{ fill: '#94a3b8' }} />
                  <YAxis type="category" dataKey="definition" width={120} label={{ value: 'Pressure definition', angle: -90, position: 'insideLeft', fill: '#cbd5e1' }} tick={{ fill: '#cbd5e1' }} />
                  <Tooltip formatter={(value, name, item) => [`${value} players · Jaccard ${item.payload.jaccard == null ? '—' : item.payload.jaccard.toFixed(2)} · 80% interval not supplied`, String(name)]} />
                  <Bar dataKey="overlap" name="Shared Top-10 players" fill="#a78bfa" />
                </BarChart>
              </ResponsiveContainer>
            </div>
            <p className="text-xs text-slate-400">Overlap uses a minimum of {Math.round(minimumPressureBalls ?? 0)} pressure balls. Flag-level intersections are unavailable from aggregate endpoints.</p>
          </>
        )}
      </ChartPanel>

      <section className="grid gap-5 lg:grid-cols-2">
        {(['1', '2'] as const).map((innings) => {
          const bins = calibrationFor(innings)
          return (
            <ChartPanel key={innings} title={`Innings ${innings} calibration`} note="Horizontal axis: probability bin (1–10); vertical axis: win probability (0–1).">
              {bins.length === 0 ? <EmptyState label={`No calibration bins for innings ${innings}.`} /> : (
                <div className="h-72">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={bins} margin={{ left: 12, right: 16, bottom: 28 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                      <XAxis dataKey="bin" type="number" domain={[1, 10]} tickFormatter={(value) => `Bin ${value}`} label={{ value: 'Probability bin (decile)', position: 'insideBottom', offset: -18, fill: '#cbd5e1' }} tick={{ fill: '#94a3b8' }} />
                      <YAxis domain={[0, 1]} tickFormatter={(value) => Number(value).toFixed(1)} label={{ value: 'Win probability (0–1)', angle: -90, position: 'insideLeft', fill: '#cbd5e1' }} tick={{ fill: '#94a3b8' }} />
                      <Tooltip formatter={(value, name, item) => {
                        const bin = item.payload
                        return [`${Number(value).toFixed(3)} · count ${bin.count} · gap ${showShare(bin.gap)} · 80% interval not supplied`, String(name)]
                      }} labelFormatter={(value) => `Bin ${value}`} />
                      <Legend />
                      <Line type="monotone" dataKey="mean_predicted" name="Mean predicted" stroke="#22d3ee" connectNulls />
                      <Line type="monotone" dataKey="mean_observed" name="Mean observed" stroke="#fbbf24" connectNulls />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              )}
            </ChartPanel>
          )
        })}
      </section>
      <p className="text-xs text-slate-500">Maximum calibration gap: {(report.calibration_max_abs_gap * 100).toFixed(2)} percentage points. Data version: {config.data_version}</p>
    </div>
  )
}
