import { Link } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { fetchJson } from '../api/client'
import { EmptyState, ErrorState, IntervalNote, LoadingState } from '../components/PageStates'
import type { ApiHealth, DataQuality } from '../types'

export function OverviewPage() {
  const query = useQuery({
    queryKey: ['overview'],
    queryFn: async () => {
      const [health, quality] = await Promise.all([
        fetchJson<ApiHealth>('/api/health'),
        fetchJson<DataQuality>('/api/data-quality'),
      ])
      return { health, quality }
    },
  })

  if (query.isLoading) return <LoadingState label="Loading dashboard overview…" />
  if (query.error) return <ErrorState message={query.error.message} onRetry={() => void query.refetch()} />
  if (!query.data) return <EmptyState label="No overview aggregates were returned." />

  const { health, quality } = query.data
  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold">Overview</h1>
      {health.matches_used === 0 || quality.balls_used === 0 ? <EmptyState label="The API returned no usable match or ball aggregates." /> : null}
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Metric label="API status" value={health.status} />
        <Metric label="Matches used" value={`${health.matches_used.toLocaleString()} matches`} />
        <Metric label="Ball rows used" value={`${quality.balls_used.toLocaleString()} rows`} />
        <Metric label="Data version" value={health.data_version} />
      </div>
      <section className="rounded-xl border border-slate-700 bg-slate-900 p-5">
        <h2 className="text-lg font-semibold">Win-probability model quality</h2>
        <dl className="mt-3 grid gap-3 sm:grid-cols-2">
          <div><dt className="text-sm text-slate-400">Innings 1 log loss</dt><dd className="mt-1">{health.innings_log_loss['1']?.toFixed(4) ?? 'Not provided'} · 80% interval not supplied</dd></div>
          <div><dt className="text-sm text-slate-400">Innings 2 log loss</dt><dd className="mt-1">{health.innings_log_loss['2']?.toFixed(4) ?? 'Not provided'} · 80% interval not supplied</dd></div>
        </dl>
        <IntervalNote>Dataset counts are descriptive and have no supplied 80% intervals.</IntervalNote>
      </section>
      <Link to="/data-health" className="inline-flex rounded-lg border border-cyan-700 px-4 py-2 text-sm text-cyan-200 hover:bg-slate-900">
        View data health and quality details
      </Link>
    </div>
  )
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl border border-slate-700 bg-slate-900 p-4">
      <p className="text-sm text-slate-400">{label}</p>
      <p className="mt-2 break-all text-lg font-semibold">{value}</p>
    </div>
  )
}
