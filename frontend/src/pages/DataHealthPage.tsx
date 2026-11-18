import { useQuery } from '@tanstack/react-query'
import { fetchJson } from '../api/client'
import { EmptyState, ErrorState, IntervalNote, LoadingState } from '../components/PageStates'
import type { DataQuality } from '../types'

const integer = new Intl.NumberFormat('en-US')
const percent = (value: number) => `${(value * 100).toFixed(1)}%`

export function DataHealthPage() {
  const query = useQuery({
    queryKey: ['data-quality'],
    queryFn: () => fetchJson<DataQuality>('/api/data-quality'),
  })

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-3xl font-bold">Data health</h1>
        <p className="mt-2 text-sm text-slate-400">Aggregate coverage, exclusions, and pressure-flag shares from the processed dataset.</p>
      </header>
      {query.isLoading && <LoadingState label="Loading data quality…" />}
      {query.error && <ErrorState message={query.error.message} onRetry={() => void query.refetch()} />}
      {query.data && (
        <>
          <section className="grid gap-4 sm:grid-cols-3">
            <Metric label="Matches used" value={integer.format(query.data.matches_used)} />
            <Metric label="Ball rows used" value={integer.format(query.data.balls_used)} />
            <Metric label="Flag-share denominator" value={integer.format(query.data.flag_shares.n_balls)} />
          </section>
          <IntervalNote>Counts are descriptive; flag-share intervals are not supplied by the data-quality API.</IntervalNote>

          <section className="space-y-3">
            <h2 className="text-xl font-semibold">Coverage by competition group</h2>
            {query.data.by_comp_group.length === 0 ? <EmptyState label="No competition-group aggregates were returned." /> : (
              <UsageTable headers={['Competition group', 'Matches', 'Ball rows', 'Legal balls']}>
                {query.data.by_comp_group.map((row) => (
                  <tr key={row.comp_group}>
                    <td className="px-4 py-3">{row.comp_group}</td>
                    <td className="px-4 py-3">{integer.format(row.matches)}</td>
                    <td className="px-4 py-3">{integer.format(row.balls)}</td>
                    <td className="px-4 py-3">{integer.format(row.legal_balls)}</td>
                  </tr>
                ))}
              </UsageTable>
            )}
          </section>

          <section className="space-y-3">
            <h2 className="text-xl font-semibold">Coverage by season</h2>
            {query.data.by_season.length === 0 ? <EmptyState label="No season aggregates were returned." /> : (
              <UsageTable headers={['Season', 'Matches', 'Ball rows', 'Legal balls']}>
                {query.data.by_season.map((row) => (
                  <tr key={row.season}>
                    <td className="px-4 py-3">{row.season}</td>
                    <td className="px-4 py-3">{integer.format(row.matches)}</td>
                    <td className="px-4 py-3">{integer.format(row.balls)}</td>
                    <td className="px-4 py-3">{integer.format(row.legal_balls)}</td>
                  </tr>
                ))}
              </UsageTable>
            )}
          </section>

          <section className="space-y-3">
            <h2 className="text-xl font-semibold">Excluded data</h2>
            {Object.keys(query.data.exclusions).length === 0 ? <EmptyState label="No exclusion aggregates were returned." /> : (
              <UsageTable headers={['Reason', 'Rows dropped', 'Balls dropped', 'Matches dropped']}>
                {Object.entries(query.data.exclusions).map(([reason, counts]) => (
                  <tr key={reason}>
                    <td className="px-4 py-3">{reason.replaceAll('_', ' ')}</td>
                    <td className="px-4 py-3">{integer.format(counts.rows_dropped)}</td>
                    <td className="px-4 py-3">{integer.format(counts.balls_dropped)}</td>
                    <td className="px-4 py-3">{integer.format(counts.matches_dropped)}</td>
                  </tr>
                ))}
              </UsageTable>
            )}
          </section>

          <section className="space-y-3">
            <h2 className="text-xl font-semibold">Pressure flag shares</h2>
            {Object.keys(query.data.flag_shares.overall).length === 0 ? <EmptyState label="No pressure flag shares were returned." /> : (
              <div className="overflow-x-auto rounded-xl border border-slate-700">
                <table className="min-w-full divide-y divide-slate-700 text-left text-sm">
                  <thead className="bg-slate-900 text-slate-300">
                    <tr>
                      <th className="px-4 py-3">Group</th>
                      {Object.keys(query.data.flag_shares.overall).map((flag) => <th className="px-4 py-3" key={flag}>{flag.replaceAll('_', ' ')} (%)</th>)}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800">
                    <tr>
                      <td className="px-4 py-3">Overall</td>
                      {Object.keys(query.data.flag_shares.overall).map((flag) => <td className="px-4 py-3" key={flag}>{percent(query.data.flag_shares.overall[flag])}</td>)}
                    </tr>
                    {Object.entries(query.data.flag_shares.by_comp_group).map(([group, shares]) => (
                      <tr key={group}>
                        <td className="px-4 py-3">{group}</td>
                        {Object.keys(query.data.flag_shares.overall).map((flag) => <td className="px-4 py-3" key={flag}>{shares[flag] == null ? '—' : percent(shares[flag])}</td>)}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </section>
        </>
      )}
    </div>
  )
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl border border-slate-700 bg-slate-900 p-4">
      <p className="text-sm text-slate-400">{label}</p>
      <p className="mt-2 text-2xl font-semibold">{value}</p>
    </div>
  )
}

function UsageTable({ headers, children }: { headers: string[]; children: React.ReactNode }) {
  return (
    <div className="overflow-x-auto rounded-xl border border-slate-700">
      <table className="min-w-full divide-y divide-slate-700 text-left text-sm">
        <thead className="bg-slate-900 text-slate-300">
          <tr>{headers.map((header) => <th className="px-4 py-3" key={header}>{header}</th>)}</tr>
        </thead>
        <tbody className="divide-y divide-slate-800">{children}</tbody>
      </table>
    </div>
  )
}
