import { useQuery } from '@tanstack/react-query'
import { fetchJson } from '../api/client'
import { EmptyState, ErrorState, LoadingState } from '../components/PageStates'
import type { SquadFitReport } from '../types'

function display(value: unknown) {
  if (value == null || value === '' || value === 'not_provided') return 'Not provided'
  return String(value)
}

export function SquadFitPage() {
  const query = useQuery({
    queryKey: ['squad-fit'],
    queryFn: () => fetchJson<SquadFitReport>('/api/squad-fit'),
  })

  if (query.isLoading) return <LoadingState label="Loading squad-fit data…" />
  if (query.error) return <ErrorState message={query.error.message} onRetry={() => void query.refetch()} />
  if (!query.data) return <EmptyState label="No squad-fit response was returned." />

  const { rows, swaps } = query.data
  if (rows.length === 0) return <EmptyState label="No squad-fit rows are available." />

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-3xl font-bold">Squad fit</h1>
      </header>
      <section className="space-y-3">
        <h2 className="text-xl font-semibold">Top 20 and proposed final 10</h2>
        <div className="overflow-x-auto rounded-xl border border-slate-700">
          <table className="min-w-full divide-y divide-slate-700 text-left text-sm">
            <thead className="bg-slate-900 text-slate-300">
              <tr>
                {['Rank', 'Name', 'Role', 'Grade', 'Pressure balls', 'IPL share', 'RR status', 'Overseas', 'Age', 'Price band', 'Flag', 'Final list'].map((column) => (
                  <th className="whitespace-nowrap px-4 py-3" key={column}>{column}</th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800">
              {rows.map((row, index) => (
                <tr key={`${display(row.player_id)}-${index}`}>
                  <td className="px-4 py-3">{display(row.rank)}</td>
                  <td className="whitespace-nowrap px-4 py-3">{display(row.player_name)}</td>
                  <td className="px-4 py-3">{display(row.role)}</td>
                  <td className="px-4 py-3">{display(row.grade)}</td>
                  <td className="px-4 py-3">{display(row.pressure_balls)}</td>
                  <td className="px-4 py-3">{display(row.ipl_share)}</td>
                  <td className="px-4 py-3">{display(row.rr_status)}</td>
                  <td className="px-4 py-3">{display(row.overseas)}</td>
                  <td className="px-4 py-3">{display(row.age)}</td>
                  <td className="px-4 py-3">{display(row.price_band)}</td>
                  <td className="px-4 py-3">{display(row.fit_flag)}</td>
                  <td className="px-4 py-3">{row.final_list === 'true' ? 'Yes' : 'No'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
      <section className="space-y-3">
        <h2 className="text-xl font-semibold">Swaps</h2>
        {swaps.length === 0 ? <EmptyState label="No swaps were made." /> : (
          <div className="overflow-x-auto rounded-xl border border-slate-700">
            <table className="min-w-full divide-y divide-slate-700 text-left text-sm">
              <thead className="bg-slate-900 text-slate-300">
                <tr><th className="px-4 py-3">From</th><th className="px-4 py-3">To</th><th className="px-4 py-3">Reason</th></tr>
              </thead>
              <tbody className="divide-y divide-slate-800">
                {swaps.map((swap, index) => (
                  <tr key={`${display(swap.old_rank)}-${index}`}>
                    <td className="px-4 py-3">{display(swap.old_rank)}: {display(swap.old_player_name)}</td>
                    <td className="px-4 py-3">{display(swap.new_rank)}: {display(swap.new_player_name)}</td>
                    <td className="px-4 py-3">{display(swap.reason)}</td>
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
