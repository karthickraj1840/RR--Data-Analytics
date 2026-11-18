export function LoadingState({ label }: { label: string }) {
  return <p role="status" className="text-sm text-slate-400">{label}</p>
}

export function EmptyState({ label }: { label: string }) {
  return <p className="rounded-lg border border-slate-700 bg-slate-900 p-4 text-sm text-slate-400">{label}</p>
}

export function ErrorState({
  message,
  onRetry,
}: {
  message: string
  onRetry?: () => void
}) {
  return (
    <div role="alert" className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-rose-800 bg-rose-950/50 p-4 text-sm text-rose-200">
      <span>{message}</span>
      {onRetry && <button type="button" onClick={onRetry} className="rounded border border-rose-700 px-3 py-1 hover:bg-rose-900">Retry</button>}
    </div>
  )
}

export function ChartPanel({
  title,
  note,
  children,
}: {
  title: string
  note?: string
  children: React.ReactNode
}) {
  return (
    <section className="space-y-3 rounded-xl border border-slate-700 bg-slate-900 p-4">
      <div>
        <h2 className="font-semibold">{title}</h2>
        {note && <p className="mt-1 text-xs text-slate-400">{note}</p>}
      </div>
      {children}
    </section>
  )
}

export function IntervalNote({ children }: { children?: React.ReactNode }) {
  return (
    <p className="text-xs text-slate-400">
      80% intervals are shown when supplied by the API. {children}
    </p>
  )
}
