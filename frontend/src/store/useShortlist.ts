import { useEffect, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { fetchJson } from '../api/client'
import type { ShortlistApiEntry, Top10Report } from '../types'
import { useAppStore, type ShortlistEntry } from './useAppStore'

const shortlistKey = ['shortlist']

function ordered(entries: ShortlistEntry[]) {
  return entries.map((entry, index) => ({
    ...entry,
    order: index + 1,
    rationale: entry.rationale ?? '',
  }))
}

export function useShortlist() {
  const [hydrated, setHydrated] = useState(false)
  const queryClient = useQueryClient()
  const entries = useAppStore((state) => state.shortlist)
  const replaceShortlist = useAppStore((state) => state.replaceShortlist)
  const query = useQuery({
    queryKey: shortlistKey,
    queryFn: () => fetchJson<ShortlistApiEntry[]>('/api/shortlist'),
  })
  const mutation = useMutation({
    mutationFn: (next: ShortlistEntry[]) => fetchJson<ShortlistApiEntry[]>('/api/shortlist', {
      method: 'PUT',
      body: JSON.stringify(ordered(next).map(({ player_id, order, rationale }) => ({
        player_id,
        order,
        rationale,
      }))),
    }),
  })

  useEffect(() => {
    if (!query.data) return
    const names = new Map(useAppStore.getState().shortlist.map((entry) => [entry.player_id, entry.display_name]))
    replaceShortlist(ordered([...query.data].sort((a, b) => a.order - b.order).map((entry) => ({
      ...entry,
      display_name: names.get(entry.player_id),
    }))))
    setHydrated(true)
  }, [query.data, replaceShortlist])

  async function save(nextEntries: ShortlistEntry[]) {
    if (nextEntries.length > 10) throw new Error('The shortlist is limited to 10 players.')
    const normalized = ordered(nextEntries)
    replaceShortlist(normalized)
    try {
      const saved = await mutation.mutateAsync(normalized)
      const names = new Map(normalized.map((entry) => [entry.player_id, entry.display_name]))
      const enriched = saved.map((entry) => ({
        ...entry,
        display_name: names.get(entry.player_id),
      }))
      replaceShortlist(enriched)
      queryClient.setQueryData(shortlistKey, saved)
    } catch (error) {
      queryClient.invalidateQueries({ queryKey: shortlistKey })
      throw error
    }
  }

  async function add(playerId: number, displayName?: string) {
    const current = useAppStore.getState().shortlist
    if (current.some((entry) => entry.player_id === playerId)) return
    if (current.length >= 10) throw new Error('The shortlist is limited to 10 players.')
    await save([...current, {
      player_id: playerId,
      order: current.length + 1,
      rationale: '',
      display_name: displayName,
    }])
  }

  async function addTop10() {
    const report = await fetchJson<Top10Report>('/api/top10')
    const current = useAppStore.getState().shortlist
    const ids = new Set(current.map((entry) => entry.player_id))
    const next = [...current]
    for (const player of report.picks) {
      if (next.length >= 10) break
      if (ids.has(player.player_id)) continue
      ids.add(player.player_id)
      next.push({
        player_id: player.player_id,
        order: next.length + 1,
        rationale: '',
        display_name: player.display_name,
      })
    }
    if (next.length === current.length) return
    await save(next)
  }

  return {
    entries,
    loading: query.isLoading || (query.data !== undefined && !hydrated),
    error: query.error,
    retry: query.refetch,
    saving: mutation.isPending,
    save,
    add,
    addTop10,
    replaceShortlist,
  }
}
