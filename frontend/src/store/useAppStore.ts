import { create } from 'zustand'

export type ShortlistEntry = {
  player_id: number
  order: number
  rationale: string
  display_name?: string
}

type Store = {
  shortlist: ShortlistEntry[]
  replaceShortlist: (entries: ShortlistEntry[]) => void
}

export const useAppStore = create<Store>((set) => ({
  shortlist: [],
  replaceShortlist: (entries) => set({ shortlist: entries }),
}))
