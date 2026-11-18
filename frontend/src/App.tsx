import { lazy, Suspense } from 'react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { BrowserRouter, Route, Routes } from 'react-router-dom'
import { Layout } from './components/Layout'
import { LoadingState } from './components/PageStates'

const OverviewPage = lazy(() => import('./pages/OverviewPage').then((module) => ({ default: module.OverviewPage })))
const Top10Page = lazy(() => import('./pages/Top10Page').then((module) => ({ default: module.Top10Page })))
const DataHealthPage = lazy(() => import('./pages/DataHealthPage').then((module) => ({ default: module.DataHealthPage })))
const PressureLabPage = lazy(() => import('./pages/PressureLabPage').then((module) => ({ default: module.PressureLabPage })))
const LeaderboardPage = lazy(() => import('./pages/LeaderboardPage').then((module) => ({ default: module.LeaderboardPage })))
const ProfilePage = lazy(() => import('./pages/ProfilePage').then((module) => ({ default: module.ProfilePage })))
const ComparePage = lazy(() => import('./pages/ComparePage').then((module) => ({ default: module.ComparePage })))
const SensitivityPage = lazy(() => import('./pages/SensitivityPage').then((module) => ({ default: module.SensitivityPage })))
const BacktestPage = lazy(() => import('./pages/BacktestPage').then((module) => ({ default: module.BacktestPage })))
const SquadFitPage = lazy(() => import('./pages/SquadFitPage').then((module) => ({ default: module.SquadFitPage })))
const ShortlistPage = lazy(() => import('./pages/ShortlistPage').then((module) => ({ default: module.ShortlistPage })))
const MethodologyPage = lazy(() => import('./pages/MethodologyPage').then((module) => ({ default: module.MethodologyPage })))

const queryClient = new QueryClient({
  defaultOptions: {
    queries: { retry: false },
    mutations: { retry: false },
  },
})

export function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Layout>
          <Suspense fallback={<LoadingState label="Loading page…" />}>
            <Routes>
              <Route path="/" element={<OverviewPage />} />
              <Route path="/overview" element={<OverviewPage />} />
              <Route path="/top10" element={<Top10Page />} />
              <Route path="/data-health" element={<DataHealthPage />} />
              <Route path="/pressure" element={<PressureLabPage />} />
              <Route path="/leaderboard" element={<LeaderboardPage />} />
              <Route path="/players/:playerId" element={<ProfilePage />} />
              <Route path="/compare" element={<ComparePage />} />
              <Route path="/sensitivity" element={<SensitivityPage />} />
              <Route path="/backtest" element={<BacktestPage />} />
              <Route path="/squad-fit" element={<SquadFitPage />} />
              <Route path="/shortlist" element={<ShortlistPage />} />
              <Route path="/methodology" element={<MethodologyPage />} />
            </Routes>
          </Suspense>
        </Layout>
      </BrowserRouter>
    </QueryClientProvider>
  )
}
