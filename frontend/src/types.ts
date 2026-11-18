export type ApiHealth = {
  data_version: string
  innings_log_loss: Record<string, number>
  matches_used: number
  status: string
}

export type GroupUsage = {
  comp_group: string
  matches: number
  balls: number
  legal_balls: number
}

export type SeasonUsage = {
  season: string
  matches: number
  balls: number
  legal_balls: number
}

export type DataQuality = {
  data_version: string
  exclusions: Record<string, {
    rows_dropped: number
    balls_dropped: number
    matches_dropped: number
  }>
  matches_used: number
  balls_used: number
  by_comp_group: GroupUsage[]
  by_season: SeasonUsage[]
  flag_shares: {
    n_balls: number
    overall: Record<string, number>
    by_comp_group: Record<string, Record<string, number>>
  }
}

export type LeaderboardRequest = {
  role?: 'batter' | 'bowler' | 'all_rounder' | 'developing' | 'combined'
  comp_group?: 'IPL' | 'International' | 'Emerging'
  min_pressure_balls: number
  pressure_definition: 'composite' | 'high_leverage' | 'hard_chase' | 'death_overs' | 'collapse' | 'tight_finish'
}

export type LeaderboardRow = {
  rank: number
  player_id: number
  player_name: string
  role: string
  comp_group: string | null
  score: number
  wpa_per_ball: number
  wpa_80_interval: [number | null, number | null]
  pressure_balls: number
  rank_stability: number
  small_sample: boolean
  small_sample_threshold: number
  pressure_definition: string
}

export type Top10Entry = {
  rank: number
  player_id: number
  display_name: string
  role: 'batter' | 'bowler' | 'all_rounder'
  role_label: 'batter' | 'bowler' | 'all_rounder'
  selection_slot: 'batter' | 'bowler'
  pressure_balls: number
  ipl_share: number
  key_metrics_vs_population: Record<string, unknown>
  wpa_80_interval: [number, number] | null
  selected_predictor_wpa_per_ball: number
  bootstrap_inclusion_probability: number
  bootstrap_rank_interval_80?: [number, number] | null
  rank_stability: number
  grade: 'A' | 'B' | 'C'
}

export type Top10Report = {
  data_version: string
  picks: Top10Entry[]
  reserves: Top10Entry[]
  grade_counts: Record<string, Record<string, number>>
  grade_thresholds: Record<string, Record<string, number>>
  eligibility: Record<string, unknown>
  role_constraint: {
    note: string
    configured_split: { batters: number; bowlers: number }
  }
}

export type PlayerProfile = {
  player_id: number
  player_name: string
  role: string
  pressure_balls: number
  pressure_vs_normal: Record<string, number | null>
  phase_splits: {
    batting_share: Record<string, number | null>
    bowling: Record<string, Record<string, number | null>>
  }
  stroke_aggregates: Array<Record<string, number | string | null>>
  season_trend: Array<Record<string, number | string | null>>
}

export type SimilarPlayer = {
  player_id: number
  similar_player_id: number
  display_name: string
  role: string
  distance: number
}

export type SensitivityPlayer = {
  player_id: number
  inclusion_probability: number
  rank_stability: number
  rank_interval_80: [number, number] | null
}

export type SensitivityReport = {
  data_version: string
  bootstrap: Record<string, number | string>
  players: SensitivityPlayer[]
  scenario_count: number
  scenario_sweep: Record<string, unknown>
}

export type BacktestMetric = {
  spearman: number | null
  precision_at_10: number | null
  random_baseline_precision_at_10: number | null
  predicted_top10_wpa_uplift: number | null
  predicted_top10_mean_realized_pressure_wpa_per_ball: number | null
  population_mean_realized_pressure_wpa_per_ball: number | null
}

export type BacktestFold = {
  train_years: Array<string | number>
  test_years: Array<string | number>
  train_match_count: number
  test_match_count: number
  predictors: Record<string, {
    by_role: Record<string, BacktestMetric>
  }>
}

export type BacktestReport = {
  data_version: string
  selected_predictor: string | Record<string, unknown>
  folds: Record<string, BacktestFold>
  predictor_decision: Record<string, unknown>
  selection_comparison: Record<string, unknown>
  random_baseline?: number
  weak_results?: unknown
}

export type ModelCalibrationBin = {
  bin: number
  mean_predicted: number | null
  mean_observed: number | null
  count: number
  gap: number | null
}

export type ModelInningsReport = {
  log_loss: number
  brier: number
  n_balls: number
  n_matches: number
  mean_wp_before: number
  calibration_max_abs_gap: number
  calibration_bins: ModelCalibrationBin[]
}

export type WinProbabilityReport = {
  data_version: string
  innings: Record<string, ModelInningsReport>
  calibration_max_abs_gap: number
}

export type AppConfig = {
  data_version: string
  config: Record<string, unknown>
}

export type SquadFitReport = {
  templates: Record<string, string>
  rows: Array<Record<string, unknown>>
  swaps: Array<Record<string, unknown>>
}

export type ShortlistApiEntry = {
  player_id: number
  order: number
  rationale: string
}
