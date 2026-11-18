from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    data_version: str
    matches_used: int
    innings_log_loss: dict[str, float]


class GroupUsage(BaseModel):
    comp_group: str
    matches: int
    balls: int
    legal_balls: int


class SeasonUsage(BaseModel):
    season: str
    matches: int
    balls: int
    legal_balls: int


class DataQualityResponse(BaseModel):
    data_version: str
    exclusions: dict[str, Any]
    matches_used: int
    balls_used: int
    by_comp_group: list[GroupUsage]
    by_season: list[SeasonUsage]
    flag_shares: dict[str, Any]


class CalibrationBin(BaseModel):
    bin: int
    mean_predicted: float | None
    mean_observed: float | None
    count: int
    gap: float | None


class InningsModelReport(BaseModel):
    log_loss: float
    brier: float
    n_balls: int
    n_matches: int
    mean_wp_before: float
    calibration_max_abs_gap: float
    calibration_bins: list[CalibrationBin]


class WinProbabilityReportResponse(BaseModel):
    data_version: str
    innings: dict[str, InningsModelReport]
    calibration_max_abs_gap: float


class ConfigResponse(BaseModel):
    data_version: str
    config: dict[str, Any]


class Top10Entry(BaseModel):
    model_config = ConfigDict(extra="allow")

    rank: int
    player_id: int
    display_name: str
    role: Literal["batter", "bowler", "all_rounder"]
    role_label: Literal["batter", "bowler", "all_rounder"]
    selection_slot: str
    pressure_balls: int
    ipl_share: float
    key_metrics_vs_population: dict[str, Any]
    wpa_80_interval: list[float] | None
    bootstrap_inclusion_probability: float
    rank_stability: float
    grade: Literal["A", "B", "C"]


class Top10Response(BaseModel):
    data_version: str
    picks: list[Top10Entry]
    reserves: list[Top10Entry]
    grade_counts: dict[str, dict[str, int]]
    grade_thresholds: dict[str, dict[str, float | int]]
    eligibility: dict[str, Any]
    role_constraint: dict[str, Any]


class PlayerExplainResponse(BaseModel):
    player_id: int
    display_name: str
    status: Literal["pick", "reserve", "eligible_not_selected", "ineligible"]
    reasons: list[str]
    eligibility: dict[str, Any]
    metrics: dict[str, Any]
    inclusion_probability: float | None
    rank_stability: float | None
    rank_interval_80: list[int] | None
    role_ranks: dict[str, int | None]


class IndexWeights(BaseModel):
    wpa: float = Field(default=0.4, ge=0)
    runs_saved: float = Field(default=0.4, ge=0)
    consistency: float = Field(default=0.2, ge=0)

    @model_validator(mode="after")
    def positive_sum(self) -> "IndexWeights":
        if self.wpa + self.runs_saved + self.consistency <= 0:
            raise ValueError("At least one index weight must be positive")
        return self


class LeaderboardRequest(BaseModel):
    role: Literal["batter", "bowler", "all_rounder", "developing", "combined"] | None = None
    comp_group: Literal["IPL", "International", "Emerging"] | None = None
    min_pressure_balls: int = Field(default=0, ge=0)
    pressure_definition: Literal[
        "composite", "high_leverage", "hard_chase", "death_overs", "collapse", "tight_finish"
    ] = "composite"
    index_weights: IndexWeights | None = None


class LeaderboardRow(BaseModel):
    rank: int
    player_id: int
    player_name: str
    role: str
    comp_group: str | None
    score: float
    wpa_per_ball: float
    wpa_80_interval: list[float | None]
    pressure_balls: int
    rank_stability: float
    small_sample: bool
    small_sample_threshold: int
    pressure_definition: str


class CompareRequest(BaseModel):
    player_ids: list[int] = Field(min_length=2, max_length=4)

    @model_validator(mode="after")
    def unique_player_ids(self) -> "CompareRequest":
        if len(set(self.player_ids)) != len(self.player_ids):
            raise ValueError("player_ids must be unique")
        return self


class PlayerProfileResponse(BaseModel):
    player_id: int
    player_name: str
    role: str
    pressure_balls: int
    pressure_vs_normal: dict[str, Any]
    phase_splits: dict[str, Any]
    stroke_aggregates: list[dict[str, Any]]
    season_trend: list[dict[str, Any]]


class SimilarPlayer(BaseModel):
    player_id: int
    similar_player_id: int
    distance: float
    display_name: str
    role: str


class SensitivityPlayer(BaseModel):
    player_id: int
    inclusion_probability: float
    rank_stability: float
    rank_interval_80: list[int] | None


class SensitivityResponse(BaseModel):
    data_version: str
    bootstrap: dict[str, Any]
    players: list[SensitivityPlayer]
    scenario_count: int
    scenario_sweep: dict[str, Any]


class BacktestResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    data_version: str
    selected_predictor: dict[str, Any]
    folds: dict[str, Any]
    predictor_decision: dict[str, Any]
    selection_comparison: dict[str, Any]


class SquadFitResponse(BaseModel):
    templates: dict[str, str]
    rows: list[dict[str, Any]]
    swaps: list[dict[str, Any]]


class ShortlistEntry(BaseModel):
    player_id: int
    order: int = Field(ge=1, le=10)
    rationale: str = ""


class ShortlistInput(BaseModel):
    player_id: int
    order: int | None = Field(default=None, ge=1, le=10)
    rationale: str = ""


class ShortlistReloadResponse(BaseModel):
    status: Literal["reloaded"] = "reloaded"
    data_version: str
