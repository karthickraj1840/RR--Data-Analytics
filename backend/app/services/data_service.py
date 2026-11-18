from __future__ import annotations

import csv
import json
import math
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = ROOT / "backend"
PROCESSED = BACKEND_ROOT / "data" / "processed"
EXTERNAL = BACKEND_ROOT / "data" / "external"
SQUAD_FIT_PATH = PROCESSED / "squad_fit.csv"
SQUAD_FIT_SWAPS_PATH = PROCESSED / "squad_fit_swaps.csv"
SHORTLIST_PATH = BACKEND_ROOT / "data" / "shortlist.json"
PRESSURE_DEFINITIONS = (
    "high_leverage",
    "hard_chase",
    "death_overs",
    "collapse",
    "tight_finish",
)
PRESSURE_FLAGS = (*PRESSURE_DEFINITIONS, "knockout")
COMPOSITE_DEFINITION = "composite"
SMALL_SAMPLE_THRESHOLD = 100
SHRINKAGE_STRENGTH = 30.0
RR_TEMPLATE_COLUMNS = ("player_id", "role", "slot")
ATTRIBUTE_TEMPLATE_COLUMNS = (
    "player_id",
    "display_name",
    "role",
    "batting_hand",
    "bowling_style",
    "nationality",
    "notes",
)


class APIDataError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status_code: int = 500,
        code: str = "processed_data_error",
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code


@dataclass
class ProcessedSnapshot:
    quality: dict[str, Any]
    win_report: dict[str, Any]
    top10: dict[str, Any]
    sensitivity: dict[str, Any]
    backtest: dict[str, Any]
    config: dict[str, Any]
    metrics: pd.DataFrame
    seasons: pd.DataFrame
    strokes: pd.DataFrame
    group_usage: list[dict[str, Any]]
    season_usage: list[dict[str, Any]]
    pressure_stats: pd.DataFrame
    pressure_priors: dict[tuple[str, str], float]


_snapshot: ProcessedSnapshot | None = None
_snapshot_lock = threading.RLock()


def _read_json(name: str) -> dict[str, Any]:
    path = PROCESSED / name
    if not path.is_file():
        raise APIDataError(
            f"Required processed aggregate is missing: {name}",
            status_code=503,
            code="processed_data_missing",
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise APIDataError(
            f"Could not read processed aggregate: {name}",
            status_code=503,
            code="processed_data_invalid",
        ) from exc
    if not isinstance(value, dict):
        raise APIDataError(
            f"Processed aggregate has an invalid shape: {name}",
            status_code=503,
            code="processed_data_invalid",
        )
    return value


def _number(value: Any, default: float = 0.0) -> float:
    if value is None or value is pd.NA:
        return default
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return default
    return converted if math.isfinite(converted) else default


def _read_parquet(name: str, columns: list[str] | None = None) -> pd.DataFrame:
    path = PROCESSED / name
    if not path.is_file():
        raise APIDataError(
            f"Required processed aggregate is missing: {name}",
            status_code=503,
            code="processed_data_missing",
        )
    try:
        return pd.read_parquet(path, columns=columns)
    except (OSError, ValueError, ImportError) as exc:
        raise APIDataError(
            f"Could not read processed aggregate: {name}",
            status_code=503,
            code="processed_data_invalid",
        ) from exc


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, np.generic):
        return _clean(value.item())
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _read_config() -> dict[str, Any]:
    path = ROOT / "config.yaml"
    if not path.is_file():
        raise APIDataError(
            "Pipeline configuration is missing",
            status_code=503,
            code="config_missing",
        )
    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise APIDataError(
            "Pipeline configuration could not be read",
            status_code=503,
            code="config_invalid",
        ) from exc
    if not isinstance(config, dict):
        raise APIDataError(
            "Pipeline configuration must be a YAML object",
            status_code=503,
            code="config_invalid",
        )
    return config


def _usage_aggregates(
    matches: pd.DataFrame, pressure_rows: pd.DataFrame
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    match_metadata = matches[["match_id", "season", "comp_group"]].copy()
    match_metadata["season"] = match_metadata["season"].astype(str)
    match_metadata["comp_group"] = match_metadata["comp_group"].fillna(
        "not_provided"
    ).astype(str)
    delivery_metadata = pressure_rows[
        ["match_id", "comp_group", "is_legal"]
    ].merge(
        match_metadata[["match_id", "season"]],
        on="match_id",
        how="left",
        validate="many_to_one",
    )
    if delivery_metadata["season"].isna().any():
        raise APIDataError(
            "Some processed deliveries do not map to a season",
            status_code=503,
            code="processed_data_invalid",
        )
    by_group_matches = (
        match_metadata.groupby("comp_group", observed=True)["match_id"]
        .nunique()
        .to_dict()
    )
    by_group_balls = delivery_metadata.groupby(
        "comp_group", observed=True
    ).agg(balls=("is_legal", "size"), legal_balls=("is_legal", "sum"))
    groups = []
    for group in sorted(set(by_group_matches) | set(by_group_balls.index)):
        groups.append(
            {
                "comp_group": str(group),
                "matches": int(by_group_matches.get(group, 0)),
                "balls": int(by_group_balls.loc[group, "balls"])
                if group in by_group_balls.index
                else 0,
                "legal_balls": int(by_group_balls.loc[group, "legal_balls"])
                if group in by_group_balls.index
                else 0,
            }
        )
    by_season_matches = (
        match_metadata.groupby("season", observed=True)["match_id"]
        .nunique()
        .to_dict()
    )
    by_season_balls = delivery_metadata.groupby("season", observed=True).agg(
        balls=("is_legal", "size"), legal_balls=("is_legal", "sum")
    )
    seasons = []
    for season in sorted(set(by_season_matches) | set(by_season_balls.index)):
        seasons.append(
            {
                "season": str(season),
                "matches": int(by_season_matches.get(season, 0)),
                "balls": int(by_season_balls.loc[season, "balls"])
                if season in by_season_balls.index
                else 0,
                "legal_balls": int(by_season_balls.loc[season, "legal_balls"])
                if season in by_season_balls.index
                else 0,
            }
        )
    return groups, seasons


def _pressure_aggregates(
    pressure_rows: pd.DataFrame, league_weights: dict[str, float]
) -> tuple[pd.DataFrame, dict[tuple[str, str], float]]:
    definitions = {
        definition: pressure_rows[definition].fillna(False).astype(bool)
        for definition in PRESSURE_DEFINITIONS
    }
    definitions[COMPOSITE_DEFINITION] = pressure_rows[
        list(PRESSURE_DEFINITIONS)
    ].fillna(False).any(axis=1)
    legal = pressure_rows["is_legal"].fillna(False).astype(bool)
    frames = []
    priors: dict[tuple[str, str], float] = {}
    for definition, mask in definitions.items():
        selected = pressure_rows.loc[legal & mask]
        for role, player_column, sign in (
            ("batter", "striker_id", 1.0),
            ("bowler", "bowler_id", -1.0),
        ):
            part = selected.dropna(subset=[player_column]).copy()
            part["player_id"] = part[player_column].astype(int)
            part["role_wpa"] = part["wpa"].fillna(0).astype(float) * sign
            part["league_weight"] = (
                part["comp_group"].map(league_weights).fillna(0.0).astype(float)
            )
            part["weighted_wpa"] = part["role_wpa"] * part["league_weight"]
            summary = (
                part.groupby(["player_id", "comp_group"], observed=True)
                .agg(
                    pressure_balls=("role_wpa", "size"),
                    weighted_balls=("league_weight", "sum"),
                    weighted_wpa=("weighted_wpa", "sum"),
                )
                .reset_index()
            )
            summary["role"] = role
            summary["definition"] = definition
            frames.append(summary)
            total_balls = float(summary["weighted_balls"].sum())
            priors[(role, definition)] = (
                float(summary["weighted_wpa"].sum() / total_balls)
                if total_balls
                else 0.0
            )
    aggregate = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(
            columns=[
                "player_id",
                "comp_group",
                "pressure_balls",
                "weighted_wpa",
                "role",
                "definition",
            ]
        )
    )
    return aggregate, priors


def _load_snapshot() -> ProcessedSnapshot:
    quality = _read_json("data_quality.json")
    win_report = _read_json("win_probability_report.json")
    top10 = _read_json("top10.json")
    sensitivity = _read_json("sensitivity.json")
    backtest = _read_json("backtest.json")
    config = _read_config()
    metrics = _read_parquet("player_metrics.parquet")
    seasons = _read_parquet("player_season_metrics.parquet")
    strokes = _read_parquet("player_stroke_metrics.parquet")
    matches = _read_parquet(
        "matches.parquet", columns=["match_id", "season", "comp_group"]
    )
    pressure_columns = [
        "match_id",
        "comp_group",
        "striker_id",
        "bowler_id",
        "wpa",
        "is_legal",
        *PRESSURE_DEFINITIONS,
    ]
    pressure_rows = _read_parquet("pressure_flags.parquet", columns=pressure_columns)
    group_usage, season_usage = _usage_aggregates(matches, pressure_rows)
    pressure_stats, pressure_priors = _pressure_aggregates(
        pressure_rows,
        {
            str(group): float(weight)
            for group, weight in config["league_weights"].items()
        },
    )
    return ProcessedSnapshot(
        quality=quality,
        win_report=win_report,
        top10=top10,
        sensitivity=sensitivity,
        backtest=backtest,
        config=config,
        metrics=metrics,
        seasons=seasons,
        strokes=strokes,
        group_usage=group_usage,
        season_usage=season_usage,
        pressure_stats=pressure_stats,
        pressure_priors=pressure_priors,
    )


def get_snapshot() -> ProcessedSnapshot:
    global _snapshot
    with _snapshot_lock:
        if _snapshot is None:
            _snapshot = _load_snapshot()
        return _snapshot


def reload_processed_data() -> str:
    global _snapshot
    with _snapshot_lock:
        fresh = _load_snapshot()
        _snapshot = fresh
        return str(fresh.quality["data_version"])


def get_health() -> dict[str, Any]:
    snapshot = get_snapshot()
    return {
        "status": "ok",
        "data_version": snapshot.quality["data_version"],
        "matches_used": int(snapshot.quality["matches_used"]),
        "innings_log_loss": {
            innings: float(snapshot.win_report["by_innings"][innings]["log_loss"])
            for innings in ("1", "2")
        },
    }


def get_data_quality() -> dict[str, Any]:
    snapshot = get_snapshot()
    return {
        "data_version": snapshot.quality["data_version"],
        "exclusions": _clean(snapshot.quality["exclusions"]),
        "matches_used": int(snapshot.quality["matches_used"]),
        "balls_used": int(snapshot.quality["balls_used"]),
        "by_comp_group": snapshot.group_usage,
        "by_season": snapshot.season_usage,
        "flag_shares": _clean(snapshot.quality["pressure_flags"]),
    }


def get_win_report() -> dict[str, Any]:
    report = get_snapshot().win_report
    innings = {}
    for label, metrics in report["by_innings"].items():
        bins = []
        for item in metrics["calibration_10_bins"]:
            predicted = item["mean_predicted"]
            observed = item["mean_observed"]
            bins.append(
                {
                    "bin": int(item["bin"]),
                    "mean_predicted": predicted,
                    "mean_observed": observed,
                    "count": int(item["count"]),
                    "gap": (
                        abs(float(predicted) - float(observed))
                        if predicted is not None and observed is not None
                        else None
                    ),
                }
            )
        innings[label] = {
            "log_loss": float(metrics["log_loss"]),
            "brier": float(metrics["brier"]),
            "n_balls": int(metrics["n_balls"]),
            "n_matches": int(metrics["n_matches"]),
            "mean_wp_before": float(metrics["mean_wp_before"]),
            "calibration_max_abs_gap": float(metrics["calibration_max_abs_gap"]),
            "calibration_bins": bins,
        }
    return {
        "data_version": report["data_version"],
        "innings": innings,
        "calibration_max_abs_gap": float(report["calibration_max_abs_gap"]),
    }


def get_config() -> dict[str, Any]:
    snapshot = get_snapshot()
    return {
        "data_version": snapshot.quality["data_version"],
        "config": _clean(snapshot.config),
    }


def get_top10() -> dict[str, Any]:
    data = get_snapshot().top10
    return {
        "data_version": data["data_version"],
        "picks": _clean(data["combined_top10"]),
        "reserves": _clean(data["reserves"]),
        "grade_counts": data["grade_counts"],
        "grade_thresholds": data["confidence_grade_thresholds"],
        "eligibility": data["eligibility"],
        "role_constraint": data["top10_role_constraint"],
    }


def _player_metrics(snapshot: ProcessedSnapshot, player_id: int) -> pd.Series:
    rows = snapshot.metrics.loc[snapshot.metrics["player_id"].eq(player_id)]
    if rows.empty:
        raise APIDataError(
            f"Player {player_id} was not found in processed player aggregates",
            status_code=404,
            code="player_not_found",
        )
    return rows.iloc[0]


def explain_top10(player_id: int) -> dict[str, Any]:
    snapshot = get_snapshot()
    player = _player_metrics(snapshot, player_id)
    selection = snapshot.top10
    selected_items = selection["items"]
    eligibility = {
        "eligible_batter": bool(player["eligible_batter"]),
        "eligible_bowler": bool(player["eligible_bowler"]),
        "eligibility_min_pressure_balls": int(
            selection["eligibility"]["eligibility_min_pressure_balls"]
        ),
        "min_ipl_share": float(selection["eligibility"]["min_ipl_share"]),
        "player_ipl_share": float(player["ipl_ball_share"]),
        "batting_pressure_balls": int(_number(player["batting_pressure_balls"])),
        "bowling_pressure_balls": int(_number(player["bowling_pressure_balls"])),
    }
    item = next(
        (entry for entry in selected_items if int(entry["player_id"]) == player_id),
        None,
    )
    if item is not None and int(item["rank"]) <= 10:
        status = "pick"
        reasons = [
            f"Selected at combined rank {item['rank']} in the constrained Top 10.",
            f"Assigned to the {item['selection_slot']} slot.",
        ]
    elif item is not None and int(item["rank"]) <= 20:
        status = "reserve"
        reasons = [f"Ranked {item['rank']} and listed as a reserve."]
    else:
        batter_ok = bool(player["eligible_batter"])
        bowler_ok = bool(player["eligible_bowler"])
        status = "eligible_not_selected" if batter_ok or bowler_ok else "ineligible"
        reasons = []
        if not batter_ok:
            reasons.append(
                "Does not meet batter pressure-ball and/or minimum IPL-share thresholds."
            )
        if not bowler_ok:
            reasons.append(
                "Does not meet bowler pressure-ball and/or minimum IPL-share thresholds."
            )
        if batter_ok or bowler_ok:
            reasons.append("Eligible, but outside the selected Top 10 and reserve ranks.")
        if not reasons:
            reasons.append("Not selected in the current aggregate ranking.")
    sensitivity_player = snapshot.sensitivity["players"].get(str(player_id), {})
    role_ranks = {}
    for role, rank_key in (("batters", "batter_rank"), ("bowlers", "bowler_rank")):
        ranking = selection["role_rankings"][role]
        ranked_player = next(
            (entry for entry in ranking if int(entry["player_id"]) == player_id),
            None,
        )
        role_ranks[role] = (
            int(ranked_player["rank"]) if ranked_player is not None else None
        )
    return {
        "player_id": player_id,
        "display_name": str(player["player_name"]),
        "status": status,
        "reasons": reasons,
        "eligibility": _clean(eligibility),
        "metrics": _clean(player.to_dict()),
        "inclusion_probability": sensitivity_player.get("top10_inclusion_probability"),
        "rank_stability": sensitivity_player.get("rank_stability_score"),
        "rank_interval_80": sensitivity_player.get("rank_interval_80"),
        "role_ranks": role_ranks,
    }


def _role_value(player: pd.Series, role: str, field: str) -> float:
    if role == "batter":
        return _number(player.get(f"batting_{field}"))
    if role == "bowler":
        return _number(player.get(f"bowling_{field}"))
    bat = _number(player.get(f"batting_{field}"))
    bowl = _number(player.get(f"bowling_{field}"))
    bat_balls = _number(player.get("batting_pressure_balls"))
    bowl_balls = _number(player.get("bowling_pressure_balls"))
    total = bat_balls + bowl_balls
    return (bat * bat_balls + bowl * bowl_balls) / total if total else 0.0


def _pressure_rows_by_player(
    snapshot: ProcessedSnapshot,
    definition: str,
    comp_group: str | None,
) -> dict[tuple[int, str], dict[str, float]]:
    subset = snapshot.pressure_stats.loc[
        snapshot.pressure_stats["definition"].eq(definition)
    ]
    if comp_group is not None:
        subset = subset.loc[subset["comp_group"].eq(comp_group)]
    grouped = subset.groupby(["player_id", "role"], observed=True).agg(
        pressure_balls=("pressure_balls", "sum"),
        weighted_balls=("weighted_balls", "sum"),
        weighted_wpa=("weighted_wpa", "sum"),
    )
    result: dict[tuple[int, str], dict[str, float]] = {}
    for (player_id, role), values in grouped.iterrows():
        raw_count = float(values["pressure_balls"])
        weighted_count = float(values["weighted_balls"])
        raw_rate = (
            float(values["weighted_wpa"] / weighted_count)
            if weighted_count
            else 0.0
        )
        prior = snapshot.pressure_priors.get((str(role), definition), 0.0)
        shrunk_rate = (
            (weighted_count * raw_rate + SHRINKAGE_STRENGTH * prior)
            / (weighted_count + SHRINKAGE_STRENGTH)
            if weighted_count + SHRINKAGE_STRENGTH
            else prior
        )
        result[(int(player_id), str(role))] = {
            "pressure_balls": raw_count,
            "wpa_per_ball": shrunk_rate,
        }
    return result


def _definition_wpa(
    player: pd.Series,
    player_id: int,
    role: str,
    definition: str,
    comp_group: str | None,
    pressure_aggregate: dict[tuple[int, str], dict[str, float]],
) -> tuple[float, int]:
    roles = ("batter", "bowler") if role == "all_rounder" else (role,)
    weighted_wpa = 0.0
    total_balls = 0
    for current_role in roles:
        aggregate = pressure_aggregate.get((player_id, current_role))
        if comp_group is not None:
            rate = aggregate["wpa_per_ball"] if aggregate else 0.0
            balls = int(aggregate["pressure_balls"]) if aggregate else 0
        elif aggregate is not None:
            rate = aggregate["wpa_per_ball"]
            balls = int(aggregate["pressure_balls"])
        else:
            rate = 0.0
            balls = 0
        weighted_wpa += rate * balls
        total_balls += balls
    return (
        weighted_wpa / total_balls if total_balls else 0.0,
        total_balls,
    )


def _zscore(values: pd.Series) -> pd.Series:
    std = float(values.std(ddof=0)) if len(values) > 1 else 0.0
    if not std or not math.isfinite(std):
        return pd.Series(0.0, index=values.index)
    return (values - float(values.mean())) / std


def get_leaderboard(
    request: dict[str, Any],
) -> list[dict[str, Any]]:
    snapshot = get_snapshot()
    role_filter = request.get("role")
    comp_group = request.get("comp_group")
    definition = request.get("pressure_definition", COMPOSITE_DEFINITION)
    minimum = int(request.get("min_pressure_balls", 0))
    default_weights = snapshot.config["index_weights"]
    weights = request.get("index_weights") or default_weights
    values = {
        "wpa": float(weights["wpa"]),
        "runs_saved": float(weights["runs_saved"]),
        "consistency": float(weights["consistency"]),
    }
    total_weight = sum(values.values())
    if total_weight <= 0:
        raise APIDataError(
            "Index weights must have a positive sum",
            status_code=422,
            code="invalid_index_weights",
        )
    pressure_aggregate = _pressure_rows_by_player(snapshot, definition, comp_group)

    candidates: list[dict[str, Any]] = []
    for _, player in snapshot.metrics.iterrows():
        player_role = str(player["role"])
        target_role = None if role_filter == "combined" else role_filter
        if target_role == "batter" and player_role not in ("batter", "all_rounder"):
            continue
        if target_role == "bowler" and player_role not in ("bowler", "all_rounder"):
            continue
        if target_role == "all_rounder" and player_role != "all_rounder":
            continue
        if target_role == "developing" and player_role != "developing":
            continue
        if target_role == "developing":
            target_role = _leaderboard_role_for_developing(player.get("primary_role"))
        if target_role is None:
            target_role = (
                _leaderboard_role_for_developing(player.get("primary_role"))
                if player_role == "developing"
                else player_role
            )
        role = target_role
        player_id = int(player["player_id"])
        wpa_rate, pressure_balls = _definition_wpa(
            player, player_id, role, definition, comp_group, pressure_aggregate
        )
        if comp_group is not None and pressure_balls == 0:
            continue
        if pressure_balls < minimum:
            continue
        candidates.append(
            {
                "player_id": player_id,
                "player_name": str(player["player_name"]),
                "role": player_role,
                "score_wpa": wpa_rate,
                "wpa_per_ball": wpa_rate,
                "pressure_balls": pressure_balls,
                "runs_saved": _role_value(player, role, "runs_saved_per_ball"),
                "consistency": _role_value(player, role, "consistency"),
                "rank_stability": _number(
                    snapshot.sensitivity["players"]
                    .get(str(player_id), {})
                    .get("rank_stability_score")
                ),
                "wpa_80_interval": [
                    (
                        (
                            _number(player.get("batting_wpa_ci80_lower"))
                            + _number(player.get("bowling_wpa_ci80_lower"))
                        )
                        / 2.0
                        if role == "all_rounder"
                        else player.get(
                            "batting_wpa_ci80_lower"
                            if role == "batter"
                            else "bowling_wpa_ci80_lower"
                        )
                    ),
                    (
                        (
                            _number(player.get("batting_wpa_ci80_upper"))
                            + _number(player.get("bowling_wpa_ci80_upper"))
                        )
                        / 2.0
                        if role == "all_rounder"
                        else player.get(
                            "batting_wpa_ci80_upper"
                            if role == "batter"
                            else "bowling_wpa_ci80_upper"
                        )
                    ),
                ],
                "small_sample": pressure_balls < SMALL_SAMPLE_THRESHOLD,
                "small_sample_threshold": SMALL_SAMPLE_THRESHOLD,
            }
        )
    if not candidates:
        return []
    frame = pd.DataFrame(candidates)
    score = (
        values["wpa"] * _zscore(frame["score_wpa"])
        + values["runs_saved"] * _zscore(frame["runs_saved"])
        + values["consistency"] * _zscore(frame["consistency"])
    ) / total_weight
    frame["score"] = score
    frame = frame.sort_values(
        ["score", "pressure_balls", "player_id"],
        ascending=[False, False, True],
        kind="stable",
    )
    rows = []
    for rank, (_, row) in enumerate(frame.iterrows(), start=1):
        rows.append(
            _clean(
                {
                    "rank": rank,
                    "player_id": int(row["player_id"]),
                    "player_name": row["player_name"],
                    "role": row["role"],
                    "comp_group": comp_group,
                    "score": float(row["score"]),
                    "wpa_per_ball": float(row["wpa_per_ball"]),
                    "wpa_80_interval": row["wpa_80_interval"],
                    "pressure_balls": int(row["pressure_balls"]),
                    "rank_stability": float(row["rank_stability"]),
                    "small_sample": bool(row["small_sample"]),
                    "small_sample_threshold": int(row["small_sample_threshold"]),
                    "pressure_definition": definition,
                }
            )
        )
    return rows


def _leaderboard_role_for_developing(primary_role: Any) -> str:
    role = str(primary_role or "").lower()
    if "batter" in role or "batting" in role:
        return "batter"
    if "bowler" in role or "bowling" in role:
        return "bowler"
    return "all_rounder"


def get_player_profile(player_id: int) -> dict[str, Any]:
    snapshot = get_snapshot()
    player = _player_metrics(snapshot, player_id)
    role = str(player["role"])
    seasons = snapshot.seasons.loc[
        snapshot.seasons["player_id"].eq(player_id)
    ].sort_values(["season", "role", "comp_group"], kind="stable")
    strokes = snapshot.strokes.loc[
        snapshot.strokes["player_id"].eq(player_id)
    ].sort_values(["legal_balls", "stroke"], ascending=[False, True], kind="stable")
    pressure_vs_normal = {
        "batting_pressure_wpa_per_ball": player.get(
            "batting_pressure_wpa_per_ball_shrunk_30"
        ),
        "batting_normal_wpa_per_ball": player.get("batting_normal_wpa_per_ball"),
        "batting_pressure_vs_normal_delta": player.get(
            "batting_wpa_pressure_vs_normal_delta"
        ),
        "batting_pressure_strike_rate": player.get("batting_strike_rate"),
        "batting_normal_strike_rate": player.get("batting_normal_strike_rate"),
        "batting_pressure_vs_normal_strike_rate_delta": player.get(
            "batting_pressure_vs_normal_strike_rate_delta"
        ),
        "bowling_pressure_wpa_per_ball": player.get(
            "bowling_pressure_wpa_per_ball_shrunk_30"
        ),
        "bowling_normal_wpa_per_ball": player.get("bowling_normal_wpa_per_ball"),
        "bowling_pressure_vs_normal_delta": player.get(
            "bowling_wpa_pressure_vs_normal_delta"
        ),
        "bowling_pressure_economy": player.get("bowling_economy"),
        "bowling_normal_economy": player.get("bowling_normal_economy"),
        "bowling_pressure_vs_normal_economy_delta": player.get(
            "bowling_pressure_vs_normal_economy_delta"
        ),
    }
    phase_splits = {
        "batting_share": {
            "powerplay": player.get("batting_phase_share_pp"),
            "middle": player.get("batting_phase_share_mid"),
            "death": player.get("batting_phase_share_death"),
        },
        "bowling": {
            "powerplay": {
                "legal_balls": player.get("bowling_legal_balls_pp"),
                "economy": player.get("bowling_economy_pp"),
                "wickets_per_24_balls": player.get(
                    "bowling_wickets_per_24_legal_balls_pp"
                ),
            },
            "middle": {
                "legal_balls": player.get("bowling_legal_balls_mid"),
                "economy": player.get("bowling_economy_mid"),
                "wickets_per_24_balls": player.get(
                    "bowling_wickets_per_24_legal_balls_mid"
                ),
            },
            "death": {
                "legal_balls": player.get("bowling_legal_balls_death"),
                "economy": player.get("bowling_economy_death"),
                "wickets_per_24_balls": player.get(
                    "bowling_wickets_per_24_legal_balls_death"
                ),
            },
        },
    }
    return _clean(
        {
            "player_id": player_id,
            "player_name": str(player["player_name"]),
            "role": role,
            "pressure_balls": int(_number(player.get("pressure_balls"))),
            "pressure_vs_normal": pressure_vs_normal,
            "phase_splits": phase_splits,
            "stroke_aggregates": strokes.to_dict("records"),
            "season_trend": seasons.to_dict("records"),
        }
    )


def get_similar_players(player_id: int, limit: int = 10) -> list[dict[str, Any]]:
    snapshot = get_snapshot()
    _player_metrics(snapshot, player_id)
    feature_columns = [
        "pressure_index",
        "wpa_per_ball",
        "pressure_delta",
        "sample_size",
    ]
    features = snapshot.metrics[feature_columns].astype(float).fillna(0.0)
    scale = features.std(ddof=0).replace(0, 1.0)
    scaled = (features - features.mean()) / scale
    player_positions = np.flatnonzero(
        snapshot.metrics["player_id"].to_numpy() == player_id
    )
    position = int(player_positions[0])
    distances = np.linalg.norm(
        scaled.to_numpy() - scaled.iloc[position].to_numpy(),
        axis=1,
    )
    distances[position] = np.inf
    nearest = np.argsort(distances, kind="stable")[:limit]
    return [
        {
            "player_id": player_id,
            "similar_player_id": int(snapshot.metrics.iloc[index]["player_id"]),
            "distance": float(distances[index]),
            "display_name": str(snapshot.metrics.iloc[index]["player_name"]),
            "role": str(snapshot.metrics.iloc[index]["role"]),
        }
        for index in nearest
    ]


def compare_players(player_ids: list[int]) -> list[dict[str, Any]]:
    snapshot = get_snapshot()
    missing = [
        player_id
        for player_id in player_ids
        if not snapshot.metrics["player_id"].eq(player_id).any()
    ]
    if missing:
        raise APIDataError(
            f"Unknown player IDs: {missing}",
            status_code=404,
            code="player_not_found",
        )
    return [get_player_profile(player_id) for player_id in player_ids]


def get_sensitivity() -> dict[str, Any]:
    snapshot = get_snapshot()
    players = []
    for player_id, values in snapshot.sensitivity["players"].items():
        players.append(
            {
                "player_id": int(player_id),
                "inclusion_probability": float(
                    values["top10_inclusion_probability"]
                ),
                "rank_stability": float(values["rank_stability_score"]),
                "rank_interval_80": values.get("rank_interval_80"),
            }
        )
    players.sort(
        key=lambda item: (-item["inclusion_probability"], item["player_id"])
    )
    return _clean(
        {
            "data_version": snapshot.sensitivity["data_version"],
            "bootstrap": snapshot.sensitivity["bootstrap"],
            "players": players,
            "scenario_count": int(snapshot.sensitivity["scenario_count"]),
            "scenario_sweep": snapshot.sensitivity["scenario_sweep"],
        }
    )


def get_backtest() -> dict[str, Any]:
    return _clean(get_snapshot().backtest)


def _ensure_external_templates() -> tuple[Path, Path]:
    EXTERNAL.mkdir(parents=True, exist_ok=True)
    squad_path = EXTERNAL / "rr_squad_template.csv"
    attributes_path = EXTERNAL / "player_attributes_template.csv"
    for path, columns in (
        (squad_path, RR_TEMPLATE_COLUMNS),
        (attributes_path, ATTRIBUTE_TEMPLATE_COLUMNS),
    ):
        if not path.exists():
            with path.open("w", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerow(columns)
    return squad_path, attributes_path


def _csv_rows(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            return list(csv.DictReader(handle))
    except (OSError, csv.Error) as exc:
        raise APIDataError(
            f"Could not read squad-fit input: {path.name}",
            status_code=503,
            code="external_data_invalid",
        ) from exc


def get_squad_fit() -> dict[str, Any]:
    if not SQUAD_FIT_PATH.is_file() or not SQUAD_FIT_SWAPS_PATH.is_file():
        raise APIDataError(
            "Squad-fit output is missing; run scripts/squad_fit.py",
            status_code=503,
            code="squad_fit_missing",
        )

    def supplied_rows(path: Path) -> list[dict[str, str]]:
        return [
            {
                key: value.strip() if value and value.strip() else "not_provided"
                for key, value in row.items()
            }
            for row in _csv_rows(path)
        ]

    return {
        "templates": {
            "squad_fit": SQUAD_FIT_PATH.name,
            "swaps": SQUAD_FIT_SWAPS_PATH.name,
        },
        "rows": supplied_rows(SQUAD_FIT_PATH),
        "swaps": supplied_rows(SQUAD_FIT_SWAPS_PATH),
    }


def get_shortlist() -> list[dict[str, Any]]:
    if not SHORTLIST_PATH.exists():
        return []
    try:
        raw = json.loads(SHORTLIST_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise APIDataError(
            "Local shortlist JSON is invalid",
            status_code=503,
            code="shortlist_invalid",
        ) from exc
    if not isinstance(raw, list):
        raise APIDataError(
            "Local shortlist must contain a JSON array",
            status_code=503,
            code="shortlist_invalid",
        )
    normalized = []
    for index, value in enumerate(raw, start=1):
        if not isinstance(value, dict):
            raise APIDataError(
                "Each shortlist entry must be an object",
                status_code=503,
                code="shortlist_invalid",
            )
        try:
            normalized.append(
                {
                    "player_id": int(value["player_id"]),
                    "order": int(value.get("order", index)),
                    "rationale": str(value.get("rationale", "")),
                }
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise APIDataError(
                "Local shortlist entries must include a numeric player_id and order",
                status_code=503,
                code="shortlist_invalid",
            ) from exc
    normalized.sort(key=lambda item: item["order"])
    if len(normalized) > 10:
        raise APIDataError(
            "Local shortlist cannot contain more than 10 players",
            status_code=503,
            code="shortlist_invalid",
        )
    if len({item["player_id"] for item in normalized}) != len(normalized):
        raise APIDataError(
            "Local shortlist contains duplicate player IDs",
            status_code=503,
            code="shortlist_invalid",
        )
    if len({item["order"] for item in normalized}) != len(normalized):
        raise APIDataError(
            "Local shortlist contains duplicate order values",
            status_code=503,
            code="shortlist_invalid",
        )
    return normalized


def save_shortlist(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(entries) > 10:
        raise APIDataError(
            "A shortlist can contain at most 10 players",
            status_code=422,
            code="shortlist_limit_exceeded",
        )
    normalized = []
    seen_ids = set()
    seen_orders = set()
    for index, entry in enumerate(entries, start=1):
        player_id = int(entry["player_id"])
        order = int(entry.get("order") or index)
        if order < 1 or order > 10:
            raise APIDataError(
                "Shortlist order must be between 1 and 10",
                status_code=422,
                code="invalid_shortlist_order",
            )
        if not get_snapshot().metrics["player_id"].eq(player_id).any():
            raise APIDataError(
                f"Player {player_id} was not found in processed player aggregates",
                status_code=404,
                code="player_not_found",
            )
        if player_id in seen_ids:
            raise APIDataError(
                "Shortlist player IDs must be unique",
                status_code=422,
                code="duplicate_shortlist_player",
            )
        if order in seen_orders:
            raise APIDataError(
                "Shortlist order values must be unique",
                status_code=422,
                code="duplicate_shortlist_order",
            )
        seen_ids.add(player_id)
        seen_orders.add(order)
        normalized.append(
            {
                "player_id": player_id,
                "order": order,
                "rationale": str(entry.get("rationale", "")),
            }
        )
    normalized.sort(key=lambda item: item["order"])
    SHORTLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = SHORTLIST_PATH.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary_path, SHORTLIST_PATH)
    return normalized
