from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "backend" / "data" / "processed"
CONFIG_PATH = ROOT / "config.yaml"
PRESSURE_FLAGS = [
    "high_leverage",
    "hard_chase",
    "death_overs",
    "collapse",
    "tight_finish",
]
BOOTSTRAP_RESAMPLES = 300
BOOTSTRAP_SEED = 20261116


def _zscore(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    selected = values[valid & np.isfinite(values)]
    mean = float(selected.mean()) if len(selected) else 0.0
    scale = float(selected.std()) if len(selected) else 0.0
    if scale == 0:
        return np.zeros(len(values), dtype=float)
    return np.nan_to_num((values - mean) / scale, nan=0.0, posinf=0.0, neginf=0.0)


def _read_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}


def _role_match_contributions(
    balls: pd.DataFrame, pressure_column: str
) -> pd.DataFrame:
    pressure = balls.loc[balls["is_legal"] & balls[pressure_column].astype(bool)]
    parts = []
    for role, player_column, sign in (
        ("batting", "striker_id", 1.0),
        ("bowling", "bowler_id", -1.0),
    ):
        selected = pressure.dropna(subset=[player_column]).copy()
        selected["player_id"] = selected[player_column].astype(int)
        selected["role_wpa"] = selected["wpa"] * sign
        selected["n_balls"] = 1
        grouped = (
            selected.groupby(
                ["player_id", "match_id", "comp_group"], observed=True
            )
            .agg(n_balls=("n_balls", "sum"), wpa_sum=("role_wpa", "sum"))
            .reset_index()
        )
        grouped["role"] = role
        parts.append(grouped)
    return pd.concat(parts, ignore_index=True)


def _wide_match_matrix(
    contributions: pd.DataFrame,
    matches: pd.DataFrame,
    player_ids: pd.Index,
    value_column: str,
    comp_group: str | None = None,
) -> np.ndarray:
    selected = contributions
    if comp_group is not None:
        selected = selected.loc[selected["comp_group"].eq(comp_group)]
    table = selected.pivot_table(
        index="match_id",
        columns="player_id",
        values=value_column,
        aggfunc="sum",
        fill_value=0,
    )
    return table.reindex(
        index=matches["match_id"], columns=player_ids, fill_value=0
    ).to_numpy(dtype=float)


def _weighted_score(
    player_metrics: pd.DataFrame,
    wpa_per_ball: np.ndarray,
    runs_saved_per_ball: np.ndarray,
    consistency: np.ndarray,
    index_weights: dict[str, float],
    valid: np.ndarray,
) -> np.ndarray:
    components = {
        "wpa": wpa_per_ball,
        "runs_saved": runs_saved_per_ball,
        "consistency": consistency,
    }
    score = np.zeros(len(player_metrics), dtype=float)
    weight_total = sum(index_weights.values())
    for key, values in components.items():
        score += index_weights[key] * _zscore(values, valid)
    return score / weight_total


def _top_ids(scores: np.ndarray, player_ids: np.ndarray, valid: np.ndarray) -> list[int]:
    indices = np.flatnonzero(valid & np.isfinite(scores))
    ordered = sorted(indices, key=lambda index: (-scores[index], int(player_ids[index])))
    return [int(player_ids[index]) for index in ordered[:10]]


def _scenario_sweep(
    metrics: pd.DataFrame,
    balls: pd.DataFrame,
    matches: pd.DataFrame,
    match_ball: dict[str, np.ndarray],
    match_wpa: dict[str, np.ndarray],
    player_ids: np.ndarray,
    config: dict[str, Any],
) -> dict[str, list[int]]:
    pressure_candidates: dict[str, list[int]] = {}
    player_index = pd.Index(player_ids)
    base_eligible = metrics["batting_qualifies"].to_numpy(bool) | metrics[
        "bowling_qualifies"
    ].to_numpy(bool)
    min_balls = int(config.get("min_balls", 30))
    for flag in PRESSURE_FLAGS:
        column = f"pri_{flag}"
        if column not in metrics:
            continue
        scores = metrics[column].to_numpy(dtype=float)
        flag_rows = balls.loc[balls["is_legal"] & balls[flag].astype(bool)]
        flag_counts = pd.concat(
            [
                flag_rows.groupby("striker_id").size(),
                flag_rows.groupby("bowler_id").size(),
            ]
        ).groupby(level=0).sum()
        counts = flag_counts.reindex(player_index, fill_value=0).to_numpy()
        valid = base_eligible & (counts >= min_balls)
        pressure_candidates[f"pressure:{flag}"] = _top_ids(scores, player_ids, valid)

    pressure_candidates["pressure:composite"] = _top_ids(
        metrics["pri_composite"].to_numpy(dtype=float),
        player_ids,
        base_eligible & (metrics["pressure_balls"].to_numpy() >= min_balls),
    )

    current_weights = {
        str(key): float(value)
        for key, value in config["league_weights"].items()
    }
    league_scenarios = {
        "configured": current_weights,
        "equal": {group: 1.0 for group in current_weights},
        "ipl_emphasis": {
            group: (1.2 if group == "IPL" else 0.8 if group == "International" else 0.4)
            for group in current_weights
        },
        "international_emphasis": {
            group: (0.8 if group == "IPL" else 1.0 if group == "International" else 0.4)
            for group in current_weights
        },
    }
    runs = metrics["runs_above_situation_average"].fillna(0).to_numpy(dtype=float)
    consistency = metrics["rank_stability"].fillna(0.5).to_numpy(dtype=float)
    default_index = {
        key: float(value) for key, value in config["index_weights"].items()
    }
    base_pressure_balls = metrics["pressure_balls"].to_numpy(dtype=float)
    for scenario_name, league_weights in league_scenarios.items():
        weighted_balls = np.zeros(len(player_ids), dtype=float)
        weighted_wpa = np.zeros(len(player_ids), dtype=float)
        for group, weight in league_weights.items():
            weighted_balls += match_ball[group].sum(axis=0) * weight
            weighted_wpa += match_wpa[group].sum(axis=0) * weight
        rates = np.divide(
            weighted_wpa,
            weighted_balls,
            out=np.zeros(len(player_ids), dtype=float),
            where=weighted_balls > 0,
        )
        score = _weighted_score(
            metrics, rates, runs, consistency, default_index, base_eligible
        )
        pressure_candidates[f"league:{scenario_name}"] = _top_ids(
            score, player_ids, base_eligible & (base_pressure_balls >= min_balls)
        )

    index_scenarios = {
        "configured": default_index,
        "wpa_heavy": {"wpa": 0.6, "runs_saved": 0.2, "consistency": 0.2},
        "runs_saved_heavy": {"wpa": 0.2, "runs_saved": 0.6, "consistency": 0.2},
        "consistency_heavy": {"wpa": 0.2, "runs_saved": 0.2, "consistency": 0.6},
    }
    rates = metrics["wpa_per_ball"].fillna(0).to_numpy(dtype=float)
    for scenario_name, weights in index_scenarios.items():
        score = _weighted_score(metrics, rates, runs, consistency, weights, base_eligible)
        pressure_candidates[f"index:{scenario_name}"] = _top_ids(
            score, player_ids, base_eligible & (base_pressure_balls >= min_balls)
        )

    for threshold in (30, 60, 80, 100):
        score = _weighted_score(
            metrics, rates, runs, consistency, default_index, base_eligible
        )
        pressure_candidates[f"min_balls:{threshold}"] = _top_ids(
            score, player_ids, base_eligible & (base_pressure_balls >= threshold)
        )
    return pressure_candidates


def build_sensitivity(data_version: str) -> dict:
    config = _read_config()
    metrics = pd.read_parquet(PROCESSED / "player_metrics.parquet").sort_values(
        "player_id"
    )
    balls = pd.read_parquet(PROCESSED / "pressure_flags.parquet")
    matches = pd.read_parquet(PROCESSED / "matches.parquet").sort_values(
        "match_id"
    )
    player_ids = metrics["player_id"].to_numpy(dtype=int)
    player_index = pd.Index(player_ids)
    composite = balls[PRESSURE_FLAGS].any(axis=1)
    balls = balls.copy()
    balls["composite_pressure"] = composite
    contributions = _role_match_contributions(balls, "composite_pressure")
    match_ids = matches["match_id"].to_numpy()
    group_names = sorted(matches["comp_group"].dropna().unique())
    match_ball: dict[str, np.ndarray] = {}
    match_wpa: dict[str, np.ndarray] = {}
    for group in group_names:
        match_ball[group] = _wide_match_matrix(
            contributions, matches, player_index, "n_balls", group
        )
        match_wpa[group] = _wide_match_matrix(
            contributions, matches, player_index, "wpa_sum", group
        )

    current_league = config["league_weights"]
    total_ball = np.zeros(len(player_ids), dtype=float)
    total_wpa = np.zeros(len(player_ids), dtype=float)
    for group in group_names:
        weight = float(current_league.get(group, 0))
        total_ball += match_ball[group].sum(axis=0) * weight
        total_wpa += match_wpa[group].sum(axis=0) * weight
    bootstrap_rate = np.divide(
        total_wpa,
        total_ball,
        out=np.zeros(len(player_ids), dtype=float),
        where=total_ball > 0,
    )
    runs = metrics["runs_above_situation_average"].fillna(0).to_numpy(dtype=float)
    consistency = metrics["rank_stability"].fillna(0.5).to_numpy(dtype=float)
    weights = {
        str(key): float(value) for key, value in config["index_weights"].items()
    }
    base_eligible = metrics["batting_qualifies"].to_numpy(bool) | metrics[
        "bowling_qualifies"
    ].to_numpy(bool)
    base_minimum = int(config.get("min_balls", 30))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    top_counts = np.zeros(len(player_ids), dtype=int)
    rank_samples: list[list[int]] = [[] for _ in player_ids]
    for _ in range(BOOTSTRAP_RESAMPLES):
        sampled_matches = rng.choice(match_ids, size=len(match_ids), replace=True)
        multiplicity = pd.Series(sampled_matches).value_counts()
        draw_counts = matches["match_id"].map(multiplicity).fillna(0).to_numpy()
        sampled_balls = np.zeros(len(player_ids), dtype=float)
        sampled_eligibility_balls = np.zeros(len(player_ids), dtype=float)
        sampled_wpa = np.zeros(len(player_ids), dtype=float)
        for group in group_names:
            weights_for_group = float(current_league.get(group, 0))
            group_balls = draw_counts @ match_ball[group]
            sampled_balls += group_balls * weights_for_group
            sampled_eligibility_balls += group_balls
            sampled_wpa += (
                draw_counts @ match_wpa[group]
            ) * weights_for_group
        sampled_rate = np.divide(
            sampled_wpa,
            sampled_balls,
            out=np.zeros(len(player_ids), dtype=float),
            where=sampled_balls > 0,
        )
        score = _weighted_score(
            metrics, sampled_rate, runs, consistency, weights, base_eligible
        )
        valid = base_eligible & (sampled_eligibility_balls >= base_minimum)
        top10 = _top_ids(score, player_ids, valid)
        top_set = set(top10)
        for index, player_id in enumerate(player_ids):
            rank_samples[index].append(
                (int(np.sum(valid & (score > score[index]))) + 1)
                if valid[index]
                else int(valid.sum()) + 1
            )
            if int(player_id) in top_set:
                top_counts[index] += 1

    inclusion = top_counts / BOOTSTRAP_RESAMPLES
    scenarios = _scenario_sweep(
        metrics,
        balls,
        matches,
        match_ball,
        match_wpa,
        player_ids,
        config,
    )
    scenario_counts = np.zeros(len(player_ids), dtype=int)
    for top_ids in scenarios.values():
        top_ids_set = set(top_ids)
        scenario_counts += np.array(
            [int(player_id) in top_ids_set for player_id in player_ids], dtype=int
        )
    scenario_share = scenario_counts / max(len(scenarios), 1)
    stability = 0.5 * inclusion + 0.5 * scenario_share

    ranks = np.asarray(rank_samples, dtype=float)
    rank_lower = np.quantile(ranks, 0.10, axis=1).astype(int)
    rank_upper = np.quantile(ranks, 0.90, axis=1).astype(int)
    player_id_strings = [str(int(player_id)) for player_id in player_ids]
    similar_features = [
        column
        for column in [
            "wpa_per_ball",
            "runs_above_situation_average",
            "pressure_index",
            "strike_rate",
            "economy",
            "ipl_ball_share",
        ]
        if column in metrics
    ]
    features = metrics[similar_features].replace([np.inf, -np.inf], np.nan)
    features = features.fillna(features.median()).to_numpy(dtype=float)
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale == 0] = 1
    standardized = (features - mean) / scale
    differences = standardized[:, None, :] - standardized[None, :, :]
    distances = np.sqrt(np.einsum("ijk,ijk->ij", differences, differences))
    nearest = {}
    for index, player_id in enumerate(player_ids):
        distances[index, index] = np.inf
        neighbor_indices = np.argsort(distances[index])[:5]
        nearest[str(int(player_id))] = [
            {
                "similar_player_id": int(player_ids[neighbor]),
                "distance": float(distances[index, neighbor]),
            }
            for neighbor in neighbor_indices
        ]

    score = _weighted_score(
        metrics,
        bootstrap_rate,
        runs,
        consistency,
        weights,
        base_eligible,
    )
    report = {
        "data_version": data_version,
        "bootstrap": {
            "unit": "match",
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
            "base_min_pressure_balls": base_minimum,
        },
        "players": {
            player_id_strings[index]: {
                "top10_inclusion_probability": float(inclusion[index]),
                "rank_interval_80": [
                    int(rank_lower[index]),
                    int(rank_upper[index]),
                ],
                "rank_stability_score": float(stability[index]),
                "scenario_top10_share": float(scenario_share[index]),
            }
            for index in range(len(player_ids))
        },
        "scenario_top10": scenarios,
        "scenario_count": len(scenarios),
        "scenario_sweep": {
            "pressure_definitions": ["composite", *PRESSURE_FLAGS],
            "league_weight_sets": ["configured", "equal", "ipl_emphasis", "international_emphasis"],
            "index_weight_sets": ["configured", "wpa_heavy", "runs_saved_heavy", "consistency_heavy"],
            "minimum_pressure_ball_thresholds": [30, 60, 80, 100],
            "rank_stability_definition": (
                "0.5 * match-bootstrap top-10 inclusion probability + "
                "0.5 * share of one-factor sensitivity scenarios in the top 10"
            ),
        },
        "similar_players": nearest,
        "score_definition": {
            "bootstrap_score": "league-weighted pressure WPA/ball + configured runs-saved and consistency z-scores",
            "rank_stability_score": "0.5 * bootstrap top-10 inclusion probability + 0.5 * scenario top-10 share",
        },
    }
    (PROCESSED / "sensitivity.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report
