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
GRADE_THRESHOLDS = {
    "A": {
        "minimum_inclusion_probability": 0.70,
        "minimum_rank_stability": 0.70,
        "minimum_pressure_balls": 200,
    },
    "B": {
        "minimum_inclusion_probability": 0.35,
        "minimum_rank_stability": 0.40,
        "minimum_pressure_balls": 100,
    },
}
POPULATION_METRICS = [
    "pressure_balls",
    "selected_predictor_wpa_per_ball",
    "runs_above_situation_average",
    "strike_rate",
    "economy",
    "ipl_ball_share",
]


def _zscore(values: pd.Series, valid: pd.Series) -> pd.Series:
    clean = values.astype(float)
    selected = clean.loc[valid & clean.notna()]
    scale = float(selected.std(ddof=0)) if len(selected) > 1 else 0.0
    if not scale:
        return pd.Series(0.0, index=clean.index)
    return (clean - float(selected.mean())) / scale


def _confidence_grade(
    inclusion_probability: float, rank_stability: float, pressure_balls: int
) -> str:
    for grade in ("A", "B"):
        thresholds = GRADE_THRESHOLDS[grade]
        if (
            inclusion_probability >= thresholds["minimum_inclusion_probability"]
            and rank_stability >= thresholds["minimum_rank_stability"]
            and pressure_balls >= thresholds["minimum_pressure_balls"]
        ):
            return grade
    return "C"


def _choose_top_ten(
    candidates: pd.DataFrame, batters_required: int, bowlers_required: int
) -> list[tuple[int, str]]:
    if batters_required + bowlers_required != 10:
        raise ValueError("Configured batter and bowler split must sum to ten")
    states: dict[tuple[int, int], tuple[float, list[tuple[int, str]]]] = {
        (0, 0): (0.0, [])
    }
    for row in candidates.itertuples(index=False):
        player_id = int(row.player_id)
        next_states = dict(states)
        for (picked, batters), (score, selected) in states.items():
            if picked >= 10:
                continue
            bowlers = picked - batters
            if bool(row.eligible_batter) and batters < batters_required:
                key = (picked + 1, batters + 1)
                value = (
                    score + float(row.batter_final_score),
                    selected + [(player_id, "batter")],
                )
                previous = next_states.get(key)
                if previous is None or value[0] > previous[0]:
                    next_states[key] = value
            if bool(row.eligible_bowler) and bowlers < bowlers_required:
                key = (picked + 1, batters)
                value = (
                    score + float(row.bowler_final_score),
                    selected + [(player_id, "bowler")],
                )
                previous = next_states.get(key)
                if previous is None or value[0] > previous[0]:
                    next_states[key] = value
        states = next_states
    result = states.get((10, batters_required))
    return result[1] if result is not None else []


def _role_rankings(metrics: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    rankings: dict[str, list[dict[str, Any]]] = {}
    for role, eligible_column, score_column in (
        ("batters", "eligible_batter", "batter_final_score"),
        ("bowlers", "eligible_bowler", "bowler_final_score"),
    ):
        ordered = metrics.loc[metrics[eligible_column]].sort_values(
            score_column, ascending=False, kind="stable"
        )
        rankings[role] = [
            {
                "rank": rank,
                "player_id": int(row.player_id),
                "display_name": str(row.player_name),
                "score": float(getattr(row, score_column)),
                "pressure_balls": int(row.pressure_balls),
                "confidence_grade": str(row.confidence_grade),
            }
            for rank, row in enumerate(ordered.itertuples(index=False), start=1)
        ]
    return rankings


def _metric_comparison(
    row: pd.Series, population: pd.DataFrame, metric_fields: list[str]
) -> dict[str, dict[str, float | None]]:
    result: dict[str, dict[str, float | None]] = {}
    for column in metric_fields:
        if column not in population or column not in row.index:
            continue
        value = float(row[column]) if pd.notna(row[column]) else None
        mean = float(population[column].mean())
        scale = float(population[column].std(ddof=0))
        result[column] = {
            "value": value,
            "population_mean": mean,
            "difference": value - mean if value is not None else None,
            "z_score": (
                (value - mean) / scale
                if value is not None and scale > 0
                else 0.0 if value is not None else None
            ),
        }
    return result


def build_selection(data_version: str) -> dict[str, Any]:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    eligibility_min = int(config.get("eligibility_min_pressure_balls", 80))
    min_ipl_share = float(config.get("min_ipl_share", 0.30))
    split = config.get("selection_role_split", {})
    batters_required = int(split.get("batters", 6))
    bowlers_required = int(split.get("bowlers", 4))
    if (
        batters_required < 0
        or bowlers_required < 0
        or batters_required + bowlers_required != 10
    ):
        raise ValueError("selection_role_split must be non-negative and sum to ten")

    metrics = pd.read_parquet(PROCESSED / "player_metrics.parquet")
    sensitivity = json.loads(
        (PROCESSED / "sensitivity.json").read_text(encoding="utf-8")
    )
    backtest = json.loads((PROCESSED / "backtest.json").read_text(encoding="utf-8"))
    if metrics["player_id"].duplicated().any():
        raise ValueError("Player metrics must contain unique player IDs")
    required = {
        "selection_eligible",
        "eligible_batter",
        "eligible_bowler",
        "ipl_ball_share",
        "batting_wpa_conservative_lower_bound",
        "bowling_wpa_conservative_lower_bound",
        "batting_overall_shrunk_wpa_per_ball",
        "bowling_overall_shrunk_wpa_per_ball",
        "batting_overall_weighted_balls",
        "bowling_overall_weighted_balls",
        "batting_pressure_wpa_per_ball_shrunk_30",
        "bowling_pressure_wpa_per_ball_shrunk_30",
        "batting_runs_saved_per_ball",
        "bowling_runs_saved_per_ball",
    }
    missing = required - set(metrics.columns)
    if missing:
        raise ValueError(f"Player metrics are missing selection columns: {sorted(missing)}")

    metrics = metrics.copy().set_index("player_id", drop=False)
    sensitivity_players = sensitivity["players"]
    metrics["inclusion_probability"] = [
        sensitivity_players.get(str(int(player_id)), {}).get(
            "top10_inclusion_probability", 0.0
        )
        for player_id in metrics["player_id"]
    ]
    metrics["rank_stability_score"] = [
        sensitivity_players.get(str(int(player_id)), {}).get(
            "rank_stability_score", 0.0
        )
        for player_id in metrics["player_id"]
    ]

    selected_predictor = backtest["selected_predictor"]
    pressure_weights = selected_predictor.get(
        "pressure_weight_by_role",
        {
            "batting": float(selected_predictor["pressure_weight"]),
            "bowling": float(selected_predictor["pressure_weight"]),
        },
    )
    batter_pressure_weight = float(pressure_weights["batting"])
    bowler_pressure_weight = float(pressure_weights["bowling"])
    combined_pressure_weight = float(selected_predictor["pressure_weight"])
    bat_volume = metrics["batting_pressure_balls"].fillna(0).astype(float)
    bowl_volume = metrics["bowling_pressure_balls"].fillna(0).astype(float)
    pressure_volume = bat_volume + bowl_volume
    pressure_predictor = np.divide(
        metrics["batting_pressure_wpa_per_ball_shrunk_30"].fillna(0) * bat_volume
        + metrics["bowling_pressure_wpa_per_ball_shrunk_30"].fillna(0) * bowl_volume,
        pressure_volume,
        out=np.zeros(len(metrics), dtype=float),
        where=pressure_volume.to_numpy() > 0,
    )
    overall_bat_volume = metrics["batting_overall_weighted_balls"].fillna(0).astype(float)
    overall_bowl_volume = metrics["bowling_overall_weighted_balls"].fillna(0).astype(float)
    overall_volume = overall_bat_volume + overall_bowl_volume
    overall_predictor = np.divide(
        metrics["batting_overall_shrunk_wpa_per_ball"].fillna(0)
        * overall_bat_volume
        + metrics["bowling_overall_shrunk_wpa_per_ball"].fillna(0)
        * overall_bowl_volume,
        overall_volume,
        out=np.zeros(len(metrics), dtype=float),
        where=overall_volume.to_numpy() > 0,
    )
    metrics["selected_predictor_wpa_per_ball"] = (
        combined_pressure_weight * pressure_predictor
        + (1.0 - combined_pressure_weight) * overall_predictor
    )
    metrics["selected_batter_wpa_per_ball"] = (
        batter_pressure_weight
        * metrics["batting_pressure_wpa_per_ball_shrunk_30"].fillna(0)
        + (1.0 - batter_pressure_weight)
        * metrics["batting_overall_shrunk_wpa_per_ball"].fillna(0)
    )
    metrics["selected_bowler_wpa_per_ball"] = (
        bowler_pressure_weight
        * metrics["bowling_pressure_wpa_per_ball_shrunk_30"].fillna(0)
        + (1.0 - bowler_pressure_weight)
        * metrics["bowling_overall_shrunk_wpa_per_ball"].fillna(0)
    )
    metrics["selected_batter_runs_saved_per_ball"] = metrics[
        "batting_runs_saved_per_ball"
    ].fillna(0)
    metrics["selected_bowler_runs_saved_per_ball"] = metrics[
        "bowling_runs_saved_per_ball"
    ].fillna(0)
    metrics["selected_batter_consistency"] = metrics["batting_consistency"].fillna(0.5)
    metrics["selected_bowler_consistency"] = metrics["bowling_consistency"].fillna(0.5)

    index_weights = backtest["rolling_only_weight_selection"]["index_weights"]
    weight_total = sum(float(value) for value in index_weights.values())
    if weight_total <= 0:
        raise ValueError("Selection index weights must have a positive sum")
    eligible_batters = metrics["eligible_batter"].astype(bool)
    eligible_bowlers = metrics["eligible_bowler"].astype(bool)
    for role, valid, predictor_column, runs_column, consistency_column in (
        (
            "batter",
            eligible_batters,
            "selected_batter_wpa_per_ball",
            "selected_batter_runs_saved_per_ball",
            "selected_batter_consistency",
        ),
        (
            "bowler",
            eligible_bowlers,
            "selected_bowler_wpa_per_ball",
            "selected_bowler_runs_saved_per_ball",
            "selected_bowler_consistency",
        ),
    ):
        metrics[f"{role}_final_score"] = (
            float(index_weights["wpa"])
            * _zscore(metrics[predictor_column], valid)
            + float(index_weights["runs_saved"])
            * _zscore(metrics[runs_column], valid)
            + float(index_weights["consistency"])
            * _zscore(metrics[consistency_column], valid)
        ) / weight_total

    overall_only_metrics = metrics.copy()
    overall_only_metrics["selected_batter_wpa_per_ball"] = metrics[
        "batting_overall_shrunk_wpa_per_ball"
    ].fillna(0)
    overall_only_metrics["selected_bowler_wpa_per_ball"] = metrics[
        "bowling_overall_shrunk_wpa_per_ball"
    ].fillna(0)
    for role, valid, predictor_column, runs_column, consistency_column in (
        (
            "batter",
            eligible_batters,
            "selected_batter_wpa_per_ball",
            "selected_batter_runs_saved_per_ball",
            "selected_batter_consistency",
        ),
        (
            "bowler",
            eligible_bowlers,
            "selected_bowler_wpa_per_ball",
            "selected_bowler_runs_saved_per_ball",
            "selected_bowler_consistency",
        ),
    ):
        overall_only_metrics[f"{role}_final_score"] = (
            float(index_weights["wpa"])
            * _zscore(overall_only_metrics[predictor_column], valid)
            + float(index_weights["runs_saved"])
            * _zscore(overall_only_metrics[runs_column], valid)
            + float(index_weights["consistency"])
            * _zscore(overall_only_metrics[consistency_column], valid)
        ) / weight_total

    metrics["confidence_grade"] = [
        _confidence_grade(
            float(metrics.loc[player_id, "inclusion_probability"]),
            float(metrics.loc[player_id, "rank_stability_score"]),
            int(metrics.loc[player_id, "pressure_balls"]),
        )
        for player_id in metrics.index
    ]
    role_rankings = _role_rankings(metrics.reset_index(drop=True))
    eligible = metrics.loc[metrics["selection_eligible"]].copy()
    eligible["combined_final_score"] = np.where(
        eligible["eligible_batter"],
        eligible["batter_final_score"],
        -np.inf,
    )
    eligible["combined_final_score"] = np.maximum(
        eligible["combined_final_score"],
        np.where(eligible["eligible_bowler"], eligible["bowler_final_score"], -np.inf),
    )
    eligible = eligible.sort_values(
        "combined_final_score", ascending=False, kind="stable"
    )
    unconstrained_top = eligible.head(10)["player_id"].astype(int).tolist()
    assignments = _choose_top_ten(
        eligible.reset_index(drop=True), batters_required, bowlers_required
    )
    if len(assignments) != 10:
        raise ValueError(
            "Unable to meet the configured batter/bowler split with eligible players"
        )
    overall_only_eligible = overall_only_metrics.loc[
        overall_only_metrics["selection_eligible"]
    ].copy()
    overall_only_eligible["combined_final_score"] = np.maximum(
        np.where(
            overall_only_eligible["eligible_batter"],
            overall_only_eligible["batter_final_score"],
            -np.inf,
        ),
        np.where(
            overall_only_eligible["eligible_bowler"],
            overall_only_eligible["bowler_final_score"],
            -np.inf,
        ),
    )
    overall_only_assignments = _choose_top_ten(
        overall_only_eligible.reset_index(drop=True),
        batters_required,
        bowlers_required,
    )
    if len(overall_only_assignments) != 10:
        raise ValueError("Unable to build the overall_only comparison Top 10")
    final_top10_ids = [player_id for player_id, _ in assignments]
    overall_only_top10_ids = [
        player_id for player_id, _ in overall_only_assignments
    ]
    selected_ids = {player_id for player_id, _ in assignments}
    remaining = eligible.loc[~eligible["player_id"].isin(selected_ids)]
    reserves = [
        (
            int(row.player_id),
            "batter"
            if bool(row.eligible_batter)
            and (
                not bool(row.eligible_bowler)
                or float(row.batter_final_score) >= float(row.bowler_final_score)
            )
            else "bowler",
        )
        for row in remaining.itertuples(index=False)
    ][:10]
    ordered = assignments + reserves
    rank_lookup = {
        player_id: rank for rank, (player_id, _) in enumerate(ordered, start=1)
    }
    batter_rank = {
        int(item["player_id"]): int(item["rank"])
        for item in role_rankings["batters"]
    }
    bowler_rank = {
        int(item["player_id"]): int(item["rank"])
        for item in role_rankings["bowlers"]
    }

    metric_fields = [
        "pressure_balls",
        "selected_predictor_wpa_per_ball",
        "runs_above_situation_average",
        "strike_rate",
        "economy",
        "ipl_ball_share",
    ]
    role_populations = {
        "batter": metrics.loc[eligible_batters],
        "bowler": metrics.loc[eligible_bowlers],
    }
    records: list[dict[str, Any]] = []
    for player_id, slot in ordered:
        row = metrics.loc[player_id]
        role = str(row["role"])
        inclusion_probability = float(row["inclusion_probability"])
        rank_stability = float(row["rank_stability_score"])
        pressure_balls = int(row["pressure_balls"])
        batting_lower, batting_upper = (
            row.get("batting_wpa_ci80_lower"),
            row.get("batting_wpa_ci80_upper"),
        )
        bowling_lower, bowling_upper = (
            row.get("bowling_wpa_ci80_lower"),
            row.get("bowling_wpa_ci80_upper"),
        )
        batting_interval = (
            [float(batting_lower), float(batting_upper)]
            if pd.notna(batting_lower) and pd.notna(batting_upper)
            else None
        )
        bowling_interval = (
            [float(bowling_lower), float(bowling_upper)]
            if pd.notna(bowling_lower) and pd.notna(bowling_upper)
            else None
        )
        selected_interval = batting_interval if slot == "batter" else bowling_interval
        population = role_populations[slot]
        comparison_row = row.copy()
        comparison_population = population.copy()
        predictor_metric = f"selected_{slot}_wpa_per_ball"
        comparison_row["selected_predictor_wpa_per_ball"] = row[predictor_metric]
        comparison_population["selected_predictor_wpa_per_ball"] = comparison_population[
            predictor_metric
        ]
        metrics_vs_population = _metric_comparison(
            comparison_row, comparison_population, metric_fields
        )
        slot_score = float(row[f"{slot}_final_score"])
        records.append(
            {
                "rank": rank_lookup[player_id],
                "designation": "pick" if rank_lookup[player_id] <= 10 else "reserve",
                "selection_slot": slot,
                "batter_rank": batter_rank.get(player_id),
                "bowler_rank": bowler_rank.get(player_id),
                "player_id": int(player_id),
                "display_name": str(row["player_name"]),
                "player_name": str(row["player_name"]),
                "role": role,
                "role_label": role,
                "pressure_balls": pressure_balls,
                "batting_pressure_balls": int(row["batting_pressure_balls"])
                if pd.notna(row["batting_pressure_balls"])
                else 0,
                "bowling_pressure_balls": int(row["bowling_pressure_balls"])
                if pd.notna(row["bowling_pressure_balls"])
                else 0,
                "ipl_share": float(row["ipl_ball_share"]),
                "ipl_ball_share": float(row["ipl_ball_share"]),
                "key_metrics_vs_population": metrics_vs_population,
                "wpa_80_interval": selected_interval,
                "batting_wpa_80_interval": batting_interval,
                "bowling_wpa_80_interval": bowling_interval,
                "bootstrap_inclusion_probability": inclusion_probability,
                "top10_inclusion_probability": inclusion_probability,
                "rank_stability": rank_stability,
                "rank_stability_score": rank_stability,
                "grade": str(row["confidence_grade"]),
                "confidence_grade": str(row["confidence_grade"]),
                "final_score": slot_score,
                "selected_predictor_wpa_per_ball": float(
                    row[f"selected_{slot}_wpa_per_ball"]
                ),
                "selected_runs_saved_per_ball": float(
                    row[f"selected_{slot}_runs_saved_per_ball"]
                ),
                "selected_consistency": float(row[f"selected_{slot}_consistency"]),
                "bootstrap_rank_interval_80": sensitivity_players.get(
                    str(player_id), {}
                ).get("rank_interval_80"),
            }
        )

    top10_records = records[:10]
    grade_counts = {
        group: {
            grade: sum(item["grade"] == grade for item in group_items)
            for grade in ("A", "B", "C")
        }
        for group, group_items in (
            ("top10", top10_records),
            ("top20", records),
        )
    }
    final_fold_predictors = backtest["folds"][
        "final_train_2022_25_test_2026"
    ]["predictors"]
    holdout = backtest["final_holdout_population"]
    holdout_players = holdout["players"]
    population_holdout_mean = float(
        holdout["population_mean_realized_pressure_wpa_per_ball"]
    )

    def list_holdout_metrics(player_ids: list[int], predictor_name: str) -> dict[str, Any]:
        values = final_fold_predictors[predictor_name]["by_role"]["combined"]
        realized_top_ids = set(holdout["realized_top20_player_ids"])
        observed = [
            float(holdout_players[str(player_id)]["realized_pressure_wpa_per_ball"])
            for player_id in player_ids
            if str(player_id) in holdout_players
        ]
        mean_realized = float(np.mean(observed)) if observed else None
        return {
            "spearman": values["spearman"],
            "precision_at_10": len(set(player_ids) & realized_top_ids) / 10.0,
            "random_baseline_precision_at_10": (
                float(holdout["realized_top20_count"])
                / float(holdout["population_count"])
            ),
            "predicted_top10_wpa_uplift": (
                mean_realized - population_holdout_mean
                if mean_realized is not None
                else None
            ),
            "predicted_top10_mean_realized_pressure_wpa_per_ball": mean_realized,
            "population_mean_realized_pressure_wpa_per_ball": population_holdout_mean,
            "predictor_level_metrics": {
                key: values.get(key)
                for key in (
                    "spearman",
                    "precision_at_10",
                    "predicted_top10_wpa_uplift",
                )
            },
        }

    final_list_backtests = {}
    for label, player_ids, predictor_name in (
        ("pressure_aware_blend", final_top10_ids, selected_predictor["name"]),
        ("overall_only", overall_only_top10_ids, "overall_only"),
    ):
        final_list_backtests[label] = list_holdout_metrics(
            player_ids, predictor_name
        )
    top10_comparison = {
        "comparison_period": "final_train_2022_25_test_2026",
        "overlap_count": len(set(final_top10_ids) & set(overall_only_top10_ids)),
        "pressure_aware_top10_player_ids": final_top10_ids,
        "overall_only_top10_player_ids": overall_only_top10_ids,
        "pressure_aware_predictor_weights": {
            "batting_pressure": batter_pressure_weight,
            "bowling_pressure": bowler_pressure_weight,
        },
        "backtest_metrics": final_list_backtests,
        "metrics_scope": (
            "Precision@10 and uplift use each exact Top-10 list against 2026 outcomes. "
            "Spearman is the out-of-time predictor-level rank correlation. These "
            "current full-sample rosters include 2026 data, so list-level outcomes "
            "are retrospective, not independent forecasts."
        ),
    }
    top10_ipl_shares = [float(item["ipl_share"]) for item in top10_records]
    report = {
        "data_version": data_version,
        "eligibility": {
            "eligibility_min_pressure_balls": eligibility_min,
            "min_ipl_share": min_ipl_share,
            "players_eligible": int(len(eligible)),
        },
        "top10_role_constraint": {
            "default_split": {"batters": 6, "bowlers": 4},
            "configured_split": {
                "batters": batters_required,
                "bowlers": bowlers_required,
            },
            "constraint_satisfied": True,
            "note": (
                "All-rounders keep the all_rounder role label and fill exactly one "
                "assigned slot, batter or bowler; they count only once toward the "
                "6-4 split."
            ),
            "changed_unconstrained_list": (
                [player_id for player_id, _ in assignments] != unconstrained_top
            ),
            "unconstrained_top10_ids": unconstrained_top,
            "constrained_top10_ids": [player_id for player_id, _ in assignments],
        },
        "ranking_predictor": selected_predictor,
        "pressure_weight_by_role": {
            "batting": batter_pressure_weight,
            "bowling": bowler_pressure_weight,
        },
        "index_weights": index_weights,
        "league_weights": backtest["rolling_only_weight_selection"]["league_weights"],
        "ipl_share_range": {
            "minimum": min(top10_ipl_shares),
            "maximum": max(top10_ipl_shares),
        },
        "top10_comparison": top10_comparison,
        "score_formula": (
            "role-specific weighted z-score of selected predictor WPA/ball, pressure "
            "runs-saved/ball, and pressure consistency using backtest-selected weights"
        ),
        "role_rankings": role_rankings,
        "items": records,
        "combined_top10": top10_records,
        "reserves": records[10:20],
        "confidence_grade_thresholds": GRADE_THRESHOLDS,
        "grade_counts": grade_counts,
    }
    (PROCESSED / "top10.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame(records).to_csv(PROCESSED / "top10.csv", index=False)
    return report
