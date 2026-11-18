from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

BACKEND_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = BACKEND_ROOT / "data" / "processed"


def _print_table(title: str, headers: list[str], rows: list[list[object]]) -> None:
    print(title)
    widths = [
        max(len(header), *(len(str(row[index])) for row in rows))
        for index, header in enumerate(headers)
    ]
    print(" | ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(str(value).ljust(widths[index]) for index, value in enumerate(row)))


def main() -> None:
    quality = json.loads((PROCESSED / "data_quality.json").read_text(encoding="utf-8"))
    win_report = json.loads(
        (PROCESSED / "win_probability_report.json").read_text(encoding="utf-8")
    )
    sensitivity = json.loads(
        (PROCESSED / "sensitivity.json").read_text(encoding="utf-8")
    )
    backtest = json.loads((PROCESSED / "backtest.json").read_text(encoding="utf-8"))
    selection = json.loads((PROCESSED / "top10.json").read_text(encoding="utf-8"))
    selection_csv = pd.read_csv(PROCESSED / "top10.csv")
    metrics = pd.read_parquet(PROCESSED / "player_metrics.parquet")

    if quality["matches_used"] < 500:
        raise AssertionError(f"Only {quality['matches_used']} cleaned matches")
    if sum(quality["matches_used_by_comp_group"].values()) != quality["matches_used"]:
        raise AssertionError("Competition group counts do not sum to matches_used")
    if quality["rows_dropped"] != sum(
        item["rows_dropped"] for item in quality["exclusions"].values()
    ):
        raise AssertionError("Exclusion rows do not sum to rows_dropped")
    if quality["max_wpa_sum_error"] > 1e-6:
        raise AssertionError("Per-match WPA sum error exceeds 1e-6")
    if metrics["player_id"].duplicated().any():
        raise AssertionError("Player metrics contain duplicate player IDs")

    for report_name, report in (
        ("win_probability_report.json", win_report),
        ("sensitivity.json", sensitivity),
        ("backtest.json", backtest),
        ("top10.json", selection),
    ):
        if report["data_version"] != quality["data_version"]:
            raise AssertionError(f"Inconsistent data_version in {report_name}")
    for output_path in PROCESSED.glob("*.parquet"):
        frame = pd.read_parquet(output_path, columns=["data_version"])
        if frame["data_version"].nunique() > 1 or (
            not frame.empty
            and frame["data_version"].iloc[0] != quality["data_version"]
        ):
            raise AssertionError(f"Inconsistent data_version in {output_path.name}")

    for innings in ("1", "2"):
        innings_metrics = win_report["by_innings"].get(innings)
        if (
            not innings_metrics
            or innings_metrics["n_matches"] < 5
            or len(innings_metrics["calibration_10_bins"]) != 10
        ):
            raise AssertionError(f"Incomplete 10-bin calibration table for innings {innings}")
    if not np.isfinite(win_report["calibration_max_abs_gap"]):
        raise AssertionError("Invalid maximum calibration gap")
    if "match_level_chase_win_rate" not in win_report:
        raise AssertionError("Missing match-level chase win rate")

    eligibility = quality["player_metrics"]
    min_bat_pressure = eligibility["all_rounder_min_batting_pressure_balls"]
    min_bowl_pressure = eligibility["all_rounder_min_bowling_pressure_balls"]
    all_rounders = metrics.loc[metrics["is_all_rounder"]]
    if (
        all_rounders["batting_pressure_balls"].lt(min_bat_pressure).any()
        or all_rounders["bowling_pressure_balls"].lt(min_bowl_pressure).any()
    ):
        raise AssertionError("An all-rounder does not meet both legal-pressure-ball thresholds")
    if min_bat_pressure != 100 or min_bowl_pressure != 100:
        raise AssertionError("All-rounder role thresholds must both be 100 legal pressure balls")

    eligibility_min = selection["eligibility"]["eligibility_min_pressure_balls"]
    min_ipl_share = selection["eligibility"]["min_ipl_share"]
    if min_ipl_share != 0.30:
        raise AssertionError("Selection eligibility must require at least 0.30 IPL share")
    eligible_batters = metrics["eligible_batter"].astype(bool)
    eligible_bowlers = metrics["eligible_bowler"].astype(bool)
    if (
        metrics.loc[eligible_batters, "batting_pressure_balls"].lt(eligibility_min).any()
        or metrics.loc[eligible_bowlers, "bowling_pressure_balls"].lt(eligibility_min).any()
        or metrics.loc[
            eligible_batters | eligible_bowlers, "ipl_ball_share"
        ].lt(min_ipl_share).any()
    ):
        raise AssertionError("Selection eligibility does not meet configured thresholds")

    items = selection["items"]
    picks = [item for item in items if item["rank"] <= 10]
    reserves = [item for item in items if 11 <= item["rank"] <= 20]
    if len(picks) != 10 or len(reserves) != 10:
        raise AssertionError("Top-10 selection must contain ten picks and ten reserves")
    required_selection_fields = {
        "player_id",
        "display_name",
        "role",
        "pressure_balls",
        "ipl_share",
        "key_metrics_vs_population",
        "wpa_80_interval",
        "bootstrap_inclusion_probability",
        "rank_stability",
        "grade",
    }
    if not required_selection_fields.issubset(selection_csv.columns):
        raise AssertionError("Top-10 CSV is missing required output fields")
    if len(selection_csv) != 20:
        raise AssertionError("Top-10 CSV must include ten picks and ten reserves")
    for item in picks:
        if not required_selection_fields.issubset(item):
            raise AssertionError(
                f"Top-10 JSON record is missing required fields for player {item['player_id']}"
            )
    if set(item["player_id"] for item in picks) - set(
        metrics.loc[eligible_batters | eligible_bowlers, "player_id"].astype(int)
    ):
        raise AssertionError("Top-10 includes a selection-ineligible player")
    pick_ids = [item["player_id"] for item in picks]
    pick_metrics = metrics.set_index("player_id").loc[pick_ids]
    can_bat = pick_metrics["eligible_batter"].astype(bool)
    can_bowl = pick_metrics["eligible_bowler"].astype(bool)
    fixed_batters = int((can_bat & ~can_bowl).sum())
    fixed_bowlers = int((can_bowl & ~can_bat).sum())
    flexible_all_rounders = int((can_bat & can_bowl).sum())
    if (
        fixed_batters + flexible_all_rounders < 3
        or fixed_bowlers + flexible_all_rounders < 3
    ):
        raise AssertionError("Top-10 role constraints are not satisfied")
    configured_split = selection["top10_role_constraint"]["configured_split"]
    assigned_batters = sum(item["selection_slot"] == "batter" for item in picks)
    assigned_bowlers = sum(item["selection_slot"] == "bowler" for item in picks)
    if (
        assigned_batters != configured_split["batters"]
        or assigned_bowlers != configured_split["bowlers"]
    ):
        raise AssertionError("Top-10 does not match the configured role split")
    if configured_split != {"batters": 6, "bowlers": 4}:
        raise AssertionError("Top-10 role split must be configured as 6 batters / 4 bowlers")
    for role, role_key in (("batters", "batter_rank"), ("bowlers", "bowler_rank")):
        ranking = selection["role_rankings"][role]
        if [item["rank"] for item in ranking] != list(range(1, len(ranking) + 1)):
            raise AssertionError(f"{role.title()} ranking has invalid rank sequence")
        if any(item[role_key] is None for item in picks if item["selection_slot"] == role[:-1]):
            raise AssertionError(f"Selected {role} are missing role-specific ranks")

    expected_folds = {
        "rolling_train_2022_23_test_2024",
        "rolling_train_2022_24_test_2025_26",
        "final_train_2022_25_test_2026",
    }
    if set(backtest["folds"]) != expected_folds:
        raise AssertionError("Backtest must contain exactly the three temporal folds")
    final_predictors = backtest["folds"][
        "final_train_2022_25_test_2026"
    ]["predictors"]
    if set(final_predictors) != {
        backtest["selected_predictor"]["name"],
        "overall_only",
    }:
        raise AssertionError("Final holdout must compare the selected blend with overall_only")
    if selection["ranking_predictor"] != backtest["selected_predictor"]:
        raise AssertionError("Selection did not use the winning backtest predictor")
    expected_note = (
        "differences between predictors were within noise; pressure-aware blend chosen "
        "to match the brief"
    )
    if backtest["predictor_decision"]["note"] != expected_note:
        raise AssertionError("Backtest is missing the pressure-aware predictor rationale")
    if selection["pressure_weight_by_role"] != {"batting": 0.75, "bowling": 0.5}:
        raise AssertionError("Role-specific pressure blend weights are incorrect")
    expected_index_weights = {"wpa": 0.4, "runs_saved": 0.4, "consistency": 0.2}
    expected_league_weights = {"IPL": 1.0, "International": 0.8, "Emerging": 0.4}
    if selection["index_weights"] != expected_index_weights:
        raise AssertionError("Selection index weights differ from requested fixed weights")
    if selection["league_weights"] != expected_league_weights:
        raise AssertionError("Selection league weights differ from requested fixed weights")
    comparison = selection["top10_comparison"]
    if backtest.get("selection_comparison") != comparison:
        raise AssertionError("Backtest and selection roster comparisons do not match")
    if comparison["overlap_count"] != len(
        set(comparison["pressure_aware_top10_player_ids"])
        & set(comparison["overall_only_top10_player_ids"])
    ):
        raise AssertionError("Top-10 overlap count is inconsistent")
    if (
        len(comparison["pressure_aware_top10_player_ids"]) != 10
        or len(comparison["overall_only_top10_player_ids"]) != 10
        or set(comparison["pressure_aware_top10_player_ids"])
        != set(item["player_id"] for item in picks)
    ):
        raise AssertionError("Top-10 comparison lists do not match the selected rosters")
    if not np.isclose(
        selection["ipl_share_range"]["minimum"],
        min(item["ipl_share"] for item in picks),
    ) or not np.isclose(
        selection["ipl_share_range"]["maximum"],
        max(item["ipl_share"] for item in picks),
    ):
        raise AssertionError("Top-10 IPL share range is inconsistent")

    top20 = [item for item in items if item["rank"] <= 20]
    grades = pd.Series([item["confidence_grade"] for item in top20]).value_counts()
    top10_pressure = pd.Series([item["pressure_balls"] for item in picks])
    grade_thresholds = selection["confidence_grade_thresholds"]

    for report_name in ("player_season_metrics.parquet", "player_stroke_metrics.parquet"):
        aggregate = pd.read_parquet(PROCESSED / report_name, columns=["data_version"])
        if not aggregate.empty and aggregate["data_version"].iloc[0] != quality["data_version"]:
            raise AssertionError(f"Inconsistent data_version in {report_name}")

    print(f"Data version: {quality['data_version']}")
    print(
        f"Matches: {quality['matches_used']} | Deliveries: "
        f"{quality['checks']['total_rows']} | Legal balls: {quality['balls_used']}"
    )
    print(
        "Calibration: 10 bins per innings | maximum absolute gap: "
        f"{win_report['calibration_max_abs_gap']:.6f}"
    )
    print(f"Maximum WPA-sum error: {quality['max_wpa_sum_error']:.12g}")
    _print_table(
        "Eligible player counts",
        ["Category", "Players"],
        [
            ["Eligible batters", int(eligible_batters.sum())],
            ["Eligible bowlers", int(eligible_bowlers.sum())],
            ["Eligible either role", int((eligible_batters | eligible_bowlers).sum())],
            ["All-rounders (100+ legal pressure balls in each role)", int(len(all_rounders))],
        ],
    )
    print(
        "Top-10 pressure balls: "
        f"median={top10_pressure.median():.1f} | minimum={top10_pressure.min()}"
    )
    print(f"Top-10 role split: {configured_split}")
    print(
        "Top-10 IPL share range: "
        f"{selection['ipl_share_range']['minimum']:.3f}-"
        f"{selection['ipl_share_range']['maximum']:.3f}"
    )
    print(
        "Selected predictor: "
        f"{selection['ranking_predictor']['name']} "
        f"(pressure weights by role={selection['pressure_weight_by_role']})"
    )
    print(f"Index weights: {selection['index_weights']}")
    print(f"League weights: {selection['league_weights']}")
    print("Confidence-grade thresholds (each criterion required):")
    for grade, thresholds in grade_thresholds.items():
        print(
            f"  {grade}: inclusion >= {thresholds['minimum_inclusion_probability']:.2f}, "
            f"stability >= {thresholds['minimum_rank_stability']:.2f}, "
            f"pressure balls >= {thresholds['minimum_pressure_balls']}"
        )
    _print_table(
        "Top-20 confidence-grade distribution",
        ["Grade", "Players"],
        [[grade, int(grades.get(grade, 0))] for grade in ("A", "B", "C")],
    )
    print(f"Top-20 grade counts: {selection['grade_counts']['top20']}")
    print(f"Top-10 grade counts: {selection['grade_counts']['top10']}")
    print(
        "Role constraint changed unconstrained top 10: "
        f"{selection['top10_role_constraint']['changed_unconstrained_list']}"
    )
    backtest_rows = []
    for fold, fold_result in backtest["folds"].items():
        for predictor_name, predictor_result in fold_result["predictors"].items():
            for role, values in predictor_result["by_role"].items():
                backtest_rows.append(
                    [
                        fold,
                        predictor_name,
                        role,
                        "n/a"
                        if values["spearman"] is None
                        else f"{values['spearman']:.4f}",
                        "n/a"
                        if values["precision_at_10"] is None
                        else f"{values['precision_at_10']:.3f}",
                        "n/a"
                        if values["random_baseline_precision_at_10"] is None
                        else f"{values['random_baseline_precision_at_10']:.3f}",
                        "n/a"
                        if values[
                            "predicted_top10_mean_realized_pressure_wpa_per_ball"
                        ]
                        is None
                        else (
                            f"{values['predicted_top10_mean_realized_pressure_wpa_per_ball']:.6f}"
                        ),
                        "n/a"
                        if values[
                            "population_mean_realized_pressure_wpa_per_ball"
                        ]
                        is None
                        else (
                            f"{values['population_mean_realized_pressure_wpa_per_ball']:.6f}"
                        ),
                    ]
                )
    _print_table(
        "Backtest: aggregate out-of-time results by fold, predictor, and role",
        [
            "Fold",
            "Predictor",
            "Role",
            "Spearman",
            "Precision@10",
            "Random baseline",
            "Top-10 realized pressure WPA/ball",
            "Population pressure WPA/ball",
        ],
        backtest_rows,
    )
    print(
        "Weight selection: "
        f"predictor={backtest['selected_predictor']['name']} "
        f"(pressure weights by role={selection['pressure_weight_by_role']}); "
        f"index={backtest['rolling_only_weight_selection']['selected_index_candidate']}; "
        f"league={backtest['rolling_only_weight_selection']['selected_league_candidate']}; "
        f"rolling precision@10="
        f"{backtest['rolling_only_weight_selection']['rolling_mean_precision_at_10']:.6f}; "
        "2026 final holdout used for selection=False"
    )
    print(
        "Top-10 overlap with overall_only: "
        f"{comparison['overlap_count']}/10"
    )
    for label, values in comparison["backtest_metrics"].items():
        print(
            f"{label} Top-10: Spearman={values['spearman']:.4f} "
            "(predictor-level), "
            f"precision@10={values['precision_at_10']:.3f} (list), "
            f"uplift={values['predicted_top10_wpa_uplift']:.6f} (list)"
        )
    evaluated_scores = [
        values["spearman"]
        for fold_result in backtest["folds"].values()
        for predictor_result in fold_result["predictors"].values()
        for values in predictor_result["by_role"].values()
        if values["spearman"] is not None
    ]
    print(
        "Weak-result aggregates (Spearman < 0.20): "
        f"{len(backtest['weak_results'])}/{len(evaluated_scores)}"
    )
    chart_names = (
        "top10_intervals.png",
        "backtest_summary.png",
        "calibration.png",
        "flag_shares.png",
    )
    missing_charts = [
        name for name in chart_names if not (PROCESSED / "charts" / name).is_file()
    ]
    if missing_charts:
        raise AssertionError(f"Missing chart exports: {missing_charts}")
    print(f"Chart exports: {len(chart_names)} present")


if __name__ == "__main__":
    main()
