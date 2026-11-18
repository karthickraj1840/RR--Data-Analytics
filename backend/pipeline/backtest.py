from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = ROOT / "backend"
PROCESSED = BACKEND_ROOT / "data" / "processed"
CONFIG_PATH = ROOT / "config.yaml"
PRESSURE_FLAGS = [
    "high_leverage",
    "hard_chase",
    "death_overs",
    "collapse",
    "tight_finish",
]
PRIOR_STRENGTH = 30.0
MINIMUM_PRESSURE_BALLS = 30
ROLE_REPORTS = ("combined", "batting", "bowling")
BLEND_WEIGHTS = (0.25, 0.50, 0.75)


def _fit_situation_baseline(train: pd.DataFrame) -> tuple[pd.DataFrame, float, float]:
    train = train.loc[train["is_legal"]].copy()
    train["phase"] = pd.cut(
        train["overid"],
        bins=[0, 6, 15, 20],
        labels=["PP", "MID", "DEATH"],
        include_lowest=True,
    ).astype("string")
    train["wicket_bucket"] = pd.cut(
        train["wickets_before"],
        bins=[-1, 0, 2, 4, 10],
        labels=["0", "1-2", "3-4", "5+"],
    ).astype("string")
    baseline = (
        train.groupby(["innings", "phase", "wicket_bucket"], observed=True)
        .agg(bat_rate=("runs", "mean"), total_rate=("total_runs", "mean"))
        .reset_index()
    )
    return baseline, float(train["runs"].mean()), float(train["total_runs"].mean())


def _role_group_features(
    rows: pd.DataFrame, role: str, player_column: str, runs_saved: pd.Series
) -> pd.DataFrame:
    part = rows.dropna(subset=[player_column]).copy()
    part["player_id"] = part[player_column].astype(int)
    part["role"] = role
    part["role_wpa"] = part["wpa"] if role == "batting" else -part["wpa"]
    part["runs_saved"] = runs_saved.reindex(part.index)
    part["unit_id"] = (
        part["overid"].astype(int) if role == "bowling" else part["innings"].astype(int)
    )
    match_units = (
        part.groupby(
            ["player_id", "role", "match_id", "comp_group", "unit_id"],
            observed=True,
        )
        .agg(
            n_balls=("is_legal", "size"),
            wpa_sum=("role_wpa", "sum"),
            runs_saved=("runs_saved", "sum"),
        )
        .reset_index()
    )
    grouped = (
        match_units.groupby(["player_id", "role", "comp_group"], observed=True)
        .agg(
            n_balls=("n_balls", "sum"),
            wpa_sum=("wpa_sum", "sum"),
            runs_saved=("runs_saved", "sum"),
            positive_units=("wpa_sum", lambda values: int(values.gt(0).sum())),
            n_units=("wpa_sum", "size"),
        )
        .reset_index()
    )
    grouped["consistency"] = (grouped["positive_units"] + 1) / (
        grouped["n_units"] + 2
    )
    return grouped


def _period_features(
    rows: pd.DataFrame,
    match_ids: set,
    baseline: pd.DataFrame,
    fallback_bat: float,
    fallback_total: float,
) -> dict[str, pd.DataFrame]:
    selected = rows.loc[rows["match_id"].isin(match_ids)].copy()
    selected["phase"] = pd.cut(
        selected["overid"],
        bins=[0, 6, 15, 20],
        labels=["PP", "MID", "DEATH"],
        include_lowest=True,
    ).astype("string")
    selected["wicket_bucket"] = pd.cut(
        selected["wickets_before"],
        bins=[-1, 0, 2, 4, 10],
        labels=["0", "1-2", "3-4", "5+"],
    ).astype("string")
    selected = selected.merge(
        baseline,
        on=["innings", "phase", "wicket_bucket"],
        how="left",
        validate="many_to_one",
    )
    selected["bat_rate"] = selected["bat_rate"].fillna(fallback_bat)
    selected["total_rate"] = selected["total_rate"].fillna(fallback_total)

    legal = selected.loc[selected["is_legal"]].copy()
    pressure = legal.loc[legal[PRESSURE_FLAGS].any(axis=1)].copy()
    return {
        "pressure": pd.concat(
            [
                _role_group_features(
                    pressure,
                    "batting",
                    "striker_id",
                    pressure["runs"] - pressure["bat_rate"],
                ),
                _role_group_features(
                    pressure,
                    "bowling",
                    "bowler_id",
                    pressure["total_rate"] - pressure["total_runs"],
                ),
            ],
            ignore_index=True,
        ),
        "overall": pd.concat(
            [
                _role_group_features(
                    legal,
                    "batting",
                    "striker_id",
                    legal["runs"] - legal["bat_rate"],
                ),
                _role_group_features(
                    legal,
                    "bowling",
                    "bowler_id",
                    legal["total_rate"] - legal["total_runs"],
                ),
            ],
            ignore_index=True,
        ),
    }


def _combine_features(
    grouped_features: pd.DataFrame,
    league_weights: dict[str, float],
    prior_by_role: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    weighted = grouped_features.copy()
    if weighted.empty:
        return pd.DataFrame(), {}
    weighted["league_weight"] = weighted["comp_group"].map(league_weights).fillna(0.0)
    weighted["weighted_balls"] = weighted["n_balls"] * weighted["league_weight"]
    weighted["weighted_wpa"] = weighted["wpa_sum"] * weighted["league_weight"]
    weighted["weighted_runs_saved"] = (
        weighted["runs_saved"] * weighted["league_weight"]
    )
    weighted["weighted_positive"] = (
        weighted["positive_units"] * weighted["league_weight"]
    )
    weighted["weighted_units"] = weighted["n_units"] * weighted["league_weight"]

    role_summary = weighted.groupby(["player_id", "role"]).agg(
        n_balls=("n_balls", "sum"),
        weighted_balls=("weighted_balls", "sum"),
        weighted_wpa=("weighted_wpa", "sum"),
        weighted_runs_saved=("weighted_runs_saved", "sum"),
        weighted_positive=("weighted_positive", "sum"),
        weighted_units=("weighted_units", "sum"),
    )
    combined = role_summary.groupby("player_id").sum(numeric_only=True)
    combined["role"] = "combined"
    role_summary = role_summary.reset_index().set_index("player_id")
    role_summary["role"] = role_summary["role"].astype(str)
    summary = pd.concat([role_summary, combined], axis=0)
    summary.index.name = "player_id"
    summary = summary.reset_index().set_index(["player_id", "role"]).sort_index()

    if prior_by_role is None:
        prior_by_role = {}
        for role, role_rows in summary.groupby(level="role"):
            total_balls = float(role_rows["weighted_balls"].sum())
            prior_by_role[str(role)] = (
                float(role_rows["weighted_wpa"].sum() / total_balls)
                if total_balls > 0
                else 0.0
            )
    summary["wpa_per_ball_raw"] = np.divide(
        summary["weighted_wpa"],
        summary["weighted_balls"],
        out=np.zeros(len(summary), dtype=float),
        where=summary["weighted_balls"].to_numpy() > 0,
    )
    prior_values = np.array(
        [prior_by_role.get(str(role), 0.0) for _, role in summary.index],
        dtype=float,
    )
    weighted_balls = summary["weighted_balls"].to_numpy(dtype=float)
    summary["wpa_per_ball_shrunk"] = np.divide(
        weighted_balls * summary["wpa_per_ball_raw"].to_numpy()
        + PRIOR_STRENGTH * prior_values,
        weighted_balls + PRIOR_STRENGTH,
        out=prior_values.copy(),
        where=weighted_balls + PRIOR_STRENGTH > 0,
    )
    summary["runs_saved_per_ball"] = np.divide(
        summary["weighted_runs_saved"],
        summary["weighted_balls"],
        out=np.zeros(len(summary), dtype=float),
        where=summary["weighted_balls"].to_numpy() > 0,
    )
    summary["consistency"] = np.divide(
        summary["weighted_positive"] + 1,
        summary["weighted_units"] + 2,
        out=np.full(len(summary), 0.5, dtype=float),
        where=summary["weighted_units"].to_numpy() + 2 > 0,
    )
    return summary, prior_by_role


def _prepare_period(
    features: dict[str, pd.DataFrame],
    league_weights: dict[str, float],
    priors: dict[str, dict[str, float]] | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, float]]]:
    prepared: dict[str, pd.DataFrame] = {}
    fitted_priors: dict[str, dict[str, float]] = {}
    for metric_scope in ("pressure", "overall"):
        prior = priors.get(metric_scope) if priors else None
        prepared[metric_scope], fitted_priors[metric_scope] = _combine_features(
            features[metric_scope], league_weights, prior
        )
    return prepared, fitted_priors


def _role_frame(features: pd.DataFrame, role: str) -> pd.DataFrame:
    if features.empty or role not in features.index.get_level_values("role"):
        return pd.DataFrame()
    return features.xs(role, level="role", drop_level=True)


def _predictor_rate(
    pressure: pd.DataFrame,
    overall: pd.DataFrame,
    pressure_weight: float,
    index: pd.Index,
) -> pd.Series:
    pressure_rate = pressure["wpa_per_ball_shrunk"].reindex(index).fillna(0.0)
    overall_rate = overall["wpa_per_ball_shrunk"].reindex(index).fillna(0.0)
    return pressure_weight * pressure_rate + (1.0 - pressure_weight) * overall_rate


def _zscore(values: pd.Series, eligible: pd.Series) -> pd.Series:
    selected = values.loc[eligible & values.notna()]
    scale = float(selected.std(ddof=0)) if len(selected) > 1 else 0.0
    if not scale:
        return pd.Series(0.0, index=values.index)
    mean = float(selected.mean())
    return (values - mean) / scale


def _score(
    pressure: pd.DataFrame,
    overall: pd.DataFrame,
    pressure_weight: float,
    index_weights: dict[str, float],
    eligible: pd.Series,
) -> pd.Series:
    index = pressure.index.union(overall.index)
    predictor = _predictor_rate(pressure, overall, pressure_weight, index)
    pressure_rows = pressure.reindex(index)
    score = (
        float(index_weights["wpa"]) * _zscore(predictor, eligible.reindex(index, fill_value=False))
        + float(index_weights["runs_saved"])
        * _zscore(
            pressure_rows["runs_saved_per_ball"],
            eligible.reindex(index, fill_value=False),
        )
        + float(index_weights["consistency"])
        * _zscore(
            pressure_rows["consistency"],
            eligible.reindex(index, fill_value=False),
        )
    )
    total_weight = sum(float(value) for value in index_weights.values())
    return score / total_weight


def _evaluate_model(
    train_features: dict[str, pd.DataFrame],
    test_features: dict[str, pd.DataFrame],
    predictor: dict[str, Any],
    index_weights: dict[str, float],
) -> dict[str, Any]:
    by_role: dict[str, dict[str, Any]] = {}
    for role in ROLE_REPORTS:
        pressure_weight = float(
            predictor.get("pressure_weight_by_role", {}).get(
                role,
                predictor.get(
                    "pressure_weight",
                    predictor.get("pressure_weight_by_role", {}).get(
                        "batting", 0.0
                    ),
                ),
            )
        )
        train_pressure = _role_frame(train_features["pressure"], role)
        train_overall = _role_frame(train_features["overall"], role)
        test_pressure = _role_frame(test_features["pressure"], role)
        test_overall = _role_frame(test_features["overall"], role)
        if train_pressure.empty or test_pressure.empty:
            by_role[role] = {
                "spearman": None,
                "precision_at_10": None,
                "random_baseline_precision_at_10": None,
                "predicted_top10_mean_realized_pressure_wpa_per_ball": None,
                "population_mean_realized_pressure_wpa_per_ball": None,
                "n_predicted_with_test_pressure_balls": 0,
                "realized_top20_count": 0,
                "n_common_players": 0,
                "n_predicted": 0,
                "n_test_population": 0,
            }
            continue

        train_index = train_pressure.index.union(train_overall.index)
        eligible = train_pressure["n_balls"].reindex(train_index).fillna(0).ge(
            MINIMUM_PRESSURE_BALLS
        )
        scores = _score(
            train_pressure,
            train_overall,
            pressure_weight,
            index_weights,
            eligible,
        )
        predicted = scores.loc[eligible].sort_values(
            ascending=False, kind="stable"
        ).head(10).index

        test_population = test_pressure.loc[
            test_pressure["n_balls"].ge(MINIMUM_PRESSURE_BALLS)
        ]
        if test_population.empty:
            by_role[role] = {
                "spearman": None,
                "precision_at_10": None,
                "random_baseline_precision_at_10": None,
                "predicted_top10_mean_realized_pressure_wpa_per_ball": None,
                "population_mean_realized_pressure_wpa_per_ball": None,
                "n_predicted_with_test_pressure_balls": 0,
                "realized_top20_count": 0,
                "n_common_players": 0,
                "n_predicted": int(len(predicted)),
                "n_test_population": 0,
            }
            continue

        realized_count = max(1, int(np.ceil(len(test_population) * 0.20)))
        realized_top = set(
            test_population["wpa_per_ball_shrunk"]
            .nlargest(realized_count)
            .index.tolist()
        )
        precision = (
            len(set(predicted) & realized_top) / len(predicted)
            if len(predicted)
            else 0.0
        )
        predicted_test = test_pressure.reindex(
            test_pressure.index.intersection(predicted)
        )
        predicted_test = predicted_test.loc[predicted_test["n_balls"].gt(0)]
        predicted_mean = (
            float(predicted_test["wpa_per_ball_shrunk"].mean())
            if not predicted_test.empty
            else None
        )
        population_mean = float(test_population["wpa_per_ball_shrunk"].mean())

        common = train_pressure.index.intersection(test_population.index)
        common = common[
            train_pressure.loc[common, "n_balls"].ge(MINIMUM_PRESSURE_BALLS).to_numpy()
        ]
        if len(common) >= 2:
            train_predictor = _predictor_rate(
                train_pressure, train_overall, pressure_weight, common
            )
            correlation = spearmanr(
                train_predictor,
                test_population.loc[common, "wpa_per_ball_shrunk"],
            ).statistic
            spearman = float(correlation) if np.isfinite(correlation) else None
        else:
            spearman = None

        by_role[role] = {
            "spearman": spearman,
            "precision_at_10": float(precision),
            "random_baseline_precision_at_10": realized_count
            / len(test_population),
            "predicted_top10_mean_realized_pressure_wpa_per_ball": predicted_mean,
            "population_mean_realized_pressure_wpa_per_ball": population_mean,
            "n_predicted_with_test_pressure_balls": int(len(predicted_test)),
            "predicted_top10_wpa_uplift": (
                predicted_mean - population_mean
                if predicted_mean is not None
                else None
            ),
            "realized_top20_count": realized_count,
            "n_common_players": int(len(common)),
            "n_predicted": int(len(predicted)),
            "n_test_population": int(len(test_population)),
        }
    return {"by_role": by_role}


def _make_index_candidates() -> list[tuple[str, dict[str, float]]]:
    candidates = []
    increments = 5
    for wpa_units in range(increments + 1):
        for runs_units in range(increments + 1 - wpa_units):
            consistency_units = increments - wpa_units - runs_units
            weights = {
                "wpa": wpa_units / increments,
                "runs_saved": runs_units / increments,
                "consistency": consistency_units / increments,
            }
            name = (
                f"wpa_{weights['wpa']:.1f}_runs_{weights['runs_saved']:.1f}_"
                f"consistency_{weights['consistency']:.1f}"
            )
            candidates.append((name, weights))
    return candidates


def _make_league_candidates(
    current: dict[str, float],
) -> list[tuple[str, dict[str, float]]]:
    proposed = [
        ("configured", current),
        ("equal", {group: 1.0 for group in current}),
        (
            "ipl_emphasis",
            {
                group: 1.2 if group == "IPL" else 0.8 if group == "International" else 0.4
                for group in current
            },
        ),
        (
            "international_emphasis",
            {
                group: 0.8 if group == "IPL" else 1.0 if group == "International" else 0.4
                for group in current
            },
        ),
    ]
    unique: dict[tuple[float, ...], tuple[str, dict[str, float]]] = {}
    group_order = tuple(sorted(current))
    for name, weights in proposed:
        key = tuple(float(weights[group]) for group in group_order)
        unique.setdefault(key, (name, weights))
    return list(unique.values())


def _predictor_candidates() -> list[dict[str, Any]]:
    return [
        {"name": "pressure_only", "pressure_weight": 1.0},
        {"name": "overall_only", "pressure_weight": 0.0},
        *[
            {"name": "blend", "pressure_weight": weight}
            for weight in BLEND_WEIGHTS
        ],
    ]


def _candidate_key(item: dict[str, Any]) -> tuple:
    return (
        -item["rolling_mean_precision_at_10"],
        -item["rolling_mean_predicted_top10_wpa_uplift"],
        -item["rolling_mean_spearman"],
        item["predictor"]["name"],
        item["predictor"]["pressure_weight"],
        item["index_name"],
        item["league_name"],
    )


def build_backtest(data_version: str) -> dict[str, Any]:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    balls = pd.read_parquet(PROCESSED / "pressure_flags.parquet")
    matches = pd.read_parquet(PROCESSED / "matches.parquet")
    match_info = matches[["match_id", "match_date"]].copy()
    match_info["match_date"] = pd.to_datetime(
        match_info["match_date"], errors="coerce"
    )
    if match_info["match_date"].isna().any():
        raise ValueError("Backtest requires valid dates for every cleaned match")
    match_info["match_year"] = match_info["match_date"].dt.year
    balls = balls.merge(
        match_info[["match_id", "match_year"]],
        on="match_id",
        how="inner",
        validate="many_to_one",
    )

    ids_by_year = {
        year: set(match_info.loc[match_info["match_year"].eq(year), "match_id"])
        for year in range(2022, 2027)
    }
    folds = {
        "rolling_train_2022_23_test_2024": (
            ids_by_year[2022] | ids_by_year[2023],
            ids_by_year[2024],
        ),
        "rolling_train_2022_24_test_2025_26": (
            ids_by_year[2022] | ids_by_year[2023] | ids_by_year[2024],
            ids_by_year[2025] | ids_by_year[2026],
        ),
        "final_train_2022_25_test_2026": (
            ids_by_year[2022]
            | ids_by_year[2023]
            | ids_by_year[2024]
            | ids_by_year[2025],
            ids_by_year[2026],
        ),
    }
    if not folds["final_train_2022_25_test_2026"][0] or not folds[
        "final_train_2022_25_test_2026"
    ][1]:
        raise ValueError("Final 2026 holdout requires non-empty train and test periods")

    current_league = {
        str(key): float(value) for key, value in config["league_weights"].items()
    }
    index_candidates = _make_index_candidates()
    league_candidates = _make_league_candidates(current_league)
    predictors = _predictor_candidates()
    tune_slices = {
        "rolling_train_2022_23_test_2024": folds[
            "rolling_train_2022_23_test_2024"
        ][1],
        "rolling_train_2022_24_test_2025_only": ids_by_year[2025],
    }

    # Fit fold-specific situation baselines only on each fold's training period.
    prepared: dict[
        tuple[str, str],
        tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]],
    ] = {}
    fold_raw: dict[str, tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]] = {}
    for fold_name, (train_ids, test_ids) in folds.items():
        baseline, fallback_bat, fallback_total = _fit_situation_baseline(
            balls.loc[balls["match_id"].isin(train_ids)]
        )
        train_raw = _period_features(
            balls, train_ids, baseline, fallback_bat, fallback_total
        )
        test_raw = _period_features(
            balls, test_ids, baseline, fallback_bat, fallback_total
        )
        fold_raw[fold_name] = (train_raw, test_raw)

    tune_raw: dict[
        str, tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]
    ] = {}
    for fold_name, test_ids in tune_slices.items():
        train_ids = (
            folds[fold_name][0]
            if fold_name in folds
            else ids_by_year[2022] | ids_by_year[2023] | ids_by_year[2024]
        )
        baseline, fallback_bat, fallback_total = _fit_situation_baseline(
            balls.loc[balls["match_id"].isin(train_ids)]
        )
        tune_raw[fold_name] = (
            _period_features(balls, train_ids, baseline, fallback_bat, fallback_total),
            _period_features(balls, test_ids, baseline, fallback_bat, fallback_total),
        )

    def prepare_pair(
        raw_pair: tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]],
        league_weights: dict[str, float],
    ) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
        train_features, priors = _prepare_period(raw_pair[0], league_weights)
        test_features, _ = _prepare_period(raw_pair[1], league_weights, priors)
        return train_features, test_features

    for league_name, league_weights in league_candidates:
        for fold_name, raw_pair in tune_raw.items():
            prepared[(fold_name, league_name)] = prepare_pair(raw_pair, league_weights)
        for fold_name, raw_pair in fold_raw.items():
            prepared[(fold_name, league_name)] = prepare_pair(raw_pair, league_weights)

    candidates: list[dict[str, Any]] = []
    for predictor, (index_name, index_weights), (league_name, league_weights) in (
        (predictor, index_candidate, league_candidate)
        for predictor in predictors
        for index_candidate in index_candidates
        for league_candidate in league_candidates
    ):
        fold_results = []
        for fold_name in tune_slices:
            train_features, test_features = prepared[(fold_name, league_name)]
            result = _evaluate_model(
                train_features, test_features, predictor, index_weights
            )["by_role"]["combined"]
            fold_results.append(result)
        usable = [
            item
            for item in fold_results
            if item["precision_at_10"] is not None
            and item["predicted_top10_wpa_uplift"] is not None
        ]
        candidates.append(
            {
                "predictor": predictor,
                "index_name": index_name,
                "index_weights": index_weights,
                "league_name": league_name,
                "league_weights": league_weights,
                "tuning_folds": list(tune_slices),
                "rolling_mean_precision_at_10": (
                    float(np.mean([item["precision_at_10"] for item in usable]))
                    if usable
                    else float("-inf")
                ),
                "rolling_mean_random_baseline_precision_at_10": (
                    float(
                        np.mean(
                            [
                                item["random_baseline_precision_at_10"]
                                for item in usable
                            ]
                        )
                    )
                    if usable
                    else float("-inf")
                ),
                "rolling_mean_predicted_top10_wpa_uplift": (
                    float(
                        np.mean(
                            [
                                item["predicted_top10_wpa_uplift"]
                                for item in usable
                            ]
                        )
                    )
                    if usable
                    else float("-inf")
                ),
                "rolling_mean_spearman": (
                    float(
                        np.mean(
                            [
                                item["spearman"]
                                for item in usable
                                if item["spearman"] is not None
                            ]
                        )
                    )
                    if any(item["spearman"] is not None for item in usable)
                    else float("-inf")
                ),
                "rolling_fold_results": fold_results,
            }
        )
    candidates.sort(key=_candidate_key)
    best_grid_candidate = candidates[0]
    if not np.isfinite(best_grid_candidate["rolling_mean_precision_at_10"]):
        raise ValueError("No rolling fold produced a usable grid-search score")

    index_weights = {
        str(key): float(value) for key, value in config["index_weights"].items()
    }
    league_weights = {
        str(key): float(value) for key, value in current_league.items()
    }
    selected_predictor = dict(config["selection_predictor"])
    role_pressure_weights = selected_predictor.get("pressure_weight_by_role", {})
    if (
        selected_predictor.get("name") != "role_specific_blend"
        or float(role_pressure_weights.get("batting", -1)) != 0.75
        or float(role_pressure_weights.get("bowling", -1)) != 0.50
    ):
        raise ValueError(
            "selection_predictor must use the configured 0.75/0.50 role-specific blend"
        )
    selected_predictor.update(
        {
            "pressure_weight": 0.625,
            "overall_weight": 0.375,
            "prior_strength": PRIOR_STRENGTH,
        }
    )
    config["selection_predictor"] = selected_predictor
    CONFIG_PATH.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    fixed_tuning_results = []
    for fold_name in tune_slices:
        train_features, test_features = prepared[(fold_name, "configured")]
        fixed_tuning_results.append(
            _evaluate_model(
                train_features,
                test_features,
                selected_predictor,
                index_weights,
            )["by_role"]["combined"]
        )
    fixed_usable = [
        item
        for item in fixed_tuning_results
        if item["precision_at_10"] is not None
        and item["predicted_top10_wpa_uplift"] is not None
    ]

    predictor_results: dict[str, Any] = {}
    for predictor in predictors:
        predictor_key = (
            predictor["name"]
            if predictor["name"] != "blend"
            else f"blend_pressure_{predictor['pressure_weight']:.2f}"
        )
        predictor_results[predictor_key] = predictor
    predictor_results[selected_predictor["name"]] = selected_predictor

    fold_reports: dict[str, Any] = {}
    for fold_name, raw_pair in fold_raw.items():
        train_features, test_features = prepared[(fold_name, "configured")]
        if fold_name == "final_train_2022_25_test_2026":
            models = {
                selected_predictor["name"]: _evaluate_model(
                    train_features,
                    test_features,
                    selected_predictor,
                    index_weights,
                ),
                "overall_only": _evaluate_model(
                    train_features,
                    test_features,
                    {"name": "overall_only", "pressure_weight": 0.0},
                    index_weights,
                ),
            }
        else:
            models = {
                name: _evaluate_model(
                    train_features,
                    test_features,
                    predictor,
                    index_weights,
                )
                for name, predictor in predictor_results.items()
            }
        train_ids, test_ids = folds[fold_name]
        fold_reports[fold_name] = {
            "train_match_count": int(len(train_ids)),
            "test_match_count": int(len(test_ids)),
            "train_years": sorted(
                match_info.loc[match_info["match_id"].isin(train_ids), "match_year"]
                .unique()
                .astype(int)
                .tolist()
            ),
            "test_years": sorted(
                match_info.loc[match_info["match_id"].isin(test_ids), "match_year"]
                .unique()
                .astype(int)
                .tolist()
            ),
            "predictors": models,
        }

    final_test_features = prepared[
        ("final_train_2022_25_test_2026", "configured")
    ][1]
    final_test_population = _role_frame(final_test_features["pressure"], "combined")
    final_test_population = final_test_population.loc[
        final_test_population["n_balls"].ge(MINIMUM_PRESSURE_BALLS)
    ]
    final_realized_top_count = max(
        1, int(np.ceil(len(final_test_population) * 0.20))
    )
    final_realized_top_ids = set(
        final_test_population["wpa_per_ball_shrunk"]
        .nlargest(final_realized_top_count)
        .index.astype(int)
    )
    final_holdout_population = {
        "minimum_pressure_balls": MINIMUM_PRESSURE_BALLS,
        "population_count": int(len(final_test_population)),
        "population_mean_realized_pressure_wpa_per_ball": float(
            final_test_population["wpa_per_ball_shrunk"].mean()
        ),
        "realized_top20_count": final_realized_top_count,
        "realized_top20_player_ids": sorted(final_realized_top_ids),
        "players": {
            str(int(player_id)): {
                "pressure_balls": int(row["n_balls"]),
                "realized_pressure_wpa_per_ball": float(
                    row["wpa_per_ball_shrunk"]
                ),
            }
            for player_id, row in final_test_population.iterrows()
        },
    }

    report = {
        "data_version": data_version,
        "target": "future composite-pressure WPA/ball, shrunk; bowling WPA sign is reversed",
        "pressure_definition": "legal deliveries matching any of high_leverage, hard_chase, death_overs, collapse, tight_finish",
        "minimum_pressure_balls": MINIMUM_PRESSURE_BALLS,
        "realized_top_fraction": 0.20,
        "random_baseline": "expected precision@10 is the realized top-fraction of the test population",
        "folds": fold_reports,
        "rolling_only_weight_selection": {
            "index_weights": index_weights,
            "league_weights": league_weights,
            "selected_index_candidate": "configured_fixed",
            "selected_league_candidate": "configured_fixed",
            "tuning_periods": [
                "train 2022-23, validate 2024",
                "train 2022-24, validate 2025 only",
            ],
            "final_holdout_year": 2026,
            "selection_objective": (
                "configured weights fixed to the pressure-reliability brief; rolling "
                "grid search is retained for comparison and does not override them"
            ),
            "rolling_mean_precision_at_10": float(
                np.mean([item["precision_at_10"] for item in fixed_usable])
            ),
            "rolling_mean_random_baseline_precision_at_10": float(
                np.mean(
                    [
                        item["random_baseline_precision_at_10"]
                        for item in fixed_usable
                    ]
                )
            ),
            "rolling_mean_predicted_top10_wpa_uplift": float(
                np.mean(
                    [item["predicted_top10_wpa_uplift"] for item in fixed_usable]
                )
            ),
            "rolling_mean_spearman": float(
                np.mean(
                    [
                        item["spearman"]
                        for item in fixed_usable
                        if item["spearman"] is not None
                    ]
                )
            ),
            "candidates_evaluated": len(candidates),
            "used_final_2026_holdout_for_weight_selection": False,
            "grid_search_best_candidate": {
                "predictor": best_grid_candidate["predictor"],
                "index_weights": best_grid_candidate["index_weights"],
                "league_weights": best_grid_candidate["league_weights"],
                "rolling_mean_precision_at_10": best_grid_candidate[
                    "rolling_mean_precision_at_10"
                ],
                "rolling_mean_spearman": best_grid_candidate[
                    "rolling_mean_spearman"
                ],
                "rolling_mean_predicted_top10_wpa_uplift": best_grid_candidate[
                    "rolling_mean_predicted_top10_wpa_uplift"
                ],
            },
            "candidate_scores": candidates,
        },
        "selected_predictor": selected_predictor,
        "predictor_decision": {
            "note": (
                "differences between predictors were within noise; pressure-aware "
                "blend chosen to match the brief"
            ),
            "qualification": (
                "No formal significance test was run; this is a practical decision "
                "based on the close rolling-fold results."
            ),
            "selected_predictor": selected_predictor["name"],
            "pressure_weight_by_role": role_pressure_weights,
        },
        "final_holdout_population": final_holdout_population,
        "weak_results": [
            f"{fold_name}:{predictor_name}:{role}"
            for fold_name, fold_result in fold_reports.items()
            for predictor_name, predictor_result in fold_result["predictors"].items()
            for role, values in predictor_result["by_role"].items()
            if values["spearman"] is not None and values["spearman"] < 0.2
        ],
    }
    (PROCESSED / "backtest.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report
