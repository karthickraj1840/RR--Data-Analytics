from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "backend" / "data" / "processed"
FEATURES = [
    "runs_before",
    "wickets_before",
    "legal_balls_before",
    "runs_needed",
    "balls_left",
    "crr",
    "rrr",
]
MONOTONIC_CONSTRAINTS = [1, -1, -1, -1, 1, 1, -1]


def _calibration_bins(labels: np.ndarray, probabilities: np.ndarray) -> list[dict]:
    indices = np.minimum((probabilities * 10).astype(int), 9)
    bins = []
    for bin_index in range(10):
        selected = indices == bin_index
        bins.append(
            {
                "bin": bin_index + 1,
                "count": int(selected.sum()),
                "mean_predicted": (
                    float(probabilities[selected].mean()) if selected.any() else None
                ),
                "mean_observed": (
                    float(labels[selected].mean()) if selected.any() else None
                ),
            }
        )
    return bins


def _train_innings(
    state: pd.DataFrame, innings: int, data_version: str
) -> tuple[np.ndarray, dict]:
    subset = state.loc[state["innings"].eq(innings)].copy()
    if subset["match_id"].nunique() < 5:
        raise ValueError(f"Innings {innings} has fewer than five matches for GroupKFold")
    X = subset[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = subset["batting_side_win"].astype(int)
    if y.nunique() != 2:
        raise ValueError(f"Innings {innings} training labels must contain both outcomes")

    probabilities = np.full(len(subset), np.nan, dtype=float)
    fold_ids = np.full(len(subset), -1, dtype=int)
    splitter = GroupKFold(n_splits=5)
    for fold, (train_indices, test_indices) in enumerate(
        splitter.split(X, y, groups=subset["match_id"])
    ):
        if y.iloc[train_indices].nunique() != 2:
            raise ValueError(f"Innings {innings}, fold {fold + 1} has one training class")
        model = HistGradientBoostingClassifier(
            learning_rate=0.06,
            max_iter=150,
            max_leaf_nodes=15,
            min_samples_leaf=40,
            monotonic_cst=MONOTONIC_CONSTRAINTS,
            early_stopping=False,
            random_state=42,
        )
        model.fit(X.iloc[train_indices], y.iloc[train_indices])
        probabilities[test_indices] = model.predict_proba(X.iloc[test_indices])[:, 1]
        fold_ids[test_indices] = fold

    if np.isnan(probabilities).any() or (fold_ids < 0).any():
        raise AssertionError(f"Innings {innings} has missing out-of-fold predictions")
    subset["wp_before"] = probabilities
    subset["oof_fold"] = fold_ids
    subset["data_version"] = data_version

    labels = y.to_numpy(dtype=int)
    metrics = {
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
        "brier": float(brier_score_loss(labels, probabilities)),
        "baselines": {"log_loss": 0.693, "brier": 0.25},
        "n_balls": int(len(subset)),
        "n_matches": int(subset["match_id"].nunique()),
        "mean_wp_before": float(probabilities.mean()),
        "mean_match_label": float(labels.mean()),
        "calibration_10_bins": _calibration_bins(labels, probabilities),
    }
    calibration_gaps = [
        abs(item["mean_predicted"] - item["mean_observed"])
        for item in metrics["calibration_10_bins"]
        if item["mean_predicted"] is not None and item["mean_observed"] is not None
    ]
    metrics["calibration_max_abs_gap"] = (
        float(max(calibration_gaps)) if calibration_gaps else 0.0
    )
    return subset["wp_before"].to_numpy(), metrics | {"_folds": fold_ids}


def build_win_probability(data_version: str) -> dict:
    state = pd.read_parquet(PROCESSED / "match_state.parquet")
    output = state.copy()
    report: dict = {"data_version": data_version, "by_innings": {}}
    for innings in (1, 2):
        subset = state.loc[state["innings"].eq(innings)].copy()
        probabilities, metrics = _train_innings(subset, innings, data_version)
        output.loc[subset.index, "wp_before"] = probabilities
        output.loc[subset.index, "oof_fold"] = metrics.pop("_folds")
        report["by_innings"][str(innings)] = metrics

    if output["wp_before"].isna().any():
        raise AssertionError("Every cleaned delivery must have an OOF win probability")
    next_probability = output.groupby(["match_id", "innings"], sort=False)[
        "wp_before"
    ].shift(-1)
    output["wp_after"] = next_probability
    final_rows = output.groupby(["match_id", "innings"], sort=False).tail(1)
    innings_two_starts = (
        output.loc[output["innings"].eq(2)]
        .groupby("match_id", sort=False)
        .head(1)
        .set_index("match_id")["wp_before"]
    )
    innings_one_ends = final_rows.loc[final_rows["innings"].eq(1)]
    output.loc[innings_one_ends.index, "wp_after"] = (
        1.0 - innings_one_ends["match_id"].map(innings_two_starts)
    )
    innings_two_ends = final_rows.loc[final_rows["innings"].eq(2)]
    output.loc[innings_two_ends.index, "wp_after"] = innings_two_ends[
        "batting_side_win"
    ].astype(float)

    output["wpa"] = output["wp_after"] - output["wp_before"]
    output["match_wpa"] = np.where(
        output["innings"].eq(1), output["wpa"], -output["wpa"]
    )
    output["abs_wpa"] = output["wpa"].abs()
    output["data_version"] = data_version

    first_innings_starts = (
        output.loc[output["innings"].eq(1)]
        .groupby("match_id", sort=False)
        .head(1)
        .set_index("match_id")
    )
    summed_wpa = output.groupby("match_id")["match_wpa"].sum()
    expected_change = (
        first_innings_starts["batting_side_win"].astype(float)
        - first_innings_starts["wp_before"]
    )
    errors = summed_wpa.reindex(expected_change.index) - expected_change
    max_wpa_error = float(errors.abs().max())
    if max_wpa_error > 1e-6:
        raise AssertionError(f"Per-match WPA sum error {max_wpa_error:.12g} exceeds 1e-6")

    per_innings = report["by_innings"]
    report["max_wpa_sum_error"] = max_wpa_error
    report["log_loss"] = float(np.mean([item["log_loss"] for item in per_innings.values()]))
    report["coin_flip_baseline"] = 0.693
    report["brier_baseline"] = 0.25
    report["win_probability_mean"] = float(output["wp_before"].mean())
    report["wpa_mean_abs"] = float(output["abs_wpa"].mean())
    report["calibration_max_abs_gap"] = max(
        values["calibration_max_abs_gap"] for values in per_innings.values()
    )
    innings_two_first_balls = (
        output.loc[output["innings"].eq(2)]
        .sort_values(["match_id", "innings_ball_num"], kind="stable")
        .groupby("match_id", sort=False)
        .head(1)
    )
    if innings_two_first_balls["match_id"].nunique() != output["match_id"].nunique():
        raise AssertionError("Every cleaned match must have a first ball of innings 2")
    chase_win_rate = float(innings_two_first_balls["batting_side_win"].mean())
    first_ball_mean_wp = float(innings_two_first_balls["wp_before"].mean())
    report["match_level_chase_win_rate"] = chase_win_rate
    report["mean_wp_before_first_ball_innings_2"] = first_ball_mean_wp
    report["innings_2_first_ball"] = {
        "n_matches": int(innings_two_first_balls["match_id"].nunique()),
        "match_level_chase_win_rate": chase_win_rate,
        "mean_wp_before": first_ball_mean_wp,
    }
    if any(len(metrics["calibration_10_bins"]) != 10 for metrics in per_innings.values()):
        raise AssertionError("Each innings requires a 10-bin calibration table")

    output.to_parquet(PROCESSED / "win_probability.parquet", index=False)
    (PROCESSED / "win_probability_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    quality_path = PROCESSED / "data_quality.json"
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    quality["win_probability"] = report
    quality["max_wpa_sum_error"] = max_wpa_error
    quality_path.write_text(json.dumps(quality, indent=2), encoding="utf-8")
    return report
