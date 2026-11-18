from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "backend" / "data" / "processed"
CHARTS = PROCESSED / "charts"


def _top10_intervals(selection: dict) -> None:
    items = selection["combined_top10"]
    labels = [f"{item['rank']}. {item['display_name']} ({item['selection_slot']})" for item in items]
    centers = []
    lower_errors = []
    upper_errors = []
    for item in items:
        interval = item.get("wpa_80_interval")
        if interval is None:
            center = float(item["selected_predictor_wpa_per_ball"])
            lower = upper = center
        else:
            lower, upper = map(float, interval)
            center = (lower + upper) / 2.0
        centers.append(center)
        lower_errors.append(max(0.0, center - lower))
        upper_errors.append(max(0.0, upper - center))

    fig, axis = plt.subplots(figsize=(11, 6.5))
    y = np.arange(len(items))
    axis.errorbar(
        centers,
        y,
        xerr=np.array([lower_errors, upper_errors]),
        fmt="o",
        capsize=3,
        color="#285f8f",
    )
    axis.axvline(0, color="#666666", linewidth=0.8)
    axis.set_yticks(y, labels)
    axis.invert_yaxis()
    axis.set_xlabel("WPA per pressure ball (80% interval)")
    axis.set_title("Selected Top 10: role-specific intervals")
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(CHARTS / "top10_intervals.png", dpi=180)
    plt.close(fig)


def _backtest_summary(backtest: dict) -> None:
    predictor = backtest["selected_predictor"]
    predictor_key = (
        predictor["name"]
        if predictor["name"] != "blend"
        else f"blend_pressure_{predictor['pressure_weight']:.2f}"
    )
    folds = list(backtest["folds"])
    short_names = [
        "2024",
        "2025-26",
        "Final 2026",
    ]
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True)
    for row_index, role in enumerate(("combined", "batting", "bowling")):
        precision_axis, spearman_axis = axes[row_index]
        precisions = []
        baselines = []
        correlations = []
        for fold in folds:
            fold_predictor_key = (
                predictor["name"]
                if fold == "final_train_2022_25_test_2026"
                else predictor_key
            )
            values = backtest["folds"][fold]["predictors"][fold_predictor_key][
                "by_role"
            ][role]
            precisions.append(
                np.nan
                if values["precision_at_10"] is None
                else values["precision_at_10"]
            )
            baselines.append(
                np.nan
                if values["random_baseline_precision_at_10"] is None
                else values["random_baseline_precision_at_10"]
            )
            correlations.append(
                np.nan if values["spearman"] is None else values["spearman"]
            )
        x = np.arange(len(folds))
        precision_axis.bar(x, precisions, color="#397aa8", width=0.65)
        precision_axis.plot(
            x, baselines, color="#b85243", marker="o", linestyle="--", label="random"
        )
        precision_axis.set_ylabel(f"{role.title()}\nPrecision@10")
        precision_axis.set_ylim(0, 1)
        precision_axis.grid(axis="y", alpha=0.2)
        if row_index == 0:
            precision_axis.legend(frameon=False)
        spearman_axis.bar(x, correlations, color="#7a9e60", width=0.65)
        spearman_axis.axhline(0, color="#666666", linewidth=0.8)
        spearman_axis.set_ylabel("Spearman")
        spearman_axis.grid(axis="y", alpha=0.2)
        precision_axis.set_xticks(x, short_names)
        spearman_axis.set_xticks(x, short_names)
    fig.suptitle(
        f"Out-of-time backtest: {predictor['name']} predictor",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(CHARTS / "backtest_summary.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _calibration(report: dict) -> None:
    fig, axis = plt.subplots(figsize=(7, 7))
    colors = {"1": "#397aa8", "2": "#d27a3c"}
    for innings, values in sorted(report["by_innings"].items(), key=lambda item: int(item[0])):
        bins = [
            item
            for item in values["calibration_10_bins"]
            if item["mean_predicted"] is not None
            and item["mean_observed"] is not None
        ]
        if not bins:
            continue
        axis.scatter(
            [item["mean_predicted"] for item in bins],
            [item["mean_observed"] for item in bins],
            s=[max(18, min(160, item["count"] / 150)) for item in bins],
            label=f"Innings {innings}",
            color=colors.get(innings),
            alpha=0.8,
        )
    axis.plot([0, 1], [0, 1], color="#555555", linestyle="--", linewidth=1)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_xlabel("Mean predicted win probability")
    axis.set_ylabel("Mean observed win rate")
    axis.set_title(
        f"Win probability calibration (max gap {report['calibration_max_abs_gap']:.3f})"
    )
    axis.legend(frameon=False)
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(CHARTS / "calibration.png", dpi=180)
    plt.close(fig)


def _flag_shares(quality: dict) -> None:
    flags = quality["pressure_flags"]
    groups = sorted(flags["by_comp_group"])
    columns = list(flags["overall"])
    frame = pd.DataFrame(
        {
            group: [flags["by_comp_group"][group][column] for column in columns]
            for group in groups
        },
        index=columns,
    )
    axis = frame.plot(kind="bar", figsize=(10, 5.5), width=0.8)
    axis.set_ylabel("Share of deliveries")
    axis.set_xlabel("Pressure flag")
    axis.set_title("Pressure flag shares by competition group")
    axis.legend(title="Competition group", frameon=False)
    axis.grid(axis="y", alpha=0.2)
    axis.tick_params(axis="x", rotation=25)
    axis.figure.tight_layout()
    axis.figure.savefig(CHARTS / "flag_shares.png", dpi=180)
    plt.close(axis.figure)


def build_charts() -> list[str]:
    CHARTS.mkdir(parents=True, exist_ok=True)
    selection_report = json.loads(
        (PROCESSED / "top10.json").read_text(encoding="utf-8")
    )
    backtest = json.loads((PROCESSED / "backtest.json").read_text(encoding="utf-8"))
    calibration_report = json.loads(
        (PROCESSED / "win_probability_report.json").read_text(encoding="utf-8")
    )
    quality = json.loads((PROCESSED / "data_quality.json").read_text(encoding="utf-8"))
    _top10_intervals(selection_report)
    _backtest_summary(backtest)
    _calibration(calibration_report)
    _flag_shares(quality)
    return [
        str(CHARTS / name)
        for name in (
            "top10_intervals.png",
            "backtest_summary.png",
            "calibration.png",
            "flag_shares.png",
        )
    ]
