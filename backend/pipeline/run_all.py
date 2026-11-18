from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import build_backtest
from .charts import build_charts
from .leverage import build_leverage
from .load_clean import load_clean_data
from .match_state import build_match_state
from .player_metrics import build_player_metrics
from .pressure_flags import build_pressure_flags
from .selection import build_selection
from .sensitivity import build_sensitivity as build_robustness
from .win_probability import build_win_probability

ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = ROOT / "backend"
CONFIG_PATH = ROOT / "config.yaml"
PROCESSED = BACKEND_ROOT / "data" / "processed"


def _new_data_version() -> str:
    config_bytes = CONFIG_PATH.read_bytes()
    config_hash = hashlib.sha256(config_bytes).hexdigest()
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )
    return f"{timestamp}|sha256:{config_hash}"


def _clear_processed_outputs() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    for path in PROCESSED.iterdir():
        if path.is_file() or path.is_symlink():
            path.unlink()


def _refresh_data_version(data_version: str) -> None:
    for path in PROCESSED.glob("*.parquet"):
        frame = pd.read_parquet(path)
        if "data_version" in frame.columns:
            frame["data_version"] = data_version
            frame.to_parquet(path, index=False)
    for path in PROCESSED.glob("*.json"):
        report = json.loads(path.read_text(encoding="utf-8"))

        def update_version(value: object) -> None:
            if isinstance(value, dict):
                if "data_version" in value:
                    value["data_version"] = data_version
                for nested in value.values():
                    update_version(nested)
            elif isinstance(value, list):
                for nested in value:
                    update_version(nested)

        update_version(report)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def build_similarity(data_version: str) -> pd.DataFrame:
    metrics = pd.read_parquet(PROCESSED / "player_metrics.parquet")
    if metrics.empty:
        empty = pd.DataFrame(columns=["player_id", "similar_player_id", "distance"])
        empty["data_version"] = pd.Series(dtype=str)
        empty.to_parquet(PROCESSED / "similarity.parquet", index=False)
        return empty
    features = metrics[
        ["pressure_index", "wpa_per_ball", "pressure_delta", "sample_size"]
    ].fillna(0)
    standard_deviation = features.std(ddof=0).replace(0, 1)
    scaled = (features - features.mean()) / standard_deviation
    rows = []
    for index, row in metrics.iterrows():
        for other_index, other in metrics.iterrows():
            if row["player_id"] == other["player_id"]:
                continue
            distance = np.linalg.norm(
                scaled.loc[index].to_numpy() - scaled.loc[other_index].to_numpy()
            )
            rows.append(
                {
                    "player_id": int(row["player_id"]),
                    "similar_player_id": int(other["player_id"]),
                    "distance": float(distance),
                    "data_version": data_version,
                }
            )
    similarities = pd.DataFrame(rows)
    if not similarities.empty:
        similarities = similarities.sort_values(
            ["player_id", "distance"], kind="stable"
        ).head(50)
    similarities.to_parquet(PROCESSED / "similarity.parquet", index=False)
    return similarities


def run_pipeline() -> dict:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Pipeline config not found: {CONFIG_PATH}")
    _clear_processed_outputs()
    data_version = _new_data_version()

    balls, matches, competitions = load_clean_data(data_version)
    state = build_match_state(data_version)
    win_report = build_win_probability(data_version)
    build_leverage(data_version)
    build_pressure_flags(data_version)
    backtest = build_backtest(data_version)
    data_version = _new_data_version()
    _refresh_data_version(data_version)
    win_report = json.loads(
        (PROCESSED / "win_probability_report.json").read_text(encoding="utf-8")
    )
    backtest = json.loads((PROCESSED / "backtest.json").read_text(encoding="utf-8"))
    player_metrics = build_player_metrics(data_version)
    sensitivity = build_robustness(data_version)
    selection = build_selection(data_version)
    backtest["selection_comparison"] = selection["top10_comparison"]
    (PROCESSED / "backtest.json").write_text(
        json.dumps(backtest, indent=2), encoding="utf-8"
    )
    build_similarity(data_version)
    charts = build_charts()

    return {
        "data_version": data_version,
        "rows": int(len(balls)),
        "matches": int(len(matches)),
        "competitions": int(len(competitions)),
        "state_rows": int(len(state)),
        "win_report": win_report,
        "backtest": backtest,
        "sensitivity": sensitivity,
        "selection": selection,
        "leaderboard_count": int(len(player_metrics)),
        "charts": charts,
    }


if __name__ == "__main__":
    result = run_pipeline()
    print("Pipeline completed")
    print(f"Data version: {result['data_version']}")
    print(f"Rows used: {result['rows']}")
    print(f"Matches used: {result['matches']}")
    print(f"Competition rows: {result['competitions']}")
    for innings, metrics in sorted(result["win_report"]["by_innings"].items()):
        print(
            f"Innings {innings}: log_loss={metrics['log_loss']:.6f}, "
            f"brier={metrics['brier']:.6f}, "
            f"mean_wp_before={metrics['mean_wp_before']:.6f}, "
            f"calibration_bins={len(metrics['calibration_10_bins'])}"
        )
    print(
        "Maximum 10-bin calibration gap: "
        f"{result['win_report']['calibration_max_abs_gap']:.6f}"
    )
    print(f"Maximum WPA-sum error: {result['win_report']['max_wpa_sum_error']:.12g}")
    selected_predictor = result["selection"]["ranking_predictor"]
    print(
        "Selected predictor: "
        f"{selected_predictor['name']} (pressure weight="
        f"{selected_predictor['pressure_weight']:.2f}; by role="
        f"{result['selection']['pressure_weight_by_role']})"
    )
    print(f"Index weights: {result['selection']['index_weights']}")
    print(f"League weights: {result['selection']['league_weights']}")
    print(
        "Top-20 grade counts: "
        f"{result['selection']['grade_counts']['top20']}"
    )
    top10_pressure = [
        item["pressure_balls"] for item in result["selection"]["combined_top10"]
    ]
    print(
        "Top-10 pressure balls: "
        f"median={float(np.median(top10_pressure)):.1f}, "
        f"minimum={min(top10_pressure)}"
    )
    print(
        "Top-10 role split: "
        f"{result['selection']['top10_role_constraint']['configured_split']}"
    )
    print(
        "Top-10 role labels: "
        + ", ".join(
            f"{item['rank']}={item['role_label']}/{item['selection_slot']}-slot"
            for item in result["selection"]["combined_top10"]
        )
    )
    print(
        "All-rounder split note: "
        f"{result['selection']['top10_role_constraint']['note']}"
    )
    print(
        "Top-10 IPL share range: "
        f"{result['selection']['ipl_share_range']['minimum']:.3f}-"
        f"{result['selection']['ipl_share_range']['maximum']:.3f}"
    )
    comparison = result["selection"]["top10_comparison"]
    print(f"Top-10 overlap with overall_only: {comparison['overlap_count']}/10")
    for label, metrics in comparison["backtest_metrics"].items():
        print(
            f"{label} Top-10: Spearman={metrics['spearman']:.4f} "
            "(predictor-level), "
            f"precision@10={metrics['precision_at_10']:.3f} (list), "
            f"uplift={metrics['predicted_top10_wpa_uplift']:.6f} (list)"
        )
    print(f"Chart exports: {len(result['charts'])}")
