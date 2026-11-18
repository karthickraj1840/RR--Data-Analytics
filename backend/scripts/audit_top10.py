from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

BACKEND_ROOT = Path(__file__).resolve().parents[1]
BACKEND_PROCESSED = BACKEND_ROOT / "data" / "processed"
ROOT_PROCESSED = BACKEND_ROOT.parent / "data" / "processed"

TOP10_COLUMNS = {
    "rank",
    "designation",
    "player_id",
    "pressure_balls",
    "ipl_share",
    "role",
    "grade",
}
METRICS_COLUMNS = {"player_id"}
EXPECTED_GRADE_COUNTS = {"A": 3, "B": 1, "C": 6}


def _report(condition: bool, description: str, detail: str = "") -> bool:
    result = "PASS" if condition else "FAIL"
    suffix = f" ({detail})" if detail else ""
    print(f"{result}: {description}{suffix}")
    return condition


def _counts(frame: pd.DataFrame, column: str) -> str:
    counts = frame[column].value_counts(dropna=False).sort_index()
    return ", ".join(
        f"{'<missing>' if pd.isna(value) else value}: {count}"
        for value, count in counts.items()
    )


def _numeric_ids(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce")


def main() -> int:
    processed = BACKEND_PROCESSED
    if not (processed / "top10.csv").is_file():
        processed = ROOT_PROCESSED

    print(f"Data directory used: {processed}")
    top10_path = processed / "top10.csv"
    metrics_path = processed / "player_metrics.parquet"
    selection_path = processed / "top10.json"
    passed = True

    top10_exists = top10_path.is_file()
    passed &= _report(
        top10_exists,
        "top10.csv input",
        "found" if top10_exists else "not found",
    )
    metrics_exists = metrics_path.is_file()
    passed &= _report(
        metrics_exists,
        "player_metrics.parquet input",
        "found" if metrics_exists else "not found",
    )
    selection_exists = selection_path.is_file()
    passed &= _report(
        selection_exists,
        "top10.json input",
        "found" if selection_exists else "not found",
    )

    top10 = pd.read_csv(top10_path) if top10_exists else None
    metrics = pd.read_parquet(metrics_path) if metrics_exists else None
    selection = (
        json.loads(selection_path.read_text(encoding="utf-8"))
        if selection_exists
        else None
    )

    missing_top10 = TOP10_COLUMNS - set(top10.columns) if top10 is not None else set()
    missing_metrics = METRICS_COLUMNS - set(metrics.columns) if metrics is not None else set()
    if missing_top10:
        print(f"Missing columns in top10.csv: {', '.join(sorted(missing_top10))}")
    if missing_metrics:
        print(
            "Missing columns in player_metrics.parquet: "
            f"{', '.join(sorted(missing_metrics))}"
        )

    passed &= _report(
        top10 is not None and len(top10) == 20,
        "exactly 20 rows",
        f"found {len(top10)}" if top10 is not None else "CSV unavailable",
    )

    ids_available = top10 is not None and "player_id" in top10.columns
    if ids_available:
        all_ids = _numeric_ids(top10["player_id"])
        unique_ids = int(all_ids.nunique(dropna=True))
        ids_are_valid = bool(all_ids.notna().all())
        passed &= _report(
            ids_are_valid and unique_ids == 20,
            "20 unique player ids",
            f"found {unique_ids} unique non-empty ids",
        )
    else:
        passed &= _report(False, "20 unique player ids", "player_id unavailable")

    ranks_available = top10 is not None and "rank" in top10.columns
    picks = None
    if ranks_available:
        numeric_ranks = pd.to_numeric(top10["rank"], errors="coerce")
        ranked = top10.assign(_audit_rank=numeric_ranks).sort_values(
            "_audit_rank", kind="stable", na_position="last"
        )
        picks = ranked.loc[ranked["_audit_rank"].between(1, 10)]
        valid_rank_sequence = bool(
            numeric_ranks.notna().all()
            and sorted(numeric_ranks.tolist()) == list(range(1, 21))
        )
        designation_matches = bool(
            "designation" in ranked.columns
            and ranked.loc[ranked["_audit_rank"].between(1, 10), "designation"]
            .eq("pick")
            .all()
            and ranked.loc[ranked["_audit_rank"].between(11, 20), "designation"]
            .eq("reserve")
            .all()
        )
        passed &= _report(
            valid_rank_sequence and designation_matches,
            "ranks 1-10 are picks and ranks 11-20 are reserves",
        )
    else:
        passed &= _report(
            False, "ranks 1-10 are picks and ranks 11-20 are reserves", "rank unavailable"
        )

    if picks is not None:
        for column, threshold, description in (
            ("pressure_balls", 80, "Top-10 all pressure_balls >= 80"),
            ("ipl_share", 0.30, "Top-10 all ipl_share >= 0.30"),
        ):
            if column in picks.columns and len(picks) == 10:
                values = pd.to_numeric(picks[column], errors="coerce")
                valid = bool(values.notna().all() and values.ge(threshold).all())
                passed &= _report(valid, description)
            else:
                passed &= _report(False, description, f"{column} unavailable or rank set invalid")

        if "role" in picks.columns:
            print(f"Top-10 role counts (ranks 1-10): {_counts(picks, 'role')}")
        else:
            print("Top-10 role counts (ranks 1-10): unavailable (missing role)")

        if "grade" in picks.columns:
            grade_counts = {
                grade: int(picks["grade"].eq(grade).sum())
                for grade in ("A", "B", "C")
            }
            print(
                "Top-10 grade counts (ranks 1-10): "
                + ", ".join(f"{grade}: {count}" for grade, count in grade_counts.items())
            )
            passed &= _report(
                grade_counts == EXPECTED_GRADE_COUNTS,
                "Top-10 grade counts match expected A:3, B:1, C:6",
            )
        else:
            print("Top-10 grade counts (ranks 1-10): unavailable (missing grade)")
            passed &= _report(
                False, "Top-10 grade counts match expected A:3, B:1, C:6"
            )
    else:
        passed &= _report(False, "Top-10 eligibility checks", "rank unavailable")
        print("Top-10 role counts (ranks 1-10): unavailable")
        print("Top-10 grade counts (ranks 1-10): unavailable")

    if (
        picks is not None
        and metrics is not None
        and "player_id" in picks.columns
        and "player_id" in metrics.columns
        and len(picks) == 10
    ):
        top_ids = _numeric_ids(picks["player_id"])
        metric_ids = _numeric_ids(metrics["player_id"])
        ids_exist = bool(
            top_ids.notna().all()
            and set(top_ids).issubset(set(metric_ids.dropna()))
        )
        passed &= _report(ids_exist, "all ranks 1-10 ids exist in player_metrics")
    else:
        passed &= _report(
            False,
            "all ranks 1-10 ids exist in player_metrics",
            "required input, column, or rank set unavailable",
        )

    if picks is not None and selection is not None and "player_id" in picks.columns:
        combined_top10 = selection.get("combined_top10")
        csv_ids = _numeric_ids(picks["player_id"])
        json_ids = None
        if isinstance(combined_top10, list):
            if all(isinstance(item, dict) for item in combined_top10):
                if all("player_id" in item and "rank" in item for item in combined_top10):
                    json_rows = sorted(combined_top10, key=lambda item: item["rank"])
                    json_ids = pd.to_numeric(
                        pd.Series([item["player_id"] for item in json_rows]),
                        errors="coerce",
                    )
            elif all(not isinstance(item, dict) for item in combined_top10):
                json_ids = pd.to_numeric(pd.Series(combined_top10), errors="coerce")
        identical = bool(
            json_ids is not None
            and len(json_ids) == 10
            and json_ids.notna().all()
            and csv_ids.notna().all()
            and csv_ids.tolist() == json_ids.tolist()
        )
        passed &= _report(
            identical, "ranks 1-10 ids match combined_top10 in top10.json"
        )
    else:
        passed &= _report(
            False,
            "ranks 1-10 ids match combined_top10 in top10.json",
            "required input or rank set unavailable",
        )

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
