from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = ROOT / "backend"
RAW_DIR = ROOT / "6af458274c1e8_data_pack" / "Data Pack"
ZIP_PATH = ROOT / "6af458274c1e8_data_pack.zip"
PROCESSED = BACKEND_ROOT / "data" / "processed"


def ensure_data_available() -> Path:
    if RAW_DIR.exists():
        return RAW_DIR
    if ZIP_PATH.exists():
        with zipfile.ZipFile(ZIP_PATH, "r") as archive:
            archive.extractall(ROOT)
        if RAW_DIR.exists():
            return RAW_DIR
    raise FileNotFoundError("Raw dataset is missing from the workspace")


def _competition_group(name: object) -> str:
    normalized = str(name).strip().lower()
    if "ipl" in normalized:
        return "IPL"
    if "emerging" in normalized or "rising stars" in normalized:
        return "Emerging"
    return "International"


def _record_exclusion(
    quality: dict,
    rule: str,
    before: pd.DataFrame,
    after: pd.DataFrame,
    before_match_ids: set,
    after_match_ids: set,
) -> None:
    removed = before.loc[~before.index.isin(after.index)]
    quality["exclusions"][rule] = {
        "rows_dropped": int(len(removed)),
        "balls_dropped": int(removed["is_legal"].sum()),
        "matches_dropped": int(len(before_match_ids - after_match_ids)),
    }


def load_clean_data(data_version: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    raw_dir = ensure_data_available()
    bbb = pd.read_csv(raw_dir / "ball_by_ball.csv", low_memory=False)
    matches = pd.read_csv(raw_dir / "matches.csv", low_memory=False)
    competitions = pd.read_csv(raw_dir / "competitions.csv", low_memory=False)
    playing_eleven = pd.read_csv(raw_dir / "playing_eleven.csv", low_memory=False)

    required_match_columns = {
        "match_id",
        "comp_id",
        "win_team_id",
        "revised_target",
        "first_inn_revised_overs",
        "second_inn_revised_overs",
    }
    missing_columns = required_match_columns - set(matches.columns)
    if missing_columns:
        raise ValueError(f"matches.csv is missing required columns: {sorted(missing_columns)}")
    if matches["match_id"].isna().any() or matches["match_id"].duplicated().any():
        raise ValueError("matches.csv must contain unique, non-null match_id values")
    if bbb["match_id"].isna().any():
        raise ValueError("ball_by_ball.csv contains null match_id values")
    if not bbb["match_id"].isin(matches["match_id"]).all():
        raise ValueError("ball_by_ball.csv contains match_id values absent from matches.csv")

    bbb = bbb.sort_values(
        ["match_id", "innings", "innings_ball_num"], kind="stable"
    ).copy()
    duplicate_count = int(
        bbb.duplicated(["match_id", "innings", "innings_ball_num"]).sum()
    )
    if duplicate_count:
        raise ValueError(f"Duplicate ball keys remain: {duplicate_count}")

    for column in ["runs", "extras1_runs", "extras_2_runs"]:
        bbb[column] = pd.to_numeric(bbb[column], errors="coerce").fillna(0)
    bbb["total_runs"] = bbb["runs"] + bbb["extras1_runs"] + bbb["extras_2_runs"]
    illegal_types = {"wide", "overstep", "other", "penalty"}
    bbb["is_legal"] = ~(
        bbb["extras_type"].fillna("").astype(str).str.strip().str.lower().isin(illegal_types)
    )
    bbb["is_wicket"] = bbb["is_wicket"].fillna(False).astype(bool)

    quality: dict = {
        "data_version": data_version,
        "duplicate_keys": duplicate_count,
        "matches_raw": int(len(matches)),
        "matches_used": 0,
        "balls_used": 0,
        "rows_dropped": 0,
        "exclusions": {},
    }
    active_match_ids = set(matches["match_id"])

    before = bbb
    before_ids = active_match_ids
    bbb = bbb.loc[bbb["innings"].isin([1, 2])].copy()
    active_match_ids = set(bbb["match_id"])
    matches = matches.loc[matches["match_id"].isin(active_match_ids)].copy()
    _record_exclusion(quality, "innings_1_and_2_only", before, bbb, before_ids, active_match_ids)

    def exclude_matches(rule: str, excluded_ids: set) -> None:
        nonlocal bbb, matches, active_match_ids
        before_rows = bbb
        before_ids = active_match_ids
        active_match_ids = before_ids - excluded_ids
        bbb = bbb.loc[bbb["match_id"].isin(active_match_ids)].copy()
        matches = matches.loc[matches["match_id"].isin(active_match_ids)].copy()
        _record_exclusion(
            quality, rule, before_rows, bbb, before_ids, active_match_ids
        )

    exclude_matches(
        "null_win_team_id",
        set(matches.loc[matches["win_team_id"].isna(), "match_id"]),
    )
    revised_columns = [
        "revised_target",
        "first_inn_revised_overs",
        "second_inn_revised_overs",
    ]
    exclude_matches(
        "revised_target_or_overs",
        set(matches.loc[matches[revised_columns].notna().any(axis=1), "match_id"]),
    )

    first_innings = bbb.loc[bbb["innings"].eq(1)].groupby("match_id").agg(
        legal_balls=("is_legal", "sum"),
        wickets=("is_wicket", "sum"),
    )
    too_short = first_innings.index[
        first_innings["legal_balls"].lt(114) & first_innings["wickets"].lt(10)
    ]
    exclude_matches("short_first_innings", set(too_short))

    innings_counts = bbb.groupby("match_id")["innings"].agg(
        lambda values: set(values.tolist())
    )
    missing_innings = set(
        innings_counts.index[
            ~innings_counts.map(lambda values: {1, 2}.issubset(values))
        ]
    )
    exclude_matches("missing_either_innings", missing_innings)

    matches = matches.loc[matches["match_id"].isin(active_match_ids)].copy()
    competition_names = competitions.set_index("comp_id")["comp_name"]
    competition_seasons = competitions.set_index("comp_id")["season"]
    matches["comp_name"] = matches["comp_id"].map(competition_names)
    if matches["comp_name"].isna().any():
        missing_comp_ids = sorted(matches.loc[matches["comp_name"].isna(), "comp_id"].unique())
        raise ValueError(f"Used matches reference unknown comp_id values: {missing_comp_ids}")
    matches["season"] = matches["comp_id"].map(competition_seasons)
    if matches["season"].isna().any():
        raise ValueError("Used matches are missing competition seasons")
    matches["comp_group"] = matches["comp_name"].map(_competition_group)

    used_ids = set(matches["match_id"])
    playing_eleven = playing_eleven.loc[
        playing_eleven["match_id"].isin(used_ids)
    ].copy()
    if "player_id" not in playing_eleven.columns:
        raise ValueError("playing_eleven.csv is missing player_id")

    matches_used_by_group = (
        matches.groupby("comp_group")["match_id"].nunique().sort_index().astype(int).to_dict()
    )
    matches_used_by_season = (
        matches.groupby("season")["match_id"].nunique().sort_index().astype(int).to_dict()
        if "season" in matches.columns
        else {}
    )
    quality.update(
        {
            "matches_used": int(matches["match_id"].nunique()),
            "balls_used": int(bbb["is_legal"].sum()),
            "matches_used_by_comp_group": matches_used_by_group,
            "matches_used_by_season": {
                str(season): count for season, count in matches_used_by_season.items()
            },
            "checks": {
                "legal_ball_count": int(bbb["is_legal"].sum()),
                "total_rows": int(len(bbb)),
                "matches": int(matches["match_id"].nunique()),
            },
            "frame_shape": {"ball_by_ball": list(bbb.shape)},
        }
    )
    quality["rows_dropped"] = sum(
        item["rows_dropped"] for item in quality["exclusions"].values()
    )
    if quality["matches_used"] < 500:
        raise AssertionError(
            f"Expected at least 500 cleaned matches; got {quality['matches_used']}"
        )

    for frame in (bbb, matches, competitions, playing_eleven):
        frame["data_version"] = data_version

    bbb.to_parquet(PROCESSED / "ball_by_ball.parquet", index=False)
    matches.to_parquet(PROCESSED / "matches.parquet", index=False)
    competitions.to_parquet(PROCESSED / "competitions.parquet", index=False)
    playing_eleven.to_parquet(PROCESSED / "playing_eleven.parquet", index=False)
    (PROCESSED / "data_quality.json").write_text(
        json.dumps(quality, indent=2), encoding="utf-8"
    )
    return bbb, matches, competitions
