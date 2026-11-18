from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "backend" / "data" / "processed"


def build_match_state(data_version: str) -> pd.DataFrame:
    balls = pd.read_parquet(PROCESSED / "ball_by_ball.parquet")
    matches = pd.read_parquet(PROCESSED / "matches.parquet")
    match_columns = [
        "match_id",
        "win_team_id",
        "knock_out",
        "comp_id",
        "comp_group",
    ]
    state = balls.merge(
        matches[match_columns],
        on="match_id",
        how="inner",
        validate="many_to_one",
    )
    state = state.sort_values(
        ["match_id", "innings", "innings_ball_num"], kind="stable"
    ).copy()
    group_keys = ["match_id", "innings"]

    def before_ball(values: pd.Series) -> pd.Series:
        return values.cumsum().shift(fill_value=0)

    state["runs_before"] = state.groupby(group_keys, sort=False)["total_runs"].transform(
        before_ball
    )
    state["wickets_before"] = state.groupby(group_keys, sort=False)["is_wicket"].transform(
        lambda values: before_ball(values.astype(int))
    )
    state["legal_balls_before"] = state.groupby(group_keys, sort=False)["is_legal"].transform(
        lambda values: before_ball(values.astype(int))
    )
    state["balls_left"] = (120 - state["legal_balls_before"]).clip(lower=0)

    first_innings_totals = (
        state.loc[state["innings"].eq(1)]
        .groupby("match_id")["total_runs"]
        .sum()
    )
    state["first_innings_total"] = state["match_id"].map(first_innings_totals)
    state["target"] = np.where(
        state["innings"].eq(2), state["first_innings_total"] + 1, np.nan
    )
    state["runs_needed"] = np.where(
        state["innings"].eq(2), state["target"] - state["runs_before"], np.nan
    )

    legal_balls = state["legal_balls_before"].to_numpy(dtype=float)
    runs_before = state["runs_before"].to_numpy(dtype=float)
    balls_left = state["balls_left"].to_numpy(dtype=float)
    runs_needed = state["runs_needed"].to_numpy(dtype=float)
    state["crr"] = np.divide(
        runs_before * 6,
        legal_balls,
        out=np.zeros(len(state), dtype=float),
        where=legal_balls > 0,
    )
    state["rr"] = state["crr"]
    state["rrr"] = np.divide(
        runs_needed * 6,
        balls_left,
        out=np.zeros(len(state), dtype=float),
        where=(state["innings"].to_numpy() == 2) & (balls_left > 0),
    )
    state["batting_side_win"] = (
        state["batting_team_id"].eq(state["win_team_id"]).astype(int)
    )
    state["data_version"] = data_version
    state.to_parquet(PROCESSED / "match_state.parquet", index=False)
    return state
