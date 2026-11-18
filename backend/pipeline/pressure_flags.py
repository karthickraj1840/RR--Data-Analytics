from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "backend" / "data" / "processed"
CONFIG_PATH = ROOT / "config.yaml"
FLAG_COLUMNS = [
    "high_leverage",
    "hard_chase",
    "death_overs",
    "collapse",
    "tight_finish",
    "knockout",
]


def build_pressure_flags(data_version: str) -> pd.DataFrame:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    pressure = config.get("pressure", {})
    state = pd.read_parquet(PROCESSED / "leverage.parquet")

    death_overs = pressure.get("death_overs", [16, 20])
    if not isinstance(death_overs, (list, tuple)) or len(death_overs) != 2:
        raise ValueError("pressure.death_overs must be a two-value [start, end] range")
    death_start, death_end = map(int, death_overs)
    if death_start > death_end:
        raise ValueError("pressure.death_overs start must not exceed its end")

    state["hard_chase"] = (
        state["innings"].eq(2)
        & state["rrr"].ge(float(pressure.get("hard_chase_rrr", 10.0)))
        & state["balls_left"].between(
            1, int(pressure.get("hard_chase_balls_left_max", 60))
        )
    )
    state["death_overs"] = state["overid"].ge(death_start)
    if death_end:
        state["death_overs"] &= state["overid"].le(death_end)
    state["collapse"] = (
        state["wickets_before"].ge(int(pressure.get("collapse_wickets", 3)))
        & state["overid"].le(int(pressure.get("collapse_over", 10)))
    )
    state["tight_finish"] = (
        state["innings"].eq(2)
        & state["balls_left"].le(
            int(pressure.get("tight_finish_balls_left_max", 30))
        )
        & state["wp_before"].between(
            float(pressure.get("tight_finish_wp_min", 0.25)),
            float(pressure.get("tight_finish_wp_max", 0.75)),
        )
    )
    knockout_signal = int(pressure.get("knockout_signal", 1))
    state["knockout"] = state["knock_out"].fillna(0).eq(knockout_signal)
    state[FLAG_COLUMNS] = state[FLAG_COLUMNS].fillna(False).astype(bool)
    state["pressure_flags"] = state[FLAG_COLUMNS].astype(int).sum(axis=1)
    state["data_version"] = data_version
    state.to_parquet(PROCESSED / "pressure_flags.parquet", index=False)

    overall = {
        flag: float(state[flag].mean()) for flag in FLAG_COLUMNS
    }
    by_comp_group = {}
    for group, group_state in state.groupby("comp_group", sort=True):
        by_comp_group[str(group)] = {
            flag: float(group_state[flag].mean()) for flag in FLAG_COLUMNS
        }
    quality_path = PROCESSED / "data_quality.json"
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    quality["pressure_flags"] = {
        "n_balls": int(len(state)),
        "overall": overall,
        "by_comp_group": by_comp_group,
    }
    quality_path.write_text(json.dumps(quality, indent=2), encoding="utf-8")
    return state
