from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "backend" / "data" / "processed"
CONFIG_PATH = ROOT / "config.yaml"


def build_leverage(data_version: str) -> pd.DataFrame:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    quantile = float(config.get("pressure", {}).get("high_leverage_quantile", 0.85))
    if not 0 <= quantile <= 1:
        raise ValueError(f"Invalid high_leverage_quantile: {quantile}")

    state = pd.read_parquet(PROCESSED / "win_probability.parquet")
    bucket_keys = [
        "match_id",
        "innings",
        "legal_balls_before",
        "wickets_before",
    ]
    bucket = state.groupby(bucket_keys, as_index=False)["abs_wpa"].mean()
    global_mean = float(bucket["abs_wpa"].mean())
    if global_mean == 0:
        bucket["li"] = 0.0
    else:
        bucket["li"] = bucket["abs_wpa"] / global_mean
    threshold = float(bucket["li"].quantile(quantile))
    bucket["high_leverage"] = bucket["li"].ge(threshold)
    state = state.merge(
        bucket[bucket_keys + ["li", "high_leverage"]],
        on=bucket_keys,
        how="left",
        validate="many_to_one",
    )
    state["high_leverage"] = state["high_leverage"].fillna(False).astype(bool)
    state["data_version"] = data_version
    state.to_parquet(PROCESSED / "leverage.parquet", index=False)
    return state
