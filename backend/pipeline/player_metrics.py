from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

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
WICKET_DISMISSALS = {
    "bowled",
    "caught",
    "stump",
    "lbw",
    "hit wicket",
}
PHASES = {"PP": (1, 6), "MID": (7, 15), "DEATH": (16, 20)}
WPA_CI_Z = 1.2815515655446004


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    valid = values.notna() & weights.notna() & weights.gt(0)
    if not valid.any():
        return 0.0
    denominator = float(weights.loc[valid].sum())
    return (
        float((values.loc[valid] * weights.loc[valid]).sum() / denominator)
        if denominator
        else 0.0
    )


def _load_config() -> tuple[dict[str, Any], dict[str, float], dict[str, float], int]:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    league_weights = {
        str(group): float(weight)
        for group, weight in config.get(
            "league_weights",
            {"IPL": 1.0, "International": 0.8, "Emerging": 0.4},
        ).items()
    }
    index_weights = {
        str(metric): float(weight)
        for metric, weight in config.get(
            "index_weights", {"wpa": 0.4, "runs_saved": 0.3, "consistency": 0.3}
        ).items()
    }
    required_weights = {"wpa", "runs_saved", "consistency"}
    if set(index_weights) != required_weights:
        raise ValueError(f"index_weights must contain exactly {sorted(required_weights)}")
    if any(not np.isfinite(weight) or weight < 0 for weight in index_weights.values()):
        raise ValueError("index_weights must be finite and non-negative")
    if sum(index_weights.values()) <= 0:
        raise ValueError("At least one index weight must be positive")
    minimum_pressure_balls = int(config.get("min_pressure_balls", config.get("min_balls", 30)))
    if minimum_pressure_balls < 1:
        raise ValueError("Minimum pressure balls must be positive")
    return config, league_weights, index_weights, minimum_pressure_balls


def _add_situation_baselines(state: pd.DataFrame) -> pd.DataFrame:
    legal = state.loc[state["is_legal"]].copy()
    legal["phase"] = pd.cut(
        legal["overid"],
        bins=[0, 6, 15, 20],
        labels=["PP", "MID", "DEATH"],
        include_lowest=True,
    ).astype("string")
    legal["wicket_bucket"] = pd.cut(
        legal["wickets_before"],
        bins=[-1, 0, 2, 4, 10],
        labels=["0", "1-2", "3-4", "5+"],
    ).astype("string")
    situation = (
        legal.groupby(["innings", "phase", "wicket_bucket"], observed=True)
        .agg(
            situation_bat_runs_per_ball=("runs", "mean"),
            situation_total_runs_per_ball=("total_runs", "mean"),
        )
        .reset_index()
    )
    return legal.merge(
        situation,
        on=["innings", "phase", "wicket_bucket"],
        how="left",
        validate="many_to_one",
    )


def _event_units(rows: pd.DataFrame, player_column: str, role: str) -> pd.DataFrame:
    if role == "batting":
        keys = [player_column, "match_id", "innings"]
    else:
        keys = [player_column, "match_id", "innings", "overid"]
    units = (
        rows.groupby(keys, observed=True)
        .agg(
            unit_wpa=("role_wpa", "sum"),
            unit_balls=("pressure_ball", "size"),
        )
        .reset_index()
    )
    units["unit_wpa_per_ball"] = np.divide(
        units["unit_wpa"],
        units["unit_balls"],
        out=np.zeros(len(units), dtype=float),
        where=units["unit_balls"].to_numpy() > 0,
    )
    if role == "batting":
        units = units.loc[units["unit_balls"].ge(6)].copy()
    return units


def _eb_strength(raw: pd.Series, rows: pd.DataFrame, n_balls: pd.Series) -> tuple[float, float]:
    raw_spread = float(raw.std(ddof=0))
    pooled_noise = float(rows["role_wpa"].var(ddof=0))
    if len(raw) < 2 or not np.isfinite(pooled_noise) or pooled_noise <= 0:
        return 1000.0, max(pooled_noise, 0.0)
    between_variance = float(raw.var(ddof=0))
    average_sampling_variance = float(
        np.mean(pooled_noise / np.maximum(n_balls.to_numpy(dtype=float), 1))
    )
    prior_variance = max(between_variance - average_sampling_variance, 0.0)
    if prior_variance <= np.finfo(float).eps:
        strength = 1000.0
    else:
        strength = float(np.clip(pooled_noise / prior_variance, 1.0, 1000.0))
    if raw_spread and strength <= 0:
        raise AssertionError("Empirical-Bayes prior strength must be positive")
    return strength, pooled_noise


def _summarize_role(
    rows: pd.DataFrame,
    player_column: str,
    role: str,
    minimum_pressure_balls: int,
    beta_prior: tuple[float, float],
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    rows = rows.copy()
    rows["role_weight"] = rows["league_weight"].where(rows["league_weight"].gt(0), 0.0)
    rows["weighted_pressure"] = rows["pressure_ball"].astype(float) * rows["role_weight"]
    rows["role_wpa"] = rows["wpa"] if role == "batting" else -rows["wpa"]
    rows["role_wpa"] = rows["role_wpa"].fillna(0.0)
    rows["role_runs_saved"] = (
        rows["runs"] - rows["situation_bat_runs_per_ball"]
        if role == "batting"
        else rows["situation_total_runs_per_ball"] - rows["total_runs"]
    )
    rows["role_runs_saved"] = rows["role_runs_saved"].fillna(0.0)

    all_summary = rows.groupby(player_column, observed=True).agg(
        legal_balls=("is_legal", "size"),
        league_weighted_balls=("role_weight", "sum"),
        pressure_balls=("pressure_ball", "sum"),
        weighted_pressure_balls=("weighted_pressure", "sum"),
        weighted_all_balls=("role_weight", "sum"),
    )
    pressure = rows.loc[rows["pressure_ball"]].copy()
    if pressure.empty:
        return pd.DataFrame(index=all_summary.index), {}

    pressure["weighted_wpa"] = pressure["role_wpa"] * pressure["role_weight"]
    pressure["weighted_runs_saved"] = (
        pressure["role_runs_saved"] * pressure["role_weight"]
    )
    pressure["weighted_bat_runs"] = pressure["runs"] * pressure["role_weight"]
    pressure["weighted_total_runs"] = pressure["total_runs"] * pressure["role_weight"]
    pressure["weighted_legal_ball"] = pressure["role_weight"]
    pressure["dismissal"] = (
        pressure["is_wicket"]
        & pressure["dismissed_player_id"].eq(pressure[player_column])
        if role == "batting"
        else pressure["is_wicket"]
        & pressure["dismissal_type"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(WICKET_DISMISSALS)
    )
    pressure["dot"] = pressure["total_runs"].eq(0)
    pressure["boundary"] = pressure["is_four"].fillna(False) | pressure["is_six"].fillna(False)

    aggregate = pressure.groupby(player_column, observed=True).agg(
        weighted_balls=("weighted_legal_ball", "sum"),
        raw_pressure_balls=("pressure_ball", "sum"),
        weighted_wpa=("weighted_wpa", "sum"),
        weighted_runs_saved=("weighted_runs_saved", "sum"),
        weighted_bat_runs=("weighted_bat_runs", "sum"),
        weighted_total_runs=("weighted_total_runs", "sum"),
        weighted_dismissals=("dismissal", lambda values: float((values * pressure.loc[values.index, "role_weight"]).sum())),
        weighted_dots=("dot", lambda values: float((values * pressure.loc[values.index, "role_weight"]).sum())),
        weighted_boundaries=("boundary", lambda values: float((values * pressure.loc[values.index, "role_weight"]).sum())),
        pressure_weight=("role_weight", "sum"),
    )
    aggregate["raw_wpa_per_ball"] = np.divide(
        aggregate["weighted_wpa"],
        aggregate["weighted_balls"],
        out=np.zeros(len(aggregate), dtype=float),
        where=aggregate["weighted_balls"].to_numpy() > 0,
    )
    aggregate["runs_saved_per_ball"] = np.divide(
        aggregate["weighted_runs_saved"],
        aggregate["weighted_balls"],
        out=np.zeros(len(aggregate), dtype=float),
        where=aggregate["weighted_balls"].to_numpy() > 0,
    )
    aggregate["runs_saved"] = aggregate["weighted_runs_saved"]
    aggregate["strike_rate"] = np.divide(
        aggregate["weighted_bat_runs"] * 100,
        aggregate["weighted_balls"],
        out=np.zeros(len(aggregate), dtype=float),
        where=aggregate["weighted_balls"].to_numpy() > 0,
    )
    aggregate["economy"] = np.divide(
        aggregate["weighted_total_runs"] * 6,
        aggregate["weighted_balls"],
        out=np.zeros(len(aggregate), dtype=float),
        where=aggregate["weighted_balls"].to_numpy() > 0,
    )
    aggregate["dot_pct"] = np.divide(
        aggregate["weighted_dots"],
        aggregate["weighted_balls"],
        out=np.zeros(len(aggregate), dtype=float),
        where=aggregate["weighted_balls"].to_numpy() > 0,
    )
    aggregate["boundary_pct"] = np.divide(
        aggregate["weighted_boundaries"],
        aggregate["weighted_balls"],
        out=np.zeros(len(aggregate), dtype=float),
        where=aggregate["weighted_balls"].to_numpy() > 0,
    )
    aggregate["dismissal_rate"] = np.divide(
        aggregate["weighted_dismissals"],
        aggregate["weighted_balls"],
        out=np.zeros(len(aggregate), dtype=float),
        where=aggregate["weighted_balls"].to_numpy() > 0,
    )
    aggregate["wickets_per_24_legal_balls"] = np.divide(
        aggregate["weighted_dismissals"] * 24,
        aggregate["weighted_balls"],
        out=np.zeros(len(aggregate), dtype=float),
        where=aggregate["weighted_balls"].to_numpy() > 0,
    )
    aggregate["qualifies"] = aggregate["raw_pressure_balls"].ge(minimum_pressure_balls)

    units = _event_units(pressure, player_column, role)
    prior_a, prior_b = beta_prior
    if units.empty:
        consistency = pd.DataFrame(columns=["consistency", "consistency_variance", "wpa_variance_across_units"])
    else:
        consistency = units.groupby(player_column, observed=True).agg(
            positive_units=("unit_wpa", lambda values: int(values.gt(0).sum())),
            n_units=("unit_wpa", "size"),
            wpa_variance_across_units=("unit_wpa_per_ball", "var"),
        )
        consistency["consistency"] = (
            consistency["positive_units"] + prior_a
        ) / (consistency["n_units"] + prior_a + prior_b)
        consistency["consistency_variance"] = (
            (consistency["positive_units"] + prior_a)
            * (consistency["n_units"] - consistency["positive_units"] + prior_b)
            / (
                (consistency["n_units"] + prior_a + prior_b) ** 2
                * (consistency["n_units"] + prior_a + prior_b + 1)
            )
        )
    aggregate = aggregate.join(consistency, how="left")
    aggregate["consistency"] = aggregate["consistency"].fillna(
        prior_a / (prior_a + prior_b)
    )
    aggregate["consistency_variance"] = aggregate["consistency_variance"].fillna(
        prior_a * prior_b
        / ((prior_a + prior_b) ** 2 * (prior_a + prior_b + 1))
    )
    aggregate["wpa_variance_across_units"] = aggregate[
        "wpa_variance_across_units"
    ].fillna(0.0)

    pooled_rows = pressure.loc[pressure["role_weight"].gt(0)]
    raw = aggregate["raw_wpa_per_ball"]
    strength, noise_variance = _eb_strength(
        raw,
        pooled_rows.assign(role_wpa=pooled_rows["role_wpa"]),
        aggregate["raw_pressure_balls"],
    )
    global_mean = _weighted_mean(
        aggregate["raw_wpa_per_ball"],
        aggregate["weighted_balls"],
    )
    n = aggregate["raw_pressure_balls"].astype(float)
    aggregate["shrunk_wpa_per_ball"] = (
        n * aggregate["raw_wpa_per_ball"] + strength * global_mean
    ) / (n + strength)
    aggregate["pressure_wpa_per_ball_shrunk_30"] = (
        aggregate["weighted_balls"] * aggregate["raw_wpa_per_ball"]
        + 30.0 * global_mean
    ) / (aggregate["weighted_balls"] + 30.0)
    aggregate["posterior_sd_wpa_per_ball"] = np.sqrt(
        max(noise_variance, 0.0) / (n + strength)
    )
    aggregate["wpa_ci80_lower"] = (
        aggregate["shrunk_wpa_per_ball"]
        - WPA_CI_Z * aggregate["posterior_sd_wpa_per_ball"]
    )
    aggregate["wpa_ci80_upper"] = (
        aggregate["shrunk_wpa_per_ball"]
        + WPA_CI_Z * aggregate["posterior_sd_wpa_per_ball"]
    )
    aggregate["wpa_conservative_lower_bound"] = aggregate["wpa_ci80_lower"]
    aggregate["eb_prior_strength"] = strength

    spread = {
        "raw": float(raw.std(ddof=0)),
        "shrunk": float(aggregate["shrunk_wpa_per_ball"].std(ddof=0)),
    }
    if len(aggregate) > 1 and spread["shrunk"] >= spread["raw"]:
        raise AssertionError(
            f"{role} empirical-Bayes shrinkage did not reduce WPA/ball spread: {spread}"
        )

    metrics = aggregate.join(all_summary, how="left", rsuffix="_all")
    metrics["normal_balls"] = (
        metrics["legal_balls"].fillna(0) - metrics["pressure_balls"].fillna(0)
    ).clip(lower=0)
    metrics["normal_weight"] = (
        rows.loc[~rows["pressure_ball"]]
        .groupby(player_column, observed=True)["role_weight"]
        .sum()
        .reindex(metrics.index, fill_value=0)
    )
    if role == "batting":
        normal_runs = (
            rows.loc[~rows["pressure_ball"]]
            .assign(weighted_runs=lambda frame: frame["runs"] * frame["role_weight"])
            .groupby(player_column, observed=True)["weighted_runs"]
            .sum()
            .reindex(metrics.index, fill_value=0)
        )
        normal_wpa = (
            rows.loc[~rows["pressure_ball"]]
            .assign(weighted_wpa=lambda frame: frame["role_wpa"] * frame["role_weight"])
            .groupby(player_column, observed=True)["weighted_wpa"]
            .sum()
            .reindex(metrics.index, fill_value=0)
        )
        metrics["normal_strike_rate"] = np.divide(
            normal_runs * 100,
            metrics["normal_weight"],
            out=np.zeros(len(metrics), dtype=float),
            where=metrics["normal_weight"].to_numpy() > 0,
        )
        metrics["pressure_vs_normal_strike_rate_delta"] = (
            metrics["strike_rate"] - metrics["normal_strike_rate"]
        )
    else:
        normal_runs = (
            rows.loc[~rows["pressure_ball"]]
            .assign(weighted_runs=lambda frame: frame["total_runs"] * frame["role_weight"])
            .groupby(player_column, observed=True)["weighted_runs"]
            .sum()
            .reindex(metrics.index, fill_value=0)
        )
        normal_wpa = (
            rows.loc[~rows["pressure_ball"]]
            .assign(weighted_wpa=lambda frame: frame["role_wpa"] * frame["role_weight"])
            .groupby(player_column, observed=True)["weighted_wpa"]
            .sum()
            .reindex(metrics.index, fill_value=0)
        )
        normal_economy = np.divide(
            normal_runs * 6,
            metrics["normal_weight"],
            out=np.zeros(len(metrics), dtype=float),
            where=metrics["normal_weight"].to_numpy() > 0,
        )
        metrics["normal_economy"] = normal_economy
        metrics["pressure_vs_normal_economy_delta"] = (
            metrics["economy"] - normal_economy
        )
    metrics["normal_wpa_per_ball"] = np.divide(
        normal_wpa,
        metrics["normal_weight"],
        out=np.zeros(len(metrics), dtype=float),
        where=metrics["normal_weight"].to_numpy() > 0,
    )
    metrics["overall_weighted_balls"] = (
        metrics["weighted_balls"].fillna(0) + metrics["normal_weight"]
    )
    overall_weighted_wpa = metrics["weighted_wpa"].fillna(0) + normal_wpa
    metrics["overall_raw_wpa_per_ball"] = np.divide(
        overall_weighted_wpa,
        metrics["overall_weighted_balls"],
        out=np.zeros(len(metrics), dtype=float),
        where=metrics["overall_weighted_balls"].to_numpy() > 0,
    )
    overall_prior = _weighted_mean(
        metrics["overall_raw_wpa_per_ball"],
        metrics["overall_weighted_balls"],
    )
    overall_balls = metrics["overall_weighted_balls"].to_numpy(dtype=float)
    metrics["overall_shrunk_wpa_per_ball"] = np.divide(
        overall_balls * metrics["overall_raw_wpa_per_ball"]
        + 30.0 * overall_prior,
        overall_balls + 30.0,
        out=np.full(len(metrics), overall_prior, dtype=float),
        where=overall_balls + 30.0 > 0,
    )
    metrics["wpa_pressure_vs_normal_delta"] = (
        metrics["raw_wpa_per_ball"] - metrics["normal_wpa_per_ball"]
    )
    return metrics, spread


def _compute_role_flag_pri(
    rows: pd.DataFrame,
    player_column: str,
    role: str,
    flag: str,
    minimum_pressure_balls: int,
    beta_prior: tuple[float, float],
    index_weights: dict[str, float],
) -> tuple[pd.DataFrame, dict[str, float]]:
    scoped = rows.copy()
    scoped["pressure_ball"] = scoped["is_legal"] & scoped[flag].astype(bool)
    metrics, _ = _summarize_role(
        scoped, player_column, role, minimum_pressure_balls, beta_prior
    )
    if metrics.empty:
        return metrics, {}
    valid = metrics["qualifies"]
    components = {
        "wpa": metrics["shrunk_wpa_per_ball"],
        "runs_saved": metrics["runs_saved_per_ball"],
        "consistency": metrics["consistency"],
    }
    index = pd.Series(0.0, index=metrics.index)
    for component, values in components.items():
        eligible = values.loc[valid]
        mean = float(eligible.mean()) if not eligible.empty else 0.0
        standard_deviation = float(eligible.std(ddof=0)) if len(eligible) > 1 else 0.0
        z_score = (
            (values - mean) / standard_deviation
            if standard_deviation > 0
            else pd.Series(0.0, index=metrics.index)
        )
        metrics[f"z_{component}"] = z_score
        index += index_weights[component] * z_score
    metrics["pri"] = index.where(valid, np.nan)
    return metrics, {
        "raw_wpa_spread": float(metrics.loc[valid, "raw_wpa_per_ball"].std(ddof=0)),
        "shrunk_wpa_spread": float(metrics.loc[valid, "shrunk_wpa_per_ball"].std(ddof=0)),
    }


def _aggregate_season_and_stroke(
    rows: pd.DataFrame, data_version: str
) -> None:
    season_parts = []
    for role, player_column in (("batting", "striker_id"), ("bowling", "bowler_id")):
        frame = rows.loc[rows["is_legal"]].copy()
        frame["role"] = role
        frame["player_id"] = frame[player_column]
        frame = frame.dropna(subset=["player_id"])
        frame["runs_saved"] = (
            frame["runs"] - frame["situation_bat_runs_per_ball"]
            if role == "batting"
            else frame["situation_total_runs_per_ball"] - frame["total_runs"]
        )
        frame["role_wpa"] = frame["wpa"] if role == "batting" else -frame["wpa"]
        frame["pressure_ball"] = frame["pressure_ball"].astype(bool)
        grouped = frame.groupby(
            ["player_id", "role", "season", "comp_group"], observed=True
        )
        summary = grouped.agg(
            legal_balls=("is_legal", "size"),
            pressure_balls=("pressure_ball", "sum"),
            runs=("runs", "sum"),
            total_runs=("total_runs", "sum"),
            runs_saved=("runs_saved", "sum"),
            mean_wpa_per_ball=("role_wpa", "mean"),
        ).reset_index()
        summary.insert(0, "data_version", data_version)
        season_parts.append(summary)
    seasons = (
        pd.concat(season_parts, ignore_index=True)
        if season_parts
        else pd.DataFrame()
    )
    seasons.to_parquet(PROCESSED / "player_season_metrics.parquet", index=False)

    batting = rows.loc[rows["is_legal"] & rows["striker_id"].notna()].copy()
    batting["player_id"] = batting["striker_id"]
    batting["weighted_runs_above_situation"] = (
        batting["runs"] - batting["situation_bat_runs_per_ball"]
    )
    strokes = (
        batting.dropna(subset=["stroke"])
        .groupby(["player_id", "stroke"], observed=True)
        .agg(
            legal_balls=("is_legal", "size"),
            runs=("runs", "sum"),
            runs_above_situation=("weighted_runs_above_situation", "sum"),
            boundaries=("is_four", lambda values: int(values.fillna(False).sum())),
            mean_wpa_per_ball=("wpa", "mean"),
        )
        .reset_index()
    )
    strokes.insert(0, "data_version", data_version)
    strokes.to_parquet(PROCESSED / "player_stroke_metrics.parquet", index=False)


def build_player_metrics(data_version: str) -> pd.DataFrame:
    config, league_weights, index_weights, minimum_pressure_balls = _load_config()
    pressure_config = config.get("player_metrics", {})
    all_rounder_config = config.get("all_rounder_eligibility", {})
    all_rounder_min_batting_pressure_balls = int(
        all_rounder_config.get("min_batting_pressure_balls", 100)
    )
    all_rounder_min_bowling_pressure_balls = int(
        all_rounder_config.get("min_bowling_pressure_balls", 100)
    )
    if (
        all_rounder_min_batting_pressure_balls < 1
        or all_rounder_min_bowling_pressure_balls < 1
    ):
        raise ValueError("All-rounder pressure-ball thresholds must be positive")
    beta_prior = (
        float(pressure_config.get("consistency_beta_alpha", 1.0)),
        float(pressure_config.get("consistency_beta_beta", 1.0)),
    )
    if min(beta_prior) <= 0:
        raise ValueError("Consistency Beta-prior parameters must be positive")

    rows = pd.read_parquet(PROCESSED / "pressure_flags.parquet")
    matches = pd.read_parquet(PROCESSED / "matches.parquet")
    season_lookup = matches.set_index("match_id")["season"]
    rows["season"] = rows["match_id"].map(season_lookup)
    if rows["season"].isna().any():
        raise ValueError("Some cleaned deliveries do not map to a competition season")
    missing_groups = sorted(set(rows["comp_group"].dropna()) - set(league_weights))
    if missing_groups:
        raise ValueError(f"Missing league weights for groups: {missing_groups}")
    rows["league_weight"] = rows["comp_group"].map(league_weights).fillna(0.0)
    rows = _add_situation_baselines(rows)
    rows["pressure_ball"] = rows["is_legal"] & rows[PRESSURE_FLAGS].any(axis=1)
    rows["is_bowler_wicket"] = (
        rows["is_wicket"]
        & rows["dismissal_type"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(WICKET_DISMISSALS)
    )

    batting_rows = rows.loc[rows["striker_id"].notna()].copy()
    bowling_rows = rows.loc[rows["bowler_id"].notna()].copy()
    bat_metrics, bat_spread = _summarize_role(
        batting_rows,
        "striker_id",
        "batting",
        minimum_pressure_balls,
        beta_prior,
    )
    bowl_metrics, bowl_spread = _summarize_role(
        bowling_rows,
        "bowler_id",
        "bowling",
        minimum_pressure_balls,
        beta_prior,
    )
    for phase, (start, end) in PHASES.items():
        phase_rows = bowling_rows.loc[bowling_rows["overid"].between(start, end)].copy()
        phase_rows["weighted_conceded_runs"] = (
            phase_rows["total_runs"] * phase_rows["league_weight"]
        )
        phase_rows["weighted_wickets"] = (
            phase_rows["is_bowler_wicket"].astype(float)
            * phase_rows["league_weight"]
        )
        phase_summary = phase_rows.groupby("bowler_id", observed=True).agg(
            weighted_legal_balls=("league_weight", "sum"),
            legal_balls=("is_legal", "size"),
            weighted_runs=("weighted_conceded_runs", "sum"),
            weighted_wickets=("weighted_wickets", "sum"),
        )
        bowl_metrics[f"legal_balls_{phase.lower()}"] = phase_summary[
            "legal_balls"
        ].reindex(bowl_metrics.index)
        bowl_metrics[f"economy_{phase.lower()}"] = np.divide(
            phase_summary["weighted_runs"].reindex(bowl_metrics.index).fillna(0) * 6,
            phase_summary["weighted_legal_balls"]
            .reindex(bowl_metrics.index)
            .fillna(0),
            out=np.zeros(len(bowl_metrics), dtype=float),
            where=phase_summary["weighted_legal_balls"]
            .reindex(bowl_metrics.index)
            .fillna(0)
            .to_numpy()
            > 0,
        )
        bowl_metrics[f"wickets_per_24_legal_balls_{phase.lower()}"] = np.divide(
            phase_summary["weighted_wickets"]
            .reindex(bowl_metrics.index)
            .fillna(0)
            * 24,
            phase_summary["weighted_legal_balls"]
            .reindex(bowl_metrics.index)
            .fillna(0),
            out=np.zeros(len(bowl_metrics), dtype=float),
            where=phase_summary["weighted_legal_balls"]
            .reindex(bowl_metrics.index)
            .fillna(0)
            .to_numpy()
            > 0,
        )

    # Role-specific summaries are computed independently for every configured flag.
    for flag in [*PRESSURE_FLAGS, "composite"]:
        flag_column = "pressure_ball" if flag == "composite" else flag
        for role, role_rows, player_column in (
            ("batting", batting_rows, "striker_id"),
            ("bowling", bowling_rows, "bowler_id"),
        ):
            flag_metrics, _ = _compute_role_flag_pri(
                role_rows,
                player_column,
                role,
                flag_column,
                minimum_pressure_balls,
                beta_prior,
                index_weights,
            )
            if flag_metrics.empty:
                continue
            role_metrics = bat_metrics if role == "batting" else bowl_metrics
            role_metrics[f"pri_{flag}"] = flag_metrics["pri"].reindex(
                role_metrics.index
            )
            suffix = "" if flag == "composite" else f"_{flag}"
            if suffix:
                role_metrics[f"shrunk_wpa_per_ball{suffix}"] = flag_metrics[
                    "shrunk_wpa_per_ball"
                ].reindex(role_metrics.index)

    batting_id = bat_metrics.index.rename("player_id")
    bowling_id = bowl_metrics.index.rename("player_id")
    player_ids = batting_id.union(bowling_id).sort_values()
    metrics = pd.concat(
        [
            bat_metrics.reindex(player_ids).add_prefix("batting_"),
            bowl_metrics.reindex(player_ids).add_prefix("bowling_"),
        ],
        axis=1,
    ).copy()
    metrics.index.name = "player_id"
    metrics["batting_qualifies"] = metrics["batting_qualifies"].fillna(False).astype(bool)
    metrics["bowling_qualifies"] = metrics["bowling_qualifies"].fillna(False).astype(bool)
    metrics["is_all_rounder"] = (
        metrics["batting_pressure_balls"]
        .fillna(0)
        .ge(all_rounder_min_batting_pressure_balls)
        & metrics["bowling_pressure_balls"]
        .fillna(0)
        .ge(all_rounder_min_bowling_pressure_balls)
    )
    metrics["role"] = np.select(
        [metrics["is_all_rounder"], metrics["bowling_qualifies"], metrics["batting_qualifies"]],
        ["all_rounder", "bowler", "batter"],
        default="developing",
    )

    batting_order = (
        batting_rows.groupby("striker_id")["batting_order"].median().rename("median_batting_order")
    )
    metrics["median_batting_order"] = batting_order.reindex(metrics.index)
    median_order = metrics["median_batting_order"]
    top_order = median_order.le(3)
    middle_order = median_order.gt(3) & median_order.le(6)
    lower_order = median_order.gt(6)
    total_batting_phase = batting_rows.groupby("striker_id").size()
    total_bowling_phase = bowling_rows.groupby("bowler_id").size()
    batting_role_volume = total_batting_phase.reindex(metrics.index).fillna(0)
    bowling_role_volume = total_bowling_phase.reindex(metrics.index).fillna(0)
    appears_more_as_bowler = bowling_role_volume.gt(batting_role_volume)
    has_batting_appearance = batting_role_volume.gt(0)
    metrics["primary_role"] = np.select(
        [
            metrics["is_all_rounder"],
            metrics["bowling_qualifies"],
            metrics["batting_qualifies"] & top_order,
            metrics["batting_qualifies"] & middle_order,
            metrics["batting_qualifies"] & lower_order,
            appears_more_as_bowler,
            has_batting_appearance & top_order,
            has_batting_appearance & middle_order,
            has_batting_appearance & lower_order,
        ],
        [
            "all_rounder",
            "bowler",
            "top_order_batter",
            "middle_order_batter",
            "lower_order_batter",
            "bowler",
            "top_order_batter",
            "middle_order_batter",
            "lower_order_batter",
        ],
        default="developing",
    )
    for phase, (start, end) in PHASES.items():
        bat_counts = batting_rows.loc[batting_rows["overid"].between(start, end)].groupby("striker_id").size()
        bowl_counts = bowling_rows.loc[bowling_rows["overid"].between(start, end)].groupby("bowler_id").size()
        metrics[f"batting_phase_share_{phase.lower()}"] = (
            bat_counts.reindex(metrics.index, fill_value=0)
            / total_batting_phase.reindex(metrics.index).fillna(0).replace(0, np.nan)
        )
        metrics[f"bowling_phase_share_{phase.lower()}"] = (
            bowl_counts.reindex(metrics.index, fill_value=0)
            / total_bowling_phase.reindex(metrics.index).fillna(0).replace(0, np.nan)
        )
    death_counts = bowling_rows.loc[bowling_rows["overid"].between(16, 20)].groupby("bowler_id").size()
    metrics["death_overs_share"] = (
        death_counts.reindex(metrics.index, fill_value=0)
        / total_bowling_phase.reindex(metrics.index).fillna(0).replace(0, np.nan)
    )
    pace_counts = bowling_rows.assign(
        _pace=bowling_rows["pace_or_spin"].fillna("").astype(str).str.lower().eq("pace")
    ).groupby("bowler_id")["_pace"].sum()
    spin_counts = bowling_rows.assign(
        _spin=bowling_rows["pace_or_spin"].fillna("").astype(str).str.lower().eq("spin")
    ).groupby("bowler_id")["_spin"].sum()
    bowling_n = total_bowling_phase.reindex(metrics.index).fillna(0).replace(0, np.nan)
    metrics["pace_share"] = pace_counts.reindex(metrics.index).fillna(0) / bowling_n
    metrics["spin_share"] = spin_counts.reindex(metrics.index).fillna(0) / bowling_n
    batting_hands = (
        batting_rows.dropna(subset=["batting_type"])
        .groupby("striker_id")["batting_type"]
        .agg(lambda values: values.mode().iloc[0] if not values.mode().empty else None)
    )
    metrics["batting_hand"] = batting_hands.reindex(metrics.index)

    name_rows = pd.concat(
        [
            batting_rows[["striker_id", "striker_name"]].rename(
                columns={"striker_id": "player_id", "striker_name": "player_name"}
            ),
            bowling_rows[["bowler_id", "bowler_name"]].rename(
                columns={"bowler_id": "player_id", "bowler_name": "player_name"}
            ),
        ],
        ignore_index=True,
    ).dropna(subset=["player_id", "player_name"])
    names = name_rows.groupby("player_id")["player_name"].agg(
        lambda values: values.mode().iloc[0] if not values.mode().empty else values.iloc[0]
    )
    names = names.reindex(metrics.index)
    duplicate_name = names.fillna("").str.strip().str.casefold().map(
        names.fillna("").str.strip().str.casefold().value_counts()
    ).gt(1)
    metrics["player_name"] = names.where(
        ~duplicate_name, names + " (ID " + metrics.index.astype(str) + ")"
    )

    total_pressure_balls = (
        metrics["batting_pressure_balls"].fillna(0)
        + metrics["bowling_pressure_balls"].fillna(0)
    )
    metrics["pressure_balls"] = total_pressure_balls.astype(int)
    metrics["sample_size"] = metrics["pressure_balls"]
    bat_volume = metrics["batting_pressure_balls"].fillna(0)
    bowl_volume = metrics["bowling_pressure_balls"].fillna(0)
    total_role_volume = bat_volume + bowl_volume
    for flag in [*PRESSURE_FLAGS, "composite"]:
        bat_index = metrics.get(f"batting_pri_{flag}", pd.Series(index=metrics.index, dtype=float))
        bowl_index = metrics.get(f"bowling_pri_{flag}", pd.Series(index=metrics.index, dtype=float))
        combined = np.divide(
            bat_index.fillna(0) * bat_volume + bowl_index.fillna(0) * bowl_volume,
            total_role_volume,
            out=np.full(len(metrics), np.nan, dtype=float),
            where=total_role_volume.to_numpy() > 0,
        )
        metrics[f"pri_{flag}"] = combined
    metrics["pressure_index"] = metrics["pri_composite"].fillna(0.0)
    metrics["wpa_per_ball"] = np.divide(
        metrics["batting_shrunk_wpa_per_ball"].fillna(0) * bat_volume
        + metrics["bowling_shrunk_wpa_per_ball"].fillna(0) * bowl_volume,
        total_role_volume,
        out=np.zeros(len(metrics), dtype=float),
        where=total_role_volume.to_numpy() > 0,
    )
    metrics["pressure_delta"] = metrics[
        "batting_pressure_vs_normal_strike_rate_delta"
    ]
    metrics["rank_stability"] = metrics["batting_consistency"].combine_first(
        metrics["bowling_consistency"]
    )
    metrics["ipl_ball_share"] = np.divide(
        (
            batting_rows.loc[batting_rows["comp_group"].eq("IPL")]
            .groupby("striker_id")["is_legal"]
            .sum()
            .reindex(metrics.index, fill_value=0)
            + bowling_rows.loc[bowling_rows["comp_group"].eq("IPL")]
            .groupby("bowler_id")["is_legal"]
            .sum()
            .reindex(metrics.index, fill_value=0)
        ),
        (
            batting_rows.groupby("striker_id")["is_legal"].sum().reindex(metrics.index, fill_value=0)
            + bowling_rows.groupby("bowler_id")["is_legal"].sum().reindex(metrics.index, fill_value=0)
        ),
        out=np.zeros(len(metrics), dtype=float),
        where=(
            batting_rows.groupby("striker_id")["is_legal"].sum().reindex(metrics.index, fill_value=0)
            + bowling_rows.groupby("bowler_id")["is_legal"].sum().reindex(metrics.index, fill_value=0)
        ).to_numpy() > 0,
    )
    eligibility_min_pressure_balls = int(
        config.get("eligibility_min_pressure_balls", 80)
    )
    min_ipl_share = float(config.get("min_ipl_share", 0.30))
    if eligibility_min_pressure_balls < 1 or not 0 <= min_ipl_share <= 1:
        raise ValueError("Invalid player eligibility thresholds in config.yaml")
    metrics["eligible_batter"] = (
        metrics["batting_pressure_balls"].fillna(0).ge(eligibility_min_pressure_balls)
        & metrics["ipl_ball_share"].ge(min_ipl_share)
    )
    metrics["eligible_bowler"] = (
        metrics["bowling_pressure_balls"].fillna(0).ge(eligibility_min_pressure_balls)
        & metrics["ipl_ball_share"].ge(min_ipl_share)
    )
    metrics["selection_eligible"] = metrics["eligible_batter"] | metrics["eligible_bowler"]

    # Expose useful shorthand columns while retaining explicitly prefixed role metrics.
    for source, target in (
        ("batting_strike_rate", "strike_rate"),
        ("batting_boundary_pct", "boundary_pct"),
        ("batting_dot_pct", "dot_pct"),
        ("batting_runs_saved", "runs_above_situation_average"),
        ("batting_raw_wpa_per_ball", "raw_wpa_per_ball"),
        ("batting_shrunk_wpa_per_ball", "shrunk_wpa_per_ball"),
        ("batting_wpa_ci80_lower", "wpa_ci80_lower"),
        ("batting_wpa_ci80_upper", "wpa_ci80_upper"),
        ("batting_wpa_conservative_lower_bound", "wpa_conservative_lower_bound"),
        ("bowling_legal_balls", "legal_balls_bowled"),
        ("bowling_economy", "economy"),
        ("bowling_wickets_per_24_legal_balls", "wickets_per_24_legal_balls"),
        ("bowling_runs_saved", "bowling_runs_saved"),
    ):
        if source in metrics:
            metrics[target] = metrics[source]
    metrics["batting_wpa_per_ball"] = metrics["batting_raw_wpa_per_ball"]
    metrics["bowling_wpa_per_ball"] = metrics["bowling_raw_wpa_per_ball"]

    metrics["data_version"] = data_version
    metrics = metrics.reset_index()
    if metrics["player_id"].duplicated().any():
        raise AssertionError("Comparable player table must contain one row per player ID")

    _aggregate_season_and_stroke(rows, data_version)
    metrics.to_parquet(PROCESSED / "player_metrics.parquet", index=False)

    quality_path = PROCESSED / "data_quality.json"
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    quality["player_metrics"] = {
        "data_version": data_version,
        "min_pressure_balls": minimum_pressure_balls,
        "n_players": int(len(metrics)),
        "qualifying_batters": int(metrics["batting_qualifies"].sum()),
        "qualifying_bowlers": int(metrics["bowling_qualifies"].sum()),
        "qualifying_all_rounders": int(metrics["is_all_rounder"].sum()),
        "all_rounder_min_batting_pressure_balls": all_rounder_min_batting_pressure_balls,
        "all_rounder_min_bowling_pressure_balls": all_rounder_min_bowling_pressure_balls,
        "eligibility_min_pressure_balls": eligibility_min_pressure_balls,
        "min_ipl_share": min_ipl_share,
        "raw_vs_shrunk_wpa_per_ball_spread": {
            "batting": bat_spread,
            "bowling": bowl_spread,
        },
        "wpa_runs_above_average_correlation": float(
            metrics[["raw_wpa_per_ball", "runs_above_situation_average"]]
            .dropna()
            .corr()
            .iloc[0, 1]
        ),
    }
    quality_path.write_text(json.dumps(quality, indent=2), encoding="utf-8")
    return metrics
