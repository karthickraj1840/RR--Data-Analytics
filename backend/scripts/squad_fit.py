from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "backend" / "data" / "processed"
EXTERNAL = ROOT / "data" / "external"
TOP10_PATH = PROCESSED / "top10.csv"
METRICS_PATH = PROCESSED / "player_metrics.parquet"
ATTRIBUTES_PATH = EXTERNAL / "player_attributes.csv"
SQUAD_FIT_PATH = PROCESSED / "squad_fit.csv"
SWAPS_PATH = PROCESSED / "squad_fit_swaps.csv"
VALID_STATUSES = {
    "retained_by_rr",
    "released_by_rr",
    "in_auction_pool",
    "retained_by_other_team",
    "unknown",
}
ATTRIBUTE_COLUMNS = (
    "rr_status",
    "overseas",
    "age",
    "price_band",
    "source_url",
    "notes",
)
FIT_COLUMNS = (
    "rank",
    "player_id",
    "player_name",
    "role",
    "grade",
    "pressure_balls",
    "ipl_share",
    *ATTRIBUTE_COLUMNS,
    "fit_flag",
    "final_list",
)
SWAP_COLUMNS = (
    "old_rank",
    "old_player_name",
    "new_rank",
    "new_player_name",
    "reason",
)


def _read_attributes() -> pd.DataFrame:
    attributes = pd.read_csv(
        ATTRIBUTES_PATH,
        dtype=str,
        keep_default_na=False,
    )
    if attributes["player_id"].duplicated().any():
        raise ValueError(f"{ATTRIBUTES_PATH} contains duplicate player_id values")
    invalid_statuses = set(attributes["rr_status"]) - VALID_STATUSES
    if invalid_statuses:
        raise ValueError(f"Invalid rr_status values: {sorted(invalid_statuses)}")
    return attributes


def _fit_flag(status: str, overseas: str) -> str:
    if status in {"retained_by_rr", "retained_by_other_team"}:
        return "SWAP"
    if status == "unknown" or not overseas:
        return "REVIEW"
    if status in {"released_by_rr", "in_auction_pool"}:
        return "KEEP"
    return "REVIEW"


def _is_overseas(value: Any) -> bool | None:
    normalized = str(value).strip().lower()
    if normalized in {"true", "yes", "1", "overseas"}:
        return True
    if normalized in {"false", "no", "0", "domestic"}:
        return False
    return None


def _load_candidates() -> list[dict[str, Any]]:
    top20 = pd.read_csv(TOP10_PATH, dtype={"player_id": str})
    if len(top20) != 20:
        raise ValueError(f"{TOP10_PATH} must contain exactly 20 players")
    metrics = pd.read_parquet(METRICS_PATH)[["player_id", "player_name"]].copy()
    metrics["player_id"] = metrics["player_id"].astype(str)
    if metrics["player_id"].duplicated().any():
        raise ValueError(f"{METRICS_PATH} contains duplicate player_id values")
    attributes = _read_attributes()
    if set(attributes["player_id"]) != set(top20["player_id"]):
        raise ValueError("player_attributes.csv IDs must exactly match top10.csv")
    attributes = attributes.rename(columns={"player_name": "attribute_player_name"})

    merged = top20.merge(
        metrics.rename(columns={"player_name": "metrics_player_name"}),
        on="player_id",
        how="left",
        validate="many_to_one",
    ).merge(
        attributes,
        on="player_id",
        how="left",
        validate="one_to_one",
    )
    merged["player_name"] = merged["metrics_player_name"].where(
        merged["metrics_player_name"].fillna("").str.strip().ne(""),
        merged["player_name"],
    )
    mismatched_names = merged[
        merged["player_name"].str.strip().str.casefold()
        != merged["display_name"].str.strip().str.casefold()
    ]
    if not mismatched_names.empty:
        ids = ", ".join(mismatched_names["player_id"].astype(str))
        raise ValueError(f"Player name mismatch between top10.csv and metrics for IDs: {ids}")
    mismatched_attributes = merged[
        merged["attribute_player_name"].str.strip().str.casefold()
        != merged["player_name"].str.strip().str.casefold()
    ]
    if not mismatched_attributes.empty:
        ids = ", ".join(mismatched_attributes["player_id"].astype(str))
        raise ValueError(f"Player name mismatch between attributes and source IDs: {ids}")

    rows: list[dict[str, Any]] = []
    for source in merged.to_dict(orient="records"):
        row = {
            "rank": int(source["rank"]),
            "player_id": source["player_id"],
            "player_name": source["player_name"],
            "role": source["role"],
            "grade": source["grade"],
            "pressure_balls": int(source["pressure_balls"]),
            "ipl_share": float(source["ipl_share"]),
        }
        for column in ATTRIBUTE_COLUMNS:
            value = source.get(column, "")
            row[column] = "" if pd.isna(value) else str(value).strip()
        if not row["rr_status"]:
            row["rr_status"] = "unknown"
        row["fit_flag"] = _fit_flag(row["rr_status"], row["overseas"])
        row["final_list"] = False
        rows.append(row)
    return sorted(rows, key=lambda row: row["rank"])


def _choose_reserve(
    reserves: list[dict[str, Any]],
    used_ids: set[str],
    current_overseas: int,
    outgoing_is_overseas: bool,
    *,
    domestic_only: bool = False,
) -> dict[str, Any] | None:
    for candidate in reserves:
        if candidate["fit_flag"] != "KEEP" or candidate["player_id"] in used_ids:
            continue
        overseas = _is_overseas(candidate["overseas"])
        if overseas is None or (domestic_only and overseas):
            continue
        next_count = current_overseas - int(outgoing_is_overseas) + int(overseas)
        if next_count <= 3:
            return candidate
    return None


def _build_final_list(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    picks = [row for row in rows if row["rank"] <= 10]
    reserves = [row for row in rows if row["rank"] > 10]
    final_list = list(picks)
    swaps: list[dict[str, Any]] = []
    messages: list[str] = []
    used_ids: set[str] = {row["player_id"] for row in final_list}
    overseas_count = sum(_is_overseas(row["overseas"]) is True for row in final_list)

    for outgoing in list(final_list):
        if outgoing["fit_flag"] != "SWAP":
            continue
        outgoing_overseas = _is_overseas(outgoing["overseas"]) is True
        reserve = _choose_reserve(
            reserves, used_ids, overseas_count, outgoing_overseas
        )
        if reserve is None:
            messages.append(f"No verified KEEP reserve available for rank {outgoing['rank']}.")
            continue
        final_list.remove(outgoing)
        final_list.append(reserve)
        used_ids.remove(outgoing["player_id"])
        used_ids.add(reserve["player_id"])
        overseas_count += int(_is_overseas(reserve["overseas"]) is True)
        overseas_count -= int(outgoing_overseas)
        swaps.append(
            {
                "old_rank": outgoing["rank"],
                "old_player_name": outgoing["player_name"],
                "new_rank": reserve["rank"],
                "new_player_name": reserve["player_name"],
                "reason": outgoing["rr_status"],
            }
        )

    while overseas_count > 3:
        outgoing = next(
            (
                row
                for row in sorted(final_list, key=lambda item: item["rank"], reverse=True)
                if _is_overseas(row["overseas"]) is True
            ),
            None,
        )
        if outgoing is None:
            break
        reserve = _choose_reserve(
            reserves,
            used_ids,
            overseas_count,
            True,
            domestic_only=True,
        )
        if reserve is None:
            messages.append("Overseas limit cannot be met with verified KEEP reserves.")
            break
        final_list.remove(outgoing)
        final_list.append(reserve)
        used_ids.remove(outgoing["player_id"])
        used_ids.add(reserve["player_id"])
        overseas_count -= 1
        swaps.append(
            {
                "old_rank": outgoing["rank"],
                "old_player_name": outgoing["player_name"],
                "new_rank": reserve["rank"],
                "new_player_name": reserve["player_name"],
                "reason": "overseas_limit",
            }
        )

    unknown_overseas = [
        row["player_name"]
        for row in final_list
        if _is_overseas(row["overseas"]) is None
    ]
    if unknown_overseas:
        messages.append(
            "Overseas limit cannot be confirmed for final-list players with blank status: "
            + ", ".join(unknown_overseas)
        )

    for row in final_list:
        row["final_list"] = True
    return sorted(final_list, key=lambda row: row["rank"]), swaps, messages


def _write_csv(path: Path, columns: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(
            {
                key: str(value).lower() if isinstance(value, bool) else value
                for key, value in row.items()
            }
            for row in rows
        )


def main() -> None:
    rows = _load_candidates()
    initial_overseas = sum(
        _is_overseas(row["overseas"]) is True for row in rows if row["rank"] <= 10
    )
    print("rank,name,role,grade,pressure_balls,ipl_share,rr_status,overseas,age,price_band,flag")
    for row in rows:
        print(
            f"{row['rank']},{row['player_name']},{row['role']},{row['grade']},"
            f"{row['pressure_balls']},{row['ipl_share']:.6f},{row['rr_status']},"
            f"{row['overseas']},{row['age']},{row['price_band']},{row['fit_flag']}"
        )
    print(f"Overseas among ranks 1-10: {initial_overseas}")
    if initial_overseas > 3:
        print("WARNING: ranks 1-10 exceed the limit of 3 overseas players.")

    final_list, swaps, messages = _build_final_list(rows)
    print("Final 10:")
    for row in final_list:
        print(f"{row['rank']},{row['player_name']}")
    print("Swaps:")
    if swaps:
        for swap in swaps:
            print(
                f"{swap['old_rank']} -> {swap['new_rank']}: "
                f"{swap['old_player_name']} -> {swap['new_player_name']} "
                f"({swap['reason']})"
            )
    else:
        print("None")
    for message in messages:
        print(message)

    final_ids = {row["player_id"] for row in final_list}
    for row in rows:
        row["final_list"] = row["player_id"] in final_ids
    _write_csv(SQUAD_FIT_PATH, FIT_COLUMNS, rows)
    _write_csv(SWAPS_PATH, SWAP_COLUMNS, swaps)


if __name__ == "__main__":
    main()
