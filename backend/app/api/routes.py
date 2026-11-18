from __future__ import annotations

import csv
import io
import zipfile
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import Response

from app.schemas.models import (
    BacktestResponse,
    CompareRequest,
    ConfigResponse,
    DataQualityResponse,
    HealthResponse,
    LeaderboardRequest,
    LeaderboardRow,
    PlayerExplainResponse,
    PlayerProfileResponse,
    SensitivityResponse,
    ShortlistEntry,
    ShortlistInput,
    ShortlistReloadResponse,
    SquadFitResponse,
    SimilarPlayer,
    Top10Response,
    WinProbabilityReportResponse,
)
from app.services import data_service
from app.services.data_service import APIDataError

router = APIRouter(prefix="/api")


@router.get("/health", response_model=HealthResponse)
def health() -> dict[str, Any]:
    return data_service.get_health()


@router.get("/data-quality", response_model=DataQualityResponse)
def data_quality() -> dict[str, Any]:
    return data_service.get_data_quality()


@router.get(
    "/win-probability/model-report",
    response_model=WinProbabilityReportResponse,
)
def win_model_report() -> dict[str, Any]:
    return data_service.get_win_report()


@router.get("/config", response_model=ConfigResponse)
def config() -> dict[str, Any]:
    return data_service.get_config()


@router.get("/top10", response_model=Top10Response)
def top10() -> dict[str, Any]:
    return data_service.get_top10()


@router.get("/top10/explain/{player_id}", response_model=PlayerExplainResponse)
def top10_explain(player_id: int) -> dict[str, Any]:
    return data_service.explain_top10(player_id)


@router.post("/leaderboard", response_model=list[LeaderboardRow])
def leaderboard(request: LeaderboardRequest) -> list[dict[str, Any]]:
    return data_service.get_leaderboard(
        request.model_dump(exclude_none=True)
    )


@router.get("/players/{player_id}", response_model=PlayerProfileResponse)
def player_profile(player_id: int) -> dict[str, Any]:
    return data_service.get_player_profile(player_id)


@router.get("/players/{player_id}/similar", response_model=list[SimilarPlayer])
def similar_players(player_id: int, limit: int = 10) -> list[dict[str, Any]]:
    if limit < 1 or limit > 20:
        raise APIDataError(
            "limit must be between 1 and 20",
            status_code=422,
            code="invalid_limit",
        )
    return data_service.get_similar_players(player_id, limit)


@router.post("/compare", response_model=list[PlayerProfileResponse])
def compare(payload: CompareRequest) -> list[dict[str, Any]]:
    return data_service.compare_players(payload.player_ids)


@router.get("/sensitivity", response_model=SensitivityResponse)
def sensitivity() -> dict[str, Any]:
    return data_service.get_sensitivity()


@router.get("/backtest", response_model=BacktestResponse)
def backtest() -> dict[str, Any]:
    return data_service.get_backtest()


@router.get("/squad-fit", response_model=SquadFitResponse)
def squad_fit() -> dict[str, Any]:
    return data_service.get_squad_fit()


@router.get("/shortlist", response_model=list[ShortlistEntry])
def shortlist_get() -> list[dict[str, Any]]:
    return data_service.get_shortlist()


@router.put("/shortlist", response_model=list[ShortlistEntry])
def shortlist_put(payload: list[ShortlistInput]) -> list[dict[str, Any]]:
    entries = [
        {
            "player_id": item.player_id,
            "order": item.order or index,
            "rationale": item.rationale,
        }
        for index, item in enumerate(payload, start=1)
    ]
    return data_service.save_shortlist(entries)


@router.post("/export/shortlist", response_class=Response)
def shortlist_export() -> Response:
    entries = data_service.get_shortlist()
    csv_buffer = io.StringIO(newline="")
    writer = csv.writer(csv_buffer)
    writer.writerow(("order", "player_id", "rationale"))
    for entry in entries:
        writer.writerow((entry["order"], entry["player_id"], entry["rationale"]))

    markdown_lines = ["# Shortlist", ""]
    for entry in entries:
        markdown_lines.append(f"- Player ID: {entry['player_id']}")
        if entry["rationale"]:
            markdown_lines.append(f"  - Rationale: {entry['rationale']}")
    markdown_lines.append("")

    archive = io.BytesIO()
    with zipfile.ZipFile(archive, mode="w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("shortlist.csv", csv_buffer.getvalue())
        bundle.writestr("shortlist.md", "\n".join(markdown_lines))
    return Response(
        content=archive.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="shortlist-export.zip"'},
    )


@router.post("/admin/reload", response_model=ShortlistReloadResponse)
def reload_processed(request: Request) -> dict[str, str]:
    client_host = request.client.host if request.client else ""
    if client_host not in {"127.0.0.1", "::1", "localhost"}:
        raise APIDataError(
            "This endpoint is available only to localhost clients",
            status_code=403,
            code="localhost_only",
        )
    return {
        "status": "reloaded",
        "data_version": data_service.reload_processed_data(),
    }
