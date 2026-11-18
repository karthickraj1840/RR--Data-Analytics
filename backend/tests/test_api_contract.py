from __future__ import annotations

import io
import zipfile

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.models import (
    BacktestResponse,
    ConfigResponse,
    DataQualityResponse,
    HealthResponse,
    LeaderboardRow,
    PlayerExplainResponse,
    PlayerProfileResponse,
    SensitivityResponse,
    ShortlistEntry,
    ShortlistReloadResponse,
    SquadFitResponse,
    SimilarPlayer,
    Top10Response,
    WinProbabilityReportResponse,
)
from app.services import data_service


@pytest.fixture(scope="module")
def api_client(tmp_path_factory: pytest.TempPathFactory):
    original_path = data_service.SHORTLIST_PATH
    data_service.SHORTLIST_PATH = tmp_path_factory.mktemp("api") / "shortlist.json"
    data_service.reload_processed_data()
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        yield client
    data_service.SHORTLIST_PATH = original_path


def _sample_player_ids() -> list[int]:
    snapshot = data_service.get_snapshot()
    return [int(value) for value in snapshot.metrics["player_id"].head(4)]


def _validate(response, model):
    assert response.status_code == 200, response.text
    return model.model_validate(response.json())


def test_health_contract(api_client: TestClient):
    body = _validate(api_client.get("/api/health"), HealthResponse)
    assert body.status == "ok"
    assert set(body.innings_log_loss) == {"1", "2"}


def test_data_quality_contract(api_client: TestClient):
    body = _validate(api_client.get("/api/data-quality"), DataQualityResponse)
    assert body.matches_used > 0
    assert body.balls_used > 0
    assert body.by_comp_group
    assert body.by_season


def test_win_probability_contract(api_client: TestClient):
    body = _validate(
        api_client.get("/api/win-probability/model-report"),
        WinProbabilityReportResponse,
    )
    assert set(body.innings) == {"1", "2"}
    assert all(len(item.calibration_bins) == 10 for item in body.innings.values())


def test_config_contract(api_client: TestClient):
    body = _validate(api_client.get("/api/config"), ConfigResponse)
    assert body.config["min_ipl_share"] == 0.3


def test_top10_contract(api_client: TestClient):
    body = _validate(api_client.get("/api/top10"), Top10Response)
    assert len(body.picks) == 10
    assert len(body.reserves) == 10
    assert set(body.grade_counts) == {"top10", "top20"}
    assert all(
        entry.role == entry.role_label
        and entry.role in {"batter", "bowler", "all_rounder"}
        for entry in body.picks
    )
    assert "count only once" in body.role_constraint["note"]


def test_top10_explain_contract(api_client: TestClient):
    player_id = int(data_service.get_snapshot().top10["combined_top10"][0]["player_id"])
    body = _validate(
        api_client.get(f"/api/top10/explain/{player_id}"),
        PlayerExplainResponse,
    )
    assert body.player_id == player_id
    assert body.inclusion_probability is not None
    assert body.rank_stability is not None


def test_leaderboard_contract(api_client: TestClient):
    response = api_client.post(
        "/api/leaderboard",
        json={
            "role": "batter",
            "comp_group": "IPL",
            "min_pressure_balls": 30,
            "pressure_definition": "composite",
            "index_weights": {
                "wpa": 0.4,
                "runs_saved": 0.4,
                "consistency": 0.2,
            },
        },
    )
    assert response.status_code == 200, response.text
    rows = [LeaderboardRow.model_validate(item) for item in response.json()]
    assert rows
    assert rows[0].pressure_balls >= 30
    assert len(rows[0].wpa_80_interval) == 2
    assert isinstance(rows[0].small_sample, bool)


def test_player_profile_contract(api_client: TestClient):
    player_id = _sample_player_ids()[0]
    body = _validate(
        api_client.get(f"/api/players/{player_id}"),
        PlayerProfileResponse,
    )
    assert body.player_id == player_id
    assert isinstance(body.pressure_vs_normal, dict)
    assert isinstance(body.season_trend, list)


def test_similar_players_contract(api_client: TestClient):
    player_id = _sample_player_ids()[0]
    response = api_client.get(f"/api/players/{player_id}/similar")
    assert response.status_code == 200, response.text
    assert all(
        SimilarPlayer.model_validate(item) for item in response.json()
    )


def test_compare_contract(api_client: TestClient):
    ids = _sample_player_ids()[:2]
    response = api_client.post("/api/compare", json={"player_ids": ids})
    assert response.status_code == 200, response.text
    players = [PlayerProfileResponse.model_validate(item) for item in response.json()]
    assert [player.player_id for player in players] == ids


def test_sensitivity_contract(api_client: TestClient):
    body = _validate(api_client.get("/api/sensitivity"), SensitivityResponse)
    assert body.players
    assert all(0 <= player.inclusion_probability <= 1 for player in body.players)


def test_backtest_contract(api_client: TestClient):
    body = _validate(api_client.get("/api/backtest"), BacktestResponse)
    assert body.folds
    assert body.selection_comparison["overlap_count"] >= 0


def test_squad_fit_contract(
    api_client: TestClient,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    squad_path = tmp_path / "squad_fit.csv"
    swaps_path = tmp_path / "squad_fit_swaps.csv"
    squad_path.write_text(
        "rank,player_name,role,grade,pressure_balls,ipl_share,rr_status,overseas,age,price_band,fit_flag,final_list\n"
        "1,Player One,batter,A,10,0.5,unknown,,,,REVIEW,true\n",
        encoding="utf-8",
    )
    swaps_path.write_text(
        "old_rank,old_player_name,new_rank,new_player_name,reason\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(data_service, "SQUAD_FIT_PATH", squad_path)
    monkeypatch.setattr(data_service, "SQUAD_FIT_SWAPS_PATH", swaps_path)
    body = _validate(api_client.get("/api/squad-fit"), SquadFitResponse)
    assert body.templates == {
        "squad_fit": "squad_fit.csv",
        "swaps": "squad_fit_swaps.csv",
    }
    assert body.rows[0]["rr_status"] == "unknown"
    assert body.rows[0]["overseas"] == "not_provided"
    assert body.swaps == []


def test_shortlist_get_contract(api_client: TestClient):
    response = api_client.get("/api/shortlist")
    assert response.status_code == 200, response.text
    assert response.json() == []


def test_shortlist_put_contract(api_client: TestClient):
    ids = _sample_player_ids()[:2]
    response = api_client.put(
        "/api/shortlist",
        json=[
            {"player_id": ids[1], "order": 2},
            {"player_id": ids[0], "order": 1, "rationale": "First user rationale"},
        ],
    )
    assert response.status_code == 200, response.text
    entries = [ShortlistEntry.model_validate(item) for item in response.json()]
    assert [entry.order for entry in entries] == [1, 2]
    assert entries[0].rationale == "First user rationale"
    assert entries[1].rationale == ""
    assert api_client.get("/api/shortlist").json() == response.json()


def test_shortlist_export_contract(api_client: TestClient):
    player_id = _sample_player_ids()[0]
    api_client.put(
        "/api/shortlist",
        json=[{"player_id": player_id, "order": 1, "rationale": "User-authored note"}],
    )
    response = api_client.post("/api/export/shortlist")
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "application/zip"
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        assert set(archive.namelist()) == {"shortlist.csv", "shortlist.md"}
        markdown = archive.read("shortlist.md").decode("utf-8")
        assert "User-authored note" in markdown
        assert "not provided" not in markdown.lower()


def test_admin_reload_contract(api_client: TestClient):
    body = _validate(api_client.post("/api/admin/reload"), ShortlistReloadResponse)
    assert body.data_version


def test_unknown_player_returns_clear_json_error(api_client: TestClient):
    response = api_client.get("/api/players/999999999")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "player_not_found"


def test_invalid_compare_request_returns_clear_json_error(api_client: TestClient):
    response = api_client.post("/api/compare", json={"player_ids": [1]})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "request_validation_error"
    assert response.json()["error"]["details"]


def test_admin_reload_rejects_non_local_client():
    with TestClient(app, client=("192.0.2.10", 50000)) as client:
        response = client.post("/api/admin/reload")
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "localhost_only"


def test_cors_allows_only_requested_local_origins(api_client: TestClient):
    allowed = api_client.get(
        "/api/health", headers={"Origin": "http://localhost:5174"}
    )
    disallowed = api_client.get(
        "/api/health", headers={"Origin": "http://127.0.0.1:5173"}
    )
    assert allowed.headers.get("access-control-allow-origin") == "http://localhost:5174"
    assert "access-control-allow-origin" not in disallowed.headers
