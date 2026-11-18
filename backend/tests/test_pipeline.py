from pathlib import Path

import pandas as pd


def test_pipeline_builds_outputs():
    from pipeline.run_all import run_pipeline

    result = run_pipeline()
    assert result["rows"] > 0
    assert result["matches"] > 0
    processed = Path(__file__).resolve().parents[1] / "data" / "processed"
    assert (processed / "ball_by_ball.parquet").exists()
    assert (processed / "data_quality.json").exists()
    assert (processed / "player_metrics.parquet").exists()


def test_legal_ball_rule():
    from pipeline.run_all import run_pipeline
    run_pipeline()
    processed = Path(__file__).resolve().parents[1] / "data" / "processed"
    bbb = pd.read_parquet(processed / "ball_by_ball.parquet")
    legal = bbb["is_legal"].mean()
    assert 0.0 < legal < 1.0
