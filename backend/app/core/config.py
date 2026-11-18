from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT / "config.yaml"


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        return {
            "pressure": {"high_leverage_quantile": 0.85},
            "league_weights": {"IPL": 1.0, "International": 0.8, "Emerging": 0.4},
            "index_weights": {"wpa": 0.4, "runs_saved": 0.3, "consistency": 0.3},
            "min_balls": 30,
            "opposition_adjustment": False,
        }
    with CONFIG_PATH.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}
