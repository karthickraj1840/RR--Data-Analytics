from __future__ import annotations

from pathlib import Path

try:
    from .run_all import run_pipeline
except ImportError:
    from run_all import run_pipeline

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
