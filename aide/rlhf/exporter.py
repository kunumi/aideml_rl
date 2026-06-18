"""Compatibility shim: delegate to offline_extractor."""
from __future__ import annotations

from pathlib import Path

from .offline_extractor import extract_logs_dir as extract_offline_decisions


def export_logs_dir(
    logs_dir: str | Path,
    out_path: str | Path,
    *,
    ctu_csv: str | Path = "data/ctu_datasets_info.csv",
    total_steps: int | None = None,
) -> int:
    """Write `offline_decisions.jsonl` rows (one per heuristic step)."""
    return extract_offline_decisions(logs_dir, ctu_csv, out_path, total_steps=total_steps)
