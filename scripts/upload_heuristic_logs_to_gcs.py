#!/usr/bin/env python3
"""Upload existing AIDE heuristic log directories to GCS (logs only)."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data.data_utils import upload_aide_log_dir

# e.g. 2-ctu-legalacts_legalacts-original__seed0
_EXP_DIR_RE = re.compile(r"^(?:\d+-)?(?P<task>.+)__seed(?P<seed>\d+)$")


def _parse_exp_dir(name: str) -> tuple[str, int] | None:
    m = _EXP_DIR_RE.match(name)
    if not m:
        return None
    return m.group("task"), int(m.group("seed"))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--logs_dir", type=str, default="data/heuristic_runs/logs")
    p.add_argument("--gcs_bucket", type=str, default="benchmark-public-data")
    p.add_argument("--gcs_prefix", type=str, default="aide-runs")
    p.add_argument("--gcs_project", type=str, default="numi-platform")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    logs_root = Path(args.logs_dir)
    if not logs_root.is_dir():
        raise SystemExit(f"Not a directory: {logs_root}")

    for child in sorted(logs_root.iterdir()):
        if not child.is_dir():
            continue
        if not (child / "journal.json").is_file():
            continue
        parsed = _parse_exp_dir(child.name)
        if parsed is None:
            print(f"skip (unrecognized name): {child.name}")
            continue
        task_name, seed = parsed
        if args.dry_run:
            print(f"would upload {child} -> aide-runs/{task_name}/seed{seed}/")
            continue
        uri = upload_aide_log_dir(
            child,
            task_name=task_name,
            seed=seed,
            bucket_name=args.gcs_bucket,
            gcs_prefix=args.gcs_prefix,
            project=args.gcs_project,
        )
        print(uri)


if __name__ == "__main__":
    main()
