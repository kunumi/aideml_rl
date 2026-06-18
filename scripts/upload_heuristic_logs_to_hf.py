#!/usr/bin/env python3
"""Upload existing AIDE heuristic log directories to Hugging Face (logs only)."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv

load_dotenv()

from data.hf_utils import (
    HF_REPO,
    aide_run_exists_on_hf,
    list_repo_files_cached,
    upload_aide_log_dir,
)

_EXP_DIR_RE = re.compile(r"^(?:\d+-)?(?P<task>.+)__seed(?P<seed>\d+)$")


def _parse_exp_dir(name: str) -> tuple[str, int] | None:
    m = _EXP_DIR_RE.match(name)
    if not m:
        return None
    return m.group("task"), int(m.group("seed"))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--logs_dir", type=str, default="data/heuristic_runs/logs")
    p.add_argument("--hf_repo", type=str, default=HF_REPO)
    p.add_argument("--hf_revision", type=str, default="main")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--skip_uploaded", action="store_true")
    args = p.parse_args()

    logs_root = Path(args.logs_dir)
    if not logs_root.is_dir():
        raise SystemExit(f"Not a directory: {logs_root}")

    repo_files = None
    if args.skip_uploaded:
        print(f"Listing files on hf://{args.hf_repo} ({args.hf_revision})...")
        repo_files = list_repo_files_cached(
            repo_id=args.hf_repo,
            revision=args.hf_revision,
        )

    uploaded = 0
    skipped = 0
    for child in sorted(logs_root.iterdir()):
        if not child.is_dir() or not (child / "journal.json").is_file():
            continue
        parsed = _parse_exp_dir(child.name)
        if parsed is None:
            print(f"skip (unrecognized name): {child.name}")
            continue
        task_name, seed = parsed
        if args.skip_uploaded and aide_run_exists_on_hf(
            task_name,
            seed,
            repo_files,
            repo_id=args.hf_repo,
            revision=args.hf_revision,
        ):
            print(f"skip (already on HF): {task_name} seed{seed}")
            skipped += 1
            continue
        if args.dry_run:
            print(f"would upload {child} -> runs/{task_name}/seed{seed}/")
            uploaded += 1
            continue
        uri = upload_aide_log_dir(
            child,
            task_name=task_name,
            seed=seed,
            repo_id=args.hf_repo,
            revision=args.hf_revision,
        )
        print(uri)
        uploaded += 1

    print(f"Done. {'Would upload' if args.dry_run else 'Uploaded'}: {uploaded}, skipped: {skipped}")


if __name__ == "__main__":
    main()
