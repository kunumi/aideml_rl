#!/usr/bin/env python3
"""
Materialize all CTU tasks from relbench and upload parquet bundles to Hugging Face.

Destination layout:
  guilhermedrud/ctu_datasets/data/<task_row_name>/{train,val,test}.parquet
  guilhermedrud/ctu_datasets/data/<task_row_name>/db_tables/*.parquet
  guilhermedrud/ctu_datasets/data/ctu_datasets_info.csv
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

from tqdm import tqdm

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from aide.rlhf.ctu_dataset import load_ctu_index, materialize_workspace_from_relbench
from data.hf_utils import (
    HF_REPO,
    ctu_task_exists_on_hf,
    list_repo_files_cached,
    upload_ctu_task_data,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--csv_path", type=str, default="data/ctu_datasets_info.csv")
    p.add_argument("--hf_repo", type=str, default=HF_REPO)
    p.add_argument("--hf_revision", type=str, default="main")
    p.add_argument("--task_offset", type=int, default=0)
    p.add_argument("--max_tasks", type=int, default=None, help="Upload first N tasks (default: all).")
    p.add_argument("--task_name", type=str, default=None, help="Upload a single task by row name.")
    p.add_argument("--skip_uploaded", action="store_true", help="Skip tasks already present on HF.")
    p.add_argument("--work_dir", type=str, default=None, help="Reuse a local staging dir.")
    p.add_argument(
        "--skip_errors",
        action="store_true",
        help="Log and continue when a task fails to materialize.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tasks = load_ctu_index(args.csv_path)

    if args.task_name:
        slice_tasks = [t for t in tasks if t.row_name == args.task_name]
        if not slice_tasks:
            raise SystemExit(f"Unknown task: {args.task_name}")
    else:
        end = None if args.max_tasks is None else args.task_offset + args.max_tasks
        slice_tasks = tasks[args.task_offset:end]

    work_root = Path(args.work_dir) if args.work_dir else Path(tempfile.mkdtemp(prefix="ctu_hf_upload_"))
    work_root.mkdir(parents=True, exist_ok=True)
    failed: list[str] = []

    repo_files = None
    if args.skip_uploaded:
        print(f"Listing files on hf://{args.hf_repo} ({args.hf_revision})...")
        repo_files = list_repo_files_cached(
            repo_id=args.hf_repo,
            revision=args.hf_revision,
        )
        print(f"Found {len(repo_files)} file(s) in repo.")

    for task in tqdm(slice_tasks, desc="upload ctu->hf"):
        if args.skip_uploaded and ctu_task_exists_on_hf(
            task.row_name,
            repo_files,
            repo_id=args.hf_repo,
            revision=args.hf_revision,
        ):
            tqdm.write(f"skip (already on HF): {task.row_name}")
            continue

        staging = work_root / task.row_name
        if staging.exists():
            shutil.rmtree(staging)
        try:
            materialize_workspace_from_relbench(task, staging)
            upload_ctu_task_data(
                staging / "input",
                task.row_name,
                repo_id=args.hf_repo,
                revision=args.hf_revision,
            )
        except Exception as exc:
            if not args.skip_errors:
                raise
            print(f"FAILED {task.row_name}: {type(exc).__name__}: {exc}")
            failed.append(task.row_name)

    from data.hf_utils import upload_file_to_hf

    upload_file_to_hf(
        args.csv_path,
        "data/ctu_datasets_info.csv",
        repo_id=args.hf_repo,
        revision=args.hf_revision,
    )
    if failed:
        print(f"Failed tasks ({len(failed)}): {', '.join(failed)}")
    print(f"Done. Data repo: https://huggingface.co/datasets/{args.hf_repo}/tree/{args.hf_revision}/data")


if __name__ == "__main__":
    main()
