#!/usr/bin/env python3
"""Download AIDE heuristic run logs from Hugging Face (runs/ prefix)."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import dotenv

dotenv.load_dotenv()

from data.hf_utils import HF_REPO, HF_RUNS_PREFIX, list_repo_files_cached

_RUN_PATH_RE = re.compile(
    rf"^{re.escape(HF_RUNS_PREFIX)}/(?P<task>[^/]+)/seed(?P<seed>\d+)/(?P<rel>.+)$"
)


def main() -> None:
    p = argparse.ArgumentParser(
        description=f"Download all runs from hf://<repo>/{HF_RUNS_PREFIX}/"
    )
    p.add_argument("--hf_repo", type=str, default=HF_REPO)
    p.add_argument("--hf_revision", type=str, default="main")
    p.add_argument("--dest_dir", type=str, default="data/heuristic_runs/logs")
    p.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip runs that already have journal.json locally",
    )
    args = p.parse_args()

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    dest_root = Path(args.dest_dir)
    dest_root.mkdir(parents=True, exist_ok=True)

    print(f"Listing files on hf://{args.hf_repo}/{HF_RUNS_PREFIX}/ ({args.hf_revision})...")
    repo_files = list_repo_files_cached(
        repo_id=args.hf_repo,
        revision=args.hf_revision,
        token=token,
    )

    run_files = sorted(
        path
        for path in repo_files
        if path.startswith(f"{HF_RUNS_PREFIX}/") and _RUN_PATH_RE.match(path)
    )
    if not run_files:
        raise SystemExit(f"No files found under {HF_RUNS_PREFIX}/ in {args.hf_repo}")

    runs_seen: set[tuple[str, int]] = set()
    downloaded = 0
    skipped_runs = 0

    for remote_path in run_files:
        m = _RUN_PATH_RE.match(remote_path)
        assert m is not None
        task_name = m.group("task")
        seed = int(m.group("seed"))
        rel = m.group("rel")
        run_key = (task_name, seed)

        dest = dest_root / f"{task_name}__seed{seed}"
        if run_key not in runs_seen:
            runs_seen.add(run_key)
            if args.skip_existing and (dest / "journal.json").is_file():
                print(f"skip (already local): {task_name} seed{seed}")
                skipped_runs += 1

        if args.skip_existing and (dest / "journal.json").is_file():
            continue

        local_path = dest / rel
        local_path.parent.mkdir(parents=True, exist_ok=True)
        cached = hf_hub_download(
            repo_id=args.hf_repo,
            filename=remote_path,
            repo_type="dataset",
            revision=args.hf_revision,
            token=token,
        )
        shutil.copy2(cached, local_path)
        downloaded += 1

    print(
        f"Done. Downloaded {downloaded} files across {len(runs_seen)} runs "
        f"to {dest_root} ({skipped_runs} runs skipped)"
    )


if __name__ == "__main__":
    main()
