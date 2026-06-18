#!/usr/bin/env python3
"""Smoke-test Hugging Face dataset repo access (read + optional write probe)."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
load_dotenv()

from data.hf_utils import HF_DATA_PREFIX, HF_REPO, HF_RUNS_PREFIX, list_hf_paths, upload_folder_to_hf


def _ok(msg: str) -> None:
    print(f"[OK]   {msg}")


def _fail(msg: str) -> None:
    print(f"[FAIL] {msg}")


def _info(msg: str) -> None:
    print(f"[INFO] {msg}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--hf_repo", default=HF_REPO)
    p.add_argument("--hf_revision", default="main")
    p.add_argument("--skip-write", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    _info(f"repo={args.hf_repo}  revision={args.hf_revision}")
    _info(f"HF token: {'set' if token else 'not set (public read only)'}")

    try:
        data_paths = list_hf_paths(HF_DATA_PREFIX, repo_id=args.hf_repo, revision=args.hf_revision, limit=10)
        runs_paths = list_hf_paths(HF_RUNS_PREFIX, repo_id=args.hf_repo, revision=args.hf_revision, limit=10)
    except Exception as exc:
        _fail(f"Could not list repo files: {exc}")
        return 1

    _ok(f"Listed {len(data_paths)} path(s) under {HF_DATA_PREFIX}/")
    for p in data_paths[:5]:
        print(f"       - {p}")
    _ok(f"Listed {len(runs_paths)} path(s) under {HF_RUNS_PREFIX}/")
    for p in runs_paths[:5]:
        print(f"       - {p}")

    if args.skip_write:
        print("\nHF connection check passed (read-only).")
        return 0

    if not token:
        _fail("Write probe requires HF_TOKEN or HUGGING_FACE_HUB_TOKEN")
        return 1

    run_id = uuid.uuid4().hex[:8]
    probe = {
        "probe": "aide-hf-check",
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with tempfile.TemporaryDirectory() as tmp:
        probe_dir = Path(tmp) / "probe"
        probe_dir.mkdir()
        (probe_dir / "probe.json").write_text(json.dumps(probe, indent=2), encoding="utf-8")
        remote = f"{HF_RUNS_PREFIX}/_connectivity_probe/{run_id}"
        try:
            upload_folder_to_hf(probe_dir, repo_id=args.hf_repo, path_in_repo=remote, revision=args.hf_revision)
            _ok(f"Uploaded write probe to hf://{args.hf_repo}/{remote}/")
        except Exception as exc:
            _fail(f"Write probe failed: {exc}")
            return 1

    print("\nHF connection check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
