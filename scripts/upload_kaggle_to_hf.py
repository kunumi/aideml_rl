#!/usr/bin/env python3
"""
Download every Kaggle dataset from GCS, build train/val/test parquets, and upload
the bundles to the existing Hugging Face dataset repo under a new ``kaggle/`` folder.

Source layout (GCS bucket benchmark-public-data):
  kaggle/<dataset>/clean_data.csv.zip
  kaggle/<dataset>/data.csv
  kaggle/<dataset>/data_info.json

Destination layout (HF dataset repo, e.g. guilhermedrud/ctu_datasets):
  kaggle/<dataset>/{train,val,test}.parquet
  kaggle/kaggle_datasets_info.csv

A local copy of the index is also written to data/kaggle_datasets_info.csv.

Auth:
  - GCS:  Application Default Credentials (`gcloud auth application-default login`)
          or GOOGLE_APPLICATION_CREDENTIALS pointing at a service-account key.
  - HF:   HF_TOKEN / HUGGING_FACE_HUB_TOKEN env var (needs write access).

HF rate limits:
  Free accounts are capped at ~128 repo commits per hour. This script uploads the
  entire local ``kaggle/`` staging folder in **one commit** at the end. If you
  already hit the limit, wait ~1 hour, then either re-run normally (with
  ``--skip_uploaded``) or upload existing staging with ``--upload_only``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from data.data_utils import download_blob_prefix, list_blobs
from data.hf_utils import (
    HF_KAGGLE_PREFIX,
    HF_REPO,
    kaggle_task_exists_on_hf,
    list_repo_files_cached,
    upload_folder_to_hf,
)

GCS_BUCKET = "benchmark-public-data"
GCS_KAGGLE_PREFIX = "kaggle"

# data_info.json "task" -> CTU-style task_type used across the codebase.
TASK_TYPE_MAP = {
    "binary": "binary_classification",
    "multi": "multiclass_classification",
    "regression": "regression",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bucket", default=GCS_BUCKET)
    p.add_argument("--gcs_prefix", default=GCS_KAGGLE_PREFIX)
    p.add_argument("--hf_repo", default=HF_REPO)
    p.add_argument("--hf_revision", default="main")
    p.add_argument(
        "--raw_dir",
        default="data/kaggle_raw",
        help="Local cache for files downloaded from GCS.",
    )
    p.add_argument(
        "--work_dir",
        default="data/kaggle_materialized",
        help="Local staging dir for the parquet bundles uploaded to HF.",
    )
    p.add_argument(
        "--info_csv",
        default="data/kaggle_datasets_info.csv",
        help="Where to write the local kaggle_datasets_info.csv.",
    )
    p.add_argument("--val_size", type=float, default=0.1)
    p.add_argument("--test_size", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--dataset",
        default=None,
        help=(
            "Process a single dataset by GCS folder name or slug "
            "(e.g. --dataset='-lionel-messi-all-club-goals' or "
            "--dataset=lionel-messi-all-club-goals)."
        ),
    )
    p.add_argument("--max_datasets", type=int, default=None)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument(
        "--skip_uploaded",
        action="store_true",
        help="Skip datasets that already have train.parquet on HF.",
    )
    p.add_argument(
        "--no_upload",
        action="store_true",
        help="Build parquets + CSV locally but do not push to HF.",
    )
    p.add_argument(
        "--upload_only",
        action="store_true",
        help="Upload existing local staging under --work_dir/kaggle/ (no GCS download).",
    )
    p.add_argument(
        "--rebuild_info_csv",
        action="store_true",
        help=(
            "Rebuild kaggle_datasets_info.csv from data_info.json in --raw_dir "
            "for all staged datasets (also runs before --upload_only)."
        ),
    )
    p.add_argument(
        "--skip_errors",
        action="store_true",
        help="Log and continue when one dataset fails.",
    )
    return p.parse_args()


def slugify(folder_name: str) -> str:
    """GCS folder name -> HF/parquet-friendly dataset slug."""
    return folder_name.strip("/").strip("-").strip()


def staging_root(work_dir: str | Path) -> Path:
    return Path(work_dir) / HF_KAGGLE_PREFIX


def _dataset_parquet_dir(dataset_dir: Path) -> Path | None:
    """Return the dir containing train.parquet for a staged dataset, if any."""
    for candidate in (dataset_dir / "input", dataset_dir):
        if (candidate / "train.parquet").is_file():
            return candidate
    return None


def normalize_staging(
    work_dir: str | Path,
    *,
    info_csv: str | Path | None = None,
) -> int:
    """
    Ensure ``<work_dir>/kaggle/<slug>/{train,val,test}.parquet`` exists.

    Older runs wrote ``<work_dir>/<slug>/input/*.parquet``; those are moved here.
    Returns the number of datasets present in the normalized staging tree.
    """
    work_dir = Path(work_dir)
    staging = staging_root(work_dir)
    staging.mkdir(parents=True, exist_ok=True)

    migrated = 0
    for child in sorted(work_dir.iterdir()):
        if not child.is_dir() or child.name == HF_KAGGLE_PREFIX:
            continue
        parquet_dir = _dataset_parquet_dir(child)
        if parquet_dir is None:
            continue
        dest = staging / child.name
        dest.mkdir(parents=True, exist_ok=True)
        for split in ("train", "val", "test"):
            src = parquet_dir / f"{split}.parquet"
            if src.is_file() and not (dest / f"{split}.parquet").is_file():
                shutil.move(str(src), str(dest / f"{split}.parquet"))
        migrated += 1
        # Drop empty legacy dirs left behind after migration.
        if parquet_dir.is_dir() and parquet_dir != dest and not any(parquet_dir.iterdir()):
            parquet_dir.rmdir()
        if child.is_dir() and child != dest and not any(child.iterdir()):
            child.rmdir()

    if info_csv is not None:
        info_path = Path(info_csv)
        if info_path.is_file():
            shutil.copy2(info_path, staging / "kaggle_datasets_info.csv")

    return sum(1 for d in staging.iterdir() if d.is_dir() and (d / "train.parquet").is_file())


def resolve_dataset_folder(name: str) -> str:
    """
    Accept either the raw GCS folder name (``-foo-bar``) or the slug (``foo-bar``).
    """
    name = name.strip().strip("/")
    if name.startswith("-"):
        return name
    return f"-{name}"


def list_kaggle_datasets(bucket: str, prefix: str) -> list[str]:
    """Return the distinct dataset folder names directly under ``prefix/``."""
    prefix = prefix.strip("/")
    names = list_blobs(bucket, f"{prefix}/")
    datasets: set[str] = set()
    for name in names:
        rest = name[len(prefix) + 1 :]
        if "/" not in rest:
            continue
        datasets.add(rest.split("/", 1)[0])
    return sorted(datasets)


def read_clean_data(dataset_dir: Path) -> pd.DataFrame:
    """Load clean_data.csv.zip (preferred) or data.csv from a dataset dir."""
    zip_path = dataset_dir / "clean_data.csv.zip"
    if zip_path.is_file():
        with zipfile.ZipFile(zip_path) as zf:
            csv_members = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if not csv_members:
                raise FileNotFoundError(f"No .csv inside {zip_path}")
            with zf.open(csv_members[0]) as fh:
                return pd.read_csv(fh)

    csv_path = dataset_dir / "data.csv"
    if csv_path.is_file():
        return pd.read_csv(csv_path)

    raise FileNotFoundError(f"No clean_data.csv.zip or data.csv in {dataset_dir}")


def _assign_splits(n: int, val_size: float, test_size: float, rng: np.random.Generator) -> np.ndarray:
    """Return an array of 'train'/'val'/'test' labels for ``n`` shuffled rows."""
    perm = rng.permutation(n)
    n_test = int(round(n * test_size))
    n_val = int(round(n * val_size))
    # Guarantee at least one train row when the dataset is tiny.
    n_test = min(n_test, max(n - 2, 0))
    n_val = min(n_val, max(n - n_test - 1, 0))
    labels = np.empty(n, dtype=object)
    labels[perm[:n_test]] = "test"
    labels[perm[n_test:n_test + n_val]] = "val"
    labels[perm[n_test + n_val:]] = "train"
    return labels


def split_dataframe(
    df: pd.DataFrame,
    target_column: str,
    task_type: str,
    *,
    val_size: float,
    test_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split into train/val/test, stratifying classification when feasible."""
    rng = np.random.default_rng(seed)
    n = len(df)
    labels = np.empty(n, dtype=object)

    is_classification = task_type in {"binary_classification", "multiclass_classification"}
    can_stratify = False
    if is_classification and target_column in df.columns:
        counts = df[target_column].value_counts(dropna=False)
        # Need enough samples per class to place one in each of the 3 splits.
        can_stratify = bool(counts.min() >= 3 and len(counts) >= 2)

    if can_stratify:
        positions = np.arange(n)
        for _, group_pos in pd.Series(positions).groupby(
            df[target_column].to_numpy(), sort=False
        ):
            idx = group_pos.to_numpy()
            labels[idx] = _assign_splits(len(idx), val_size, test_size, rng)
    else:
        labels = _assign_splits(n, val_size, test_size, rng)

    train = df[labels == "train"].reset_index(drop=True)
    val = df[labels == "val"].reset_index(drop=True)
    test = df[labels == "test"].reset_index(drop=True)
    return train, val, test


def build_info_row(slug: str, data_info: dict) -> dict:
    """Build a kaggle_datasets_info.csv row mirroring the CTU index columns."""
    raw_task = str(data_info.get("task", "")).strip()
    task_type = TASK_TYPE_MAP.get(raw_task, raw_task)
    info = dict(data_info)
    info["slug"] = slug
    info["task_type"] = task_type
    return {
        "name": slug,
        "info": json.dumps(info, ensure_ascii=False),
        "task": task_type,
    }


def process_dataset(
    dataset_folder: str,
    args: argparse.Namespace,
) -> dict:
    slug = slugify(dataset_folder)
    raw_dir = Path(args.raw_dir) / dataset_folder
    remote_prefix = f"{args.gcs_prefix.strip('/')}/{dataset_folder}"

    download_blob_prefix(args.bucket, remote_prefix, raw_dir)

    info_path = raw_dir / "data_info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing data_info.json for {dataset_folder}")
    data_info = json.loads(info_path.read_text(encoding="utf-8"))

    target_column = data_info.get("target_column", "")
    raw_task = str(data_info.get("task", "")).strip()
    task_type = TASK_TYPE_MAP.get(raw_task, raw_task)

    df = read_clean_data(raw_dir)
    if target_column and target_column not in df.columns:
        raise ValueError(
            f"target_column '{target_column}' not in columns for {dataset_folder}"
        )

    train, val, test = split_dataframe(
        df,
        target_column,
        task_type,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
    )

    input_dir = staging_root(args.work_dir) / slug
    input_dir.mkdir(parents=True, exist_ok=True)
    train.to_parquet(input_dir / "train.parquet", index=False)
    val.to_parquet(input_dir / "val.parquet", index=False)
    test.to_parquet(input_dir / "test.parquet", index=False)

    return build_info_row(slug, data_info)


def write_info_csv(rows: list[dict], info_csv: Path, staging: Path | None = None) -> None:
    info_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=["name", "info", "task"])
    df.sort_values("name", inplace=True)
    df.to_csv(info_csv, index=False)
    if staging is not None:
        staging.mkdir(parents=True, exist_ok=True)
        df.to_csv(staging / "kaggle_datasets_info.csv", index=False)


def rebuild_info_csv(
    raw_dir: str | Path,
    *,
    work_dir: str | Path,
    info_csv: str | Path,
    staged_only: bool = True,
) -> int:
    """
    Rebuild the index CSV from ``data_info.json`` files under ``raw_dir``.

    By default only includes datasets that already have staged parquets.
    """
    raw_dir = Path(raw_dir)
    work_dir = Path(work_dir)
    staging = staging_root(work_dir)
    normalize_staging(work_dir)

    staged_slugs: set[str] = set()
    if staged_only and staging.is_dir():
        staged_slugs = {
            d.name
            for d in staging.iterdir()
            if d.is_dir() and (d / "train.parquet").is_file()
        }

    rows: list[dict] = []
    missing_info: list[str] = []
    for child in sorted(raw_dir.iterdir()):
        if not child.is_dir():
            continue
        slug = slugify(child.name)
        if staged_only and staged_slugs and slug not in staged_slugs:
            continue
        info_path = child / "data_info.json"
        if not info_path.is_file():
            missing_info.append(slug)
            continue
        data_info = json.loads(info_path.read_text(encoding="utf-8"))
        rows.append(build_info_row(slug, data_info))

    if missing_info:
        print(f"Warning: {len(missing_info)} staged dataset(s) missing data_info.json in {raw_dir}")

    write_info_csv(rows, Path(info_csv), staging=staging)
    return len(rows)


def upload_kaggle_staging(
    work_dir: str | Path,
    *,
    hf_repo: str,
    hf_revision: str,
    info_csv: str | Path | None = None,
) -> str:
    """
    Upload the entire local ``kaggle/`` staging tree in a single HF commit.

    This avoids the free-tier limit of ~128 repo commits per hour that happens
    when uploading one dataset at a time.
    """
    n = normalize_staging(work_dir)
    local_kaggle = staging_root(work_dir)
    if info_csv is not None:
        info_path = Path(info_csv)
        if info_path.is_file():
            local_kaggle.mkdir(parents=True, exist_ok=True)
            shutil.copy2(info_path, local_kaggle / "kaggle_datasets_info.csv")

    if n == 0:
        raise FileNotFoundError(
            f"No staged datasets found under {local_kaggle}. "
            f"Expected {local_kaggle}/<slug>/train.parquet "
            f"(or legacy {Path(work_dir)}/<slug>/input/train.parquet)."
        )
    print(f"Uploading {n} dataset(s) from {local_kaggle} ...")
    return upload_folder_to_hf(
        local_kaggle,
        repo_id=hf_repo,
        path_in_repo=HF_KAGGLE_PREFIX,
        revision=hf_revision,
    )


def main() -> None:
    args = parse_args()
    staging = staging_root(args.work_dir)

    if args.rebuild_info_csv and not args.upload_only:
        n = rebuild_info_csv(
            args.raw_dir,
            work_dir=args.work_dir,
            info_csv=args.info_csv,
        )
        print(f"Rebuilt {n} row(s) -> {args.info_csv}")
        return

    if args.upload_only:
        if args.no_upload:
            raise SystemExit("Use either --upload_only or --no_upload, not both.")
        if Path(args.raw_dir).is_dir():
            n = rebuild_info_csv(
                args.raw_dir,
                work_dir=args.work_dir,
                info_csv=args.info_csv,
            )
            print(f"Rebuilt {n} row(s) -> {args.info_csv}")
        upload_kaggle_staging(
            args.work_dir,
            hf_repo=args.hf_repo,
            hf_revision=args.hf_revision,
            info_csv=args.info_csv,
        )
        print(
            f"Done. Data: https://huggingface.co/datasets/{args.hf_repo}/tree/"
            f"{args.hf_revision}/{HF_KAGGLE_PREFIX}"
        )
        return

    if args.dataset:
        datasets = [resolve_dataset_folder(args.dataset)]
    else:
        print(f"Listing datasets under gs://{args.bucket}/{args.gcs_prefix}/ ...")
        datasets = list_kaggle_datasets(args.bucket, args.gcs_prefix)
        end = None if args.max_datasets is None else args.offset + args.max_datasets
        datasets = datasets[args.offset:end]
    print(f"Found {len(datasets)} dataset(s) to process.")

    repo_files = None
    if args.skip_uploaded and not args.no_upload:
        print(f"Listing files on hf://{args.hf_repo} ({args.hf_revision}) ...")
        repo_files = list_repo_files_cached(
            repo_id=args.hf_repo, revision=args.hf_revision
        )

    rows: list[dict] = []
    failed: list[str] = []
    for dataset_folder in tqdm(datasets, desc="kaggle->hf"):
        slug = slugify(dataset_folder)
        if (
            args.skip_uploaded
            and not args.no_upload
            and kaggle_task_exists_on_hf(
                slug, repo_files, repo_id=args.hf_repo, revision=args.hf_revision
            )
        ):
            tqdm.write(f"skip (already on HF): {slug}")
            continue
        try:
            rows.append(process_dataset(dataset_folder, args))
        except Exception as exc:  # noqa: BLE001
            if not args.skip_errors:
                raise
            tqdm.write(f"FAILED {dataset_folder}: {type(exc).__name__}: {exc}")
            failed.append(dataset_folder)

    # Rebuild the full index from raw data_info.json when available.
    info_csv = Path(args.info_csv)
    if Path(args.raw_dir).is_dir():
        n = rebuild_info_csv(
            args.raw_dir,
            work_dir=args.work_dir,
            info_csv=info_csv,
        )
    else:
        if info_csv.is_file():
            existing = pd.read_csv(info_csv).to_dict("records")
            by_name = {r["name"]: r for r in existing}
            for r in rows:
                by_name[r["name"]] = r
            rows = list(by_name.values())
        write_info_csv(rows, info_csv, staging=staging)
        n = len(rows)
    print(f"Wrote {n} row(s) -> {info_csv}")

    if not args.no_upload:
        print(
            f"Uploading all staged datasets under {staging} in a single HF commit ..."
        )
        upload_kaggle_staging(
            args.work_dir,
            hf_repo=args.hf_repo,
            hf_revision=args.hf_revision,
            info_csv=args.info_csv,
        )
        print(
            f"Done. Data: https://huggingface.co/datasets/{args.hf_repo}/tree/"
            f"{args.hf_revision}/{HF_KAGGLE_PREFIX}"
        )

    if failed:
        print(f"Failed datasets ({len(failed)}): {', '.join(failed)}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
