"""GCS helpers for AIDE heuristic run artifacts (project: numi-platform)."""

from __future__ import annotations

import os
from pathlib import Path

from google.cloud import storage

# Defaults for AIDE heuristic log uploads
GCS_PROJECT = "numi-platform"
GCS_BUCKET = "benchmark-public-data"
GCS_AIDE_RUNS_PREFIX = "aide-runs"

def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"

    # The ID of your GCS object
    # source_blob_name = "storage-object-name"

    # The path to which the file should be downloaded
    # destination_file_name = "local/path/to/file"

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)

    # Construct a client side representation of a blob.
    # Note `Bucket.blob` differs from `Bucket.get_blob` as it doesn't retrieve
    # any content from Google Cloud Storage. As we don't need additional data,
    # using `Bucket.blob` is preferred here.
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

    print(
        "Downloaded storage object {} from bucket {} to local file {}.".format(
            source_blob_name, bucket_name, destination_file_name
        )
    )

def list_blobs(bucket_name, prefix=""):
    """Lists all the blobs in the bucket under an optional prefix."""
    # bucket_name = "your-bucket-name"

    storage_client = storage.Client()

    # Note: Client.list_blobs requires at least package version 1.17.0.
    blobs = storage_client.list_blobs(bucket_name, prefix=prefix or None)

    # Note: The call returns a response only when the iterator is consumed.
    available_files = [blob.name for blob in blobs]
    return available_files


def download_blob_prefix(
    bucket_name: str,
    prefix: str,
    destination_dir: str | Path,
    *,
    project: str | None = None,
    skip_existing: bool = True,
) -> list[Path]:
    """
    Download every object under ``prefix`` into ``destination_dir``.

    The blob path relative to ``prefix`` is preserved on disk. Returns the list
    of local files written (or already present when ``skip_existing``).
    """
    destination_dir = Path(destination_dir)
    storage_client = _storage_client(project)

    prefix = prefix.strip("/")
    blobs = list(storage_client.list_blobs(bucket_name, prefix=f"{prefix}/" if prefix else None))

    written: list[Path] = []
    for blob in blobs:
        # Skip "directory placeholder" objects.
        if blob.name.endswith("/"):
            continue
        relative = blob.name[len(prefix) + 1 :] if prefix else blob.name
        local_path = destination_dir / relative
        local_path.parent.mkdir(parents=True, exist_ok=True)
        if skip_existing and local_path.is_file() and local_path.stat().st_size > 0:
            written.append(local_path)
            continue
        blob.download_to_filename(str(local_path))
        written.append(local_path)
    return written

def _storage_client(project: str | None = None) -> storage.Client:
    if project:
        return storage.Client(project=project)
    return storage.Client(project=GCS_PROJECT)


def upload_folder_to_gcs(
    bucket_name: str,
    source_folder: str | Path,
    destination_blob_prefix: str = "",
    *,
    project: str | None = None,
) -> list[str]:
    """
    Upload a local folder to GCS recursively.

    Returns list of gs:// URIs uploaded.
    """
    source_folder = Path(source_folder)
    if not source_folder.is_dir():
        raise NotADirectoryError(source_folder)

    prefix = destination_blob_prefix.strip("/")
    storage_client = _storage_client(project)
    bucket = storage_client.bucket(bucket_name)
    uploaded: list[str] = []

    for root, _, files in os.walk(source_folder):
        for file in files:
            local_path = Path(root) / file
            relative_path = local_path.relative_to(source_folder).as_posix()
            blob_path = f"{prefix}/{relative_path}" if prefix else relative_path

            blob = bucket.blob(blob_path)
            blob.upload_from_filename(str(local_path))
            uri = f"gs://{bucket_name}/{blob_path}"
            uploaded.append(uri)
            print(f"Uploaded {local_path} -> {uri}")

    return uploaded


def upload_aide_log_dir(
    local_log_dir: str | Path,
    *,
    task_name: str,
    seed: int,
    bucket_name: str = GCS_BUCKET,
    gcs_prefix: str = GCS_AIDE_RUNS_PREFIX,
    project: str | None = None,
) -> str:
    """
    Upload one AIDE experiment log directory (journal.json, config.yaml, etc.)
    to gs://<bucket>/<gcs_prefix>/<task_name>/seed<seed>/.

    Example local path:
      data/heuristic_runs/logs/2-ctu-legalacts_legalacts-original__seed0/
    """
    local_log_dir = Path(local_log_dir)
    if not local_log_dir.is_dir():
        raise NotADirectoryError(f"Log directory not found: {local_log_dir}")

    dest_prefix = f"{gcs_prefix.strip('/')}/{task_name}/seed{seed}"
    upload_folder_to_gcs(
        bucket_name,
        local_log_dir,
        dest_prefix,
        project=project or GCS_PROJECT,
    )
    return f"gs://{bucket_name}/{dest_prefix}/"


def validate_dataset_dict(data):
    errors = []

    required_keys = {
        "name": str,
        "columns": dict,
        "description": str,
        "task_description": str,
        "task": str,
        "target_column": str
    }

    for key, expected_type in required_keys.items():
        if key not in data:
            errors.append(f"{data['name']}: Missing required key: '{key}'")
        elif not isinstance(data[key], expected_type):
            errors.append(f"{data['name']}: Key '{key}' must be of type {expected_type.__name__}")

    if "task" in data and data["task"] not in {"regression", "multi", "binary"}:
        errors.append(f"{data['name']}: Key 'task' must be either 'regression' or 'multi' or 'binary'")

    if "columns" in data and isinstance(data["columns"], dict):
        for col_name, col_desc in data["columns"].items():
            if not isinstance(col_name, str):
                errors.append(f"{data['name']}: All column names in 'columns' must be strings")
            if not isinstance(col_desc, dict):
                errors.append(f"{data['name']}: Description for column '{col_name}' must be a dict")

    if "target_column" in data and "columns" in data:
        if data["target_column"] not in data["columns"]:
            errors.append(f"{data['name']}: The 'target_column' must be one of the keys in the 'columns' dictionary")

    return (len(errors) == 0, errors)

