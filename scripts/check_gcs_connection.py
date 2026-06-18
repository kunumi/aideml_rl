#!/usr/bin/env python3
"""
Smoke-test GCS access (local or Polyaxon).

Checks:
  1. Credentials env / file
  2. Storage client init
  3. Bucket exists and is readable
  4. List objects under aide-runs prefix
  5. Round-trip upload + download of a tiny probe file

Exit 0 on success, 1 on failure.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

# Same defaults as data/data_utils.py
GCS_PROJECT = "numi-platform"
GCS_BUCKET = "benchmark-public-data"
GCS_AIDE_RUNS_PREFIX = "aide-runs"


def _ok(msg: str) -> None:
    print(f"[OK]   {msg}")


def _fail(msg: str) -> None:
    print(f"[FAIL] {msg}")


def _info(msg: str) -> None:
    print(f"[INFO] {msg}")


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}")


# Polyaxon default when gcs connection secret uses mountPath: /plx-context/.gc
_POLYAXON_GCS_PATHS = (
    "/plx-context/.gc/gc-secret.json",
    "/etc/gcs/gc-secret.json",
)


def _write_keyfile_from_env(env_name: str) -> str | None:
    raw = os.getenv(env_name)
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        _warn(f"{env_name} is set but is not valid JSON")
        return None
    path = Path(tempfile.gettempdir()) / f"aide-gcs-{env_name.lower()}.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(path)
    _ok(f"Wrote credentials from {env_name} -> {path}")
    return str(path)


def _scan_plx_context() -> None:
    root = Path("/plx-context")
    if not root.is_dir():
        _info("/plx-context not present (not running on Polyaxon or connection not mounted)")
        return
    _info("Contents of /plx-context:")
    for path in sorted(root.rglob("*")):
        if path.is_file():
            print(f"       {path} ({path.stat().st_size} bytes)")


def resolve_credentials() -> str | None:
    """
    Pick a working credentials source for google-cloud-storage.

    Order: existing valid file env -> Polyaxon paths -> keyfile dict envs -> ADC.
    """
    gac = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if gac:
        p = Path(gac)
        if p.is_file():
            _ok(f"GOOGLE_APPLICATION_CREDENTIALS -> {p}")
            return str(p)
        _warn(
            f"GOOGLE_APPLICATION_CREDENTIALS points to missing file: {gac} "
            "(unsetting; will try other sources)"
        )
        os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS", None)

    gc_key_path = os.getenv("GC_KEY_PATH")
    if gc_key_path and Path(gc_key_path).is_file():
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gc_key_path
        _ok(f"GC_KEY_PATH -> {gc_key_path}")
        return gc_key_path

    for candidate in _POLYAXON_GCS_PATHS:
        p = Path(candidate)
        if p.is_file():
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(p)
            _ok(f"Found Polyaxon GCS key at {p}")
            return str(p)

    for env_name in ("GC_KEYFILE_DICT", "GOOGLE_KEYFILE_DICT"):
        written = _write_keyfile_from_env(env_name)
        if written:
            return written

    _scan_plx_context()
    _info("No key file found; will try Application Default Credentials (GKE workload identity / gcloud)")
    return None


def check_client(project: str):
    from google.cloud import storage

    client = storage.Client(project=project)
    _ok(f"storage.Client(project={project!r})")
    return client


def check_bucket(client, bucket_name: str) -> bool:
    bucket = client.bucket(bucket_name)
    try:
        exists = bucket.exists()
    except Exception as exc:
        err = str(exc)
        _fail(f"Bucket access check failed for gs://{bucket_name}: {exc}")
        if "storage.buckets.get" in err or "403" in err:
            print(
                "\nIAM hint: grant this service account on the bucket, e.g.\n"
                "  roles/storage.objectAdmin  (read/write objects)\n"
                "or at minimum storage.objects.create + storage.objects.list + storage.objects.get\n"
                "  gsutil iam ch serviceAccount:<sa>@<project>.iam.gserviceaccount.com:objectAdmin \\\n"
                f"    gs://{bucket_name}"
            )
        return False
    if not exists:
        _fail(f"Bucket not found: gs://{bucket_name}")
        return False
    _ok(f"Bucket exists: gs://{bucket_name}")
    return True


def check_list_prefix(client, bucket_name: str, prefix: str, limit: int) -> bool:
    blobs = list(client.list_blobs(bucket_name, prefix=prefix, max_results=limit))
    _ok(f"Listed {len(blobs)} object(s) under gs://{bucket_name}/{prefix} (max {limit})")
    for blob in blobs[:5]:
        print(f"       - {blob.name}")
    if len(blobs) > 5:
        print(f"       ... and {len(blobs) - 5} more")
    return True


def check_round_trip(
    client,
    bucket_name: str,
    prefix: str,
    *,
    keep_probe: bool,
) -> bool:
    run_id = uuid.uuid4().hex[:8]
    blob_name = f"{prefix.rstrip('/')}/_connectivity_probe/{run_id}.json"
    payload = {
        "probe": "aide-gcs-check",
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    body = json.dumps(payload, indent=2)

    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    with tempfile.TemporaryDirectory() as tmp:
        local_up = Path(tmp) / "probe.json"
        local_down = Path(tmp) / "probe_downloaded.json"
        local_up.write_text(body, encoding="utf-8")

        blob.upload_from_filename(str(local_up), content_type="application/json")
        _ok(f"Uploaded probe -> gs://{bucket_name}/{blob_name}")

        blob.download_to_filename(str(local_down))
        downloaded = local_down.read_text(encoding="utf-8")
        if downloaded != body:
            _fail("Downloaded probe content does not match upload")
            return False
        _ok("Downloaded probe matches upload")

    if keep_probe:
        _info(f"Probe kept at gs://{bucket_name}/{blob_name}")
    else:
        blob.delete()
        _ok(f"Deleted probe gs://{bucket_name}/{blob_name}")

    return True


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Smoke-test GCS connectivity for AIDE/Polyaxon.")
    p.add_argument("--project", default=GCS_PROJECT)
    p.add_argument("--bucket", default=GCS_BUCKET)
    p.add_argument("--prefix", default=GCS_AIDE_RUNS_PREFIX)
    p.add_argument("--list-limit", type=int, default=10)
    p.add_argument(
        "--keep-probe",
        action="store_true",
        help="Leave the uploaded probe object in GCS (default: delete after check).",
    )
    p.add_argument(
        "--skip-write",
        action="store_true",
        help="Only check credentials, client, bucket, and list (no upload).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    _info(f"Polyaxon run UUID: {os.getenv('POLYAXON_RUN_UUID', '(not set)')}")
    _info(f"Project={args.project}  bucket={args.bucket}  prefix={args.prefix}")

    resolve_credentials()

    try:
        client = check_client(args.project)
    except Exception as exc:
        _fail(f"Could not create storage client: {exc}")
        print(
            "\nPolyaxon troubleshooting:\n"
            "  1. Ensure the project has a GCS connection named 'gcs-artifacts'.\n"
            "  2. The connection secret must mount a key file, e.g.:\n"
            "       secret.mountPath: /plx-context/.gc\n"
            "       kubectl create secret generic gcs-secret \\\n"
            "         --from-file=gc-secret.json=path/to/sa-key.json -n <namespace>\n"
            "  3. Do NOT set GOOGLE_APPLICATION_CREDENTIALS in the YAML unless the file\n"
            "     is guaranteed to exist; use connections: [gcs-artifacts] only.\n"
            "  4. Ask a cluster admin to verify the connection in Polyaxon settings."
        )
        return 1

    ok = True

    ok &= check_bucket(client, args.bucket)
    if not ok:
        return 1

    try:
        check_list_prefix(client, args.bucket, args.prefix, args.list_limit)
    except Exception as exc:
        _fail(f"List under prefix failed: {exc}")
        ok = False

    if not args.skip_write and ok:
        try:
            ok &= check_round_trip(
                client,
                args.bucket,
                args.prefix,
                keep_probe=args.keep_probe,
            )
        except Exception as exc:
            _fail(f"Upload/download probe failed: {exc}")
            ok = False

    if ok:
        print("\nGCS connection check passed.")
        return 0

    print("\nGCS connection check failed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
