"""
S3 utilities for the MLOps Fine-Tuning Hub.

Reads config from environment variables:
    S3_BUCKET             — required: S3 bucket name
    AWS_REGION            — optional: AWS region (default: us-east-1)
    AWS_ACCESS_KEY_ID     — optional: explicit creds (prefer IAM Role on EC2)
    AWS_SECRET_ACCESS_KEY — optional: explicit creds (prefer IAM Role on EC2)

All functions are safe no-ops when S3_BUCKET is not set.
"""

from __future__ import annotations

import os
from pathlib import Path


def is_configured() -> bool:
    """Return True if S3_BUCKET env var is set."""
    return bool(os.environ.get("S3_BUCKET", "").strip())


def get_bucket() -> str:
    return os.environ.get("S3_BUCKET", "").strip()


def s3_uri(key: str) -> str:
    return f"s3://{get_bucket()}/{key}"


def _client():
    import boto3
    return boto3.client(
        "s3",
        region_name=os.environ.get("AWS_REGION", "us-east-1"),
    )


def upload_file(local_path: str | Path, s3_key: str) -> str:
    """
    Upload a single file to S3.
    Returns the s3:// URI on success, empty string if S3 not configured or on error.
    """
    if not is_configured():
        return ""
    try:
        _client().upload_file(str(local_path), get_bucket(), s3_key)
        return s3_uri(s3_key)
    except Exception as exc:
        print(f"[S3] upload_file failed for {local_path}: {exc}")
        return ""


def upload_folder(local_dir: str | Path, s3_prefix: str) -> list[str]:
    """
    Upload every file in local_dir (recursively) to S3 under s3_prefix.
    Returns list of s3:// URIs that were uploaded.
    """
    if not is_configured():
        return []
    local_dir = Path(local_dir)
    uploaded = []
    for file_path in sorted(local_dir.rglob("*")):
        if not file_path.is_file():
            continue
        rel = file_path.relative_to(local_dir)
        key = f"{s3_prefix.rstrip('/')}/{str(rel).replace(os.sep, '/')}"
        uri = upload_file(file_path, key)
        if uri:
            uploaded.append(uri)
    return uploaded


def download_file(s3_key: str, local_path: str | Path) -> bool:
    """Download a single file from S3. Returns True on success."""
    if not is_configured():
        return False
    try:
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        _client().download_file(get_bucket(), s3_key, str(local_path))
        return True
    except Exception as exc:
        print(f"[S3] download_file failed for {s3_key}: {exc}")
        return False


def download_folder(s3_prefix: str, local_dir: str | Path) -> bool:
    """
    Download all objects under s3_prefix into local_dir, preserving folder structure.
    Returns True if at least one file was downloaded.
    """
    if not is_configured():
        return False
    try:
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        client = _client()
        paginator = client.get_paginator("list_objects_v2")
        downloaded = 0
        prefix = s3_prefix.rstrip("/") + "/"
        for page in paginator.paginate(Bucket=get_bucket(), Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                rel = key[len(prefix):]
                if not rel:
                    continue
                dest = local_dir / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                client.download_file(get_bucket(), key, str(dest))
                downloaded += 1
        return downloaded > 0
    except Exception as exc:
        print(f"[S3] download_folder failed for {s3_prefix}: {exc}")
        return False


def list_adapter_names() -> list[str]:
    """
    List experiment names that have a final_adapter saved in S3.
    Returns experiment names (e.g. ['exp_20260410_1230', ...]).
    """
    if not is_configured():
        return []
    try:
        client = _client()
        paginator = client.get_paginator("list_objects_v2")
        names = []
        for page in paginator.paginate(
            Bucket=get_bucket(), Prefix="outputs/", Delimiter="/"
        ):
            for prefix_obj in page.get("CommonPrefixes", []):
                exp_prefix = prefix_obj["Prefix"]          # e.g. "outputs/exp_xxx/"
                adapter_key = f"{exp_prefix}final_adapter/adapter_config.json"
                try:
                    client.head_object(Bucket=get_bucket(), Key=adapter_key)
                    exp_name = exp_prefix.rstrip("/").split("/")[-1]
                    names.append(exp_name)
                except Exception:
                    pass
        return names
    except Exception as exc:
        print(f"[S3] list_adapter_names failed: {exc}")
        return []


def restore_adapter(exp_name: str, local_outputs_dir: str | Path) -> str:
    """
    Download a trained adapter from S3 to local disk if not already present.
    Returns the local adapter path on success, empty string on failure.
    """
    if not is_configured():
        return ""
    local_adapter = Path(local_outputs_dir) / exp_name / "final_adapter"
    if local_adapter.exists():
        return str(local_adapter)
    s3_prefix = f"outputs/{exp_name}/final_adapter"
    ok = download_folder(s3_prefix, local_adapter)
    return str(local_adapter) if ok else ""


def backup_db(local_db_path: str | Path) -> str:
    """
    Back up the MLflow SQLite DB to S3 at backups/mlflow.db.
    Returns s3:// URI on success, empty string on failure.
    """
    if not is_configured():
        return ""
    if not Path(local_db_path).exists():
        return ""
    return upload_file(local_db_path, "backups/mlflow.db")
