"""Model packages: install from zip (safely) and sync from Amazon S3.

A package (built on EC2 by training/train_system.py) is <task_id>_package.zip containing
<task_id>/package.json and <task_id>/members/*.onnx.
"""
import json
import logging
import os
import shutil
import zipfile

log = logging.getLogger(__name__)


def install_zip(zip_path, models_dir):
    """Extract a package zip into models_dir/<task_id>/ (replacing an older version)."""
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        tops = {n.split("/")[0] for n in names if n.strip("/")}
        if len(tops) != 1 or f"{next(iter(tops))}/package.json" not in names:
            raise ValueError(f"{zip_path}: not a model package (package.json missing)")
        for n in names:  # refuse absolute paths and '..' (zip slip)
            if os.path.isabs(n) or ".." in n.replace("\\", "/").split("/"):
                raise ValueError(f"{zip_path}: unsafe path in archive: {n}")
        task_id = next(iter(tops))
        tmp = os.path.join(models_dir, f".{task_id}.tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        zf.extractall(tmp)
    target = os.path.join(models_dir, task_id)
    shutil.rmtree(target, ignore_errors=True)
    os.replace(os.path.join(tmp, task_id), target)
    shutil.rmtree(tmp, ignore_errors=True)
    return task_id


def sync_from_s3(s3_uri, models_dir):
    """Download every *_package.zip under s3_uri that changed since the last sync (ETag)."""
    import boto3  # credentials come from the instance / task IAM role

    bucket, _, prefix = s3_uri.removeprefix("s3://").partition("/")
    s3 = boto3.client("s3")
    os.makedirs(models_dir, exist_ok=True)
    state_path = os.path.join(models_dir, ".s3_etags.json")
    state = json.load(open(state_path)) if os.path.exists(state_path) else {}
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith("_package.zip") or state.get(key) == obj["ETag"]:
                continue
            local = os.path.join(models_dir, os.path.basename(key))
            log.info("downloading s3://%s/%s", bucket, key)
            s3.download_file(bucket, key, local)
            task_id = install_zip(local, models_dir)
            os.remove(local)
            state[key] = obj["ETag"]
            log.info("installed model package %s", task_id)
    with open(state_path, "w") as f:
        json.dump(state, f)
