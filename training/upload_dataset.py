"""Загрузка подготовленного датасета эмбеддингов в объектное хранилище (MinIO/S3).

Файл кладётся по версионированному пути и не перезаписывается:
новые данные = новая версия (v2, v3, ...). Именно этот URI потом
указывается как source датасета в MLflow.

Пример:
    uv run --group training python training/upload_dataset.py \
        --file training/data/out/embeddings.parquet \
        --manifest training/data/out/manifest.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from botocore.exceptions import ClientError
from common import Config, file_sha256, parse_s3_uri, s3_client


def ensure_bucket(client, bucket: str) -> None:
    try:
        client.head_bucket(Bucket=bucket)
    except ClientError:
        client.create_bucket(Bucket=bucket)
        print(f"Создан бакет {bucket}")


def key_exists(client, bucket: str, key: str) -> bool:
    try:
        client.head_object(Bucket=bucket, Key=key)
    except ClientError:
        return False
    return True


def main() -> None:
    cfg = Config.from_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--uri", default=cfg.dataset_uri)
    parser.add_argument("--force", action="store_true", help="перезаписать версию")
    args = parser.parse_args()

    bucket, key = parse_s3_uri(args.uri)
    client = s3_client(cfg)
    ensure_bucket(client, bucket)

    if key_exists(client, bucket, key) and not args.force:
        raise SystemExit(
            f"{args.uri} уже существует. Для новых данных используйте новую версию "
            "(например .../v2/embeddings.parquet) или --force."
        )

    sha = file_sha256(args.file)
    client.upload_file(
        str(args.file), bucket, key, ExtraArgs={"Metadata": {"sha256": sha}}
    )
    print(f"Загружено: {args.uri} (sha256={sha})")

    if args.manifest:
        manifest_key = str(Path(key).with_name("manifest.json"))
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        manifest["uploaded_file_sha256"] = sha
        client.put_object(
            Bucket=bucket,
            Key=manifest_key,
            Body=json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"),
            ContentType="application/json",
        )
        print(f"Загружено: s3://{bucket}/{manifest_key}")


if __name__ == "__main__":
    main()
