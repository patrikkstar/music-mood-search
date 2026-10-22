"""Общие функции для EDA, обучения и работы с Model Registry.

Скрипты запускаются из корня репозитория:
    uv run --group training python training/<script>.py ...

Настройки берутся из переменных окружения (см. .env.example).
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd

CATEGORIES = ("genre", "instrument", "mood")
EMBEDDING_DIM = 1280
EMBEDDING_COLUMNS = [f"e{i}" for i in range(EMBEDDING_DIM)]
META_COLUMNS = [
    "track_id",
    "artist_id",
    "album_id",
    "duration",
    "split",
    "genre",
    "instrument",
    "mood",
]
LABEL_SEP = "---"
DATASET_NAME = "mtg-jamendo-moodtheme-effnet"


@dataclass(frozen=True)
class Config:
    tracking_uri: str
    s3_endpoint_url: str
    dataset_uri: str
    experiment: str
    eda_experiment: str
    registered_model: str
    cache_dir: Path

    @classmethod
    def from_env(cls) -> Config:
        return cls(
            tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
            s3_endpoint_url=os.getenv(
                "MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000"
            ),
            dataset_uri=os.getenv(
                "DATASET_URI",
                "s3://datasets/mtg-jamendo-moodtheme/v1/embeddings.parquet",
            ),
            experiment=os.getenv("MLFLOW_EXPERIMENT", "music-tagger"),
            eda_experiment=os.getenv("MLFLOW_EDA_EXPERIMENT", "music-tagger-eda"),
            registered_model=os.getenv("MODEL_NAME", "music-tagger"),
            cache_dir=Path(os.getenv("DATA_CACHE_DIR", ".cache/datasets")),
        )


# ------------------------------------------------------------------ storage


def s3_client(cfg: Config):
    import boto3

    return boto3.client("s3", endpoint_url=cfg.s3_endpoint_url)


def parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"Not an s3:// URI: {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def resolve_local_path(uri: str, cfg: Config, refresh: bool = False) -> Path:
    """Для s3:// скачивает файл в локальный кэш, для пути возвращает его же."""
    if not uri.startswith("s3://"):
        path = Path(uri)
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    bucket, key = parse_s3_uri(uri)
    local = cfg.cache_dir / bucket / key
    if refresh or not local.exists():
        local.parent.mkdir(parents=True, exist_ok=True)
        s3_client(cfg).download_file(bucket, key, str(local))
    return local


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_dataset(
    uri: str, cfg: Config, refresh: bool = False
) -> tuple[pd.DataFrame, str]:
    """Возвращает датафрейм и sha256 исходного parquet-файла."""
    path = resolve_local_path(uri, cfg, refresh=refresh)
    df = pd.read_parquet(path)
    validate_dataset(df)
    for col in CATEGORIES:
        df[col] = df[col].fillna("").astype(str)
    return df, file_sha256(path)


def validate_dataset(df: pd.DataFrame) -> None:
    missing = [c for c in META_COLUMNS + EMBEDDING_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset has no columns: {missing[:5]}...")
    if df["track_id"].duplicated().any():
        raise ValueError("Duplicate track_id in dataset")


# ------------------------------------------------------------------ dataset tracking


def dataset_digest(df: pd.DataFrame) -> str:
    """Хэш содержимого: изменился хоть один трек или тег, изменится и digest."""
    hashed = pd.util.hash_pandas_object(df, index=False).to_numpy()
    return hashlib.sha256(hashed.tobytes()).hexdigest()[:16]


def mlflow_dataset(df: pd.DataFrame, source: str, name: str):
    import mlflow

    return mlflow.data.from_pandas(
        df,
        source=source,
        name=name,
        digest=dataset_digest(df),
    )


def log_dataset_inputs(df: pd.DataFrame, source: str, splits: Iterable[str]) -> None:
    """Регистрирует каждую часть выборки как отдельный input текущего Run."""
    import mlflow

    contexts = {"train": "training", "validation": "validation", "test": "testing"}
    for split in splits:
        part = df[df["split"] == split]
        ds = mlflow_dataset(part, source=source, name=f"{DATASET_NAME}-{split}")
        mlflow.log_input(ds, context=contexts.get(split, split))


# ------------------------------------------------------------------ labels


def split_tags(value: str) -> list[str]:
    return [t for t in str(value).split("|") if t]


def tag_counts(df: pd.DataFrame, category: str) -> pd.Series:
    exploded = df[category].map(split_tags).explode().dropna()
    return exploded.value_counts()


def select_labels(
    train_df: pd.DataFrame,
    min_count: int,
    max_per_category: int | None = None,
) -> list[str]:
    """Теги, у которых в train не меньше min_count положительных примеров."""
    labels: list[str] = []
    for category in CATEGORIES:
        counts = tag_counts(train_df, category)
        counts = counts[counts >= min_count]
        if max_per_category:
            counts = counts.head(max_per_category)
        labels += [f"{category}{LABEL_SEP}{tag}" for tag in sorted(counts.index)]
    return labels


def label_matrix(df: pd.DataFrame, labels: list[str]) -> np.ndarray:
    index = {label: i for i, label in enumerate(labels)}
    y = np.zeros((len(df), len(labels)), dtype=np.int8)
    for row, rec in enumerate(df[list(CATEGORIES)].itertuples(index=False)):
        for category, value in zip(CATEGORIES, rec, strict=True):
            for tag in split_tags(value):
                j = index.get(f"{category}{LABEL_SEP}{tag}")
                if j is not None:
                    y[row, j] = 1
    return y


def features(df: pd.DataFrame) -> np.ndarray:
    return df[EMBEDDING_COLUMNS].to_numpy(dtype=np.float32)


def label_category(label: str) -> str:
    return label.split(LABEL_SEP, 1)[0]


def label_tag(label: str) -> str:
    return label.split(LABEL_SEP, 1)[1]
