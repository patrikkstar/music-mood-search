"""Общие функции для EDA, обучения и работы с Model Registry.

Скрипты запускаются из корня репозитория:
    uv run --group training python training/<script>.py ...

Настройки берутся из переменных окружения (см. .env.example).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
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
DATASET_NAME = "mtg-jamendo-moodtheme-effnet"

# Как получен эмбеддинг (см. training/data/extract_embeddings.py). Сохраняется
# в метаданных модели: сервис корректно работает только с такими векторами.
EMBEDDING_CONTRACT = {
    "model": "discogs-effnet-bs64-1",
    "output": "PartitionedCall:1",
    "sample_rate": 16000,
    "channels": "mono",
    "pooling": "mean over patches of the full track",
    "dim": EMBEDDING_DIM,
}


@dataclass(frozen=True)
class Config:
    tracking_uri: str
    s3_endpoint_url: str
    dataset_uri: str | None
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
            # Версия данных выбирается явно: без DATASET_URI (или --data)
            # скрипты остановятся, а не возьмут молча устаревший датасет.
            dataset_uri=os.getenv("DATASET_URI") or None,
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


# Целевые категории модели. mood и theme строятся из колонки "mood"
# (в MTG-Jamendo настроения и темы лежат вместе) по группам из tag_groups.json.
TARGETS = ("genre", "instrument", "mood", "theme")
DEFAULT_GROUPS_FILE = Path(__file__).with_name("tag_groups.json")


def load_tag_groups(path: Path) -> dict[str, dict[str, list[str]]]:
    """{"mood": {группа: [теги]}, "theme": {группа: [теги]}} с проверкой повторов."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    groups = {kind: dict(data.get(kind, {})) for kind in ("mood", "theme")}
    seen: dict[str, str] = {}
    for kind_groups in groups.values():
        for group, tags in kind_groups.items():
            for tag in tags:
                if tag in seen:
                    raise ValueError(f"Тег {tag} есть в группах {seen[tag]} и {group}")
                seen[tag] = group
    return groups


def file_digest(path: Path, length: int = 12) -> str:
    return file_sha256(path)[:length]


def targets_for(groups: dict | None) -> tuple[str, ...]:
    """С группами модель предсказывает 4 категории, без них 3 (теги как есть)."""
    return TARGETS if groups else CATEGORIES


def target_labels(
    df: pd.DataFrame, target: str, groups: dict | None = None
) -> pd.Series:
    """Метки каждого трека для целевой категории (список строк).

    genre, instrument: теги как есть. mood, theme при заданных группах: названия
    групп, в которые попадают теги трека из колонки "mood"; без групп mood
    это исходные теги mood/theme.
    """
    if target in ("genre", "instrument") or (target == "mood" and not groups):
        return df[target].map(split_tags)
    if not groups:
        raise ValueError(f"Категория {target} требует файла групп")
    mapping = {tag: g for g, tags in groups[target].items() for tag in tags}
    return df["mood"].map(
        lambda v: sorted({mapping[t] for t in split_tags(v) if t in mapping})
    )


def label_counts(labels: pd.Series) -> pd.Series:
    return labels.explode().dropna().value_counts()


def label_matrix(labels: pd.Series, names: list[str]) -> np.ndarray:
    index = {name: i for i, name in enumerate(names)}
    y = np.zeros((len(labels), len(names)), dtype=np.int8)
    for row, values in enumerate(labels):
        for value in values:
            j = index.get(value)
            if j is not None:
                y[row, j] = 1
    return y


def features(df: pd.DataFrame) -> np.ndarray:
    return df[EMBEDDING_COLUMNS].to_numpy(dtype=np.float32)


DATA_VERSION_RE = re.compile(r"v\d+")


def data_version(uri: str, file_sha: str | None = None) -> str:
    """Версия датасета по соглашению о путях: .../vN/embeddings.parquet -> vN.

    Если путь не следует соглашению (например, локальный файл
    training/data/out/embeddings.parquet), версия = "sha-" и первые символы
    sha256 файла. Настоящий идентификатор данных всё равно sha256: версия
    нужна только как понятная подпись в MLflow и Model Registry.
    """
    folder = Path(urlparse(uri).path if "://" in uri else uri).parent.name
    if DATA_VERSION_RE.fullmatch(folder):
        return folder
    return f"sha-{file_sha[:8]}" if file_sha else "unknown"


def require_dataset_uri(uri: str | None) -> str:
    """Датасет должен быть указан явно: --data или DATASET_URI в .env."""
    if not uri:
        raise SystemExit(
            "Не указан датасет: передайте --data "
            "s3://datasets/mtg-jamendo-moodtheme/vN/embeddings.parquet "
            "или задайте DATASET_URI в .env"
        )
    return uri
