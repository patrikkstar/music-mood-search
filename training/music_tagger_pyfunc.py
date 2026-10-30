"""MLflow pyfunc-обёртка: независимые классификаторы категорий тегов.

Этот файл и tagger_heads.py сохраняются вместе с моделью (code_paths),
поэтому приложению не нужен код из training/.

Вход:  массив float32 формы (n, 1280), эмбеддинги EffNet-Discogs.
Выход: список словарей вида
    {"genre": [{"tag": "rock", "score": 0.91}, ...], "instrument": [...],
     "mood": [...], "theme": [...]}
Категория, для которой в модели нет классификатора, возвращается пустым списком.
В каждой категории возвращаются теги выше порога (не больше max_tags).
Если ни один тег категории не прошёл порог, возвращается самый вероятный,
чтобы у каждого трека были метки всех категорий модели.
"""

from __future__ import annotations

import json
from typing import Any

import joblib
import mlflow
import numpy as np

CATEGORIES = ("genre", "instrument", "mood", "theme")


def decode_category(
    scores: np.ndarray,
    tags: list[str],
    thresholds: np.ndarray,
    max_tags: int = 3,
) -> list[list[dict[str, Any]]]:
    """Оценки одной категории (n x n_tags) -> списки тегов для каждого трека."""
    result = []
    for row in scores:
        order = np.argsort(-row)
        chosen = [j for j in order if row[j] >= thresholds[j]][:max_tags]
        if not chosen and len(order):
            chosen = [order[0]]
        result.append(
            [{"tag": tags[j], "score": round(float(row[j]), 4)} for j in chosen]
        )
    return result


def predict_tags(
    heads: dict[str, dict[str, Any]], x: np.ndarray, max_tags: int = 3
) -> list[dict[str, list[dict[str, Any]]]]:
    """Прогоняет эмбеддинги через классификатор каждой категории."""
    per_category = {}
    for category in CATEGORIES:
        head = heads.get(category)
        if head is None:
            per_category[category] = [[] for _ in range(len(x))]
            continue
        scores = head["model"].predict_proba(x)
        per_category[category] = decode_category(
            scores, head["tags"], np.asarray(head["thresholds"]), max_tags
        )
    return [
        {category: per_category[category][i] for category in CATEGORIES}
        for i in range(len(x))
    ]


class MusicTaggerModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context) -> None:
        self.heads: dict[str, dict[str, Any]] = joblib.load(context.artifacts["heads"])
        with open(context.artifacts["meta"], encoding="utf-8") as f:
            meta = json.load(f)
        self.max_tags = int(meta.get("max_tags", 3))
        if any(h["family"].startswith("mlp") for h in self.heads.values()):
            import torch

            # Модель маленькая: один поток быстрее и не мешает соседним запросам.
            torch.set_num_threads(1)

    def predict(self, context, model_input, params=None):
        x = np.asarray(model_input, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
        return predict_tags(self.heads, x, self.max_tags)
