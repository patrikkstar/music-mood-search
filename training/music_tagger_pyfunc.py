"""MLflow pyfunc-обёртка классификатора тегов.

Этот файл сохраняется вместе с моделью (code_paths), поэтому приложению
не нужен код из training/: всё необходимое лежит в артефакте модели.

Вход:  массив float32 формы (n, 1280), эмбеддинги EffNet-Discogs.
Выход: список словарей вида
    {"genre": [{"tag": "rock", "score": 0.91}, ...], "instrument": [...], "mood": [...]}
В каждой категории возвращаются теги выше порога (пороги подобраны на validation).
Если ни один тег категории не прошёл порог, возвращается один самый вероятный,
чтобы у каждого трека было хотя бы одно настроение, жанр и инструмент.
"""

from __future__ import annotations

import json
from typing import Any

import joblib
import mlflow
import numpy as np

CATEGORIES = ("genre", "instrument", "mood")
LABEL_SEP = "---"


def decode_predictions(
    proba: np.ndarray,
    labels: list[str],
    thresholds: np.ndarray,
    max_tags: int = 5,
) -> list[dict[str, list[dict[str, Any]]]]:
    categories = np.array([label.split(LABEL_SEP, 1)[0] for label in labels])
    tags = [label.split(LABEL_SEP, 1)[1] for label in labels]
    results = []
    for row in proba:
        item: dict[str, list[dict[str, Any]]] = {}
        for category in CATEGORIES:
            idx = np.flatnonzero(categories == category)
            if idx.size == 0:
                item[category] = []
                continue
            order = idx[np.argsort(-row[idx])]
            chosen = [j for j in order if row[j] >= thresholds[j]][:max_tags]
            if not chosen:
                chosen = [order[0]]
            item[category] = [
                {"tag": tags[j], "score": round(float(row[j]), 4)} for j in chosen
            ]
        results.append(item)
    return results


class MusicTaggerModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context) -> None:
        self.pipeline = joblib.load(context.artifacts["pipeline"])
        with open(context.artifacts["labels"], encoding="utf-8") as f:
            meta = json.load(f)
        self.labels: list[str] = meta["labels"]
        self.thresholds = np.asarray(meta["thresholds"], dtype=np.float32)
        self.max_tags = int(meta.get("max_tags", 5))

    def predict(self, context, model_input, params=None):
        x = np.asarray(model_input, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
        proba = self.pipeline.predict_proba(x)
        return decode_predictions(proba, self.labels, self.thresholds, self.max_tags)
