"""Метрики качества классификаторов тегов.

Все функции принимают:
    y      : матрица 0/1 формы (n_tracks, n_tags), разметка;
    scores : матрица оценок той же формы (чем больше, тем вероятнее тег).

Ранжирующие метрики (precision@k, PR-AUC, ROC-AUC, lift, hit@k) не зависят
от порогов. Пороги нужны только для F1 и итоговых тегов в ответе сервиса.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)


def evaluable_tags(y: np.ndarray) -> np.ndarray:
    """Теги, у которых в выборке есть и положительные, и отрицательные примеры."""
    positives = y.sum(axis=0)
    return np.flatnonzero((positives > 0) & (positives < len(y)))


def per_tag_ap_auc(y: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """AP и ROC-AUC по каждому тегу. NaN, если тег нельзя оценить."""
    ap = np.full(y.shape[1], np.nan)
    auc = np.full(y.shape[1], np.nan)
    for j in evaluable_tags(y):
        ap[j] = average_precision_score(y[:, j], scores[:, j])
        auc[j] = roc_auc_score(y[:, j], scores[:, j])
    return ap, auc


def per_tag_precision_at_k(y: np.ndarray, scores: np.ndarray, k: int) -> np.ndarray:
    """Precision@k по каждому тегу: доля размеченных этим тегом среди k треков
    с наибольшей оценкой. NaN, если положительных примеров меньше k
    (иначе метрика ограничена сверху числом примеров, а не качеством модели)."""
    result = np.full(y.shape[1], np.nan)
    k = min(k, len(y))
    if k == 0:
        return result
    top = np.argsort(-scores, axis=0, kind="stable")[:k]
    for j in np.flatnonzero(y.sum(axis=0) >= k):
        result[j] = y[top[:, j], j].mean()
    return result


def hit_at_k(y: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Доля треков, у которых среди k тегов с наибольшей оценкой есть хотя бы
    один верный. Считается по трекам, у которых есть хотя бы один тег."""
    has_tags = y.sum(axis=1) > 0
    if not has_tags.any():
        return float("nan")
    k = min(k, y.shape[1])
    top = np.argsort(-scores[has_tags], axis=1, kind="stable")[:, :k]
    hits = np.take_along_axis(y[has_tags], top, axis=1).max(axis=1)
    return float(hits.mean())


def tune_thresholds(y: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Порог на тег, максимизирующий F1 (0.5, если положительных примеров нет)."""
    thresholds = np.full(y.shape[1], 0.5, dtype=np.float32)
    for j in range(y.shape[1]):
        if y[:, j].sum() == 0:
            continue
        precision, recall, cut = precision_recall_curve(y[:, j], scores[:, j])
        f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
        # Последняя точка PR-кривой не имеет порога.
        thresholds[j] = cut[int(np.argmax(f1[:-1]))]
    return thresholds


def _nanmean(values: np.ndarray) -> float:
    values = values[~np.isnan(values)]
    return float(values.mean()) if values.size else float("nan")


def category_metrics(
    prefix: str,
    y: np.ndarray,
    scores: np.ndarray,
    thresholds: np.ndarray | None = None,
    k: int = 10,
    hit_k: int = 3,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Все метрики одной категории.

    prefix, например "val_mood": имена метрик будут val_mood_precision_at_10 и т.д.
    Возвращает словарь метрик для MLflow и метрики по каждому тегу.
    """
    ap, auc = per_tag_ap_auc(y, scores)
    p_at_k = per_tag_precision_at_k(y, scores, k)
    prevalence = y.mean(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        lift = np.where(prevalence > 0, ap / prevalence, np.nan)

    metrics = {
        f"{prefix}_precision_at_{k}": _nanmean(p_at_k),
        f"{prefix}_pr_auc_macro": _nanmean(ap),
        f"{prefix}_roc_auc_macro": _nanmean(auc),
        f"{prefix}_lift": _nanmean(lift),
        f"{prefix}_hit_at_{hit_k}": hit_at_k(y, scores, hit_k),
        f"{prefix}_n_tracks": float(len(y)),
        f"{prefix}_n_tags_evaluated": float(np.sum(~np.isnan(ap))),
        f"{prefix}_n_tags_precision_at_{k}": float(np.sum(~np.isnan(p_at_k))),
    }
    per_tag = {
        "ap": ap,
        "roc_auc": auc,
        f"precision_at_{k}": p_at_k,
        "lift": lift,
        "positives": y.sum(axis=0),
    }
    if thresholds is not None:
        pred = (scores >= thresholds).astype(np.int8)
        metrics[f"{prefix}_f1_micro"] = float(
            f1_score(y, pred, average="micro", zero_division=0)
        )
        metrics[f"{prefix}_f1_macro"] = float(
            f1_score(y, pred, average="macro", zero_division=0)
        )
        per_tag["f1"] = f1_score(y, pred, average=None, zero_division=0)
        per_tag["threshold"] = thresholds
    return metrics, per_tag


def macro_pr_auc(y: np.ndarray, scores: np.ndarray) -> float:
    return _nanmean(per_tag_ap_auc(y, scores)[0])


def macro_precision_at_k(y: np.ndarray, scores: np.ndarray, k: int = 10) -> float:
    return _nanmean(per_tag_precision_at_k(y, scores, k))


def resampling_ci(
    y: np.ndarray,
    scores: np.ndarray,
    metric,
    n_resamples: int = 200,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """Доверительный интервал метрики методом half-sampling.

    n_resamples раз берётся случайная половина треков (без повторений) и
    считается метрика. Разброс метрики на половинах примерно равен разбросу
    метрики на всей выборке (для средних это точное равенство с поправкой на
    конечную выборку), поэтому интервал = оценка на всей выборке ± z * std.

    Классический bootstrap (выбор с повторениями) здесь не подходит: дубликаты
    положительных треков стоят рядом в ранжировании и завышают precision@k и AP.

    Интервал показывает, насколько метрика зависит от состава выборки: если
    интервалы двух моделей сильно пересекаются, разница может быть случайной.
    """
    from statistics import NormalDist

    point = metric(y, scores)
    rng = np.random.default_rng(seed)
    half = len(y) // 2
    values = []
    for _ in range(n_resamples):
        idx = rng.choice(len(y), size=half, replace=False)
        value = metric(y[idx], scores[idx])
        if np.isfinite(value):
            values.append(value)
    if len(values) < 2 or not np.isfinite(point):
        return float("nan"), float("nan")
    z = NormalDist().inv_cdf(1 - alpha / 2)
    spread = z * float(np.std(values, ddof=1))
    # Метрики качества лежат в [0, 1].
    return float(max(point - spread, 0.0)), float(min(point + spread, 1.0))
