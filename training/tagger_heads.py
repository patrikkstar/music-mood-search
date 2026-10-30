"""Классификаторы одной категории тегов (mood, genre или instrument).

У всех классов одинаковый интерфейс:
    head.fit(x, y, groups=None) -> head
    head.predict_proba(x) -> np.ndarray формы (n_tracks, n_tags)

x: эмбеддинги EffNet-Discogs (float32, n x 1280), y: матрица 0/1 (n x n_tags),
groups: artist_id трека (нужен MLP, чтобы отложить часть артистов для ранней
остановки, а не часть треков).

Файл сохраняется вместе с моделью в MLflow (code_paths), поэтому импортирует
только numpy, scikit-learn и (лениво, внутри методов) torch.
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np

# Сетка гиперпараметров каждого семейства. Лучшая точка выбирается
# кросс-валидацией внутри train (см. train.py).
FAMILY_GRIDS: dict[str, dict[str, list[Any]]] = {
    "logreg": {"C": [0.01, 0.1, 1.0]},
    "knn": {"k": [10, 20, 50]},
    "mlp_bce": {"hidden": [256, 512], "dropout": [0.2, 0.5]},
    "mlp_asl": {"hidden": [256, 512], "dropout": [0.2, 0.5], "gamma_neg": [2, 4]},
}
FAMILIES = tuple(FAMILY_GRIDS)


def grid_points(family: str, smoke: bool = False) -> list[dict[str, Any]]:
    """Все комбинации гиперпараметров семейства (в smoke-режиме только первая)."""
    grid = FAMILY_GRIDS[family]
    keys = list(grid)
    points = [
        dict(zip(keys, values, strict=True))
        for values in itertools.product(*grid.values())
    ]
    return points[:1] if smoke else points


def build_head(
    family: str, params: dict[str, Any], seed: int = 42, smoke: bool = False
):
    if family == "logreg":
        return LogRegHead(C=params["C"])
    if family == "knn":
        return KNNHead(k=params["k"])
    if family in ("mlp_bce", "mlp_asl"):
        return TorchMLPHead(
            hidden=params["hidden"],
            dropout=params["dropout"],
            loss="asl" if family == "mlp_asl" else "bce",
            gamma_neg=params.get("gamma_neg", 4.0),
            max_epochs=5 if smoke else 200,
            seed=seed,
        )
    raise ValueError(f"Unknown family: {family}")


# ------------------------------------------------------------------ logreg


class LogRegHead:
    """Baseline: StandardScaler + логистическая регрессия на каждый тег."""

    def __init__(self, C: float = 1.0) -> None:
        self.C = C
        self.pipeline = None

    def fit(self, x: np.ndarray, y: np.ndarray, groups=None) -> LogRegHead:
        from sklearn.linear_model import LogisticRegression
        from sklearn.multiclass import OneVsRestClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        self.pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    OneVsRestClassifier(
                        LogisticRegression(C=self.C, max_iter=2000), n_jobs=-1
                    ),
                ),
            ]
        )
        self.pipeline.fit(x, y)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.pipeline.predict_proba(x), dtype=np.float32)


# ------------------------------------------------------------------ knn


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


class KNNHead:
    """k ближайших соседей по косинусной близости.

    Оценка тега = доля соседей с этим тегом, взвешенная по близости.
    Модель хранит нормированные эмбеддинги и разметку train.
    """

    def __init__(self, k: int = 20, chunk_size: int = 1024) -> None:
        self.k = k
        self.chunk_size = chunk_size
        self.x_train: np.ndarray | None = None
        self.y_train: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray, groups=None) -> KNNHead:
        self.x_train = _l2_normalize(x)
        self.y_train = np.asarray(y, dtype=np.float32)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        xq = _l2_normalize(x)
        k = min(self.k, len(self.x_train))
        out = np.empty((len(xq), self.y_train.shape[1]), dtype=np.float32)
        for start in range(0, len(xq), self.chunk_size):
            sims = xq[start : start + self.chunk_size] @ self.x_train.T
            idx = np.argpartition(-sims, k - 1, axis=1)[:, :k]
            w = np.clip(np.take_along_axis(sims, idx, axis=1), 0.0, None)
            # Если все соседи "противоположны" (близость <= 0), веса равные.
            w[w.sum(axis=1) == 0] = 1.0
            neighbours = self.y_train[idx]  # (batch, k, n_tags)
            out[start : start + len(w)] = np.einsum(
                "bk,bkt->bt", w, neighbours
            ) / w.sum(axis=1, keepdims=True)
        return out


# ------------------------------------------------------------------ torch mlp


def asymmetric_loss(logits, targets, gamma_neg=4.0, gamma_pos=0.0, clip=0.05):
    """Asymmetric Loss для многометочной классификации.

    Уменьшает вклад "лёгких" отрицательных примеров (gamma_neg) и полностью
    обнуляет штраф, если модель уверена в отсутствии тега (clip). Это
    подходит для неполной разметки: отсутствие тега не всегда значит,
    что он не подходит треку.
    """
    import torch
    import torch.nn.functional as F

    p = torch.sigmoid(logits)
    p_neg = (1.0 - p + clip).clamp(max=1.0) if clip > 0 else 1.0 - p
    log_pos = F.logsigmoid(logits)
    log_neg = torch.log(p_neg.clamp(min=1e-8))
    pt = p * targets + p_neg * (1.0 - targets)
    gamma = gamma_pos * targets + gamma_neg * (1.0 - targets)
    weight = torch.pow(1.0 - pt, gamma)
    loss = targets * log_pos + (1.0 - targets) * log_neg
    return -(loss * weight).mean()


def _mean_ap(y: np.ndarray, scores: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    positives = y.sum(axis=0)
    cols = np.flatnonzero((positives > 0) & (positives < len(y)))
    if cols.size == 0:
        return float("nan")
    return float(
        np.mean([average_precision_score(y[:, j], scores[:, j]) for j in cols])
    )


class TorchMLPHead:
    """MLP на PyTorch: Linear -> ReLU -> Dropout -> Linear, сигмоида на каждый тег.

    loss="bce": обычная бинарная кросс-энтропия, loss="asl": Asymmetric Loss.
    Ранняя остановка по mean AP на отложенной части train (holdout по артистам).
    Веса хранятся как numpy-массивы, поэтому объект сериализуется через joblib
    без зависимости от версии torch.
    """

    def __init__(
        self,
        hidden: int = 512,
        dropout: float = 0.3,
        loss: str = "bce",
        gamma_neg: float = 4.0,
        gamma_pos: float = 0.0,
        clip: float = 0.05,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 64,
        max_epochs: int = 200,
        patience: int = 15,
        holdout: float = 0.1,
        seed: int = 42,
    ) -> None:
        self.hidden = hidden
        self.dropout = dropout
        self.loss = loss
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.holdout = holdout
        self.seed = seed
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.state_: dict[str, np.ndarray] | None = None
        self.n_in_: int | None = None
        self.n_out_: int | None = None
        self.best_epoch_: int | None = None
        self._net = None

    # -- pickling: сеть не сериализуем, только веса numpy
    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_net"] = None
        return state

    def _build_net(self):
        from torch import nn

        return nn.Sequential(
            nn.Linear(self.n_in_, self.hidden),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden, self.n_out_),
        )

    def _loss_fn(self, logits, targets):
        import torch.nn.functional as F

        if self.loss == "asl":
            return asymmetric_loss(
                logits, targets, self.gamma_neg, self.gamma_pos, self.clip
            )
        return F.binary_cross_entropy_with_logits(logits, targets)

    def _split(self, n: int, groups) -> tuple[np.ndarray, np.ndarray]:
        from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit

        if groups is not None and len(np.unique(groups)) > 1:
            splitter = GroupShuffleSplit(
                n_splits=1, test_size=self.holdout, random_state=self.seed
            )
            return next(splitter.split(np.zeros(n), groups=groups))
        splitter = ShuffleSplit(
            n_splits=1, test_size=self.holdout, random_state=self.seed
        )
        return next(splitter.split(np.zeros(n)))

    def fit(self, x: np.ndarray, y: np.ndarray, groups=None) -> TorchMLPHead:
        import torch

        torch.manual_seed(self.seed)
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        self.n_in_, self.n_out_ = x.shape[1], y.shape[1]
        self.mean_ = x.mean(axis=0)
        self.std_ = np.maximum(x.std(axis=0), 1e-6)
        xs = (x - self.mean_) / self.std_

        fit_idx, hold_idx = self._split(len(xs), groups)
        x_fit = torch.from_numpy(xs[fit_idx])
        y_fit = torch.from_numpy(y[fit_idx])
        x_hold = torch.from_numpy(xs[hold_idx])
        y_hold = y[hold_idx]

        net = self._build_net()
        optimizer = torch.optim.AdamW(
            net.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        generator = torch.Generator().manual_seed(self.seed)

        best_score, best_state, best_epoch, stale = -np.inf, None, 0, 0
        for epoch in range(1, self.max_epochs + 1):
            net.train()
            order = torch.randperm(len(x_fit), generator=generator)
            for start in range(0, len(order), self.batch_size):
                batch = order[start : start + self.batch_size]
                optimizer.zero_grad()
                loss = self._loss_fn(net(x_fit[batch]), y_fit[batch])
                loss.backward()
                optimizer.step()

            net.eval()
            with torch.no_grad():
                hold_scores = torch.sigmoid(net(x_hold)).numpy()
            score = _mean_ap(y_hold, hold_scores)
            if np.isnan(score):
                # В отложенной части нет ни одного оцениваемого тега:
                # ориентируемся на функцию потерь (чем меньше, тем лучше).
                with torch.no_grad():
                    score = -float(self._loss_fn(net(x_hold), torch.from_numpy(y_hold)))
            if score > best_score:
                best_score, best_epoch, stale = score, epoch, 0
                best_state = {
                    k: v.detach().clone() for k, v in net.state_dict().items()
                }
            else:
                stale += 1
                if stale >= self.patience:
                    break

        net.load_state_dict(best_state)
        net.eval()
        self.best_epoch_ = best_epoch
        self.state_ = {k: v.cpu().numpy() for k, v in net.state_dict().items()}
        self._net = net
        return self

    def _network(self):
        if self._net is None:
            import torch

            net = self._build_net()
            net.load_state_dict(
                {k: torch.from_numpy(v) for k, v in self.state_.items()}
            )
            net.eval()
            self._net = net
        return self._net

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        import torch

        xs = (np.asarray(x, dtype=np.float32) - self.mean_) / self.std_
        net = self._network()
        with torch.no_grad():
            scores = torch.sigmoid(net(torch.from_numpy(xs))).numpy()
        return scores.astype(np.float32)
