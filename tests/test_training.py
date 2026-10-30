"""Тесты кода обучения: метрики, классификаторы категорий, обёртка модели, promote.

Код training/ не является пакетом (скрипты запускаются как
python training/train.py), поэтому добавляем папку в sys.path.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))

from common import (  # noqa: E402
    DEFAULT_GROUPS_FILE,
    data_version,
    label_matrix,
    load_tag_groups,
    require_dataset_uri,
    target_labels,
    targets_for,
)
from metrics import (  # noqa: E402
    category_metrics,
    hit_at_k,
    macro_pr_auc,
    macro_precision_at_k,
    per_tag_precision_at_k,
    resampling_ci,
    tune_thresholds,
)
from music_tagger_pyfunc import predict_tags  # noqa: E402
from registry import pick_best  # noqa: E402
from tagger_heads import (  # noqa: E402
    KNNHead,
    LogRegHead,
    build_head,
    grid_points,
)


@pytest.fixture
def toy():
    """Два хорошо разделимых тега: тег j "включает" свою координату эмбеддинга."""
    rng = np.random.default_rng(0)
    n, dim = 240, 16
    y = (rng.random((n, 2)) < 0.3).astype(np.int8)
    y[y.sum(axis=1) == 0, 0] = 1
    x = rng.normal(size=(n, dim)).astype(np.float32)
    x[:, 0] += 3 * y[:, 0]
    x[:, 1] += 3 * y[:, 1]
    groups = np.repeat(np.arange(n // 4), 4).astype(str)
    return x, y, groups


# ------------------------------------------------------------------ metrics


def test_precision_at_k_counts_hits_in_top_k():
    y = np.array([[1], [0], [1], [0], [1]])
    scores = np.array([[0.9], [0.8], [0.7], [0.1], [0.2]])

    # top-2: треки 0 и 1, верный только трек 0
    assert per_tag_precision_at_k(y, scores, k=2)[0] == pytest.approx(0.5)


def test_precision_at_k_skips_tags_with_too_few_positives():
    y = np.array([[1, 1], [0, 1], [0, 1], [0, 0]])
    scores = np.random.default_rng(0).random((4, 2))

    result = per_tag_precision_at_k(y, scores, k=3)

    assert np.isnan(result[0])  # 1 положительный < k
    assert not np.isnan(result[1])


def test_hit_at_k_ignores_tracks_without_tags():
    y = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
    scores = np.array([[0.9, 0.5, 0.1], [0.1, 0.2, 0.9], [0.3, 0.3, 0.3]])

    # трек 0: top-2 = {0, 1}, есть тег 1 -> попадание; трек 1: top-2 = {2, 1} -> нет
    assert hit_at_k(y, scores, k=2) == pytest.approx(0.5)


def test_tune_thresholds_separates_perfect_scores():
    y = np.array([[0], [0], [1], [1]])
    scores = np.array([[0.1], [0.2], [0.7], [0.9]])

    threshold = tune_thresholds(y, scores)[0]

    assert 0.2 < threshold <= 0.7
    assert tune_thresholds(np.zeros((3, 1)), np.ones((3, 1)))[0] == pytest.approx(0.5)


def test_category_metrics_names(toy):
    x, y, _ = toy

    metrics, per_tag = category_metrics("val_mood", y, x[:, :2], np.array([0.5, 0.5]))

    assert metrics["val_mood_precision_at_10"] == pytest.approx(1.0)
    assert metrics["val_mood_pr_auc_macro"] > 0.9
    assert {"val_mood_hit_at_3", "val_mood_lift", "val_mood_f1_micro"} <= set(metrics)
    assert set(per_tag) >= {"ap", "precision_at_10", "threshold"}


# ------------------------------------------------------------------ heads


@pytest.mark.parametrize("head", [LogRegHead(C=1.0), KNNHead(k=10)])
def test_head_learns_and_survives_pickling(head, toy, tmp_path):
    x, y, groups = toy
    head.fit(x[:180], y[:180], groups[:180])
    scores = head.predict_proba(x[180:])

    assert scores.shape == (60, 2)
    metrics, _ = category_metrics("t", y[180:], scores)
    assert metrics["t_pr_auc_macro"] > 0.8

    joblib.dump(head, tmp_path / "head.joblib")
    restored = joblib.load(tmp_path / "head.joblib")
    np.testing.assert_allclose(restored.predict_proba(x[180:]), scores, rtol=1e-5)


def test_grid_points_and_smoke():
    assert len(grid_points("mlp_asl")) == 8
    assert grid_points("logreg", smoke=True) == [{"C": 0.01}]
    with pytest.raises(ValueError):
        build_head("unknown", {})


def test_torch_mlp_heads(toy, tmp_path):
    pytest.importorskip("torch")
    x, y, groups = toy
    for family in ("mlp_bce", "mlp_asl"):
        params = {"hidden": 32, "dropout": 0.2, "gamma_neg": 4}
        head = build_head(family, params, seed=0)
        head.max_epochs = 30
        head.fit(x[:180], y[:180], groups[:180])
        scores = head.predict_proba(x[180:])

        metrics, _ = category_metrics("t", y[180:], scores)
        assert metrics["t_pr_auc_macro"] > 0.8, family
        assert head.best_epoch_ >= 1

        joblib.dump(head, tmp_path / f"{family}.joblib")
        restored = joblib.load(tmp_path / f"{family}.joblib")
        np.testing.assert_allclose(restored.predict_proba(x[180:]), scores, rtol=1e-5)


# ------------------------------------------------------------------ pyfunc output


def test_predict_tags_returns_every_category(toy):
    x, y, _ = toy
    head = KNNHead(k=5).fit(x, y)
    heads = {
        "mood": {"model": head, "tags": ["calm", "happy"], "thresholds": [2.0, 2.0]},
        "genre": {"model": head, "tags": ["rock", "pop"], "thresholds": [0.0, 0.0]},
    }

    result = predict_tags(heads, x[:2], max_tags=1)

    assert len(result) == 2
    # порог не пройден -> один самый вероятный тег
    assert len(result[0]["mood"]) == 1
    # порог пройден обоими тегами, но max_tags=1
    assert len(result[0]["genre"]) == 1
    # категории без классификатора -> пустой список
    assert result[0]["instrument"] == []
    assert result[0]["theme"] == []


# ------------------------------------------------------------------ promote

OVERALL = "val_pr_auc_overall"


def _row(version, sha, overall, genre=0.5, groups="g1", aliases=()):
    return {
        "version": version,
        "aliases": list(aliases),
        "data": "v",
        "data_sha256": sha,
        "groups_sha256": groups,
        "families": "-",
        "metrics": {
            OVERALL: overall,
            "val_mood_pr_auc_macro": 0.3,
            "val_genre_pr_auc_macro": genre,
        },
    }


def test_promote_compares_only_same_dataset_and_groups():
    rows = [
        _row(1, "old", overall=0.9),
        _row(2, "new", overall=0.6, groups="g0"),  # другая группировка тегов
        _row(3, "new", overall=0.4),
        _row(4, "new", overall=0.5),
    ]

    best, _ = pick_best(rows, OVERALL, None, None, 0.0)

    assert best["version"] == 4


def test_promote_guard_blocks_regression_in_any_category():
    champion = _row(2, "new", overall=0.4, genre=0.6, aliases=["champion"])
    rows = [champion, _row(3, "new", overall=0.5, genre=0.3)]

    best, reason = pick_best(rows, OVERALL, None, champion, 0.0)

    assert best is None
    assert "val_genre_pr_auc_macro" in reason

    best, _ = pick_best(rows, OVERALL, None, champion, 0.5)
    assert best["version"] == 3


def test_promote_skips_guard_for_champion_in_other_setup():
    champion = _row(1, "old", overall=0.9, genre=0.9, aliases=["champion"])
    rows = [champion, _row(2, "new", overall=0.4, genre=0.1)]

    best, reason = pick_best(rows, OVERALL, None, champion, 0.0)

    assert best["version"] == 2
    assert "другой постановке" in reason


# ------------------------------------------------------------------ labels


GROUPS = {
    "mood": {"calm": ["calm", "relaxing"], "happy": ["happy"]},
    "theme": {"cinematic": ["film", "trailer"]},
}


def _frame():
    return pd.DataFrame(
        {
            "genre": ["rock|pop", ""],
            "instrument": ["", "piano"],
            "mood": ["calm|relaxing|film", "melodic"],
        }
    )


def test_target_labels_map_tags_to_groups():
    df = _frame()

    assert target_labels(df, "mood", GROUPS).tolist() == [["calm"], []]
    assert target_labels(df, "theme", GROUPS).tolist() == [["cinematic"], []]
    assert target_labels(df, "genre", GROUPS).tolist() == [["rock", "pop"], []]
    # без групп mood это исходные теги, а theme недоступна
    assert target_labels(df, "mood").tolist() == [
        ["calm", "relaxing", "film"],
        ["melodic"],
    ]
    with pytest.raises(ValueError):
        target_labels(df, "theme")


def test_label_matrix_and_targets():
    labels = target_labels(_frame(), "mood", GROUPS)

    y = label_matrix(labels, ["calm", "happy"])

    assert y.tolist() == [[1, 0], [0, 0]]
    assert targets_for(GROUPS) == ("genre", "instrument", "mood", "theme")
    assert targets_for(None) == ("genre", "instrument", "mood")


def test_load_tag_groups_rejects_duplicates(tmp_path):
    path = tmp_path / "groups.json"
    path.write_text(
        json.dumps({"mood": {"a": ["calm"]}, "theme": {"b": ["calm"]}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_tag_groups(path)


def test_repository_tag_groups_are_valid():
    groups = load_tag_groups(DEFAULT_GROUPS_FILE)

    assert groups["mood"] and groups["theme"]


def test_resampling_ci_is_centered_on_point_estimate(toy):
    x, y, _ = toy
    rng = np.random.default_rng(1)
    scores = x[:, :2] + rng.normal(scale=3.0, size=(len(x), 2))  # шумная модель

    point = macro_pr_auc(y, scores)
    low, high = resampling_ci(y, scores, macro_pr_auc, n_resamples=50)

    assert low < point < high
    assert high - point == pytest.approx(point - low)
    assert 0 < high - low < 0.5
    assert macro_precision_at_k(y, x[:, :2], 10) == pytest.approx(1.0)


# ------------------------------------------------------------------ data version


def test_data_version_from_path_convention():
    uri = "s3://datasets/mtg-jamendo-moodtheme/v12/embeddings.parquet"

    assert data_version(uri) == "v12"
    # путь не по соглашению: версия по хэшу файла
    local = "training/data/out/embeddings.parquet"
    assert data_version(local, file_sha="abcdef0123456789") == "sha-abcdef01"
    assert data_version(local) == "unknown"
    assert data_version("s3://datasets/x/version2/e.parquet", "ff00aa11bb") == (
        "sha-ff00aa11"
    )


def test_dataset_uri_is_required():
    with pytest.raises(SystemExit):
        require_dataset_uri(None)
    with pytest.raises(SystemExit):
        require_dataset_uri("")
    assert require_dataset_uri("s3://datasets/a/v1/e.parquet").endswith("e.parquet")
