"""Обучение классификатора тегов: отдельная модель для каждой категории.

Категории (с файлом групп training/tag_groups.json, по умолчанию):
    genre, instrument  теги MTG-Jamendo как есть;
    mood, theme        группы тегов mood/theme (например, calm = calm, relaxing, ...).
Без групп (--no-groups) категорий три: genre, instrument, mood (исходные теги).

Схема одного запуска (эксперимент music-tagger):

    родительский Run "tagger-<версия данных>-groups"
      ├── вложенный Run "mood/logreg"        CV по артистам внутри train,
      ├── вложенный Run "mood/knn"           лучшие гиперпараметры,
      ├── ...                                метрики на validation
      └── "theme/mlp_asl"
    затем в родительском Run:
      выбор лучшего семейства в каждой категории по PR-AUC на validation,
      сборка классификаторов в одну pyfunc-модель,
      финальная оценка на test (один раз, с доверительными интервалами),
      общая метрика (взвешенное среднее по категориям), регистрация версии.

Роли выборок:
    train      - обучение, подбор гиперпараметров и порогов (GroupKFold по artist_id);
    validation - только сравнение семейств моделей внутри категории;
    test       - только итоговая оценка собранной модели.

Каждая категория обучается и оценивается только на треках, где она размечена
(например, классификатор инструментов не видит треки без тегов инструментов).

Примеры (из корня репозитория, перед каждой командой
"uv run --env-file .env --group training"):
    python training/train.py --data s3://datasets/mtg-jamendo-moodtheme/v2/embeddings.parquet
    python training/train.py --smoke --no-register
    python training/train.py --families logreg knn --categories mood theme
    python training/train.py --weights mood=2 theme=1 genre=1 instrument=1
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import mlflow
import numpy as np
import pandas as pd
import sklearn
from common import (
    CATEGORIES,
    DEFAULT_GROUPS_FILE,
    EMBEDDING_CONTRACT,
    EMBEDDING_DIM,
    TARGETS,
    Config,
    data_version,
    features,
    file_digest,
    label_counts,
    label_matrix,
    load_dataset,
    load_tag_groups,
    log_dataset_inputs,
    require_dataset_uri,
    target_labels,
    targets_for,
)
from metrics import (
    category_metrics,
    macro_pr_auc,
    resampling_ci,
    tune_thresholds,
)
from mlflow.models import ModelSignature
from mlflow.types.schema import Schema, TensorSpec
from music_tagger_pyfunc import MusicTaggerModel
from sklearn.model_selection import GroupKFold
from tagger_heads import FAMILIES, build_head, grid_points

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).parent
CODE_PATHS = [str(HERE / "music_tagger_pyfunc.py"), str(HERE / "tagger_heads.py")]
SPLITS = ("train", "validation", "test")

K = 10  # precision@K: сколько треков пользователь видит по запросу
HIT_K = 3  # hit@K: сколько тегов показываем у трека
# PR-AUC учитывает все положительные примеры, а не 10 верхних треков,
# поэтому стабильнее P@10: по нему подбираются гиперпараметры и семейство.
CV_METRIC = "pr_auc_macro"
PRIMARY_METRIC = "val_pr_auc_overall"  # сравнение версий в registry.py promote


def selection_metric(category: str) -> str:
    return f"val_{category}_pr_auc_macro"


def report_metric(category: str, split: str = "val") -> str:
    """Метрика продукта: из K треков по запросу сколько подходят."""
    return f"{split}_{category}_precision_at_{K}"


def log_metrics(metrics: dict[str, float]) -> None:
    """Логирует только конечные значения: NaN (например, P@10 без подходящих
    тегов) не попадает в MLflow, чтобы не мешать сортировке и сравнению Runs."""
    mlflow.log_metrics({k: float(v) for k, v in metrics.items() if np.isfinite(v)})


# ------------------------------------------------------------------ data


@dataclass
class CategoryData:
    category: str
    tags: list[str]
    x: dict[str, np.ndarray] = field(default_factory=dict)
    y: dict[str, np.ndarray] = field(default_factory=dict)
    groups: dict[str, np.ndarray] = field(default_factory=dict)


def prepare_category(
    df: pd.DataFrame, category: str, min_count: int, groups: dict | None
) -> CategoryData:
    """Только треки, у которых категория размечена: отсутствие меток категории
    это пропуск в разметке, а не отрицательный пример."""
    labels = target_labels(df, category, groups)
    labeled = (labels.map(len) > 0).to_numpy()
    train_mask = (df["split"] == "train").to_numpy() & labeled
    counts = label_counts(labels[train_mask])
    tags = sorted(counts[counts >= min_count].index)
    data = CategoryData(category=category, tags=tags)
    for split in SPLITS:
        mask = (df["split"] == split).to_numpy() & labeled
        part = df[mask]
        data.x[split] = features(part)
        data.y[split] = label_matrix(labels[mask], tags)
        data.groups[split] = part["artist_id"].astype(str).to_numpy()
    return data


# ------------------------------------------------------------------ candidates


@dataclass
class Candidate:
    category: str
    family: str
    run_id: str
    params: dict[str, Any]
    model: Any
    thresholds: np.ndarray
    val_metrics: dict[str, float]
    val_per_tag: dict[str, np.ndarray]
    cv_mean: float


def cross_validate(
    family: str,
    params: dict[str, Any],
    data: CategoryData,
    args: argparse.Namespace,
) -> tuple[float, float, np.ndarray]:
    """GroupKFold по артистам внутри train. Возвращает среднее и std метрики
    по фолдам и out-of-fold оценки (по ним подбираются пороги)."""
    x, y, groups = data.x["train"], data.y["train"], data.groups["train"]
    folds = GroupKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
    oof = np.zeros(y.shape, dtype=np.float32)
    fold_scores = []
    for train_idx, test_idx in folds.split(x, y, groups):
        head = build_head(family, params, seed=args.seed, smoke=args.smoke)
        head.fit(x[train_idx], y[train_idx], groups[train_idx])
        scores = head.predict_proba(x[test_idx])
        oof[test_idx] = scores
        metrics, _ = category_metrics("cv", y[test_idx], scores, k=K, hit_k=HIT_K)
        fold_scores.append(metrics[f"cv_{CV_METRIC}"])
    fold_scores = np.asarray(fold_scores, dtype=float)
    return float(np.nanmean(fold_scores)), float(np.nanstd(fold_scores)), oof


def train_candidate(
    family: str, data: CategoryData, args: argparse.Namespace
) -> Candidate:
    category = data.category
    with mlflow.start_run(run_name=f"{category}/{family}", nested=True) as run:
        mlflow.set_tags({"category": category, "family": family, "stage": "candidate"})
        rows, best = [], None
        for params in grid_points(family, smoke=args.smoke):
            started = time.perf_counter()
            mean, std, oof = cross_validate(family, params, data, args)
            rows.append(
                {
                    **params,
                    f"cv_{CV_METRIC}_mean": mean,
                    f"cv_{CV_METRIC}_std": std,
                    "seconds": round(time.perf_counter() - started, 1),
                }
            )
            print(
                f"  {category}/{family} {params}: cv {CV_METRIC}={mean:.4f}±{std:.4f}"
            )
            if best is None or mean > best[1]:
                best = (params, mean, std, oof)
        params, cv_mean, cv_std, oof = best

        # Пороги по out-of-fold предсказаниям train: validation в этом не участвует.
        thresholds = tune_thresholds(data.y["train"], oof)

        started = time.perf_counter()
        model = build_head(family, params, seed=args.seed, smoke=args.smoke)
        model.fit(data.x["train"], data.y["train"], data.groups["train"])
        fit_seconds = time.perf_counter() - started

        val_scores = model.predict_proba(data.x["validation"])
        val_metrics, val_per_tag = category_metrics(
            f"val_{category}",
            data.y["validation"],
            val_scores,
            thresholds,
            k=K,
            hit_k=HIT_K,
        )

        mlflow.log_params(
            {
                "category": category,
                "family": family,
                "cv.folds": args.cv_folds,
                "cv.metric": CV_METRIC,
                "grid": json.dumps(grid_points(family, smoke=args.smoke)),
                **{f"best.{k}": v for k, v in params.items()},
                "labels.n_tags": len(data.tags),
                "data.n_train_tracks": len(data.y["train"]),
            }
        )
        extra = {}
        if getattr(model, "best_epoch_", None) is not None:
            extra["best_epoch"] = float(model.best_epoch_)
        log_metrics(
            {
                f"cv_{category}_{CV_METRIC}": cv_mean,
                f"cv_{category}_{CV_METRIC}_std": cv_std,
                "fit_seconds": fit_seconds,
                **val_metrics,
                **extra,
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            pd.DataFrame(rows).to_csv(tmp_dir / "cv_results.csv", index=False)
            per_tag_frame(data.tags, val_per_tag, prefix="val").to_csv(
                tmp_dir / "per_tag_val.csv", index=False
            )
            mlflow.log_artifacts(str(tmp_dir), artifact_path="diagnostics")

        shown = (selection_metric(category), report_metric(category))
        print(
            f"[{category}/{family}] best={params} "
            + " ".join(f"{name}={val_metrics[name]:.4f}" for name in shown)
        )
        return Candidate(
            category=category,
            family=family,
            run_id=run.info.run_id,
            params=params,
            model=model,
            thresholds=thresholds,
            val_metrics=val_metrics,
            val_per_tag=val_per_tag,
            cv_mean=cv_mean,
        )


def choose(candidates: list[Candidate]) -> Candidate:
    """Лучший кандидат по PR-AUC на validation (при равенстве по P@K)."""

    def key(c: Candidate) -> tuple[float, float]:
        main = c.val_metrics[selection_metric(c.category)]
        tie = c.val_metrics[report_metric(c.category)]
        return (
            -np.inf if np.isnan(main) else main,
            -np.inf if np.isnan(tie) else tie,
        )

    return max(candidates, key=key)


# ------------------------------------------------------------------ artifacts


def per_tag_frame(
    tags: list[str], per_tag: dict[str, np.ndarray], prefix: str
) -> pd.DataFrame:
    frame = pd.DataFrame({"tag": tags})
    for name, values in per_tag.items():
        frame[f"{prefix}_{name}"] = values
    return frame


def plot_candidates(table: pd.DataFrame, path: Path) -> None:
    pivot = table.pivot(index="family", columns="category", values="val_pr_auc_macro")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    pivot.plot.bar(ax=ax, rot=0)
    ax.set_ylabel("PR-AUC macro (validation)")
    ax.set_title("Сравнение семейств моделей по категориям")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_ap_vs_frequency(frames: dict[str, pd.DataFrame], path: Path) -> None:
    fig, axes = plt.subplots(
        1, len(frames), figsize=(5 * len(frames), 4), squeeze=False
    )
    for ax, (category, frame) in zip(axes[0], frames.items(), strict=True):
        data = frame.dropna(subset=["test_ap"])
        ax.scatter(data["train_positives"], data["test_ap"], alpha=0.7)
        ax.set_xscale("log")
        ax.set_title(category)
        ax.set_xlabel("Примеров в train (log)")
        ax.set_ylabel("AP на test")
    fig.suptitle("Качество тега vs его частота")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def inference_latency_ms(wrapper: MusicTaggerModel, x: np.ndarray, repeats: int = 30):
    single = x[:1]
    wrapper.predict(None, single)  # прогрев
    timings = []
    for _ in range(repeats):
        started = time.perf_counter()
        wrapper.predict(None, single)
        timings.append((time.perf_counter() - started) * 1000)
    started = time.perf_counter()
    wrapper.predict(None, x)
    batch_ms = (time.perf_counter() - started) * 1000 / max(len(x), 1)
    return float(np.median(timings)), float(batch_ms)


def pip_requirements(families: set[str]) -> list[str]:
    reqs = [
        f"mlflow=={mlflow.__version__}",
        f"scikit-learn=={sklearn.__version__}",
        f"numpy=={np.__version__}",
        f"joblib=={joblib.__version__}",
    ]
    if any(f.startswith("mlp") for f in families):
        import torch

        reqs.append(f"torch=={torch.__version__.split('+')[0]}")
    return reqs


# ------------------------------------------------------------------ main flow


def overall_metrics(
    metrics: dict[str, float], selected: dict[str, Candidate], weights: dict[str, float]
) -> dict[str, float]:
    """Взвешенное среднее метрик по категориям: одно число для сравнения версий."""
    result = {}
    for split in ("val", "test"):
        for name in ("pr_auc_macro", f"precision_at_{K}"):
            pairs = [
                (weights.get(c, 1.0), metrics.get(f"{split}_{c}_{name}", np.nan))
                for c in selected
            ]
            pairs = [(w, v) for w, v in pairs if w > 0 and np.isfinite(v)]
            if pairs:
                total = sum(w for w, _ in pairs)
                short = "pr_auc" if name == "pr_auc_macro" else name
                result[f"{split}_{short}_overall"] = (
                    sum(w * v for w, v in pairs) / total
                )
    return result


def assemble_and_log(
    df: pd.DataFrame,
    file_sha: str,
    datasets: dict[str, CategoryData],
    selected: dict[str, Candidate],
    candidates: list[Candidate],
    cfg: Config,
    args: argparse.Namespace,
) -> tuple[dict[str, float], str | None]:
    heads = {
        category: {
            "model": cand.model,
            "tags": datasets[category].tags,
            "thresholds": [float(t) for t in cand.thresholds],
            "family": cand.family,
            "params": cand.params,
            "run_id": cand.run_id,
        }
        for category, cand in selected.items()
    }

    # Финальная оценка на test: один раз, только для выбранных моделей.
    metrics: dict[str, float] = {}
    per_tag_frames: dict[str, pd.DataFrame] = {}
    for category, cand in selected.items():
        data = datasets[category]
        test_scores = cand.model.predict_proba(data.x["test"])
        test_metrics, test_per_tag = category_metrics(
            f"test_{category}",
            data.y["test"],
            test_scores,
            cand.thresholds,
            k=K,
            hit_k=HIT_K,
        )
        # Доверительный интервал PR-AUC (half-sampling): насколько метрика зависит
        # от того, какие треки попали в test. Для P@K интервал не считаем: на
        # половине треков меняется сам набор тегов с >= K примерами, и разброс
        # получается завышенным.
        low, high = resampling_ci(
            data.y["test"],
            test_scores,
            macro_pr_auc,
            n_resamples=args.ci_resamples,
            seed=args.seed,
        )
        test_metrics[f"test_{category}_pr_auc_macro_ci_low"] = low
        test_metrics[f"test_{category}_pr_auc_macro_ci_high"] = high
        metrics.update(cand.val_metrics)
        metrics.update(test_metrics)
        frame = per_tag_frame(data.tags, cand.val_per_tag, prefix="val").merge(
            per_tag_frame(data.tags, test_per_tag, prefix="test"), on="tag"
        )
        frame.insert(1, "train_positives", data.y["train"].sum(axis=0))
        per_tag_frames[category] = frame
    metrics.update(overall_metrics(metrics, selected, args.weights))
    log_metrics(metrics)
    mlflow.set_tags({f"selected.{c}": cand.family for c, cand in selected.items()})
    mlflow.log_params(
        {
            **{f"selected.{c}.family": cand.family for c, cand in selected.items()},
            **{f"selected.{c}.run_id": cand.run_id for c, cand in selected.items()},
            **{
                f"selected.{c}.params": json.dumps(cand.params)
                for c, cand in selected.items()
            },
        }
    )

    wrapper = MusicTaggerModel()
    wrapper.heads = heads
    wrapper.max_tags = args.max_tags
    test_df = df[df["split"] == "test"]
    single_ms, batch_ms = inference_latency_ms(wrapper, features(test_df))
    log_metrics(
        {"inference_ms_single": single_ms, "inference_ms_per_track_batch": batch_ms}
    )

    table = pd.DataFrame(
        [
            {
                "category": c.category,
                "family": c.family,
                "params": json.dumps(c.params),
                "cv_pr_auc_macro": c.cv_mean,
                "val_precision_at_k": c.val_metrics[report_metric(c.category)],
                "val_pr_auc_macro": c.val_metrics[f"val_{c.category}_pr_auc_macro"],
                f"val_hit_at_{HIT_K}": c.val_metrics[
                    f"val_{c.category}_hit_at_{HIT_K}"
                ],
                "selected": selected[c.category].run_id == c.run_id,
                "run_id": c.run_id,
            }
            for c in candidates
        ]
    )

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        diag = tmp_dir / "diagnostics"
        diag.mkdir()
        table.to_csv(diag / "candidates.csv", index=False)
        plot_candidates(table, diag / "candidates_val_pr_auc.png")
        for category, frame in per_tag_frames.items():
            frame.to_csv(diag / f"per_tag_metrics_{category}.csv", index=False)
        plot_ap_vs_frequency(per_tag_frames, diag / "ap_vs_frequency.png")

        sample_df = test_df.head(5)
        sample = [
            {"track_id": tid, "true": true, "predicted": pred}
            for tid, true, pred in zip(
                sample_df["track_id"],
                sample_df[list(CATEGORIES)].to_dict("records"),
                wrapper.predict(None, features(sample_df)),
                strict=True,
            )
        ]
        (diag / "sample_predictions.json").write_text(
            json.dumps(sample, ensure_ascii=False, indent=2)
        )
        mlflow.log_artifacts(str(diag), artifact_path="diagnostics")
        if args.groups_path is not None:
            mlflow.log_artifact(str(args.groups_path), artifact_path="labels")

        heads_path = tmp_dir / "heads.joblib"
        meta_path = tmp_dir / "meta.json"
        joblib.dump(heads, heads_path)
        meta = {
            "max_tags": args.max_tags,
            "embedding": EMBEDDING_CONTRACT,
            "label_groups_sha256": args.groups_sha,
            "categories": {
                c: {
                    "family": h["family"],
                    "params": h["params"],
                    "n_tags": len(h["tags"]),
                }
                for c, h in heads.items()
            },
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))

        signature = ModelSignature(
            inputs=Schema([TensorSpec(np.dtype(np.float32), (-1, EMBEDDING_DIM))])
        )
        model_info = mlflow.pyfunc.log_model(
            name="model",
            python_model=MusicTaggerModel(),
            artifacts={"heads": str(heads_path), "meta": str(meta_path)},
            code_paths=CODE_PATHS,
            signature=signature,
            input_example=features(test_df.head(2)),
            pip_requirements=pip_requirements({h["family"] for h in heads.values()}),
            metadata={
                "embedding": EMBEDDING_CONTRACT,
                "data_version": args.data_version,
                "data_sha256": file_sha,
                "label_groups_sha256": args.groups_sha,
                "categories": list(heads),
            },
            registered_model_name=None if args.no_register else cfg.registered_model,
        )

    version = getattr(model_info, "registered_model_version", None)
    if version is not None:
        client = mlflow.MlflowClient()
        shown = [PRIMARY_METRIC, *(selection_metric(c) for c in selected)]
        version_tags = {
            "data_uri": args.data,
            "data_version": args.data_version,
            "data_sha256": file_sha,
            "label_groups_sha256": args.groups_sha,
            **{f"family.{c}": cand.family for c, cand in selected.items()},
            **{
                name: f"{metrics[name]:.4f}"
                for name in shown
                if np.isfinite(metrics.get(name, np.nan))
            },
        }
        for key, value in version_tags.items():
            client.set_model_version_tag(cfg.registered_model, str(version), key, value)
        summary = ", ".join(
            f"{c}: {cand.family} PR-AUC={metrics[selection_metric(c)]:.3f}"
            for c, cand in selected.items()
        )
        client.update_model_version(
            cfg.registered_model,
            str(version),
            description=(
                f"data {args.data_version}, groups {args.groups_sha}; "
                f"overall PR-AUC={metrics[PRIMARY_METRIC]:.3f}; {summary}"
            ),
        )
    return metrics, None if version is None else str(version)


def parse_weights(items: list[str] | None) -> dict[str, float]:
    """["mood=2", "theme=1"] -> {"mood": 2.0, "theme": 1.0}; остальные веса 1."""
    weights: dict[str, float] = {}
    for item in items or []:
        name, _, value = item.partition("=")
        if name not in TARGETS or not value:
            raise SystemExit(f"Неверный вес {item!r}, нужен вид mood=2")
        weights[name] = float(value)
    return weights


def main() -> None:
    cfg = Config.from_env()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", default=cfg.dataset_uri, help="s3:// URI или путь")
    parser.add_argument("--refresh-data", action="store_true")
    parser.add_argument(
        "--groups",
        type=Path,
        default=DEFAULT_GROUPS_FILE,
        help="файл групп тегов mood/theme",
    )
    parser.add_argument(
        "--no-groups",
        action="store_true",
        help="без групп: исходные теги mood/theme одной категорией mood",
    )
    parser.add_argument("--categories", nargs="+", choices=TARGETS, default=None)
    parser.add_argument(
        "--families", nargs="+", choices=FAMILIES, default=list(FAMILIES)
    )
    parser.add_argument(
        "--weights",
        nargs="+",
        default=None,
        help="веса категорий в общей метрике, например mood=2 theme=1 (по умолчанию 1)",
    )
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--min-count", type=int, default=20)
    parser.add_argument("--max-tags", type=int, default=3)
    parser.add_argument(
        "--ci-resamples",
        type=int,
        default=200,
        help="число половинных подвыборок для доверительных интервалов",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-register", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="быстрая проверка: 2 фолда, одна точка сетки, 5 эпох MLP",
    )
    args = parser.parse_args()
    if args.smoke:
        args.cv_folds = 2
        args.ci_resamples = min(args.ci_resamples, 20)
    args.weights = parse_weights(args.weights)

    groups = None if args.no_groups else load_tag_groups(args.groups)
    args.groups_path = None if args.no_groups else args.groups
    args.groups_sha = "none" if args.no_groups else file_digest(args.groups)
    categories = args.categories or list(targets_for(groups))
    unsupported = [c for c in categories if c not in targets_for(groups)]
    if unsupported:
        raise SystemExit(f"Категории {unsupported} требуют файла групп")

    np.random.seed(args.seed)
    mlflow.set_tracking_uri(cfg.tracking_uri)
    mlflow.set_experiment(cfg.experiment)
    args.data = require_dataset_uri(args.data)
    df, file_sha = load_dataset(args.data, cfg, refresh=args.refresh_data)
    args.data_version = data_version(args.data, file_sha)

    datasets = {}
    for category in categories:
        data = prepare_category(df, category, args.min_count, groups)
        if not data.tags:
            raise SystemExit(
                f"{category}: нет меток с >= {args.min_count} примерами, "
                "уменьшите --min-count"
            )
        datasets[category] = data

    run_name = (
        f"tagger-{args.data_version}-{'tags' if args.no_groups else 'groups'}"
        + ("-smoke" if args.smoke else "")
    )
    with mlflow.start_run(run_name=run_name) as parent:
        mlflow.set_tags({"stage": "smoke" if args.smoke else "assembled"})
        mlflow.log_params(
            {
                "data.uri": args.data,
                "data.version": args.data_version,
                "data.file_sha256": file_sha,
                **{f"data.n_{s}": int((df["split"] == s).sum()) for s in SPLITS},
                "labels.groups_file": "none" if args.no_groups else str(args.groups),
                "labels.groups_sha256": args.groups_sha,
                "labels.categories": ",".join(categories),
                "cv.folds": args.cv_folds,
                "cv.group": "artist_id",
                "cv.metric": CV_METRIC,
                "selection.metric": "val_<category>_pr_auc_macro",
                "overall.weights": json.dumps(
                    {c: args.weights.get(c, 1.0) for c in categories}
                ),
                "labels.min_count": args.min_count,
                "families": ",".join(args.families),
                "max_tags": args.max_tags,
                "ci.method": "half-sampling",
                "ci.resamples": args.ci_resamples,
                "seed": args.seed,
                "sklearn_version": sklearn.__version__,
                **{f"labels.n_{c}": len(d.tags) for c, d in datasets.items()},
                **{f"data.n_train_{c}": len(d.y["train"]) for c, d in datasets.items()},
            }
        )
        log_dataset_inputs(df, source=args.data, splits=SPLITS)

        candidates: list[Candidate] = []
        selected: dict[str, Candidate] = {}
        for category, data in datasets.items():
            print(
                f"== {category}: {len(data.tags)} меток, "
                f"train {len(data.y['train'])} треков"
            )
            per_category = [
                train_candidate(family, data, args) for family in args.families
            ]
            candidates += per_category
            selected[category] = choose(per_category)
            print(f"== {category}: выбрано {selected[category].family}")

        metrics, version = assemble_and_log(
            df, file_sha, datasets, selected, candidates, cfg, args
        )

    print(f"\nRun {parent.info.run_id}, версия модели: {version}")
    for category, cand in selected.items():
        m = metrics
        pr, p10 = "pr_auc_macro", f"precision_at_{K}"
        print(
            f"  {category:<10} {cand.family:<8} "
            f"val PR-AUC={m[f'val_{category}_{pr}']:.3f}  "
            f"test PR-AUC={m[f'test_{category}_{pr}']:.3f} "
            f"[{m[f'test_{category}_{pr}_ci_low']:.3f}; "
            f"{m[f'test_{category}_{pr}_ci_high']:.3f}]  "
            f"test P@{K}={m[f'test_{category}_{p10}']:.3f}"
        )
    print(
        f"  overall    val PR-AUC={metrics.get('val_pr_auc_overall', np.nan):.3f}  "
        f"test PR-AUC={metrics.get('test_pr_auc_overall', np.nan):.3f}  "
        f"test P@{K}={metrics.get(f'test_precision_at_{K}_overall', np.nan):.3f}"
    )


if __name__ == "__main__":
    main()
