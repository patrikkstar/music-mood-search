"""Обучение классификатора тегов (genre / instrument / mood) поверх эмбеддингов.

Каждая модель = отдельный MLflow Run в эксперименте music-tagger:
параметры, датасеты (train/validation/test), метрики, артефакты и
pyfunc-модель, зарегистрированная в Model Registry как новая версия.

Примеры:
    uv run --group training python training/train.py --model logreg
    uv run --group training python training/train.py --model all
    uv run --group training python training/train.py --model mlp --min-count 30
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import joblib
import matplotlib
import mlflow
import numpy as np
import pandas as pd
import sklearn
from common import (
    CATEGORIES,
    EMBEDDING_DIM,
    Config,
    features,
    label_category,
    label_matrix,
    label_tag,
    load_dataset,
    log_dataset_inputs,
    select_labels,
)
from mlflow.models import ModelSignature
from mlflow.types.schema import Schema, TensorSpec
from music_tagger_pyfunc import MusicTaggerModel, decode_predictions
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PYFUNC_FILE = Path(__file__).with_name("music_tagger_pyfunc.py")
PRIMARY_METRIC = "val_pr_auc_macro"

# ------------------------------------------------------------------ models

MODEL_CONFIGS: dict[str, dict] = {
    # baseline: линейная модель, один классификатор на тег
    "logreg": {"kind": "logreg", "C": 1.0, "class_weight": None},
    # вариант 1: учитываем дисбаланс классов (редкие теги)
    "logreg_balanced": {"kind": "logreg", "C": 1.0, "class_weight": "balanced"},
    # вариант 2: другой preprocessing, сжимаем 1280 -> 256 через PCA
    "pca_logreg": {
        "kind": "logreg",
        "C": 1.0,
        "class_weight": "balanced",
        "pca_components": 256,
    },
    # вариант 3: нелинейная модель, общий скрытый слой для всех тегов
    "mlp": {
        "kind": "mlp",
        "hidden_layer_sizes": [512],
        "alpha": 1e-3,
        "learning_rate_init": 1e-3,
        "max_iter": 200,
    },
}


def build_pipeline(cfg: dict, seed: int) -> Pipeline:
    steps: list = [("scaler", StandardScaler())]
    if cfg.get("pca_components"):
        steps.append(
            ("pca", PCA(n_components=cfg["pca_components"], random_state=seed))
        )
    if cfg["kind"] == "logreg":
        clf = OneVsRestClassifier(
            LogisticRegression(
                C=cfg["C"],
                class_weight=cfg["class_weight"],
                max_iter=2000,
            ),
            n_jobs=-1,
        )
    elif cfg["kind"] == "mlp":
        clf = MLPClassifier(
            hidden_layer_sizes=tuple(cfg["hidden_layer_sizes"]),
            alpha=cfg["alpha"],
            learning_rate_init=cfg["learning_rate_init"],
            max_iter=cfg["max_iter"],
            early_stopping=True,
            random_state=seed,
        )
    else:
        raise ValueError(f"Unknown model kind: {cfg['kind']}")
    steps.append(("clf", clf))
    return Pipeline(steps)


# ------------------------------------------------------------------ metrics


def per_label_scores(y: np.ndarray, proba: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """AP и ROC-AUC по каждому тегу. NaN, если в выборке нет положительных."""
    ap = np.full(y.shape[1], np.nan)
    auc = np.full(y.shape[1], np.nan)
    for j in range(y.shape[1]):
        positives = y[:, j].sum()
        if 0 < positives < len(y):
            ap[j] = average_precision_score(y[:, j], proba[:, j])
            auc[j] = roc_auc_score(y[:, j], proba[:, j])
    return ap, auc


def tune_thresholds(y_val: np.ndarray, proba_val: np.ndarray) -> np.ndarray:
    """Порог на тег, максимизирующий F1 на validation (0.5, если нет примеров)."""
    thresholds = np.full(y_val.shape[1], 0.5, dtype=np.float32)
    grid = np.linspace(0.05, 0.95, 19)
    for j in range(y_val.shape[1]):
        if y_val[:, j].sum() == 0:
            continue
        scores = [
            f1_score(y_val[:, j], proba_val[:, j] >= t, zero_division=0) for t in grid
        ]
        thresholds[j] = grid[int(np.argmax(scores))]
    return thresholds


def split_metrics(
    prefix: str,
    y: np.ndarray,
    proba: np.ndarray,
    thresholds: np.ndarray,
    labels: list[str],
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    ap, auc = per_label_scores(y, proba)
    pred = (proba >= thresholds).astype(int)
    metrics = {
        f"{prefix}_pr_auc_macro": float(np.nanmean(ap)),
        f"{prefix}_roc_auc_macro": float(np.nanmean(auc)),
        f"{prefix}_f1_micro": float(
            f1_score(y, pred, average="micro", zero_division=0)
        ),
        f"{prefix}_f1_macro": float(
            f1_score(y, pred, average="macro", zero_division=0)
        ),
        f"{prefix}_n_evaluated_labels": float(np.sum(~np.isnan(ap))),
    }
    categories = np.array([label_category(label) for label in labels])
    for category in CATEGORIES:
        mask = categories == category
        if mask.any() and not np.all(np.isnan(ap[mask])):
            metrics[f"{prefix}_pr_auc_{category}"] = float(np.nanmean(ap[mask]))
            metrics[f"{prefix}_roc_auc_{category}"] = float(np.nanmean(auc[mask]))
    return metrics, ap, auc


def inference_latency_ms(model: MusicTaggerModel, x: np.ndarray, repeats: int = 30):
    """Задержка полного inference (модель + декодирование тегов) на 1 трек."""
    single = x[:1]
    model.predict(None, single)  # прогрев
    timings = []
    for _ in range(repeats):
        started = time.perf_counter()
        model.predict(None, single)
        timings.append((time.perf_counter() - started) * 1000)
    started = time.perf_counter()
    model.predict(None, x)
    batch_ms = (time.perf_counter() - started) * 1000 / max(len(x), 1)
    return float(np.median(timings)), float(batch_ms)


# ------------------------------------------------------------------ artifacts


def plot_worst_tags(per_tag: pd.DataFrame, path: Path, n: int = 20) -> None:
    data = per_tag.dropna(subset=["test_ap"]).nsmallest(n, "test_ap")
    fig, ax = plt.subplots(figsize=(8, 0.35 * len(data) + 1.5))
    ax.barh(data["label"], data["test_ap"], color="#c0504d")
    ax.invert_yaxis()
    ax.set_xlabel("Average precision (test)")
    ax.set_title(f"{n} худших тегов по AP на test")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_ap_vs_frequency(per_tag: pd.DataFrame, path: Path) -> None:
    data = per_tag.dropna(subset=["test_ap"])
    fig, ax = plt.subplots(figsize=(7, 5))
    for category in CATEGORIES:
        part = data[data["category"] == category]
        ax.scatter(part["train_positives"], part["test_ap"], label=category, alpha=0.7)
    ax.set_xscale("log")
    ax.set_xlabel("Число положительных примеров в train (log)")
    ax.set_ylabel("Average precision (test)")
    ax.set_title("Качество тега vs его частота")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_pr_curves(
    y: np.ndarray, proba: np.ndarray, labels: list[str], path: Path, n: int = 8
) -> None:
    top = np.argsort(-y.sum(axis=0))[:n]
    fig, ax = plt.subplots(figsize=(7, 6))
    for j in top:
        if y[:, j].sum() == 0:
            continue
        precision, recall, _ = precision_recall_curve(y[:, j], proba[:, j])
        ap = average_precision_score(y[:, j], proba[:, j])
        ax.plot(recall, precision, label=f"{labels[j]} (AP={ap:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("PR-кривые самых частых тегов (test)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ------------------------------------------------------------------ main flow


def train_one(
    name: str,
    df: pd.DataFrame,
    file_sha: str,
    cfg: Config,
    args: argparse.Namespace,
) -> str:
    model_cfg = MODEL_CONFIGS[name]
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "validation"]
    test_df = df[df["split"] == "test"]

    labels = select_labels(train_df, args.min_count, args.max_per_category)
    if not labels:
        raise SystemExit(
            "Нет тегов с достаточным числом примеров, уменьшите --min-count"
        )

    x_train, x_val, x_test = features(train_df), features(val_df), features(test_df)
    y_train = label_matrix(train_df, labels)
    y_val = label_matrix(val_df, labels)
    y_test = label_matrix(test_df, labels)

    with mlflow.start_run(run_name=name) as run:
        mlflow.set_tags({"model_type": name, "stage": "experiment"})
        mlflow.log_params({f"model.{k}": v for k, v in model_cfg.items()})
        mlflow.log_params(
            {
                "data.uri": args.data,
                "data.file_sha256": file_sha,
                "data.n_train": len(train_df),
                "data.n_validation": len(val_df),
                "data.n_test": len(test_df),
                "labels.min_count": args.min_count,
                "labels.max_per_category": args.max_per_category,
                "labels.n_total": len(labels),
                **{
                    f"labels.n_{c}": sum(label_category(lb) == c for lb in labels)
                    for c in CATEGORIES
                },
                "seed": args.seed,
                "sklearn_version": sklearn.__version__,
            }
        )
        log_dataset_inputs(df, source=args.data, splits=("train", "validation", "test"))

        pipeline = build_pipeline(model_cfg, args.seed)
        started = time.perf_counter()
        pipeline.fit(x_train, y_train)
        mlflow.log_metric("fit_seconds", time.perf_counter() - started)

        proba_val = pipeline.predict_proba(x_val)
        proba_test = pipeline.predict_proba(x_test)
        thresholds = tune_thresholds(y_val, proba_val)

        val_metrics, val_ap, _ = split_metrics(
            "val", y_val, proba_val, thresholds, labels
        )
        test_metrics, test_ap, test_auc = split_metrics(
            "test", y_test, proba_test, thresholds, labels
        )
        mlflow.log_metrics({**val_metrics, **test_metrics})

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            pipeline_path = tmp_dir / "pipeline.joblib"
            labels_path = tmp_dir / "labels.json"
            joblib.dump(pipeline, pipeline_path)
            labels_meta = {
                "labels": labels,
                "thresholds": [float(t) for t in thresholds],
                "max_tags": args.max_tags,
                "embedding_dim": EMBEDDING_DIM,
            }
            labels_path.write_text(
                json.dumps(labels_meta, ensure_ascii=False, indent=2)
            )

            # Метрики по каждому тегу.
            test_pred = (proba_test >= thresholds).astype(int)
            per_tag = pd.DataFrame(
                {
                    "label": labels,
                    "category": [label_category(lb) for lb in labels],
                    "tag": [label_tag(lb) for lb in labels],
                    "train_positives": y_train.sum(axis=0),
                    "test_positives": y_test.sum(axis=0),
                    "threshold": thresholds,
                    "val_ap": val_ap,
                    "test_ap": test_ap,
                    "test_roc_auc": test_auc,
                    "test_f1": f1_score(
                        y_test, test_pred, average=None, zero_division=0
                    ),
                }
            )
            per_tag.to_csv(tmp_dir / "per_tag_metrics.csv", index=False)
            plot_worst_tags(per_tag, tmp_dir / "worst_tags.png")
            plot_ap_vs_frequency(per_tag, tmp_dir / "ap_vs_frequency.png")
            plot_pr_curves(y_test, proba_test, labels, tmp_dir / "pr_curves_test.png")

            # Пример предсказаний на нескольких треках из test.
            examples = decode_predictions(proba_test[:5], labels, thresholds)
            sample = [
                {"track_id": tid, "true": true, "predicted": pred}
                for tid, true, pred in zip(
                    test_df["track_id"].head(5),
                    test_df[list(CATEGORIES)].head(5).to_dict("records"),
                    examples,
                    strict=True,
                )
            ]
            (tmp_dir / "sample_predictions.json").write_text(
                json.dumps(sample, ensure_ascii=False, indent=2)
            )
            for artifact in (
                "per_tag_metrics.csv",
                "worst_tags.png",
                "ap_vs_frequency.png",
                "pr_curves_test.png",
                "sample_predictions.json",
                "labels.json",
            ):
                mlflow.log_artifact(
                    str(tmp_dir / artifact), artifact_path="diagnostics"
                )

            # Задержка inference через ту же обёртку, что будет в сервисе.
            wrapper = MusicTaggerModel()
            wrapper.pipeline = pipeline
            wrapper.labels = labels
            wrapper.thresholds = thresholds
            wrapper.max_tags = args.max_tags
            single_ms, batch_ms = inference_latency_ms(wrapper, x_test)
            mlflow.log_metrics(
                {
                    "inference_ms_single": single_ms,
                    "inference_ms_per_track_batch": batch_ms,
                }
            )

            signature = ModelSignature(
                inputs=Schema([TensorSpec(np.dtype(np.float32), (-1, EMBEDDING_DIM))])
            )
            model_info = mlflow.pyfunc.log_model(
                name="model",
                python_model=MusicTaggerModel(),
                artifacts={
                    "pipeline": str(pipeline_path),
                    "labels": str(labels_path),
                },
                code_paths=[str(PYFUNC_FILE)],
                signature=signature,
                input_example=x_val[:2],
                pip_requirements=[
                    f"mlflow=={mlflow.__version__}",
                    f"scikit-learn=={sklearn.__version__}",
                    f"numpy=={np.__version__}",
                    f"joblib=={joblib.__version__}",
                ],
                registered_model_name=None
                if args.no_register
                else cfg.registered_model,
            )

        version = getattr(model_info, "registered_model_version", None)
        if version is not None:
            client = mlflow.MlflowClient()
            client.set_model_version_tag(
                cfg.registered_model,
                str(version),
                PRIMARY_METRIC,
                f"{val_metrics[PRIMARY_METRIC]:.4f}",
            )
            client.set_model_version_tag(
                cfg.registered_model, str(version), "model_type", name
            )
            client.update_model_version(
                cfg.registered_model,
                str(version),
                description=(
                    f"{name}: val PR-AUC macro {val_metrics[PRIMARY_METRIC]:.4f}, "
                    f"test PR-AUC macro {test_metrics['test_pr_auc_macro']:.4f}, "
                    f"{len(labels)} tags"
                ),
            )

        print(
            f"[{name}] run_id={run.info.run_id} version={version} "
            f"val_pr_auc_macro={val_metrics['val_pr_auc_macro']:.4f} "
            f"test_pr_auc_macro={test_metrics['test_pr_auc_macro']:.4f}"
        )
        return run.info.run_id


def main() -> None:
    cfg = Config.from_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=[*MODEL_CONFIGS, "all"], default="logreg")
    parser.add_argument("--data", default=cfg.dataset_uri, help="s3:// URI или путь")
    parser.add_argument("--min-count", type=int, default=20)
    parser.add_argument("--max-per-category", type=int, default=None)
    parser.add_argument("--max-tags", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-register", action="store_true")
    parser.add_argument("--refresh-data", action="store_true")
    args = parser.parse_args()

    mlflow.set_tracking_uri(cfg.tracking_uri)
    mlflow.set_experiment(cfg.experiment)

    df, file_sha = load_dataset(args.data, cfg, refresh=args.refresh_data)
    names = list(MODEL_CONFIGS) if args.model == "all" else [args.model]
    for name in names:
        train_one(name, df, file_sha, cfg, args)


if __name__ == "__main__":
    main()
