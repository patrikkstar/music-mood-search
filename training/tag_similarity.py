"""Анализ похожести тегов mood/theme: помогает решить, какие теги объединять в группы.

Идея: если треки с двумя тегами в среднем звучат одинаково (их средние
эмбеддинги почти совпадают), модель не сможет надёжно различить эти теги,
и их разумно объединить в одну группу.

Что считается (только по train, validation и test не используются):
  * похожесть тегов: косинусная близость средних эмбеддингов их треков
    (эмбеддинги стандартизованы, поэтому общий "средний трек" вычтен);
  * иерархическая кластеризация тегов и дендрограмма;
  * совместная встречаемость тегов (коэффициент Жаккара);
  * для черновика групп из tag_groups.json: средняя похожесть внутри группы
    и с ближайшей другой группой (хорошая группа: внутри заметно выше).

Результат: PNG и CSV в --out и в MLflow (эксперимент EDA, Run "tag-similarity").

Пример (из корня репозитория):
    uv run --env-file .env --group training python training/tag_similarity.py
    uv run --env-file .env --group training python training/tag_similarity.py \
        --n-clusters 12 --min-count 30
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from common import (
    Config,
    data_version,
    features,
    load_dataset,
    require_dataset_uri,
    split_tags,
    tag_counts,
)
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).parent
SOURCE_COLUMN = "mood"  # в MTG-Jamendo настроения и темы лежат в одной колонке


# ------------------------------------------------------------------ groups


def load_groups(path: Path) -> dict[str, dict[str, list[str]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {kind: data.get(kind, {}) for kind in ("mood", "theme")}


def tag_to_group(groups: dict[str, dict[str, list[str]]]) -> dict[str, tuple[str, str]]:
    """tag -> (kind, group), например "calm" -> ("mood", "calm")."""
    mapping = {}
    for kind, kind_groups in groups.items():
        for group, tags in kind_groups.items():
            for tag in tags:
                mapping[tag] = (kind, group)
    return mapping


# ------------------------------------------------------------------ similarity


def tag_centroids(
    x: np.ndarray, tags_per_track: list[list[str]], tags: list[str]
) -> np.ndarray:
    """Средний стандартизованный эмбеддинг треков каждого тега."""
    z = (x - x.mean(axis=0)) / np.maximum(x.std(axis=0), 1e-6)
    index = {tag: i for i, tag in enumerate(tags)}
    sums = np.zeros((len(tags), x.shape[1]), dtype=np.float64)
    counts = np.zeros(len(tags))
    for row, track_tags in enumerate(tags_per_track):
        for tag in track_tags:
            i = index.get(tag)
            if i is not None:
                sums[i] += z[row]
                counts[i] += 1
    return sums / counts[:, None]


def cosine_matrix(vectors: np.ndarray) -> np.ndarray:
    unit = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    sim = unit @ unit.T
    np.fill_diagonal(sim, 1.0)
    return np.clip(sim, -1.0, 1.0)


def jaccard_matrix(tags_per_track: list[list[str]], tags: list[str]) -> np.ndarray:
    index = {tag: i for i, tag in enumerate(tags)}
    y = np.zeros((len(tags_per_track), len(tags)), dtype=np.float32)
    for row, track_tags in enumerate(tags_per_track):
        for tag in track_tags:
            if tag in index:
                y[row, index[tag]] = 1
    both = y.T @ y
    counts = np.diag(both)
    union = counts[:, None] + counts[None, :] - both
    return np.where(union > 0, both / np.maximum(union, 1), 0.0)


def cluster_order(sim: np.ndarray, n_clusters: int) -> tuple[np.ndarray, np.ndarray]:
    """Порядок листьев дендрограммы и номера кластеров (средняя связь)."""
    distance = squareform(1.0 - sim, checks=False)
    tree = linkage(distance, method="average")
    order = dendrogram(tree, no_plot=True)["leaves"]
    clusters = fcluster(tree, t=min(n_clusters, len(sim)), criterion="maxclust")
    return np.asarray(order), clusters


# ------------------------------------------------------------------ reports


def group_cohesion(sim: pd.DataFrame, groups: dict[str, list[str]]) -> pd.DataFrame:
    """Похожесть внутри каждой группы черновика и с ближайшей другой группой."""
    present = {g: [t for t in tags if t in sim.index] for g, tags in groups.items()}
    present = {g: tags for g, tags in present.items() if tags}
    rows = []
    for group, tags in present.items():
        block = sim.loc[tags, tags].to_numpy()
        within = (
            float(block[~np.eye(len(tags), dtype=bool)].mean())
            if len(tags) > 1
            else np.nan
        )
        nearest_group, nearest_sim = None, -np.inf
        for other, other_tags in present.items():
            if other == group:
                continue
            between = float(sim.loc[tags, other_tags].to_numpy().mean())
            if between > nearest_sim:
                nearest_group, nearest_sim = other, between
        rows.append(
            {
                "group": group,
                "tags": ", ".join(tags),
                "n_tags": len(tags),
                "within_sim": within,
                "nearest_group": nearest_group,
                "nearest_group_sim": nearest_sim if nearest_group else np.nan,
                "margin": within - nearest_sim if nearest_group else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("margin", na_position="last")


def nearest_tags(sim: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    rows = []
    for tag in sim.index:
        others = sim.loc[tag].drop(tag).sort_values(ascending=False).head(k)
        rows.append(
            {
                "tag": tag,
                **{
                    f"nearest_{i + 1}": f"{t} ({v:.2f})"
                    for i, (t, v) in enumerate(others.items())
                },
            }
        )
    return pd.DataFrame(rows)


def plot_heatmap(
    matrix: pd.DataFrame,
    order: np.ndarray,
    labels: list[str],
    title: str,
    path: Path,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "RdBu_r",
) -> None:
    data = matrix.to_numpy()[np.ix_(order, order)]
    ticks = [labels[i] for i in order]
    size = max(6.0, 0.32 * len(ticks) + 2.5)
    fig, ax = plt.subplots(figsize=(size + 1.5, size))
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(ticks)), ticks, rotation=90, fontsize=8)
    ax.set_yticks(range(len(ticks)), ticks, fontsize=8)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_dendrogram(sim: np.ndarray, labels: list[str], title: str, path: Path) -> None:
    tree = linkage(squareform(1.0 - sim, checks=False), method="average")
    fig, ax = plt.subplots(figsize=(7, max(4.0, 0.28 * len(labels) + 1.5)))
    dendrogram(tree, labels=labels, orientation="right", ax=ax, leaf_font_size=8)
    ax.set_xlabel("1 - косинусная близость (чем левее слияние, тем похожее теги)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def off_diagonal_max(matrix: np.ndarray) -> float:
    values = np.abs(matrix[~np.eye(len(matrix), dtype=bool)])
    return max(float(values.max()), 1e-6) if values.size else 1.0


def label_for(tag: str, count: int, mapping: dict[str, tuple[str, str]]) -> str:
    group = mapping.get(tag)
    suffix = f" [{group[1]}]" if group else " [?]"
    return f"{tag} ({count}){suffix}"


def analyze(
    kind: str,
    tags: list[str],
    x: np.ndarray,
    tags_per_track: list[list[str]],
    counts: pd.Series,
    groups: dict[str, dict[str, list[str]]],
    mapping: dict[str, tuple[str, str]],
    out: Path,
    n_clusters: int,
) -> dict[str, float]:
    if len(tags) < 3:
        print(f"{kind}: меньше 3 тегов, пропускаю")
        return {}
    sim = pd.DataFrame(
        cosine_matrix(tag_centroids(x, tags_per_track, tags)), index=tags, columns=tags
    )
    order, clusters = cluster_order(sim.to_numpy(), n_clusters)
    labels = [label_for(t, int(counts[t]), mapping) for t in tags]

    plot_heatmap(
        sim,
        order,
        labels,
        f"{kind}: похожесть тегов по звучанию (косинус средних эмбеддингов)",
        out / f"{kind}_similarity.png",
        # Шкала по максимуму вне диагонали, иначе на реальных данных всё бледное.
        vmin=-off_diagonal_max(sim.to_numpy()),
        vmax=off_diagonal_max(sim.to_numpy()),
    )
    plot_dendrogram(
        sim.to_numpy(),
        labels,
        f"{kind}: кластеры тегов",
        out / f"{kind}_dendrogram.png",
    )
    jac = pd.DataFrame(jaccard_matrix(tags_per_track, tags), index=tags, columns=tags)
    plot_heatmap(
        jac,
        order,
        labels,
        f"{kind}: совместная встречаемость (Жаккар)",
        out / f"{kind}_cooccurrence.png",
        vmin=0,
        vmax=off_diagonal_max(jac.to_numpy()),
        cmap="viridis",
    )
    sim.round(4).to_csv(out / f"{kind}_similarity.csv")
    nearest_tags(sim).to_csv(out / f"{kind}_nearest_tags.csv", index=False)

    suggested = (
        pd.DataFrame(
            {"tag": tags, "cluster": clusters, "count": [int(counts[t]) for t in tags]}
        )
        .sort_values(["cluster", "count"], ascending=[True, False])
        .groupby("cluster")["tag"]
        .apply(lambda s: ", ".join(s))
    )
    suggested.to_csv(out / f"{kind}_suggested_clusters.csv")
    print(
        f"\n=== {kind}: {len(tags)} тегов, "
        f"предложение кластеризации ({n_clusters} кластеров)"
    )
    for cluster, members in suggested.items():
        print(f"  {cluster:>2}: {members}")

    metrics: dict[str, float] = {f"{kind}_n_tags": float(len(tags))}
    if groups.get(kind):
        cohesion = group_cohesion(sim, groups[kind])
        cohesion.to_csv(out / f"{kind}_draft_groups.csv", index=False)
        print(
            f"\n=== {kind}: черновик групп (margin < 0: группа похожа "
            "на соседнюю сильнее, чем сама на себя)"
        )
        with pd.option_context("display.width", 160, "display.max_colwidth", 60):
            print(cohesion.round(3).to_string(index=False))
        metrics[f"{kind}_draft_mean_within_sim"] = float(
            np.nanmean(cohesion["within_sim"])
        )
        metrics[f"{kind}_draft_mean_margin"] = float(np.nanmean(cohesion["margin"]))
    return metrics


def main() -> None:
    cfg = Config.from_env()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", default=cfg.dataset_uri)
    parser.add_argument("--groups", type=Path, default=HERE / "tag_groups.json")
    parser.add_argument(
        "--min-count", type=int, default=20, help="минимум треков с тегом в train"
    )
    parser.add_argument(
        "--n-clusters", type=int, default=10, help="для предложения кластеров"
    )
    parser.add_argument("--out", type=Path, default=Path(".cache/tag_similarity"))
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument("--refresh-data", action="store_true")
    args = parser.parse_args()

    args.data = require_dataset_uri(args.data)
    df, file_sha = load_dataset(args.data, cfg, refresh=args.refresh_data)
    train = df[df["split"] == "train"]
    x = features(train)
    tags_per_track = [split_tags(v) for v in train[SOURCE_COLUMN]]
    counts = tag_counts(train, SOURCE_COLUMN)
    frequent = sorted(counts[counts >= args.min_count].index)
    rare = sorted(counts[counts < args.min_count].index)

    groups = (
        load_groups(args.groups) if args.groups.exists() else {"mood": {}, "theme": {}}
    )
    mapping = tag_to_group(groups)
    themes = {t for t, (kind, _) in mapping.items() if kind == "theme"}
    by_kind = {
        "mood": [t for t in frequent if t not in themes],
        "theme": [t for t in frequent if t in themes],
        "all": frequent,
    }
    unassigned = [t for t in frequent if t not in mapping]
    missing = sorted(t for t in mapping if t not in counts.index)

    args.out.mkdir(parents=True, exist_ok=True)
    print(
        f"train: {len(train)} треков, тегов mood/theme: {len(counts)}, "
        f">= {args.min_count} треков: {len(frequent)}"
    )
    if rare:
        print(f"Редкие теги (не анализируются): {', '.join(rare)}")
    if unassigned:
        print(f"Нет в черновике групп: {', '.join(unassigned)}")
    if missing:
        print(f"Есть в черновике, но нет в данных: {', '.join(missing)}")

    metrics: dict[str, float] = {}
    for kind, tags in by_kind.items():
        metrics.update(
            analyze(
                kind,
                tags,
                x,
                tags_per_track,
                counts,
                groups,
                mapping,
                args.out,
                args.n_clusters,
            )
        )
    counts.rename("n_train_tracks").to_csv(args.out / "tag_counts_train.csv")

    if args.no_mlflow:
        print(f"\nГрафики и таблицы: {args.out}")
        return

    import mlflow
    from common import log_dataset_inputs

    mlflow.set_tracking_uri(cfg.tracking_uri)
    mlflow.set_experiment(cfg.eda_experiment)
    groups_sha = (
        hashlib.sha256(args.groups.read_bytes()).hexdigest()[:12]
        if args.groups.exists()
        else "none"
    )
    with mlflow.start_run(run_name="tag-similarity"):
        mlflow.set_tags({"stage": "eda", "analysis": "tag-similarity"})
        mlflow.log_params(
            {
                "data.uri": args.data,
                "data.version": data_version(args.data, file_sha),
                "data.file_sha256": file_sha,
                "min_count": args.min_count,
                "n_clusters": args.n_clusters,
                "groups.file": str(args.groups),
                "groups.sha256": groups_sha,
                "similarity": "cosine of mean standardized embeddings (train)",
            }
        )
        log_dataset_inputs(df, source=args.data, splits=("train",))
        mlflow.log_metrics(metrics)
        mlflow.log_artifacts(str(args.out), artifact_path="tag_similarity")
        if args.groups.exists():
            mlflow.log_artifact(str(args.groups), artifact_path="tag_similarity")
    print(
        f"\nГрафики и таблицы: {args.out} и MLflow (эксперимент {cfg.eda_experiment})"
    )


if __name__ == "__main__":
    main()
