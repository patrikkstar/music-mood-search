"""EDA датасета эмбеддингов. Все графики и выводы сохраняются в MLflow.

Пример:
    uv run --group training python training/eda.py
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import matplotlib
import mlflow
import numpy as np
import pandas as pd
from common import (
    CATEGORIES,
    Config,
    data_version,
    features,
    load_dataset,
    log_dataset_inputs,
    require_dataset_uri,
    split_tags,
    tag_counts,
)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SPLITS = ("train", "validation", "test")


def save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_top_tags(df: pd.DataFrame, out: Path, top: int = 25) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    for ax, category in zip(axes, CATEGORIES, strict=True):
        counts = tag_counts(df, category).head(top)
        ax.barh(counts.index[::-1], counts.values[::-1])
        ax.set_title(f"{category}: top-{top} тегов")
        ax.set_xlabel("Число треков")
    save(fig, out)


def plot_long_tail(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for category in CATEGORIES:
        counts = tag_counts(df, category).to_numpy()
        ax.plot(np.arange(1, len(counts) + 1), counts, marker=".", label=category)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Ранг тега (log)")
    ax.set_ylabel("Число треков (log)")
    ax.set_title("Длинный хвост распределения тегов")
    ax.legend()
    save(fig, out)


def plot_tags_per_track(df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, category in zip(axes, CATEGORIES, strict=True):
        n = df[category].map(lambda v: len(split_tags(v)))
        ax.hist(n, bins=range(0, int(n.max()) + 2), align="left", rwidth=0.85)
        ax.set_title(f"{category}: тегов на трек")
        ax.set_xlabel("Число тегов")
    save(fig, out)


def plot_cooccurrence(df: pd.DataFrame, out: Path, top: int = 15) -> None:
    moods = tag_counts(df, "mood").head(top).index
    genres = tag_counts(df, "genre").head(top).index
    mood_sets = df["mood"].map(lambda v: set(split_tags(v)))
    genre_sets = df["genre"].map(lambda v: set(split_tags(v)))
    matrix = np.zeros((len(moods), len(genres)))
    for i, mood in enumerate(moods):
        has_mood = mood_sets.map(lambda s, m=mood: m in s)
        subset = genre_sets[has_mood]
        for j, genre in enumerate(genres):
            matrix[i, j] = subset.map(lambda s, g=genre: g in s).mean()
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=max(matrix.max(), 1e-6))
    ax.set_xticks(range(len(genres)), genres, rotation=60, ha="right")
    ax.set_yticks(range(len(moods)), moods)
    ax.set_title("P(жанр | настроение) для частых тегов")
    fig.colorbar(im, ax=ax)
    save(fig, out)


def plot_split_balance(df: pd.DataFrame, out: Path, top: int = 20) -> pd.DataFrame:
    rows = []
    for category in CATEGORIES:
        for tag in (
            tag_counts(df, category).head(top // 2 if category != "mood" else top).index
        ):
            row = {"label": f"{category}:{tag}"}
            for split in SPLITS:
                part = df[df["split"] == split]
                row[split] = (
                    part[category].map(lambda v, t=tag: t in split_tags(v)).mean()
                )
            rows.append(row)
    table = pd.DataFrame(rows).set_index("label")
    fig, ax = plt.subplots(figsize=(10, 0.3 * len(table) + 2))
    table.plot.barh(ax=ax)
    ax.invert_yaxis()
    ax.set_xlabel("Доля треков с тегом")
    ax.set_title("Баланс частых тегов между train / validation / test")
    save(fig, out)
    return table


def plot_pca(df: pd.DataFrame, out_scatter: Path, out_var: Path) -> dict[str, float]:
    x = StandardScaler().fit_transform(features(df))
    n_components = min(256, x.shape[0], x.shape[1])
    pca = PCA(n_components=n_components, random_state=42).fit(x)
    cumulative = np.cumsum(pca.explained_variance_ratio_)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(1, len(cumulative) + 1), cumulative)
    ax.axhline(0.9, color="grey", linestyle="--")
    ax.set_xlabel("Число компонент")
    ax.set_ylabel("Объяснённая дисперсия")
    ax.set_title("PCA эмбеддингов: накопленная дисперсия")
    save(fig, out_var)

    coords = pca.transform(x)[:, :2]
    top_genres = list(tag_counts(df, "genre").head(6).index)
    main_genre = df["genre"].map(
        lambda v: next((g for g in top_genres if g in split_tags(v)), "other")
    )
    fig, ax = plt.subplots(figsize=(8, 7))
    for genre in [*top_genres, "other"]:
        mask = (main_genre == genre).to_numpy()
        ax.scatter(coords[mask, 0], coords[mask, 1], s=6, alpha=0.6, label=genre)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Эмбеддинги в проекции PCA, цвет = жанр")
    ax.legend(markerscale=3)
    save(fig, out_scatter)

    # Если 90% не набирается за n_components, метрика = -1.
    reached = np.flatnonzero(cumulative >= 0.9)
    return {
        "pca_components_for_90pct": float(reached[0] + 1) if reached.size else -1.0,
        f"pca_variance_{len(cumulative)}_components": float(cumulative[-1]),
        "pca_variance_top2": float(cumulative[1]),
    }


def plot_duration(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df["duration"] / 60, bins=50)
    ax.set_xlabel("Длительность, мин")
    ax.set_ylabel("Число треков")
    ax.set_title("Длительность треков")
    save(fig, out)


def main() -> None:
    cfg = Config.from_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=cfg.dataset_uri)
    parser.add_argument("--refresh-data", action="store_true")
    args = parser.parse_args()

    mlflow.set_tracking_uri(cfg.tracking_uri)
    mlflow.set_experiment(cfg.eda_experiment)
    args.data = require_dataset_uri(args.data)
    df, file_sha = load_dataset(args.data, cfg, refresh=args.refresh_data)

    artists = {s: set(df.loc[df["split"] == s, "artist_id"]) for s in SPLITS}
    summary: dict[str, float] = {
        "n_tracks": len(df),
        **{f"n_{s}": int((df["split"] == s).sum()) for s in SPLITS},
        "n_artists": df["artist_id"].nunique(),
        "n_albums": df["album_id"].nunique(),
        "artist_overlap_train_test": len(artists["train"] & artists["test"]),
        "artist_overlap_train_validation": len(
            artists["train"] & artists["validation"]
        ),
        "duration_median_min": float(df["duration"].median() / 60),
    }
    for category in CATEGORIES:
        n_tags = df[category].map(lambda v: len(split_tags(v)))
        counts = tag_counts(df, category)
        summary[f"{category}_n_unique_tags"] = len(counts)
        summary[f"{category}_tags_per_track_mean"] = float(n_tags.mean())
        summary[f"{category}_share_tracks_without_tag"] = float((n_tags == 0).mean())
        summary[f"{category}_tags_with_lt_20_tracks"] = int((counts < 20).sum())
        summary[f"{category}_top_tag_share"] = float(counts.iloc[0] / len(df))

    with mlflow.start_run(run_name="eda"), tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        mlflow.set_tags({"stage": "eda"})
        mlflow.log_params(
            {
                "data.uri": args.data,
                "data.version": data_version(args.data, file_sha),
                "data.file_sha256": file_sha,
            }
        )
        log_dataset_inputs(df, source=args.data, splits=SPLITS)

        plot_top_tags(df, tmp_dir / "01_top_tags.png")
        plot_long_tail(df, tmp_dir / "02_long_tail.png")
        plot_tags_per_track(df, tmp_dir / "03_tags_per_track.png")
        plot_cooccurrence(df, tmp_dir / "04_mood_genre_cooccurrence.png")
        balance = plot_split_balance(df, tmp_dir / "05_split_balance.png")
        summary.update(
            plot_pca(df, tmp_dir / "06_pca_genres.png", tmp_dir / "07_pca_variance.png")
        )
        plot_duration(df, tmp_dir / "08_duration.png")

        balance.to_csv(tmp_dir / "split_balance.csv")
        for category in CATEGORIES:
            tag_counts(df, category).rename("n_tracks").to_csv(
                tmp_dir / f"tag_counts_{category}.csv"
            )
        (tmp_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2)
        )

        mlflow.log_metrics({k: float(v) for k, v in summary.items()})
        mlflow.log_artifacts(str(tmp_dir), artifact_path="eda")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
