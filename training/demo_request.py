"""Демонстрация POST /process на реальных треках из test-части датасета.

Берёт несколько треков, отправляет их эмбеддинги в сервис и печатает
версию модели, предсказанные и настоящие теги.

Пример:
    uv run --env-file .env --group training python training/demo_request.py --n 3
"""

from __future__ import annotations

import argparse
import json
import urllib.request

from common import (
    DEFAULT_GROUPS_FILE,
    EMBEDDING_COLUMNS,
    Config,
    load_dataset,
    load_tag_groups,
    require_dataset_uri,
    target_labels,
)


def main() -> None:
    cfg = Config.from_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000/api/v1/process")
    parser.add_argument("--data", default=cfg.dataset_uri)
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    df, _ = load_dataset(require_dataset_uri(args.data), cfg)
    sample = df[df["split"] == "test"].sample(args.n, random_state=args.seed)
    payload = {"embeddings": sample[EMBEDDING_COLUMNS].to_numpy().tolist()}

    request = urllib.request.Request(
        args.url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        result = json.loads(response.read())

    model = result["model"]
    print(f"Модель: {model['name']} v{model['version']} (@{model['alias']})")
    print(f"Inference: {result['inference_ms']} ms\n")
    # Настоящие метки: жанры и инструменты как есть, mood и theme по группам.
    groups = (
        load_tag_groups(DEFAULT_GROUPS_FILE) if DEFAULT_GROUPS_FILE.exists() else None
    )
    true = {
        c: target_labels(sample, c, groups)
        for c in ("genre", "instrument", "mood", "theme")
        if c in ("genre", "instrument", "mood") or groups
    }
    for i, ((_, row), pred) in enumerate(
        zip(sample.iterrows(), result["predictions"], strict=True)
    ):
        print(f"{row['track_id']}  (исходные теги mood/theme: {row['mood'] or '-'})")
        for category, tags in pred.items():
            predicted = ", ".join(t["tag"] for t in tags) or "-"
            expected = ", ".join(true[category].iloc[i]) if category in true else "?"
            print(f"  {category:<10} pred: {predicted}")
            print(f"  {'':<10} true: {expected or '-'}")
        print()


if __name__ == "__main__":
    main()
