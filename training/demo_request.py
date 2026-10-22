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

from common import CATEGORIES, EMBEDDING_COLUMNS, Config, load_dataset


def main() -> None:
    cfg = Config.from_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000/api/v1/process")
    parser.add_argument("--data", default=cfg.dataset_uri)
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    df, _ = load_dataset(args.data, cfg)
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
    for (_, row), pred in zip(sample.iterrows(), result["predictions"], strict=True):
        print(row["track_id"])
        for category in CATEGORIES:
            predicted = ", ".join(t["tag"] for t in pred[category])
            print(f"  {category:<10} pred: {predicted}")
            print(f"  {'':<10} true: {row[category].replace('|', ', ') or '-'}")
        print()


if __name__ == "__main__":
    main()
