"""Работа с Model Registry: список версий, выбор лучшей, переключение alias.

Примеры:
    uv run --group training python training/registry.py list
    uv run --group training python training/registry.py promote
    uv run --group training python training/registry.py set-alias --version 2
    uv run --group training python training/registry.py set-alias --version 1 \
        --alias champion

При переключении champion старая версия получает alias "previous",
чтобы откат делался одной командой.
"""

from __future__ import annotations

import argparse

import mlflow
from common import Config
from mlflow import MlflowClient

PRIMARY_METRIC = "val_pr_auc_macro"


def versions_with_metrics(client: MlflowClient, name: str) -> list[dict]:
    rows = []
    for mv in client.search_model_versions(f"name='{name}'"):
        run = client.get_run(mv.run_id) if mv.run_id else None
        full = client.get_model_version(name, mv.version)
        version_tags = dict(getattr(full, "tags", {}) or {})
        # Метрики берём из Run; если связи с Run нет, из тегов версии,
        # которые проставляет train.py.
        metrics = dict(run.data.metrics) if run else {}
        for key, value in version_tags.items():
            try:
                metrics.setdefault(key, float(value))
            except ValueError:
                continue
        rows.append(
            {
                "version": int(mv.version),
                "aliases": list(getattr(full, "aliases", []) or []),
                "model_type": version_tags.get("model_type")
                or (run.data.tags.get("model_type") if run else None),
                "run_id": mv.run_id,
                "val_pr_auc_macro": metrics.get("val_pr_auc_macro"),
                "test_pr_auc_macro": metrics.get("test_pr_auc_macro"),
                "test_roc_auc_macro": metrics.get("test_roc_auc_macro"),
                "metrics": metrics,
            }
        )
    return sorted(rows, key=lambda r: r["version"])


def print_table(rows: list[dict]) -> None:
    def fmt(value):
        return f"{value:.4f}" if isinstance(value, float) else str(value)

    header = (
        "version",
        "aliases",
        "model_type",
        "val_pr_auc_macro",
        "test_pr_auc_macro",
        "test_roc_auc_macro",
    )
    print(" | ".join(header))
    for row in rows:
        values = [row[h] if h != "aliases" else ",".join(row[h]) for h in header]
        print(" | ".join(fmt(v) for v in values))


def set_alias(client: MlflowClient, name: str, alias: str, version: int) -> None:
    if alias == "champion":
        try:
            current = client.get_model_version_by_alias(name, alias)
        except Exception:  # noqa: BLE001
            current = None
        if current is not None and int(current.version) != version:
            client.set_registered_model_alias(name, "previous", current.version)
            print(f"previous -> v{current.version}")
    client.set_registered_model_alias(name, alias, str(version))
    print(f"{alias} -> v{version}")


def main() -> None:
    cfg = Config.from_env()
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="версии модели с метриками и alias")

    promote = sub.add_parser("promote", help="назначить champion лучшей версии")
    promote.add_argument("--metric", default=PRIMARY_METRIC)
    promote.add_argument("--alias", default="champion")

    set_cmd = sub.add_parser("set-alias", help="назначить alias конкретной версии")
    set_cmd.add_argument("--version", type=int, required=True)
    set_cmd.add_argument("--alias", default="champion")

    args = parser.parse_args()

    mlflow.set_tracking_uri(cfg.tracking_uri)
    client = MlflowClient(tracking_uri=cfg.tracking_uri)
    name = cfg.registered_model

    if args.command == "list":
        print_table(versions_with_metrics(client, name))
    elif args.command == "promote":
        rows = [
            r
            for r in versions_with_metrics(client, name)
            if r["metrics"].get(args.metric) is not None
        ]
        if not rows:
            raise SystemExit(f"Нет версий с метрикой {args.metric}")
        best = max(rows, key=lambda r: r["metrics"][args.metric])
        print(
            f"Лучшая версия по {args.metric}: v{best['version']} ({best['model_type']})"
        )
        set_alias(client, name, args.alias, best["version"])
    elif args.command == "set-alias":
        set_alias(client, name, args.alias, args.version)


if __name__ == "__main__":
    main()
