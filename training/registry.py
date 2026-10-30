"""Работа с Model Registry: список версий, выбор лучшей, переключение alias.

Примеры (из корня репозитория, перед каждой командой
"uv run --env-file .env --group training"):
    python training/registry.py list
    python training/registry.py promote
    python training/registry.py promote --alias candidate
    python training/registry.py set-alias --version 7

promote сравнивает только версии с одинаковой постановкой задачи: тот же
датасет (sha256 файла) и те же группы тегов (sha256 tag_groups.json). Метрики
на разных validation-выборках или разных наборах меток несравнимы.
Основная метрика: val_pr_auc_overall (взвешенное среднее PR-AUC по категориям).
Новая версия не проходит, если хотя бы одна категория хуже, чем у текущего
champion с той же постановкой, больше чем на --tolerance.

При переключении champion старая версия получает alias "previous",
чтобы откат делался одной командой.
"""

from __future__ import annotations

import argparse

import mlflow
from common import Config
from mlflow import MlflowClient

CATEGORIES = ("mood", "theme", "genre", "instrument")
PRIMARY_METRIC = "val_pr_auc_overall"
GUARD_METRICS = tuple(f"val_{c}_pr_auc_macro" for c in CATEGORIES)
SHOWN_METRICS = (
    "val_pr_auc_overall",
    "test_pr_auc_overall",
    *GUARD_METRICS,
)
SHORT_NAMES = {
    "val_pr_auc_overall": "val_overall",
    "test_pr_auc_overall": "test_overall",
    **{f"val_{c}_pr_auc_macro": f"val_{c}" for c in CATEGORIES},
}


def versions_with_metrics(client: MlflowClient, name: str) -> list[dict]:
    rows = []
    for mv in client.search_model_versions(f"name='{name}'"):
        full = client.get_model_version(name, mv.version)
        run = client.get_run(mv.run_id) if mv.run_id else None
        version_tags = dict(getattr(full, "tags", {}) or {})
        params = dict(run.data.params) if run else {}
        metrics = dict(run.data.metrics) if run else {}
        # Если связи с Run нет, берём метрики из тегов версии (их ставит train.py).
        for key, value in version_tags.items():
            try:
                metrics.setdefault(key, float(value))
            except ValueError:
                continue
        metrics = {k: v for k, v in metrics.items() if v == v}  # без NaN
        data_sha = version_tags.get("data_sha256") or params.get("data.file_sha256")
        groups_sha = (
            version_tags.get("label_groups_sha256")
            or params.get("labels.groups_sha256")
            or "none"
        )
        rows.append(
            {
                "version": int(mv.version),
                "aliases": list(getattr(full, "aliases", []) or []),
                "data": version_tags.get("data_version")
                or params.get("data.version")
                or (data_sha[:8] if data_sha else "-"),
                "data_sha256": data_sha,
                "groups_sha256": groups_sha,
                "families": "/".join(
                    version_tags.get(f"family.{c}", "-") for c in CATEGORIES
                ),
                "metrics": metrics,
            }
        )
    return sorted(rows, key=lambda r: r["version"])


def print_table(rows: list[dict]) -> None:
    header = [
        "version",
        "aliases",
        "data",
        "groups",
        "mood/theme/genre/instr",
        *(SHORT_NAMES[m] for m in SHOWN_METRICS),
    ]
    print(" | ".join(header))
    for row in rows:
        values = [
            str(row["version"]),
            ",".join(row["aliases"]) or "-",
            str(row["data"]),
            row["groups_sha256"][:8],
            row["families"],
        ]
        for metric in SHOWN_METRICS:
            value = row["metrics"].get(metric)
            values.append("-" if value is None else f"{value:.4f}")
        print(" | ".join(values))


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


def setup(row: dict) -> tuple[str | None, str]:
    """Постановка задачи версии: (датасет, группы тегов)."""
    return row["data_sha256"], row.get("groups_sha256", "none")


def pick_best(
    rows: list[dict],
    metric: str,
    data_sha: str | None,
    champion: dict | None,
    tolerance: float,
) -> tuple[dict | None, str]:
    """Лучшая версия с той же постановкой, что у самой новой версии
    (или у версий на датасете data_sha). Возвращает (версия, пояснение)."""
    pool = [r for r in rows if r["data_sha256"]]
    if data_sha is not None:
        pool = [r for r in pool if r["data_sha256"] == data_sha]
    latest = max(pool, key=lambda r: r["version"], default=None)
    if latest is None:
        return None, "нет версий с информацией о датасете"
    target = setup(latest)

    comparable = [
        r for r in rows if setup(r) == target and r["metrics"].get(metric) is not None
    ]
    if not comparable:
        return None, (
            f"нет версий с метрикой {metric} для датасета {target[0][:8]} "
            f"и групп {target[1][:8]}"
        )
    best = max(comparable, key=lambda r: r["metrics"][metric])

    if champion is None or champion["version"] == best["version"]:
        return best, "текущего champion нет или он уже лучший"
    if setup(champion) != target:
        return best, (
            f"champion v{champion['version']} обучен в другой постановке "
            "(другие данные или группы тегов), проверка по категориям пропущена"
        )
    for guard in GUARD_METRICS:
        new, old = best["metrics"].get(guard), champion["metrics"].get(guard)
        if new is not None and old is not None and new < old - tolerance:
            return None, (
                f"v{best['version']} хуже champion v{champion['version']} по {guard}: "
                f"{new:.4f} < {old:.4f}"
            )
    return best, "проверка по категориям пройдена"


def main() -> None:
    cfg = Config.from_env()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="версии модели с метриками и alias")

    promote = sub.add_parser("promote", help="назначить alias лучшей версии")
    promote.add_argument("--metric", default=PRIMARY_METRIC)
    promote.add_argument("--alias", default="champion")
    promote.add_argument(
        "--data-sha",
        default=None,
        help="sha256 датасета (по умолчанию датасет самой новой версии)",
    )
    promote.add_argument(
        "--tolerance",
        type=float,
        default=0.0,
        help="допустимое ухудшение любой категории относительно champion",
    )

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
        rows = versions_with_metrics(client, name)
        champion = next((r for r in rows if "champion" in r["aliases"]), None)
        best, reason = pick_best(
            rows, args.metric, args.data_sha, champion, args.tolerance
        )
        print(reason)
        if best is None:
            raise SystemExit("Alias не изменён")
        print(
            f"Лучшая версия по {args.metric}: v{best['version']} "
            f"({best['families']}, data {best['data']})"
        )
        set_alias(client, name, args.alias, best["version"])
    elif args.command == "set-alias":
        set_alias(client, name, args.alias, args.version)


if __name__ == "__main__":
    main()
