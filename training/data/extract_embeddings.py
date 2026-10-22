"""Извлечение эмбеддингов EffNet-Discogs для подвыборки MTG-Jamendo (mood/theme).

Схема работы по частям (архив за архивом):
    скачать .tar -> проверить sha256 -> распаковать только нужные треки
    -> посчитать эмбеддинги -> сохранить parts/part_XX.parquet -> удалить аудио.

Пока считается текущая часть, следующая скачивается в фоне.
Уже готовые части пропускаются, поэтому скрипт можно перезапускать.

Источник треков и тегов: официальный split-0 (data/splits/split-0/autotagging-*.tsv),
только треки с хотя бы одним mood/theme-тегом. Аудио: autotagging_moodtheme, audio-low.

Пример:
    python extract_embeddings.py \
        --repo-dir ./mtg-jamendo-dataset \
        --exclude-csv ./song_describer.csv \
        --parts 0-29 \
        --out-dir ./out

    python extract_embeddings.py \
        --repo-dir ./mtg-jamendo-dataset --out-dir ./out --merge-only
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import shutil
import tarfile
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

DATASET = "autotagging_moodtheme"
DATA_TYPE = "audio-low"
MIRRORS = {
    "mtg-fast": "https://cdn.freesound.org/mtg-jamendo",
    "mtg": "https://essentia.upf.edu/documentation/datasets/mtg-jamendo",
}
MODEL_NAME = "discogs-effnet-bs64-1"
MODEL_URL = (
    f"https://essentia.upf.edu/models/feature-extractors/discogs-effnet/{MODEL_NAME}.pb"
)
SAMPLE_RATE = 16000
EMB_DIM = 1280
CATEGORIES = {"genre": "genre", "instrument": "instrument", "mood/theme": "mood"}
CHUNK = 1024 * 1024

log = logging.getLogger("extract")


# ---------------------------------------------------------------- metadata


def read_split_tsv(path: Path, split: str) -> list[dict]:
    """Читает tsv MTG-Jamendo: 5 полей метаданных, дальше переменное число тегов."""
    rows = []
    with path.open() as fp:
        reader = csv.reader(fp, delimiter="\t")
        next(reader)
        for r in reader:
            tags = {v: [] for v in CATEGORIES.values()}
            for t in r[5:]:
                cat, tag = t.split("---", 1)
                tags[CATEGORIES[cat]].append(tag)
            rows.append(
                {
                    "track_id": r[0],
                    "artist_id": r[1],
                    "album_id": r[2],
                    "path": r[3],
                    "duration": float(r[4]),
                    "split": split,
                    **{k: "|".join(sorted(v)) for k, v in tags.items()},
                }
            )
    return rows


def load_exclude_ids(path: Path | None) -> set[str]:
    """ID треков Song Describer (колонка track_id: 1234 или track_0001234)."""
    if path is None:
        return set()
    df = pd.read_csv(path)
    col = next(c for c in df.columns if c.lower() == "track_id")
    ids = set()
    for v in df[col].dropna():
        s = str(v).strip()
        num = int(s.split("_")[-1]) if "_" in s else int(float(s))
        ids.add(f"track_{num:07d}")
    return ids


def build_metadata(repo_dir: Path, exclude: set[str]) -> pd.DataFrame:
    split_dir = repo_dir / "data" / "splits" / "split-0"
    rows = []
    for split in ("train", "validation", "test"):
        rows += read_split_tsv(split_dir / f"autotagging-{split}.tsv", split)
    df = pd.DataFrame(rows)
    n_all = len(df)
    df = df[df["mood"] != ""]
    n_mood = len(df)
    df = df[~df["track_id"].isin(exclude)]
    log.info(
        "split-0: %d треков, с mood: %d, после исключения Song Describer: %d "
        "(исключено %d)",
        n_all,
        n_mood,
        len(df),
        n_mood - len(df),
    )
    # имя файла внутри архива audio-low: 14/214.mp3 -> 14/214.low.mp3
    df["member"] = df["path"].str.replace(".mp3", ".low.mp3", regex=False)
    df["part"] = df["path"].str.split("/").str[0].astype(int)
    return df.reset_index(drop=True)


def read_sha256(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text().splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            out[name.strip()] = digest
    return out


# ---------------------------------------------------------------- download


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path, expected_sha: str | None, retries: int = 3) -> Path:
    if dest.exists() and expected_sha and sha256_file(dest) == expected_sha:
        return dest
    for attempt in range(1, retries + 1):
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        try:
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with tmp.open("wb") as f:
                    for chunk in r.iter_content(CHUNK):
                        f.write(chunk)
            if expected_sha and sha256_file(tmp) != expected_sha:
                raise ValueError(f"sha256 не совпал: {dest.name}")
            tmp.replace(dest)
            return dest
        except Exception as e:  # noqa: BLE001
            tmp.unlink(missing_ok=True)
            log.warning(
                "Загрузка %s, попытка %d/%d: %s", dest.name, attempt, retries, e
            )
            time.sleep(5 * attempt)
    raise RuntimeError(f"Не удалось скачать {url}")


# ---------------------------------------------------------------- embeddings


class Embedder:
    """EffNet-Discogs: аудио 16 кГц моно -> (n_patches, 1280) -> среднее (1280,).

    Эта же функция должна использоваться в сервисе для загруженных треков.
    """

    def __init__(self, model_path: Path, max_seconds: float | None):
        from essentia.standard import MonoLoader, TensorflowPredictEffnetDiscogs

        self._loader_cls = MonoLoader
        self.model = TensorflowPredictEffnetDiscogs(
            graphFilename=str(model_path), output="PartitionedCall:1"
        )
        self.max_seconds = max_seconds

    def __call__(self, path: Path) -> np.ndarray:
        audio = self._loader_cls(
            filename=str(path), sampleRate=SAMPLE_RATE, resampleQuality=4
        )()
        audio = center_crop(audio, self.max_seconds)
        emb = self.model(audio)
        return np.asarray(emb, dtype=np.float32).mean(axis=0)


def center_crop(audio: np.ndarray, max_seconds: float | None) -> np.ndarray:
    if not max_seconds:
        return audio
    n = int(max_seconds * SAMPLE_RATE)
    if len(audio) <= n:
        return audio
    start = (len(audio) - n) // 2
    return audio[start : start + n]


def ensure_model(model_path: Path) -> Path:
    if not model_path.exists():
        log.info("Скачиваю модель %s", MODEL_URL)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        download(MODEL_URL, model_path, expected_sha=None)
    return model_path


# ---------------------------------------------------------------- per part


def process_part(
    part: int,
    tar_path: Path,
    meta_part: pd.DataFrame,
    track_sha: dict[str, str],
    embed,
    work_dir: Path,
) -> tuple[pd.DataFrame, list[dict]]:
    extract_dir = work_dir / f"extract_{part:02d}"
    wanted = set(meta_part["member"])
    with tarfile.open(tar_path) as tar:
        members = [m for m in tar.getmembers() if m.isfile() and m.name in wanted]
        try:
            tar.extractall(extract_dir, members=members, filter="data")
        except TypeError:  # Python < 3.11.4 без параметра filter
            tar.extractall(extract_dir, members=members)

    records, failed = [], []
    for row in meta_part.itertuples(index=False):
        f = extract_dir / row.member
        try:
            if not f.exists():
                raise FileNotFoundError("нет в архиве")
            if row.member in track_sha and sha256_file(f) != track_sha[row.member]:
                raise ValueError("sha256 трека не совпал")
            t0 = time.perf_counter()
            emb = embed(f)
            rec = row._asdict()
            rec["embed_seconds"] = round(time.perf_counter() - t0, 3)
            rec.update({f"e{i}": float(v) for i, v in enumerate(emb)})
            records.append(rec)
        except Exception as e:  # noqa: BLE001
            failed.append({"track_id": row.track_id, "part": part, "error": str(e)})
            log.warning("%s: %s", row.track_id, e)

    shutil.rmtree(extract_dir, ignore_errors=True)
    df = pd.DataFrame(records)
    if not df.empty:
        emb_cols = [f"e{i}" for i in range(EMB_DIM)]
        df[emb_cols] = df[emb_cols].astype(np.float32)
    return df, failed


def parse_parts(spec: str) -> list[int]:
    parts: list[int] = []
    for chunk in spec.split(","):
        if "-" in chunk:
            a, b = chunk.split("-")
            parts += list(range(int(a), int(b) + 1))
        else:
            parts.append(int(chunk))
    return sorted(set(parts))


def merge(out_dir: Path, manifest: dict) -> Path:
    files = sorted((out_dir / "parts").glob("part_*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df.sort_values("track_id").reset_index(drop=True)
    out = out_dir / "embeddings.parquet"
    df.to_parquet(out, index=False)
    manifest.update(
        {
            "n_tracks": len(df),
            "parts_merged": [int(f.stem.split("_")[1]) for f in files],
            "split_counts": df["split"].value_counts().to_dict(),
            "file_sha256": sha256_file(out),
        }
    )
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2)
    )
    log.info("Итог: %s, %d треков, %s", out, len(df), manifest["split_counts"])
    return out


# ---------------------------------------------------------------- main


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--repo-dir", type=Path, required=True, help="клон MTG/mtg-jamendo-dataset"
    )
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="куда качать архивы (по умолчанию out-dir/tmp)",
    )
    p.add_argument(
        "--parts", default="0-29", help="номера архивов 0..99, например 0-29 или 0,5,7"
    )
    p.add_argument(
        "--exclude-csv",
        type=Path,
        default=None,
        help="csv Song Describer с колонкой track_id",
    )
    p.add_argument(
        "--model-path", type=Path, default=Path("models") / f"{MODEL_NAME}.pb"
    )
    p.add_argument(
        "--max-seconds",
        type=float,
        default=0,
        help="центральный кусок трека, 0 = весь трек",
    )
    p.add_argument("--mirror", choices=list(MIRRORS), default="mtg-fast")
    p.add_argument(
        "--keep-tars", action="store_true", help="не удалять архивы после обработки"
    )
    p.add_argument(
        "--merge-only", action="store_true", help="только склеить готовые части"
    )
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    out_dir = args.out_dir
    parts_dir = out_dir / "parts"
    work_dir = args.work_dir or out_dir / "tmp"
    for d in (parts_dir, work_dir):
        d.mkdir(parents=True, exist_ok=True)

    exclude = load_exclude_ids(args.exclude_csv)
    if not exclude:
        log.warning("Список Song Describer не задан: треки не исключаются")

    manifest = {
        "source": f"MTG-Jamendo {DATASET}/{DATA_TYPE}, split-0 autotagging-*.tsv",
        "mirror": MIRRORS[args.mirror],
        "model": MODEL_NAME,
        "sample_rate": SAMPLE_RATE,
        "pooling": "mean over patches",
        "max_seconds": args.max_seconds or None,
        "excluded_song_describer": len(exclude),
    }

    if args.merge_only:
        merge(out_dir, manifest)
        return

    meta = build_metadata(args.repo_dir, exclude)
    dl_dir = args.repo_dir / "data" / "download"
    tar_sha = read_sha256(dl_dir / f"{DATASET}_{DATA_TYPE}_sha256_tars.txt")
    track_sha = read_sha256(dl_dir / f"{DATASET}_{DATA_TYPE}_sha256_tracks.txt")

    parts = [
        n
        for n in parse_parts(args.parts)
        if not (parts_dir / f"part_{n:02d}.parquet").exists()
    ]
    log.info("К обработке частей: %d %s", len(parts), parts)
    if not parts:
        merge(out_dir, manifest)
        return

    embed = Embedder(ensure_model(args.model_path), args.max_seconds)
    manifest["model_sha256"] = sha256_file(args.model_path)

    def fetch(n: int) -> Path:
        name = f"{DATASET}_{DATA_TYPE}-{n:02d}.tar"
        url = f"{MIRRORS[args.mirror]}/{DATASET}/{DATA_TYPE}/{name}"
        return download(url, work_dir / name, tar_sha.get(name))

    failed_all: list[dict] = []
    with ThreadPoolExecutor(max_workers=1) as pool:
        future: Future = pool.submit(fetch, parts[0])
        for i, n in enumerate(parts):
            tar_path = future.result()
            if i + 1 < len(parts):
                future = pool.submit(fetch, parts[i + 1])  # prefetch следующей части

            t0 = time.perf_counter()
            meta_part = meta[meta["part"] == n]
            df, failed = process_part(
                n, tar_path, meta_part, track_sha, embed, work_dir
            )
            failed_all += failed
            df.to_parquet(parts_dir / f"part_{n:02d}.parquet", index=False)
            if not args.keep_tars:
                tar_path.unlink(missing_ok=True)
            log.info(
                "Часть %02d: %d треков, ошибок %d, %.0f c",
                n,
                len(df),
                len(failed),
                time.perf_counter() - t0,
            )

    if failed_all:
        failed_path = out_dir / "failed.csv"
        pd.DataFrame(failed_all).to_csv(
            failed_path, mode="a", index=False, header=not failed_path.exists()
        )
    merge(out_dir, manifest)


if __name__ == "__main__":
    main()
