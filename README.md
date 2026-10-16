# Music Mood Search

Асинхронный веб-сервис на FastAPI, разработанный в рамках курса MLOps.

В дальнейшем проект будет использоваться как основа для сервиса **Music Mood Search** — системы поиска музыкальных треков по текстовому описанию настроения, жанра и инструментов.

На текущем этапе реализована инфраструктурная часть проекта:

- асинхронное FastAPI-приложение;
- подключение к PostgreSQL;
- health-check эндпоинты;
- тестирование и измерение покрытия;
- статический анализ и форматирование;
- Docker и Docker Compose;
- pre-commit hooks;
- CI pipeline;
- CD pipeline;
- версионирование Docker-образов;
- публикация образов в GitHub Container Registry;
- структурированное JSON-логирование.

## ДЗ3: MLflow

Добавлен полный ML-цикл для классификатора тегов (genre / instrument / mood)
поверх эмбеддингов EffNet-Discogs:

- MLflow Tracking Server в Docker Compose: Backend Store в PostgreSQL
  (`mlflow-db`), Artifact Store в MinIO (S3);
- EDA, Dataset Tracking, эксперименты и Model Registry (`training/`);
- модель загружается из Registry по alias `champion` при старте приложения;
- `POST /api/v1/process` и `GET /api/v1/model`.

Быстрый старт:

```bash
cp .env.example .env
uv lock && uv sync --group training
docker compose up --build -d
uv run --env-file .env --group training python training/upload_dataset.py \
    --file training/data/out/embeddings.parquet \
    --manifest training/data/out/manifest.json
uv run --env-file .env --group training python training/eda.py
uv run --env-file .env --group training python training/train.py --model all
uv run --env-file .env --group training python training/registry.py promote
docker compose restart app
uv run --env-file .env --group training python training/demo_request.py
```

## Структура проекта

```text
music-mood-search/
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── cd.yml
│
├── docker/
│   └── mlflow/
│       └── Dockerfile          # образ MLflow Tracking Server
│
├── src/
│   └── music_mood_search/
│       ├── api/
│       │   ├── __init__.py
│       │   ├── health.py
│       │   └── process.py      # POST /api/v1/process, GET /api/v1/model
│       │
│       ├── services/
│       │   ├── __init__.py
│       │   ├── health.py
│       │   └── model.py        # загрузка модели из Model Registry
│       │
│       ├── __init__.py
│       ├── app.py
│       ├── config.py
│       ├── db.py
│       ├── logging_config.py
│       └── schemas.py
│
├── training/
│   ├── data/
│   │   ├── extract_embeddings.py   # эмбеддинги EffNet-Discogs из аудио
│   │   └── requirements.txt
│   ├── common.py                   # датасет, Dataset Tracking, метки
│   ├── demo_request.py             # демонстрационный запрос к /process
│   ├── eda.py                      # EDA, артефакты в MLflow
│   ├── music_tagger_pyfunc.py      # pyfunc-обёртка модели
│   ├── registry.py                 # версии, alias champion
│   ├── train.py                    # Runs, метрики, регистрация моделей
│   └── upload_dataset.py           # загрузка датасета в MinIO
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── fakes.py
│   ├── test_health.py
│   ├── test_model_service.py
│   └── test_process.py
│
├── .dockerignore
├── .env.example
├── .gitattributes
├── .gitignore
├── .pre-commit-config.yaml
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── uv.lock
└── README.md
```

## Используемые технологии

- Python 3.12
- FastAPI
- Uvicorn
- PostgreSQL 17
- asyncpg
- Pydantic Settings
- MLflow
- MinIO
- scikit-learn
- NumPy
- pandas
- Essentia
- uv
- pytest
- pytest-asyncio
- pytest-cov
- HTTPX
- Ruff
- pre-commit
- Docker
- Docker Compose
- GitHub Actions
- GitHub Container Registry

## API

В приложении реализованы пять эндпоинтов.

### `GET /healthz`

Быстрая проверка того, что само приложение работает.


Эндпоинт намеренно не находится внутри `/api/v1`, так как используется не пользователями API, а инфраструктурой: Docker, оркестраторами и системами мониторинга.

### `GET /api/v1/version`

Возвращает текущую версию приложения.


Версия не хранится отдельно внутри endpoint. Она получается из metadata установленного Python-пакета, а источником версии является `pyproject.toml`.

### `GET /api/v1/health`

Выполняет end-to-end health-check сторонних компонентов приложения.

На текущем этапе проверяются PostgreSQL и модель из MLflow Model Registry.

Эндпоинт возвращает:

- общий статус;
- состояние каждого компонента (`database`, `model`);
- версию компонента (для модели: имя, версия и alias);
- время ответа;
- информацию об ошибке при недоступности компонента.

Если PostgreSQL недоступен или модель не загружена, endpoint возвращает HTTP-код `503 Service Unavailable`.

### `GET /api/v1/model`

Показывает, какая модель обслуживает запросы: имя, alias, версию, `run_id` в MLflow и URI загруженной версии.

Если модель не загружена, возвращает `503 Service Unavailable`.

### `POST /api/v1/process`

Принимает до 64 эмбеддингов EffNet-Discogs (по 1280 чисел в каждом) и возвращает для каждого трека теги трёх категорий: `genre`, `instrument` и `mood`. У каждого тега есть оценка (`score`).

Пример запроса:

```json
{ "embeddings": [[0.01, 0.02, "... 1280 чисел ..."]] }
```

В ответе:

- `model` с информацией о версии модели, которая дала предсказание;
- `predictions` со списками тегов для каждого трека;
- `inference_ms`, время работы модели.


## Конфигурация

Настройки приложения загружаются через `pydantic-settings`.

Для локального запуска необходимо создать `.env` на основе `.env.example`.

При работе приложения внутри Docker Compose используется внутренний адрес PostgreSQL:

```text
postgresql://postgres:postgres@db:5432/postgres
http://mlflow:5000
```

Настройки модели:

| Переменная | Назначение | По умолчанию |
|---|---|---|
| `MLFLOW_TRACKING_URI` | адрес MLflow Tracking Server | `http://mlflow:5000` внутри Compose |
| `MODEL_NAME` | имя модели в Model Registry | `music-tagger` |
| `MODEL_ALIAS` | alias версии, которая обслуживает запросы | `champion` |



## Ruff

Для статического анализа и форматирования используется Ruff.

Проверка кода:

```bash
uv run ruff check .
```

Проверка форматирования:

```bash
uv run ruff format --check .
```

Автоматическое форматирование:

```bash
uv run ruff format .
```

В конфигурации Ruff используются правила для:

- стандартных ошибок Python;
- ошибок оформления;
- сортировки импортов;
- поиска распространённых потенциальных ошибок;
- использования современного синтаксиса Python.

## Тестирование

Запуск тестов:

```bash
uv run pytest -v
```

Запуск тестов с измерением покрытия:

```bash
uv run pytest --cov=music_mood_search --cov-report=term-missing
```

Проверяются следующие сценарии:

- успешный `/healthz`;
- успешный `/api/v1/version`;
- успешный `/api/v1/health`;
- ситуации, когда PostgreSQL недоступен или модель не загружена;
- загрузка модели: alias превращается в конкретную версию, повторные попытки при ошибке;
- `POST /api/v1/process`: успешный запрос, неверная размерность эмбеддинга, слишком большой пакет, ответ `503` без модели;
- `GET /api/v1/model`.

Минимально допустимое покрытие:

```text
80%
```

Если coverage становится ниже 80%, проверка завершается ошибкой.

## Pre-commit

Для автоматических проверок перед каждым коммитом используется `pre-commit`.

Установка Git hook:

```bash
uv run pre-commit install
```

Ручной запуск всех проверок:

```bash
uv run pre-commit run --all-files
```

В pre-commit настроены:

- удаление trailing whitespace;
- проверка конца файлов;
- проверка YAML;
- проверка TOML;
- проверка больших файлов;
- проверка merge conflicts;
- проверка конфликтов регистра имён файлов;
- нормализация line endings;
- Ruff lint;
- Ruff format.

При выполнении:

```bash
git commit
```

pre-commit запускается автоматически.

## Docker

Сборка Docker-образа:

```bash
docker build -t music-mood-search:local .
```

Dockerfile использует:

- `python:3.12-slim`;
- установку зависимостей через `uv`;
- фиксированные зависимости из `uv.lock`;
- отдельные Docker layers для зависимостей и исходного кода;
- только production-зависимости;
- непривилегированного пользователя `appuser`.

Проверить пользователя контейнера:

Приложение внутри контейнера запускается не от `root`.

## Docker Compose

Запустить приложение, PostgreSQL и MLflow с хранилищем:

```bash
docker compose up --build -d
```

Проверить состояние контейнеров:

```bash
docker compose ps
```

Все основные сервисы должны иметь статус `healthy`:

```text
app          healthy
db           healthy
mlflow       healthy
mlflow-db    healthy
minio        healthy
```

Сервис `minio-init` разовый: он создаёт бакеты `mlflow` и `datasets` и завершается со статусом `Exited (0)`.

Адреса с хостовой машины:

```text
API:           http://localhost:8000
MLflow UI:     http://localhost:5000
MinIO консоль: http://localhost:9001
PostgreSQL:    localhost:55432
```

Docker Compose содержит:

- сервис приложения;
- PostgreSQL приложения;
- MLflow Tracking Server и Model Registry;
- отдельный PostgreSQL для Backend Store MLflow;
- MinIO (S3) как Artifact Store;
- отдельную Docker network;
- persistent volumes для обеих баз и для MinIO;
- healthcheck каждого сервиса;
- ограничения памяти;
- ротацию логов;
- restart policy.


Просмотр логов приложения:

```bash
docker compose logs app
```

Просмотр логов в реальном времени:

```bash
docker compose logs -f app
```

Остановка сервисов:

```bash
docker compose down
```

## Логирование

В приложении реализовано структурированное JSON-логирование.


Логируются:

- запуск приложения;
- создание пула соединений с PostgreSQL;
- HTTP-метод;
- URL запроса;
- HTTP status code;
- время выполнения запроса;
- завершение приложения;
- закрытие connection pool;
- загрузка модели из MLflow (URI, версия, `run_id`, номер попытки);
- каждый вызов inference (версия модели, размер пакета, время).

## CI

Continuous Integration реализован через GitHub Actions.

CI запускается:

- при push в `main`;
- при pull request в `main`.

Pipeline выполняет:

```text
Checkout
   ↓
Установка uv и Python
   ↓
Установка зависимостей
   ↓
Ruff lint
   ↓
Ruff format check
   ↓
pytest
   ↓
coverage check
```

Pipeline завершится ошибкой, если:

- Ruff обнаружит ошибку;
- код неправильно отформатирован;
- хотя бы один тест не пройдёт;
- coverage станет ниже 80%.

## CD

Continuous Delivery также реализован через GitHub Actions.

CD запускается только при push Git-тега формата:

```text
v*.*.*
```

Перед публикацией Docker-образа pipeline сравнивает версию Git-тега с версией приложения из `pyproject.toml`.

Например:

```text
pyproject.toml: 0.1.1
Git tag:        v0.1.1
```

Если версии не совпадают, CD завершается ошибкой.

Такой подход предотвращает случайную публикацию образа с неправильной версией.

## GitHub Container Registry

Docker-образы публикуются в:

```text
ghcr.io/patrikkstar/music-mood-search
```

Для релиза `v0.1.1` создаются следующие теги:

```text
0.1.1
0.1
latest
sha-1dc2e5d
```

Назначение тегов:

- `0.1.1` — конкретная версия приложения;
- `0.1` — последняя patch-версия ветки `0.1`;
- `latest` — последний опубликованный релиз;
- `sha-*` — образ, соответствующий конкретному Git-коммиту.

Скачать конкретную версию:

```bash
docker pull ghcr.io/patrikkstar/music-mood-search:0.1.1
```

## Git workflow

Полный процесс разработки выглядит следующим образом:

```text
изменение кода
      ↓
git commit
      ↓
pre-commit
      ↓
git push
      ↓
CI
      ↓
создание version tag
      ↓
CD
      ↓
GitHub Container Registry
```

Обычный push в `main` запускает CI, но не создаёт новый Docker-релиз.

Для выпуска новой версии необходимо явно создать Git-тег:

```bash
git tag -a v0.1.1 -m "Release v0.1.1"
git push origin v0.1.1
```

После этого автоматически запускается CD и публикуется новый Docker-образ.

## Версии

Текущие релизы:

```text
v0.1.0 — первая рабочая инфраструктурная версия
v0.1.1 — добавлено структурированное логирование
```

## Развитие Music Mood Search

Текущая версия проекта содержит инфраструктурную основу будущего ML-сервиса.

Планируемая функциональность Music Mood Search:

1. Извлечение музыкальных эмбеддингов с помощью Essentia EffNet-Discogs.
2. Обучение лёгких классификаторов для тегов:
   - mood;
   - genre;
   - instrument.
3. Использование MTG-Jamendo Dataset для обучения классификатора.
4. Использование Song Describer Dataset для построения поисковой базы.
5. Объединение предсказанных музыкальных тегов с текстовыми описаниями треков.
6. Индексация треков в Chroma.
7. Поиск треков по текстовому описанию пользователя.
8. Возможность загрузки собственного трека и добавления его в поисковую коллекцию.
