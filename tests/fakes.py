from music_mood_search.services.model import ModelInfo, ModelNotLoadedError

EMBEDDING_CONTRACT = {"model": "discogs-effnet-bs64-1", "dim": 1280}

PREDICTION = {
    "genre": [{"tag": "rock", "score": 0.91}],
    "instrument": [{"tag": "guitar", "score": 0.77}],
    "mood": [{"tag": "energetic", "score": 0.66}, {"tag": "happy", "score": 0.51}],
    "theme": [{"tag": "cinematic", "score": 0.58}],
}


class FakeModelService:
    """Подменяет ModelService в тестах API, чтобы не ходить в MLflow."""

    def __init__(self, loaded: bool = True) -> None:
        self._loaded = loaded
        self.last_error = None if loaded else "registry unavailable"
        self.calls: list = []

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def info(self) -> ModelInfo:
        if not self._loaded:
            raise ModelNotLoadedError("Model is not loaded")
        return ModelInfo(
            name="music-tagger",
            alias="champion",
            version="3",
            run_id="run-123",
            uri="models:/music-tagger/3",
            loaded_at=0.0,
            embedding=EMBEDDING_CONTRACT,
        )

    def predict(self, embeddings) -> list[dict]:
        self.calls.append(embeddings)
        return [PREDICTION for _ in embeddings]
