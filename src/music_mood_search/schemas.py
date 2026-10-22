from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str


class VersionResponse(BaseModel):
    version: str


class ComponentHealth(BaseModel):
    status: str
    version: str
    response_time_ms: float
    error: str | None = None


class DetailedHealthResponse(BaseModel):
    status: str
    components: dict[str, ComponentHealth]


class ProcessRequest(BaseModel):
    embeddings: list[list[float]] = Field(
        ...,
        min_length=1,
        description="Эмбеддинги EffNet-Discogs (1280 чисел на трек).",
    )


class TagScore(BaseModel):
    tag: str
    score: float


class TrackPrediction(BaseModel):
    genre: list[TagScore]
    instrument: list[TagScore]
    mood: list[TagScore]


class ModelInfoResponse(BaseModel):
    name: str
    alias: str
    version: str
    run_id: str | None
    uri: str


class ProcessResponse(BaseModel):
    model: ModelInfoResponse
    predictions: list[TrackPrediction]
    inference_ms: float
