from pydantic import BaseModel


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
