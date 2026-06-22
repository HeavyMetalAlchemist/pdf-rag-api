from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

# Always resolves to the project root regardless of
# where the process is started from
ROOT_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    Pydantic validates all fields at startup — missing required
    variables cause an immediate failure with a clear error message.
    """

    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # AWS
    aws_region: str
    aws_access_key_id: str
    aws_secret_access_key: str
    aws_session_token: str | None = None  # needed for SSO profile

    # S3
    s3_bucket_name: str

    # S3 Vectors
    s3_vector_bucket_name: str
    s3_vector_index_name: str

    # Bedrock
    bedrock_embedding_model_id: str
    bedrock_generation_model_id: str

    # RAG
    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k: int = 5

    # OpenTelemetry
    otel_service_name: str
    otel_exporter_otlp_endpoint: str
    generation_max_tokens: int = 1024
    generation_temperature: float = 0.0


@lru_cache
def get_settings() -> Settings:
    """
    Returns a cached Settings instance.
    lru_cache ensures .env is only read once regardless of how many
    times get_settings() is called across the application.
    """
    return Settings()


# Module-level singleton for direct imports
settings = get_settings()
