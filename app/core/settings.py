from functools import lru_cache
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Industrial RAG API"

    openai_api_key: SecretStr
    openai_embedding_model: str = "text-embedding-3-small"
    openai_llm_model: str = "gpt-5.4-mini"

    retrieval_top_k: int = Field(default=5, ge=1, le=50)
    retrieval_score_threshold: float = Field(default=0.0, ge=0.0)

    vector_db_host: str = "localhost"
    vector_db_port: int = Field(default=6333, ge=1, le=65535)
    vector_db_collection_name: str = "my_collection"

    # NOTE:
    # In local development, settings can be loaded from .env.
    # In Docker/Compose, environment variables are expected
    # to be injected at container runtime.
    model_config = SettingsConfigDict(
        # Optional local source for development.
        # The application can still work without this file
        # if required variables are provided by the process environment (i.e. in production, when running in Docker).
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()

if __name__ == "__main__":
    settings = get_settings()
    print(settings.dict())