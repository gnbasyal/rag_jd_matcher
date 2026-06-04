from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM model names — API keys are supplied at runtime via the UI
    openai_model: str
    anthropic_model: str

    # Embeddings
    embedding_model: str

    # Vector store
    chroma_persist_dir: str
    jd_collection_name: str

    # Source
    jd_source_dir: str


settings = Settings()
