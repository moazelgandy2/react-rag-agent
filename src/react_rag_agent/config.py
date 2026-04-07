from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class Settings(BaseSettings):
    ollama_base_url: str = "http://localhost:11434"
    reasoning_model: str = "qwen3.5:9b"
    embedding_model: str = "nomic-embed-text"
    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k: int = 5
    max_iterations: int = 5
    temperature: float = 0.1
    chroma_persist_dir: str = "./chroma_db"
    collection_name: str = "documents"
    data_dir: str = "./data/raw"
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    cors_origins: str = "http://localhost:5173"
    max_upload_mb: int = 25
    session_ttl_minutes: int = 240
    session_max_messages: int = 40
    session_max_count: int = 200
    orchestrator_enabled: bool = True
    orchestrator_model: str = "glm-4.7-flash"
    orchestrator_temperature: float = 0.0

    model_config = SettingsConfigDict(
        env_prefix="",
        case_sensitive=False,
        frozen=True,
    )


settings = Settings()
