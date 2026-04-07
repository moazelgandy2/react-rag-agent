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
    api_workers: int = 1
    cors_origins: str = "http://localhost:5173"
    max_upload_mb: int = 25
    session_ttl_minutes: int = 240
    session_max_messages: int = 40
    session_max_count: int = 200
    orchestrator_enabled: bool = True
    orchestrator_model: str = "glm-4.7-flash"
    orchestrator_temperature: float = 0.0
    reasoning_num_ctx: int = 8192
    reasoning_num_predict: int = 768
    reasoning_keep_alive: str = "30m"
    orchestrator_num_ctx: int = 2048
    orchestrator_num_predict: int = 128
    orchestrator_keep_alive: str = "30m"
    kv_cache_max_entries: int = 2000
    kv_cache_ttl_seconds: int = 300
    retrieval_cache_ttl_seconds: int = 120
    orchestrator_cache_ttl_seconds: int = 600
    response_cache_ttl_seconds: int = 300
    api_log_level: str = "info"

    model_config = SettingsConfigDict(
        env_prefix="",
        case_sensitive=False,
        frozen=True,
    )


settings = Settings()
