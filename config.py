import os
from pydantic_settings import BaseSettings
from typing import Optional, Literal

class Settings(BaseSettings):
    # Core settings
    google_api_key: Optional[str] = None
    google_credentials_path: Optional[str] = None
    openai_api_key: Optional[str] = None
    deepgram_api_key: Optional[str] = None
    google_cloud_api: Optional[str] = None

    # Service selection
    stt_service: Literal["google", "openai", "local"] = "google"
    tts_service: Literal["google", "openai", "local"] = "google"
    llm_service: Literal["gemini", "openai", "local"] = "gemini"

    # Local service settings
    local_stt_url: Optional[str] = "http://localhost:8001/stt"
    local_tts_url: Optional[str] = "http://localhost:8002/tts"
    local_llm_url: Optional[str] = "http://localhost:8003/llm"

    # ChromaDB settings
    use_cloud_chromadb: bool = False
    chroma_persist_dir: str = "./chroma_db"
    documents_dir: str = "./documents"
    chroma_api_key: Optional[str] = None
    chroma_tenant: Optional[str] = None
    chroma_database: Optional[str] = None
    
    # Gemini settings
    gemini_model: str = "gemini-1.5-flash"
    gemini_temperature: float = 0.7
    
    # OpenAI settings
    openai_model: str = "gpt-4o"
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    fastapi_url: str = "ws://localhost:8000/ws"
    reload: bool = False
    
    # Logging settings
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Create a single instance of the settings
settings = Settings()

# Optionally: set GOOGLE_APPLICATION_CREDENTIALS globally
if settings.google_credentials_path:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = settings.google_credentials_path
