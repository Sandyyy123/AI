# src/config.py
from dataclasses import dataclass
from dotenv import load_dotenv
import os

@dataclass(frozen=True)
class Settings:
    # Providers / keys
    GOOGLE_API_KEY: str | None = None
    OPENAI_API_KEY: str | None = None
    OPENROUTER_API_KEY: str | None = None
    OPENROUTER_BASE_URL: str | None = None

    # Default models
    GEMINI_EMBED_MODEL: str = "gemini-embedding-001"
    ST_DEFAULT_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

def load_settings(env_path: str | None = None) -> Settings:
    if env_path:
        load_dotenv(env_path)
    else:
        load_dotenv()

    return Settings(
        GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY"),
        OPENAI_API_KEY=os.getenv("OPENAI_API_KEY"),
        OPENROUTER_API_KEY=os.getenv("OPENROUTER_API_KEY"),
        OPENROUTER_BASE_URL=os.getenv("OPENROUTER_BASE_URL"),
    )
