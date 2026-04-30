# app/config.py
import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")


def _require(key: str) -> str:
    """Fetch a required environment variable or raise clearly."""
    value = os.getenv(key)
    if not value:
        raise EnvironmentError(
            f"Required environment variable '{key}' is not set. "
            f"Check your .env file against .env.example."
        )
    return value


# --- External services ---
GROQ_API_KEY: str = _require("GROQ_API_KEY")
QDRANT_URL: str = _require("QDRANT_URL")
QDRANT_API_KEY: str = _require("QDRANT_API_KEY")

# --- Vector store ---
COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "doc_intelligence")
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# --- Chunking ---
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))

# --- Retrieval ---
TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", "5"))

# --- LLM ---
LLM_MODEL: str = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

# --- App ---
APP_ENV: str = os.getenv("APP_ENV", "development")