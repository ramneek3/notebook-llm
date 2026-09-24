"""Central configuration loaded from environment variables (.env)."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
CHROMA_DIR = DATA_DIR / "chroma"

# Make sure runtime folders exist
for _dir in (UPLOAD_DIR, CHROMA_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------- chat LLM
# Any OpenAI-compatible chat endpoint works. OpenRouter's :free models are
# the zero-cost default; point CHAT_MODEL at gpt-4o-mini + set OPENAI_API_KEY
# to use OpenAI instead.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

CHAT_MODEL = os.getenv("CHAT_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free")

# ------------------------------------------------------------- embeddings
# "local"  -> free, offline, private (sentence-transformers MiniLM)
# "openai" -> text-embedding-3-small via the OpenAI API
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "local")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Chunking: ~4 characters per token is a decent rule of thumb
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150
RETRIEVER_K = 4


def chat_credentials() -> dict:
    """Return kwargs for ChatOpenAI based on which provider key is set."""
    if OPENROUTER_API_KEY:
        return {
            "api_key": OPENROUTER_API_KEY,
            "base_url": OPENROUTER_BASE_URL,
        }
    if OPENAI_API_KEY:
        return {"api_key": OPENAI_API_KEY}
    raise RuntimeError(
        "No chat API key configured. Set OPENROUTER_API_KEY (free models) "
        "or OPENAI_API_KEY in .env."
    )


def require_api_key() -> str:
    """Return the OpenAI API key or raise a clear error if it is missing.

    Only needed when EMBEDDING_BACKEND=openai.
    """
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Either set it in .env or switch "
            "EMBEDDING_BACKEND to 'local' (free, offline)."
        )
    return OPENAI_API_KEY
