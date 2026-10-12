"""
Unified configuration for CCMS v2.

Single source of truth for ALL settings.

Usage:
    from config.settings import settings

    # or import individual values:
    from config.settings import MODEL_NAME, UPLOAD_DIR
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


# ==========================================================
# BASE
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent


# ==========================================================
# LLM
# ==========================================================

LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "ollama")
LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "http://localhost:11434")
MODEL_NAME: str = os.getenv("MODEL_NAME", "qwen2.5:3b-instruct")

# Hosted "/generate" server (used when LLM_PROVIDER=generate), e.g. the
# Qwen3-14B "qwen13" or Gemma servers.
GENERATE_URL: str = os.getenv(
    "GENERATE_URL", "http://101.53.137.13:8004/qwen13/generate"
)
GENERATE_MAX_TOKENS: int = int(os.getenv("GENERATE_MAX_TOKENS", "1500"))
# Sent with every request. Must stay 1.0: the vLLM wrapper's default of
# 1.15 stops the model copying Act names from the text (0/8 vs 8/8 Acts
# found in testing). Servers without this parameter ignore it.
GENERATE_REPETITION_PENALTY: float = float(
    os.getenv("GENERATE_REPETITION_PENALTY", "1.0")
)
# Append Qwen3's "/no_think" (much faster; thinking adds nothing here).
GENERATE_NO_THINK: bool = (
    os.getenv("GENERATE_NO_THINK", "true").lower() in ("1", "true", "yes")
)

# Any OpenAI-compatible chat API (used when LLM_PROVIDER=openai),
# e.g. the hosted vLLM Qwen.
OPENAI_BASE_URL: str = os.getenv(
    "OPENAI_BASE_URL", "http://101.53.137.13:8004/vllm_openai/v1"
)
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "qwen2.5-3b-instruct")
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_MAX_TOKENS: int = int(os.getenv("OPENAI_MAX_TOKENS", "1024"))

# LLM request settings
REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", "120"))
MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "3"))


# ==========================================================
# API
# ==========================================================

API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
API_PORT: int = int(os.getenv("API_PORT", "8000"))


# ==========================================================
# OCR
# ==========================================================

TESSERACT_CMD: str = os.getenv("TESSERACT_CMD", "tesseract")
POPPLER_PATH: str = os.getenv("POPPLER_PATH", "")

# Page range
FIRST_PAGE: int = int(os.getenv("FIRST_PAGE", "2"))
LAST_PAGE: int = int(os.getenv("LAST_PAGE", "12"))

# OCR quality
SEARCH_PAGES: int = int(os.getenv("SEARCH_PAGES", "30"))
FAST_DPI: int = int(os.getenv("FAST_DPI", "150"))
FULL_DPI: int = int(os.getenv("FULL_DPI", "220"))
MAX_WORKERS: int = int(os.getenv("MAX_WORKERS", str(min(8, os.cpu_count() or 4))))


# ==========================================================
# EMBEDDINGS
# ==========================================================

EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")

# local  = run BGE-M3 on this machine (~4 GB RAM in float32)
# remote = hosted BGE-M3 service (EMBEDDING_URL), no local model memory
EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "local")
EMBEDDING_URL: str = os.getenv(
    "EMBEDDING_URL", "http://101.53.137.13:8004/smarana-bge-m3/embed"
)

# CPU dtype for the embedding model (GPU always uses float16):
#   float32  - fastest on most CPUs, ~4 GB
#   bfloat16 - ~1 GB less, so Ollama still fits on 8 GB machines,
#              but ~5x slower on CPUs without native bfloat16
#   auto     - float32 if enough memory is free at load time, else bfloat16
EMBEDDING_CPU_DTYPE: str = os.getenv("EMBEDDING_CPU_DTYPE", "auto")

# "auto" picks float32 only if at least this much memory (GB, RAM + page
# file) is free: ~4 GB for BGE-M3 plus room for the Ollama model.
EMBEDDING_FLOAT32_MIN_FREE_GB: float = float(
    os.getenv("EMBEDDING_FLOAT32_MIN_FREE_GB", "7")
)


# ==========================================================
# VECTOR DB
# ==========================================================

# memory = keep the current documents' vectors in memory (no Docker needed)
# qdrant = Qdrant server at QDRANT_HOST:QDRANT_PORT
VECTOR_STORE: str = os.getenv("VECTOR_STORE", "memory")

QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "legal_documents")


# ==========================================================
# DIRECTORIES
# ==========================================================

DATA_DIR: Path = BASE_DIR / "data"
UPLOAD_DIR: Path = BASE_DIR / os.getenv("UPLOAD_DIR", "uploads")
TEMP_DIR: Path = BASE_DIR / os.getenv("TEMP_DIR", "temp")
OUTPUT_TEXT_DIR: Path = BASE_DIR / os.getenv("OUTPUT_TEXT_DIR", "output_text")
OUTPUT_JSON_DIR: Path = BASE_DIR / os.getenv("OUTPUT_JSON_DIR", "output_json")
LOG_DIR: Path = BASE_DIR / os.getenv("LOG_DIR", "logs")
SECTION_OUTPUT_DIR: Path = BASE_DIR / "section_output"
RESULTS_DIR: Path = BASE_DIR / "results"


# ==========================================================
# RAG
# ==========================================================

CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "450"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))


# ==========================================================
# LOGGING
# ==========================================================

LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")


# ==========================================================
# AUTO-CREATE DIRECTORIES
# ==========================================================

_dirs_to_create = [
    DATA_DIR,
    UPLOAD_DIR,
    TEMP_DIR,
    OUTPUT_TEXT_DIR,
    OUTPUT_JSON_DIR,
    LOG_DIR,
    SECTION_OUTPUT_DIR,
    RESULTS_DIR,
]

for _folder in _dirs_to_create:
    _folder.mkdir(parents=True, exist_ok=True)
