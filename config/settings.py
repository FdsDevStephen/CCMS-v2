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


# ==========================================================
# VECTOR DB
# ==========================================================

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
