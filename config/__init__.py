"""
CCMS Configuration Package.

Single source of truth: config.settings

Usage:
    from config import MODEL_NAME, UPLOAD_DIR, settings  (if settings is added)

    # or
    from config.settings import MODEL_NAME, UPLOAD_DIR
"""

from config.settings import (
    BASE_DIR,
    LLM_PROVIDER,
    LLM_BASE_URL,
    MODEL_NAME,
    REQUEST_TIMEOUT,
    MAX_RETRIES,
    API_HOST,
    API_PORT,
    TESSERACT_CMD,
    POPPLER_PATH,
    FIRST_PAGE,
    LAST_PAGE,
    SEARCH_PAGES,
    FAST_DPI,
    FULL_DPI,
    MAX_WORKERS,
    EMBEDDING_MODEL,
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_COLLECTION,
    DATA_DIR,
    UPLOAD_DIR,
    TEMP_DIR,
    OUTPUT_TEXT_DIR,
    OUTPUT_JSON_DIR,
    LOG_DIR,
    SECTION_OUTPUT_DIR,
    RESULTS_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    LOG_LEVEL,
)

__all__ = [
    "BASE_DIR",
    "LLM_PROVIDER",
    "LLM_BASE_URL",
    "MODEL_NAME",
    "REQUEST_TIMEOUT",
    "MAX_RETRIES",
    "API_HOST",
    "API_PORT",
    "TESSERACT_CMD",
    "POPPLER_PATH",
    "FIRST_PAGE",
    "LAST_PAGE",
    "SEARCH_PAGES",
    "FAST_DPI",
    "FULL_DPI",
    "MAX_WORKERS",
    "EMBEDDING_MODEL",
    "QDRANT_HOST",
    "QDRANT_PORT",
    "QDRANT_COLLECTION",
    "DATA_DIR",
    "UPLOAD_DIR",
    "TEMP_DIR",
    "OUTPUT_TEXT_DIR",
    "OUTPUT_JSON_DIR",
    "LOG_DIR",
    "SECTION_OUTPUT_DIR",
    "RESULTS_DIR",
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "LOG_LEVEL",
]
