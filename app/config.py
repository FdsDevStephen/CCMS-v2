"""
App Configuration — BACKWARD-COMPATIBILITY SHIM.

All settings live in config.settings.
This file re-exports them so that existing imports like:

    from app.config import MODEL_NAME

continue to work without changes.

Preferred import going forward:
    from config.settings import MODEL_NAME
"""

from config.settings import (
    BASE_DIR,
    LLM_PROVIDER,
    LLM_BASE_URL,
    LLM_BASE_URL as OLLAMA_HOST,
    MODEL_NAME,
    REQUEST_TIMEOUT,
    MAX_RETRIES,
    TESSERACT_CMD as TESSERACT_PATH,
    POPPLER_PATH,
    FIRST_PAGE,
    LAST_PAGE,
    DATA_DIR,
)

# ==========================================================
# COMPATIBILITY ALIASES
#
# config/settings.py uses UPLOAD_DIR, OUTPUT_TEXT_DIR, etc.
# app/config.py historically used UPLOAD_FOLDER, OUTPUT_TEXT_FOLDER, etc.
# These aliases keep any code using the old names working.
# ==========================================================

from config.settings import (
    UPLOAD_DIR as UPLOAD_FOLDER,
    OUTPUT_TEXT_DIR as OUTPUT_TEXT_FOLDER,
    OUTPUT_JSON_DIR as OUTPUT_JSON_FOLDER,
    TEMP_DIR as TEMP_FOLDER,
    LOG_DIR as LOG_FOLDER,
)

# Backward-compat: app/config.py created directories on import.
# config/settings.py now handles this automatically.
