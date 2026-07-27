from pathlib import Path
import os

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

# =========================
# LLM
# =========================

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434")
MODEL_NAME = os.getenv("MODEL_NAME", "qwen2.5:7b-instruct")

# =========================
# API
# =========================

API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))

# =========================
# Directories
# =========================

UPLOAD_DIR = BASE_DIR / os.getenv("UPLOAD_DIR", "uploads")
TEMP_DIR = BASE_DIR / os.getenv("TEMP_DIR", "temp")
OUTPUT_TEXT_DIR = BASE_DIR / os.getenv("OUTPUT_TEXT_DIR", "output_text")
OUTPUT_JSON_DIR = BASE_DIR / os.getenv("OUTPUT_JSON_DIR", "output_json")
LOG_DIR = BASE_DIR / os.getenv("LOG_DIR", "logs")

# =========================
# OCR
# =========================

TESSERACT_CMD = os.getenv("TESSERACT_CMD", "tesseract")
POPPLER_PATH = os.getenv("POPPLER_PATH", "")

# =========================
# Logging
# =========================

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")