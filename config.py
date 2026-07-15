from pathlib import Path

# ==========================================================
# LLM CONFIGURATION
# ==========================================================

# Current Provider
# Future Options:
#   - ollama
#   - vllm
LLM_PROVIDER = "ollama"

# Ollama Configuration
OLLAMA_HOST = "http://localhost:11434"
MODEL_NAME = "qwen2.5:3b-instruct"

# LLM Settings
REQUEST_TIMEOUT = 120          # Seconds
MAX_RETRIES = 3

# ==========================================================
# OCR CONFIGURATION
# ==========================================================

POPPLER_PATH = (
    r"C:\Users\steph\Downloads\Release-26.02.0-0\poppler-26.02.0\Library\bin"
)

# Set to None if Tesseract is already in your system PATH
TESSERACT_PATH = None

FIRST_PAGE = 2
LAST_PAGE = 12

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 150

# ==========================================================
# PROJECT PATHS
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent

UPLOAD_FOLDER = BASE_DIR / "uploads"
OUTPUT_TEXT_FOLDER = BASE_DIR / "output_text"
OUTPUT_JSON_FOLDER = BASE_DIR / "output_json"
TEMP_FOLDER = BASE_DIR / "temp"
LOG_FOLDER = BASE_DIR / "logs"

# ==========================================================
# CREATE PROJECT DIRECTORIES
# ==========================================================

for folder in (
    UPLOAD_FOLDER,
    OUTPUT_TEXT_FOLDER,
    OUTPUT_JSON_FOLDER,
    TEMP_FOLDER,
    LOG_FOLDER,
):
    folder.mkdir(parents=True, exist_ok=True)