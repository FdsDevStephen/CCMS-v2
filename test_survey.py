"""
Test survey number + survey location extraction on one document.

Usage:
    python test_survey.py                          # default PDF below
    python test_survey.py path\\to\\file.pdf       # any PDF (runs OCR)
    python test_survey.py path\\to\\file.txt       # saved OCR text (fast)
    python test_survey.py file.pdf --no-llm        # rules only, no Ollama

Does not need Qdrant. Uses the same OCR settings as app.py.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# ==========================================================
# DEFAULT DOCUMENT (used when no path is given)
# ==========================================================
DEFAULT_PATH = Path(__file__).parent / "uploads" / "WP-7049-2021-B.pdf"
# ==========================================================

args = [a for a in sys.argv[1:] if not a.startswith("--")]
use_llm = "--no-llm" not in sys.argv

path = Path(args[0]) if args else DEFAULT_PATH

if not path.exists():
    print(f"File not found: {path}")
    sys.exit(1)

print(f"Document: {path.name}\n")

# ----------------------------------------------------------
# 1. Text
# ----------------------------------------------------------

if path.suffix.lower() == ".txt":
    text = path.read_text(encoding="utf-8", errors="ignore")
else:
    from ocr_v2 import OCRProcessor

    start = time.perf_counter()
    text, _ = OCRProcessor(
        output_folder=Path(__file__).parent / "section_output",
        tesseract_path=r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        search_pages=30,
        fast_dpi=150,
        full_dpi=220,
        max_workers=4,
        page_start=2,
        page_end=13,
        prefer_text_layer=True,
        denoise=True,
    ).process(path)
    print(f"OCR: {len(text)} chars in {time.perf_counter() - start:.1f}s\n")

# ----------------------------------------------------------
# 2. Survey numbers + locations
# ----------------------------------------------------------

from extractor.normalizer import Normalizer
from extractor.survey_extractor import SurveyExtractor
from extractor.survey_location import SurveyLocationExtractor

survey_numbers = Normalizer.normalize_survey_numbers(
    SurveyExtractor(text).extract()
)

start = time.perf_counter()
locations = SurveyLocationExtractor(text, use_llm=use_llm).extract(
    survey_numbers
)
elapsed = time.perf_counter() - start

# ----------------------------------------------------------
# 3. Report
# ----------------------------------------------------------

print("=" * 70)
print(f"  SURVEY NUMBERS ({len(survey_numbers)})")
print("=" * 70)
print(", ".join(survey_numbers) or "None found.")

print()
print("=" * 70)
print("  SURVEY LOCATIONS")
print("=" * 70)

for location in locations:
    parts = [
        f"{field.title()}: {location[field]}"
        for field in ("village", "hobli", "taluk", "district")
        if location[field]
    ]
    print(f"{location['survey_number']:>12}  {', '.join(parts) or '-'}")

located = sum(
    any(loc[f] for f in ("village", "hobli", "taluk", "district"))
    for loc in locations
)
print("=" * 70)
print(
    f"Located {located}/{len(locations)} in {elapsed:.2f}s"
    f"{'' if use_llm else '  (LLM disabled)'}"
)

if "--json" in sys.argv:
    print(json.dumps(locations, indent=2, ensure_ascii=False))
