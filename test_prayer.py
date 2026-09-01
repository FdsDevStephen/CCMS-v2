"""
Paste your PDF path below, then run:  python test_prayer.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# ==========================================================
# PASTE YOUR PDF PATH HERE
# ==========================================================
PDF_PATH = r"C:\Users\steph\OneDrive\Desktop\CEG\CCMS v2\uploads\WP-7049-2021-B.pdf"
# ==========================================================

pdf = Path(PDF_PATH)
if not pdf.exists():
    print(f"File not found: {pdf}")
    sys.exit(1)

print(f"PDF: {pdf.name}\n")

from ocr_v2 import OCRConfig, process_document

cfg = OCRConfig(
    search_pages=30,
    fast_dpi=150,
    full_dpi=220,
    max_workers=4,
    page_start=2,
    page_end=13,
    prefer_text_layer=True,
    denoise=True,
)

result = process_document(pdf, cfg)
ocr_text = result.get("sections", {}).get("body", "") if isinstance(result, dict) else str(result)
print(f"OCR: {len(ocr_text)} chars\n")

from extractor.prayer_extractor import PrayerExtractor

prayer = PrayerExtractor().extract(ocr_text)

print("=" * 60)
print("  CLEANED PRAYER")
print("=" * 60)
print(prayer if prayer else "No prayer found.")
print("=" * 60)
