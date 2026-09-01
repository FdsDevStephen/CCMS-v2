"""
Test prayer extraction on ALL PDFs in uploads/ folder.
Body pages 2-13 always OCR'd + prayer pages found via fast scan.
Run:  python test_prayer_all.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ocr_v2 import OCRConfig, process_document
from extractor.prayer_extractor import PrayerExtractor

upload_dir = Path("uploads")
pdfs = sorted(upload_dir.glob("*.pdf"))

print(f"Found {len(pdfs)} PDFs in uploads/\n")

for i, pdf in enumerate(pdfs, 1):
    print("=" * 60)
    print(f"  [{i}/{len(pdfs)}] {pdf.name}")
    print("=" * 60)

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

    try:
        result = process_document(pdf, cfg)
        ocr_text = result.get("sections", {}).get("body", "") if isinstance(result, dict) else str(result)

        prayer = PrayerExtractor().extract(ocr_text)

        if prayer:
            print(f"  OCR: {len(ocr_text)} chars | Prayer: {len(prayer)} chars\n")
            print(prayer[:500])
            if len(prayer) > 500:
                print(f"\n  ... ({len(prayer) - 500} more chars)")
        else:
            print(f"  OCR: {len(ocr_text)} chars | Prayer: NOT FOUND")
    except Exception as e:
        print(f"  ERROR: {e}")

    print()
