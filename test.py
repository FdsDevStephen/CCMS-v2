from pathlib import Path
import json

from extractor.extractor import LegalExtractor


# ==========================================================
# PDF
# ==========================================================

PDF_PATH = Path(
    r"C:\Users\steph\OneDrive\Desktop\WP-13437-2022-B.pdf"
)

# ==========================================================
# RUN PIPELINE
# ==========================================================

extractor = LegalExtractor()

result = extractor.extract(PDF_PATH)

# ==========================================================
# OUTPUT
# ==========================================================

print("\n" + "=" * 80)
print("FINAL RESULT")
print("=" * 80)

print(json.dumps(result, indent=4, ensure_ascii=False))