from pathlib import Path
import sys
import time

from ocr import OCRProcessor


# ==========================================================
# CONFIG
# ==========================================================

PDF_PATH = Path(
    r"C:\Users\steph\OneDrive\Desktop\CEG\CCMS v2\.streamlit_uploads\WA-710-2024-B.pdf"
)

TESSERACT_PATH = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

OUTPUT_FOLDER = Path("ocr_test_output")


# ==========================================================
# TEST
# ==========================================================

def main():

    print()
    print("=" * 80)
    print("WHOLE DOCUMENT OCR TEST")
    print("=" * 80)

    # ------------------------------------------------------
    # Check PDF
    # ------------------------------------------------------

    if not PDF_PATH.exists():
        print()
        print("ERROR: PDF not found")
        print(f"Path: {PDF_PATH}")
        print()
        sys.exit(1)

    print(f"PDF: {PDF_PATH}")
    print(f"Size: {PDF_PATH.stat().st_size / (1024 * 1024):.2f} MB")

    # ------------------------------------------------------
    # Check Tesseract
    # ------------------------------------------------------

    if not Path(TESSERACT_PATH).exists():
        print()
        print("ERROR: Tesseract not found")
        print(f"Path: {TESSERACT_PATH}")
        print()
        sys.exit(1)

    # ------------------------------------------------------
    # Create OCR processor
    # ------------------------------------------------------

    processor = OCRProcessor(
        tesseract_path=TESSERACT_PATH,
        output_folder=OUTPUT_FOLDER,

        # Whole-document OCR settings
        dpi=220,
        psm=6,
        language="eng",

        # Parallel OCR
        max_workers=8,

        # Preprocessing
        denoise=True,
    )

    # ------------------------------------------------------
    # Run OCR
    # ------------------------------------------------------

    print()
    print("-" * 80)
    print("STARTING OCR")
    print("-" * 80)

    start = time.perf_counter()

    try:

        text, output_path = processor.process(
            PDF_PATH
        )

    except Exception as exc:

        print()
        print("=" * 80)
        print("OCR FAILED")
        print("=" * 80)
        print()
        print(type(exc).__name__)
        print(str(exc))
        print()

        raise

    elapsed = time.perf_counter() - start

    # ------------------------------------------------------
    # Results
    # ------------------------------------------------------

    print()
    print("=" * 80)
    print("OCR TEST COMPLETE")
    print("=" * 80)

    print()
    print(f"Time taken       : {elapsed:.2f} seconds")
    print(f"Characters       : {len(text):,}")
    print(f"Output file      : {output_path}")
    print(
        f"Output size      : "
        f"{output_path.stat().st_size / 1024:.2f} KB"
    )

    # ------------------------------------------------------
    # Basic validation
    # ------------------------------------------------------

    print()
    print("-" * 80)
    print("VALIDATION")
    print("-" * 80)

    if not text.strip():
        print("❌ FAIL: OCR returned no text")

    else:
        print("✅ PASS: OCR returned text")

    if output_path.exists():
        print("✅ PASS: TXT file was created")

    else:
        print("❌ FAIL: TXT file was not created")

    # ------------------------------------------------------
    # Preview
    # ------------------------------------------------------

    print()
    print("-" * 80)
    print("FIRST 5,000 CHARACTERS")
    print("-" * 80)
    print()

    print(text[:5000])

    print()
    print("-" * 80)
    print("END PREVIEW")
    print("-" * 80)

    # ------------------------------------------------------
    # Last 2,000 characters
    # ------------------------------------------------------

    print()
    print("-" * 80)
    print("LAST 2,000 CHARACTERS")
    print("-" * 80)
    print()

    print(text[-2000:])

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()