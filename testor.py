from pathlib import Path

from ocr_v2 import OCRProcessor


INPUT_FOLDER = Path(
    r"C:\Users\steph\OneDrive\Desktop\CEG\CCMS v2\uploads"
)


ocr = OCRProcessor(
    output_folder="section_output",
    search_pages=30,
    fast_dpi=150,
    full_dpi=220,
    page_start=2,
    page_end=13,
    prefer_text_layer=True,
    denoise=True,
)


pdf_files = sorted(INPUT_FOLDER.glob("*.pdf"))

print(f"Found {len(pdf_files)} PDF files.")

for index, pdf_path in enumerate(pdf_files, start=1):

    print("\n" + "=" * 80)
    print(f"Processing {index}/{len(pdf_files)}: {pdf_path.name}")
    print("=" * 80)

    try:
        text, output_path = ocr.process(pdf_path)

        print(f"Completed: {pdf_path.name}")
        print(f"Saved to: {output_path}")

    except Exception as exc:
        print(f"FAILED: {pdf_path.name}")
        print(f"Error: {exc}")


print("\n" + "=" * 80)
print("ALL PDF FILES PROCESSED")
print("=" * 80)