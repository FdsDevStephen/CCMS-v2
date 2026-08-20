from pathlib import Path
from app.ocr import OCRProcessor

PDF_PATH = r"C:\Users\steph\Downloads\7.PTCL ACT 1978.pdf"

POPPLER_PATH = r"C:\Users\steph\Downloads\Release-26.02.0-0\poppler-26.02.0\Library\bin"

TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

OUTPUT_FOLDER = Path(r"C:\Users\steph\OneDrive\Desktop\CEG\CCMS v2\data\output_text")

ocr = OCRProcessor(
    poppler_path=POPPLER_PATH,
    output_folder=OUTPUT_FOLDER,
    tesseract_path=TESSERACT_PATH,
)

text, path = ocr.process(PDF_PATH)

print(text)
print(f"Saved to: {path}")