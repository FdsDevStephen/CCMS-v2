from pathlib import Path

from config import (
    FIRST_PAGE,
    LAST_PAGE,
    OUTPUT_TEXT_FOLDER,
    POPPLER_PATH,
    TESSERACT_PATH,
)

from extractor.text_chunker import TextChunker
from ocr import OCRProcessor
from extractor.llm.factory import get_llm_client
from extractor.prompts import build_act_extraction_prompt
from extractor.parser import LLMResponseParser


# ==========================================================
# PDF
# ==========================================================

PDF_PATH = Path(
    r"C:\Users\steph\OneDrive\Desktop\WP-19110-2024-B.pdf"
)

# ==========================================================
# OCR
# ==========================================================

ocr = OCRProcessor(
    poppler_path=POPPLER_PATH,
    output_folder=OUTPUT_TEXT_FOLDER,
    tesseract_path=TESSERACT_PATH,
    first_page=FIRST_PAGE,
    last_page=LAST_PAGE,
)

text, _ = ocr.process(PDF_PATH)

chunker = TextChunker(
    chunk_size=3000,
    overlap=300,
)

chunks = chunker.split(text)

# Test ONLY the first chunk
prompt = build_act_extraction_prompt(chunks[0])

llm = get_llm_client()

response = llm.generate(prompt)

print(response)

# ==========================================================
# RAW RESPONSE
# ==========================================================

print("=" * 80)
print("RAW LLM RESPONSE")
print("=" * 80)

print(response)

# ==========================================================
# PARSED RESPONSE
# ==========================================================

parsed = LLMResponseParser.parse(response)

print("\n")
print("=" * 80)
print("PARSED RESPONSE")
print("=" * 80)

for key, value in parsed.items():
    print(f"{key}: {value}")