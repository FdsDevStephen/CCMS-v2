from __future__ import annotations

import json
import time
from pathlib import Path


# ==========================================================
# TIMER
# ==========================================================

def stage_time(
    label: str,
    start: float,
) -> None:
    elapsed = time.perf_counter() - start

    print(
        f">>> {label}: {elapsed:.2f}s",
        flush=True,
    )


# ==========================================================
# IMPORTS
# ==========================================================

start = time.perf_counter()

print(
    "1. Importing OCRProcessor...",
    flush=True,
)

from ocr_v2 import OCRProcessor

print(
    f"   Done: "
    f"{time.perf_counter() - start:.2f}s",
    flush=True,
)


start = time.perf_counter()

print(
    "2. Importing LegalExtractor...",
    flush=True,
)

from extractor.extractor import LegalExtractor

print(
    f"   Done: "
    f"{time.perf_counter() - start:.2f}s",
    flush=True,
)


start = time.perf_counter()

print(
    "3. Importing SurveyLocationExtractor...",
    flush=True,
)

from extractor.survey_location import (
    SurveyLocationExtractor,
)

print(
    f"   Done: "
    f"{time.perf_counter() - start:.2f}s",
    flush=True,
)


start = time.perf_counter()

print(
    "4. Importing LegalTextChunker...",
    flush=True,
)

from RAG.chunker import LegalTextChunker

print(
    f"   Done: "
    f"{time.perf_counter() - start:.2f}s",
    flush=True,
)


start = time.perf_counter()

print(
    "5. Importing EmbeddingModel...",
    flush=True,
)

from RAG.embedding import EmbeddingModel

print(
    f"   Done: "
    f"{time.perf_counter() - start:.2f}s",
    flush=True,
)


start = time.perf_counter()

print(
    "6. Importing LegalRetriever...",
    flush=True,
)

from RAG.retriever import LegalRetriever

print(
    f"   Done: "
    f"{time.perf_counter() - start:.2f}s",
    flush=True,
)


start = time.perf_counter()

print(
    "7. Importing HybridRetriever...",
    flush=True,
)

from RAG.hybrid_retreiver import (
    HybridRetriever,
)

print(
    f"   Done: "
    f"{time.perf_counter() - start:.2f}s",
    flush=True,
)


start = time.perf_counter()

print(
    "8. Importing QdrantVectorStore...",
    flush=True,
)

from RAG.vector_store import QdrantVectorStore

print(
    f"   Done: "
    f"{time.perf_counter() - start:.2f}s",
    flush=True,
)


start = time.perf_counter()

print(
    "9. Importing ActExtractor...",
    flush=True,
)

from RAG.act_extractor import ActExtractor

print(
    f"   Done: "
    f"{time.perf_counter() - start:.2f}s",
    flush=True,
)


print()
print("=" * 70)
print("ALL IMPORTS COMPLETE")
print("=" * 70)


# ==========================================================
# CONFIG
# ==========================================================

PDF_PATH = Path(
    r"c:\Users\steph\OneDrive\Desktop\data\KLR- LG Case Copy\WP-625-2023-B.pdf"
)

OUTPUT_FOLDER = Path(
    r"C:\Users\steph\OneDrive\Desktop\CEG\CCMS v2\section_output"
)

TESSERACT_PATH = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


# ==========================================================
# PIPELINE TIMER
# ==========================================================

pipeline_start = time.perf_counter()


# ==========================================================
# 1. OCR PROCESSING
# ==========================================================

print()
print("=" * 70)
print("1. OCR PROCESSING")
print("=" * 70)

if not PDF_PATH.exists():
    raise FileNotFoundError(
        f"PDF file not found: {PDF_PATH}"
    )

stage_start = time.perf_counter()

ocr_processor = OCRProcessor(
    output_folder=OUTPUT_FOLDER,
    tesseract_path=TESSERACT_PATH,
    search_pages=30,
    fast_dpi=150,
    full_dpi=220,
    max_workers=8,
    page_start=2,
    page_end=13,
    prefer_text_layer=True,
    denoise=True,
)

ocr_text, txt_path = (
    ocr_processor.process(PDF_PATH)
)

stage_time(
    "OCR",
    stage_start,
)

print(
    f"OCR text saved to: {txt_path}",
    flush=True,
)

print(
    f"OCR text length: "
    f"{len(ocr_text):,} characters",
    flush=True,
)


# ==========================================================
# 2. BASIC LEGAL EXTRACTION
# ==========================================================

print()
print("=" * 70)
print("2. LEGAL EXTRACTION")
print("=" * 70)

stage_start = time.perf_counter()

document_name = PDF_PATH.stem

legal_extractor = LegalExtractor()

base_result = legal_extractor.extract(
    text=ocr_text,
    case_number=document_name,
)

stage_time(
    "LEGAL EXTRACTION",
    stage_start,
)


# ==========================================================
# 3. LOAD BGE-M3 ONCE
# ==========================================================

print()
print("=" * 70)
print("3. LOADING EMBEDDING MODEL")
print("=" * 70)

stage_start = time.perf_counter()

embedding_model = EmbeddingModel()

stage_time(
    "BGE-M3 LOAD",
    stage_start,
)


# ==========================================================
# 4. CHUNK DOCUMENT
# ==========================================================

print()
print("=" * 70)
print("4. CHUNKING")
print("=" * 70)

stage_start = time.perf_counter()

chunker = LegalTextChunker(
    chunk_size=450,
    overlap=50,
    tokenizer=embedding_model.tokenizer,
)

chunks = chunker.chunk_file(
    txt_path
)

if not chunks:
    raise RuntimeError(
        "No chunks were created. "
        "Check LegalTextChunker section parsing."
    )

stage_time(
    "CHUNKING",
    stage_start,
)

print(
    f"Chunks created: {len(chunks)}",
    flush=True,
)


# ==========================================================
# 5. GENERATE EMBEDDINGS
# ==========================================================

print()
print("=" * 70)
print("5. GENERATING EMBEDDINGS")
print("=" * 70)

stage_start = time.perf_counter()

texts = [
    chunk["text"]
    for chunk in chunks
]

embeddings = embedding_model.encode(
    texts,
    batch_size=12,
)

stage_time(
    "EMBEDDINGS",
    stage_start,
)

print(
    f"Embedding shape: "
    f"{embeddings.shape}",
    flush=True,
)


# ==========================================================
# 6. STORE DOCUMENT IN QDRANT
# ==========================================================

print()
print("=" * 70)
print("6. QDRANT")
print("=" * 70)

stage_start = time.perf_counter()

vector_store = QdrantVectorStore()

vector_store.insert(
    chunks,
    embeddings,
)

stage_time(
    "QDRANT INSERT",
    stage_start,
)


# ==========================================================
# 7. SURVEY LOCATION EXTRACTION
# ==========================================================

print()
print("=" * 70)
print("7. SURVEY LOCATION EXTRACTION")
print("=" * 70)

stage_start = time.perf_counter()

survey_numbers = base_result[
    "survey_numbers"
]

if survey_numbers:

    location_extractor = (
        SurveyLocationExtractor(
            ocr_text
        )
    )

    survey_locations = (
        location_extractor.extract(
            survey_numbers
        )
    )

else:

    survey_locations = []

base_result[
    "survey_locations"
] = survey_locations

stage_time(
    "SURVEY LOCATION",
    stage_start,
)


# ==========================================================
# 8. BUILD HYBRID RETRIEVER
# ==========================================================

print()
print("=" * 70)
print("8. BUILDING HYBRID RETRIEVER")
print("=" * 70)

stage_start = time.perf_counter()

retriever = HybridRetriever(
    vector_retriever=LegalRetriever(
        embedding_model=embedding_model,
    )
)

retriever.build_bm25(
    chunks
)

stage_time(
    "BM25 + HYBRID RETRIEVER",
    stage_start,
)


# ==========================================================
# 9. ACT + SECTION EXTRACTION
# ==========================================================

print()
print("=" * 70)
print("9. ACT + SECTION EXTRACTION")
print("=" * 70)

stage_start = time.perf_counter()

act_extractor = ActExtractor(
    chunks,
    retriever=retriever,
)

act_result = act_extractor.extract(
    document=document_name,
    sections=base_result["sections"],
    top_k=5,
)

stage_time(
    "ACT EXTRACTION",
    stage_start,
)


# ==========================================================
# 10. ADD ACT RESULT
# ==========================================================

base_result["acts"] = act_result.get(
    "acts",
    [],
)

base_result["sections"] = act_result.get(
    "sections",
    base_result["sections"],
)

base_result[
    "act_section_mapping"
] = act_result.get(
    "act_section_mapping",
    [],
)


# ==========================================================
# 11. TOTAL PIPELINE TIME
# ==========================================================

pipeline_end = time.perf_counter()

total_time = (
    pipeline_end - pipeline_start
)


# ==========================================================
# 12. FINAL OUTPUT
# ==========================================================

print()
print("=" * 80)
print("FINAL OUTPUT")
print("=" * 80)

print(
    json.dumps(
        base_result,
        indent=4,
        ensure_ascii=False,
    )
)

print("=" * 80)

print(
    f"TOTAL PIPELINE TIME: "
    f"{total_time:.2f} seconds"
)

print("=" * 80)

print(
    "FULL LEGAL DOCUMENT PIPELINE COMPLETED"
)

print("=" * 80)