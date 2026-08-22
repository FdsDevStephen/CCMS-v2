from __future__ import annotations

import json
import time

from pathlib import Path

import time

start = time.perf_counter()

print("1. Importing LegalExtractor...", flush=True)
from extractor.extractor import LegalExtractor
print(f"   Done: {time.perf_counter() - start:.2f}s", flush=True)

start = time.perf_counter()

print("2. Importing SurveyLocationExtractor...", flush=True)
from extractor.survey_location import SurveyLocationExtractor
print(f"   Done: {time.perf_counter() - start:.2f}s", flush=True)

start = time.perf_counter()

print("3. Importing LegalTextChunker...", flush=True)
from RAG.chunker import LegalTextChunker
print(f"   Done: {time.perf_counter() - start:.2f}s", flush=True)

start = time.perf_counter()

print("4. Importing EmbeddingModel...", flush=True)
from RAG.embedding import EmbeddingModel
print(f"   Done: {time.perf_counter() - start:.2f}s", flush=True)

start = time.perf_counter()

print("5. Importing LegalRetriever...", flush=True)
from RAG.retriever import LegalRetriever
print(f"   Done: {time.perf_counter() - start:.2f}s", flush=True)

start = time.perf_counter()

print("6. Importing HybridRetriever...", flush=True)
from RAG.hybrid_retreiver import HybridRetriever
print(f"   Done: {time.perf_counter() - start:.2f}s", flush=True)

start = time.perf_counter()

print("7. Importing QdrantVectorStore...", flush=True)
from RAG.vector_store import QdrantVectorStore
print(f"   Done: {time.perf_counter() - start:.2f}s", flush=True)

start = time.perf_counter()

print("8. Importing ActExtractor...", flush=True)
from RAG.act_extractor import ActExtractor
print(f"   Done: {time.perf_counter() - start:.2f}s", flush=True)

print()
print("=" * 60)
print("ALL IMPORTS COMPLETE")
print("=" * 60)


# ==========================================================
# CONFIG
# ==========================================================

TXT_PATH = Path(
    r"C:\Users\steph\OneDrive\Desktop\CEG\CCMS v2\section_output\WP-14650-2021-B.txt"
)


# ==========================================================
# TIMER
# ==========================================================

pipeline_start = time.perf_counter()


# ==========================================================
# 1. READ OCR TEXT
# ==========================================================

if not TXT_PATH.exists():
    raise FileNotFoundError(
        f"Text file not found: {TXT_PATH}"
    )

try:
    text = TXT_PATH.read_text(
        encoding="utf-8"
    )

except UnicodeDecodeError:

    try:
        text = TXT_PATH.read_text(
            encoding="cp1252"
        )

    except UnicodeDecodeError:

        text = TXT_PATH.read_text(
            encoding="latin-1"
        )


document_name = TXT_PATH.stem


# ==========================================================
# 2. EXTRACT CASE INFORMATION
# ==========================================================

legal_extractor = LegalExtractor()

base_result = legal_extractor.extract(
    text=text,
    case_number=document_name,
)

print(">>> LEGAL EXTRACTION COMPLETE <<<", flush=True)
print(">>> STARTING CHUNKING <<<", flush=True)

# ==========================================================
# 3. CHUNK DOCUMENT
# ==========================================================

chunker = LegalTextChunker()

chunks = chunker.chunk_file(
    TXT_PATH
)

if not chunks:
    raise RuntimeError(
        "No chunks were created. "
        "Check LegalTextChunker section parsing."
    )


# ==========================================================
# 4. GENERATE EMBEDDINGS
# ==========================================================

embedding_model = EmbeddingModel()

texts = [
    chunk["text"]
    for chunk in chunks
]

embeddings = embedding_model.encode(
    texts
)


# ==========================================================
# 5. STORE DOCUMENT IN QDRANT
# ==========================================================

vector_store = QdrantVectorStore()

vector_store.insert(
    chunks,
    embeddings,
)


# ==========================================================
# 6. SURVEY LOCATION EXTRACTION
# ==========================================================

survey_numbers = base_result[
    "survey_numbers"
]

if survey_numbers:

    location_extractor = (
        SurveyLocationExtractor(
            text
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


# ==========================================================
# 7. ACT + SECTION EXTRACTION
# ==========================================================

act_extractor = ActExtractor(
    chunks,
    retriever=HybridRetriever(
        vector_retriever=LegalRetriever(
            # Reuse the encoder built in step 4. A second bge-m3 is another
            # 2.27 GB in fp32; on an 8 GB machine the forward pass on the
            # second copy segfaults instead of raising MemoryError.
            embedding_model=embedding_model,
        )
    ),
)

act_result = act_extractor.extract(
    document=document_name,
    sections=base_result["sections"],
    top_k=5,
)


# ==========================================================
# 8. ADD ACT RESULT
# ==========================================================

base_result["acts"] = act_result.get(
    "acts",
    []
)

base_result["sections"] = act_result.get(
    "sections",
    []
)

base_result[
    "act_section_mapping"
] = act_result.get(
    "act_section_mapping",
    []
)


# ==========================================================
# 9. TOTAL PIPELINE TIME
# ==========================================================

pipeline_end = time.perf_counter()

total_time = (
    pipeline_end - pipeline_start
)


# ==========================================================
# 10. FINAL OUTPUT ONLY
# ==========================================================

print("\n" + "=" * 80)

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