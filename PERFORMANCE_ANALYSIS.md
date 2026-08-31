# Performance Optimizations Applied

## Changes Made

### 1. OCR Multi-Threading (CRITICAL)
**File:** `app.py` — `get_ocr_processor()`
**Before:** `max_workers=1` (fully sequential)
**After:** `max_workers=min(4, os.cpu_count())` (4 threads)
**Impact:** OCR is the slowest stage. This gives ~3-4x speedup on multi-core machines.
pytesseract shells out to the Tesseract binary and releases the GIL, so threads give real parallelism.

### 2. Legal Extraction + Prayer in Parallel (MEDIUM)
**File:** `app.py` — `run_pipeline()` stages 2+3
**Before:** Sequential — Legal Extraction then Prayer
**After:** Concurrent via `ThreadPoolExecutor(max_workers=2)`
**Impact:** Both are pure text processing with no dependency on each other. Saves ~1-3s per document.

### 3. QdrantVectorStore Cached (HIGH)
**File:** `app.py` — `get_vector_store()`
**Before:** `QdrantVectorStore()` created fresh every pipeline call
**After:** `@st.cache_resource` singleton — created once, reused
**Impact:** Eliminates repeated HTTP connection setup to Qdrant per document.

### 4. LegalRetriever Cached (HIGH)
**File:** `app.py` — `get_legal_retriever()`
**Before:** `LegalRetriever()` created fresh every pipeline call
**After:** `@st.cache_resource` singleton — created once, reused
**Impact:** Eliminates repeated Qdrant client init per document.

### 5. Embedding Batch Size (MEDIUM)
**File:** `app.py` — `run_pipeline()` stage 6
**Before:** `batch_size=12`
**After:** `batch_size=32`
**Impact:** Larger batches = fewer forward passes through BGE-M3. ~2x faster on GPU, ~1.5x on CPU.

### 6. Duplicate normalize() Removed (LOW)
**File:** `ocr_v2.py` line 142-146
**Before:** `normalize()` defined twice in the same file
**After:** Single definition
**Impact:** Code clarity, no runtime effect (Python silently uses the last definition).

---

## Expected Speedup (per document)

| Stage | Before | After | Notes |
|-------|--------|-------|-------|
| OCR | ~15-30s | ~5-10s | 3-4x faster with 4 threads |
| Legal + Prayer | ~3-5s | ~2-3s | Parallel execution |
| Embeddings | ~2-4s | ~1-2s | batch_size 12→32 |
| Qdrant connect | ~0.5s | ~0s | Cached singleton |
| Retriever init | ~0.5s | ~0s | Cached singleton |
| **Total** | **~25-45s** | **~10-18s** | **~50-60% faster** |

---

## Domain Models Created

New `domain/` package with typed, immutable data structures:

| Model | Purpose |
|-------|---------|
| `Section` | Legal section number (e.g. "79A", "136(2)") |
| `Act` | Named statute (e.g. "Karnataka Land Reforms Act, 1961") |
| `ActSectionMapping` | Act → Sections relationship |
| `SurveyNumber` | Land parcel identifier |
| `SurveyLocation` | Administrative hierarchy (village→hobli→taluk→district) |
| `Prayer` | Relief sought in the document |
| `LegalDocument` | Aggregate root — carries identity through pipeline |
| `DocumentChunk` | Text chunk for embedding/RAG |
| `ExtractionResult` | Final pipeline output (replaces raw dict) |
| `ExtractionTimings` | Performance tracking |

These are backward-compatible: `ExtractionResult.to_dict()` produces the same dict format as before.
