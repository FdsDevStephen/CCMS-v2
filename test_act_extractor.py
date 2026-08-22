from pathlib import Path
import json

from RAG.chunker import LegalTextChunker
from RAG.embedding import EmbeddingModel
from RAG.vector_store import QdrantVectorStore
from RAG.act_extractor import ActExtractor


# ==========================================================
# CONFIG
# ==========================================================

TXT_PATH = Path(
    r"C:\Users\steph\OneDrive\Desktop\CEG\CCMS v2\section_output\WP-7049-2021-B.txt"
)


# ==========================================================
# START
# ==========================================================

print("=" * 80)
print("ACT + SECTION RAG PIPELINE")
print("=" * 80)


# ==========================================================
# 1. READ TXT
# ==========================================================

print("\n[1] READING TXT")

if not TXT_PATH.exists():
    raise FileNotFoundError(
        f"TXT file not found: {TXT_PATH}"
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

print(
    f"Document  : {document_name}"
)

print(
    f"Characters: {len(text)}"
)


# ==========================================================
# 2. CHUNK DOCUMENT
# ==========================================================

print("\n" + "=" * 80)
print("[2] CHUNKING DOCUMENT")
print("=" * 80)

chunker = LegalTextChunker(
    chunk_size=450,
    overlap=100,
)

chunks = chunker.chunk_file(
    TXT_PATH
)

print(
    f"Chunks created: {len(chunks)}"
)

if not chunks:
    raise RuntimeError(
        "No chunks were created."
    )


# ==========================================================
# 3. ADD DOCUMENT METADATA
# ==========================================================

print("\n" + "=" * 80)
print("[3] PREPARING CHUNKS")
print("=" * 80)

for chunk in chunks:
    chunk["document"] = document_name

print(
    f"Prepared {len(chunks)} chunks"
)

print(
    f"Document metadata: {document_name}"
)


# ==========================================================
# 4. VERIFY CHUNKS
# ==========================================================

print("\n" + "=" * 80)
print("[4] VERIFYING CHUNKS")
print("=" * 80)

for index, chunk in enumerate(
    chunks[:5],
    start=1,
):

    print("\n" + "-" * 80)

    print(
        f"CHUNK {index}"
    )

    print(
        f"Document : {chunk.get('document')}"
    )

    print(
        f"Section  : {chunk.get('section')}"
    )

    print(
        f"Chunk ID : {chunk.get('chunk_id')}"
    )

    print(
        f"Characters: "
        f"{len(chunk.get('text', ''))}"
    )

    # print("\nTEXT:")

    print(
        chunk.get(
            "text",
            "",
        )
    )


# ==========================================================
# 5. GENERATE EMBEDDINGS
# ==========================================================

print("\n" + "=" * 80)
print("[5] GENERATING EMBEDDINGS")
print("=" * 80)

embedding_model = EmbeddingModel()

texts = [
    chunk["text"]
    for chunk in chunks
]

embeddings = embedding_model.encode(
    texts
)

print(
    f"Embedding shape: {embeddings.shape}"
)


# ==========================================================
# 6. STORE EMBEDDINGS IN QDRANT
# ==========================================================

print("\n" + "=" * 80)
print("[6] STORING EMBEDDINGS IN QDRANT")
print("=" * 80)

vector_store = QdrantVectorStore()

vector_store.insert(
    chunks,
    embeddings,
)

print(
    f"Stored vectors: {len(chunks)}"
)


# ==========================================================
# 7. VERIFY QDRANT DOCUMENT
# ==========================================================

print("\n" + "=" * 80)
print("[7] VERIFYING QDRANT STORAGE")
print("=" * 80)

print(
    "Document filter:"
)

print(
    f"    {document_name}"
)


# ==========================================================
# 8. ACT + SECTION EXTRACTION
# ==========================================================

print("\n" + "=" * 80)
print("[8] ACT + SECTION EXTRACTION")
print("=" * 80)

extractor = ActExtractor(
    chunks
)

result = extractor.extract(
    document=document_name,
    top_k=3,
)


# ==========================================================
# 9. FINAL LLM RESULT
# ==========================================================

print("\n" + "=" * 80)
print("[9] FINAL LLM RESPONSE")
print("=" * 80)

print(
    json.dumps(
        result,
        indent=4,
        ensure_ascii=False,
    )
)


# ==========================================================
# 10. ACTS
# ==========================================================

print("\n" + "=" * 80)
print("[10] EXTRACTED ACTS")
print("=" * 80)

print(
    json.dumps(
        result.get(
            "acts",
            [],
        ),
        indent=4,
        ensure_ascii=False,
    )
)


# ==========================================================
# 11. SECTIONS
# ==========================================================

print("\n" + "=" * 80)
print("[11] EXTRACTED SECTIONS")
print("=" * 80)

print(
    json.dumps(
        result.get(
            "sections",
            [],
        ),
        indent=4,
        ensure_ascii=False,
    )
)


# ==========================================================
# 12. ACT → SECTION MAPPING
# ==========================================================

print("\n" + "=" * 80)
print("[12] ACT → SECTION MAPPING")
print("=" * 80)

print(
    json.dumps(
        result.get(
            "act_section_mapping",
            [],
        ),
        indent=4,
        ensure_ascii=False,
    )
)


# ==========================================================
# END
# ==========================================================

print("\n" + "=" * 80)
print("PIPELINE COMPLETED")
print("=" * 80)