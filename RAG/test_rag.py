from pathlib import Path

from chunker import LegalTextChunker
from embedding import EmbeddingModel
from vector_store import QdrantVectorStore
from retriever import LegalRetriever


TXT_FILE = Path(
    r"C:\Users\steph\OneDrive\Desktop\CEG\CCMS v2\section_output\WP-12860-2025-B.txt"
)


print("\n" + "=" * 80)
print("SINGLE DOCUMENT RAG TEST")
print("=" * 80)


# ==========================================================
# 1. CHECK FILE
# ==========================================================

if not TXT_FILE.exists():

    raise FileNotFoundError(
        f"TXT file not found:\n{TXT_FILE}"
    )

print(
    f"\nDocument: {TXT_FILE.name}"
)


# ==========================================================
# 2. LOAD COMPONENTS
# ==========================================================

print("\nLoading RAG components...")

chunker = LegalTextChunker(
    chunk_size=200,
    overlap=50,
)

embedding_model = EmbeddingModel()

vector_store = QdrantVectorStore()

retriever = LegalRetriever()

print("Components loaded.")


# ==========================================================
# 3. CHUNK DOCUMENT
# ==========================================================

print("\n" + "=" * 80)
print("CHUNKING")
print("=" * 80)

chunks = chunker.chunk_file(
    TXT_FILE
)

print(
    f"Total chunks created: {len(chunks)}"
)

if not chunks:

    raise RuntimeError(
        "No chunks were created."
    )


# ==========================================================
# 4. SHOW CHUNK INFORMATION
# ==========================================================

print("\n" + "=" * 80)
print("CHUNK INFORMATION")
print("=" * 80)

for chunk in chunks[:5]:

    print("\n" + "-" * 80)

    print(
        f"Document : {chunk['document']}"
    )

    print(
        f"Section  : {chunk['section']}"
    )

    print(
        f"Chunk ID : {chunk['chunk_id']}"
    )

    print(
        f"Text     :\n{chunk['text'][:500]}"
    )


# ==========================================================
# 5. GENERATE EMBEDDINGS
# ==========================================================

print("\n" + "=" * 80)
print("GENERATING EMBEDDINGS")
print("=" * 80)

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
# 6. STORE IN QDRANT
# ==========================================================

print("\n" + "=" * 80)
print("STORING IN QDRANT")
print("=" * 80)

vector_store.insert(
    chunks,
    embeddings,
)

print(
    f"Stored {len(chunks)} vectors."
)


# ==========================================================
# 7. RETRIEVE
# ==========================================================

print("\n" + "=" * 80)
print("RETRIEVAL TEST")
print("=" * 80)

query = (
    "What Acts are mentioned "
    "in this case?"
)

document = TXT_FILE.stem

print(
    f"\nQuery: {query}"
)

print(
    f"Document filter: {document}"
)


results = retriever.search(
    query=query,
    top_k=5,
    document=document,
)


# ==========================================================
# 8. DISPLAY RESULTS
# ==========================================================

print("\n" + "=" * 80)
print(
    f"RETRIEVED {len(results)} CHUNKS"
)
print("=" * 80)


for index, result in enumerate(
    results,
    start=1,
):

    print("\n" + "-" * 80)

    print(
        f"RESULT {index}"
    )

    print(
        f"Score    : {result.score}"
    )

    print(
        f"Document : "
        f"{result.payload.get('document')}"
    )

    print(
        f"Section  : "
        f"{result.payload.get('section')}"
    )

    print(
        f"Chunk ID : "
        f"{result.payload.get('chunk_id')}"
    )

    print("\nTEXT:")

    print(
        result.payload.get(
            "text",
            "",
        )
    )


# ==========================================================
# 9. FINISHED
# ==========================================================

print("\n" + "=" * 80)
print("SINGLE DOCUMENT RAG TEST COMPLETED")
print("=" * 80)