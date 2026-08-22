print("TEST STARTED", flush=True)

from chunker import LegalTextChunker

print("CHUNKER IMPORTED", flush=True)

from embedding import EmbeddingModel

print("EMBEDDING IMPORTED", flush=True)

TXT_PATH = "../section_output/WP-14650-2021-B.txt"

chunker = LegalTextChunker()

print("CHUNKER CREATED", flush=True)

chunks = chunker.chunk_file(TXT_PATH)

print(
    f"CHUNKS CREATED: {len(chunks)}",
    flush=True
)

texts = [
    chunk["text"]
    for chunk in chunks
]

print("TEXTS CREATED", flush=True)

print("LOADING BGE-M3...", flush=True)

model = EmbeddingModel()

print("MODEL LOADED", flush=True)

embeddings = model.encode(texts)

print("EMBEDDINGS CREATED", flush=True)

print(
    "Embedding shape:",
    embeddings.shape,
    flush=True
)