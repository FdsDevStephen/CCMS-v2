from chunker import LegalTextChunker
from embedding import EmbeddingModel
from vector_store import QdrantVectorStore


TXT_PATH = "../section_output/WP-14650-2021-B.txt"


print("=" * 80)
print("CHUNKING")
print("=" * 80)

chunker = LegalTextChunker()

chunks = chunker.chunk_file(TXT_PATH)

print(f"Chunks: {len(chunks)}")


print("\n" + "=" * 80)
print("LOADING EMBEDDING MODEL")
print("=" * 80)

embedding_model = EmbeddingModel()

texts = [
    chunk["text"]
    for chunk in chunks
]


print("\n" + "=" * 80)
print("GENERATING EMBEDDINGS")
print("=" * 80)

embeddings = embedding_model.encode(texts)

print(
    f"Embedding shape: {embeddings.shape}"
)


print("\n" + "=" * 80)
print("CONNECTING TO QDRANT")
print("=" * 80)

vector_store = QdrantVectorStore()


print("\nRecreating collection...")

vector_store.client.recreate_collection(
    collection_name=vector_store.COLLECTION_NAME,
    vectors_config={
        "size": 1024,
        "distance": "Cosine",
    },
)


print("\n" + "=" * 80)
print("INSERTING VECTORS")
print("=" * 80)

vector_store.insert(
    chunks,
    embeddings,
)


print("\n" + "=" * 80)
print("RESULT")
print("=" * 80)

print(
    "Vectors stored:",
    vector_store.count()
)