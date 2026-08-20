from sentence_transformers import SentenceTransformer

print("1. Starting...")

model = SentenceTransformer("BAAI/bge-m3")

print("2. Model loaded!")

embedding = model.encode(
    "Survey Number 171/1",
    normalize_embeddings=True
)

print("3. Embedding created!")
print("Shape:", embedding.shape)