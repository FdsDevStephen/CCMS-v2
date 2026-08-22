from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
)


class QdrantVectorStore:

    COLLECTION_NAME = "legal_documents"

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
    ):
        self.client = QdrantClient(
            host=host,
            port=port,
        )

    def create_collection(
        self,
        vector_size: int,
    ):

        collections = self.client.get_collections()

        exists = any(
            collection.name == self.COLLECTION_NAME
            for collection in collections.collections
        )

        if not exists:
            self.client.create_collection(
                collection_name=self.COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                ),
            )

    def insert(
        self,
        chunks: list[dict],
        embeddings,
    ):

        points = []

        for index, (chunk, embedding) in enumerate(
            zip(chunks, embeddings)
        ):

            points.append(
                PointStruct(
                    id=index,
                    vector=embedding.tolist(),
                    payload={
                        "document": chunk["document"],
                        "section": chunk["section"],
                        "chunk_id": chunk["chunk_id"],
                        "text": chunk["text"],
                    },
                )
            )

        self.client.upsert(
            collection_name=self.COLLECTION_NAME,
            points=points,
        )

    def count(self):

        result = self.client.count(
            collection_name=self.COLLECTION_NAME,
        )

        return result.count