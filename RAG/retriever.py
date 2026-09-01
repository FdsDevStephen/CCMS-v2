from __future__ import annotations

from rag.embedding import EmbeddingModel
from rag.vector_store import QdrantVectorStore
from qdrant_client.models import FieldCondition, Filter, MatchValue


class LegalRetriever:
    """Vector retrieval over the Qdrant legal-document collection."""

    def __init__(
        self,
        embedding_model: EmbeddingModel | None = None,
    ):
        self.embedding_model = embedding_model or EmbeddingModel()
        self.vector_store = QdrantVectorStore()

    def search(
        self,
        query: str,
        top_k: int = 5,
        document: str | None = None,
    ):
        if not query or not query.strip():
            return []

        if top_k <= 0:
            return []

        query_embedding = self.embedding_model.encode([query])[0]

        query_filter = None

        if document:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="document",
                        match=MatchValue(value=document),
                    )
                ]
            )

        response = self.vector_store.client.query_points(
            collection_name=self.vector_store.COLLECTION_NAME,
            query=query_embedding.tolist(),
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )

        return response.points
