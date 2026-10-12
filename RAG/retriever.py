from __future__ import annotations

from rag.embedding import EmbeddingModel
from rag.vector_store import get_vector_store


class LegalRetriever:
    """Vector retrieval over the configured vector store."""

    def __init__(
        self,
        embedding_model: EmbeddingModel | None = None,
        vector_store=None,
    ):
        self.embedding_model = embedding_model or EmbeddingModel()
        self.vector_store = vector_store or get_vector_store()

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

        return self.vector_store.search(
            query_embedding,
            top_k=top_k,
            document=document,
        )
