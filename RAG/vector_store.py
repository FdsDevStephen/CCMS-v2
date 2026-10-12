"""
Vector stores for document chunks.

- InMemoryVectorStore (default): keeps the vectors of recently processed
  documents in memory. CCMS only ever searches the current document's
  chunks, so no database (and no Docker) is needed.
- QdrantVectorStore: the previous Qdrant-backed store (needs a running
  Qdrant server). Select it with VECTOR_STORE=qdrant.

Both expose insert(chunks, embeddings) and
search(query_vector, top_k, document) -> results with .score and .payload.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np


@dataclass
class VectorSearchResult:
    score: float
    payload: dict


def _payload(chunk: dict) -> dict:
    return {
        "document": chunk["document"],
        "section": chunk["section"],
        "chunk_id": chunk["chunk_id"],
        "text": chunk["text"],
    }


class InMemoryVectorStore:
    """Cosine-similarity search over the chunks of recent documents."""

    def __init__(self, max_documents: int = 20):
        self.max_documents = max_documents
        self._documents: OrderedDict[str, tuple[list[dict], np.ndarray]] = (
            OrderedDict()
        )
        self._lock = threading.Lock()

    def insert(
        self,
        chunks: list[dict],
        embeddings,
    ):
        vectors = np.asarray(embeddings, dtype=np.float32)

        if not len(chunks):
            return

        # Normalise so a dot product is the cosine similarity.
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.clip(norms, 1e-12, None)

        by_document: dict[str, list[int]] = {}

        for index, chunk in enumerate(chunks):
            by_document.setdefault(chunk["document"], []).append(index)

        with self._lock:
            for document, indexes in by_document.items():
                # Re-processing a document replaces its old chunks.
                self._documents.pop(document, None)
                self._documents[document] = (
                    [_payload(chunks[i]) for i in indexes],
                    vectors[indexes],
                )

            while len(self._documents) > self.max_documents:
                self._documents.popitem(last=False)

    def search(
        self,
        query_vector,
        top_k: int = 5,
        document: str | None = None,
    ) -> list[VectorSearchResult]:
        query = np.asarray(query_vector, dtype=np.float32)
        query = query / max(float(np.linalg.norm(query)), 1e-12)

        with self._lock:
            if document is not None:
                entries = (
                    [self._documents[document]]
                    if document in self._documents
                    else []
                )
            else:
                entries = list(self._documents.values())

        results: list[VectorSearchResult] = []

        for payloads, vectors in entries:
            scores = vectors @ query

            results.extend(
                VectorSearchResult(float(score), payload)
                for score, payload in zip(scores, payloads)
            )

        results.sort(key=lambda result: result.score, reverse=True)

        return results[:top_k]

    def count(self) -> int:
        with self._lock:
            return sum(len(p) for p, _ in self._documents.values())


class QdrantVectorStore:

    COLLECTION_NAME = "legal_documents"

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
    ):
        from qdrant_client import QdrantClient

        self.client = QdrantClient(
            host=host,
            port=port,
        )

    def create_collection(
        self,
        vector_size: int,
    ):
        from qdrant_client.models import Distance, VectorParams

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
        from qdrant_client.models import PointStruct

        points = []

        for index, (chunk, embedding) in enumerate(
            zip(chunks, embeddings)
        ):

            points.append(
                PointStruct(
                    id=index,
                    vector=embedding.tolist(),
                    payload=_payload(chunk),
                )
            )

        self.client.upsert(
            collection_name=self.COLLECTION_NAME,
            points=points,
        )

    def search(
        self,
        query_vector,
        top_k: int = 5,
        document: str | None = None,
    ):
        from qdrant_client.models import FieldCondition, Filter, MatchValue

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

        response = self.client.query_points(
            collection_name=self.COLLECTION_NAME,
            query=np.asarray(query_vector).tolist(),
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )

        return response.points

    def count(self):

        result = self.client.count(
            collection_name=self.COLLECTION_NAME,
        )

        return result.count


_default_store = None
_default_lock = threading.Lock()


def get_vector_store():
    """The configured store (VECTOR_STORE=memory|qdrant), shared per process."""
    global _default_store

    with _default_lock:
        if _default_store is None:
            from config.settings import QDRANT_HOST, QDRANT_PORT, VECTOR_STORE

            if VECTOR_STORE.lower() == "qdrant":
                _default_store = QdrantVectorStore(
                    host=QDRANT_HOST,
                    port=QDRANT_PORT,
                )
            else:
                _default_store = InMemoryVectorStore()

        return _default_store
