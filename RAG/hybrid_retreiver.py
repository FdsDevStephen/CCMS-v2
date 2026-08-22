from __future__ import annotations

import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from RAG.retriever import LegalRetriever


@dataclass
class HybridResult:
    score: float
    payload: dict
    vector_score: float = 0.0
    bm25_score: float = 0.0
    legal_score: float = 0.0


class HybridRetriever:
    """
    Hybrid legal retriever.

    Combines:
    - dense vector similarity
    - BM25 lexical similarity
    - lightweight legal-reference relevance

    The final score is normalized so that one raw BM25/legal score cannot
    overwhelm the vector score.
    """

    def __init__(
        self,
        vector_retriever: LegalRetriever | None = None,
        vector_weight: float = 0.55,
        bm25_weight: float = 0.30,
        legal_weight: float = 0.15,
    ):
        total = vector_weight + bm25_weight + legal_weight
        if total <= 0:
            raise ValueError("Hybrid weights must sum to a positive value.")

        self.vector_weight = vector_weight / total
        self.bm25_weight = bm25_weight / total
        self.legal_weight = legal_weight / total

        self.vector_retriever = vector_retriever or LegalRetriever()
        self.bm25: BM25Okapi | None = None
        self.chunks: list[dict] = []

    def build_bm25(self, chunks: list[dict]) -> None:
        self.chunks = list(chunks)

        documents = [str(chunk.get("text", "")) for chunk in self.chunks]
        tokenized_documents = [self._tokenize(text) for text in documents]

        if not tokenized_documents:
            self.bm25 = None
            return

        self.bm25 = BM25Okapi(tokenized_documents)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(
            r"u/s|section|sections|act|acts|provision|provisions|"
            r"\d+(?:\([a-zA-Z0-9]+\))*|"
            r"[a-zA-Z]+",
            text.lower(),
        )

    @staticmethod
    def _chunk_key(chunk: dict) -> tuple:
        """
        chunk_id alone is not globally unique.

        Include document + section so chunks from different documents/sections
        cannot overwrite each other during hybrid merging.
        """
        return (
            chunk.get("document"),
            chunk.get("section"),
            chunk.get("chunk_id"),
        )

    def _legal_reference_score(self, text: str, query: str) -> float:
        text_lower = text.lower()
        query_lower = query.lower()

        score = 0.0

        if re.search(r"\b(?:section|sections|u/s)\b", text_lower):
            score += 1.0

        if re.search(r"\b(?:act|acts)\b", text_lower):
            score += 1.0

        if re.search(r"\b(?:under\s+section|section\s+\d|u/s\s*\d)\b", text_lower):
            score += 1.0

        query_sections = set(
            re.findall(r"\b\d+(?:\([a-zA-Z0-9]+\))*\b", query_lower)
        )
        text_sections = set(
            re.findall(r"\b\d+(?:\([a-zA-Z0-9]+\))*\b", text_lower)
        )

        if query_sections & text_sections:
            score += 3.0

        if re.search(r"\bsection\b.*\bact\b", text_lower, re.DOTALL):
            score += 1.0

        return score

    @staticmethod
    def _normalize(values: list[float]) -> list[float]:
        if not values:
            return []

        minimum = min(values)
        maximum = max(values)

        if maximum <= minimum:
            return [1.0 if maximum > 0 else 0.0 for _ in values]

        return [(value - minimum) / (maximum - minimum) for value in values]

    def search(
        self,
        query: str,
        top_k: int = 10,
        document: str | None = None,
        candidate_k: int | None = None,
    ) -> list[HybridResult]:
        if not query or not query.strip():
            return []

        if top_k <= 0:
            return []

        if self.bm25 is None:
            raise RuntimeError("BM25 index has not been built.")

        candidate_k = candidate_k or max(top_k * 3, 20)

        vector_results = self.vector_retriever.search(
            query=query,
            top_k=candidate_k,
            document=document,
        )

        query_tokens = self._tokenize(query)
        bm25_scores = self.bm25.get_scores(query_tokens)

        bm25_indexes = sorted(
            range(len(bm25_scores)),
            key=bm25_scores.__getitem__,
            reverse=True,
        )

        candidates: dict[tuple, dict] = {}

        for result in vector_results:
            payload = result.payload or {}
            if not payload:
                continue

            if document and payload.get("document") != document:
                continue

            key = self._chunk_key(payload)

            candidates[key] = {
                "chunk": payload,
                "vector_score": float(result.score),
                "bm25_score": 0.0,
            }

        added = 0
        for index in bm25_indexes:
            chunk = self.chunks[index]

            if document and chunk.get("document") != document:
                continue

            key = self._chunk_key(chunk)

            if key not in candidates:
                candidates[key] = {
                    "chunk": chunk,
                    "vector_score": 0.0,
                    "bm25_score": float(bm25_scores[index]),
                }
            else:
                candidates[key]["bm25_score"] = float(bm25_scores[index])

            added += 1
            if added >= candidate_k:
                break

        if not candidates:
            return []

        candidate_list = list(candidates.values())

        vector_values = [item["vector_score"] for item in candidate_list]
        bm25_values = [item["bm25_score"] for item in candidate_list]
        legal_values = [
            self._legal_reference_score(item["chunk"].get("text", ""), query)
            for item in candidate_list
        ]

        normalized_vector = self._normalize(vector_values)
        normalized_bm25 = self._normalize(bm25_values)
        normalized_legal = self._normalize(legal_values)

        scored: list[HybridResult] = []

        for index, item in enumerate(candidate_list):
            final_score = (
                normalized_vector[index] * self.vector_weight
                + normalized_bm25[index] * self.bm25_weight
                + normalized_legal[index] * self.legal_weight
            )

            scored.append(
                HybridResult(
                    score=float(final_score),
                    payload=item["chunk"],
                    vector_score=item["vector_score"],
                    bm25_score=item["bm25_score"],
                    legal_score=legal_values[index],
                )
            )

        scored.sort(key=lambda result: result.score, reverse=True)
        return scored[:top_k]
