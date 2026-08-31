"""
DocumentChunk — Value Object.

A chunk of text from a legal document, ready for embedding and retrieval.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DocumentChunk:
    """A single chunk of legal text for embedding and RAG retrieval."""

    document: str          # case_number / stem
    section: str           # e.g. "pages_2_13", "full_prayer"
    chunk_id: int          # sequence within the section
    text: str              # the actual text content

    # ── Derived ──────────────────────────────────────────

    @property
    def key(self) -> tuple[str, str, int]:
        """Unique identifier for deduplication across vector + BM25."""
        return (self.document, self.section, self.chunk_id)

    @property
    def word_count(self) -> int:
        return len(self.text.split()) if self.text else 0

    @property
    def token_estimate(self) -> int:
        """Rough token estimate (words × 1.3)."""
        return int(self.word_count * 1.3)

    # ── Factory ────────────────────────────────────────────

    @classmethod
    def from_dict(cls, data: dict) -> DocumentChunk:
        """Create from the dict format used by LegalTextChunker."""
        return cls(
            document=str(data.get("document", "")),
            section=str(data.get("section", "")),
            chunk_id=int(data.get("chunk_id", 0)),
            text=str(data.get("text", "")),
        )

    def to_dict(self) -> dict:
        """Convert to the dict format expected by downstream services."""
        return {
            "document": self.document,
            "section": self.section,
            "chunk_id": self.chunk_id,
            "text": self.text,
        }
