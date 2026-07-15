"""
Text Chunking Utility.
"""

from __future__ import annotations


class TextChunker:
    """
    Split OCR text into overlapping chunks.
    """

    def __init__(
        self,
        chunk_size: int = 1500,
        overlap: int = 150,
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, text: str) -> list[str]:
        """
        Split text into overlapping chunks.

        Returns:
            List of text chunks.
        """

        if not text.strip():
            return []

        chunks = []

        start = 0

        while start < len(text):

            end = start + self.chunk_size

            chunks.append(text[start:end])

            start += self.chunk_size - self.overlap

        return chunks