"""
Prayer — Value Object.

The prayer (relief sought) extracted from a legal document.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Prayer:
    """The prayer / relief sought in the legal document."""

    text: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "text", (self.text or "").strip())

    @property
    def is_empty(self) -> bool:
        return not self.text

    @property
    def word_count(self) -> int:
        return len(self.text.split()) if self.text else 0

    @property
    def has_annexure_reference(self) -> bool:
        """True if the prayer references an annexure."""
        return "annexure" in self.text.lower()

    @classmethod
    def from_raw(cls, text: str) -> Prayer | None:
        """Create from raw text, or None if empty/invalid."""
        if not text or not text.strip():
            return None
        return cls(text=text)
