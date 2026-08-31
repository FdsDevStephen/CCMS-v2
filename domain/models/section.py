"""
Section — Value Object.

Represents a legal section number (e.g. "79A", "136(2)").
Immutable: once created, a Section never changes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# Valid section suffixes recognized by Karnataka courts
VALID_SUFFIXES: frozenset[str] = frozenset({
    "A", "B", "C", "D",
    "AA", "AB", "AC", "AD",
    "BA", "BB", "BC",
})


@dataclass(frozen=True, order=True)
class Section:
    """
    A single legal section number.

    Examples: "79A", "136(2)", "19(1)(a)"
    """

    number: str = field(compare=True)

    def __post_init__(self) -> None:
        if not self.number or not self.number.strip():
            raise ValueError("Section number cannot be empty.")
        # Freeze the string (already frozen by dataclass, but guard)
        object.__setattr__(self, "number", self.number.strip())

    # ── Queries ────────────────────────────────────────────

    @property
    def base_number(self) -> str:
        """The numeric part before any suffix or sub-section.
        
        '79A' → '79'
        '136(2)' → '136'
        """
        match = re.match(r"(\d+)", self.number)
        return match.group(1) if match else self.number

    @property
    def has_suffix(self) -> bool:
        """True if the section has a letter suffix like A, B, AA."""
        match = re.fullmatch(r"(\d+)([A-Za-z]+)", self.number.replace("-", ""))
        if not match:
            return False
        return match.group(2).upper() in VALID_SUFFIXES

    @property
    def sub_sections(self) -> tuple[str, ...]:
        """Extract sub-section identifiers.
        
        '136(2)' → ('2',)
        '19(1)(a)' → ('1', 'a')
        """
        return tuple(re.findall(r"\(([^)]+)\)", self.number))

    # ── Factory ────────────────────────────────────────────

    @classmethod
    def from_raw(cls, raw: str) -> Section | None:
        """Create a Section from raw OCR text, or None if invalid."""
        if not raw or not raw.strip():
            return None
        cleaned = raw.strip()
        # Basic validation: must start with a digit
        if not re.match(r"^\d", cleaned):
            return None
        return cls(number=cleaned)


@dataclass(frozen=True)
class ActSectionMapping:
    """Maps an Act name to its explicitly referenced sections."""

    act: str
    sections: tuple[Section, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.act or not self.act.strip():
            raise ValueError("Act name cannot be empty.")
        object.__setattr__(self, "act", self.act.strip())

    @property
    def section_numbers(self) -> tuple[str, ...]:
        """Flat list of section number strings."""
        return tuple(s.number for s in self.sections)
