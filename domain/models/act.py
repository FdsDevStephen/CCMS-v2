"""
Act — Value Object.

Represents a named legal statute (e.g. "Karnataka Land Reforms Act, 1961").
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class Act:
    """A single named Act, exactly as written in the document."""

    name: str

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("Act name cannot be empty.")
        object.__setattr__(self, "name", self.name.strip())

    @property
    def year(self) -> str | None:
        """Extract the year if present: 'Act, 1961' → '1961'."""
        import re
        match = re.search(r",?\s*(\d{4})\s*$", self.name)
        return match.group(1) if match else None

    @property
    def short_name(self) -> str:
        """Name without the year: 'Karnataka Land Reforms Act, 1961' → 'Karnataka Land Reforms Act'."""
        import re
        return re.sub(r",?\s*\d{4}\s*$", "", self.name).strip()

    @classmethod
    def from_raw(cls, name: str) -> Act | None:
        """Create an Act from raw text, or None if invalid."""
        if not name or not name.strip():
            return None
        cleaned = name.strip()
        # Must contain the word "Act" to be valid
        if "act" not in cleaned.lower():
            return None
        return cls(name=cleaned)
