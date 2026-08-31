"""
Survey — Value Objects.

SurveyNumber: a numeric land parcel identifier.
SurveyLocation: the administrative hierarchy for a survey number.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True, order=True)
class SurveyNumber:
    """A single survey/land parcel number."""

    number: str

    def __post_init__(self) -> None:
        if not self.number or not self.number.strip():
            raise ValueError("Survey number cannot be empty.")
        object.__setattr__(self, "number", self.number.strip())

    @classmethod
    def from_raw(cls, raw: str) -> SurveyNumber | None:
        """Create from raw OCR text, or None if invalid."""
        if not raw or not raw.strip():
            return None
        # Strip common prefixes
        cleaned = re.sub(
            r"^(?:Sy\.?\s*Nos?\.?|S\.?\s*Nos?\.?|Survey\s*Nos?\.?|"
            r"Re-Sy\.?\s*Nos?\.?|Re-Survey\s*Nos?\.?)\s*",
            "",
            raw.strip(),
            flags=re.IGNORECASE,
        ).strip()
        # Must contain at least one digit
        if not re.search(r"\d", cleaned):
            return None
        return cls(number=cleaned)


@dataclass(frozen=True)
class SurveyLocation:
    """Administrative location for a survey number: village → hobli → taluk → district."""

    survey_number: str
    village: str | None = None
    hobli: str | None = None
    taluk: str | None = None
    district: str | None = None

    @property
    def completeness(self) -> int:
        """How many location fields are filled (0-4)."""
        return sum(
            1 for f in (self.village, self.hobli, self.taluk, self.district)
            if f is not None
        )

    @property
    def location_chain(self) -> str:
        """Human-readable location: 'Kenchammanahalli, Anegodu, Davangere'."""
        parts = [p for p in (self.village, self.hobli, self.taluk, self.district) if p]
        return ", ".join(parts)

    def fill_from(self, other: SurveyLocation) -> SurveyLocation:
        """Return a new SurveyLocation filling missing fields from `other`."""
        if self.survey_number != other.survey_number:
            return self
        return SurveyLocation(
            survey_number=self.survey_number,
            village=self.village or other.village,
            hobli=self.hobli or other.hobli,
            taluk=self.taluk or other.taluk,
            district=self.district or other.district,
        )
