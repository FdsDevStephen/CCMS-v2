"""
ExtractionResult — Aggregate.

The final output of the entire extraction pipeline.
Assembles all extracted data into a single typed object.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from domain.models.section import Section, ActSectionMapping
from domain.models.act import Act
from domain.models.survey import SurveyNumber, SurveyLocation
from domain.models.prayer import Prayer
from domain.enums import ExtractionStatus


@dataclass
class ExtractionTimings:
    """Tracks how long each pipeline stage took (in seconds)."""

    ocr: float = 0.0
    legal_extraction: float = 0.0
    prayer: float = 0.0
    embedding_model: float = 0.0
    chunking: float = 0.0
    embeddings: float = 0.0
    vector_store: float = 0.0
    hybrid_retriever: float = 0.0
    act_extraction: float = 0.0
    survey_locations: float = 0.0
    total: float = 0.0

    def as_dict(self) -> dict[str, float]:
        """Convert to the format app.py used: {'OCR': 1.2, 'Prayer': 0.3, ...}"""
        return {
            "OCR": self.ocr,
            "Legal Extraction": self.legal_extraction,
            "Prayer": self.prayer,
            "BGE-M3": self.embedding_model,
            "Chunking": self.chunking,
            "Embeddings": self.embeddings,
            "Qdrant": self.vector_store,
            "BM25 + Hybrid": self.hybrid_retriever,
            "Act Extraction": self.act_extraction,
            "Survey Locations": self.survey_locations,
            "Total": self.total,
        }


@dataclass
class ExtractionResult:
    """
    The final output of the extraction pipeline.

    This is the typed replacement for the raw dict that app.py currently returns.
    """

    # ── Identity ──────────────────────────────────────────

    case_number: str = ""

    # ── Extracted data ────────────────────────────────────

    sections: list[Section] = field(default_factory=list)
    acts: list[Act] = field(default_factory=list)
    act_section_mapping: list[ActSectionMapping] = field(default_factory=list)
    survey_numbers: list[SurveyNumber] = field(default_factory=list)
    survey_locations: list[SurveyLocation] = field(default_factory=list)
    prayer: Prayer | None = None
    primary_act: Act | None = None

    # ── Metadata ──────────────────────────────────────────

    status: ExtractionStatus = ExtractionStatus.PENDING
    timings: ExtractionTimings = field(default_factory=ExtractionTimings)

    # ── Convenience queries ───────────────────────────────

    @property
    def section_numbers(self) -> list[str]:
        """Flat list of section number strings."""
        return [s.number for s in self.sections]

    @property
    def act_names(self) -> list[str]:
        """Flat list of act name strings."""
        return [a.name for a in self.acts]

    @property
    def survey_number_strings(self) -> list[str]:
        """Flat list of survey number strings."""
        return [s.number for s in self.survey_numbers]

    @property
    def prayer_text(self) -> str:
        """Prayer text or empty string."""
        return self.prayer.text if self.prayer else ""

    # ── Conversion ────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to the raw dict format expected by:
        - Excel export in app.py
        - JSON serialization
        - Backward-compatible code
        """
        return {
            "case_number": self.case_number,
            "sections": self.section_numbers,
            "acts": [a.name for a in self.acts],
            "act_section_mapping": [
                {"act": m.act, "sections": list(m.section_numbers)}
                for m in self.act_section_mapping
            ],
            "survey_numbers": self.survey_number_strings,
            "survey_locations": [
                {
                    "survey_number": loc.survey_number,
                    "village": loc.village,
                    "hobli": loc.hobli,
                    "taluk": loc.taluk,
                    "district": loc.district,
                }
                for loc in self.survey_locations
            ],
            "prayer": self.prayer_text,
            "primary_act": self.primary_act.name if self.primary_act else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ExtractionResult:
        """Create from the raw dict format (backward compatibility)."""
        sections = [
            Section(number=s) for s in data.get("sections", [])
            if s and isinstance(s, str)
        ]
        acts = [
            Act(name=a) for a in data.get("acts", [])
            if a and isinstance(a, str)
        ]
        mappings = []
        for m in data.get("act_section_mapping", []):
            if isinstance(m, dict):
                act_name = m.get("act", "")
                sec_numbers = m.get("sections", [])
                sec_objs = tuple(Section(number=s) for s in sec_numbers if s)
                if act_name and sec_objs:
                    mappings.append(ActSectionMapping(act=act_name, sections=sec_objs))

        survey_numbers = [
            SurveyNumber(number=s) for s in data.get("survey_numbers", [])
            if s and isinstance(s, str)
        ]

        survey_locations = []
        for loc in data.get("survey_locations", []):
            if isinstance(loc, dict):
                survey_locations.append(SurveyLocation(
                    survey_number=loc.get("survey_number", ""),
                    village=loc.get("village"),
                    hobli=loc.get("hobli"),
                    taluk=loc.get("taluk"),
                    district=loc.get("district"),
                ))

        prayer = Prayer.from_raw(data.get("prayer", ""))

        primary_act_name = data.get("primary_act")
        primary_act = Act(name=primary_act_name) if primary_act_name else None

        return cls(
            case_number=data.get("case_number", ""),
            sections=sections,
            acts=acts,
            act_section_mapping=mappings,
            survey_numbers=survey_numbers,
            survey_locations=survey_locations,
            prayer=prayer,
            primary_act=primary_act,
        )
