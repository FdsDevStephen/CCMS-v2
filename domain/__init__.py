"""Domain layer — pure business logic with ZERO external dependencies."""

from domain.models.section import Section, ActSectionMapping
from domain.models.act import Act
from domain.models.survey import SurveyNumber, SurveyLocation
from domain.models.prayer import Prayer
from domain.models.document import LegalDocument
from domain.models.chunk import DocumentChunk
from domain.models.extraction import ExtractionResult, ExtractionTimings
from domain.enums import DocumentType
from domain.errors import DomainError, ExtractionError, OCRError, LLMError

__all__ = [
    "Section",
    "ActSectionMapping",
    "Act",
    "SurveyNumber",
    "SurveyLocation",
    "Prayer",
    "LegalDocument",
    "DocumentChunk",
    "ExtractionResult",
    "ExtractionTimings",
    "DocumentType",
    "DomainError",
    "ExtractionError",
    "OCRError",
    "LLMError",
]
