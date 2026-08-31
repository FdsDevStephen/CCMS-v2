"""
Domain Enums.

Classify documents without depending on any external library.
"""

from enum import Enum, StrEnum


class DocumentType(StrEnum):
    """Legal document types found in Karnataka High Court."""
    WRIT_PETITION = "WP"
    WRIT_APPEAL = "WA"
    CIVIL_MISC = "CM"
    MISC = "MISC"


class Court(StrEnum):
    """Courts this system handles."""
    KARNATAKA_HIGH_COURT = "Karnataka High Court"


class ExtractionStatus(StrEnum):
    """Pipeline processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"
