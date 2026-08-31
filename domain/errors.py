"""
Domain Errors.

All domain-specific exceptions inherit from DomainError,
making it easy to catch business logic failures separately
from infrastructure errors.
"""


class DomainError(Exception):
    """Base class for all domain errors."""
    pass


class ExtractionError(DomainError):
    """Raised when the extraction pipeline fails."""
    pass


class OCRError(DomainError):
    """Raised when OCR processing fails."""
    pass


class LLMError(DomainError):
    """Raised when the LLM fails to respond or parse."""
    pass


class ValidationError(DomainError):
    """Raised when extracted data fails validation rules."""
    pass


class DocumentNotFoundError(DomainError):
    """Raised when a document cannot be located."""
    pass
