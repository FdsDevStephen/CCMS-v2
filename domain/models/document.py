"""
Document — Aggregate Root.

Represents a legal document being processed.
This is the main entity that carries identity through the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from domain.enums import DocumentType, ExtractionStatus


def _detect_document_type(stem: str) -> DocumentType:
    """Detect document type from filename stem like 'WP-19110-2024-B'."""
    upper = stem.upper()
    if upper.startswith("WP"):
        return DocumentType.WRIT_PETITION
    if upper.startswith("WA"):
        return DocumentType.WRIT_APPEAL
    return DocumentType.MISC


@dataclass
class LegalDocument:
    """
    The aggregate root for a legal document.

    Carries identity and state through the entire pipeline:
    PDF → OCR → Extraction → RAG → Export.
    """

    # ── Identity ──────────────────────────────────────────

    case_number: str
    filename: str = ""
    document_type: DocumentType = field(init=False)

    # ── Raw data ──────────────────────────────────────────

    pdf_bytes: bytes = field(default=b"", repr=False)
    pdf_path: Path | None = None
    ocr_text: str = ""
    txt_path: Path | None = None

    # ── State ─────────────────────────────────────────────

    status: ExtractionStatus = ExtractionStatus.PENDING
    error_message: str = ""

    def __post_init__(self) -> None:
        if not self.case_number:
            self.case_number = Path(self.filename).stem if self.filename else "unknown"
        self.document_type = _detect_document_type(self.case_number)

    # ── Queries ────────────────────────────────────────────

    @property
    def has_ocr_text(self) -> bool:
        return bool(self.ocr_text)

    @property
    def has_pdf(self) -> bool:
        return bool(self.pdf_bytes) or (self.pdf_path is not None and self.pdf_path.exists())

    @property
    def display_name(self) -> str:
        """Human-readable name: 'WP-19110-2024-B'."""
        return self.case_number

    # ── Commands ───────────────────────────────────────────

    def mark_processing(self) -> None:
        self.status = ExtractionStatus.PROCESSING

    def mark_success(self) -> None:
        self.status = ExtractionStatus.SUCCESS

    def mark_failed(self, error: str) -> None:
        self.status = ExtractionStatus.FAILED
        self.error_message = error

    # ── Factory ────────────────────────────────────────────

    @classmethod
    def from_pdf(cls, pdf_bytes: bytes, filename: str) -> LegalDocument:
        """Create from uploaded PDF bytes."""
        return cls(
            case_number=Path(filename).stem,
            filename=filename,
            pdf_bytes=pdf_bytes,
        )

    @classmethod
    def from_file(cls, pdf_path: str | Path) -> LegalDocument:
        """Create from a PDF file on disk."""
        pdf_path = Path(pdf_path)
        return cls(
            case_number=pdf_path.stem,
            filename=pdf_path.name,
            pdf_path=pdf_path,
        )
