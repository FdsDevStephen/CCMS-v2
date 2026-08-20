from __future__ import annotations

import cv2
import numpy as np
import pytesseract

from pathlib import Path
from pdf2image import convert_from_path, convert_from_bytes


class OCRProcessor:
    """
    OCR Processor for PDF documents using Tesseract.

    Supports:
    - OCR from a PDF file path
    - OCR directly from PDF bytes (for Streamlit)
    """

    def __init__(
        self,
        poppler_path: str,
        output_folder: Path,
        tesseract_path: str | None = None,
        first_page: int | None = None,
        last_page: int | None = None,
    ) -> None:

        self.poppler_path = poppler_path
        self.output_folder = Path(output_folder)

        self.first_page = first_page
        self.last_page = last_page

        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path

        self.output_folder.mkdir(parents=True, exist_ok=True)

    # ==========================================================
    # Internal OCR Logic
    # ==========================================================

    def _ocr_images(self, images) -> str:
        """
        Perform OCR on a list of PIL images.

        Args:
            images: List of PIL Images.

        Returns:
            OCR extracted text.
        """

        pages = []

        for page_number, image in enumerate(
            images,
            start=self.first_page or 1,
        ):

            image = np.array(image)

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            gray = cv2.medianBlur(gray, 3)

            text = pytesseract.image_to_string(
                gray,
                lang="eng",
                config="--oem 3 --psm 6",
            )

            pages.append(
                f"\n\n========== PAGE {page_number} ==========\n\n{text}"
            )

        return "".join(pages)

    # ==========================================================
    # OCR From PDF Path
    # ==========================================================

    def extract_text(self, pdf_path: str | Path) -> str:
        """
        Perform OCR on a PDF file.

        Args:
            pdf_path: Path to PDF.

        Returns:
            OCR extracted text.
        """

        images = convert_from_path(
            pdf_path,
            poppler_path=self.poppler_path,
            first_page=self.first_page,
            last_page=self.last_page,
        )

        return self._ocr_images(images)

    # ==========================================================
    # OCR From PDF Bytes (Streamlit)
    # ==========================================================

    def extract_text_from_bytes(self, pdf_bytes: bytes) -> str:
        """
        Perform OCR on PDF bytes.

        Args:
            pdf_bytes: PDF bytes.

        Returns:
            OCR extracted text.
        """

        images = convert_from_bytes(
            pdf_bytes,
            poppler_path=self.poppler_path,
            first_page=self.first_page,
            last_page=self.last_page,
        )

        return self._ocr_images(images)

    # ==========================================================
    # Save OCR Text
    # ==========================================================

    def save_text(self, pdf_path: str | Path, text: str) -> Path:
        """
        Save OCR text.

        Args:
            pdf_path: Original PDF path.
            text: OCR text.

        Returns:
            Saved text file path.
        """

        pdf_path = Path(pdf_path)

        output_path = self.output_folder / f"{pdf_path.stem}.txt"

        output_path.write_text(
            text,
            encoding="utf-8",
        )

        return output_path

    # ==========================================================
    # Complete Pipeline (File)
    # ==========================================================

    def process(self, pdf_path: str | Path) -> tuple[str, Path]:
        """
        OCR pipeline using a PDF file.

        Returns:
            (OCR text, saved text path)
        """

        text = self.extract_text(pdf_path)

        txt_path = self.save_text(pdf_path, text)

        return text, txt_path


    def process_bytes(self, pdf_bytes: bytes) -> str:
        """
        OCR pipeline using PDF bytes.

        Returns:
            OCR text only.
        """

        return self.extract_text_from_bytes(pdf_bytes)