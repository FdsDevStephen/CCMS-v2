from __future__ import annotations
from pathlib import Path

from config import (
    FIRST_PAGE,
    LAST_PAGE,
    OUTPUT_TEXT_FOLDER,
    POPPLER_PATH,
    TESSERACT_PATH,
)

from ocr import OCRProcessor

from extractor.text_chunker import TextChunker
from extractor.llm.factory import get_llm_client
from extractor.parser import LLMResponseParser
from extractor.prompts import build_act_extraction_prompt
from extractor.regex_extractor import RegexExtractor
from extractor.utils import get_case_number_from_filename
from extractor.validator import Validator


class LegalExtractor:
    """
    Main Legal Information Extraction Pipeline.
    """

    def __init__(self):

        self.llm = get_llm_client()

        self.chunker = TextChunker(
            chunk_size=3000,
            overlap=300,
        )

        self.ocr = OCRProcessor(
            poppler_path=POPPLER_PATH,
            output_folder=OUTPUT_TEXT_FOLDER,
            tesseract_path=TESSERACT_PATH,
            first_page=FIRST_PAGE,
            last_page=LAST_PAGE,
        )

    def extract(self, pdf_path: str | Path) -> dict:

        pdf_path = Path(pdf_path)

        # ======================================================
        # OCR
        # ======================================================

        text, _ = self.ocr.process(pdf_path)

        # ======================================================
        # Case Number
        # ======================================================

        case_number = get_case_number_from_filename(pdf_path)

        # ======================================================
        # Regex Extraction
        # ======================================================

        regex = RegexExtractor(text)

        regex_result = regex.extract_all()
        
        print("=" * 80)
        print("RAW SURVEY NUMBERS")
        print("=" * 80)

        for survey in regex_result["survey_numbers"]:
            print(repr(survey))

        regex_result["case_number"] = case_number

        # ======================================================
        # Chunk OCR Text
        # ======================================================

        chunks = self.chunker.split(text)

        all_acts = []

        print(f"\nTotal Chunks : {len(chunks)}\n")

        # ======================================================
        # LLM Extraction (Chunk by Chunk)
        # ======================================================

        for index, chunk in enumerate(chunks, start=1):

            prompt = build_act_extraction_prompt(chunk)

            try:

                llm_response = self.llm.generate(prompt)

                result = LLMResponseParser.parse(llm_response)

                acts = result.get("acts", [])

                print(f"Acts Found : {acts}")

                all_acts.extend(acts)

            except Exception as e:

                print(f"Chunk {index} Failed")

                print(e)

        # ======================================================
        # Merge LLM Result
        # ======================================================

        llm_result = {
            "acts": all_acts
        }

        # ======================================================
        # Final Output
        # ======================================================

        final_result = {
            "case_number": case_number,
            "survey_numbers": regex_result["survey_numbers"],
            "sections": regex_result["sections"],
            "acts": llm_result["acts"],
            "primary_act": None,
        }

        final_result = Validator.validate(final_result)
        
        print("=" * 80)
        print("FINAL SURVEY NUMBERS")
        print("=" * 80)

        for survey in final_result["survey_numbers"]:
            print(repr(survey))

        return final_result