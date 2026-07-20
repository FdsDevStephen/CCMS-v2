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
from extractor.normalizer import Normalizer
from extractor.survey_location import SurveyLocationExtractor
from concurrent.futures import ThreadPoolExecutor, as_completed


class LegalExtractor:
    """
    Main Legal Information Extraction Pipeline.
    """

    def __init__(self):

        self.llm = get_llm_client()

        self.chunker = TextChunker(
            chunk_size=6000,
            overlap=150,
        )

        self.ocr = OCRProcessor(
            poppler_path=POPPLER_PATH,
            output_folder=OUTPUT_TEXT_FOLDER,
            tesseract_path=TESSERACT_PATH,
            first_page=FIRST_PAGE,
            last_page=LAST_PAGE,
        )

    def _extract_acts_from_chunk(
        self,
        chunk: str,
        sections: list[str],
    ) -> tuple[list, list]:

        prompt = build_act_extraction_prompt(
            chunk,
            sections,
        )

        llm_response = self.llm.generate(prompt)

        result = LLMResponseParser.parse(llm_response)

        acts = result.get("acts", [])
        mappings = result.get("act_section_mapping", [])

        return acts, mappings

    def extract(self, pdf_path: str | Path) -> dict:

        pdf_path = Path(pdf_path)

        text, _ = self.ocr.process(pdf_path)

        case_number = get_case_number_from_filename(pdf_path)


        regex = RegexExtractor(text)

        regex_result = regex.extract_all()

        location_extractor = SurveyLocationExtractor(text)

        survey_locations = location_extractor.extract(regex_result["survey_numbers"])

        regex_result["survey_locations"] = survey_locations

        regex_result["sections"] = Normalizer.normalize_sections(
            regex_result["sections"]
        )

        regex_result["case_number"] = case_number

        print("=" * 80)
        print("RAW SURVEY NUMBERS")
        print("=" * 80)

        for survey in regex_result["survey_numbers"]:
            print(repr(survey))

        chunks = self.chunker.split(text)

        all_acts = []
        all_mappings = []

        print(f"\nTotal Chunks : {len(chunks)}\n")

        MAX_WORKERS = 4

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:

            futures = {
                executor.submit(
                    self._extract_acts_from_chunk,
                    chunk,
                    regex_result["sections"],
                ): index

                for index, chunk in enumerate(chunks, start=1)
            }

            for future in as_completed(futures):

                chunk_index = futures[future]

                try:
                    acts, mappings = future.result()

                    all_acts.extend(acts)
                    all_mappings.extend(mappings)

                    print(f"Chunk {chunk_index} Completed")

                except Exception as e:

                    print(f"Chunk {chunk_index} Failed")

                    print(e)



        llm_result = {
            "acts": Normalizer.normalize_acts(all_acts),
            "act_section_mapping": Normalizer.normalize_act_section_mapping(
                all_mappings
            ),
        }


        final_result = {
            "case_number": case_number,
            "survey_numbers": regex_result["survey_numbers"],
            "survey_locations": regex_result["survey_locations"],
            "sections": regex_result["sections"],
            "acts": llm_result["acts"],
            "act_section_mapping": llm_result["act_section_mapping"],
            "primary_act": None,
        }

        print("=" * 80)
        print("SURVEY LOCATIONS BEFORE VALIDATOR")
        print("=" * 80)
        print(regex_result["survey_locations"])

        final_result = Validator.validate(final_result)

        print("=" * 80)
        print("FINAL SURVEY NUMBERS")
        print("=" * 80)

        for survey in final_result["survey_numbers"]:
            print(repr(survey))

        print("=" * 80)
        print("FINAL RESULT")
        print("=" * 80)

        import json

        print(json.dumps(final_result, indent=4))

        return final_result
