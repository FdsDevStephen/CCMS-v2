from __future__ import annotations

from pathlib import Path
import json

from extractor.section_extractor import SectionExtractor
from extractor.survey_extractor import SurveyExtractor
from extractor.normalizer import Normalizer
from extractor.utils import get_case_number_from_filename
from extractor.validator import Validator


class LegalExtractor:
    """
    Legal Information Extraction Pipeline.

    Input:
        Already extracted OCR text containing:
        - Synopsis
        - Brief Facts of the Case
        - Prayer

    Output:
        - Case Number
        - Sections
        - Survey Numbers
    """

    def __init__(self):
        pass

    def extract(
        self,
        text: str,
        case_number: str = "",
    ) -> dict:

        # ---------------------------------------------------------
        # 1. Extract Sections
        # ---------------------------------------------------------

        print(
            ">>> EXTRACTING SECTIONS <<<"
        )

        section_extractor = SectionExtractor(
            text
        )

        sections = section_extractor.extract()

        sections = Normalizer.normalize_sections(
            sections
        )

        # ---------------------------------------------------------
        # 2. Extract Survey Numbers
        # ---------------------------------------------------------

        print(
            ">>> EXTRACTING SURVEY NUMBERS <<<"
        )

        survey_extractor = SurveyExtractor(
            text
        )

        survey_numbers = survey_extractor.extract()

        # ---------------------------------------------------------
        # 3. Assemble Result
        # ---------------------------------------------------------

        final_result = {

            "case_number": case_number,

            "survey_numbers": survey_numbers,

            "survey_locations": [],

            "sections": sections,

            "acts": [],

            "act_section_mapping": [],

            "primary_act": None,
        }

        # ---------------------------------------------------------
        # 4. Validate
        # ---------------------------------------------------------

        final_result = Validator.validate(
            final_result
        )

        # ---------------------------------------------------------
        # 5. Print Result
        # ---------------------------------------------------------

        print(
            "=" * 80
        )

        print(
            "FINAL RESULT"
        )

        print(
            "=" * 80
        )

        print(
            json.dumps(
                final_result,
                indent=4,
            )
        )

        return final_result

    def extract_from_file(
        self,
        txt_path: str | Path,
    ) -> dict:

        txt_path = Path(
            txt_path
        )

        text = txt_path.read_text(
            encoding="utf-8"
        )

        case_number = (
            txt_path.stem
        )

        return self.extract(
            text=text,
            case_number=case_number,
        )