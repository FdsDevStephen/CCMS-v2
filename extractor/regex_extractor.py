"""
Regex Extraction Pipeline.
"""

from __future__ import annotations

from extractor.normalizer import Normalizer
from extractor.section_extractor import SectionExtractor
from extractor.survey_extractor import SurveyExtractor


class RegexExtractor:
    """
    Runs all regex-based extractors.
    """

    def __init__(self, text: str):
        self.text = text

        self.survey_extractor = SurveyExtractor(text)
        self.section_extractor = SectionExtractor(text)

    def extract_survey_numbers(self) -> list[str]:
        """
        Extract Survey Numbers.
        """
        return self.survey_extractor.extract()

    def extract_sections(self) -> list[str]:
        """
        Extract Sections.
        """
        return self.section_extractor.extract()

    def extract_all(self) -> dict:
        """
        Run all regex extractors.
        """
        raw_survey_numbers = self.extract_survey_numbers()

        survey_numbers = Normalizer.normalize_survey_numbers(
            raw_survey_numbers
        )

        return {
            "survey_numbers": survey_numbers,
            "sections": self.extract_sections(),
        }