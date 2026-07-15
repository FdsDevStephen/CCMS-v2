"""
Regex Extraction Pipeline.
"""

from __future__ import annotations

from extractor.survey_extractor import SurveyExtractor
from extractor.section_extractor import SectionExtractor


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

        return {
            "survey_numbers": self.extract_survey_numbers(),
            "sections": self.extract_sections(),
        }