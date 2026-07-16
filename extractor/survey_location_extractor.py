"""
Survey location extractor.
"""

from __future__ import annotations

import re

from extractor.normalizer import Normalizer


class SurveyLocationExtractor:
    SURVEY_PATTERN = re.compile(
        r"""
        (?:
            \bS\.?\s*Nos?\.?|
            \bSy\.?\s*Nos?\.?|
            \bRe-Sy\.?\s*Nos?\.?|
            \bRe-Survey\s*Nos?\.?|
            \bSurvey\s*Nos?\.?|
            \bSurvey\s*Numbers?|
            \bSurvey\s*Number
        )
        [\s:.-]*
        (?P<survey_number>\d+(?:/[A-Za-z0-9*]+)*)
        """,
        re.IGNORECASE | re.VERBOSE,
    )

    NAME_TOKEN = (
        r"(?!village\b|hobli\b|taluk\b|taluka\b|district\b|"
        r"of\b|in\b|at\b|and\b|the\b)"
        r"[A-Za-z][A-Za-z.'-]*"
    )

    VILLAGE_PATTERN = re.compile(
        rf"""
        (?P<village>{NAME_TOKEN}(?:\s+{NAME_TOKEN}){{0,3}})
        \s+
        village
        """,
        re.IGNORECASE | re.VERBOSE,
    )

    HOBLI_PATTERN = re.compile(
        rf"""
        (?:of|in|at|,)
        \s+
        (?P<hobli>{NAME_TOKEN}(?:\s+{NAME_TOKEN}){{0,3}})
        \s+
        Hobli
        """,
        re.IGNORECASE | re.VERBOSE,
    )

    TALUK_PATTERN = re.compile(
        rf"""
        (?:of|in|at|,)
        \s+
        (?P<taluk>{NAME_TOKEN}(?:\s+{NAME_TOKEN}){{0,3}})
        \s+
        Taluk(?:a)?
        """,
        re.IGNORECASE | re.VERBOSE,
    )

    DISTRICT_PATTERN = re.compile(
        rf"""
        (?:of|in|at|,)
        \s+
        (?P<district>{NAME_TOKEN}(?:\s+{NAME_TOKEN}){{0,3}})
        \s+
        District
        """,
        re.IGNORECASE | re.VERBOSE,
    )

    def __init__(self, text: str):
        self.text = text

    def extract(self, survey_numbers: list[str]) -> list[dict]:
        results = []
        seen = set()

        for match in self.SURVEY_PATTERN.finditer(self.text):
            raw_survey = match.group("survey_number")
            normalized = Normalizer.normalize_survey_numbers([raw_survey])

            if not normalized:
                continue

            survey_number = normalized[0]

            if survey_number not in survey_numbers:
                continue

            start = max(0, match.start() - 150)
            end = min(len(self.text), match.end() + 900)

            context = self._clean_spaces(self.text[start:end])
            location = self._extract_location(context)

            if not any(location.values()):
                continue

            if survey_number in seen:
                continue

            seen.add(survey_number)

            results.append(
                {
                    "survey_number": survey_number,
                    **location,
                }
            )

        return results

    def _extract_location(self, context: str) -> dict:
        return {
            "village": self._find_group(self.VILLAGE_PATTERN, context, "village"),
            "hobli": self._find_group(self.HOBLI_PATTERN, context, "hobli"),
            "taluk": self._find_group(self.TALUK_PATTERN, context, "taluk"),
            "district": self._find_group(self.DISTRICT_PATTERN, context, "district"),
        }

    @staticmethod
    def _find_group(pattern: re.Pattern, text: str, group_name: str) -> str | None:
        matches = list(pattern.finditer(text))

        if not matches:
            return None

        return SurveyLocationExtractor._clean_name(matches[-1].group(group_name))

    @staticmethod
    def _clean_name(value: str | None) -> str | None:
        if not value:
            return None

        value = SurveyLocationExtractor._clean_spaces(value)
        value = value.strip(" ,.;:-")

        # OCR/list junk cleanup:
        # "t Hejamadi" -> "Hejamadi"
        # "t Lingammanahalli" -> "Lingammanahalli"
        value = re.sub(
            r"^[a-z]\s+(?=[A-Z])",
            "",
            value,
        )

        return value or None

    @staticmethod
    def _clean_spaces(value: str) -> str:
        return " ".join(value.split())