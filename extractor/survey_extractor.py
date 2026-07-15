"""
Survey Number Extractor.
"""

from __future__ import annotations

import re


class SurveyExtractor:
    CASE_NUMBER_PATTERN = re.compile(
        r"""
        \b
        [O0]\s*[\.,]?\s*S\s*\.?\s*
        Nos?\.?
        \s*[:.-]?\s*
        \d+
        (?:/\d+)?
        """,
        re.IGNORECASE | re.VERBOSE,
    )

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

(?:i\.?\s*e\.?,?\s*)?

(
    \d+
    (?-i:(?:(?<!\.)\s+(?:A|B|C|AA|AB|AC|AD))?)
    (?:/[A-Za-z0-9*]+)*

    (?:
        \s*
        (?:,|&|\band\b)
        \s*

        \d+
        (?-i:(?:(?<!\.)\s+(?:A|B|C|AA|AB|AC|AD))?)
        (?:/[A-Za-z0-9*]+)*
    )*
)
""",
        re.IGNORECASE | re.VERBOSE,
    )

    def __init__(self, text: str):
        self.text = text

    def extract(self) -> list[str]:
        seen = set()
        surveys = []

        # Remove civil suit case numbers before extracting survey numbers.
        cleaned_text = self.CASE_NUMBER_PATTERN.sub(" ", self.text)

        for match in self.SURVEY_PATTERN.finditer(cleaned_text):
            survey = match.group(1).strip()

            if survey not in seen:
                seen.add(survey)
                surveys.append(survey)

        return surveys