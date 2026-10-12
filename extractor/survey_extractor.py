"""
Survey Number Extractor.
"""

from __future__ import annotations

import re
from typing import Iterator


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

    # Prefixes: Sy.No, S.No, Survey No/Number, Re-Sy.No, Re-Survey No,
    # R.S.No / RS.No, plus common OCR variants (S¥.No, Sy.Na, Sy.n0).
    SURVEY_PATTERN = re.compile(
        r"""
(?:
    \bR\.?\s*S\.?\s*N[o0]s?\.?|
    \bRe-?\s*Sy\.?\s*N[o0]s?\.?|
    \bRe-?\s*Survey\s*N[o0]s?\.?|
    \bS[y¥]?\.?\s*N[o0a]s?\.?|
    \bSurvey\s*Numbers?|
    \bSurvey\s*N[o0]s?\.?
)

[\s:.-]*

(?:i\.?\s*e\.?,?\s*)?

(
    \d+
    (?-i:(?:(?<!\.)\s+(?:A|B|C|AA|AB|AC|AD))?)
    (?:/[A-Za-z0-9*]+)*

    (?:
        \s*\.?\s*
        (?:,\s*(?:and\b|&)?|&|\band\b)
        \s*

        \d+(?!\d|(?:st|nd|rd|th)\b)
        (?-i:(?:(?<!\.)\s+(?:A|B|C|AA|AB|AC|AD))?)
        (?:/[A-Za-z0-9*]+)*
    )*
)
""",
        re.IGNORECASE | re.VERBOSE,
    )

    def __init__(self, text: str):
        self.text = text

    def iter_mentions(self) -> Iterator[re.Match]:
        """
        Yield every survey-number mention in the text.

        Match offsets refer to the original text, so callers can look at
        the surrounding context (used by SurveyLocationExtractor).
        """
        # Blank out civil suit case numbers (O.S.No.158/1984) without
        # shifting offsets.
        cleaned_text = self.CASE_NUMBER_PATTERN.sub(
            lambda m: " " * len(m.group(0)),
            self.text,
        )

        yield from self.SURVEY_PATTERN.finditer(cleaned_text)

    def extract(self) -> list[str]:
        seen = set()
        surveys = []

        for match in self.iter_mentions():
            survey = match.group(1).strip()

            if survey not in seen:
                seen.add(survey)
                surveys.append(survey)

        return surveys
