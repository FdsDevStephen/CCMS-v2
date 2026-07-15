"""
Section Extractor.
"""

from __future__ import annotations

import re


class SectionExtractor:

    SECTION_PATTERN = re.compile(
        r"""
(?:
    under\s+Sections?|
    u/s|
    U/S|
    Sections?|
    Sec(?:tion)?\.?|
    Secs\.?|
    \bS\.
)

[\s:\-./]*

(

    \d+
(?:\s*-\s*[A-Za-z]{1,3}|\s*[A-Za-z]{1,3})?
(?:\s*\([A-Za-z0-9]{1,4}\))?

    (?:

        \s*(?:,|&|\band\b)\s*

        \(?

        \d+
(?:\s*-\s*[A-Za-z]{1,3}|\s*[A-Za-z]{1,3})?
(?:\s*\([A-Za-z0-9]{1,4}\))?

        \)?

    )*

)

""",
        re.IGNORECASE | re.VERBOSE,
    )

    def __init__(self, text: str):

        self.text = text

    def extract(self) -> list[str]:

        seen = set()

        sections = []

        for match in self.SECTION_PATTERN.finditer(self.text):

            section = " ".join(match.group(1).split())

            if section not in seen:

                seen.add(section)

                sections.append(section)

        return sections
