from __future__ import annotations

import re


VALID_SECTION_SUFFIXES = {
    "A",
    "B",
    "C",
    "D",
    "AA",
    "AB",
    "AC",
    "AD",
    "BA",
    "BB",
    "BC",
}


class SectionExtractor:

    SECTION_VALUE = r"""
        \d+
        (?:
            \s*-\s*[A-Za-z]{1,3}
            |
            \s*[A-Za-z]{1,2}
        )?
        (?:
            \s*\([A-Za-z0-9]{1,4}\)
        )*
    """

    SECTION_PATTERNS = [
        rf"""
        \b
        (?:Section|Sec\.?|S\.)
        \s*
        ({SECTION_VALUE})
        (?![A-Za-z0-9])
        """,

        rf"""
        \b
        Sections
        \s*
        ({SECTION_VALUE})
        (?![A-Za-z0-9])
        """,

        rf"""
        \b
        u/s
        \s*
        ({SECTION_VALUE})
        (?![A-Za-z0-9])
        """,

        rf"""
        \b
        under\s+(?:Section|Sec\.?)
        \s*
        ({SECTION_VALUE})
        (?![A-Za-z0-9])
        """,
    ]

    def __init__(
        self,
        text: str | None = None,
    ):

        self.text = text or ""

    # ======================================================
    # PUBLIC METHOD
    # ======================================================

    def extract(
        self,
        text: str | None = None,
    ) -> list[str]:

        if text is not None:

            self.text = text

        if not self.text:

            return []

        sections = []

        for pattern in self.SECTION_PATTERNS:

            matches = re.finditer(
                pattern,
                self.text,
                flags=re.IGNORECASE
                | re.VERBOSE,
            )

            for match in matches:

                if not match.groups():
                    continue

                value = match.group(1)

                if not value:
                    continue

                value = self._clean_value(
                    value
                )

                if not value:
                    continue

                if not self._is_valid_candidate(
                    value
                ):
                    continue

                if value not in sections:

                    sections.append(
                        value
                    )

        return sections

    # ======================================================
    # CLEAN VALUE
    # ======================================================

    def _clean_value(
        self,
        value: str,
    ) -> str:

        value = value.strip()

        # Normalize whitespace
        value = re.sub(
            r"\s+",
            " ",
            value,
        )

        # Normalize spaces around hyphen
        value = re.sub(
            r"\s*-\s*",
            "-",
            value,
        )

        # Normalize spaces inside brackets
        value = re.sub(
            r"\(\s*",
            "(",
            value,
        )

        value = re.sub(
            r"\s*\)",
            ")",
            value,
        )

        value = self._drop_word_suffix(
            value.strip()
        )

        return value.strip()

    # ======================================================
    # DROP WORD SUFFIX
    # ======================================================

    @staticmethod
    def _drop_word_suffix(
        value: str,
    ) -> str:
        """
        The optional letter suffix in SECTION_VALUE also swallows the
        next word: "U/s 67 of the Act" -> "67 of". Keep the number and
        drop letters that are not a real suffix (94A, 94-B, 79 AA).

        A suffix separated by a space must be upper-case, so
        "Section 4 a notification" gives "4", not "4A".
        """

        match = re.fullmatch(
            r"(\d+)(\s?-?)([A-Za-z]{1,3})((?:\([A-Za-z0-9]{1,4}\))*)",
            value,
        )

        if not match:
            return value

        number, separator, suffix, brackets = match.groups()

        is_suffix = (
            # A hyphen is deliberate notation ("94-G", "5-G").
            "-" in separator
            or (
                suffix.upper() in VALID_SECTION_SUFFIXES
                and (
                    " " not in separator
                    or suffix.isupper()
                )
            )
        )

        if is_suffix:
            return value

        return number + brackets

    # ======================================================
    # VALIDATE SECTION
    # ======================================================

    def _is_valid_candidate(
        self,
        value: str,
    ) -> bool:

        if not value:
            return False

        # --------------------------------------------------
        # Basic section format
        #
        # Valid examples:
        #
        # 56
        # 57
        # 94-A
        # 94-B
        # 94A
        # 94B
        # 19(1)
        # 19(1)(a)
        # --------------------------------------------------

        if not re.fullmatch(
            r"""
            \d+
            (?:
                \s*-\s*[A-Za-z]{1,3}
                |
                \s*[A-Za-z]{1,2}
            )?
            (?:
                \s*\([A-Za-z0-9]{1,4}\)
            )*
            """,
            value,
            flags=re.VERBOSE,
        ):
            return False

        # --------------------------------------------------
        # Reject OCR words attached to section numbers
        #
        # Examples rejected:
        #
        # 57of
        # 25the
        # 11and
        # 94from
        # 56with
        #
        # Examples accepted:
        #
        # 94A
        # 94B
        # 94-AA
        # 94-AB
        # --------------------------------------------------

        compact = re.sub(
            r"\s+",
            "",
            value,
        )

        suffix_match = re.fullmatch(
            r"(\d+)([A-Za-z]{1,3})",
            compact,
        )

        if suffix_match:

            suffix = (
                suffix_match.group(2)
                .upper()
            )

            if (
                suffix
                not in VALID_SECTION_SUFFIXES
            ):

                return False

        # --------------------------------------------------
        # Reject obviously invalid values
        # --------------------------------------------------

        number_match = re.match(
            r"^\d+",
            compact,
        )

        if not number_match:
            return False

        number = int(
            number_match.group()
        )

        # Section numbers should not be absurdly large.
        if number > 9999:
            return False

        return True