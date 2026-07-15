"""
Normalization utilities for extracted legal information.
"""

from __future__ import annotations

import re


class Normalizer:
    """
    Normalize extracted legal information.
    """

    # ==========================================================
    # Case Number
    # ==========================================================

    @staticmethod
    def normalize_case_number(case_number: str | None) -> str | None:

        if not case_number:
            return None

        return " ".join(case_number.split())

    # ==========================================================
    # Survey Numbers
    # ==========================================================

    @staticmethod
    def normalize_survey_numbers(
        survey_numbers: list[str],
    ) -> list[str]:

        normalized = []
        seen = set()

        for survey in survey_numbers:

            survey = survey.strip()

            # ----------------------------------------------
            # Remove prefixes
            # ----------------------------------------------

            survey = re.sub(
                r"^(Sy(?:urvey)?\.?\s*Nos?\.?|Survey\s*Numbers?|Survey\s*Nos?\.?|Survey\s*Number)\s*",
                "",
                survey,
                flags=re.IGNORECASE,
            )

            # ----------------------------------------------
            # Remove "i.e."
            # ----------------------------------------------

            survey = re.sub(
                r"^i\.?e\.?,?\s*",
                "",
                survey,
                flags=re.IGNORECASE,
            )

            survey = " ".join(survey.split())

            survey = re.sub(
                r"\..*$",
                "",
                survey,
            )

            survey = re.sub(
                r"(?<=\d)\s*and\s*(?=\d)",
                " and ",
                survey,
                flags=re.IGNORECASE,
            )
            # ----------------------------------------------
            # OCR cleanup
            # 82 A -> 82A
            # ----------------------------------------------

            survey = re.sub(
                r"(\d+)\s+([A-Za-z]{1,2})\b",
                r"\1\2",
                survey,
            )

            values = []

            # ----------------------------------------------
            # Split multiple survey numbers
            # 87,88,89
            # 87 & 88
            # 87 and 88
            # ----------------------------------------------

            if re.search(
                r"\s*(?:,|&|\band\b)\s*",
                survey,
                flags=re.IGNORECASE,
            ):

                values = [
                    x.strip()
                    for x in re.split(
                        r"\s*(?:,|&|\band\b)\s*",
                        survey,
                        flags=re.IGNORECASE,
                    )
                    if x.strip()
                ]

            else:

                values = [survey]

            # ----------------------------------------------
            # Remove duplicates
            # ----------------------------------------------

            for value in values:

                value = value.strip()

                value = value.rstrip(".,;:")

                if value and value not in seen:

                    seen.add(value)

                    normalized.append(value)

        return normalized

    # ==========================================================
    # Sections
    # ==========================================================

    @staticmethod
    def normalize_sections(
        sections: list[str],
    ) -> list[str]:

        normalized = []
        seen = set()

        VALID_SUFFIXES = {
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

        for section in sections:

            section = section.strip()

            section = re.sub(
                r"^(Section|Sections|Sec\.?|Secs\.?|S\.|u/s|U/S)\s*",
                "",
                section,
                flags=re.IGNORECASE,
            )

            section = " ".join(section.split())

            # ----------------------------------------------
            # Remove trailing connector words
            # Example:
            # 28 of -> 28
            # 79B under -> 79B
            # ----------------------------------------------

            section = re.sub(
                r"\s+(?:of|under|the|to|for)\b.*$",
                "",
                section,
                flags=re.IGNORECASE,
            )

            # ----------------------------------------------
            # OCR cleanup
            # 79 B -> 79B
            # 95 AB -> 95AB
            # 192 A -> 192A
            # ----------------------------------------------

            m = re.fullmatch(
                r"(\d+)\s+([A-Za-z]{1,3})",
                section,
            )

            if m and m.group(2).upper() in VALID_SUFFIXES:

                section = f"{m.group(1)}{m.group(2).upper()}"

            values = []

            # ----------------------------------------------
            # 29(2) & (3)
            # 136(2) & (3)
            # ----------------------------------------------

            m = re.fullmatch(
                r"(\d+[A-Za-z]?)\s*\((\d+[A-Za-z]?)\)\s*(?:and|&|,)\s*\((\d+[A-Za-z]?)\)",
                section,
                flags=re.IGNORECASE,
            )

            if m:

                base = m.group(1)

                values = [
                    f"{base}({m.group(2)})",
                    f"{base}({m.group(3)})",
                ]

            # ----------------------------------------------
            # 19(1) and 19
            # ----------------------------------------------

            elif re.fullmatch(
                r"\d+\(\d+\)\s*(?:and|&|,)\s*\d+",
                section,
                flags=re.IGNORECASE,
            ):

                left, right = re.split(
                    r"\s*(?:and|&|,)\s*",
                    section,
                )

                values = [left, right]

            # ----------------------------------------------
            # 11,19,38
            # 11A and 11B
            # ----------------------------------------------

            elif re.search(
                r"\s*(?:,|&|\band\b)\s*",
                section,
                flags=re.IGNORECASE,
            ):

                values = [
                    x.strip()
                    for x in re.split(
                        r"\s*(?:,|&|\band\b)\s*",
                        section,
                        flags=re.IGNORECASE,
                    )
                    if x.strip()
                ]

            else:

                values = [section]

            # ----------------------------------------------
            # Final Cleanup
            # ----------------------------------------------

            for value in values:

                value = value.strip()

                value = " ".join(value.split())

                value = re.sub(r"\s+\(", "(", value)
                value = re.sub(r"\)\s+", ")", value)
                value = re.sub(r"\s*,\s*", ",", value)

                value = re.sub(
                    r"\s*-\s*",
                    "-",
                    value,
                )

                value = re.sub(
                    r"(Sec|Section|Sections)$",
                    "",
                    value,
                    flags=re.IGNORECASE,
                ).strip()

                value = value.rstrip(".,;:")

                if value and value not in seen:

                    seen.add(value)
                    normalized.append(value)

        return normalized

    # ==========================================================
    # Acts
    # ==========================================================

    @staticmethod
    def normalize_acts(
        acts: list[str],
    ) -> list[str]:

        normalized = []
        seen = set()

        # ----------------------------------------------
        # Remove exact duplicates
        # ----------------------------------------------

        for act in acts:

            act = " ".join(act.split())

            act = act.rstrip(".,;:")

            if act and act not in seen:

                seen.add(act)

                normalized.append(act)

        # ----------------------------------------------
        # Remove partial Act names
        # Example:
        # Karnataka Land Revenue Act
        # Land Revenue Act
        # ----------------------------------------------

        filtered = []

        for act in normalized:

            keep = True

            for other in normalized:

                if act == other:
                    continue

                if len(other) <= len(act):
                    continue

                # Ignore year while comparing
                act_compare = re.sub(r",\s*\d{4}$", "", act, flags=re.IGNORECASE)
                other_compare = re.sub(r",\s*\d{4}$", "", other, flags=re.IGNORECASE)

                if other_compare.lower().endswith(act_compare.lower()):

                    keep = False
                    break

            if keep:

                filtered.append(act)

        return filtered
