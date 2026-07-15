"""
Normalization utilities for extracted legal information.
"""

from __future__ import annotations

import re

from extractor.act_normalizer import ActNormalizer

ACT_NORMALIZER = ActNormalizer()

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

        for act in acts:

            act = ACT_NORMALIZER.normalize(act)

            act = " ".join(act.split())

            if act and act not in seen:

                seen.add(act)

                normalized.append(act)

        return normalized


    @staticmethod
    def normalize_act_section_mapping(
        mappings: list[dict],
    ) -> list[dict]:
        """
        Normalize Act -> Section mapping.
        """

        merged = {}

        for item in mappings:

            act = ACT_NORMALIZER.normalize(
                item.get("act", "")
            )

            act = " ".join(act.split())

            if not act:
                continue

            if act not in merged:

                merged[act] = []

            for section in item.get("sections", []):

                section = Normalizer.normalize_section_name(section)
                
                if re.fullmatch(r"\d+\.\d+", section):
                    continue

                # ---------------------------------------
                # Expand combined sections
                # 19(1)and(2)
                # 19(1) and 19(2)
                # 19(1)&19(2)
                # ---------------------------------------

                values = []

                m = re.fullmatch(
                    r"(\d+)\((\d+)\)\s*(?:and|&|,)\s*(?:\1)?\(?(\d+)\)?",
                    section,
                    flags=re.IGNORECASE,
                )

                if m:

                    values = [
                        f"{m.group(1)}({m.group(2)})",
                        f"{m.group(1)}({m.group(3)})",
                    ]

                else:

                    values = [section]

                for value in values:

                    if value not in merged[act]:

                        merged[act].append(value)

        result = []

        for act, sections in merged.items():

            result.append(
                {
                    "act": act,
                    "sections": sections,
                }
            )

        return result
    
    @staticmethod
    def normalize_section_name(section: str) -> str:
        """
        Normalize a single Section/Rule representation.
        """

        import re

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

        section = section.strip()

        section = re.sub(
            r"^(Section|Sections|Sec\.?|Secs\.?|S\.|u/s|U/S)\s*",
            "",
            section,
            flags=re.IGNORECASE,
        )

        section = " ".join(section.split())
        
        
        
        section = re.sub(
            r"(\))(?=(and|or)\b)",
            r"\1 ",
            section,
            flags=re.IGNORECASE,
        )

        section = re.sub(
            r"([)&,])(?=\S)",
            r"\1 ",
            section,
        )
        # ----------------------------------------------
        # Remove trailing connector words
        # ----------------------------------------------

        section = re.sub(
            r"\s+(?:of|under|the|to|for)\b.*$",
            "",
            section,
            flags=re.IGNORECASE,
        )

        # ----------------------------------------------
        # OCR Cleanup
        # 79 B -> 79B
        # 95 AB -> 95AB
        # ----------------------------------------------

        m = re.fullmatch(
            r"(\d+)\s+([A-Za-z]{1,3})",
            section,
        )

        if m and m.group(2).upper() in VALID_SUFFIXES:

            section = f"{m.group(1)}{m.group(2).upper()}"

        # ----------------------------------------------
        # Rule 108 D (3) -> Rule 108-D(3)
        # ----------------------------------------------

        section = re.sub(
            r"Rule\s+(\d+)\s+([A-Z])\s*\((\d+)\)",
            r"Rule \1-\2(\3)",
            section,
            flags=re.IGNORECASE,
        )

        # ----------------------------------------------
        # Rule 108-D-3 -> Rule 108-D(3)
        # ----------------------------------------------

        section = re.sub(
            r"Rule\s+(\d+)-([A-Z])-([0-9]+)",
            r"Rule \1-\2(\3)",
            section,
            flags=re.IGNORECASE,
        )

        # ----------------------------------------------
        # 108 D (3) -> 108-D(3)
        # ----------------------------------------------

        section = re.sub(
            r"(\d+)\s+([A-Z])\s*\((\d+)\)",
            r"\1-\2(\3)",
            section,
            flags=re.IGNORECASE,
        )

        # ----------------------------------------------
        # 108-D-3 -> 108-D(3)
        # ----------------------------------------------

        section = re.sub(
            r"(\d+)-([A-Z])-([0-9]+)",
            r"\1-\2(\3)",
            section,
            flags=re.IGNORECASE,
        )

        # ----------------------------------------------
        # 136 (2) -> 136(2)
        # ----------------------------------------------

        section = re.sub(
            r"(\d+)\s+\((\d+)\)",
            r"\1(\2)",
            section,
        )

        # ----------------------------------------------
        # Final Cleanup
        # ----------------------------------------------

        section = re.sub(r"\s+\(", "(", section)
        section = re.sub(r"\)\s+", ")", section)
        section = re.sub(r"\s*,\s*", ",", section)
        section = re.sub(r"\s*-\s*", "-", section)

        section = re.sub(
            r"(Sec|Section|Sections)$",
            "",
            section,
            flags=re.IGNORECASE,
        ).strip()

        section = section.rstrip(".,;:")

        return section