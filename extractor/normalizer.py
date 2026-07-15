"""
Normalization utilities for extracted legal information.
"""

from __future__ import annotations

import re
from typing import Any

from extractor.act_normalizer import ActNormalizer


ACT_NORMALIZER = ActNormalizer()

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

SECTION_PREFIX_PATTERN = re.compile(
    r"^(Section|Sections|Sec\.?|Secs\.?|S\.|u/s|U/S)\s*",
    flags=re.IGNORECASE,
)

SURVEY_PREFIX_PATTERN = re.compile(
    r"^(Sy(?:urvey)?\.?\s*Nos?\.?|Survey\s*Numbers?|Survey\s*Nos?\.?|Survey\s*Number)\s*",
    flags=re.IGNORECASE,
)

MULTI_VALUE_SEPARATOR_PATTERN = re.compile(
    r"\s*(?:,|&|\band\b)\s*",
    flags=re.IGNORECASE,
)

VALID_SECTION_PATTERN = re.compile(
    r"""
    ^
    (
        Rule\s+\d+(?:-[A-Z]+)?(?:\(\d+[A-Z]?\))?
        |
        \d+[A-Z]{0,2}(?:\(\d+[A-Z]?\))?
        |
        \d+-[A-Z](?:\(\d+\))?
    )
    $
    """,
    flags=re.VERBOSE | re.IGNORECASE,
)


class Normalizer:
    """Normalize extracted legal information."""

    @staticmethod
    def normalize_case_number(case_number: str | None) -> str | None:
        """Normalize case number spacing."""
        if not case_number:
            return None

        return clean_spaces(case_number)

    @staticmethod
    def normalize_survey_numbers(survey_numbers: list[str]) -> list[str]:
        """Normalize, split, and deduplicate survey numbers."""
        normalized: list[str] = []
        seen: set[str] = set()

        for survey in survey_numbers:
            survey = normalize_survey_text(survey)
            values = split_multi_value_text(survey)

            for value in values:
                value = value.strip().rstrip(".,;:")

                if value and value not in seen:
                    seen.add(value)
                    normalized.append(value)

        return normalized

    @staticmethod
    def normalize_sections(sections: list[str]) -> list[str]:
        """Normalize, split, and deduplicate section names."""
        normalized: list[str] = []
        seen: set[str] = set()

        for section in sections:
            section = Normalizer.normalize_section_name(section)
            values = expand_or_split_section(section)

            for value in values:
                value = finalize_section_text(value)

                if value and value not in seen:
                    seen.add(value)
                    normalized.append(value)

        return normalized

    @staticmethod
    def normalize_acts(acts: list[str]) -> list[str]:
        """Normalize and deduplicate act names."""
        normalized: list[str] = []
        seen: set[str] = set()

        for act in acts:
            act = clean_spaces(ACT_NORMALIZER.normalize(act))

            if act and act not in seen:
                seen.add(act)
                normalized.append(act)

        return normalized

    @staticmethod
    def normalize_act_section_mapping(
        mappings: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Normalize Act -> Section mapping."""
        merged: dict[str, list[str]] = {}

        for item in mappings:
            act = clean_spaces(ACT_NORMALIZER.normalize(item.get("act", "")))

            if not act:
                continue

            merged.setdefault(act, [])

            for section in item.get("sections", []):
                section = Normalizer.normalize_section_name(section)

                if not is_valid_mapping_section(section):
                    continue

                for value in expand_combined_section(section):
                    if value not in merged[act]:
                        merged[act].append(value)

        return [
            {
                "act": act,
                "sections": sections,
            }
            for act, sections in merged.items()
        ]

    @staticmethod
    def normalize_section_name(section: str) -> str:
        """Normalize a single section or rule representation."""
        section = section.strip()
        section = SECTION_PREFIX_PATTERN.sub("", section)
        section = clean_spaces(section)

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

        section = remove_trailing_connector_text(section)
        section = normalize_section_suffix(section)
        section = normalize_rule_format(section)
        section = normalize_parentheses(section)

        return finalize_section_text(section)


def clean_spaces(value: str) -> str:
    """Collapse repeated whitespace into single spaces."""
    return " ".join(value.split())


def normalize_survey_text(survey: str) -> str:
    """Clean a single survey number string before splitting."""
    survey = survey.strip()
    survey = SURVEY_PREFIX_PATTERN.sub("", survey)
    survey = re.sub(r"^i\.?e\.?,?\s*", "", survey, flags=re.IGNORECASE)
    survey = clean_spaces(survey)

    # 9. A -> 9
    # 9.A -> 9
    survey = re.sub(
        r"^(\d+)\.\s*[A-Za-z]{1,2}\b.*$",
        r"\1",
        survey,
    )

    survey = re.sub(r"\..*$", "", survey)

    survey = re.sub(
        r"(?<=\d)\s*and\s*(?=\d)",
        " and ",
        survey,
        flags=re.IGNORECASE,
    )

    # 9 A -> 9
    # 9A -> 9
    # Prevent dotted/list references from becoming survey suffixes.
    survey = re.sub(
        r"\b([1-9])\s*([A-Za-z]{1,2})\b",
        r"\1",
        survey,
    )

    # OCR cleanup: 82 A -> 82A
    # Only join suffixes for two-or-more digit survey numbers.
    survey = re.sub(
        r"\b(\d{2,})\s+([A-Za-z]{1,2})\b",
        r"\1\2",
        survey,
    )

    return survey


def split_multi_value_text(value: str) -> list[str]:
    """Split comma, ampersand, or 'and' separated values."""
    if not MULTI_VALUE_SEPARATOR_PATTERN.search(value):
        return [value]

    return [
        item.strip()
        for item in MULTI_VALUE_SEPARATOR_PATTERN.split(value)
        if item.strip()
    ]


def remove_trailing_connector_text(section: str) -> str:
    """Remove trailing connector text such as 'of', 'under', or 'for'."""
    return re.sub(
        r"\s+(?:of|under|the|to|for)\b.*$",
        "",
        section,
        flags=re.IGNORECASE,
    )


def normalize_section_suffix(section: str) -> str:
    """Normalize OCR spacing in section suffixes, like 79 B -> 79B."""
    match = re.fullmatch(r"(\d+)\s+([A-Za-z]{1,3})", section)

    if not match:
        return section

    number, suffix = match.groups()
    suffix = suffix.upper()

    if suffix not in VALID_SECTION_SUFFIXES:
        return section

    return f"{number}{suffix}"


def normalize_rule_format(section: str) -> str:
    """Normalize rule formats such as Rule 108 D (3) -> Rule 108-D(3)."""
    replacements = [
        (
            r"Rule\s+(\d+)\s+([A-Z])\s*\((\d+)\)",
            r"Rule \1-\2(\3)",
        ),
        (
            r"Rule\s+(\d+)-([A-Z])-([0-9]+)",
            r"Rule \1-\2(\3)",
        ),
        (
            r"(\d+)\s+([A-Z])\s*\((\d+)\)",
            r"\1-\2(\3)",
        ),
        (
            r"(\d+)-([A-Z])-([0-9]+)",
            r"\1-\2(\3)",
        ),
    ]

    for pattern, replacement in replacements:
        section = re.sub(
            pattern,
            replacement,
            section,
            flags=re.IGNORECASE,
        )

    return section


def normalize_parentheses(section: str) -> str:
    """Normalize spacing around section parentheses."""
    return re.sub(
        r"(\d+)\s+\((\d+)\)",
        r"\1(\2)",
        section,
    )


def expand_or_split_section(section: str) -> list[str]:
    """Expand or split section expressions into individual section values."""
    match = re.fullmatch(
        r"(\d+[A-Za-z]?)\s*\((\d+[A-Za-z]?)\)\s*(?:and|&|,)\s*\((\d+[A-Za-z]?)\)",
        section,
        flags=re.IGNORECASE,
    )

    if match:
        base = match.group(1)
        return [
            f"{base}({match.group(2)})",
            f"{base}({match.group(3)})",
        ]

    if re.fullmatch(
        r"\d+\(\d+\)\s*(?:and|&|,)\s*\d+",
        section,
        flags=re.IGNORECASE,
    ):
        return [
            item.strip()
            for item in MULTI_VALUE_SEPARATOR_PATTERN.split(section)
            if item.strip()
        ]

    return split_multi_value_text(section)


def expand_combined_section(section: str) -> list[str]:
    """
    Expand combined sections.

    Examples:
    - 19(1)and(2)
    - 19(1) and 19(2)
    - 19(1)&19(2)
    """
    match = re.fullmatch(
        r"(\d+)\((\d+)\)\s*(?:and|&|,)\s*(?:\1)?\(?(\d+)\)?",
        section,
        flags=re.IGNORECASE,
    )

    if not match:
        return [section]

    base, first, second = match.groups()

    return [
        f"{base}({first})",
        f"{base}({second})",
    ]


def finalize_section_text(section: str) -> str:
    """Apply final formatting cleanup to a section value."""
    section = section.strip()
    section = clean_spaces(section)

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

    return section.rstrip(".,;:")


def is_valid_mapping_section(section: str) -> bool:
    """Return whether a section is valid for Act -> Section mapping."""
    if not section:
        return False

    if re.fullmatch(r"(18|19|20)\d{2}", section):
        return False

    if re.fullmatch(r"(18|19|20)\d{2}-(18|19|20)\d{2}", section):
        return False

    if re.fullmatch(r"\d+\.\d+", section):
        return False

    return bool(VALID_SECTION_PATTERN.fullmatch(section))