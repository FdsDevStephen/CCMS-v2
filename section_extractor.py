from __future__ import annotations

import re

from rapidfuzz.fuzz import ratio


# ==========================================================
# SECTION HEADINGS
# ==========================================================

SECTION_PATTERNS = {
    "synopsis": [
        "SYNOPSIS",
        "LIST OF EVENTS WITH SYNOPSIS",
        "LIST OF EVENTS / SYNOPSIS",
        "EVENTS WITH SYNOPSIS",
    ],

    "brief_facts": [
        "BRIEF FACTS OF THE CASE",
        "FACTS IN BRIEF",
        "BRIEF FACTS",
        "FACTS OF THE CASE",
    ],

    "interim_prayer": [
        "INTERIM PRAYER",
    ],

    "prayer": [
        "PRAYER",
    ],
}


# ==========================================================
# GENERAL STOP HEADINGS
# ==========================================================

STOP_PATTERNS = {
    "synopsis": [
        "BRIEF FACTS OF THE CASE",
        "FACTS IN BRIEF",
        "BRIEF FACTS",
        "FACTS OF THE CASE",
    ],

    "brief_facts": [
        "GROUNDS",
        "GROUNDS FOR INTERIM PRAYER",
        "GROUNDS FOR INTERIM RELIEF",
        "PRAYER",
        "INTERIM PRAYER",
    ],

    "prayer": [
        "INTERIM PRAYER",
        "INTERIM RELIEF",
        "GROUNDS FOR INTERIM PRAYER",
        "GROUNDS FOR INTERIM RELIEF",
    ],

    "interim_prayer": [
        "PLACE",
        "DATE",
        "ADVOCATE FOR PETITIONER",
        "ADVOCATE FOR THE PETITIONER",
        "ADVOCATE FOR RESPONDENT",
        "ADVOCATE FOR THE RESPONDENT",
    ],
}


# ==========================================================
# SIGNATURE / END MARKERS
# ==========================================================

SIGNATURE_STOP_PATTERNS = [
    r"ADVOCATE\s+FOR\s+PETITIONER",
    r"ADVOCATE\s+FOR\s+THE\s+PETITIONER",
    r"ADVOCATE\s+FOR\s+RESPONDENT",
    r"ADVOCATE\s+FOR\s+THE\s+RESPONDENT",
]


# ==========================================================
# NORMALIZE
# ==========================================================

def normalize(text: str) -> str:
    return " ".join(text.upper().split())


# ==========================================================
# PAGE NUMBER
# ==========================================================

def get_page_number(line: str) -> int | None:

    match = re.match(
        r"=+\s*PAGE\s+(\d+)\s*=+",
        line.strip(),
        re.IGNORECASE,
    )

    if match:
        return int(match.group(1))

    return None


# ==========================================================
# HEADING MATCH
# ==========================================================

def is_heading(line: str, heading: str) -> bool:

    line_normalized = normalize(line)
    heading_normalized = normalize(heading)

    if line_normalized == heading_normalized:
        return True

    return ratio(
        line_normalized,
        heading_normalized,
    ) >= 85


# ==========================================================
# SIGNATURE MATCH
# ==========================================================

def is_signature_stop(line: str) -> bool:

    normalized = normalize(line)

    for pattern in SIGNATURE_STOP_PATTERNS:

        if re.search(
            pattern,
            normalized,
            re.IGNORECASE,
        ):
            return True

    return False


# ==========================================================
# SECTION STOP MATCH
# ==========================================================

def is_section_stop(
    section: str,
    line: str,
) -> bool:

    for heading in STOP_PATTERNS.get(section, []):

        if is_heading(line, heading):
            return True

    return False


# ==========================================================
# FIND SECTION HEADINGS
# ==========================================================

def find_headings(text: str) -> list[dict]:

    lines = text.splitlines()

    matches = []

    current_page = None

    # IMPORTANT:
    # INTERIM PRAYER must be checked before PRAYER.
    section_order = [
        "interim_prayer",
        "synopsis",
        "brief_facts",
        "prayer",
    ]

    for index, raw_line in enumerate(lines):

        line = raw_line.strip()

        if not line:
            continue

        page_number = get_page_number(line)

        if page_number is not None:
            current_page = page_number
            continue

        for section in section_order:

            matched = False

            for heading in SECTION_PATTERNS[section]:

                if is_heading(line, heading):

                    matches.append(
                        {
                            "section": section,
                            "line_index": index,
                            "page": current_page,
                            "heading": line,
                        }
                    )

                    matched = True
                    break

            if matched:
                break

    return matches


# ==========================================================
# EXTRACT SECTIONS
# ==========================================================

def extract_sections(text: str) -> dict:

    lines = text.splitlines()

    matches = find_headings(text)

    sections = {
        "synopsis": "",
        "brief_facts": "",
        "prayer": "",
        "interim_prayer": "",
    }

    pages = {
        "synopsis": [],
        "brief_facts": [],
        "prayer": [],
        "interim_prayer": [],
    }

    # ------------------------------------------------------
    # Nothing found
    # ------------------------------------------------------

    if not matches:

        return {
            "sections": sections,
            "pages": pages,
        }

    # ------------------------------------------------------
    # Keep FIRST occurrence of every section
    # ------------------------------------------------------

    first_occurrence = {}

    for match in matches:

        section = match["section"]

        if section not in first_occurrence:

            first_occurrence[section] = match

    # ------------------------------------------------------
    # Order sections according to document position
    # ------------------------------------------------------

    ordered = sorted(
        first_occurrence.values(),
        key=lambda item: item["line_index"],
    )

    # ------------------------------------------------------
    # Extract each section
    # ------------------------------------------------------

    for position, current in enumerate(ordered):

        section = current["section"]

        start = current["line_index"] + 1

        # Default boundary:
        # next detected requested section
        if position + 1 < len(ordered):

            end = ordered[
                position + 1
            ]["line_index"]

        else:

            end = len(lines)

        current_page = current["page"]

        extracted_lines = []

        # --------------------------------------------------
        # Read lines
        # --------------------------------------------------

        for raw_line in lines[start:end]:

            line = raw_line.strip()

            if not line:
                continue

            # ----------------------------------------------
            # Page marker
            # ----------------------------------------------

            page_number = get_page_number(line)

            if page_number is not None:

                current_page = page_number

                continue

            # ----------------------------------------------
            # Signature / advocate boundary
            # ----------------------------------------------

            if is_signature_stop(line):

                break

            # ----------------------------------------------
            # Natural section boundary
            # ----------------------------------------------

            if is_section_stop(
                section,
                line,
            ):

                break

            # ----------------------------------------------
            # Keep actual content
            # ----------------------------------------------

            extracted_lines.append(line)

            if current_page is not None:

                pages[section].append(
                    current_page
                )

        sections[section] = "\n".join(
            extracted_lines
        ).strip()

    # ------------------------------------------------------
    # Remove duplicate pages
    # ------------------------------------------------------

    for section in pages:

        pages[section] = sorted(
            set(pages[section])
        )

    return {
        "sections": sections,
        "pages": pages,
    }