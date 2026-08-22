from __future__ import annotations

import re
from pathlib import Path
from time import perf_counter
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pytesseract
from pdf2image import convert_from_path


PDF_PATH = Path(
    r"C:\Users\steph\OneDrive\Desktop\CEG\CCMS V2\uploads\WP-7049-2021-B.pdf"
)

POPPLER_PATH = (
    r"C:\Users\steph\.cache\codex-runtimes"
    r"\codex-primary-runtime"
    r"\dependencies\native\poppler\Library\bin"
)

OUTPUT_DIR = Path("section_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_WORKERS = 4


# ==========================================================
# FAST SCAN RESULTS
# ==========================================================

CANDIDATE_PAGES = {
    "synopsis": [3],
    "brief_facts": [4],
    "prayer": [9],
    "interim_prayer": [10],
}


# ==========================================================
# SECTION HEADINGS
# ==========================================================

SECTION_HEADINGS = {
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
    "prayer": [
        "PRAYER",
    ],
    "interim_prayer": [
        "INTERIM PRAYER",
    ],
}


# ==========================================================
# STOP HEADINGS
# ==========================================================

STOP_HEADINGS = {
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
        "ADVOCATE FOR PETITIONER",
        "ADVOCATE FOR THE PETITIONER",
        "ADVOCATE FOR RESPONDENT",
        "ADVOCATE FOR THE RESPONDENT",
        "DATE",
        "PLACE",
    ],
}


SIGNATURE_PATTERNS = [
    r"ADVOCATE\s+FOR\s+PETITIONER",
    r"ADVOCATE\s+FOR\s+THE\s+PETITIONER",
    r"ADVOCATE\s+FOR\s+RESPONDENT",
    r"ADVOCATE\s+FOR\s+THE\s+RESPONDENT",
]


# ==========================================================
# HELPERS
# ==========================================================

def normalize(text: str) -> str:
    return " ".join(text.upper().split())


def is_heading(line: str, headings: list[str]) -> bool:
    normalized = normalize(line)

    for heading in headings:
        if normalized == normalize(heading):
            return True

    return False


def is_signature(line: str) -> bool:
    normalized = normalize(line)

    for pattern in SIGNATURE_PATTERNS:
        if re.search(pattern, normalized, re.IGNORECASE):
            return True

    return False


def is_stop_heading(section: str, line: str) -> bool:
    return is_heading(
        line,
        STOP_HEADINGS.get(section, []),
    )


def page_numbers_in_window(
    page: int,
    total_pages: int,
    radius: int = 1,
) -> list[int]:

    pages = []

    for offset in range(-radius, radius + 1):

        candidate = page + offset

        if 1 <= candidate <= total_pages:
            pages.append(candidate)

    return pages


# ==========================================================
# FULL OCR ONE PAGE
# ==========================================================

def ocr_page(args):

    page_number, image = args

    image = np.array(image)

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY,
    )

    gray = cv2.medianBlur(
        gray,
        3,
    )

    text = pytesseract.image_to_string(
        gray,
        lang="eng",
        config="--oem 3 --psm 6",
    )

    return page_number, text


# ==========================================================
# EXTRACT SECTIONS
# ==========================================================

def extract_sections(
    pages: dict[int, str],
) -> dict:

    sections = {
        "synopsis": "",
        "brief_facts": "",
        "prayer": "",
        "interim_prayer": "",
    }

    ordered_pages = sorted(pages.items())

    all_text = []

    for page_number, text in ordered_pages:

        all_text.append(
            f"\n========== PAGE {page_number} ==========\n"
        )

        all_text.append(text)

    full_text = "\n".join(all_text)

    lines = full_text.splitlines()

    matches = []

    current_page = None

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

        page_match = re.match(
            r"=+\s*PAGE\s+(\d+)\s*=+",
            line,
            re.IGNORECASE,
        )

        if page_match:
            current_page = int(page_match.group(1))
            continue

        for section in section_order:

            if is_heading(
                line,
                SECTION_HEADINGS[section],
            ):

                matches.append(
                    {
                        "section": section,
                        "line_index": index,
                        "page": current_page,
                    }
                )

                break

    # Keep first occurrence
    first_occurrence = {}

    for match in matches:

        section = match["section"]

        if section not in first_occurrence:
            first_occurrence[section] = match

    ordered_matches = sorted(
        first_occurrence.values(),
        key=lambda x: x["line_index"],
    )

    for i, current in enumerate(ordered_matches):

        section = current["section"]

        start = current["line_index"] + 1

        if i + 1 < len(ordered_matches):
            end = ordered_matches[i + 1]["line_index"]
        else:
            end = len(lines)

        extracted = []

        for raw_line in lines[start:end]:

            line = raw_line.strip()

            if not line:
                continue

            if is_signature(line):
                break

            if is_stop_heading(section, line):
                break

            page_match = re.match(
                r"=+\s*PAGE\s+(\d+)\s*=+",
                line,
                re.IGNORECASE,
            )

            if page_match:
                continue

            extracted.append(line)

        sections[section] = "\n".join(extracted).strip()

    return sections


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    print("\nLoading candidate pages...")

    # ------------------------------------------------------
    # Get unique candidate pages
    # ------------------------------------------------------

    raw_candidate_pages = set()

    for pages in CANDIDATE_PAGES.values():
        raw_candidate_pages.update(pages)

    raw_candidate_pages = sorted(raw_candidate_pages)

    print(
        "Candidate pages:",
        raw_candidate_pages,
    )

    # ------------------------------------------------------
    # Render ONLY candidate pages
    # ------------------------------------------------------

    render_start = perf_counter()

    images = convert_from_path(
        PDF_PATH,
        dpi=300,
        poppler_path=POPPLER_PATH,
        first_page=min(raw_candidate_pages),
        last_page=max(raw_candidate_pages),
    )

    # Because convert_from_path renders a continuous range,
    # map them to their actual page numbers.
    first_page = min(raw_candidate_pages)

    page_jobs = []

    for index, image in enumerate(images):
        page_number = first_page + index

        if page_number in raw_candidate_pages:
            page_jobs.append(
                (page_number, image)
            )

    render_time = perf_counter() - render_start

    # ------------------------------------------------------
    # Full OCR
    # ------------------------------------------------------

    print("\nRunning full OCR on candidate pages...")

    ocr_start = perf_counter()

    page_text = {}

    with ThreadPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:

        results = executor.map(
            ocr_page,
            page_jobs,
        )

        for page_number, text in results:

            page_text[page_number] = text

            print(
                f"OCR completed: page {page_number}"
            )

    ocr_time = perf_counter() - ocr_start

    # ------------------------------------------------------
    # Extract
    # ------------------------------------------------------

    print("\nExtracting sections...")

    extraction_start = perf_counter()

    sections = extract_sections(
        page_text
    )

    extraction_time = (
        perf_counter()
        - extraction_start
    )

    # ------------------------------------------------------
    # Save
    # ------------------------------------------------------

    output_file = (
        OUTPUT_DIR
        / f"{PDF_PATH.stem}_sections.txt"
    )

    with output_file.open(
        "w",
        encoding="utf-8",
    ) as file:

        file.write("=" * 80)
        file.write("\nSYNOPSIS\n")
        file.write("=" * 80)
        file.write("\n\n")
        file.write(
            sections["synopsis"]
            or "NOT FOUND"
        )
        file.write("\n\n")

        file.write("=" * 80)
        file.write("\nBRIEF FACTS OF THE CASE\n")
        file.write("=" * 80)
        file.write("\n\n")
        file.write(
            sections["brief_facts"]
            or "NOT FOUND"
        )
        file.write("\n\n")

        file.write("=" * 80)
        file.write("\nPRAYER\n")
        file.write("=" * 80)
        file.write("\n\n")
        file.write(
            sections["prayer"]
            or "NOT FOUND"
        )
        file.write("\n\n")

        file.write("=" * 80)
        file.write("\nINTERIM PRAYER\n")
        file.write("=" * 80)
        file.write("\n\n")
        file.write(
            sections["interim_prayer"]
            or "NOT FOUND"
        )
        file.write("\n\n")

        file.write("=" * 80)
        file.write("\nTIMING\n")
        file.write("=" * 80)
        file.write("\n\n")

        file.write(
            f"Candidate Pages : {raw_candidate_pages}\n"
        )

        file.write(
            f"Rendering Time  : {render_time:.2f} seconds\n"
        )

        file.write(
            f"Full OCR Time   : {ocr_time:.2f} seconds\n"
        )

        file.write(
            f"Extraction Time : {extraction_time:.4f} seconds\n"
        )

        file.write(
            f"Total Time      : "
            f"{render_time + ocr_time + extraction_time:.2f} seconds\n"
        )

    # ------------------------------------------------------
    # Final log
    # ------------------------------------------------------

    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)

    print(
        f"Candidate Pages : {raw_candidate_pages}"
    )

    print(
        f"Rendering Time  : {render_time:.2f} sec"
    )

    print(
        f"Full OCR Time   : {ocr_time:.2f} sec"
    )

    print(
        f"Extraction Time : {extraction_time:.4f} sec"
    )

    print(
        f"Total Time      : "
        f"{render_time + ocr_time + extraction_time:.2f} sec"
    )

    print(
        f"\nSaved to: {output_file}"
    )