from __future__ import annotations

import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from time import perf_counter
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pytesseract
from pdf2image import convert_from_path
from rapidfuzz.fuzz import ratio


# ==========================================================
# CONFIG
# ==========================================================

# ONLY CHANGE THIS
PDF_NAME = "WP-18309-2025-B.pdf"

PDF_FOLDER = Path(
    r"C:\Users\steph\OneDrive\Desktop\CEG\CCMS v2\uploads"
)

PDF_PATH = PDF_FOLDER / PDF_NAME

POPPLER_PATH = (
    r"C:\Users\steph\.cache\codex-runtimes"
    r"\codex-primary-runtime"
    r"\dependencies\native\poppler\Library\bin"
)

OUTPUT_DIR = Path("section_output")

FAST_DPI = 80
FULL_DPI = 300
MAX_WORKERS = min(8, os.cpu_count() or 4)


# ==========================================================
# DATA
# ==========================================================

@dataclass
class Candidate:
    section: str
    page: int
    score: float
    line: str
    kind: str


@dataclass
class SectionResult:
    text: str = ""
    start_page: int | None = None
    end_page: int | None = None
    heading: str = ""
    score: float = 0.0


# ==========================================================
# NORMALIZATION
# ==========================================================

def normalize(text: str) -> str:
    return " ".join(text.upper().split())


def compact(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", normalize(text))


# ==========================================================
# OCR
# ==========================================================

def ocr_image(
    image: np.ndarray,
    psm: int,
    fast: bool,
) -> str:

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY,
    )

    if not fast:
        gray = cv2.medianBlur(gray, 3)

    return pytesseract.image_to_string(
        gray,
        lang="eng",
        config=f"--oem 3 --psm {psm}",
    )


def fast_ocr_page(args):
    page, image = args

    return page, ocr_image(
        np.array(image),
        psm=11,
        fast=True,
    )


def fast_scan(pdf_path: Path):
    start = perf_counter()

    print("\nFast scanning PDF...")

    images = convert_from_path(
        pdf_path,
        dpi=FAST_DPI,
        poppler_path=POPPLER_PATH,
        thread_count=MAX_WORKERS,
    )

    total_pages = len(images)

    jobs = [
        (page, image)
        for page, image in enumerate(
            images,
            start=1,
        )
    ]

    page_text = {}

    with ThreadPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:

        for page, text in executor.map(
            fast_ocr_page,
            jobs,
        ):
            page_text[page] = text

    return (
        page_text,
        total_pages,
        perf_counter() - start,
    )


def full_ocr_page(args):
    page, image = args

    return page, ocr_image(
        np.array(image),
        psm=6,
        fast=False,
    )


def full_ocr_range(
    pdf_path: Path,
    start_page: int,
    end_page: int,
):
    images = convert_from_path(
        pdf_path,
        dpi=FULL_DPI,
        poppler_path=POPPLER_PATH,
        first_page=start_page,
        last_page=end_page,
        thread_count=MAX_WORKERS,
    )

    jobs = [
        (page, image)
        for page, image in enumerate(
            images,
            start=start_page,
        )
    ]

    results = {}

    with ThreadPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:

        for page, text in executor.map(
            full_ocr_page,
            jobs,
        ):
            results[page] = text

    return results


# ==========================================================
# PATTERNS
# ==========================================================

SYNOPSIS_PATTERNS = [
    "SYNOPSIS",
    "LIST OF DATES AND SYNOPSIS",
    "LIST OF DATES & SYNOPSIS",
    "LIST OF EVENTS AND SYNOPSIS",
    "LIST OF EVENTS WITH SYNOPSIS",
    "LIST OF EVENTS / SYNOPSIS",
    "EVENTS WITH SYNOPSIS",
    "BRIEF SYNOPSIS",
]

BRIEF_FACTS_PATTERNS = [
    "BRIEF FACTS",
    "BRIEF FACTS OF THE CASE",
    "FACTS IN BRIEF",
    "FACTS OF THE CASE",
    "SUBJECT IN BRIEF",
]

SIGNATURE_MARKERS = [
    "ADVOCATE FOR PETITIONER",
    "ADVOCATE FOR THE PETITIONER",
    "ADVOCATE FOR RESPONDENT",
    "ADVOCATE FOR THE RESPONDENT",
    "GOVT. PLEADER",
    "GOVERNMENT PLEADER",
    "HIGH COURT GOVT. PLEADER",
    "ADVOCATE FOR THE APPELLANTS",
    "ADVOCATE FOR APPELLANTS",
    "IDENTIFIED BY ME",
]


# ==========================================================
# HELPERS
# ==========================================================

def is_date_line(line: str) -> bool:
    return bool(
        re.search(
            r"\b\d{1,2}\s*[/.-]\s*\d{1,2}\s*[/.-]\s*\d{2,4}\b",
            line,
        )
    )


def is_signature(line: str) -> bool:
    normalized = normalize(line)

    return any(
        marker in normalized
        for marker in SIGNATURE_MARKERS
    )


def is_new_document(line: str) -> bool:
    normalized = normalize(line)

    return (
        normalized.startswith("AFFIDAVIT")
        or normalized.startswith("VERIFYING AFFIDAVIT")
        or (
            normalized.startswith("IN THE HIGH COURT")
            and (
                "BETWEEN" in normalized
                or "WRIT PETITION" in normalized
                or "WRIT APPEAL" in normalized
            )
        )
    )


def is_brief_facts_heading(line: str) -> bool:
    normalized = normalize(line)
    compact_line = compact(line)

    for target in BRIEF_FACTS_PATTERNS:
        target_normalized = normalize(target)

        if (
            normalized == target_normalized
            or target_normalized in normalized
        ):
            return True

    if "BRIEF" in compact_line:
        if (
            "FACT" in compact_line
            or "FCTS" in compact_line
            or "CTS" in compact_line
        ):
            return len(compact_line) <= 70

    if len(normalized) <= 70:
        if ratio(
            normalized,
            "BRIEF FACTS",
        ) >= 58:
            return True

    return False


def is_interim_prayer_marker(line: str) -> bool:
    normalized = normalize(line)
    compact_line = compact(line)

    if "INTERIMPRAYER" in compact_line:
        return True

    if "INTENIMPRAYER" in compact_line:
        return True

    if normalized in {
        "IL PRAYER",
        "II PRAYER",
        "INT PRAYER",
        "INTE PRAYER",
    }:
        return True

    if (
        "PRAYER" in compact_line
        and (
            "INTER" in compact_line
            or "INTE" in compact_line
            or "INTENIM" in compact_line
        )
    ):
        return True

    return False


def is_prayer_heading(line: str) -> bool:
    normalized = normalize(line)

    if is_interim_prayer_marker(line):
        return False

    if normalized == "PRAYER":
        return True

    stripped = re.sub(
        r"^[\s\W]*(?:I|II|III|IV|V|VI|VII|VIII|IX|X)[\s\.\-:]+",
        "",
        normalized,
    )

    if stripped == "PRAYER":
        return True

    return (
        normalized.startswith("PRAYER ")
        and len(normalized) <= 35
    )


# ==========================================================
# CANDIDATE DETECTION
# ==========================================================

def heading_score(
    line: str,
    target: str,
) -> float:
    normalized = normalize(line)
    target = normalize(target)

    score = ratio(
        normalized,
        target,
    )

    if normalized == target:
        score += 100

    if target in normalized:
        score += 35

    if len(normalized) <= 80:
        score += 10

    if len(normalized) > 100:
        score -= 40

    return score


def synopsis_table_score(text: str) -> float:
    lines = [
        normalize(line)
        for line in text.splitlines()
        if line.strip()
    ]

    has_date = any(
        "DATE" in line
        for line in lines
    )

    has_event = any(
        "EVENT" in line
        for line in lines
    )

    date_count = sum(
        is_date_line(line)
        for line in lines
    )

    score = 0.0

    if has_date:
        score += 45

    if has_event:
        score += 45

    if date_count >= 2:
        score += 60

    if date_count >= 5:
        score += 20

    return score


def prayer_page_score(text: str) -> float:
    normalized = normalize(text)

    score = 0.0

    for phrase, weight in [
        ("WHEREFORE", 80),
        ("MOST RESPECTFULLY PRAYS", 80),
        ("MOST RESPECTFULLY PRAY", 60),
        ("MAY BE PLEASED TO", 35),
        ("WRIT, ORDER OR DIRECTION", 35),
        ("PASS SUCH OTHER ORDERS", 30),
        ("GRANT SUCH OTHER RELIEFS", 30),
        ("IN THE INTEREST OF JUSTICE AND EQUITY", 25),
    ]:
        if phrase in normalized:
            score += weight

    roman_items = len(
        re.findall(
            r"\(\s*(?:I|II|III|IV|V|VI|VII|VIII|IX|X)\s*\)",
            normalized,
        )
    )

    score += min(
        roman_items * 15,
        75,
    )

    if (
        "INTERIM PRAYER" in normalized
        or "INTENIM PRAYER" in normalized
        or "PENDING DISPOSAL" in normalized
    ):
        score -= 200

    return score


def collect_candidates(page_text):
    candidates = {
        "synopsis": [],
        "brief_facts": [],
        "prayer": [],
    }

    for page, text in page_text.items():

        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

        for line in lines:

            for target in SYNOPSIS_PATTERNS:

                score = heading_score(
                    line,
                    target,
                )

                if score >= 105:

                    candidates[
                        "synopsis"
                    ].append(
                        Candidate(
                            "synopsis",
                            page,
                            score,
                            line,
                            "heading",
                        )
                    )

                    break

        table_score = synopsis_table_score(text)

        if table_score >= 130:

            candidates[
                "synopsis"
            ].append(
                Candidate(
                    "synopsis",
                    page,
                    table_score,
                    "DATE / EVENTS TABLE",
                    "table",
                )
            )

        for line in lines:

            if is_brief_facts_heading(line):

                score = max(
                    heading_score(
                        line,
                        target,
                    )
                    for target in BRIEF_FACTS_PATTERNS
                )

                candidates[
                    "brief_facts"
                ].append(
                    Candidate(
                        "brief_facts",
                        page,
                        score,
                        line,
                        "heading",
                    )
                )

            if is_prayer_heading(line):

                candidates[
                    "prayer"
                ].append(
                    Candidate(
                        "prayer",
                        page,
                        180,
                        line,
                        "heading",
                    )
                )

        body_score = prayer_page_score(text)

        if body_score >= 100:

            candidates[
                "prayer"
            ].append(
                Candidate(
                    "prayer",
                    page,
                    body_score,
                    "PRAYER BODY",
                    "body",
                )
            )

    return candidates


# ==========================================================
# SECTION SELECTION
# ==========================================================

def select_synopsis(
    candidates,
    total_pages,
):
    front = [
        c
        for c in candidates
        if c.page <= min(
            15,
            total_pages,
        )
    ]

    if not front:
        front = candidates

    headings = [
        c
        for c in front
        if c.kind == "heading"
    ]

    if headings:
        return min(
            headings,
            key=lambda c: c.page,
        )

    return min(
        front,
        key=lambda c: c.page,
    ) if front else None


def find_primary_start(
    page_text,
    after_page,
):
    patterns = [
        "MEMORANDUM OF WRIT PETITION",
        "MEMORANDUM OF WRIT APPEAL",
        "MEMORANDUM OF PETITION",
        "WRIT PETITION UNDER ARTICLE",
        "WRIT PETITION UNDER ARTICLES",
    ]

    for page in sorted(page_text):

        if page < after_page:
            continue

        text = normalize(
            page_text[page]
        )

        if any(
            pattern in text
            for pattern in patterns
        ):
            return page

    return after_page + 1


def select_brief_facts(
    candidates,
    synopsis_page,
):
    valid = [
        c
        for c in candidates
        if c.page > synopsis_page
        and c.page <= synopsis_page + 15
    ]

    if not valid:
        return None

    return min(
        valid,
        key=lambda c: (
            c.page,
            -c.score,
        ),
    )


def select_prayer(
    candidates,
    brief_page,
    total_pages,
):
    valid = [
        c
        for c in candidates
        if c.page > brief_page
    ]

    if not valid:
        return None

    # Real pleading prayer should precede
    # later embedded annexures/old petitions.
    primary_window = [
        c
        for c in valid
        if c.page <= min(
            total_pages,
            brief_page + 25,
        )
    ]

    if primary_window:
        valid = primary_window

    headings = [
        c
        for c in valid
        if c.kind == "heading"
    ]

    if headings:
        return min(
            headings,
            key=lambda c: c.page,
        )

    return max(
        valid,
        key=lambda c: (
            c.score,
            -c.page,
        ),
    )


# ==========================================================
# IMPORTANT: FIND THE END OF BRIEF FACTS
# ==========================================================

def find_brief_facts_end(
    page_text,
    start_page,
    total_pages,
):
    """
    Brief Facts in these documents is a short front-matter
    section. Do NOT let it bleed into the Memorandum.

    We stop at the first strong transition:
      - signature block
      - Grounds
      - next formal pleading/document
      - or a strong page-level signature/footer.
    """

    max_page = min(
        total_pages,
        start_page + 10,
    )

    for page in range(
        start_page,
        max_page + 1,
    ):

        text = page_text.get(
            page,
            "",
        )

        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

        # --------------------------------------------------
        # Strong page-level signature test
        # --------------------------------------------------

        has_advocate = any(
            "ADVOCATE" in normalize(line)
            for line in lines
        )

        has_place = any(
            normalize(line).startswith("PLACE")
            for line in lines
        )

        has_date = any(
            normalize(line).startswith("DATE")
            for line in lines
        )

        # A short front-matter Brief Facts page ending
        # with advocate/place/date is the clean boundary.
        if (
            has_advocate
            and (
                has_place
                or has_date
            )
        ):
            return page

        # --------------------------------------------------
        # Normal textual stops
        # --------------------------------------------------

        for line in lines:

            normalized = normalize(line)

            if (
                normalized == "GROUNDS"
                or normalized.startswith("GROUNDS ")
            ):
                return page

            if normalized.startswith(
                "MEMORANDUM OF WRIT "
            ):
                return page

            if normalized.startswith(
                "MEMORANDUM OF PETITION"
            ):
                return page

            if normalized.startswith(
                "MEMORANDUM OF APPEAL"
            ):
                return page

            if normalized.startswith(
                "IN THE HIGH COURT"
            ) and page > start_page:
                return page

    return min(
        total_pages,
        start_page + 2,
    )


# ==========================================================
# FIND END OF PRAYER
# ==========================================================

def find_prayer_end(
    page_text,
    start_page,
    total_pages,
):
    max_page = min(
        total_pages,
        start_page + 8,
    )

    for page in range(
        start_page,
        max_page + 1,
    ):

        lines = [
            line.strip()
            for line in page_text.get(
                page,
                "",
            ).splitlines()
            if line.strip()
        ]

        for line in lines:

            if (
                is_interim_prayer_marker(
                    line
                )
                or is_signature(line)
                or is_new_document(line)
            ):
                return page

    return max_page


# ==========================================================
# EXTRACT TEXT
# ==========================================================

def find_section_start(
    text,
    section,
):
    lines = text.splitlines()

    for index, raw_line in enumerate(lines):

        line = raw_line.strip()

        if not line:
            continue

        normalized = normalize(line)

        if section == "synopsis":

            if normalized == "SYNOPSIS":
                return index + 1

            if (
                "DATE" in normalized
                and "EVENT" in normalized
            ):
                return index + 1

            if normalized in {
                "DATE",
                "DATES",
            }:
                return index + 1

            for pattern in SYNOPSIS_PATTERNS:

                if normalize(pattern) in normalized:
                    return index + 1

        elif section == "brief_facts":

            if is_brief_facts_heading(line):
                return index + 1

        elif section == "prayer":

            if is_prayer_heading(line):
                return index + 1

            if (
                "WHEREFORE" in normalized
                or "MOST RESPECTFULLY PRAY"
                in normalized
            ):
                return index

    return None


def extract_synopsis(
    text,
):
    start = find_section_start(
        text,
        "synopsis",
    )

    if start is None:
        return ""

    lines = text.splitlines()
    collected = []

    for line in lines[start:]:

        line = line.strip()

        if not line:
            continue

        if is_brief_facts_heading(line):
            break

        if is_signature(line):
            break

        collected.append(line)

    return "\n".join(
        collected
    ).strip()


def extract_brief_facts(
    text,
):
    start = find_section_start(
        text,
        "brief_facts",
    )

    if start is None:
        return ""

    lines = text.splitlines()
    collected = []

    for line in lines[start:]:

        line = line.strip()

        if not line:
            continue

        # Critical: Brief Facts must never continue
        # into the actual Memorandum.
        if is_new_document(line):
            break

        if (
            normalize(line) == "GROUNDS"
            or normalize(line).startswith(
                "GROUNDS "
            )
        ):
            break

        if is_prayer_heading(line):
            break

        if is_signature(line):
            break

        collected.append(line)

    return "\n".join(
        collected
    ).strip()


def extract_prayer(
    text,
):
    start = find_section_start(
        text,
        "prayer",
    )

    if start is None:
        return ""

    lines = text.splitlines()
    collected = []

    for line in lines[start:]:

        line = line.strip()

        if not line:
            continue

        if is_interim_prayer_marker(line):
            break

        if is_signature(line):
            break

        if is_new_document(line):
            break

        collected.append(line)

    return "\n".join(
        collected
    ).strip()


# ==========================================================
# CENTRALIZED EXTRACTOR
# ==========================================================

class CentralizedSectionExtractor:

    def __init__(
        self,
        output_dir: Path = OUTPUT_DIR,
    ):
        self.output_dir = Path(
            output_dir
        )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def process(
        self,
        pdf_path: Path,
    ):
        total_start = perf_counter()

        # --------------------------------------------------
        # FAST SCAN
        # --------------------------------------------------

        (
            fast_text,
            total_pages,
            fast_time,
        ) = fast_scan(
            pdf_path
        )

        candidates = collect_candidates(
            fast_text
        )

        # --------------------------------------------------
        # LOCATE
        # --------------------------------------------------

        synopsis = select_synopsis(
            candidates["synopsis"],
            total_pages,
        )

        if synopsis is None:

            locations = {
                "synopsis": None,
                "brief_facts": None,
                "prayer": None,
            }

        else:

            brief_facts = select_brief_facts(
                candidates["brief_facts"],
                synopsis.page,
            )

            prayer = (
                select_prayer(
                    candidates["prayer"],
                    brief_facts.page
                    if brief_facts
                    else synopsis.page,
                    total_pages,
                )
                if brief_facts
                else None
            )

            locations = {
                "synopsis": synopsis,
                "brief_facts": brief_facts,
                "prayer": prayer,
            }

        # --------------------------------------------------
        # RANGES
        # --------------------------------------------------

        ranges = {}

        if locations["synopsis"]:

            ranges["synopsis"] = (
                locations[
                    "synopsis"
                ].page,
                find_brief_facts_end(
                    fast_text,
                    locations[
                        "synopsis"
                    ].page,
                    total_pages,
                ),
            )

        if locations["brief_facts"]:

            ranges["brief_facts"] = (
                locations[
                    "brief_facts"
                ].page,
                find_brief_facts_end(
                    fast_text,
                    locations[
                        "brief_facts"
                    ].page,
                    total_pages,
                ),
            )

        if locations["prayer"]:

            ranges["prayer"] = (
                locations[
                    "prayer"
                ].page,
                find_prayer_end(
                    fast_text,
                    locations[
                        "prayer"
                    ].page,
                    total_pages,
                ),
            )

        # --------------------------------------------------
        # FULL OCR
        # --------------------------------------------------

        full_start = perf_counter()

        full_texts = {}

        for section, (
            start_page,
            end_page,
        ) in ranges.items():

            # Ensure at least one page.
            end_page = max(
                start_page,
                end_page,
            )

            full_texts[section] = (
                full_ocr_range(
                    pdf_path,
                    start_page,
                    end_page,
                )
            )

        full_time = (
            perf_counter()
            - full_start
        )

        # --------------------------------------------------
        # EXTRACTION
        # --------------------------------------------------

        extraction_start = (
            perf_counter()
        )

        extracted_sections = {
            "synopsis": "",
            "brief_facts": "",
            "prayer": "",
        }

        for section in extracted_sections:

            pages = full_texts.get(
                section,
                {},
            )

            combined = "\n".join(
                pages[page]
                for page in sorted(pages)
            )

            if section == "synopsis":

                extracted_sections[
                    section
                ] = extract_synopsis(
                    combined
                )

            elif section == "brief_facts":

                extracted_sections[
                    section
                ] = extract_brief_facts(
                    combined
                )

            elif section == "prayer":

                extracted_sections[
                    section
                ] = extract_prayer(
                    combined
                )

        extraction_time = (
            perf_counter()
            - extraction_start
        )

        total_time = (
            perf_counter()
            - total_start
        )

        result = {
            "pdf": str(pdf_path),
            "total_pages": total_pages,
            "locations": {
                section: (
                    asdict(candidate)
                    if candidate
                    else None
                )
                for section, candidate
                in locations.items()
            },
            "ranges": {
                section: {
                    "start": pages[0],
                    "end": pages[1],
                }
                for section, pages
                in ranges.items()
            },
            "sections": extracted_sections,
            "timing": {
                "fast_scan": round(
                    fast_time,
                    3,
                ),
                "full_ocr": round(
                    full_time,
                    3,
                ),
                "extraction": round(
                    extraction_time,
                    4,
                ),
                "total": round(
                    total_time,
                    3,
                ),
            },
        }

        output_path = self.save(
            pdf_path,
            result,
        )

        result["output"] = str(
            output_path
        )

        return result

    def save(
        self,
        pdf_path,
        result,
    ):
        output_path = (
            self.output_dir
            / f"{pdf_path.stem}_sections.txt"
        )

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as file:

            for section, title in [
                (
                    "synopsis",
                    "SYNOPSIS",
                ),
                (
                    "brief_facts",
                    "BRIEF FACTS OF THE CASE",
                ),
                (
                    "prayer",
                    "PRAYER",
                ),
            ]:

                file.write(
                    "=" * 80
                )
                file.write(
                    f"\n{title}\n"
                )
                file.write(
                    "=" * 80
                )
                file.write("\n\n")

                file.write(
                    result[
                        "sections"
                    ][section]
                    or "NOT FOUND"
                )

                file.write("\n\n")

            file.write(
                "=" * 80
            )
            file.write(
                "\nPROCESS INFORMATION\n"
            )
            file.write(
                "=" * 80
            )
            file.write("\n\n")

            file.write(
                f"PDF: "
                f"{result['pdf']}\n"
            )

            file.write(
                f"Total Pages: "
                f"{result['total_pages']}\n\n"
            )

            for section, location in result[
                "locations"
            ].items():

                file.write(
                    f"{section}: "
                    f"{location}\n"
                )

            file.write("\n")

            for section, pages in result[
                "ranges"
            ].items():

                file.write(
                    f"{section} range: "
                    f"{pages['start']}-"
                    f"{pages['end']}\n"
                )

            file.write("\n")

            for name, value in result[
                "timing"
            ].items():

                file.write(
                    f"{name}: "
                    f"{value:.3f} sec\n"
                )

        return output_path


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    print(
        f"\nDocument: {PDF_NAME}"
    )

    print(
        f"PDF Path: {PDF_PATH}"
    )

    if not PDF_PATH.exists():

        raise FileNotFoundError(
            f"\nPDF not found:\n{PDF_PATH}\n\n"
            "Change PDF_NAME at the top."
        )

    extractor = (
        CentralizedSectionExtractor()
    )

    result = extractor.process(
        PDF_PATH
    )

    print("\n")
    print("=" * 80)
    print("SELECTED SECTION LOCATIONS")
    print("=" * 80)

    for section, location in result[
        "locations"
    ].items():

        if location is None:

            print(
                f"{section:<15}: NOT FOUND"
            )

        else:

            print(
                f"{section:<15}: "
                f"page={location['page']}, "
                f"score={location['score']:.1f}, "
                f"type={location['kind']}, "
                f"line={location['line']!r}"
            )

    print("\n")
    print("=" * 80)
    print("SECTION RANGES")
    print("=" * 80)

    for section, pages in result[
        "ranges"
    ].items():

        print(
            f"{section:<15}: "
            f"{pages['start']}-"
            f"{pages['end']}"
        )

    print("\n")
    print("=" * 80)
    print("TIMING")
    print("=" * 80)

    for key, value in result[
        "timing"
    ].items():

        print(
            f"{key:<15}: "
            f"{value:.3f} sec"
        )

    print(
        "\nOutput:",
        result["output"],
    )
