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

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None


# ==========================================================
# CONFIG
# ==========================================================

PDF_PATH = Path(
    r"C:\Users\steph\OneDrive\Desktop\CEG\CCMS v2\uploads\WP-7049-2021-B.pdf"
)

POPPLER_PATH = (
    r"C:\Users\steph\.cache\codex-runtimes"
    r"\codex-primary-runtime"
    r"\dependencies\native\poppler\Library\bin"
)

OUTPUT_DIR = Path("section_output")

SEARCH_PAGES = 30

FAST_DPI = 80
FULL_DPI = 300

MAX_WORKERS = min(
    8,
    os.cpu_count() or 4,
)


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


# ==========================================================
# NORMALIZATION
# ==========================================================

def normalize(text: str) -> str:
    return " ".join(
        text.upper().split()
    )


def compact(text: str) -> str:
    return re.sub(
        r"[^A-Z0-9]",
        "",
        normalize(text),
    )


def is_date_line(line: str) -> bool:
    return bool(
        re.search(
            r"\b\d{1,2}\s*[/.-]\s*\d{1,2}\s*[/.-]\s*\d{2,4}\b",
            line,
        )
    )


def looks_like_year(line: str) -> bool:
    return bool(
        re.search(
            r"\b(?:19|20)\d{2}\b",
            line,
        )
    )


def has_numbered_row(line: str) -> bool:
    return bool(
        re.match(
            r"^\s*(?:[0-9]{1,2}|[IVX]{1,5})[\s.)-]",
            line,
            re.IGNORECASE,
        )
    )


# ==========================================================
# PDF PAGE COUNT
# ==========================================================

def get_total_pages(
    pdf_path: Path,
) -> int:

    if PdfReader is not None:

        try:
            return len(
                PdfReader(
                    str(pdf_path)
                ).pages
            )

        except Exception:
            pass

    try:
        import fitz

        document = fitz.open(
            str(pdf_path)
        )

        try:
            return document.page_count

        finally:
            document.close()

    except Exception:
        pass

    raise RuntimeError(
        "Unable to determine PDF page count. "
        "Install pypdf or PyMuPDF."
    )


# ==========================================================
# OCR
# ==========================================================

def ocr_image(
    image: np.ndarray,
    psm: int,
) -> str:

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY,
    )

    return pytesseract.image_to_string(
        gray,
        lang="eng",
        config=f"--oem 3 --psm {psm}",
    )


def fast_ocr_page(args):

    page, image = args

    return (
        page,
        ocr_image(
            np.array(image),
            11,
        ),
    )


def full_ocr_page(args):

    page, image = args

    image_array = np.array(
        image
    )

    gray = cv2.cvtColor(
        image_array,
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

    return (
        page,
        text,
    )


# ==========================================================
# FAST SCAN - FIRST 30 PAGES
# ==========================================================

def fast_scan(
    pdf_path: Path,
):

    start = perf_counter()

    total_pages = get_total_pages(
        pdf_path
    )

    scan_pages = min(
        SEARCH_PAGES,
        total_pages,
    )

    print(
        f"\nFast scanning first "
        f"{scan_pages} pages..."
    )

    images = convert_from_path(
        pdf_path,
        dpi=FAST_DPI,
        poppler_path=POPPLER_PATH,
        first_page=1,
        last_page=scan_pages,
        thread_count=MAX_WORKERS,
    )

    jobs = [
        (
            page,
            image,
        )
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


# ==========================================================
# RANGE MERGING
# ==========================================================

def merge_ranges(
    ranges: list[tuple[int, int]],
) -> list[tuple[int, int]]:

    if not ranges:
        return []

    sorted_ranges = sorted(
        (
            min(start, end),
            max(start, end),
        )
        for start, end in ranges
    )

    merged = [
        sorted_ranges[0]
    ]

    for start, end in sorted_ranges[1:]:

        prev_start, prev_end = (
            merged[-1]
        )

        if start <= prev_end + 1:

            merged[-1] = (
                prev_start,
                max(
                    prev_end,
                    end,
                ),
            )

        else:

            merged.append(
                (start, end)
            )

    return merged


# ==========================================================
# FULL OCR
# ==========================================================

def full_ocr_ranges(
    pdf_path: Path,
    ranges: list[tuple[int, int]],
) -> dict[int, str]:

    page_text: dict[int, str] = {}

    for start_page, end_page in merge_ranges(
        ranges
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
            (
                page,
                image,
            )
            for page, image in enumerate(
                images,
                start=start_page,
            )
        ]

        with ThreadPoolExecutor(
            max_workers=MAX_WORKERS
        ) as executor:

            for page, text in executor.map(
                full_ocr_page,
                jobs,
            ):
                page_text[page] = text

    return page_text


# ==========================================================
# SECTION PATTERNS
# ==========================================================

SYNOPSIS_PATTERNS = [
    "SYNOPSIS",
    "BRIEF SYNOPSIS",
    "LIST OF DATES AND SYNOPSIS",
    "LIST OF DATES & SYNOPSIS",
    "LIST OF EVENTS AND SYNOPSIS",
    "LIST OF EVENTS WITH SYNOPSIS",
    "LIST OF EVENTS / SYNOPSIS",
    "EVENTS WITH SYNOPSIS",
]

BRIEF_FACTS_PATTERNS = [
    "BRIEF FACTS OF THE CASE",
    "BRIEF FACTS",
    "FACTS IN BRIEF",
    "FACTS OF THE CASE",
    "SUBJECT IN BRIEF",
]

GROUNDS_PATTERNS = [
    "GROUNDS",
    "SALIENT GROUNDS",
    "GROUNDS OF CHALLENGE",
    "GROUNDS FOR INTERIM RELIEF",
    "GROUNDS FOR INTERIM PRAYER",
]

SIGNATURE_MARKERS = [
    "ADVOCATE FOR PETITIONER",
    "ADVOCATE FOR THE PETITIONER",
    "ADVOCATE FOR RESPONDENT",
    "ADVOCATE FOR THE RESPONDENT",
    "ADVOCATE FOR THE APPELLANTS",
    "ADVOCATE FOR APPELLANTS",
    "ADVOCATES FOR PETITIONER",
    "ADVOCATES FOR THE PETITIONER",
    "ADVOCATES FOR APPELLANTS",
    "ADVOCATE FOR THE PETITIONERS",
    "ADVOCATE FOR PETITIONERS",
    "HIGH COURT GOVT. PLEADER",
    "GOVT. PLEADER",
    "GOVERNMENT PLEADER",
    "IDENTIFIED BY ME",
]


# ==========================================================
# BASIC HELPERS
# ==========================================================

def is_signature(
    line: str,
) -> bool:

    normalized = normalize(line)

    return any(
        marker in normalized
        for marker in SIGNATURE_MARKERS
    )


def is_new_document(
    line: str,
) -> bool:

    normalized = normalize(line)

    return (
        normalized.startswith(
            "AFFIDAVIT"
        )
        or normalized.startswith(
            "VERIFYING AFFIDAVIT"
        )
        or (
            normalized.startswith(
                "IN THE HIGH COURT"
            )
            and (
                "BETWEEN"
                in normalized
                or "WRIT PETITION"
                in normalized
                or "WRIT APPEAL"
                in normalized
            )
        )
    )


def is_grounds_heading(
    line: str,
) -> bool:

    normalized = normalize(line)

    for target in GROUNDS_PATTERNS:

        if normalized == target:
            return True

        if (
            len(normalized) <= 90
            and ratio(
                normalized,
                target,
            ) >= 85
        ):
            return True

    return False


def is_place_heading(
    line: str,
) -> bool:

    normalized = normalize(line)

    return (
        normalized == "PLACE"
        or normalized.startswith(
            "PLACE:"
        )
        or normalized.startswith(
            "PLACE "
        )
    )


# ==========================================================
# SYNOPSIS HEADING
# ==========================================================

def is_synopsis_heading(
    line: str,
) -> bool:

    normalized = normalize(line)
    compact_line = compact(line)

    if not normalized:
        return False

    for target in SYNOPSIS_PATTERNS:

        target_compact = compact(
            target
        )

        if compact_line == target_compact:
            return True

        if (
            len(normalized) <= 90
            and ratio(
                normalized,
                target,
            ) >= 86
        ):
            return True

        if (
            target_compact in compact_line
            and len(compact_line) <= 90
        ):
            return True

    return False


# ==========================================================
# BRIEF FACTS HEADING
# ==========================================================

def is_brief_facts_heading(
    line: str,
) -> bool:

    normalized = normalize(line)
    compact_line = compact(line)

    if not normalized:
        return False

    for target in BRIEF_FACTS_PATTERNS:

        target_normalized = normalize(
            target
        )

        if (
            normalized == target_normalized
            or (
                len(normalized) <= 90
                and target_normalized in normalized
            )
        ):
            return True

        if (
            len(normalized) <= 90
            and ratio(
                normalized,
                target_normalized,
            ) >= 84
        ):
            return True

    if (
        "BRIEF" in compact_line
        and (
            "FACT" in compact_line
            or "FCTS" in compact_line
            or "CTS" in compact_line
        )
        and len(compact_line) <= 80
    ):
        return True

    if (
        len(normalized) <= 80
        and ratio(
            normalized,
            "BRIEF FACTS",
        ) >= 65
    ):
        return True

    return False


# ==========================================================
# INTERIM PRAYER
# ==========================================================

def is_interim_prayer_marker(
    line: str,
) -> bool:

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


# ==========================================================
# PRAYER HEADING
# ==========================================================

def is_prayer_heading(
    line: str,
) -> bool:

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
        and len(normalized) <= 40
    )


# ==========================================================
# HEADING SCORE
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


# ==========================================================
# INDEX PAGE
# ==========================================================

def is_index_page(
    text: str,
) -> bool:

    normalized = normalize(text)

    if "INDEX" not in normalized:
        return False

    strong_index_signals = 0

    for phrase in [
        "PAGE NO",
        "PAGE NOS",
        "PARTICULARS",
        "DESCRIPTION",
        "MEMORANDUM OF WRIT PETITION",
        "VAKALATH",
        "ANNEXURE",
        "VERIFYING AFFIDAVIT",
    ]:

        if phrase in normalized:
            strong_index_signals += 1

    return strong_index_signals >= 2


# ==========================================================
# SYNOPSIS TABLE
# ==========================================================

def synopsis_table_score(
    text: str,
) -> float:

    lines = [
        normalize(line)
        for line in text.splitlines()
        if line.strip()
    ]

    page_text = "\n".join(lines)

    if is_index_page(text):
        return 0.0

    has_date = (
        "DATE" in page_text
        or "DATES" in page_text
    )

    has_events = (
        "EVENT" in page_text
        or "EVENTS" in page_text
    )

    has_descriptions = (
        "DESCRIPTION" in page_text
        or "DESCRIPTIONS" in page_text
    )

    has_particulars = (
        "PARTICULAR" in page_text
        or "PARTICULARS" in page_text
    )

    has_content_column = (
        has_events
        or has_descriptions
        or has_particulars
    )

    has_synopsis_heading = any(
        is_synopsis_heading(line)
        for line in lines
    )

    has_sl_no = any(
        token in page_text
        for token in [
            "SL. NO",
            "SL NO",
            "SI. NO",
            "SI NO",
            "SL.NO",
            "SI.NO",
        ]
    )

    numeric_rows = sum(
        1
        for line in lines
        if has_numbered_row(line)
    )

    nil_rows = sum(
        1
        for line in lines
        if "NIL" in line
    )

    date_rows = sum(
        1
        for line in lines
        if (
            is_date_line(line)
            or looks_like_year(line)
        )
    )

    if not has_date:
        return 0.0

    if not has_content_column:
        return 0.0

    score = 0.0

    if has_synopsis_heading:
        score += 170

    if has_sl_no:
        score += 55

    if has_events:
        score += 50

    if has_descriptions:
        score += 65

    if has_particulars:
        score += 55

    if numeric_rows >= 2:
        score += 45

    if date_rows >= 2:
        score += 35

    if nil_rows >= 1:
        score += 20

    if (
        has_synopsis_heading
        and has_date
        and has_content_column
    ):
        score += 80

    return score


# ==========================================================
# PRAYER BODY SCORE
# ==========================================================

def prayer_page_score(
    text: str,
) -> float:

    normalized = normalize(text)

    score = 0.0

    signals = [
        ("WHEREFORE", 80),
        ("MOST RESPECTFULLY PRAYS", 80),
        ("MOST RESPECTFULLY PRAY", 65),
        ("MAY BE PLEASED TO", 35),
        ("WRIT, ORDER OR DIRECTION", 35),
        ("PASS SUCH OTHER ORDERS", 30),
        ("GRANT SUCH OTHER RELIEFS", 30),
        (
            "IN THE INTEREST OF JUSTICE AND EQUITY",
            25,
        ),
    ]

    for phrase, weight in signals:

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


# ==========================================================
# COLLECT CANDIDATES
# ==========================================================

def collect_candidates(
    page_text: dict[int, str],
):

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

        index_page = is_index_page(text)

        if not index_page:

            for line in lines:

                if is_synopsis_heading(line):

                    score = max(
                        heading_score(
                            line,
                            target,
                        )
                        for target in SYNOPSIS_PATTERNS
                    )

                    candidates["synopsis"].append(
                        Candidate(
                            "synopsis",
                            page,
                            score,
                            line,
                            "heading",
                        )
                    )

        table_score = synopsis_table_score(text)

        if table_score >= 150:

            candidates["synopsis"].append(
                Candidate(
                    "synopsis",
                    page,
                    table_score,
                    "SYNOPSIS TABLE",
                    "table",
                )
            )

        if not index_page:

            for line in lines:

                if is_brief_facts_heading(line):

                    score = max(
                        heading_score(
                            line,
                            target,
                        )
                        for target in BRIEF_FACTS_PATTERNS
                    )

                    candidates["brief_facts"].append(
                        Candidate(
                            "brief_facts",
                            page,
                            score,
                            line,
                            "heading",
                        )
                    )

                if is_prayer_heading(line):

                    candidates["prayer"].append(
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

                candidates["prayer"].append(
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
# SELECT SYNOPSIS
# ==========================================================

def select_synopsis(
    candidates,
    page_text,
    total_pages,
):

    valid = []

    for candidate in candidates:

        text = page_text.get(
            candidate.page,
            "",
        )

        if candidate.kind == "heading":

            if is_index_page(text):
                continue

            line_normalized = normalize(
                candidate.line
            )

            is_list_of_dates_heading = (
                "LIST OF DATES"
                in line_normalized
            )

            if is_list_of_dates_heading:

                if synopsis_table_score(text) < 150:
                    continue

            elif not is_synopsis_heading(
                candidate.line
            ):
                continue

        elif candidate.kind == "table":

            if candidate.page > SEARCH_PAGES:
                continue

            if synopsis_table_score(text) < 150:
                continue

        valid.append(candidate)

    if not valid:
        return None

    tables = [
        c
        for c in valid
        if c.kind == "table"
    ]

    if tables:

        return min(
            tables,
            key=lambda c: (
                c.page,
                -c.score,
            ),
        )

    standalone_headings = [
        c
        for c in valid
        if (
            c.kind == "heading"
            and "LIST OF DATES"
            not in normalize(c.line)
        )
    ]

    if standalone_headings:

        return min(
            standalone_headings,
            key=lambda c: (
                c.page,
                -c.score,
            ),
        )

    return min(
        valid,
        key=lambda c: (
            c.page,
            -c.score,
        ),
    )


# ==========================================================
# SELECT BRIEF FACTS
# ==========================================================

def select_brief_facts(
    candidates,
    synopsis_page,
    total_pages,
):

    valid = [
        c
        for c in candidates
        if c.page >= synopsis_page
        and c.page <= min(
            SEARCH_PAGES,
            total_pages,
            synopsis_page + 15,
        )
    ]

    if not valid:
        return None

    same_page = [
        c
        for c in valid
        if c.page == synopsis_page
    ]

    if same_page:

        return max(
            same_page,
            key=lambda c: c.score,
        )

    return min(
        valid,
        key=lambda c: (
            c.page,
            -c.score,
        ),
    )


# ==========================================================
# SELECT PRAYER
# ==========================================================

def select_prayer(
    candidates,
    after_page,
    total_pages,
):

    valid = [
        c
        for c in candidates
        if c.page >= after_page
        and c.page <= min(
            SEARCH_PAGES,
            total_pages,
        )
    ]

    if not valid:
        return None

    headings = [
        c
        for c in valid
        if c.kind == "heading"
    ]

    if headings:

        return min(
            headings,
            key=lambda c: (
                c.page,
                -c.score,
            ),
        )

    body = [
        c
        for c in valid
        if c.kind == "body"
    ]

    if body:

        return min(
            body,
            key=lambda c: (
                c.page,
                -c.score,
            ),
        )

    return None


# ==========================================================
# FIND SYNOPSIS END
# ==========================================================

def find_synopsis_end(
    page_text,
    start_page,
    total_pages,
):

    max_page = min(
        SEARCH_PAGES,
        total_pages,
        start_page + 15,
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
                is_brief_facts_heading(line)
                or is_grounds_heading(line)
                or is_prayer_heading(line)
                or is_new_document(line)
            ):
                return page

    return max_page


# ==========================================================
# FIND BRIEF FACTS END
# ==========================================================

def find_brief_facts_end(
    page_text,
    start_page,
    total_pages,
):

    max_page = min(
        SEARCH_PAGES,
        total_pages,
        start_page + 15,
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

        for index, line in enumerate(lines):

            normalized = normalize(line)

            if is_signature(line):
                return page

            if (
                "ADVOCATE" in normalized
                and (
                    "PETITIONER" in normalized
                    or "PETITIONERS" in normalized
                    or "APPELLANT" in normalized
                    or "APPELLANTS" in normalized
                    or "RESPONDENT" in normalized
                )
            ):
                return page

            if is_place_heading(line):
                return page

            if normalized.startswith("DATE"):

                nearby = " ".join(
                    normalize(x)
                    for x in lines[
                        index:min(
                            index + 5,
                            len(lines),
                        )
                    ]
                )

                if (
                    "ADVOCATE" in nearby
                    or "PETITIONER" in nearby
                    or "APPELLANT" in nearby
                ):
                    return page

            if is_grounds_heading(line):
                return page

            if is_prayer_heading(line):
                return page

            if is_new_document(line):
                return page

    return min(
        total_pages,
        start_page + 2,
    )


# ==========================================================
# FIND PRAYER END
#
# IMPORTANT:
# Do NOT stop Prayer on its starting page because of
# OCR signature artifacts.
# ==========================================================

def find_prayer_end(
    page_text,
    start_page,
    total_pages,
):

    max_page = min(
        SEARCH_PAGES,
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

        for index, line in enumerate(lines):

            normalized = normalize(line)

            # --------------------------------------------------
            # These are hard boundaries even on start page.
            # --------------------------------------------------

            if is_interim_prayer_marker(line):
                return page

            if is_new_document(line):
                return page

            if is_grounds_heading(line):
                return page

            # --------------------------------------------------
            # IMPORTANT:
            # Never terminate Prayer on starting page due to
            # Place / Advocate / signature OCR artifacts.
            # --------------------------------------------------

            if page == start_page:
                continue

            # --------------------------------------------------
            # Subsequent-page signature block.
            # --------------------------------------------------

            if is_signature(line):
                return page

            if (
                "ADVOCATE" in normalized
                and (
                    "PETITIONER" in normalized
                    or "PETITIONERS" in normalized
                    or "APPELLANT" in normalized
                    or "APPELLANTS" in normalized
                    or "RESPONDENT" in normalized
                )
            ):
                return page

            if is_place_heading(line):
                return page

            if normalized.startswith("DATE"):

                nearby = " ".join(
                    normalize(x)
                    for x in lines[
                        index:min(
                            index + 5,
                            len(lines),
                        )
                    ]
                )

                if (
                    "ADVOCATE" in nearby
                    or "PETITIONER" in nearby
                    or "APPELLANT" in nearby
                ):
                    return page

    return max_page


# ==========================================================
# SYNOPSIS TABLE START
# ==========================================================

def find_table_synopsis_start(
    lines,
):

    for index, raw_line in enumerate(lines):

        if not is_synopsis_heading(raw_line):
            continue

        nearby = lines[
            index:min(
                index + 25,
                len(lines),
            )
        ]

        nearby_text = normalize(
            "\n".join(nearby)
        )

        has_date = (
            "DATE" in nearby_text
            or "DATES" in nearby_text
        )

        has_content = (
            "EVENT" in nearby_text
            or "EVENTS" in nearby_text
            or "DESCRIPTION" in nearby_text
            or "DESCRIPTIONS" in nearby_text
            or "PARTICULAR" in nearby_text
            or "PARTICULARS" in nearby_text
        )

        if (
            has_date
            and has_content
        ):
            return index + 1

    for index, raw_line in enumerate(lines):

        normalized = normalize(raw_line)

        if not any(
            phrase in normalized
            for phrase in [
                "LIST OF DATES AND SYNOPSIS",
                "LIST OF DATES & SYNOPSIS",
                "LIST OF EVENTS AND SYNOPSIS",
                "LIST OF EVENTS WITH SYNOPSIS",
                "LIST OF EVENTS / SYNOPSIS",
            ]
        ):
            continue

        nearby = lines[
            index:min(
                index + 25,
                len(lines),
            )
        ]

        nearby_text = normalize(
            "\n".join(nearby)
        )

        has_date = (
            "DATE" in nearby_text
            or "DATES" in nearby_text
        )

        has_content = (
            "EVENT" in nearby_text
            or "DESCRIPTION" in nearby_text
            or "PARTICULAR" in nearby_text
        )

        if (
            has_date
            and has_content
        ):
            return index + 1

    return None


# ==========================================================
# SECTION START
# ==========================================================

def find_section_start(
    text,
    section,
    candidate_kind=None,
):

    lines = text.splitlines()

    if section == "synopsis":

        table_start = (
            find_table_synopsis_start(lines)
        )

        if table_start is not None:
            return table_start

        for index, line in enumerate(lines):

            if is_synopsis_heading(line):
                return index + 1

        return None

    if section == "brief_facts":

        for index, line in enumerate(lines):

            if is_brief_facts_heading(line):
                return index + 1

        return None

    if section == "prayer":

        for index, line in enumerate(lines):

            if is_prayer_heading(line):
                return index + 1

        for index, line in enumerate(lines):

            normalized = normalize(line)

            if (
                "WHEREFORE" in normalized
                or "MOST RESPECTFULLY PRAY"
                in normalized
            ):
                return index

        return None

    return None


# ==========================================================
# EXTRACT SYNOPSIS
# ==========================================================

def extract_synopsis(
    text,
    candidate_kind=None,
):

    start = find_section_start(
        text,
        "synopsis",
        candidate_kind,
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

        if is_grounds_heading(line):
            break

        if is_prayer_heading(line):
            break

        if is_new_document(line):
            break

        if is_signature(line):
            break

        normalized = normalize(line)

        if (
            "ADVOCATE" in normalized
            and (
                "PETITIONER" in normalized
                or "PETITIONERS" in normalized
                or "APPELLANT" in normalized
                or "APPELLANTS" in normalized
                or "RESPONDENT" in normalized
            )
        ):
            break

        if is_place_heading(line):
            break

        collected.append(line)

    return "\n".join(collected).strip()


# ==========================================================
# BRIEF FACTS FOOTER DETECTOR
# ==========================================================

def is_brief_facts_footer(
    lines,
    index,
) -> bool:

    line = lines[index]
    normalized = normalize(line)

    if is_signature(line):
        return True

    if is_place_heading(line):
        return True

    if normalized.startswith("DATE"):
        return True

    if (
        "ADVOCATE" in normalized
        and (
            "PETITIONER" in normalized
            or "PETITIONERS" in normalized
            or "APPELLANT" in normalized
            or "APPELLANTS" in normalized
            or "RESPONDENT" in normalized
        )
    ):
        return True

    # Common city/footer appearing at very bottom.
    if (
        index >= max(
            0,
            len(lines) - 6,
        )
        and normalized in {
            "BENGALURU",
            "BANGALORE",
            "DHARWAD",
            "MYSURU",
            "MYSORE",
            "KALABURAGI",
            "BELAGAVI",
            "HUBBALLI",
            "MANGALURU",
        }
    ):
        return True

    return False


# ==========================================================
# EXTRACT BRIEF FACTS
# ==========================================================

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

    for index in range(
        start,
        len(lines),
    ):

        line = lines[index].strip()

        if not line:
            continue

        normalized = normalize(line)

        if is_brief_facts_footer(
            lines,
            index,
        ):
            break

        if is_new_document(line):
            break

        if is_grounds_heading(line):
            break

        if is_prayer_heading(line):
            break

        if is_interim_prayer_marker(line):
            break

        collected.append(line)

    return "\n".join(collected).strip()


# ==========================================================
# EXTRACT PRAYER
#
# Multi-page Prayer is preserved.
# ==========================================================

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

    for index in range(
        start,
        len(lines),
    ):

        line = lines[index].strip()

        if not line:
            continue

        normalized = normalize(line)

        # --------------------------------------------------
        # Hard boundary: Interim Prayer.
        # --------------------------------------------------

        if is_interim_prayer_marker(line):
            break

        # --------------------------------------------------
        # Hard boundary: new document.
        # --------------------------------------------------

        if is_new_document(line):
            break

        # --------------------------------------------------
        # Hard boundary: Grounds.
        # --------------------------------------------------

        if is_grounds_heading(line):
            break

        # --------------------------------------------------
        # Signature block.
        # --------------------------------------------------

        if is_place_heading(line):
            break

        if normalized.startswith("DATE"):

            nearby = " ".join(
                normalize(x)
                for x in lines[
                    index:min(
                        index + 5,
                        len(lines),
                    )
                ]
            )

            if (
                "ADVOCATE" in nearby
                or "PETITIONER" in nearby
                or "APPELLANT" in nearby
            ):
                break

        if (
            "ADVOCATE" in normalized
            and (
                "PETITIONER" in normalized
                or "PETITIONERS" in normalized
                or "APPELLANT" in normalized
                or "APPELLANTS" in normalized
                or "RESPONDENT" in normalized
            )
        ):
            break

        # Raw signature OCR artifacts are ignored,
        # not treated as a section boundary.
        if is_signature(line):
            continue

        collected.append(line)

    return "\n".join(collected).strip()


# ==========================================================
# PROCESS DOCUMENT
# ==========================================================

def process_document(
    pdf_path: Path,
):

    total_start = perf_counter()

    # ------------------------------------------------------
    # FAST SCAN
    # ------------------------------------------------------

    (
        fast_text,
        total_pages,
        fast_time,
    ) = fast_scan(
        pdf_path
    )

    # ------------------------------------------------------
    # CANDIDATES
    # ------------------------------------------------------

    candidates = collect_candidates(
        fast_text
    )

    # ------------------------------------------------------
    # SYNOPSIS
    # ------------------------------------------------------

    synopsis = select_synopsis(
        candidates["synopsis"],
        fast_text,
        total_pages,
    )

    # ------------------------------------------------------
    # BRIEF FACTS
    # ------------------------------------------------------

    brief_facts = None

    if synopsis:

        brief_facts = select_brief_facts(
            candidates["brief_facts"],
            synopsis.page,
            total_pages,
        )

    # ------------------------------------------------------
    # PRAYER
    # ------------------------------------------------------

    prayer_start = 1

    if brief_facts:
        prayer_start = brief_facts.page

    elif synopsis:
        prayer_start = synopsis.page

    prayer = select_prayer(
        candidates["prayer"],
        prayer_start,
        total_pages,
    )

    # ------------------------------------------------------
    # TARGETED FALLBACK
    # ------------------------------------------------------

    if (
        synopsis is None
        or brief_facts is None
        or prayer is None
    ):

        print(
            "\nTargeted 120-DPI scan "
            "for first 30 pages..."
        )

        scan_pages = min(
            SEARCH_PAGES,
            total_pages,
        )

        fallback_images = convert_from_path(
            pdf_path,
            dpi=120,
            poppler_path=POPPLER_PATH,
            first_page=1,
            last_page=scan_pages,
            thread_count=MAX_WORKERS,
        )

        fallback_jobs = [
            (
                page,
                image,
            )
            for page, image in enumerate(
                fallback_images,
                start=1,
            )
        ]

        fallback_text = {}

        def fallback_ocr(args):

            page, image = args

            gray = cv2.cvtColor(
                np.array(image),
                cv2.COLOR_RGB2GRAY,
            )

            text = pytesseract.image_to_string(
                gray,
                lang="eng",
                config="--oem 3 --psm 6",
            )

            return (
                page,
                text,
            )

        with ThreadPoolExecutor(
            max_workers=MAX_WORKERS
        ) as executor:

            for page, text in executor.map(
                fallback_ocr,
                fallback_jobs,
            ):
                fallback_text[page] = text

        fallback_candidates = collect_candidates(
            fallback_text
        )

        fallback_synopsis = select_synopsis(
            fallback_candidates["synopsis"],
            fallback_text,
            total_pages,
        )

        if fallback_synopsis:
            synopsis = fallback_synopsis

        if synopsis:

            fallback_brief = select_brief_facts(
                fallback_candidates["brief_facts"],
                synopsis.page,
                total_pages,
            )

            if fallback_brief:
                brief_facts = fallback_brief

        prayer_start = 1

        if brief_facts:
            prayer_start = brief_facts.page

        elif synopsis:
            prayer_start = synopsis.page

        fallback_prayer = select_prayer(
            fallback_candidates["prayer"],
            prayer_start,
            total_pages,
        )

        if fallback_prayer:
            prayer = fallback_prayer

        fast_text.update(
            fallback_text
        )

    # ------------------------------------------------------
    # LOCATIONS
    # ------------------------------------------------------

    locations = {
        "synopsis": synopsis,
        "brief_facts": brief_facts,
        "prayer": prayer,
    }

    # ------------------------------------------------------
    # RANGES
    # ------------------------------------------------------

    ranges = {}

    if synopsis:

        if brief_facts:
            synopsis_end = brief_facts.page

        else:
            synopsis_end = find_synopsis_end(
                fast_text,
                synopsis.page,
                total_pages,
            )

        ranges["synopsis"] = (
            synopsis.page,
            synopsis_end,
        )

    if brief_facts:

        ranges["brief_facts"] = (
            brief_facts.page,
            find_brief_facts_end(
                fast_text,
                brief_facts.page,
                total_pages,
            ),
        )

    if prayer:

        ranges["prayer"] = (
            prayer.page,
            find_prayer_end(
                fast_text,
                prayer.page,
                total_pages,
            ),
        )

    # ------------------------------------------------------
    # FULL OCR
    # ------------------------------------------------------

    full_start = perf_counter()

    full_text_by_page = full_ocr_ranges(
        pdf_path,
        list(ranges.values()),
    )

    full_time = (
        perf_counter()
        - full_start
    )

    # ------------------------------------------------------
    # EXTRACTION
    # ------------------------------------------------------

    extraction_start = perf_counter()

    sections = {
        "synopsis": "",
        "brief_facts": "",
        "prayer": "",
    }

    for section in sections:

        if section not in ranges:
            continue

        start_page, end_page = ranges[section]

        section_text = "\n".join(
            full_text_by_page[page]
            for page in sorted(
                full_text_by_page
            )
            if (
                start_page
                <= page
                <= end_page
            )
        )

        candidate = locations[section]

        if section == "synopsis":

            sections["synopsis"] = extract_synopsis(
                section_text,
                candidate.kind
                if candidate
                else None,
            )

        elif section == "brief_facts":

            sections["brief_facts"] = extract_brief_facts(
                section_text
            )

        else:

            sections["prayer"] = extract_prayer(
                section_text
            )

    extraction_time = (
        perf_counter()
        - extraction_start
    )

    total_time = (
        perf_counter()
        - total_start
    )

    return {
        "pdf": str(pdf_path),
        "total_pages": total_pages,
        "search_pages": min(
            SEARCH_PAGES,
            total_pages,
        ),
        "locations": {
            section: (
                asdict(candidate)
                if candidate
                else None
            )
            for section, candidate in locations.items()
        },
        "ranges": {
            section: {
                "start": pages[0],
                "end": pages[1],
            }
            for section, pages in ranges.items()
        },
        "sections": sections,
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


# ==========================================================
# SAVE OUTPUT
# ==========================================================

def save_output(
    pdf_path: Path,
    result: dict,
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR
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

            file.write("=" * 80)
            file.write(f"\n{title}\n")
            file.write("=" * 80)
            file.write("\n\n")

            file.write(
                result["sections"][section]
                or "NOT FOUND"
            )

            file.write("\n\n")

        file.write("=" * 80)
        file.write("\nPROCESS INFORMATION\n")
        file.write("=" * 80)
        file.write("\n\n")

        file.write(
            f"PDF: {result['pdf']}\n"
        )

        file.write(
            f"Total Pages: "
            f"{result['total_pages']}\n"
        )

        file.write(
            f"Pages Scanned for Discovery: "
            f"{result['search_pages']}\n\n"
        )

        for section, location in result["locations"].items():

            file.write(
                f"{section}: "
                f"{location}\n"
            )

        file.write("\n")

        for section, pages in result["ranges"].items():

            file.write(
                f"{section} range: "
                f"{pages['start']}-"
                f"{pages['end']}\n"
            )

        file.write("\n")

        for key, value in result["timing"].items():

            file.write(
                f"{key}: "
                f"{value:.3f} sec\n"
            )

    return output_path


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    print(
        f"\nDocument: "
        f"{PDF_PATH.name}"
    )

    print(
        f"PDF Path: "
        f"{PDF_PATH}"
    )

    if not PDF_PATH.exists():

        raise FileNotFoundError(
            f"\nPDF not found:\n"
            f"{PDF_PATH}"
        )

    result = process_document(
        PDF_PATH
    )

    output_path = save_output(
        PDF_PATH,
        result
    )

    print("\n")
    print("=" * 80)
    print("SELECTED SECTION LOCATIONS")
    print("=" * 80)

    for section, location in result["locations"].items():

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

    for section, pages in result["ranges"].items():

        print(
            f"{section:<15}: "
            f"{pages['start']}-"
            f"{pages['end']}"
        )

    print("\n")
    print("=" * 80)
    print("TIMING")
    print("=" * 80)

    for key, value in result["timing"].items():

        print(
            f"{key:<25}: "
            f"{value:.3f} sec"
        )

    print(
        "\nOutput:",
        output_path,
    )