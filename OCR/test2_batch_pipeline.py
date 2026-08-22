
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

INPUT_DIR = Path(
    r"C:\Users\steph\OneDrive\Desktop\CEG\CCMS v2\uploads"
)

OUTPUT_DIR = Path("section_output")

POPPLER_PATH = (
    r"C:\Users\steph\.cache\codex-runtimes"
    r"\codex-primary-runtime"
    r"\dependencies\native\poppler\Library\bin"
)

SEARCH_PAGES = 30
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


# ==========================================================
# NORMALIZATION
# ==========================================================

def normalize(text: str) -> str:
    return " ".join(text.upper().split())


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


def clean_lines(text: str) -> list[str]:
    return [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]


# ==========================================================
# PDF PAGE COUNT
# ==========================================================

def get_total_pages(pdf_path: Path) -> int:

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

    except Exception as exc:
        raise RuntimeError(
            "Unable to determine PDF page count. "
            "Install pypdf or PyMuPDF."
        ) from exc


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

        prev_start, prev_end = merged[-1]

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
                "BETWEEN" in normalized
                or "WRIT PETITION" in normalized
                or "WRIT APPEAL" in normalized
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
        or normalized.startswith("PLACE:")
        or normalized.startswith("PLACE ")
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

        target_compact = compact(target)

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

        target_normalized = normalize(target)

        if normalized == target_normalized:
            return True

        if (
            len(normalized) <= 100
            and target_normalized in normalized
        ):
            return True

        if (
            len(normalized) <= 100
            and ratio(
                normalized,
                target_normalized,
            ) >= 80
        ):
            return True

    # OCR variants:
    # BRIEF FACTS
    # BRIEF-FACTS
    # RIEF FACTS
    # BRIF FACTS
    if re.match(
        r"^(?:B?RIEF|BRIF)[\s\-.:]+FACTS"
        r"(?:\s+OF\s+THE\s+CASE)?$",
        normalized,
    ):
        return True

    if (
        "FACTS" in compact_line
        and (
            "BRIEF" in compact_line
            or "BRIF" in compact_line
            or "RIEF" in compact_line
        )
    ):
        return True

    if (
        len(compact_line) <= 100
        and (
            compact_line.startswith("BRIEFFACTS")
            or compact_line.startswith("RIEFFACTS")
            or compact_line.startswith("BRIFFACTS")
        )
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

    if not normalized:
        return False

    if "GROUNDS FOR INTERIM PRAYER" in normalized:
        return False

    stripped = re.sub(
        r"^[\s\W]*(?:[0-9]{1,3}|[IVX]{1,5})[\s.\-:)]+",
        "",
        normalized,
    )

    compact_line = compact(stripped)

    return compact_line in {
        "INTERIMPRAYER",
        "INTENIMPRAYER",
        "INTERMPRAYER",
        "INTERIMPRAYE",
    }


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
# SYNOPSIS TABLE SCORE
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

    if numeric_rows >= 4:
        score += 30

    if date_rows >= 2:
        score += 50

    if date_rows >= 4:
        score += 40

    if nil_rows >= 1:
        score += 20

    if (
        has_date
        and has_events
        and date_rows >= 3
    ):
        score += 130

    if (
        has_date
        and has_descriptions
        and date_rows >= 3
    ):
        score += 130

    if (
        has_date
        and has_particulars
        and date_rows >= 3
    ):
        score += 130

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

        lines = clean_lines(text)
        index_page = is_index_page(text)

        # Synopsis heading
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

        # Synopsis table
        table_score = synopsis_table_score(text)

        if table_score >= 170:

            candidates["synopsis"].append(
                Candidate(
                    "synopsis",
                    page,
                    table_score,
                    "SYNOPSIS TABLE",
                    "table",
                )
            )

        # Brief Facts + Prayer
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

    page_limit = min(
        12,
        total_pages,
        SEARCH_PAGES,
    )

    for candidate in candidates:

        if candidate.page > page_limit:
            continue

        text = page_text.get(
            candidate.page,
            "",
        )

        if candidate.kind == "table":

            score = synopsis_table_score(text)

            if score < 170:
                continue

            valid.append(
                candidate
            )

            continue

        if candidate.kind == "heading":

            if is_index_page(text):
                continue

            line_normalized = normalize(
                candidate.line
            )

            if "LIST OF DATES" in line_normalized:

                if synopsis_table_score(text) < 170:
                    continue

            elif not is_synopsis_heading(candidate.line):
                continue

            valid.append(
                candidate
            )

    if not valid:
        return None

    tables = [
        c
        for c in valid
        if c.kind == "table"
    ]

    if tables:

        # Prefer the EARLIEST real synopsis table.
        return min(
            tables,
            key=lambda c: (
                c.page,
                -c.score,
            ),
        )

    standalone = [
        c
        for c in valid
        if (
            c.kind == "heading"
            and "LIST OF DATES"
            not in normalize(c.line)
        )
    ]

    if standalone:

        return min(
            standalone,
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
        if (
            synopsis_page
            <= c.page
            <= min(
                SEARCH_PAGES,
                total_pages,
                synopsis_page + 15,
            )
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

        # Select the first real Brief Facts heading,
        # not an OCR body artifact.
        return min(
            same_page,
            key=lambda c: (
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
        if (
            c.page >= after_page
            and c.page <= min(
                SEARCH_PAGES,
                total_pages,
            )
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
# FIND SYNOPSIS END PAGE
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

        lines = clean_lines(
            page_text.get(
                page,
                "",
            )
        )

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
# FIND BRIEF FACTS END PAGE
# ==========================================================

def find_brief_facts_end(
    page_text,
    start_page,
    total_pages,
    prayer_page=None,
):

    max_page = min(
        SEARCH_PAGES,
        total_pages,
        start_page + 10,
    )

    # Brief Facts is allowed to continue to the next page.
    first_boundary_page = min(
        start_page + 1,
        max_page,
    )

    # If Prayer has already been found earlier than our
    # heuristic boundary, never consume the Prayer page.
    if prayer_page is not None:

        max_page = min(
            max_page,
            prayer_page,
        )

    for page in range(
        first_boundary_page,
        max_page + 1,
    ):

        lines = clean_lines(
            page_text.get(
                page,
                "",
            )
        )

        for index, line in enumerate(lines):

            normalized = normalize(line)

            if is_new_document(line):
                return page

            if is_grounds_heading(line):
                return page

            if is_prayer_heading(line):
                return page

            if is_interim_prayer_marker(line):
                return page

            if is_place_heading(line):
                return page

            # Do NOT stop on random OCR signature-looking text.
            if (
                "ADVOCATE" in normalized
                and any(
                    token in normalized
                    for token in [
                        "PETITIONER",
                        "PETITIONERS",
                        "APPELLANT",
                        "APPELLANTS",
                        "RESPONDENT",
                    ]
                )
            ):
                return page

            # Date is only a footer boundary when an advocate
            # marker is nearby.
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

                if "ADVOCATE" in nearby:
                    return page

    return max_page


# ==========================================================
# FIND PRAYER END PAGE
#
# Important:
# If Prayer has no Interim Prayer, retain the entire
# detected Prayer page. This fixes documents where Prayer,
# (a), (b), (c), Place and Advocate all share one page.
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

        lines = clean_lines(
            page_text.get(
                page,
                "",
            )
        )

        for line in lines:

            if is_interim_prayer_marker(line):
                return page

            if page > start_page:

                if is_new_document(line):
                    return page

                if is_grounds_heading(line):
                    return page

    # Prayer can continue onto the immediately
    # following page even when there is no
    # Interim Prayer section.
    return min(
        start_page + 1,
        max_page,
    )


# ==========================================================
# TABLE SYNOPSIS START
# ==========================================================

def find_table_synopsis_start(
    lines,
):

    # Explicit Synopsis heading.
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

        if has_date and has_content:
            return index + 1

    # List of Dates / Synopsis.
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

        if (
            "DATE" in nearby_text
            and (
                "EVENT" in nearby_text
                or "DESCRIPTION" in nearby_text
                or "PARTICULAR" in nearby_text
            )
        ):
            return index + 1

    # Structural fallback.
    page_text = normalize(
        "\n".join(lines)
    )

    has_date_header = (
        "DATE" in page_text
        or "DATES" in page_text
    )

    has_content_header = (
        "EVENT" in page_text
        or "EVENTS" in page_text
        or "DESCRIPTION" in page_text
        or "DESCRIPTIONS" in page_text
        or "PARTICULAR" in page_text
        or "PARTICULARS" in page_text
    )

    if (
        has_date_header
        and has_content_header
    ):

        date_rows = sum(
            1
            for line in lines
            if (
                is_date_line(line)
                or looks_like_year(line)
            )
        )

        if date_rows >= 2:

            for index, line in enumerate(lines):

                normalized = normalize(line)

                if (
                    "DATE" in normalized
                    and (
                        "EVENT" in normalized
                        or "DESCRIPTION" in normalized
                        or "PARTICULAR" in normalized
                    )
                ):
                    return index + 1

            for index, line in enumerate(lines):

                if "DATE" in normalize(line):
                    return index + 1

            return 0

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

        if candidate_kind == "table":
            return 0

        for index, line in enumerate(lines):

            if is_synopsis_heading(line):
                return index + 1

        table_start = find_table_synopsis_start(
            lines
        )

        if table_start is not None:
            return table_start

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
                or "MOST RESPECTFULLY PRAY" in normalized
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

    for index in range(
        start,
        len(lines),
    ):

        line = lines[index].strip()

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

        if is_place_heading(line):
            break

        normalized = normalize(line)

        if (
            "ADVOCATE" in normalized
            and any(
                token in normalized
                for token in [
                    "PETITIONER",
                    "PETITIONERS",
                    "APPELLANT",
                    "APPELLANTS",
                    "RESPONDENT",
                ]
            )
        ):
            break

        collected.append(line)

    return "\n".join(
        collected
    ).strip()


# ==========================================================
# BRIEF FACTS FOOTER
# ==========================================================

def is_brief_facts_footer(
    lines,
    index,
) -> bool:

    line = lines[index]
    normalized = normalize(line)

    # Do not stop on standalone dates.
    # Do not stop on random OCR signature artifacts.

    if is_place_heading(line):
        return True

    if (
        "ADVOCATE" in normalized
        and any(
            token in normalized
            for token in [
                "PETITIONER",
                "PETITIONERS",
                "APPELLANT",
                "APPELLANTS",
                "RESPONDENT",
            ]
        )
    ):
        return True

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

        if "ADVOCATE" in nearby:
            return True

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

    return "\n".join(
        collected
    ).strip()


# ==========================================================
# EXTRACT PRAYER
#
# This intentionally does NOT stop on generic OCR
# signature-like noise. It stops on real structural
# boundaries only.
# ==========================================================

def extract_prayer(
    text: str,
):

    lines = text.splitlines()

    if not lines:
        return ""

    start = find_section_start(
        text,
        "prayer",
    )

    if start is None:
        return ""

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
        # REAL SECTION BOUNDARY
        # --------------------------------------------------

        if is_interim_prayer_marker(line):
            break

        if is_new_document(line):
            break

        if is_grounds_heading(line):
            break

        # --------------------------------------------------
        # REAL FOOTER
        # --------------------------------------------------

        if is_place_heading(line):
            break

        if (
            "ADVOCATE FOR" in normalized
            or (
                "ADVOCATE" in normalized
                and any(
                    token in normalized
                    for token in [
                        "PETITIONER",
                        "PETITIONERS",
                        "APPELLANT",
                        "APPELLANTS",
                        "RESPONDENT",
                    ]
                )
            )
        ):
            break

        # Only treat Date/Dated as footer when an advocate
        # marker is nearby.
        if (
            normalized.startswith("DATE")
            or normalized.startswith("DATED")
        ):

            nearby = " ".join(
                normalize(x)
                for x in lines[
                    index:min(
                        index + 6,
                        len(lines),
                    )
                ]
            )

            if (
                "ADVOCATE" in nearby
                or "ADDRESS FOR SERVICE" in nearby
            ):
                break

        # IMPORTANT:
        # No is_signature(line) check here.
        collected.append(line)

    return "\n".join(
        collected
    ).strip()


# ==========================================================
# TARGETED FALLBACK OCR
# ==========================================================

def targeted_fallback_scan(
    pdf_path: Path,
    total_pages: int,
):

    scan_pages = min(
        SEARCH_PAGES,
        total_pages,
    )

    print(
        "\nTargeted 120-DPI scan "
        "for first 30 pages..."
    )

    images = convert_from_path(
        pdf_path,
        dpi=120,
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
            jobs,
        ):

            fallback_text[page] = text

    return fallback_text


# ==========================================================
# PROCESS ONE DOCUMENT
# ==========================================================

def process_document(
    pdf_path: Path,
):

    total_start = perf_counter()

    # ------------------------------------------------------
    # FAST DISCOVERY
    # ------------------------------------------------------

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

    # ------------------------------------------------------
    # SELECT SYNOPSIS
    # ------------------------------------------------------

    synopsis = select_synopsis(
        candidates["synopsis"],
        fast_text,
        total_pages,
    )

    # ------------------------------------------------------
    # SELECT BRIEF FACTS
    # ------------------------------------------------------

    brief_facts = None

    if synopsis:

        brief_facts = select_brief_facts(
            candidates["brief_facts"],
            synopsis.page,
            total_pages,
        )

    # ------------------------------------------------------
    # SELECT PRAYER
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

        fallback_text = targeted_fallback_scan(
            pdf_path,
            total_pages,
        )

        fallback_candidates = collect_candidates(
            fallback_text
        )

        if synopsis is None:

            fallback_synopsis = select_synopsis(
                fallback_candidates["synopsis"],
                fallback_text,
                total_pages,
            )

            if fallback_synopsis:
                synopsis = fallback_synopsis

        if brief_facts is None and synopsis:

            fallback_brief = select_brief_facts(
                fallback_candidates["brief_facts"],
                synopsis.page,
                total_pages,
            )

            if fallback_brief:
                brief_facts = fallback_brief

        if prayer is None:

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

        # Use fallback OCR for discovery.
        fast_text.update(
            fallback_text
        )

        candidates = fallback_candidates

    # ------------------------------------------------------
    # SECTION RANGES
    # ------------------------------------------------------

    ranges: dict[str, tuple[int, int]] = {}

    # Synopsis
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

    # Brief Facts
    if brief_facts:

        prayer_page = (
            prayer.page
            if prayer
            else None
        )

        brief_facts_end = find_brief_facts_end(
            fast_text,
            brief_facts.page,
            total_pages,
            prayer_page,
        )

        ranges["brief_facts"] = (
            brief_facts.page,
            brief_facts_end,
        )

    # Prayer
    if prayer:

        prayer_end = find_prayer_end(
            fast_text,
            prayer.page,
            total_pages,
        )

        ranges["prayer"] = (
            prayer.page,
            prayer_end,
        )

    # ------------------------------------------------------
    # Always OCR the complete selected pages.
    # This deliberately preserves the old working behavior.
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

        start_page, end_page = ranges[
            section
        ]

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

        candidate = {
            "synopsis": synopsis,
            "brief_facts": brief_facts,
            "prayer": prayer,
        }[section]

        if section == "synopsis":

            sections["synopsis"] = extract_synopsis(
                section_text,
                candidate.kind
                if candidate
                else None,
            )

        elif section == "brief_facts":

            sections["brief_facts"] = (
                extract_brief_facts(
                    section_text
                )
            )

        else:

            sections["prayer"] = (
                extract_prayer(
                    section_text
                )
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
            "synopsis": (
                asdict(synopsis)
                if synopsis
                else None
            ),
            "brief_facts": (
                asdict(brief_facts)
                if brief_facts
                else None
            ),
            "prayer": (
                asdict(prayer)
                if prayer
                else None
            ),
        },
        "ranges": {
            section: {
                "start": pages[0],
                "end": pages[1],
            }
            for section, pages
            in ranges.items()
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
# SAVE DOCUMENT OUTPUT
#
# Exactly:
# DocumentName.txt
# ==========================================================

def save_document_output(
    pdf_path: Path,
    result: dict,
    output_dir: Path,
):

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / f"{pdf_path.stem}.txt"
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        for title, key in [
            (
                "SYNOPSIS",
                "synopsis",
            ),
            (
                "BRIEF FACTS OF THE CASE",
                "brief_facts",
            ),
            (
                "PRAYER",
                "prayer",
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

            file.write(
                "\n\n"
            )

            file.write(
                result["sections"].get(
                    key,
                    "",
                )
                or "NOT FOUND"
            )

            file.write(
                "\n\n"
            )

    return output_path


# ==========================================================
# PROCESS ONE PDF
# ==========================================================

def process_one_pdf(
    pdf_path: Path,
    output_dir: Path,
):

    started = perf_counter()

    print(
        "\n"
        + "=" * 80
    )

    print(
        f"PROCESSING: {pdf_path.name}"
    )

    print(
        "=" * 80
    )

    try:

        result = process_document(
            pdf_path
        )

        output_path = save_document_output(
            pdf_path,
            result,
            output_dir,
        )

        print(
            "\nRESULT"
        )

        print(
            "-" * 80
        )

        for section in [
            "synopsis",
            "brief_facts",
            "prayer",
        ]:

            location = result[
                "locations"
            ].get(
                section
            )

            if location is None:

                print(
                    f"{section:<15}: "
                    f"NOT FOUND"
                )

            else:

                print(
                    f"{section:<15}: "
                    f"page={location['page']}, "
                    f"score={location['score']:.1f}, "
                    f"type={location['kind']}"
                )

        print(
            "\nRanges"
        )

        for section, pages in result[
            "ranges"
        ].items():

            print(
                f"{section:<15}: "
                f"{pages['start']}-"
                f"{pages['end']}"
            )

        print(
            f"\nOutput: {output_path}"
        )

        print(
            f"Time: "
            f"{perf_counter() - started:.2f} sec"
        )

        return True

    except Exception as exc:

        print(
            f"\nFAILED: "
            f"{pdf_path.name}"
        )

        print(
            f"Error: {exc}"
        )

        return False


# ==========================================================
# BATCH PIPELINE
# ==========================================================

def process_folder(
    input_dir: Path,
    output_dir: Path,
):

    if not input_dir.exists():

        raise FileNotFoundError(
            f"Input folder not found:\n"
            f"{input_dir}"
        )

    pdf_files = sorted(
        [
            path
            for path in input_dir.iterdir()
            if (
                path.is_file()
                and path.suffix.lower() == ".pdf"
            )
        ],
        key=lambda p: p.name.lower(),
    )

    if not pdf_files:

        print(
            f"No PDF files found in:\n"
            f"{input_dir}"
        )

        return

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    total_started = perf_counter()

    success = 0
    failed = 0

    print(
        "\n"
        + "=" * 80
    )

    print(
        "BATCH LEGAL DOCUMENT PIPELINE"
    )

    print(
        "=" * 80
    )

    print(
        f"Input Folder : {input_dir}"
    )

    print(
        f"Output Folder: {output_dir.resolve()}"
    )

    print(
        f"PDF Count    : {len(pdf_files)}"
    )

    print(
        "=" * 80
    )

    for index, pdf_path in enumerate(
        pdf_files,
        start=1,
    ):

        print(
            f"\n[{index}/{len(pdf_files)}]"
        )

        ok = process_one_pdf(
            pdf_path,
            output_dir,
        )

        if ok:
            success += 1
        else:
            failed += 1

    total_time = (
        perf_counter()
        - total_started
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "BATCH COMPLETE"
    )

    print(
        "=" * 80
    )

    print(
        f"Total PDFs : {len(pdf_files)}"
    )

    print(
        f"Successful  : {success}"
    )

    print(
        f"Failed      : {failed}"
    )

    print(
        f"Total Time  : {total_time:.2f} sec"
    )

    print(
        f"Output      : {output_dir.resolve()}"
    )

    print(
        "=" * 80
    )


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    process_folder(
        INPUT_DIR,
        OUTPUT_DIR,
    )
