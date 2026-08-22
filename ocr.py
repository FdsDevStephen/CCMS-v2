from __future__ import annotations

import os
import re
import tempfile

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

DEFAULT_OUTPUT_DIR = Path("section_output")

DEFAULT_POPPLER_PATH = (
    r"C:\Users\steph\.cache\codex-runtimes"
    r"\codex-primary-runtime"
    r"\dependencies\native\poppler\Library\bin"
)

SEARCH_PAGES = 30

PAGE_START = 2
PAGE_END = 13

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

def normalize(
    text: str,
) -> str:

    return " ".join(
        text.upper().split()
    )


def compact(
    text: str,
) -> str:

    return re.sub(
        r"[^A-Z0-9]",
        "",
        normalize(text),
    )


# ==========================================================
# BASIC HELPERS
# ==========================================================

def clean_lines(
    text: str,
) -> list[str]:

    return [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]


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
# GROUNDS
# ==========================================================

GROUNDS_PATTERNS = [
    "GROUNDS",
    "SALIENT GROUNDS",
    "GROUNDS OF CHALLENGE",
    "GROUNDS FOR INTERIM RELIEF",
    "GROUNDS FOR INTERIM PRAYER",
]


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

    except Exception as exc:

        raise RuntimeError(
            "Unable to determine PDF page count."
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


def fast_ocr_page(
    args,
):

    page, image = args

    return (
        page,
        ocr_image(
            np.array(image),
            11,
        ),
    )


def full_ocr_page(
    args,
):

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
# FAST OCR
# SEARCH FIRST 30 PAGES
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
        f"\nFast OCR: scanning pages "
        f"1-{scan_pages}..."
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

    elapsed = (
        perf_counter()
        - start
    )

    return (
        page_text,
        total_pages,
        elapsed,
    )


# ==========================================================
# MERGE OCR RANGES
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

        previous_start, previous_end = (
            merged[-1]
        )

        if start <= previous_end + 1:

            merged[-1] = (
                previous_start,
                max(
                    previous_end,
                    end,
                ),
            )

        else:

            merged.append(
                (
                    start,
                    end,
                )
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

    merged_ranges = merge_ranges(
        ranges
    )

    print(
        f"\nFull OCR ranges: "
        f"{merged_ranges}"
    )

    for start_page, end_page in merged_ranges:

        print(
            f"Full OCR: pages "
            f"{start_page}-{end_page}..."
        )

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
# INTERIM PRAYER
# ==========================================================

def is_interim_prayer_marker(
    line: str,
) -> bool:

    normalized = normalize(line)

    if not normalized:
        return False

    if (
        "GROUNDS FOR INTERIM PRAYER"
        in normalized
    ):
        return False

    stripped = re.sub(
        r"^[\s\W]*(?:[0-9]{1,3}|[IVX]{1,5})[\s.\-:)]+",
        "",
        normalized,
    )

    compact_line = compact(
        stripped
    )

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

    if is_interim_prayer_marker(
        line
    ):
        return False

    if normalized == "PRAYER":
        return True

    stripped = re.sub(
        r"^[\s\W]*(?:I|II|III|IV|V|VI|VII|VIII|IX|X)[\s.\-:]+",
        "",
        normalized,
    )

    if stripped == "PRAYER":
        return True

    return (
        normalized.startswith(
            "PRAYER "
        )
        and len(normalized) <= 40
    )


# ==========================================================
# PRAYER PAGE SCORE
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
#
# Prayer logic retained from the working implementation.
# ==========================================================

def collect_candidates(
    page_text: dict[int, str],
):

    candidates = {
        "prayer": [],
    }

    for page, text in page_text.items():

        lines = clean_lines(text)

        # --------------------------------------------------
        # Prayer Heading
        # --------------------------------------------------

        for line in lines:

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

        # --------------------------------------------------
        # Prayer Body
        # --------------------------------------------------

        body_score = prayer_page_score(
            text
        )

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

    # ------------------------------------------------------
    # FALLBACK
    #
    # Sometimes fast OCR only recognizes WHEREFORE.
    # WHEREFORE is a strong Prayer indicator.
    # Only use this fallback if no normal Prayer
    # candidate was found.
    # ------------------------------------------------------

    if not candidates["prayer"]:

        for page, text in page_text.items():

            normalized = normalize(text)

            if (
                "WHEREFORE" in normalized
                and (
                    "PRAYS" in normalized
                    or "PRAY" in normalized
                    or "PLEASED TO" in normalized
                )
            ):

                candidates["prayer"].append(
                    Candidate(
                        "prayer",
                        page,
                        95,
                        "PRAYER BODY FALLBACK",
                        "body",
                    )
                )

    return candidates


# ==========================================================
# SELECT PRAYER
# ==========================================================

def select_prayer(
    candidates,
    total_pages,
):

    valid = [
        candidate
        for candidate in candidates
        if (
            1
            <= candidate.page
            <= min(
                SEARCH_PAGES,
                total_pages,
            )
        )
    ]

    if not valid:
        return None

    # Prefer an actual Prayer heading.
    headings = [
        candidate
        for candidate in valid
        if candidate.kind == "heading"
    ]

    if headings:

        return min(
            headings,
            key=lambda candidate: (
                candidate.page,
                -candidate.score,
            ),
        )

    # If OCR missed the heading,
    # fall back to Prayer body score.
    body = [
        candidate
        for candidate in valid
        if candidate.kind == "body"
    ]

    if body:

        return min(
            body,
            key=lambda candidate: (
                candidate.page,
                -candidate.score,
            ),
        )

    return None


# ==========================================================
# FIND PRAYER END
#
# Uses the same range-detection behavior.
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

            if is_interim_prayer_marker(
                line
            ):
                return page

            if page > start_page:

                if is_new_document(
                    line
                ):
                    return page

                if is_grounds_heading(
                    line
                ):
                    return page

    return min(
        start_page + 1,
        max_page,
    )


# ==========================================================
# FIND PRAYER START
# ==========================================================

def find_section_start(
    text: str,
    section: str,
):

    lines = text.splitlines()

    if section != "prayer":
        return None

    # ----------------------------------------------
    # First try the actual Prayer heading.
    # ----------------------------------------------

    for index, line in enumerate(
        lines
    ):

        if is_prayer_heading(
            line
        ):

            return index + 1

    # ----------------------------------------------
    # If heading was missed by OCR,
    # locate the Prayer body.
    # ----------------------------------------------

    for index, line in enumerate(
        lines
    ):

        normalized = normalize(
            line
        )

        if (
            "WHEREFORE" in normalized
            or "MOST RESPECTFULLY PRAY"
            in normalized
        ):

            return index

    return None


# ==========================================================
# EXTRACT FULL PRAYER
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

        normalized = normalize(
            line
        )

        # ----------------------------------------------
        # Prayer boundaries
        # ----------------------------------------------

        if is_interim_prayer_marker(
            line
        ):
            break

        if is_new_document(
            line
        ):
            break

        if is_grounds_heading(
            line
        ):
            break

        # ----------------------------------------------
        # Footer boundaries
        # ----------------------------------------------

        if is_place_heading(
            line
        ):
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

        # ----------------------------------------------
        # Date / Dated footer
        # ----------------------------------------------

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
                or "ADDRESS FOR SERVICE"
                in nearby
            ):
                break

        collected.append(
            line
        )

    return "\n".join(
        collected
    ).strip()


# ==========================================================
# PAGE TEXT HELPER
# ==========================================================

def combine_pages(
    page_text: dict[int, str],
    start_page: int,
    end_page: int,
) -> str:

    return "\n".join(
        page_text[page]
        for page in sorted(
            page_text
        )
        if (
            start_page
            <= page
            <= end_page
        )
    ).strip()


# ==========================================================
# PROCESS ONE DOCUMENT
# ==========================================================

def process_document(
    pdf_path: Path,
):

    total_start = perf_counter()

    # ======================================================
    # STEP 1
    # FAST OCR FIRST 30 PAGES
    # ======================================================

    (
        fast_text,
        total_pages,
        fast_time,
    ) = fast_scan(
        pdf_path
    )

    # ======================================================
    # STEP 2
    # FIND PRAYER
    # ======================================================

    candidates = collect_candidates(
        fast_text
    )

    prayer = select_prayer(
        candidates["prayer"],
        total_pages,
    )

    # ======================================================
    # STEP 3
    # FIND FULL PRAYER RANGE
    # ======================================================

    prayer_range = None

    if prayer:

        prayer_end = find_prayer_end(
            fast_text,
            prayer.page,
            total_pages,
        )

        prayer_range = (
            prayer.page,
            prayer_end,
        )

        print(
            f"Prayer found: "
            f"Page {prayer.page}"
        )

        print(
            f"Prayer range: "
            f"Pages {prayer.page}-{prayer_end}"
        )

    else:

        print(
            "Prayer NOT FOUND in first "
            f"{min(SEARCH_PAGES, total_pages)} pages."
        )

    # ======================================================
    # STEP 4
    # FIXED PAGES 2-13
    # ======================================================

    page_2_to_13_end = min(
        PAGE_END,
        total_pages,
    )

    ranges = []

    if total_pages >= PAGE_START:

        ranges.append(
            (
                PAGE_START,
                page_2_to_13_end,
            )
        )

    # ======================================================
    # STEP 5
    # ADD FULL PRAYER RANGE
    #
    # If Prayer is inside Pages 2-13,
    # merge_ranges() prevents duplicate OCR.
    # ======================================================

    if prayer_range:

        ranges.append(
            prayer_range
        )

    # ======================================================
    # STEP 6
    # FULL OCR
    # ======================================================

    full_start = perf_counter()

    full_text_by_page = (
        full_ocr_ranges(
            pdf_path,
            ranges,
        )
    )

    full_time = (
        perf_counter()
        - full_start
    )

    # ======================================================
    # STEP 7
    # EXTRACT PAGES 2-13
    # ======================================================

    pages_2_to_13 = ""

    if total_pages >= PAGE_START:

        pages_2_to_13 = combine_pages(
            full_text_by_page,
            PAGE_START,
            page_2_to_13_end,
        )

    # ======================================================
    # STEP 8
    # EXTRACT FULL PRAYER
    # ======================================================

    prayer_text = ""

    if prayer_range:

        prayer_start_page, prayer_end_page = (
            prayer_range
        )

        prayer_full_text = combine_pages(
            full_text_by_page,
            prayer_start_page,
            prayer_end_page,
        )

        prayer_text = extract_prayer(
            prayer_full_text
        )

    # ======================================================
    # TOTAL TIME
    # ======================================================

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
            "prayer": (
                asdict(prayer)
                if prayer
                else None
            ),
        },

        "ranges": {
            "pages_2_to_13": (
                {
                    "start": PAGE_START,
                    "end": page_2_to_13_end,
                }
                if total_pages >= PAGE_START
                else None
            ),

            "prayer": (
                {
                    "start": prayer_range[0],
                    "end": prayer_range[1],
                }
                if prayer_range
                else None
            ),
        },

        "sections": {
            "pages_2_to_13": pages_2_to_13,
            "prayer": prayer_text,
        },

        "timing": {
            "fast_scan": round(
                fast_time,
                3,
            ),

            "full_ocr": round(
                full_time,
                3,
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

    sections = result[
        "sections"
    ]

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "=" * 80
        )

        file.write(
            "\nPAGES 2-13\n"
        )

        file.write(
            "=" * 80
        )

        file.write(
            "\n\n"
        )

        file.write(
            sections.get(
                "pages_2_to_13",
                "",
            )
            or "NOT FOUND"
        )

        file.write(
            "\n\n"
        )

        file.write(
            "=" * 80
        )

        file.write(
            "\nFULL PRAYER\n"
        )

        file.write(
            "=" * 80
        )

        file.write(
            "\n\n"
        )

        file.write(
            sections.get(
                "prayer",
                "",
            )
            or "NOT FOUND"
        )

        file.write(
            "\n\n"
        )

    return output_path


# ==========================================================
# OCR PROCESSOR
# ==========================================================

class OCRProcessor:
    """
    OCR Processor.

    Pipeline:

        PDF
          ↓
        Fast OCR - first 30 pages
          ↓
        Find Prayer
          ↓
        Find complete Prayer range
          ↓
        Full OCR - Pages 2-13 + Prayer range
          ↓
        Extract Pages 2-13
          ↓
        Extract complete Prayer
    """

    def __init__(
        self,
        poppler_path: str,
        output_folder: Path,
        tesseract_path: str | None = None,
        search_pages: int = 30,
        fast_dpi: int = 80,
        full_dpi: int = 300,
        max_workers: int | None = None,
    ) -> None:

        self.poppler_path = (
            poppler_path
        )

        self.output_folder = Path(
            output_folder
        )

        self.search_pages = (
            search_pages
        )

        self.fast_dpi = (
            fast_dpi
        )

        self.full_dpi = (
            full_dpi
        )

        self.max_workers = (
            max_workers
            if max_workers is not None
            else min(
                8,
                os.cpu_count() or 4,
            )
        )

        if tesseract_path:

            pytesseract.pytesseract.tesseract_cmd = (
                tesseract_path
            )

        self.output_folder.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ======================================================
    # CONFIGURE
    # ======================================================

    def _configure(
        self,
    ) -> None:

        global POPPLER_PATH
        global SEARCH_PAGES
        global FAST_DPI
        global FULL_DPI
        global MAX_WORKERS

        POPPLER_PATH = (
            self.poppler_path
        )

        SEARCH_PAGES = (
            self.search_pages
        )

        FAST_DPI = (
            self.fast_dpi
        )

        FULL_DPI = (
            self.full_dpi
        )

        MAX_WORKERS = (
            self.max_workers
        )

    # ======================================================
    # FORMAT OUTPUT
    # ======================================================

    def _format_output(
        self,
        result: dict,
    ) -> str:

        sections = result.get(
            "sections",
            {},
        )

        output = []

        output.append(
            "=" * 80
        )

        output.append(
            "PAGES 2-13"
        )

        output.append(
            "=" * 80
        )

        output.append("")

        output.append(
            sections.get(
                "pages_2_to_13",
                "",
            )
            or "NOT FOUND"
        )

        output.append("")

        output.append(
            "=" * 80
        )

        output.append(
            "FULL PRAYER"
        )

        output.append(
            "=" * 80
        )

        output.append("")

        output.append(
            sections.get(
                "prayer",
                "",
            )
            or "NOT FOUND"
        )

        output.append("")

        return "\n".join(
            output
        ).strip()

    # ======================================================
    # SAVE TEXT
    # ======================================================

    def save_text(
        self,
        pdf_path: str | Path,
        text: str,
    ) -> Path:

        pdf_path = Path(
            pdf_path
        )

        output_path = (
            self.output_folder
            / f"{pdf_path.stem}.txt"
        )

        output_path.write_text(
            text,
            encoding="utf-8",
        )

        return output_path

    # ======================================================
    # PROCESS PDF
    # ======================================================

    def process(
        self,
        pdf_path: str | Path,
    ) -> tuple[str, Path]:

        self._configure()

        pdf_path = Path(
            pdf_path
        )

        if not pdf_path.exists():

            raise FileNotFoundError(
                f"PDF file not found: {pdf_path}"
            )

        result = process_document(
            pdf_path
        )

        text = self._format_output(
            result
        )

        txt_path = self.save_text(
            pdf_path,
            text,
        )

        return (
            text,
            txt_path,
        )

    # ======================================================
    # PROCESS BYTES
    # ======================================================

    def process_bytes(
        self,
        pdf_bytes: bytes,
        filename: str = "document.pdf",
    ) -> str:

        self._configure()

        if not pdf_bytes:

            raise ValueError(
                "PDF bytes are empty."
            )

        suffix = (
            Path(filename).suffix
            or ".pdf"
        )

        with tempfile.NamedTemporaryFile(
            suffix=suffix,
            delete=False,
        ) as temp_file:

            temp_file.write(
                pdf_bytes
            )

            temp_path = Path(
                temp_file.name
            )

        try:

            result = process_document(
                temp_path
            )

            text = self._format_output(
                result
            )

            output_path = (
                self.output_folder
                / f"{Path(filename).stem}.txt"
            )

            output_path.write_text(
                text,
                encoding="utf-8",
            )

            return text

        finally:

            try:

                temp_path.unlink()

            except FileNotFoundError:
                pass