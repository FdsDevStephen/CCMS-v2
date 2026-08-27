"""
OCR Section Extractor.

Pipeline:

    PDF
      ↓
    Text layer probe (free) — skip OCR where the PDF already has text
      ↓
    Fast pass over the first N pages
      ↓
    Locate Prayer
      ↓
    Resolve the complete Prayer page range
      ↓
    Full pass over Body range + Prayer range
      ↓
    Extract Body + complete Prayer


Notes on this version:

    - All tunables live on OCRConfig and are threaded through explicitly.
      There are no module-level mutable globals, so the free functions are
      safe to call directly and safe under a threaded server.

    - Rendering uses PyMuPDF (fitz) directly, at grayscale, one page at a
      time. This removes the poppler dependency, cuts page memory ~6x, and
      lets render + OCR overlap in the worker pool.

    - Tesseract is always told the true DPI. Without that it estimates
      resolution from glyph size, which is unreliable below ~150 DPI.
"""

from __future__ import annotations

import os
import re
import tempfile

from dataclasses import dataclass, field, asdict, replace
from pathlib import Path
from time import perf_counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import fitz
import numpy as np
import pytesseract

from rapidfuzz.fuzz import ratio

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None


# ==========================================================
# CONFIG
# ==========================================================

@dataclass(frozen=True)
class OCRConfig:
    """
    All OCR tunables. Passed explicitly rather than held in globals.
    """

    # ----- Page selection -----

    search_pages: int = 30
    page_start: int = 2
    page_end: int = 13

    # ----- Resolution -----
    #
    # fast_dpi was 80. Tesseract is unreliable below ~150 DPI, which is
    # why heading detection previously needed a WHEREFORE fallback.
    #
    # full_dpi was 300. 220 is ~1.9x faster (OCR cost tracks pixel count)
    # and holds up on clean legal body text. Raise it back toward 300 for
    # badly degraded scans.

    fast_dpi: int = 150
    full_dpi: int = 220

    # ----- Text layer -----
    #
    # Digital-born e-filed PDFs already carry text. Reading it is ~1000x
    # cheaper than OCR and cleaner, so downstream normalization has less
    # to repair. The probe is cheap enough to always run.

    prefer_text_layer: bool = True
    min_text_layer_chars: int = 200

    # ----- Preprocessing -----
    #
    # Median blur helps noisy scans and slightly erodes thin strokes on
    # clean renders. Only ever applied to pages actually sent to OCR.

    denoise: bool = True

    # ----- Prayer range -----

    prayer_max_span: int = 8
    prayer_edge_slack: int = 4
    prayer_extend_pages: int = 8

    # ----- Output -----
    #
    # NOTE: RAG/chunker.py::_extract_sections parses this exact literal.
    # If you change it, update that regex in the same commit or chunking
    # silently returns [].

    body_label: str = "PAGES 2-13"
    prayer_label: str = "FULL PRAYER"

    # ----- Concurrency -----

    max_workers: int = field(
        default_factory=lambda: min(8, os.cpu_count() or 4)
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
    return " ".join(text.upper().split())

def normalize(text: str) -> str:
    return " ".join(text.upper().split())


def clean_ocr_text(text: str) -> str:
    """
    Clean the complete OCR output after ALL pages have been extracted.

    Flow:
        OCR all pages
            ↓
        combine all extracted text
            ↓
        clean_ocr_text()
            ↓
        save .txt

    This does NOT paraphrase or summarize the legal document.
    """

    if not text:
        return ""

    # ==========================================================
    # 1. NORMALIZE
    # ==========================================================

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    lines = []

    for raw_line in text.split("\n"):

        line = raw_line.strip()

        if not line:
            if lines and lines[-1] != "":
                lines.append("")
            continue

        # ======================================================
        # 2. REMOVE TABLE / SCAN ARTIFACTS
        # ======================================================

        # Remove vertical table borders ANYWHERE in the line.
        line = re.sub(r"[|¦]", " ", line)

        # Remove long horizontal OCR lines.
        line = re.sub(r"[-_=]{4,}", " ", line)
        line = re.sub(r"_{3,}", " ", line)

        # Remove obvious decorative OCR symbols.
        line = re.sub(
            r"(?<!\w)[*@#~`^°¢£€©®•·]+(?!\w)",
            " ",
            line,
        )

        # Remove isolated backslashes.
        line = re.sub(r"(?<!\w)\\(?!\w)", " ", line)

        # ======================================================
        # 3. HIGH-CONFIDENCE OCR CORRECTIONS
        # ======================================================

        corrections = {
            # Court
            r"\bCORT\b": "COURT",
            r"\bCOUT\b": "COURT",
            r"\bCOORT\b": "COURT",

            # Karnataka
            r"\bKamataka\b": "Karnataka",
            r"\bKarnatake\b": "Karnataka",

            # Common OCR errors
            r"\bfram\b": "from",
            r"\bfrorm\b": "from",
            r"\bforma!\b": "formal",
            r"\brnarked\b": "marked",
            r"\bmace\b": "made",
            r"\bthrougn\b": "through",

            # Legal words
            r"\bpetitloner\b": "petitioner",
            r"\bPetitloner\b": "Petitioner",
            r"\bpetitlon\b": "petition",
            r"\brespondant\b": "respondent",

            r"\bGommissioner\b": "Commissioner",
            r"\bCommisioner\b": "Commissioner",
            r"\bCommissloner\b": "Commissioner",

            r"\bAsslstant\b": "Assistant",
            r"\bReyenue\b": "Revenue",

            r"\bGovemment\b": "Government",
            r"\bgoverment\b": "government",

            r"\bappllcation\b": "application",
            r"\bapproprlate\b": "appropriate",
            r"\bopportunlty\b": "opportunity",

            r"\bunauthorlzed\b": "unauthorized",
            r"\bregularizatlon\b": "regularization",
            r"\brepresentatlon\b": "representation",
            r"\bcancellatlon\b": "cancellation",

            r"\bproceedlng\b": "proceeding",
            r"\bproceedlngs\b": "proceedings",

            r"\bnotlce\b": "notice",

            r"\bOrignial\b": "Original",
            r"\bAmnexure\b": "Annexure",
            r"\bANNEXCURE\b": "ANNEXURE",

            r"\bAlfidavil\b": "Affidavit",
            r"\battidavit\b": "affidavit",
            r"\bVeritying\b": "Verifying",
            r"\bMemorand\b": "Memorandum",

            # Other recurring OCR errors
            r"\bLatter\b": "Letter",
            r"\bDio\b": "D/o",
            r"\bWio\b": "W/o",
        }

        for pattern, replacement in corrections.items():
            line = re.sub(
                pattern,
                replacement,
                line,
            )

        # ======================================================
        # 4. CONTEXT-SPECIFIC LEGAL CORRECTIONS
        # ======================================================

        # HIGH CORT OF KARNATAKA
        line = re.sub(
            r"\bHIGH\s+CORT\b",
            "HIGH COURT",
            line,
            flags=re.IGNORECASE,
        )

        # Farr/Far No.53 -> Form No.53
        line = re.sub(
            r"\bFarr\s+No\.",
            "Form No.",
            line,
            flags=re.IGNORECASE,
        )

        line = re.sub(
            r"\bFar\s+No\.",
            "Form No.",
            line,
            flags=re.IGNORECASE,
        )

        # Rule 108 D (3)
        line = re.sub(
            r"\bRule\s+108\s+D\s*\(\s*3\s*\)",
            "Rule 108-D(3)",
            line,
            flags=re.IGNORECASE,
        )

        # Rule 108-D-3
        line = re.sub(
            r"\bRule\s+108-D-3\b",
            "Rule 108-D(3)",
            line,
            flags=re.IGNORECASE,
        )

        # ======================================================
        # 5. WHITESPACE
        # ======================================================

        line = re.sub(r"[ \t]+", " ", line)

        # Space before punctuation
        line = re.sub(
            r"\s+([,.;:!?])",
            r"\1",
            line,
        )

        # Missing space after punctuation
        line = re.sub(
            r"([,.;:!?])(?=[A-Za-z])",
            r"\1 ",
            line,
        )

        line = line.strip()

        if not line:
            continue

        # ======================================================
        # 6. REMOVE OBVIOUS GARBAGE LINES
        # ======================================================

        # Standalone page number
        if re.fullmatch(r"\d{1,3}", line):
            continue

        # Standalone punctuation/symbols
        if re.fullmatch(r"[\W_]+", line):
            continue

        # Tiny OCR garbage
        if re.fullmatch(r"[A-Za-z]{1,2}", line):
            if line.upper() not in {
                "IN",
                "OF",
                "TO",
                "BY",
                "OR",
                "NO",
                "RS",
                "MR",
                "MS",
                "DR",
                "VS",
                "WP",
                "IA",
                "RA",
                "AND",
                "AS",
                "IS",
                "ON",
                "AT",
                "A",
                "I",
            }:
                continue

        lines.append(line)

    # ==========================================================
    # 7. REBUILD BROKEN OCR LINES
    # ==========================================================

    final_lines = []

    for line in lines:

        if not line:
            if final_lines and final_lines[-1] != "":
                final_lines.append("")
            continue

        # Never merge headings.
        is_heading = (
            line.isupper()
            and len(line) <= 100
        )

        # Never merge numbered legal paragraphs.
        is_numbered = bool(
            re.match(
                r"^(?:\d+[.)]|\([A-Za-z0-9]+\)|[A-Za-z][.)])\s+",
                line,
            )
        )

        if (
            final_lines
            and final_lines[-1]
            and not is_heading
            and not is_numbered
        ):

            previous = final_lines[-1]

            previous_is_heading = (
                previous.isupper()
                and len(previous) <= 100
            )

            if not previous_is_heading:

                # Join obvious wrapped sentences.
                if (
                    not previous.endswith(
                        (".", ":", ";", "?", "!")
                    )
                    and not line.startswith(
                        (
                            "ANNEXURE",
                            "INDEX",
                            "SYNOPSIS",
                            "WHEREFORE",
                            "BENGALURU",
                            "DATE:",
                            "ADVOCATE",
                            "BETWEEN:",
                            "AND:",
                        )
                    )
                ):
                    final_lines[-1] = (
                        previous.rstrip()
                        + " "
                        + line.lstrip()
                    )
                    continue

        final_lines.append(line)

    # ==========================================================
    # 8. FINAL BLANK-LINE CLEANUP
    # ==========================================================

    output = []

    for line in final_lines:

        if not line.strip():

            if output and output[-1] != "":
                output.append("")

        else:
            output.append(line.rstrip())

    while output and not output[0].strip():
        output.pop(0)

    while output and not output[-1].strip():
        output.pop()

    return "\n".join(output)

def compact(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", normalize(text))


# ==========================================================
# BASIC HELPERS
# ==========================================================

def clean_lines(text: str) -> list[str]:
    return [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]


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


def is_place_heading(line: str) -> bool:

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


def is_grounds_heading(line: str) -> bool:

    normalized = normalize(line)

    for target in GROUNDS_PATTERNS:

        if normalized == target:
            return True

        if len(normalized) <= 90 and ratio(normalized, target) >= 85:
            return True

    return False


# ==========================================================
# TABLE OF CONTENTS
# ==========================================================

# Dot leaders, underscore leaders, or a trailing page number. A synopsis
# index line such as "PRAYER ............ 12" previously matched as a real
# Prayer heading, and select_prayer takes the earliest match, so a single
# TOC line hijacked the whole extraction.

TOC_LINE_PATTERN = re.compile(r"\.{3,}|_{3,}|\s\d{1,3}$")


def looks_like_toc_entry(normalized: str) -> bool:
    return bool(TOC_LINE_PATTERN.search(normalized))


# ==========================================================
# PDF PAGE COUNT
# ==========================================================

def get_total_pages(pdf_path: Path) -> int:

    try:
        document = fitz.open(str(pdf_path))

        try:
            return document.page_count

        finally:
            document.close()

    except Exception:
        pass

    if PdfReader is not None:

        try:
            return len(PdfReader(str(pdf_path)).pages)

        except Exception:
            pass

    raise RuntimeError(
        f"Unable to determine PDF page count: {pdf_path}"
    )


# ==========================================================
# TEXT LAYER
# ==========================================================

def text_layer_pages(
    pdf_path: Path,
    first_page: int,
    last_page: int,
    min_chars: int,
) -> dict[int, str]:
    """
    Return {page: text} for pages that already carry a usable text layer.

    Pages below min_chars are omitted so the caller falls back to OCR for
    them. Scanned pages typically yield 0 characters; a page with only a
    letterhead or stamp yields a handful.
    """

    found: dict[int, str] = {}

    try:
        document = fitz.open(str(pdf_path))

    except Exception:
        return found

    try:
        last_page = min(last_page, document.page_count)

        for page_number in range(first_page, last_page + 1):

            try:
                # sort=True orders blocks by visual position rather than
                # content-stream order. Without it, list markers such as
                # "(c)" can surface after the text they label, and stray
                # page numbers get inlined mid-sentence.
                text = document[page_number - 1].get_text("text", sort=True)

            except Exception:
                continue

            if len(text.strip()) >= min_chars:
                found[page_number] = text

    finally:
        document.close()

    return found


# ==========================================================
# RENDERING
# ==========================================================

def render_page_gray(
    pdf_path: Path,
    page_number: int,
    dpi: int,
) -> np.ndarray:
    """
    Render one page to a grayscale numpy array.

    The document is opened per call because fitz.Document is not thread
    safe. Open cost is a few milliseconds against tens of milliseconds of
    render and hundreds of milliseconds of OCR, so this is a good trade
    for being able to render inside the worker pool.
    """

    document = fitz.open(str(pdf_path))

    try:
        pixmap = document[page_number - 1].get_pixmap(
            dpi=dpi,
            colorspace=fitz.csGRAY,
        )

        return np.frombuffer(
            pixmap.samples,
            dtype=np.uint8,
        ).reshape(pixmap.height, pixmap.width)

    finally:
        document.close()


# ==========================================================
# OCR
# ==========================================================

def ocr_gray(
    gray: np.ndarray,
    psm: int,
    dpi: int,
) -> str:
    """
    Run Tesseract on a grayscale array.

    user_defined_dpi matters: a numpy array carries no resolution
    metadata, so without it Tesseract estimates DPI from glyph size and
    segments characters poorly at lower resolutions.
    """

    return pytesseract.image_to_string(
        gray,
        lang="eng",
        config=f"--oem 3 --psm {psm} -c user_defined_dpi={dpi}",
    )


def ocr_page_fast(
    pdf_path: Path,
    page_number: int,
    cfg: OCRConfig,
) -> tuple[int, str]:
    """
    Fast pass. PSM 11 (sparse text) because this pass only needs to spot
    headings and marker phrases, not reproduce layout.
    """

    gray = render_page_gray(pdf_path, page_number, cfg.fast_dpi)

    return page_number, ocr_gray(gray, 11, cfg.fast_dpi)


def ocr_page_full(
    pdf_path: Path,
    page_number: int,
    cfg: OCRConfig,
) -> tuple[int, str]:
    """
    Full pass. PSM 6 (uniform block of text) for legal body copy.
    """

    gray = render_page_gray(pdf_path, page_number, cfg.full_dpi)

    if cfg.denoise:
        gray = cv2.medianBlur(gray, 3)

    return page_number, ocr_gray(gray, 6, cfg.full_dpi)


# ==========================================================
# PAGE PASSES
# ==========================================================

def _run_pages(
    pdf_path: Path,
    pages: list[int],
    cfg: OCRConfig,
    worker,
) -> dict[int, str]:
    """
    Render + OCR the given pages concurrently.

    pytesseract shells out to the tesseract binary, so threads are the
    right primitive here — the GIL is released for the duration.
    """

    page_text: dict[int, str] = {}

    if not pages:
        return page_text

    with ThreadPoolExecutor(max_workers=cfg.max_workers) as executor:

        futures = {
            executor.submit(worker, pdf_path, page, cfg): page
            for page in pages
        }

        for future in as_completed(futures):

            page = futures[future]

            try:
                _, text = future.result()
                page_text[page] = text

            except Exception as exc:
                print(f"  OCR failed on page {page}: {exc}")
                page_text[page] = ""

    return page_text


def fast_scan_range(
    pdf_path: Path,
    first_page: int,
    last_page: int,
    cfg: OCRConfig,
    text_layer: dict[int, str] | None = None,
) -> dict[int, str]:
    """
    Fast pass over a page range, reusing the text layer where present.
    """

    if last_page < first_page:
        return {}

    text_layer = text_layer or {}

    wanted = range(first_page, last_page + 1)

    reused = {
        page: text_layer[page]
        for page in wanted
        if page in text_layer
    }

    todo = [page for page in wanted if page not in reused]

    if reused:
        print(
            f"  Text layer: {len(reused)} page(s) reused, "
            f"{len(todo)} to OCR"
        )

    page_text = _run_pages(pdf_path, todo, cfg, ocr_page_fast)

    page_text.update(reused)

    return page_text


def fast_scan(
    pdf_path: Path,
    total_pages: int,
    cfg: OCRConfig,
    text_layer: dict[int, str] | None = None,
) -> tuple[dict[int, str], float]:

    start = perf_counter()

    scan_pages = min(cfg.search_pages, total_pages)

    print(f"\nFast pass: pages 1-{scan_pages}...")

    page_text = fast_scan_range(
        pdf_path,
        1,
        scan_pages,
        cfg,
        text_layer,
    )

    return page_text, perf_counter() - start


# ==========================================================
# MERGE OCR RANGES
# ==========================================================

def merge_ranges(
    ranges: list[tuple[int, int]],
) -> list[tuple[int, int]]:

    if not ranges:
        return []

    sorted_ranges = sorted(
        (min(start, end), max(start, end))
        for start, end in ranges
    )

    merged = [sorted_ranges[0]]

    for start, end in sorted_ranges[1:]:

        previous_start, previous_end = merged[-1]

        if start <= previous_end + 1:

            merged[-1] = (
                previous_start,
                max(previous_end, end),
            )

        else:
            merged.append((start, end))

    return merged


# ==========================================================
# FULL OCR
# ==========================================================

def full_ocr_ranges(
    pdf_path: Path,
    ranges: list[tuple[int, int]],
    cfg: OCRConfig,
    text_layer: dict[int, str] | None = None,
) -> dict[int, str]:

    text_layer = text_layer or {}

    merged_ranges = merge_ranges(ranges)

    print(f"\nFull pass ranges: {merged_ranges}")

    wanted: list[int] = []

    for start_page, end_page in merged_ranges:
        wanted.extend(range(start_page, end_page + 1))

    reused = {
        page: text_layer[page]
        for page in wanted
        if page in text_layer
    }

    todo = [page for page in wanted if page not in reused]

    if reused:
        print(
            f"  Text layer: {len(reused)} page(s) reused, "
            f"{len(todo)} to OCR"
        )

    if todo:
        print(f"  Full OCR: {len(todo)} page(s) at {cfg.full_dpi} DPI...")

    page_text = _run_pages(pdf_path, todo, cfg, ocr_page_full)

    page_text.update(reused)

    return page_text


# ==========================================================
# INTERIM PRAYER
# ==========================================================

def is_interim_prayer_marker(line: str) -> bool:

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

    return compact(stripped) in {
        "INTERIMPRAYER",
        "INTENIMPRAYER",
        "INTERMPRAYER",
        "INTERIMPRAYE",
    }


# ==========================================================
# PRAYER HEADING
# ==========================================================

def is_prayer_heading(line: str) -> bool:

    normalized = normalize(line)

    if is_interim_prayer_marker(line):
        return False

    # Reject synopsis / index entries.
    if looks_like_toc_entry(normalized):
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
        normalized.startswith("PRAYER ")
        and len(normalized) <= 40
    )


# ==========================================================
# PRAYER PAGE SCORE
# ==========================================================

def prayer_page_score(text: str) -> float:

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
        ("IN THE INTEREST OF JUSTICE AND EQUITY", 25),
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

    score += min(roman_items * 15, 75)

    # An explicit interim heading is a real veto.
    #
    # "PENDING DISPOSAL" is not. It appears verbatim inside main prayers
    # ("pending disposal of this writ petition"), and a -200 penalty
    # against a ~155 signal ceiling guaranteed the page was rejected.
    # Demote it to a nudge, and skip it entirely when the page carries
    # the strongest main-prayer marker.

    if (
        "INTERIM PRAYER" in normalized
        or "INTENIM PRAYER" in normalized
    ):
        score -= 200

    elif (
        "PENDING DISPOSAL" in normalized
        and "WHEREFORE" not in normalized
    ):
        score -= 60

    return score


# ==========================================================
# COLLECT CANDIDATES
# ==========================================================

def collect_candidates(page_text: dict[int, str]):

    candidates = {"prayer": []}

    for page, text in page_text.items():

        lines = clean_lines(text)

        # --------------------------------------------------
        # Prayer Heading
        # --------------------------------------------------

        for line in lines:

            if is_prayer_heading(line):

                candidates["prayer"].append(
                    Candidate("prayer", page, 180, line, "heading")
                )

        # --------------------------------------------------
        # Prayer Body
        # --------------------------------------------------

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

    # ------------------------------------------------------
    # FALLBACK
    #
    # Retained as a safety net, though it should fire far less often now
    # that the fast pass runs at a resolution Tesseract can work with and
    # is told its true DPI.
    # ------------------------------------------------------

    if not candidates["prayer"]:

        for page, text in page_text.items():

            normalized = normalize(text)

            if "WHEREFORE" in normalized and (
                "PRAYS" in normalized
                or "PRAY" in normalized
                or "PLEASED TO" in normalized
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
    total_pages: int,
    cfg: OCRConfig,
    scanned_through: int | None = None,
):

    limit = (
        scanned_through
        if scanned_through is not None
        else min(cfg.search_pages, total_pages)
    )

    valid = [
        candidate
        for candidate in candidates
        if 1 <= candidate.page <= limit
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
            key=lambda candidate: (candidate.page, -candidate.score),
        )

    # If the heading was missed, fall back to Prayer body score.
    body = [
        candidate
        for candidate in valid
        if candidate.kind == "body"
    ]

    if body:

        return min(
            body,
            key=lambda candidate: (candidate.page, -candidate.score),
        )

    return None


# ==========================================================
# FIND PRAYER END
# ==========================================================

def find_prayer_end(
    page_text: dict[int, str],
    start_page: int,
    total_pages: int,
    cfg: OCRConfig,
    scanned_through: int | None = None,
):
    """
    Walk forward from the Prayer start until a terminator appears.

    The window is bounded by what was actually scanned — you cannot
    inspect pages you have no text for. process_document extends the fast
    pass when the Prayer lands near the edge of the scanned window, which
    is what previously caused prayers to be truncated mid-relief.
    """

    scanned_through = (
        scanned_through
        if scanned_through is not None
        else min(cfg.search_pages, total_pages)
    )

    max_page = min(
        scanned_through,
        total_pages,
        start_page + cfg.prayer_max_span,
    )

    for page in range(start_page, max_page + 1):

        lines = clean_lines(page_text.get(page, ""))

        for line in lines:

            if is_interim_prayer_marker(line):
                return page

            if page > start_page:

                if is_new_document(line):
                    return page

                if is_grounds_heading(line):
                    return page

    return min(start_page + 1, max_page)


# ==========================================================
# FIND PRAYER START
# ==========================================================

def find_section_start(text: str, section: str):

    lines = text.splitlines()

    if section != "prayer":
        return None

    # ----------------------------------------------
    # First try the actual Prayer heading.
    # ----------------------------------------------

    for index, line in enumerate(lines):

        if is_prayer_heading(line):
            return index + 1

    # ----------------------------------------------
    # If the heading was missed, locate the body.
    # ----------------------------------------------

    for index, line in enumerate(lines):

        normalized = normalize(line)

        if (
            "WHEREFORE" in normalized
            or "MOST RESPECTFULLY PRAY" in normalized
        ):
            return index

    return None


# ==========================================================
# EXTRACT FULL PRAYER
# ==========================================================

def extract_prayer(text: str) -> str:

    lines = text.splitlines()

    if not lines:
        return ""

    start = find_section_start(text, "prayer")

    if start is None:
        return ""

    collected = []

    for index in range(start, len(lines)):

        line = lines[index].strip()

        if not line:
            continue

        normalized = normalize(line)

        # ----------------------------------------------
        # Prayer boundaries
        # ----------------------------------------------

        if is_interim_prayer_marker(line):
            break

        if is_new_document(line):
            break

        if is_grounds_heading(line):
            break

        # ----------------------------------------------
        # Footer boundaries
        # ----------------------------------------------

        if is_place_heading(line):
            break

        if "ADVOCATE FOR" in normalized or (
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

        # ----------------------------------------------
        # Date / Dated footer
        # ----------------------------------------------

        if (
            normalized.startswith("DATE")
            or normalized.startswith("DATED")
        ):

            nearby = " ".join(
                normalize(x)
                for x in lines[index:min(index + 6, len(lines))]
            )

            if (
                "ADVOCATE" in nearby
                or "ADDRESS FOR SERVICE" in nearby
            ):
                break

        collected.append(line)

    return "\n".join(collected).strip()


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
        for page in sorted(page_text)
        if start_page <= page <= end_page
    ).strip()


# ==========================================================
# PROCESS ONE DOCUMENT
# ==========================================================

def process_document(
    pdf_path: Path,
    cfg: OCRConfig | None = None,
) -> dict:

    cfg = cfg or OCRConfig()

    total_start = perf_counter()

    pdf_path = Path(pdf_path)

    total_pages = get_total_pages(pdf_path)

    # ======================================================
    # STEP 0
    # TEXT LAYER PROBE
    #
    # Computed once for the widest region either pass could need, then
    # shared by both. A page with a text layer needs no OCR at all, so
    # there is no fast/full distinction for it.
    # ======================================================

    text_layer: dict[int, str] = {}

    if cfg.prefer_text_layer:

        probe_last = min(
            total_pages,
            max(
                cfg.search_pages + cfg.prayer_extend_pages,
                cfg.page_end,
            ),
        )

        text_layer = text_layer_pages(
            pdf_path,
            1,
            probe_last,
            cfg.min_text_layer_chars,
        )

        if text_layer:
            print(
                f"\nText layer found on {len(text_layer)} of "
                f"{probe_last} probed page(s)."
            )
        else:
            print("\nNo usable text layer — OCR required.")

    # ======================================================
    # STEP 1
    # FAST PASS
    # ======================================================

    fast_text, fast_time = fast_scan(
        pdf_path,
        total_pages,
        cfg,
        text_layer,
    )

    scanned_through = min(cfg.search_pages, total_pages)

    # ======================================================
    # STEP 2
    # FIND PRAYER
    # ======================================================

    candidates = collect_candidates(fast_text)

    prayer = select_prayer(
        candidates["prayer"],
        total_pages,
        cfg,
        scanned_through,
    )

    # ======================================================
    # STEP 3
    # EXTEND THE SCAN IF PRAYER SITS NEAR THE EDGE
    #
    # A Prayer starting at or near the last scanned page left
    # find_prayer_end a 1-2 page window, so it fell through to its
    # default and the Prayer was cut off mid-relief.
    # ======================================================

    if (
        prayer
        and prayer.page > scanned_through - cfg.prayer_edge_slack
        and scanned_through < total_pages
    ):

        extend_to = min(
            scanned_through + cfg.prayer_extend_pages,
            total_pages,
        )

        print(
            f"\nPrayer near scan edge (page {prayer.page}); "
            f"extending fast pass to page {extend_to}..."
        )

        extra = fast_scan_range(
            pdf_path,
            scanned_through + 1,
            extend_to,
            cfg,
            text_layer,
        )

        fast_text.update(extra)

        scanned_through = extend_to

    # ======================================================
    # STEP 4
    # RESOLVE FULL PRAYER RANGE
    # ======================================================

    prayer_range = None

    if prayer:

        prayer_end = find_prayer_end(
            fast_text,
            prayer.page,
            total_pages,
            cfg,
            scanned_through,
        )

        prayer_range = (prayer.page, prayer_end)

        print(f"Prayer found: page {prayer.page}")
        print(f"Prayer range: pages {prayer.page}-{prayer_end}")

    else:
        print(f"Prayer NOT FOUND in first {scanned_through} pages.")

    # ======================================================
    # STEP 5
    # BODY RANGE
    # ======================================================

    body_end = min(cfg.page_end, total_pages)

    ranges: list[tuple[int, int]] = []

    if total_pages >= cfg.page_start:
        ranges.append((cfg.page_start, body_end))

    # ======================================================
    # STEP 6
    # ADD PRAYER RANGE
    #
    # merge_ranges collapses the overlap when the Prayer falls inside
    # the body range, so no page is processed twice.
    # ======================================================

    if prayer_range:
        ranges.append(prayer_range)

    # ======================================================
    # STEP 7
    # FULL PASS
    # ======================================================

    full_start = perf_counter()

    full_text_by_page = full_ocr_ranges(
        pdf_path,
        ranges,
        cfg,
        text_layer,
    )

    full_time = perf_counter() - full_start

    # ======================================================
    # STEP 8
    # EXTRACT BODY
    # ======================================================

    body_text = ""

    if total_pages >= cfg.page_start:

        body_text = combine_pages(
            full_text_by_page,
            cfg.page_start,
            body_end,
        )

    # ======================================================
    # STEP 9
    # EXTRACT FULL PRAYER
    # ======================================================

    prayer_text = ""

    if prayer_range:

        prayer_start_page, prayer_end_page = prayer_range

        prayer_full_text = combine_pages(
            full_text_by_page,
            prayer_start_page,
            prayer_end_page,
        )

        prayer_text = extract_prayer(prayer_full_text)

    # ======================================================
    # RESULT
    # ======================================================

    total_time = perf_counter() - total_start

    return {
        "pdf": str(pdf_path),

        "total_pages": total_pages,

        "search_pages": scanned_through,

        "text_layer_pages": sorted(text_layer),

        "locations": {
            "prayer": asdict(prayer) if prayer else None,
        },

        "ranges": {
            "body": (
                {"start": cfg.page_start, "end": body_end}
                if total_pages >= cfg.page_start
                else None
            ),

            "prayer": (
                {"start": prayer_range[0], "end": prayer_range[1]}
                if prayer_range
                else None
            ),
        },

        "sections": {
            "body": body_text,
            "prayer": prayer_text,
        },

        "timing": {
            "fast_scan": round(fast_time, 3),
            "full_ocr": round(full_time, 3),
            "total": round(total_time, 3),
        },
    }


# ==========================================================
# OCR PROCESSOR
# ==========================================================

class OCRProcessor:
    """
    Thin, reusable wrapper around process_document.

    Instances are independent: settings live on self.cfg rather than in
    module globals, so two processors with different DPIs can run
    concurrently without interfering.
    """

    def __init__(
        self,
        poppler_path: str | None = None,
        output_folder: Path | str = "section_output",
        tesseract_path: str | None = None,
        search_pages: int = 30,
        fast_dpi: int = 150,
        full_dpi: int = 220,
        max_workers: int | None = None,
        page_start: int = 2,
        page_end: int = 13,
        prefer_text_layer: bool = True,
        denoise: bool = True,
    ) -> None:

        # poppler_path is accepted for backward compatibility only.
        # Rendering now goes through PyMuPDF, so poppler is unused.
        self.poppler_path = poppler_path

        self.output_folder = Path(output_folder)

        self.cfg = OCRConfig(
            search_pages=search_pages,
            page_start=page_start,
            page_end=page_end,
            fast_dpi=fast_dpi,
            full_dpi=full_dpi,
            prefer_text_layer=prefer_text_layer,
            denoise=denoise,
            **(
                {"max_workers": max_workers}
                if max_workers is not None
                else {}
            ),
        )

        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path

        self.output_folder.mkdir(parents=True, exist_ok=True)

    # ======================================================
    # FORMAT OUTPUT
    # ======================================================

    def _format_output(self, result: dict) -> str:

        sections = result.get("sections", {})

        blocks = [
            (self.cfg.body_label, sections.get("body", "")),
            (self.cfg.prayer_label, sections.get("prayer", "")),
        ]

        output: list[str] = []

        for label, body in blocks:

            output.append("=" * 80)
            output.append(label)
            output.append("=" * 80)
            output.append("")
            output.append(body or "NOT FOUND")
            output.append("")

        return "\n".join(output).strip()

    # ======================================================
    # SAVE TEXT
    # ======================================================

    def save_text(
        self,
        pdf_path: str | Path,
        text: str,
    ) -> Path:

        stem = Path(pdf_path).stem or "document"

        output_path = self.output_folder / f"{stem}.txt"

        output_path.write_text(text, encoding="utf-8")

        return output_path

    # ======================================================
    # PROCESS PDF
    # ======================================================

    def process(
        self,
        pdf_path: str | Path,
        **overrides,
    ) -> tuple[str, Path]:
        """
        Process a PDF on disk.

        Per-call overrides (e.g. full_dpi=300) produce a copy of the
        config rather than mutating shared state.
        """

        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        cfg = replace(self.cfg, **overrides) if overrides else self.cfg

        result = process_document(pdf_path, cfg)

        text = self._format_output(result)

        # Clean the complete extracted text before saving
        text = clean_ocr_text(text)

        txt_path = self.save_text(pdf_path, text)

        return text, txt_path

    # ======================================================
    # PROCESS BYTES
    # ======================================================

    def process_bytes(
        self,
        pdf_bytes: bytes,
        filename: str = "document.pdf",
        **overrides,
    ) -> str:

        if not pdf_bytes:
            raise ValueError("PDF bytes are empty.")

        cfg = replace(self.cfg, **overrides) if overrides else self.cfg

        # Take only the final path component so an uploaded filename
        # cannot steer the write outside output_folder.
        safe_stem = Path(Path(filename).name).stem or "document"

        suffix = Path(filename).suffix or ".pdf"

        with tempfile.NamedTemporaryFile(
            suffix=suffix,
            delete=False,
        ) as temp_file:

            temp_file.write(pdf_bytes)
            temp_path = Path(temp_file.name)

        try:
            result = process_document(temp_path, cfg)

            # ALL pages have already been extracted here
            text = self._format_output(result)

            # CLEAN THE COMPLETE OCR OUTPUT
            text = clean_ocr_text(text)

            # ONLY NOW SAVE IT
            output_path = self.output_folder / f"{safe_stem}.txt"

            output_path.write_text(
                text,
                encoding="utf-8"
            )

            return text

        finally:
            # missing_ok plus OSError: on Windows the tesseract subprocess
            # can still hold a handle, which raises PermissionError rather
            # than FileNotFoundError.
            try:
                temp_path.unlink(missing_ok=True)

            except OSError as exc:
                print(f"  Could not remove temp file {temp_path}: {exc}")
