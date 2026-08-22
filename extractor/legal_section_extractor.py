from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import pytesseract
from pdf2image import convert_from_path
from rapidfuzz.fuzz import ratio


@dataclass
class PageOCR:
    page: int
    text: str


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

# Section labels that should terminate an extraction even though we don't return them.
STOP_HEADINGS = [
    "FACTS IN BRIEF",
    "BRIEF FACTS OF THE CASE",
    "BRIEF FACTS",
    "FACTS OF THE CASE",
    "GROUNDS",
    "GROUNDS FOR INTERIM PRAYER",
    "GROUNDS FOR INTERIM RELIEF",
    "PRAYER",
    "INTERIM PRAYER",
    "AFFIDAVIT",
    "VERIFYING AFFIDAVIT",
    "VERIFICATION",
    "ADDRESS FOR SERVICE",
    "VAKALATH",
]


def normalize(text: str) -> str:
    return re.sub(r"[^A-Z0-9]+", " ", text.upper()).strip()


def is_heading_line(line: str, heading: str, threshold: int = 88) -> bool:
    n_line = normalize(line)
    n_heading = normalize(heading)
    if not n_line:
        return False
    if n_line == n_heading:
        return True
    return ratio(n_line, n_heading) >= threshold


def find_heading(line: str):
    # Longer headings first so INTERIM PRAYER isn't confused with PRAYER.
    choices = []
    for section, headings in SECTION_HEADINGS.items():
        for heading in headings:
            choices.append((len(normalize(heading)), section, heading))
    choices.sort(reverse=True)

    for _, section, heading in choices:
        if is_heading_line(line, heading):
            return section, heading
    return None


def find_stop_heading(line: str):
    for heading in sorted(STOP_HEADINGS, key=lambda x: len(normalize(x)), reverse=True):
        if is_heading_line(line, heading):
            return heading
    return None


class LegalSectionExtractor:
    def __init__(
        self,
        poppler_path: str | None = None,
        tesseract_path: str | None = None,
        scan_dpi: int = 120,
        extract_dpi: int = 220,
        initial_scan_pages: int = 20,
    ) -> None:
        self.poppler_path = poppler_path
        self.scan_dpi = scan_dpi
        self.extract_dpi = extract_dpi
        self.initial_scan_pages = initial_scan_pages

        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path

    def _render(self, pdf_path: str | Path, first_page: int, last_page: int, dpi: int):
        kwargs = {
            "dpi": dpi,
            "first_page": first_page,
            "last_page": last_page,
            "thread_count": 4,
        }
        if self.poppler_path:
            kwargs["poppler_path"] = self.poppler_path
        return convert_from_path(str(pdf_path), **kwargs)

    @staticmethod
    def _prep(image):
        arr = np.array(image)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        gray = cv2.medianBlur(gray, 3)
        return gray

    def _ocr_detection(self, image) -> str:
        gray = self._prep(image)
        return pytesseract.image_to_string(
            gray,
            lang="eng",
            config="--oem 3 --psm 11",
        )

    def _ocr_extraction(self, image) -> str:
        gray = self._prep(image)
        return pytesseract.image_to_string(
            gray,
            lang="eng",
            config="--oem 3 --psm 6",
        )

    @staticmethod
    def _is_toc_page(text: str) -> bool:
        n = normalize(text)
        return (
            " INDEX " in f" {n} "
            or "CONTENTS" in n
            or ("PAGE NO" in n and "ANNEXURE" in n)
        )

    def _detect_candidates(self, pages: List[PageOCR]):
        candidates = []
        for page in pages:
            lines = [line.strip() for line in page.text.splitlines() if line.strip()]
            toc = self._is_toc_page(page.text)
            for i, line in enumerate(lines):
                match = find_heading(line)
                if not match:
                    continue
                section, heading = match
                after_text = " ".join(lines[i + 1 :])
                candidates.append({
                    "page": page.page,
                    "line_index": i,
                    "section": section,
                    "heading": heading,
                    "line": line,
                    "toc": toc,
                    "content_after_chars": len(after_text),
                })
        return candidates

    def _choose_candidates(self, candidates):
        chosen = {}
        for section in SECTION_HEADINGS:
            opts = [x for x in candidates if x["section"] == section]
            if not opts:
                continue

            # Prefer non-TOC pages and headings followed by real body text.
            opts.sort(
                key=lambda x: (
                    not x["toc"],
                    x["content_after_chars"],
                    x["page"],
                ),
                reverse=True,
            )
            chosen[section] = opts[0]
        return chosen

    @staticmethod
    def _page_lines(text: str):
        return [line.strip() for line in text.splitlines()]

    def _extract_between(self, pages: Dict[int, str], start, stop_predicates):
        start_page = start["page"]
        start_line = start["line_index"]

        collected: List[str] = []
        started = False

        for page_no in sorted(pages):
            if page_no < start_page:
                continue

            lines = self._page_lines(pages[page_no])
            begin = start_line + 1 if page_no == start_page else 0

            for idx in range(begin, len(lines)):
                line = lines[idx].strip()
                if not line:
                    continue

                if not started:
                    started = True

                # Stop only when a heading occurs after the start line.
                if page_no == start_page and idx == start_line:
                    continue

                for pred in stop_predicates:
                    if is_heading_line(line, pred):
                        text = "\n".join(collected).strip()
                        return text

                collected.append(line)

        return "\n".join(collected).strip()

    def extract(self, pdf_path: str | Path) -> dict:
        pdf_path = Path(pdf_path)

        # PASS 1: cheap scan. Scan first N pages.
        scan_end = self.initial_scan_pages
        scan_images = self._render(pdf_path, 1, scan_end, self.scan_dpi)
        scan_pages = [PageOCR(i, self._ocr_detection(img)) for i, img in enumerate(scan_images, 1)]

        candidates = self._detect_candidates(scan_pages)
        chosen = self._choose_candidates(candidates)

        # If one or more requested headings were not found, scan the entire document once.
        missing = [s for s in SECTION_HEADINGS if s not in chosen]
        if missing:
            # Determine page count via a lightweight conversion attempt beyond current scan.
            # pdfinfo is simpler but not required; render until no pages isn't ideal, so use pypdf.
            try:
                from pypdf import PdfReader
                total_pages = len(PdfReader(str(pdf_path)).pages)
            except Exception:
                total_pages = scan_end

            if total_pages > scan_end:
                extra_images = self._render(pdf_path, scan_end + 1, total_pages, self.scan_dpi)
                extra_pages = [PageOCR(scan_end + i, self._ocr_detection(img)) for i, img in enumerate(extra_images, 1)]
                scan_pages.extend(extra_pages)
                candidates = self._detect_candidates(scan_pages)
                chosen = self._choose_candidates(candidates)

        # PASS 2: only OCR pages around the detected sections.
        relevant_pages = set()
        for item in chosen.values():
            p = item["page"]
            relevant_pages.update({p, p + 1, p - 1})
        relevant_pages = {p for p in relevant_pages if p >= 1}

        # We need continuity between sections. Include all pages from first relevant section to last.
        if relevant_pages:
            lo, hi = min(relevant_pages), max(relevant_pages)
        else:
            lo, hi = 1, min(scan_end, 20)

        # Expand enough to cover the section chain.
        hi = max(hi, min(scan_end, max([x["page"] for x in chosen.values()], default=scan_end) + 2))

        extract_images = self._render(pdf_path, lo, hi, self.extract_dpi)
        detailed = {}
        for offset, img in enumerate(extract_images):
            detailed[lo + offset] = self._ocr_extraction(img)

        # Re-detect headings on accurate OCR pages because line positions can change.
        detailed_candidates = self._detect_candidates([PageOCR(p, t) for p, t in detailed.items()])
        detailed_chosen = self._choose_candidates(detailed_candidates)

        # If detailed OCR didn't recover a heading, use detection candidate.
        for section, item in chosen.items():
            detailed_chosen.setdefault(section, item)

        result = {
            "document": pdf_path.name,
            "sections": {},
            "detected_headings": detailed_chosen,
        }

        # Extract each requested section independently.
        for section in SECTION_HEADINGS:
            start = detailed_chosen.get(section)
            if not start:
                result["sections"][section] = {"pages": [], "text": None}
                continue

            if section == "synopsis":
                stops = [x for x in STOP_HEADINGS if normalize(x) not in {normalize(h) for h in SECTION_HEADINGS["synopsis"]}]
                # Ensure brief-facts headings terminate synopsis.
                stops += SECTION_HEADINGS["brief_facts"]
            elif section == "brief_facts":
                stops = ["GROUNDS", "GROUNDS FOR INTERIM PRAYER", "GROUNDS FOR INTERIM RELIEF", "PRAYER", "INTERIM PRAYER", "AFFIDAVIT", "VERIFICATION", "ADDRESS FOR SERVICE", "VAKALATH"]
            elif section == "prayer":
                stops = ["INTERIM PRAYER", "ADDRESS FOR SERVICE", "VAKALATH", "VERIFICATION", "AFFIDAVIT"]
            else:
                stops = ["ADDRESS FOR SERVICE", "VAKALATH", "VERIFICATION", "AFFIDAVIT"]

            text = self._extract_between(detailed, start, stops)
            page_nums = []
            if text:
                start_page = start["page"]
                # approximate end page by search in collected detailed pages
                page_nums = list(range(start_page, max(detailed.keys()) + 1))

            result["sections"][section] = {"pages": page_nums, "text": text or None}

        return result


if __name__ == "__main__":
    import sys
    extractor = LegalSectionExtractor(scan_dpi=120, extract_dpi=180, initial_scan_pages=20)
    for arg in sys.argv[1:]:
        print(f"\nPROCESSING {arg}")
        result = extractor.extract(arg)
        print(json.dumps(result, indent=2, ensure_ascii=False))
