from __future__ import annotations

import re
import time
from pathlib import Path


# ==========================================================
# CONFIG
# ==========================================================

INPUT_FOLDER = Path(
    r"C:\Users\steph\OneDrive\Desktop\CEG\CCMS v2\section_output"
)

OUTPUT_FOLDER = INPUT_FOLDER / "prayer_output"


# ==========================================================
# PRAYER EXTRACTOR
# ==========================================================

class PrayerExtractor:

    START_PATTERNS = [
        r"\bFULL\s+PRAYER\b",
        r"\bWHEREFORE\b",
    ]

    END_PATTERNS = [
        r"\bINTERIM\s+PRAYER\b",
        r"\bSCHEDULE\s+LAND\b",
        r"\bADDRESS\s+FOR\s+SERVICE\b",
        r"\bAFFIDAVIT\b",
        r"\bVERIFICATION\b",
    ]

    def extract(
        self,
        text: str,
    ) -> str:

        if not text:
            return ""

        text = text.replace(
            "\r\n",
            "\n",
        ).replace(
            "\r",
            "\n",
        )

        # --------------------------------------------------
        # 1. Prefer FULL PRAYER
        # --------------------------------------------------

        match = re.search(
            r"\bFULL\s+PRAYER\b",
            text,
            flags=re.IGNORECASE,
        )

        if match:

            prayer = text[
                match.end():
            ]

            prayer = self._clean_section(
                prayer,
                full_prayer=True,
            )

            if prayer:
                return prayer

        # --------------------------------------------------
        # 2. Fallback to WHEREFORE
        # --------------------------------------------------

        match = re.search(
            r"\bWHEREFORE\b",
            text,
            flags=re.IGNORECASE,
        )

        if match:

            prayer = text[
                match.start():
            ]

            prayer = self._clean_section(
                prayer,
                full_prayer=False,
            )

            if prayer:
                return prayer

        return ""

    def _clean_section(
        self,
        text: str,
        full_prayer: bool,
    ) -> str:

        # --------------------------------------------------
        # Find end of prayer
        # --------------------------------------------------

        end_positions = []

        for pattern in self.END_PATTERNS:

            match = re.search(
                pattern,
                text,
                flags=re.IGNORECASE,
            )

            if match:

                end_positions.append(
                    match.start()
                )

        if end_positions:

            text = text[
                :min(end_positions)
            ]

        # --------------------------------------------------
        # Clean OCR lines
        # --------------------------------------------------

        cleaned_lines = []

        for line in text.splitlines():

            line = line.strip()

            if not line:
                continue

            # Remove OCR table/column marker
            line = re.sub(
                r"^\|\s*",
                "",
                line,
            )

            # Remove isolated separators
            if re.fullmatch(
                r"[-_=~]+",
                line,
            ):
                continue

            # Remove isolated page numbers
            if re.fullmatch(
                r"\d+",
                line,
            ):
                continue

            # Remove lines containing only symbols
            if re.fullmatch(
                r"[^A-Za-z0-9]+",
                line,
            ):
                continue

            # Normalize whitespace
            line = re.sub(
                r"\s+",
                " ",
                line,
            )

            # Remove spaces before punctuation
            line = re.sub(
                r"\s+([,.;:])",
                r"\1",
                line,
            )

            cleaned_lines.append(
                line
            )

        if not cleaned_lines:
            return ""

        # --------------------------------------------------
        # Preserve prayer paragraphs / clauses
        # --------------------------------------------------

        return "\n".join(
            cleaned_lines
        ).strip()


# ==========================================================
# TEST ALL FILES
# ==========================================================

print()
print("=" * 80)
print("PRAYER EXTRACTION - FULL FOLDER TEST")
print("=" * 80)

if not INPUT_FOLDER.exists():

    raise FileNotFoundError(
        f"Input folder not found:\n{INPUT_FOLDER}"
    )

OUTPUT_FOLDER.mkdir(
    parents=True,
    exist_ok=True,
)

txt_files = sorted(
    INPUT_FOLDER.glob("*.txt")
)

if not txt_files:

    raise FileNotFoundError(
        f"No TXT files found in:\n{INPUT_FOLDER}"
    )

print(
    f"Input folder : {INPUT_FOLDER}"
)

print(
    f"TXT files    : {len(txt_files)}"
)

print(
    f"Output folder: {OUTPUT_FOLDER}"
)

print("=" * 80)

extractor = PrayerExtractor()

total_start = time.perf_counter()

success_count = 0
failed_count = 0

results = []


# ==========================================================
# PROCESS EACH FILE
# ==========================================================

for index, txt_path in enumerate(
    txt_files,
    start=1,
):

    start = time.perf_counter()

    try:

        text = txt_path.read_text(
            encoding="utf-8",
            errors="replace",
        )

        prayer = extractor.extract(
            text
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        # --------------------------------------------------
        # Save extracted prayer
        # --------------------------------------------------

        output_path = (
            OUTPUT_FOLDER
            / txt_path.name
        )

        output_path.write_text(
            prayer,
            encoding="utf-8",
        )

        if prayer:

            success_count += 1
            status = "FOUND"

        else:

            failed_count += 1
            status = "NOT FOUND"

        results.append(
            {
                "file": txt_path.name,
                "status": status,
                "characters": len(prayer),
                "time": elapsed,
            }
        )

        print(
            f"[{index:03d}/{len(txt_files):03d}] "
            f"{status:<10} "
            f"{txt_path.name:<45} "
            f"{len(prayer):>7,} chars "
            f"{elapsed:.3f}s"
        )

    except Exception as e:

        failed_count += 1

        elapsed = (
            time.perf_counter()
            - start
        )

        results.append(
            {
                "file": txt_path.name,
                "status": "ERROR",
                "characters": 0,
                "time": elapsed,
            }
        )

        print(
            f"[{index:03d}/{len(txt_files):03d}] "
            f"{'ERROR':<10} "
            f"{txt_path.name:<45} "
            f"{str(e)}"
        )


# ==========================================================
# SUMMARY
# ==========================================================

total_elapsed = (
    time.perf_counter()
    - total_start
)

print()
print("=" * 80)
print("TEST SUMMARY")
print("=" * 80)

print(
    f"Total files     : {len(txt_files)}"
)

print(
    f"Prayer found    : {success_count}"
)

print(
    f"Prayer not found: {failed_count}"
)

print(
    f"Total time      : {total_elapsed:.3f} seconds"
)

if txt_files:

    print(
        f"Average/file    : "
        f"{total_elapsed / len(txt_files):.3f} seconds"
    )

print(
    f"Output folder   : {OUTPUT_FOLDER}"
)

print("=" * 80)


# ==========================================================
# SHOW FAILED FILES
# ==========================================================

not_found = [
    r["file"]
    for r in results
    if r["status"] in {
        "NOT FOUND",
        "ERROR",
    }
]

if not_found:

    print()
    print("=" * 80)
    print("FILES REQUIRING REVIEW")
    print("=" * 80)

    for filename in not_found:
        print(filename)

    print("=" * 80)