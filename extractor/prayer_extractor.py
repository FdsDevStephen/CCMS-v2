from __future__ import annotations

import re


class PrayerExtractor:
    """
    Strict prayer extractor for OCR legal documents.

    Strategy:
    1. Prefer FULL PRAYER.
    2. Otherwise use PRAYER heading.
    3. Otherwise use WHEREFORE.
    4. Otherwise use THEREFORE.
    5. Remove obvious OCR/page artifacts.
    6. Preserve legal wording.
    7. Preserve ANNEXURE references.
    8. Preserve prayer clauses.
    9. Stop before subsequent document sections.
    """

    # ==========================================================
    # START PATTERNS
    # ==========================================================

    FULL_PRAYER_PATTERN = re.compile(
        r"\bFULL\s+PRAYER\b",
        re.IGNORECASE,
    )

    PRAYER_HEADING_PATTERN = re.compile(
        r"""
        ^\s*
        [*|._\-~]*
        \s*
        PRAYER
        \s*
        [*|._\-~:]*
        \s*$
        """,
        re.IGNORECASE | re.VERBOSE,
    )

    WHEREFORE_PATTERN = re.compile(
        r"\bWHEREFORE\b",
        re.IGNORECASE,
    )

    THEREFORE_PATTERN = re.compile(
        r"\bTHEREFORE\b",
        re.IGNORECASE,
    )

    # ==========================================================
    # END PATTERNS
    # ==========================================================

    END_HEADING_PATTERNS = [
        re.compile(
            r"^\s*[*|._\-~]*\s*"
            r"INTERIM\s+PRAYER"
            r"\s*[:.\-_*|~]*\s*$",
            re.IGNORECASE,
        ),

        re.compile(
            r"^\s*[*|._\-~]*\s*"
            r"SCHEDULE(?:\s+LAND)?"
            r"\s*[:.\-_*|~]*\s*$",
            re.IGNORECASE,
        ),

        re.compile(
            r"^\s*[*|._\-~]*\s*"
            r"ADDRESS\s+FOR\s+SERVICE"
            r"\s*[:.\-_*|~]*\s*$",
            re.IGNORECASE,
        ),

        re.compile(
            r"^\s*[*|._\-~]*\s*"
            r"LIST\s+OF\s+"
            r"(?:AUTHORITIES|CITATIONS)"
            r"\s*[:.\-_*|~]*\s*$",
            re.IGNORECASE,
        ),

        re.compile(
            r"^\s*[*|._\-~]*\s*"
            r"VERIFICATION"
            r"\s*[:.\-_*|~]*\s*$",
            re.IGNORECASE,
        ),
    ]

    # ==========================================================
    # CLAUSE PATTERN
    # ==========================================================

    CLAUSE_PATTERN = re.compile(
        r"""
        ^\s*
        (?:
            \([a-z]\)
            |
            \([ivxlcdm]+\)
            |
            [a-z]\)
            |
            [ivxlcdm]+\)
            |
            [IVXLCDM]+\.
            |
            [a-z]\.
        )
        \s+
        """,
        re.IGNORECASE | re.VERBOSE,
    )

    # ==========================================================
    # SAFE STRUCTURAL CORRECTIONS
    # ==========================================================

    SAFE_CORRECTIONS = {
        "(ili)": "(iii)",
        "(ill)": "(iii)",
        "(lil)": "(iii)",

        "3r4": "3rd",
        "3™": "3rd",

        "7‘?": "7th",
        "7?": "7th",

        "8?": "8th",

        "9t*": "9th",
        "9t'": "9th",
    }

    # ==========================================================
    # KNOWN OCR ARTIFACTS
    # ==========================================================

    KNOWN_GARBAGE = {
        "gf",
        "gf.",
        "ie",
        "ie.",
        "t",
        "t.",
        "y",
        "y.",
        "ae",
        "ae.",
        "l",
        "ll",
        "lll",
        "—",
        "-",
        "_",
        "|",
        "*",
        "?",
    }

    # ==========================================================
    # PUBLIC METHOD
    # ==========================================================

    def extract(
        self,
        text: str,
    ) -> str:

        if not text:
            return ""

        text = self._normalize_text(
            text
        )

        # ======================================================
        # 1. FULL PRAYER
        # ======================================================

        match = self.FULL_PRAYER_PATTERN.search(
            text
        )

        if match:

            prayer = text[
                match.end():
            ]

            prayer = self._extract_until_end(
                prayer
            )

            prayer = self._clean(
                prayer
            )

            if prayer:
                return prayer

        # ======================================================
        # 2. PRAYER HEADING
        # ======================================================

        position = self._find_prayer_heading(
            text
        )

        if position is not None:

            prayer = text[
                position:
            ]

            prayer = self._extract_until_end(
                prayer
            )

            prayer = self._clean(
                prayer
            )

            if prayer:
                return prayer

        # ======================================================
        # 3. WHEREFORE
        # ======================================================

        match = self.WHEREFORE_PATTERN.search(
            text
        )

        if match:

            prayer = text[
                match.start():
            ]

            prayer = self._extract_until_end(
                prayer
            )

            prayer = self._clean(
                prayer
            )

            if prayer:
                return prayer

        # ======================================================
        # 4. THEREFORE
        # ======================================================

        match = self.THEREFORE_PATTERN.search(
            text
        )

        if match:

            prayer = text[
                match.start():
            ]

            prayer = self._extract_until_end(
                prayer
            )

            prayer = self._clean(
                prayer
            )

            if prayer:
                return prayer

        # ======================================================
        # 5. PETITIONER PRAYS FALLBACK
        # ======================================================

        match = re.search(
            r"\bthe\s+petitioners?\s+"
            r"(?:most\s+respectfully\s+)?"
            r"pray(?:s)?\b",
            text,
            flags=re.IGNORECASE,
        )

        if match:

            prayer = text[
                match.start():
            ]

            prayer = self._extract_until_end(
                prayer
            )

            prayer = self._clean(
                prayer
            )

            if prayer:
                return prayer

        return ""

    # ==========================================================
    # FIND PRAYER HEADING
    # ==========================================================

    def _find_prayer_heading(
        self,
        text: str,
    ) -> int | None:

        position = 0

        for line in text.splitlines(
            keepends=True
        ):

            if self.PRAYER_HEADING_PATTERN.match(
                line.strip()
            ):

                return (
                    position
                    + len(line)
                )

            position += len(line)

        return None

    # ==========================================================
    # EXTRACT UNTIL END
    # ==========================================================

    def _extract_until_end(
        self,
        text: str,
    ) -> str:

        output = []

        for line in text.splitlines():

            stripped = line.strip()

            if self._is_end_heading(
                stripped
            ):
                break

            output.append(
                line
            )

        return "\n".join(
            output
        )

    # ==========================================================
    # END HEADING
    # ==========================================================

    def _is_end_heading(
        self,
        line: str,
    ) -> bool:

        if not line:
            return False

        for pattern in self.END_HEADING_PATTERNS:

            if pattern.match(line):
                return True

        return False

    # ==========================================================
    # MAIN CLEANER
    # ==========================================================

    def _clean(
        self,
        text: str,
    ) -> str:

        raw_lines = text.splitlines()

        lines = []

        for raw_line in raw_lines:

            line = raw_line.strip()

            if not line:
                continue

            # --------------------------------------------------
            # Remove leading OCR symbols
            # --------------------------------------------------

            line = re.sub(
                r"^[*•·|¦_=~]+\s*",
                "",
                line,
            )

            # --------------------------------------------------
            # Remove trailing OCR symbols
            # --------------------------------------------------

            line = re.sub(
                r"\s+[|¦_=~]+\s*$",
                "",
                line,
            )

            line = line.strip()

            if not line:
                continue

            # --------------------------------------------------
            # Remove page numbers
            # --------------------------------------------------

            if re.fullmatch(
                r"\d{1,4}",
                line,
            ):
                continue

            # --------------------------------------------------
            # Remove pure punctuation
            # --------------------------------------------------

            if re.fullmatch(
                r"[^A-Za-z0-9]+",
                line,
            ):
                continue

            # --------------------------------------------------
            # Remove known OCR garbage
            # --------------------------------------------------

            if self._is_known_garbage(
                line
            ):
                continue

            # --------------------------------------------------
            # Remove suspicious short OCR fragments
            # --------------------------------------------------

            if self._is_short_noise(
                line
            ):
                continue

            # --------------------------------------------------
            # Structural OCR corrections
            # --------------------------------------------------

            line = self._apply_safe_corrections(
                line
            )

            # --------------------------------------------------
            # Clean spaces
            # --------------------------------------------------

            line = re.sub(
                r"[ \t]+",
                " ",
                line,
            )

            line = re.sub(
                r"\s+([,.;:])",
                r"\1",
                line,
            )

            line = re.sub(
                r"\(\s+",
                "(",
                line,
            )

            line = re.sub(
                r"\s+\)",
                ")",
                line,
            )

            line = line.strip()

            if line:
                lines.append(
                    line
                )

        if not lines:
            return ""

        # ======================================================
        # JOIN CLAUSES
        # ======================================================

        clauses = []

        current = ""

        for line in lines:

            if self._is_clause_start(
                line
            ):

                if current:

                    clauses.append(
                        current.strip()
                    )

                current = line

            else:

                if current:

                    current += " " + line

                else:

                    current = line

        if current:

            clauses.append(
                current.strip()
            )

        # ======================================================
        # FINAL CLEAN
        # ======================================================

        final = []

        for clause in clauses:

            clause = re.sub(
                r"\s+",
                " ",
                clause,
            )

            clause = re.sub(
                r"\s+([,.;:])",
                r"\1",
                clause,
            )

            clause = clause.strip()

            if clause:
                final.append(
                    clause
                )

        return "\n\n".join(
            final
        ).strip()

    # ==========================================================
    # CLAUSE DETECTION
    # ==========================================================

    def _is_clause_start(
        self,
        line: str,
    ) -> bool:

        return bool(
            self.CLAUSE_PATTERN.match(
                line
            )
        )

    # ==========================================================
    # KNOWN GARBAGE
    # ==========================================================

    def _is_known_garbage(
        self,
        line: str,
    ) -> bool:

        normalized = line.strip().lower()

        return (
            normalized
            in self.KNOWN_GARBAGE
        )

    # ==========================================================
    # SHORT OCR NOISE
    # ==========================================================

    def _is_short_noise(
        self,
        line: str,
    ) -> bool:

        # Never remove clause markers
        if self._is_clause_start(
            line
        ):
            return False

        # Never remove numbers mixed with legal text
        if re.search(
            r"\d",
            line,
        ):
            return False

        words = line.split()

        # One/two-character fragments
        if len(line) <= 2:
            return True

        # Very short isolated lowercase OCR fragments
        if (
            len(words) == 1
            and len(line) <= 3
            and line.isalpha()
            and line.islower()
        ):
            return True

        return False

    # ==========================================================
    # SAFE CORRECTIONS
    # ==========================================================

    def _apply_safe_corrections(
        self,
        line: str,
    ) -> str:

        corrections = sorted(
            self.SAFE_CORRECTIONS.items(),
            key=lambda item: len(
                item[0]
            ),
            reverse=True,
        )

        for wrong, correct in corrections:

            line = re.sub(
                re.escape(wrong),
                correct,
                line,
                flags=re.IGNORECASE,
            )

        return line

    # ==========================================================
    # NORMALIZE TEXT
    # ==========================================================

    def _normalize_text(
        self,
        text: str,
    ) -> str:

        text = text.replace(
            "\r\n",
            "\n",
        )

        text = text.replace(
            "\r",
            "\n",
        )

        text = text.replace(
            "\u00a0",
            " ",
        )

        return text