from __future__ import annotations

import re
import json
from extractor.normalizer import Normalizer
from extractor.llm.factory import get_llm_client
from extractor.location_prompt import build_location_prompt


class SurveyLocationExtractor:
    SURVEY_PATTERN = re.compile(
        r"""
        (?:
            \bS\.?\s*Nos?\.?|
            \bSy\.?\s*Nos?\.?|
            \bRe-Sy\.?\s*Nos?\.?|
            \bRe-Survey\s*Nos?\.?|
            \bSurvey\s*Nos?\.?|
            \bSurvey\s*Numbers?|
            \bSurvey\s*Number
        )
        [\s:.-]*
        (?P<survey_number>\d+(?:/[A-Za-z0-9*]+)*)
        """,
        re.IGNORECASE | re.VERBOSE,
    )

    CONTEXT_BEFORE = 5
    CONTEXT_AFTER = 250

    def __init__(self, text: str):
        self.text = text
        self.client = get_llm_client()

    def extract(self, survey_numbers: list[str]):
        contexts = self._build_contexts(survey_numbers)

        if not contexts:
            return []

        return self._extract_locations(contexts)

    def _build_contexts(self, survey_numbers: list[str]):

        contexts = []

        matches = list(self.SURVEY_PATTERN.finditer(self.text))

        for i, match in enumerate(matches):

            # -----------------------------
            # Current survey
            # -----------------------------
            current_survey = Normalizer.normalize_survey_numbers([match.group(1)])[0]

            if current_survey not in survey_numbers:
                continue

            # -----------------------------
            # Context Start
            # -----------------------------
            start = match.start()

            # -----------------------------
            # Context End
            # -----------------------------
            # -----------------------------
            # Context End
            # -----------------------------
            MAX_CONTEXT = 300

            if i + 1 < len(matches):
                end = min(matches[i + 1].start(), start + MAX_CONTEXT)
            else:
                end = min(len(self.text), start + MAX_CONTEXT)

            context = self.text[start:end]

            contexts.append(
                {
                    "survey_number": current_survey,
                    "context": context,
                }
            )

        return contexts



    def _extract_locations(self, contexts: list[dict]) -> list[dict]:
        results = []

        for context in contexts:
            
            prompt = build_location_prompt([context])

            response = self.client.generate(prompt)

            try:
                result = json.loads(response)

                if isinstance(result, list):
                    results.extend(result)
                else:
                    results.append(result)

            except Exception:
                print(response)

        # -------------------------------------------------
        # Find the record with the most information
        # -------------------------------------------------
        best_location = None
        best_score = -1

        for location in results:
            score = sum(
                field is not None
                for field in (
                    location.get("village"),
                    location.get("hobli"),
                    location.get("taluk"),
                    location.get("district"),
                )
            )

            if score > best_score:
                best_score = score
                best_location = location

        # -------------------------------------------------
        # Fill missing fields from the best record
        # -------------------------------------------------
        if best_location:
            for location in results:

                if location is best_location:
                    continue

                if location.get("village") is None:
                    location["village"] = best_location.get("village")

                if location.get("hobli") is None:
                    location["hobli"] = best_location.get("hobli")

                if location.get("taluk") is None:
                    location["taluk"] = best_location.get("taluk")

                if location.get("district") is None:
                    location["district"] = best_location.get("district")

        return results