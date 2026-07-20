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
        print(">>> SURVEY LOCATION EXTRACTOR VERSION 2 <<<")
        contexts = self._build_contexts(survey_numbers)

        print("\n" + "=" * 100)
        print("SURVEY CONTEXTS")
        print("=" * 100)

        for i, context in enumerate(contexts, start=1):
            print(f"\nContext {i}")
            print("-" * 100)
            print(f"Survey Number : {context['survey_number']}")
            print("-" * 100)
            print(context["context"])
            print("-" * 100)

        if not contexts:
            return []

        return self._extract_locations(contexts)

    def _build_contexts(self, survey_numbers: list[str]):

        contexts = []

        matches = list(self.SURVEY_PATTERN.finditer(self.text))

        MAX_CONTEXTS = 5

        for i, match in enumerate(matches):

            # Stop after collecting 10 contexts
            if len(contexts) >= MAX_CONTEXTS:
                break

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
        print(f"Built {len(contexts)} contexts")
        return contexts



    def _extract_locations(self, contexts: list[dict]) -> list[dict]:
        results = []

        # Process one context at a time
        for context in contexts:
            prompt = build_location_prompt([context])

            response = self.client.generate(prompt)

            try:
                print("=" * 80)
                print(response)
                print("=" * 80)

                result = json.loads(response)
                results = self._fill_missing_locations(results)

                if isinstance(result, list):
                    results.extend(result)
                else:
                    results.append(result)

            except Exception:
                print(f"Failed to parse response:\n{response}")

        # ------------------------------------------------------------------
        # STEP 1: Merge all occurrences of the same survey number
        # ------------------------------------------------------------------
        merged = {}

        for location in results:
            survey = location["survey_number"]

            if survey not in merged:
                merged[survey] = location.copy()
            else:
                existing = merged[survey]

                for field in ("village", "hobli", "taluk", "district"):
                    if existing.get(field) is None and location.get(field) is not None:
                        existing[field] = location[field]

        results = list(merged.values())

        # ------------------------------------------------------------------
        # STEP 2: Find the most complete location
        # ------------------------------------------------------------------
        best_location = None
        best_score = -1

        for location in results:
            score = sum(
                location.get(field) is not None
                for field in ("village", "hobli", "taluk", "district")
            )

            if score > best_score:
                best_score = score
                best_location = location

        # ------------------------------------------------------------------
        # STEP 3: Copy best location to surveys with NO location information
        # ------------------------------------------------------------------
        if best_location:
            for location in results:

                score = sum(
                    location.get(field) is not None
                    for field in ("village", "hobli", "taluk", "district")
                )

                if score == 0:
                    for field in ("village", "hobli", "taluk", "district"):
                        location[field] = best_location.get(field)

        return results
    
    
    def _fill_missing_locations(self, locations: list[dict]) -> list[dict]:
        """
        Fill missing location fields using the most complete record.
        """

        if not locations:
            return locations

        # Find the most complete location
        best = max(
            locations,
            key=lambda x: sum(
                x.get(field) is not None
                for field in ("village", "hobli", "taluk", "district")
            ),
        )

        for location in locations:
            for field in ("village", "hobli", "taluk", "district"):
                if location.get(field) is None:
                    location[field] = best.get(field)

        return locations