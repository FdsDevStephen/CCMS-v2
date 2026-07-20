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

    def _build_contexts(self, survey_numbers: list[str]) -> list[dict]:
        contexts = []
        seen = set()

        for match in self.SURVEY_PATTERN.finditer(self.text):
            raw_survey = match.group("survey_number")

            normalized = Normalizer.normalize_survey_numbers([raw_survey])

            if not normalized:
                continue

            survey_number = normalized[0]

            if survey_number not in survey_numbers:
                continue

            if survey_number in seen:
                continue

            seen.add(survey_number)

            start = max(0, match.start() - self.CONTEXT_BEFORE)
            end = min(len(self.text), match.end() + self.CONTEXT_AFTER)

            context = " ".join(self.text[start:end].split())

            contexts.append(
                {
                    "survey_number": survey_number,
                    "context": context,
                }
            )

        return contexts



    def _extract_locations(self, contexts: list[dict]) -> list[dict]:
        results = []

        # Process one context at a time
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
                print(f"Failed to parse response:\n{response}")

        # Merge results for the same survey number
        merged = {}

        for location in results:
            survey_number = location.get("survey_number")

            if survey_number not in merged:
                merged[survey_number] = location
            else:
                existing = merged[survey_number]

                for field in ("village", "hobli", "taluk", "district"):
                    if existing.get(field) is None and location.get(field) is not None:
                        existing[field] = location[field]

        return list(merged.values())

        
