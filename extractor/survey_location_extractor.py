from __future__ import annotations

import json

from extractor.llm.factory import get_llm_client
from extractor.location_prompt import build_location_prompt

from RAG.retriever import LegalRetriever


class SurveyLocationExtractor:

    def __init__(self):
        self.client = get_llm_client()
        self.retriever = LegalRetriever()

    def extract(
        self,
        survey_numbers: list[str],
        top_k: int = 3,
    ):

        print(">>> SURVEY LOCATION RAG EXTRACTOR <<<")

        contexts = self._build_contexts(
            survey_numbers,
            top_k,
        )

        print("\n" + "=" * 100)
        print("SURVEY CONTEXTS")
        print("=" * 100)

        for i, context in enumerate(
            contexts,
            start=1,
        ):

            print(f"\nContext {i}")
            print("-" * 100)

            print(
                f"Survey Number : "
                f"{context['survey_number']}"
            )

            print("-" * 100)

            print(
                context["context"]
            )

            print("-" * 100)

        if not contexts:
            return []

        return self._extract_locations(
            contexts
        )

    # ==========================================================
    # Retrieve Context From Qdrant
    # ==========================================================

    def _build_contexts(
        self,
        survey_numbers: list[str],
        top_k: int,
    ):

        contexts = []

        for survey_number in survey_numbers:

            query = (
                f"Survey Number {survey_number} "
                f"land village hobli taluk district"
            )

            results = self.retriever.search(
                query=query,
                top_k=top_k,
            )

            for result in results:

                contexts.append(
                    {
                        "survey_number": survey_number,
                        "context": result.payload[
                            "text"
                        ],
                    }
                )

        print(
            f"Built {len(contexts)} contexts"
        )

        return contexts

    # ==========================================================
    # LLM Location Extraction
    # ==========================================================

    def _extract_locations(
        self,
        contexts: list[dict],
    ) -> list[dict]:

        results = []

        for context in contexts:

            prompt = build_location_prompt(
                [context]
            )

            response = self.client.generate(
                prompt
            )

            try:

                print("=" * 80)
                print(response)
                print("=" * 80)

                result = json.loads(
                    response
                )

                if isinstance(
                    result,
                    list,
                ):
                    results.extend(
                        result
                    )

                else:
                    results.append(
                        result
                    )

            except Exception:

                print(
                    "Failed to parse response:"
                )

                print(response)

        return self._merge_results(
            results
        )

    # ==========================================================
    # Merge Results
    # ==========================================================

    def _merge_results(
        self,
        results: list[dict],
    ) -> list[dict]:

        merged = {}

        for location in results:

            survey = location.get(
                "survey_number"
            )

            if not survey:
                continue

            if survey not in merged:

                merged[survey] = (
                    location.copy()
                )

            else:

                existing = merged[
                    survey
                ]

                for field in (
                    "village",
                    "hobli",
                    "taluk",
                    "district",
                ):

                    if (
                        existing.get(field)
                        is None
                        and location.get(
                            field
                        ) is not None
                    ):

                        existing[field] = (
                            location[field]
                        )

        results = list(
            merged.values()
        )

        # Find most complete location

        best_location = None
        best_score = -1

        for location in results:

            score = sum(
                location.get(field)
                is not None
                for field in (
                    "village",
                    "hobli",
                    "taluk",
                    "district",
                )
            )

            if score > best_score:

                best_score = score
                best_location = location

        # Copy complete location only to
        # surveys having absolutely no data

        if best_location:

            for location in results:

                score = sum(
                    location.get(field)
                    is not None
                    for field in (
                        "village",
                        "hobli",
                        "taluk",
                        "district",
                    )
                )

                if score == 0:

                    for field in (
                        "village",
                        "hobli",
                        "taluk",
                        "district",
                    ):

                        location[field] = (
                            best_location.get(
                                field
                            )
                        )

        return results