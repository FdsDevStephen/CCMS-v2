"""
Test survey location extraction with pipeline-style chunking, using regex only.
"""

from __future__ import annotations

import json
from pathlib import Path

from extractor.normalizer import Normalizer
from extractor.survey_extractor import SurveyExtractor
from extractor.survey_location_extractor import SurveyLocationExtractor
from extractor.text_chunker import TextChunker


TEXT_FILE = Path(r"C:\Users\steph\OneDrive\Desktop\CEG\CCMS v2\output_text\WP-19110-2024-B.txt")


def main() -> None:
    text = TEXT_FILE.read_text(encoding="utf-8", errors="ignore")

    chunker = TextChunker(
        chunk_size=3000,
        overlap=300,
    )

    chunks = chunker.split(text)

    all_survey_numbers = []
    all_survey_locations = []

    print("=" * 80)
    print(f"Total chunks: {len(chunks)}")
    print("=" * 80)

    for index, chunk in enumerate(chunks, start=1):
        print(f"\nProcessing chunk {index}/{len(chunks)}")
        print("-" * 80)

        raw_survey_numbers = SurveyExtractor(chunk).extract()
        survey_numbers = Normalizer.normalize_survey_numbers(raw_survey_numbers)

        survey_locations = SurveyLocationExtractor(chunk).extract(survey_numbers)
        survey_locations = Normalizer.normalize_survey_locations(survey_locations)

        print("Survey numbers:")
        print(json.dumps(survey_numbers, indent=4, ensure_ascii=False))

        print("Survey locations:")
        print(json.dumps(survey_locations, indent=4, ensure_ascii=False))

        all_survey_numbers.extend(survey_numbers)
        all_survey_locations.extend(survey_locations)

    final_survey_numbers = Normalizer.normalize_survey_numbers(all_survey_numbers)
    final_survey_locations = Normalizer.normalize_survey_locations(
        all_survey_locations
    )

    print("\n" + "=" * 80)
    print("FINAL SURVEY NUMBERS")
    print("=" * 80)
    print(json.dumps(final_survey_numbers, indent=4, ensure_ascii=False))

    print("\n" + "=" * 80)
    print("FINAL SURVEY LOCATIONS")
    print("=" * 80)
    print(json.dumps(final_survey_locations, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    main()