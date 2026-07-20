from pathlib import Path

from extractor.survey_extractor import SurveyExtractor
from extractor.survey_location import SurveyLocationExtractor
from extractor.normalizer import Normalizer


def main():

    text = Path(
        r"C:\Users\steph\OneDrive\Desktop\CEG\CCMS v2\output_text\WP-13437-2022-B.txt"
    ).read_text(
        encoding="utf-8",
        errors="ignore",
    )

    # -----------------------------
    # Extract Survey Numbers
    # -----------------------------
    survey_numbers = SurveyExtractor(text).extract()
    survey_numbers = Normalizer.normalize_survey_numbers(survey_numbers)

    print("=" * 80)
    print("SURVEY NUMBERS")
    print("=" * 80)

    for survey in survey_numbers:
        print(survey)

    # -----------------------------
    # Extract Locations
    # -----------------------------
    # -----------------------------
    # Build Contexts
    # -----------------------------
    extractor = SurveyLocationExtractor(text)

    contexts = extractor._build_contexts(survey_numbers)

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

    # -----------------------------
    # Extract Locations
    # -----------------------------
    locations = extractor.extract(survey_numbers)

    locations = Normalizer.normalize_survey_locations(locations)

    print("\n" + "=" * 80)
    print("SURVEY LOCATIONS")
    print("=" * 80)

    for location in locations:
        print(location)


if __name__ == "__main__":
    main()