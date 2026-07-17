from pathlib import Path
import json

from extractor.survey_extractor import SurveyExtractor
from extractor.survey_location import SurveyLocationExtractor


def main():

    text = Path(
        r"C:\Users\steph\OneDrive\Desktop\CEG\CCMS v2\output_text\WA-100531-2023-D.txt"
    ).read_text(
        encoding="utf-8",
        errors="ignore",
    )

    survey_numbers = SurveyExtractor(text).extract()

    print("=" * 100)
    print("SURVEY NUMBERS")
    print("=" * 100)
    print(survey_numbers)
    print(f"\nTotal Survey Numbers : {len(survey_numbers)}")

    extractor = SurveyLocationExtractor(text)

    # -----------------------------
    # Print Contexts
    # -----------------------------
    contexts = extractor._build_contexts(survey_numbers)

    print("\n")
    print("=" * 100)
    print("SURVEY CONTEXTS")
    print("=" * 100)

    for i, item in enumerate(contexts, start=1):
        print(f"\nContext #{i}")
        print(f"Survey Number : {item['survey_number']}")
        print("-" * 100)
        print(item["context"])
        print("-" * 100)

    # -----------------------------
    # LLM Extraction
    # -----------------------------
    locations = extractor.extract(survey_numbers)

    print("\n")
    print("=" * 100)
    print("LLM OUTPUT")
    print("=" * 100)

    print(f"Type : {type(locations)}")

    if isinstance(locations, list):
        print(f"Items : {len(locations)}")
        print(json.dumps(locations, indent=4, ensure_ascii=False))

    elif isinstance(locations, dict):
        print(json.dumps(locations, indent=4, ensure_ascii=False))

    else:
        print(locations)


if __name__ == "__main__":
    main()