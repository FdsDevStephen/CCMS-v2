from pathlib import Path

from extractor.survey_extractor import SurveyExtractor
from extractor.normalizer import Normalizer


def main():

    text = Path(
        r"C:\Users\steph\OneDrive\Desktop\CEG\CCMS v2\output_text\WP-30154-2024-B.txt"
    ).read_text(
        encoding="utf-8",
        errors="ignore",
    )

    extractor = SurveyExtractor(text)

    surveys = extractor.extract()

    print(f"Before Normalization : {len(surveys)}")
    print(surveys)

    normalized = Normalizer.normalize_survey_numbers(surveys)

    print(f"\nAfter Normalization : {len(normalized)}")
    print(normalized)


if __name__ == "__main__":
    main()