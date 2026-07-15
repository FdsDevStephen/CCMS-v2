from pathlib import Path

from extractor.normalizer import Normalizer
from extractor.survey_extractor import SurveyExtractor
from extractor.section_extractor import SectionExtractor


text = Path(
    r"C:\Users\steph\OneDrive\Desktop\CEG\CCMS v2\output_text\WP-30154-2024-B.txt"
).read_text(
    encoding="utf-8",
    errors="ignore",
)

# ==========================================================
# Survey Numbers
# ==========================================================

survey_extractor = SurveyExtractor(text)

surveys = survey_extractor.extract()

print("=" * 70)
print("SURVEY NUMBERS")
print("=" * 70)

for survey in surveys:
    print(survey)

print(f"\nTotal Survey Numbers : {len(surveys)}")

print()

# ==========================================================
# Sections
# ==========================================================

section_extractor = SectionExtractor(text)

sections = section_extractor.extract()

print("=" * 70)
print("SECTIONS")
print("=" * 70)

normalized_sections = Normalizer.normalize_sections(sections)

for section in normalized_sections:
    print(section)

print(f"\nTotal Sections : {len(normalized_sections)}")