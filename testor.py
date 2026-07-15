from pathlib import Path
import json

from extractor.regex_extractor import RegexExtractor
from extractor.text_chunker import TextChunker
from extractor.prompts import build_act_extraction_prompt
from extractor.llm.factory import get_llm_client
from extractor.parser import LLMResponseParser
from extractor.normalizer import Normalizer

# ==========================================================
# INPUT TEXT FILE
# ==========================================================

TEXT_FILE = Path(
    r"C:\Users\steph\OneDrive\Desktop\CEG\CCMS v2\output_text\WP-202220-2023-G.txt"
)

text = TEXT_FILE.read_text(
    encoding="utf-8",
    errors="ignore",
)

# ==========================================================
# STEP 1 - REGEX
# ==========================================================

regex = RegexExtractor(text)

sections = regex.extract_sections()

sections = Normalizer.normalize_sections(sections)

print("=" * 80)
print("REGEX SECTIONS")
print("=" * 80)

for section in sections:
    print(section)

# ==========================================================
# STEP 2 - CHUNKING
# ==========================================================

chunker = TextChunker(
    chunk_size=3000,
    overlap=300,
)

chunks = chunker.split(text)

print("\n")
print("=" * 80)
print(f"TOTAL CHUNKS : {len(chunks)}")
print("=" * 80)

# ==========================================================
# STEP 3 - LLM
# ==========================================================

llm = get_llm_client()

all_acts = []

all_mappings = []

# ==========================================================
# PROCESS CHUNKS
# ==========================================================

for index, chunk in enumerate(chunks, start=1):

    print("\n")
    print("=" * 80)
    print(f"CHUNK {index}")
    print("=" * 80)

    prompt = build_act_extraction_prompt(
        chunk,
        sections,
    )

    response = llm.generate(prompt)

    print("\nRAW RESPONSE\n")

    print(response)

    result = LLMResponseParser.parse(response)

    all_acts.extend(
        result.get("acts", [])
    )

    all_mappings.extend(
        result.get("act_section_mapping", [])
    )

# ==========================================================
# STEP 4 - NORMALIZE
# ==========================================================

all_acts = Normalizer.normalize_acts(
    all_acts
)

all_mappings = Normalizer.normalize_act_section_mapping(
    all_mappings
)

# ==========================================================
# FINAL RESULT
# ==========================================================

final_result = {
    "acts": all_acts,
    "act_section_mapping": all_mappings,
}

print("\n")
print("=" * 80)
print("FINAL RESULT")
print("=" * 80)

print(
    json.dumps(
        final_result,
        indent=4,
        ensure_ascii=False,
    )
)

# ==========================================================
# PRETTY PRINT
# ==========================================================

print("\n")
print("=" * 80)
print("ACT → SECTION MAPPING")
print("=" * 80)

for item in final_result["act_section_mapping"]:

    print(f"\n📘 {item['act']}")

    if item["sections"]:

        for section in item["sections"]:

            print(f"   • {section}")

    else:

        print("   No Sections")