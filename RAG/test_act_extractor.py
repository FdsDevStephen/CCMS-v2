import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent)
)

from act_extractor import ActExtractor


print("=" * 80)
print("ACT EXTRACTION")
print("=" * 80)


extractor = ActExtractor()

result = extractor.extract(
    top_k=5
)


print("\nRESULT:")
print(result)