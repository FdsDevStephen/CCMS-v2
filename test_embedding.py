from pathlib import Path
from extractor.extractor import LegalExtractor

# 1. Provide a path to any sample PDF from your dataset
SAMPLE_PDF = "path/to/your/sample_document.pdf"

def run_test():
    pdf_path = Path(SAMPLE_PDF)
    
    if not pdf_path.exists():
        print(f"Error: Could not find file at '{pdf_path}'. Please replace it with a valid PDF path.")
        return

    print("=" * 60)
    print(f"Starting test on: {pdf_path.name}")
    print("=" * 60)

    # 2. Initialize extractor and run
    extractor = LegalExtractor()
    result = extractor.extract(pdf_path)

    # 3. Print output summary
    print("\n" + "=" * 60)
    print("TEST COMPLETED - EXTRACTED DATA")
    print("=" * 60)
    print(f"Case Number:      {result.get('case_number')}")
    print(f"Acts Found:        {result.get('acts')}")
    print(f"Survey Locations:  {result.get('survey_locations')}")
    print(f"Sections (Regex):  {result.get('sections')}")
    print("=" * 60)

if __name__ == "__main__":
    run_test()