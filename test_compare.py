"""Read Excel and compare with extraction results."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import openpyxl

wb = openpyxl.load_workbook("results/legal_extraction_results.xlsx")
ws = wb.active

print(f"Sheet: {ws.title}, Rows: {ws.max_row}, Cols: {ws.max_column}")
print()

# Get headers
headers = [c.value for c in next(ws.iter_rows(max_row=1))]
print(f"Headers: {headers}\n")

# Print all data rows
for row in ws.iter_rows(min_row=2, values_only=True):
    d = dict(zip(headers, row))
    # Print key fields
    case = d.get("Case Number", d.get("case_number", "?"))
    sections = d.get("Sections", d.get("sections", ""))
    acts = d.get("Acts", d.get("acts", ""))
    surveys = d.get("Survey Numbers", d.get("survey_numbers", ""))
    locations = d.get("Survey Locations", d.get("survey_locations", ""))
    prayer = d.get("Prayer", d.get("prayer", ""))
    
    print(f"=== {case} ===")
    print(f"  Sections: {sections}")
    print(f"  Acts: {acts}")
    print(f"  Surveys: {surveys}")
    print(f"  Locations: {locations}")
    if prayer:
        p = str(prayer)[:100]
        print(f"  Prayer: {p}...")
    print()
