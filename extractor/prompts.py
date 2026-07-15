"""
Prompt templates for Legal Information Extraction.
"""


def build_act_extraction_prompt(document_text: str) -> str:

    return f"""
You are an expert Indian Legal Information Extraction Engine.

Your task is to identify EVERY Act explicitly mentioned in the document.

Read the COMPLETE document from beginning to end before answering.

Instructions:

- Extract every Act mentioned anywhere in the document.
- Continue scanning until the END of the document.
- An Act is legislation whose name contains the word "Act".
- Preserve the exact wording exactly as written.
- If a year is present, include the year.
- If no year is present, return only the Act name.
- Return every unique Act only once.
- Preserve the order of first appearance.

Do NOT:

- Infer missing words.
- Guess Act names.
- Expand abbreviations.
- Modify Act names.
- Return anything other than Acts.

Never extract:

- Constitution of India
- Articles
- Sections
- Rules
- Regulations
- Notifications
- Government Orders
- Circulars
- Ordinances
- Survey Numbers
- Case Numbers
- Dates
- Persons
- Places

Return ONLY valid JSON.

Output Format:

{{
    "acts": []
}}

=========================================================
Examples
=========================================================

Input:

Section 28 of the Karnataka Industrial Areas Development Act, 1966.

Output:

{{
    "acts": [
        "Karnataka Industrial Areas Development Act, 1966"
    ]
}}

---------------------------------------------------------

Input:

Proceedings were initiated under the Karnataka Land Revenue Act.

Output:

{{
    "acts": [
        "Karnataka Land Revenue Act"
    ]
}}

---------------------------------------------------------

Input:

The dispute involves the Karnataka Industrial Areas Development Act, 1966, the Karnataka Land Revenue Act, 1964 and the Indian Forest Act.

Output:

{{
    "acts": [
        "Karnataka Industrial Areas Development Act, 1966",
        "Karnataka Land Revenue Act, 1964",
        "Indian Forest Act"
    ]
}}

---------------------------------------------------------

Input:

The petition is filed under Article 226 of the Constitution of India.

Output:

{{
    "acts": []
}}

=========================================================
Document
=========================================================

{document_text}

=========================================================
Return ONLY valid JSON.
"""