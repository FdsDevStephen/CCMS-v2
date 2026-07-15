"""
Prompt templates for Legal Information Extraction.
"""


def build_act_extraction_prompt(
    document_text: str,
    sections: list[str],
) -> str:

    sections_text = "\n".join(sections)

    return f"""
You are an expert Indian Legal Information Extraction Engine.

Read the COMPLETE document before answering.

Your tasks are:

1. Extract every Act explicitly mentioned in the document.
2. Associate ONLY the provided Sections with the Act they explicitly belong to.

=========================================================
EXTRACTED SECTIONS
=========================================================

The following Sections have already been extracted.

Use ONLY these Sections.

{sections_text}

=========================================================
ACT EXTRACTION
=========================================================

Extract every Act explicitly mentioned in the document.

Rules:

- An Act is legislation whose name contains the word "Act".
- Preserve the exact wording.
- Include the year if present.
- If no year is written, return only the Act name.
- Return each unique Act only once.
- Preserve order of first appearance.

=========================================================
ACT → SECTION MAPPING
=========================================================

The Sections above are the ONLY Sections you may use.

Do NOT:

- Create new Sections.
- Modify Section numbers.
- Guess missing Sections.
- Infer legal relationships from your own knowledge.

A Section belongs to an Act ONLY if the document explicitly associates them.

Examples of explicit association:

✓ Section 19 of the Karnataka Land Revenue Act, 1964.

✓ Sections 19 and 20 under the Karnataka Land Revenue Act, 1964.

✓ Section 136(2) read with the Karnataka Land Revenue Act, 1964.

✓ Rule 108-D(3) of the Karnataka Land Revenue Rules, 1966.

The following are NOT explicit associations:

✗ Section 19 appears in one paragraph.

✗ Karnataka Land Revenue Act appears somewhere else on the page.

✗ The Act and Section appear in different contexts.

Never associate a Section merely because an Act appears nearby.

If multiple Acts are mentioned:

- Determine independently which Sections belong to each Act.
- A Section may belong to ONLY ONE Act.
- Never duplicate the same Section under multiple Acts.

If the document does NOT explicitly identify the Act for a Section:

DO NOT map that Section.

If an Act has no mapped Sections:

Return an empty list for that Act.

=========================================================
GENERAL RULES
=========================================================

- Read the COMPLETE document.
- Preserve exact wording.
- Preserve capitalization.
- Preserve punctuation.
- Remove duplicate Acts.
- Remove duplicate Sections.
- Preserve order of first appearance.
- Return ONLY valid JSON.
- Never explain your answer.

Never extract:

- Constitution of India
- Articles
- Rules (unless they are already provided in the extracted Sections)
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

=========================================================
OUTPUT FORMAT
=========================================================

{{
    "acts": [],
    "act_section_mapping": [
        {{
            "act": "",
            "sections": []
        }}
    ]
}}

=========================================================
EXAMPLES
=========================================================

Input

Sections

79A
79B
80

Document

Sections 79A, 79B and 80 of the Karnataka Land Reforms Act, 1961.

Output

{{
    "acts": [
        "Karnataka Land Reforms Act, 1961"
    ],
    "act_section_mapping": [
        {{
            "act": "Karnataka Land Reforms Act, 1961",
            "sections": [
                "79A",
                "79B",
                "80"
            ]
        }}
    ]
}}

---------------------------------------------------------

Input

Sections

136(2)
108(K)

Document

Section 136(2) and Section 108(K) of the Karnataka Land Revenue Act, 1964.

Output

{{
    "acts": [
        "Karnataka Land Revenue Act, 1964"
    ],
    "act_section_mapping": [
        {{
            "act": "Karnataka Land Revenue Act, 1964",
            "sections": [
                "136(2)",
                "108(K)"
            ]
        }}
    ]
}}

---------------------------------------------------------

Input

Sections

19
136(2)

Document

Section 19 of the Karnataka Land Reforms Act, 1961.

Section 136(2) of the Karnataka Land Revenue Act, 1964.

Output

{{
    "acts": [
        "Karnataka Land Reforms Act, 1961",
        "Karnataka Land Revenue Act, 1964"
    ],
    "act_section_mapping": [
        {{
            "act": "Karnataka Land Reforms Act, 1961",
            "sections": [
                "19"
            ]
        }},
        {{
            "act": "Karnataka Land Revenue Act, 1964",
            "sections": [
                "136(2)"
            ]
        }}
    ]
}}

---------------------------------------------------------

Input

Sections

19
136(2)

Document

Section 19 was referred.

The Karnataka Land Revenue Act, 1964 was also referred.

Output

{{
    "acts": [
        "Karnataka Land Revenue Act, 1964"
    ],
    "act_section_mapping": []
}}

=========================================================
DOCUMENT
=========================================================

{document_text}

=========================================================
Return ONLY valid JSON.
"""