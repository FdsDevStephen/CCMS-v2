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

Read the COMPLETE document carefully before answering.

Your job is ONLY to extract:

1. Acts explicitly mentioned in the document.
2. Mapping between those Acts and the already-extracted Sections provided below.

Do NOT extract Constitutional provisions, Articles, Rules, Regulations, Notifications,
Government Orders, Circulars, Ordinances, case numbers, survey numbers, dates, people,
places, or any other legal references.

=========================================================
ALREADY EXTRACTED SECTIONS
=========================================================

Use ONLY the Sections listed below.

Do not create, modify, guess, rewrite, expand, or infer any Section numbers.

{sections_text}

=========================================================
WHAT TO EXTRACT
=========================================================

Extract only Acts.

An Act is valid ONLY if its name explicitly contains the word "Act".

Examples of valid Acts:

- Karnataka Land Reforms Act, 1961
- Karnataka Land Revenue Act, 1964
- Transfer of Property Act, 1882

Preserve the Act name exactly as written in the document.

Rules for Acts:

- Include the year if it is present.
- If no year is present, return only the Act name.
- Return each unique Act only once.
- Preserve the order of first appearance.
- Do not normalize, rename, shorten, or expand Act names.

=========================================================
STRICT EXCLUSIONS
=========================================================

Never extract or map any of the following:

- Constitution of India
- Articles of the Constitution
- Article 14
- Article 19
- Article 21
- Article 226
- Article 227
- Constitutional provisions
- Writ jurisdiction references
- Rules
- Regulations
- Notifications
- Government Orders
- Circulars
- Ordinances
- Bye-laws
- Case numbers
- Survey numbers
- Dates
- Names of judges, advocates, parties, people, or places

Important:

- "Constitution of India" is NOT an Act.
- "Article" is NOT a Section.
- Do not include Article numbers in the output.
- Do not map Articles to any Act.
- Do not extract Rules even if they look similar to Sections.
- Only map the provided Sections to Acts when the document explicitly says they belong to that Act.

=========================================================
ACT TO SECTION MAPPING RULES
=========================================================

The Sections listed above are the ONLY Sections you may use.

A Section belongs to an Act ONLY when the document explicitly associates that Section
with that Act.

Valid explicit associations include:

- Section 19 of the Karnataka Land Revenue Act, 1964.
- Sections 79A, 79B and 80 of the Karnataka Land Reforms Act, 1961.
- Section 136(2) under the Karnataka Land Revenue Act, 1964.
- Section 4 read with Section 5 of the Transfer of Property Act, 1882.

Invalid associations:

- A Section appears in one paragraph and an Act appears elsewhere.
- An Act appears nearby but is not grammatically linked to the Section.
- The relationship is based on legal knowledge instead of document text.
- The Section belongs to the Constitution, an Article, a Rule, or a Regulation.
- The document merely discusses an Act and Section separately.

Do NOT:

- Create new Sections.
- Modify Section numbers.
- Guess missing Sections.
- Infer relationships from legal knowledge.
- Map the same Section to multiple Acts unless the document explicitly does so.
- Map a Section if the Act is unclear.
- Include Sections that are not present in the provided extracted Sections list.

If an Act is mentioned but no provided Section is explicitly mapped to it,
include the Act in "acts" but do not add unmapped Sections.

If no Act-to-Section mapping exists, return an empty list for "act_section_mapping".

=========================================================
OUTPUT RULES
=========================================================

Return ONLY valid JSON.

Do not include explanations, notes, markdown, comments, or extra text.

The JSON must have exactly these keys:

{{
    "acts": [],
    "act_section_mapping": [
        {{
            "act": "",
            "sections": []
        }}
    ]
}}

Rules for JSON:

- "acts" must contain only Act names.
- "act_section_mapping" must contain only Acts with explicitly mapped Sections.
- Every mapped Act must also appear in "acts".
- Every mapped Section must come from the provided Sections list.
- Remove duplicate Acts.
- Remove duplicate Sections within each Act.
- Preserve order of first appearance.

=========================================================
EXAMPLES
=========================================================

Example 1

Sections:

79A
79B
80

Document:

Sections 79A, 79B and 80 of the Karnataka Land Reforms Act, 1961.

Output:

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

Example 2

Sections:

19
136(2)

Document:

Section 19 of the Karnataka Land Reforms Act, 1961.

Section 136(2) of the Karnataka Land Revenue Act, 1964.

The petition was filed under Articles 226 and 227 of the Constitution of India.

Output:

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

Example 3

Sections:

226
227

Document:

The writ petition is filed under Articles 226 and 227 of the Constitution of India.

Output:

{{
    "acts": [],
    "act_section_mapping": []
}}

---------------------------------------------------------

Example 4

Sections:

19
136(2)

Document:

Section 19 was referred.

The Karnataka Land Revenue Act, 1964 was also mentioned later in the judgment.

Output:

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