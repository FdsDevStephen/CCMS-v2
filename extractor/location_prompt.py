def build_location_prompt(contexts: list[dict]) -> str:
    prompt = f"""
You are an expert legal information extraction system.

Extract the administrative location for EVERY survey number.

There are exactly {len(contexts)} survey numbers.
Return exactly {len(contexts)} JSON objects.

Output format:

[
    {{
        "survey_number": "",
        "village": null,
        "hobli": null,
        "taluk": null,
        "district": null
    }}
]

Rules:

1. Return ONE JSON object for EVERY survey number.
2. Never skip a survey number.
3. Use ONLY the supplied context.
4. Do NOT infer or hallucinate information.
5. Preserve the survey number exactly.
6. Preserve the spelling of place names exactly.
7. Return ONLY place names.
8. Do NOT include the words Village, Hobli, Taluk, Taluka, Tq, Tk, District or Dist in the extracted values.
9. Ignore petitioner, respondent, advocate, court, police station and postal addresses.
10. If a field is missing, return null.
11. Return ONLY valid JSON.
12. Do NOT use markdown.
13. Do NOT explain anything.

Never return administrative labels as extracted values.

The values for village, hobli, taluk and district must always be place names.

Words such as Village, Hobli, Taluk, Taluka, Tatuk, Taiuk, District, Dist, Dt, Tq and Tk are labels, not values, and must never be returned.

Administrative hierarchy:

Village → Hobli → Taluk → District

Administrative abbreviations:

Village = Village
Hobli = Hobli
Taluk = Taluk
Taluka = Taluk
Tal = Taluk
Tq = Taluk
Tk = Taluk
District = District
Dist = District
Dt = District

Special Rules

If the document contains

"Tq & Dist : Kalaburagi"

then return

"taluk": "Kalaburagi"
"district": "Kalaburagi"

If the document contains

"Taluk & District : Hassan"

then return

"taluk": "Hassan"
"district": "Hassan"

==================================================
LOCATION VERIFICATION
==================================================

Before returning any location, verify that it belongs to the SAME administrative hierarchy as the survey number.

Do NOT collect Village, Hobli, Taluk and District independently from different parts of the document.

Treat the administrative hierarchy as one connected chain.

Village
↓
Hobli
↓
Taluk
↓
District

A higher-level location (Taluk or District) may be returned ONLY if it belongs to the same Village/Hobli chain mentioned with the survey number.

Ignore locations that belong to:

- petitioner addresses
- respondent addresses
- advocate addresses
- court addresses
- government office addresses
- police station addresses
- postal addresses
- any unrelated property
- any unrelated survey number

If multiple districts appear in the supplied context, return only the district that is administratively connected to the extracted Village/Hobli/Taluk.

Never choose a district simply because it appears later in the text.

If the relationship between the survey number and a location is uncertain, return null.

Prefer correctness over completeness.

Examples

Example 1

Context

Sy.No.173/2 of village Nandur-K,
Tq & Dist : Kalaburagi

Output

{{
    "survey_number":"173/2",
    "village":"Nandur-K",
    "hobli":null,
    "taluk":"Kalaburagi",
    "district":"Kalaburagi"
}}

Example 2

Context

Sy.No.2 measuring 23 Acres 34 Guntas situated at
Kenchammanahalli village of
Anegodu Hobli of
Davangere District

Output

{{
    "survey_number":"2",
    "village":"Kenchammanahalli",
    "hobli":"Anegodu",
    "taluk":null,
    "district":"Davangere"
}}

Example 3

Context

Survey No.54 situated at
Hosahalli Village,
Kasaba Hobli,
Tumakuru Taluk,
Tumakuru District

Output

{{
    "survey_number":"54",
    "village":"Hosahalli",
    "hobli":"Kasaba",
    "taluk":"Tumakuru",
    "district":"Tumakuru"
}}

Survey Contexts

"""

    for i, item in enumerate(contexts, start=1):

        prompt += f"""

-------------------------------------------------------

Survey #{i}

Survey Number

{item["survey_number"]}

Context

{item["context"]}

"""

    prompt += """

Return ONLY the JSON array.

"""

    return prompt