from __future__ import annotations


def build_location_prompt(contexts: list[dict]) -> str:

    survey_numbers = list(
        dict.fromkeys(
            context["survey_number"]
            for context in contexts
        )
    )

    context_blocks = []

    for index, context in enumerate(contexts, start=1):

        context_blocks.append(
            f"""
================ CONTEXT {index} ================

SURVEY NUMBER:
{context["survey_number"]}

TEXT:
{context["context"]}

===================================================
"""
        )

    combined_context = "\n".join(context_blocks)

    return f"""
You are a HIGH-PRECISION Indian legal document information extraction engine.

Your ONLY task is to extract land location information explicitly associated
with the supplied survey numbers.

IMPORTANT:

NULL IS CORRECT.

NOISE IS WRONG.

GUESSING IS FORBIDDEN.

INFERENCE IS FORBIDDEN.


===================================================
SURVEY NUMBERS
===================================================

{", ".join(survey_numbers)}


===================================================
ABSOLUTE RULES
===================================================

1. Return EXACTLY ONE JSON object for every unique survey number.

2. Every survey number must appear EXACTLY ONCE.

3. Use ONLY the supplied context.

4. NEVER use outside knowledge.

5. NEVER infer a location.

6. NEVER guess a location.

7. NEVER reconstruct a location from OCR fragments.

8. If there is ANY uncertainty, return null.

9. FALSE POSITIVE IS WORSE THAN NULL.

10. A location must be explicitly associated with the survey number
    or with the land/property identified by that survey number.

11. Do NOT use a location merely because it appears somewhere nearby.

12. Do NOT copy a location from another survey number.

13. Do NOT combine unrelated text.

14. Do NOT use geographical knowledge to complete missing information.


===================================================
FIELDS
===================================================

Return ONLY these fields:

- village
- hobli
- taluk
- district


===================================================
LOCATION VALUE RULES
===================================================

Return ONLY the actual location name.

Do NOT include:

Village
Hobli
Taluk
Taluka
District
Dist
Tq
Tq & Dist
Tq & District


Example:

"Lalithadripura Village"

MUST become:

"village": "Lalithadripura"


Example:

"Varuna Hobli"

MUST become:

"hobli": "Varuna"


Example:

"Mysore Taluk"

MUST become:

"taluk": "Mysore"


Example:

"Mysore District"

MUST become:

"district": "Mysore"


===================================================
CRITICAL OCR NOISE RULE
===================================================

STOP the location value when the administrative location name ends.

NEVER continue reading into the next OCR text.

Example:

Mysore Taluk
Nityanand Naik
| ; 20

CORRECT:

"taluk": "Mysore"

WRONG:

"taluk": "Mysore Nityanand"

WRONG:

"taluk": "Mysore Nityanand Naik"

WRONG:

"taluk": "Mysore Nityanand Naik | ; 20"


The following are NEVER part of a location:

- people's names
- signatures
- page numbers
- OCR fragments
- random symbols
- respondent numbers
- dates
- annexure numbers
- headers
- footers


===================================================
PERSON NAME PROTECTION
===================================================

A person's name MUST NEVER be included in:

- village
- hobli
- taluk
- district

For example:

"Mysore Taluk
Nityanand Naik"

means:

"taluk": "Mysore"

NOT:

"taluk": "Mysore Nityanand"

NOT:

"taluk": "Mysore Nityanand Naik"


===================================================
VILLAGE
===================================================

Extract ONLY the village name.

Examples:

"Lalithadripura Village"
→ "Lalithadripura"

"village Nandur-K"
→ "Nandur-K"

"Village Nandur-K"
→ "Nandur-K"


===================================================
HOBLI
===================================================

Extract ONLY the Hobli name.

Examples:

"Varuna Hobli"
→ "Varuna"

"Anegodu Hobli"
→ "Anegodu"

"in Anegod Hobli"
→ "Anegod"


===================================================
TALUK
===================================================

Extract ONLY the Taluk name.

Examples:

"Mysore Taluk"
→ "Mysore"

"Davangere Taluka"
→ "Davangere"


===================================================
DISTRICT
===================================================

Extract ONLY the District name.

Examples:

"Mysore District"
→ "Mysore"

"District: Kalaburagi"
→ "Kalaburagi"


===================================================
TALUK AND DISTRICT
===================================================

DO NOT assume that a Taluk is a District.

Example:

"Mysore Taluk"

Return:

"taluk": "Mysore"
"district": null


Do NOT automatically return:

"district": "Mysore"


Example:

"Mysore District"

Return:

"district": "Mysore"
"taluk": null


Only return both when both are explicitly stated.


===================================================
TQ & DIST
===================================================

If the text explicitly says:

"Tq & Dist: Kalaburagi"

and this statement is explicitly associated with the survey number,
then return:

"taluk": "Kalaburagi"
"district": "Kalaburagi"


If the text only says:

"District: Kalaburagi"

return:

"district": "Kalaburagi"

and:

"taluk": null


Do NOT infer missing administrative levels.


===================================================
SURVEY NUMBER ASSOCIATION
===================================================

A location is valid only when the text explicitly connects it to the
survey number.

VALID:

"Sy.No.171/1 of Village Nandur-K"

VALID:

"land bearing Sy.No.171/1 situated at Village Nandur-K"

VALID:

"Sy.No.76 in Lalithadripura Village, Varuna Hobli, Mysore Taluk"


INVALID:

"Sy.No.76 ..."

followed later by:

"Lalithadripura Village"

unless the text explicitly connects them.


===================================================
NO CROSS-SURVEY COPYING
===================================================

Never copy a location from another survey number.

If:

Sy.No.171/1 → Nandur-K

and:

Sy.No.172/3 → no explicit location

then:

Sy.No.172/3 MUST remain null.

Do not assume both properties have the same location.


===================================================
NO CROSS-CONTEXT MIXING
===================================================

Do NOT construct one location from unrelated pieces of text.

For example:

Context 1:
Village = Nandur-K

Context 2:
Taluk = Kalaburagi

Context 3:
District = Gulbarga

Do NOT combine them unless the text explicitly establishes
that they belong to the same survey-number land.


===================================================
IGNORE
===================================================

Never extract:

- case numbers
- W.P. numbers
- O.S. numbers
- R.A. numbers
- C.C.C. numbers
- years
- dates
- acres
- guntas
- road names
- building names
- court addresses
- petitioner addresses
- respondent addresses
- advocate addresses
- government office addresses
- police station addresses
- Act names
- Sections
- Articles
- people's names


===================================================
EXAMPLES
===================================================

Example 1:

TEXT:

"Sy.No.171/1 ... Village Nandur-K, Tq & District: Kalaburagi."

OUTPUT:

{{
    "survey_number": "171/1",
    "village": "Nandur-K",
    "hobli": null,
    "taluk": "Kalaburagi",
    "district": "Kalaburagi"
}}


Example 2:

TEXT:

"Sy.No.172/3 of village Nandur-K"

OUTPUT:

{{
    "survey_number": "172/3",
    "village": "Nandur-K",
    "hobli": null,
    "taluk": null,
    "district": null
}}


Example 3:

TEXT:

"Sy.No.76 in Lalithadripura Village, Varuna Hobli, Mysore Taluk."

OUTPUT:

{{
    "survey_number": "76",
    "village": "Lalithadripura",
    "hobli": "Varuna",
    "taluk": "Mysore",
    "district": null
}}


Example 4:

TEXT:

"Sy.No.76 in Lalithadripura Village, Varuna Hobli,
Mysore Taluk, Mysore District."

OUTPUT:

{{
    "survey_number": "76",
    "village": "Lalithadripura",
    "hobli": "Varuna",
    "taluk": "Mysore",
    "district": "Mysore"
}}


Example 5:

TEXT:

"Mysore Taluk
Nityanand Naik
| ; 20"

OUTPUT:

{{
    "survey_number": "76",
    "village": null,
    "hobli": null,
    "taluk": "Mysore",
    "district": null
}}


===================================================
FINAL VALIDATION
===================================================

Before returning every non-null field, verify:

1. Is it explicitly present?

2. Is it explicitly associated with this survey number?

3. Is it actually a location?

4. Is the administrative type correct?

5. Does the value contain only the location name?

6. Did I accidentally include Village/Hobli/Taluk/District?

7. Did I accidentally include a person's name?

8. Did I accidentally include OCR noise?

9. Did I accidentally include a page number?

10. Did I infer anything?

11. Did I copy information from another survey?

12. Did I confuse Taluk and District?

13. Did I combine unrelated text?

If ANY answer is uncertain:

RETURN NULL.


===================================================
OUTPUT FORMAT
===================================================

Return ONLY valid JSON.

No markdown.

No explanation.

No comments.

No extra keys.

There must be exactly {len(survey_numbers)} objects.

Every survey number must appear exactly once.

[
  {{
    "survey_number": "76",
    "village": null,
    "hobli": null,
    "taluk": null,
    "district": null
  }}
]


===================================================
LEGAL CONTEXT
===================================================

{combined_context}


===================================================

REMEMBER:

NULL IS CORRECT.

NOISE IS WRONG.

GUESSING IS FORBIDDEN.

INFERENCE IS FORBIDDEN.
"""