def build_location_prompt(contexts: list[dict]) -> str:
    prompt = f"""
You are an expert legal information extraction system.

Your task is to extract the administrative location for EVERY survey number.

There are exactly {len(contexts)} survey numbers.

You MUST return exactly {len(contexts)} JSON objects.

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

1. Return ONE object for EVERY survey number.
2. Do NOT skip any survey number.
3. Use ONLY the supplied context.
4. Do NOT infer missing values.
5. Preserve the survey number exactly.
6. Preserve the original spelling of locations.
7. Ignore petitioner, respondent, advocate, court and police station addresses.
8. If a field is unavailable, return null.
9. Return ONLY a JSON array.
10. Do NOT wrap the JSON inside markdown.
11. Do NOT explain anything.

Survey Contexts:

"""

    for index, item in enumerate(contexts, start=1):
        prompt += f"""

======================================================

Survey #{index}

Survey Number:
{item['survey_number']}

Context:
{item['context']}

Administrative abbreviations:

Village = Village
Hobli = Hobli
Tq = Taluk
Tal = Taluk
Tk = Taluk
Taluka = Taluk
Dist = District
Dt = District

If the text contains "Tq & Dist: Kalaburagi", then BOTH

"taluk": "Kalaburagi"
"district": "Kalaburagi"

must be returned.

"""

    return prompt