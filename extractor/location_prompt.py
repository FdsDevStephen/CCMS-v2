from __future__ import annotations


def build_location_prompt(survey_number: str, contexts: list[str]) -> str:
    """
    Compact prompt for a single survey number.

    Only used as a fallback when the rule-based extractor finds nothing, so
    it is kept short enough for small local models and their default
    context window.
    """

    context_text = "\n\n---\n\n".join(
        context.strip() for context in contexts
    )

    return f"""Extract the location of the land bearing survey number {survey_number} from the Indian legal text below.

Rules:
- Use only the text below. Never guess or use outside knowledge.
- Use a location only if the text links it to survey number {survey_number} or to the land it identifies.
- Ignore addresses of petitioners, respondents, advocates and government offices.
- Return only the name, without words like Village, Hobli, Taluk, Taluka, Tq, District, Dist.
- Do not include people's names, numbers, dates or OCR noise.
- Use null for anything not explicitly stated.

Example:
Text: "land bearing Sy.No.76 in Lalithadripura Village, Varuna Hobli, Mysore Taluk"
Output: {{"village": "Lalithadripura", "hobli": "Varuna", "taluk": "Mysore", "district": null}}

Text:
{context_text}

Return exactly one JSON object with the keys "village", "hobli", "taluk", "district".
"""
