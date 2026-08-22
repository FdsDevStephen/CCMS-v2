from __future__ import annotations

import json
import re

from extractor.llm.factory import get_llm_client
from RAG.hybrid_retreiver import HybridRetriever


class ActExtractor:
    """Extract Act names and map them only to already-extracted sections."""

    def __init__(
        self,
        chunks: list[dict],
        retriever: HybridRetriever | None = None,
    ):
        self.retriever = retriever or HybridRetriever()
        self.retriever.build_bm25(chunks)
        self.llm_client = get_llm_client()

    def extract(
        self,
        document: str,
        sections: list,
        top_k: int = 5,
    ) -> dict:
        section_numbers = self._normalize_sections(sections)

        if not section_numbers:
            return self._empty_result()

        query = (
            "Indian legal Act section provision "
            "u/s under section of Act "
            + " ".join(section_numbers)
        )

        results = self.retriever.search(
            query=query,
            top_k=top_k,
            document=document,
            candidate_k=max(top_k * 3, 15),
        )

        if not results:
            return self._empty_result()

        context_parts: list[str] = []
        seen_texts: set[str] = set()

        for result in results:
            text = str(result.payload.get("text", "")).strip()

            if text and text not in seen_texts:
                context_parts.append(text)
                seen_texts.add(text)

        if not context_parts:
            return self._empty_result()

        context = "\n\n--- CHUNK ---\n\n".join(context_parts)

        prompt = self._build_prompt(section_numbers, context)

        response = self.llm_client.generate(prompt)
        result = self._parse_json(response)

        return self._validate_result(result, section_numbers)

    @staticmethod
    def _normalize_sections(sections: list) -> list[str]:
        normalized: list[str] = []

        for section in sections or []:
            if isinstance(section, dict):
                number = section.get("number", "")
            else:
                number = section

            number = str(number).strip()

            if number and number not in normalized:
                normalized.append(number)

        return normalized

    @staticmethod
    def _empty_result() -> dict:
        return {
            "acts": [],
            "act_section_mapping": [],
        }

    @staticmethod
    def _build_prompt(section_numbers: list[str], context: str) -> str:
        return f"""
You are extracting legal information from OCR text.

TASK:
Identify Act names that are EXPLICITLY connected in the supplied text
to the EXISTING section numbers below.

Existing section numbers:
{json.dumps(section_numbers, ensure_ascii=False)}

STRICT RULES:
1. Extract ONLY Act names explicitly present in the supplied text.
2. Never infer an Act from a section number.
3. Never create or add a section number.
4. Use ONLY the supplied text.
5. Preserve the Act name exactly as written, except remove surrounding whitespace.
6. A mapping is allowed only when the supplied text explicitly connects
   that Act with one or more existing section numbers.
7. A section can be mapped only if it appears in the existing section list.
8. Do not extract Articles, Rules, Regulations, Notifications, Government
   Orders, Constitution references, case numbers, dates, persons or places
   as Acts.
9. Remove duplicate Acts.
10. Remove duplicate section numbers inside each mapping.
11. Return ONLY valid JSON. No markdown and no explanation.
12. If nothing is explicitly supported, return empty arrays.

OUTPUT FORMAT:
{{
  "acts": [
    {{"name": "Exact Act Name"}}
  ],
  "act_section_mapping": [
    {{
      "act": "Exact Act Name",
      "sections": ["Existing Section"]
    }}
  ]
}}

SUPPLIED TEXT:
{context}
""".strip()

    @staticmethod
    def _parse_json(response: str) -> dict:
        if not response:
            return {}

        text = response.strip()

        # Remove accidental markdown fences.
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Recover the first JSON object if the local model added text.
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if not match:
                return {}

            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return {}

    @staticmethod
    def _validate_result(
        result: dict,
        valid_sections: list[str],
    ) -> dict:
        valid_section_set = set(valid_sections)

        raw_acts = result.get("acts", [])
        if not isinstance(raw_acts, list):
            raw_acts = []

        acts: list[dict] = []
        seen_acts: set[str] = set()

        for item in raw_acts:
            if not isinstance(item, dict):
                continue

            name = str(item.get("name", "")).strip()

            if not name or name in seen_acts:
                continue

            seen_acts.add(name)
            acts.append({"name": name})

        raw_mappings = result.get("act_section_mapping", [])
        if not isinstance(raw_mappings, list):
            raw_mappings = []

        cleaned_mappings: list[dict] = []
        seen_mapping_keys: set[tuple] = set()

        for item in raw_mappings:
            if not isinstance(item, dict):
                continue

            act = str(item.get("act", "")).strip()
            sections = item.get("sections", [])

            if not act or not isinstance(sections, list):
                continue

            cleaned_sections: list[str] = []

            for section in sections:
                section = str(section).strip()

                if section in valid_section_set and section not in cleaned_sections:
                    cleaned_sections.append(section)

            if not cleaned_sections:
                continue

            # Mapping Act must also be one of the extracted Acts.
            if act not in seen_acts:
                continue

            key = (act, tuple(cleaned_sections))
            if key in seen_mapping_keys:
                continue

            seen_mapping_keys.add(key)
            cleaned_mappings.append(
                {
                    "act": act,
                    "sections": cleaned_sections,
                }
            )

        return {
            "acts": acts,
            "act_section_mapping": cleaned_mappings,
        }
