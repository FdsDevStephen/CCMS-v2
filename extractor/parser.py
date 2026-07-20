"""
Parser for LLM JSON responses.
"""

from __future__ import annotations

import json
import re
from typing import Any


class LLMResponseParser:
    """
    Parses and repairs JSON returned by the LLM.
    """

    @staticmethod
    def parse(response: str) -> dict[str, Any]:
        """
        Parse the LLM response into a Python dictionary.
        """

        try:
            data = json.loads(response)

            return {
                "acts": data.get("acts", []),
                "act_section_mapping": data.get("act_section_mapping", []),
                "sections": data.get("sections", []),
                "primary_act": data.get("primary_act"),
            }

        except json.JSONDecodeError:
            pass


        match = re.search(r"\{.*\}", response, re.DOTALL)

        if match:

            try:

                data = json.loads(match.group(0))

                return {
                    "acts": data.get("acts", []),
                    "act_section_mapping": data.get("act_section_mapping", []),
                    "sections": data.get("sections", []),
                    "primary_act": data.get("primary_act"),
                }

            except json.JSONDecodeError:
                pass

        cleaned = response.replace("```json", "").replace("```", "").strip()

        try:

            data = json.loads(cleaned)

            return {
                "acts": data.get("acts", []),
                "act_section_mapping": data.get("act_section_mapping", []),
                "sections": data.get("sections", []),
                "primary_act": data.get("primary_act"),
            }

        except json.JSONDecodeError:

            raise ValueError("Unable to parse LLM response.")

    @staticmethod
    def empty_response() -> dict[str, Any]:
        """
        Default response.
        """

        return {
            "acts": [],
            "act_section_mapping": [],
            "sections": [],
            "primary_act": None,
        }