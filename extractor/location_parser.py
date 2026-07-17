"""
Parser for Survey Location JSON responses.
"""

from __future__ import annotations

import json
import re
from typing import Any


class LocationResponseParser:

    @staticmethod
    def parse(response: str) -> dict[str, Any]:

        # -----------------------
        # Direct JSON
        # -----------------------

        try:

            data = json.loads(response)

            return {
                "village": data.get("village"),
                "hobli": data.get("hobli"),
                "taluk": data.get("taluk"),
                "district": data.get("district"),
            }

        except json.JSONDecodeError:
            pass

        # -----------------------
        # Extract JSON
        # -----------------------

        match = re.search(
            r"\{.*\}",
            response,
            re.DOTALL,
        )

        if match:

            try:

                data = json.loads(match.group(0))

                return {
                    "village": data.get("village"),
                    "hobli": data.get("hobli"),
                    "taluk": data.get("taluk"),
                    "district": data.get("district"),
                }

            except json.JSONDecodeError:
                pass

        # -----------------------
        # Remove Markdown
        # -----------------------

        cleaned = (
            response
            .replace("```json", "")
            .replace("```", "")
            .strip()
        )

        try:

            data = json.loads(cleaned)

            return {
                "village": data.get("village"),
                "hobli": data.get("hobli"),
                "taluk": data.get("taluk"),
                "district": data.get("district"),
            }

        except json.JSONDecodeError:

            raise ValueError(
                "Unable to parse location response."
            )

    @staticmethod
    def empty():

        return {
            "village": None,
            "hobli": None,
            "taluk": None,
            "district": None,
        }