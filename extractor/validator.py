from __future__ import annotations

from typing import Any
from unittest import result

from extractor.normalizer import Normalizer


class Validator:
    """
    Validates and normalizes the final extraction result.
    """

    REQUIRED_KEYS = {
        "case_number": None,
        "survey_numbers": [],
        "acts": [],
        "sections": [],
        "act_section_mapping": [],
        "primary_act": None,
    }

    @classmethod
    def validate(cls, data: dict[str, Any]) -> dict[str, Any]:
        """
        Validate and normalize the final output.

        Args:
            data: Combined extraction output.

        Returns:
            Validated JSON.
        """

        result = cls.REQUIRED_KEYS.copy()

        result.update(data)

        # Ensure correct types
        if not isinstance(result["survey_numbers"], list):
            result["survey_numbers"] = []

        if not isinstance(result["acts"], list):
            result["acts"] = []

        if not isinstance(result["sections"], list):
            result["sections"] = []

        if not isinstance(result["act_section_mapping"], list):
            result["act_section_mapping"] = []

        # Normalize values
        result["case_number"] = Normalizer.normalize_case_number(result["case_number"])

        result["survey_numbers"] = Normalizer.normalize_survey_numbers(
            result["survey_numbers"]
        )

        result["sections"] = Normalizer.normalize_sections(result["sections"])

        result["acts"] = Normalizer.normalize_acts(result["acts"])

        result["act_section_mapping"] = Normalizer.normalize_act_section_mapping(
            result["act_section_mapping"]
        )

        # Validate primary_act
        if result["primary_act"]:

            result["primary_act"] = " ".join(result["primary_act"].split())

            if result["primary_act"] not in result["acts"]:
                result["acts"].append(result["primary_act"])

        return result
