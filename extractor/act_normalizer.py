
from __future__ import annotations

import json
from pathlib import Path

from rapidfuzz import fuzz


class ActNormalizer:

    def __init__(
        self,
        acts_file: str | Path = "data/acts.json",
    ):

        acts_file = Path(acts_file)

        with open(
            acts_file,
            "r",
            encoding="utf-8",
        ) as f:

            self.acts = json.load(f)

    # Normalize

    def normalize(
        self,
        act: str,
        threshold: int = 90,
    ) -> str:

        if not act:
            return ""

        act = " ".join(act.split())


        # Exact Canonical Match


        for item in self.acts:

            if act.lower() == item["name"].lower():

                return item["name"]
            
        # Exact Alias Match

        for item in self.acts:

            for alias in item.get("aliases", []):

                if act.lower() == alias.lower():

                    return item["name"]

        # Fuzzy Canonical Match
        best_name = act
        best_score = 0

        for item in self.acts:

            score = fuzz.ratio(
                act.lower(),
                item["name"].lower(),
            )

            if score > best_score:

                best_score = score
                best_name = item["name"]

        # Fuzzy Alias Match


        for item in self.acts:

            for alias in item.get("aliases", []):

                score = fuzz.ratio(
                    act.lower(),
                    alias.lower(),
                )

                if score > best_score:

                    best_score = score
                    best_name = item["name"]

        # Return Best Match

        if best_score >= threshold:

            return best_name

        return act