"""
Standardise place-name spellings.

- Taluk / district: matched against the Karnataka gazetteer (official names,
  old names such as Mysore -> Mysuru, and close OCR variants such as
  Baltari -> Ballari).
- Village / hobli (and taluks/districts not in the gazetteer): there is no
  master list, so near-identical spellings within the same document are
  merged into the most frequent one (Lalithadrapura -> Lalithadripura).
"""

from __future__ import annotations

import re
from collections import Counter
from difflib import SequenceMatcher, get_close_matches

from extractor.karnataka_gazetteer import (
    DISTRICT_ALIASES,
    DISTRICT_OF_TALUK,
    DISTRICTS,
    TALUK_ALIASES,
    TALUKS,
)

# Fuzzy thresholds (SequenceMatcher ratio on lower-case letters only).
GAZETTEER_CUTOFF = 0.85
DOCUMENT_CUTOFF = 0.88
MIN_FUZZY_LENGTH = 5


def _key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def _build_index(names: list[str], aliases: dict[str, str]) -> dict[str, str]:
    index = {_key(name): name for name in names}
    index.update({_key(alias): name for alias, name in aliases.items()})
    return index


_INDEX = {
    "taluk": _build_index(TALUKS, TALUK_ALIASES),
    "district": _build_index(DISTRICTS, DISTRICT_ALIASES),
}


def tidy(name: str) -> str:
    """Collapse spaces and fix ALL-CAPS OCR ("NANDUR-K" -> "Nandur-K")."""
    name = " ".join(name.split())

    if name.isupper() and len(name) > 3:
        name = name.title()

    return name


def gazetteer_name(kind: str, name: str) -> str | None:
    """Official name for a taluk/district, or None if not recognised."""
    index = _INDEX.get(kind)

    if not index:
        return None

    key = _key(name)

    if key in index:
        return index[key]

    if len(key) < MIN_FUZZY_LENGTH:
        return None

    match = get_close_matches(key, index, n=1, cutoff=GAZETTEER_CUTOFF)

    return index[match[0]] if match else None


def district_of_taluk(taluk: str | None) -> str | None:
    return DISTRICT_OF_TALUK.get(taluk) if taluk else None


def _similar(a: str, b: str) -> bool:
    a, b = _key(a), _key(b)

    if min(len(a), len(b)) < MIN_FUZZY_LENGTH:
        return a == b

    # One OCR-substituted letter ("Kuragsd" / "Kuragod").
    if len(a) == len(b) >= 6 and sum(x != y for x, y in zip(a, b)) == 1:
        return True

    return SequenceMatcher(None, a, b).ratio() >= DOCUMENT_CUTOFF


class PlaceNameStandardizer:
    """
    Standardiser for one document.

    Built from every place name found in the document, so spelling variants
    of the same village are merged into the most frequent spelling.
    """

    def __init__(self, names_by_kind: dict[str, list[str]]):
        self._canonical: dict[str, dict[str, str]] = {}

        for kind, names in names_by_kind.items():
            counts = Counter(self._base(kind, name) for name in names)
            representatives: list[str] = []
            mapping: dict[str, str] = {}

            for name, _ in counts.most_common():
                target = next(
                    (r for r in representatives if _similar(name, r)),
                    None,
                )

                if target is None:
                    representatives.append(name)
                    target = name

                mapping[_key(name)] = target

            self._canonical[kind] = mapping

    @staticmethod
    def _base(kind: str, name: str) -> str:
        name = tidy(name)
        return gazetteer_name(kind, name) or name

    def __call__(self, kind: str, name: str | None) -> str | None:
        if not name:
            return name

        name = self._base(kind, name)
        mapping = self._canonical.get(kind, {})

        if _key(name) in mapping:
            return mapping[_key(name)]

        # Names not seen while building (e.g. from the LLM): snap to a
        # close spelling already used in the document.
        return next(
            (r for r in set(mapping.values()) if _similar(name, r)),
            name,
        )
