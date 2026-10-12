"""
Survey Number -> Location (village / hobli / taluk / district) extraction.

Strategy (cheapest and most reliable first):

1. Rule-based: find every administrative "location chain" in the text
   (e.g. "Rayasandra Village, Kasaba Hobli, Kanakapura Taluk") and link each
   survey-number mention to the nearest chain after it (or right before it).
   Every number in a list ("Sy.Nos.4 and 6 of ...") gets the same chain.
   Chains that are party addresses ("R/at ...", "... - 573 201") are ignored
   for linking.
2. Fill gaps from facts stated elsewhere in the same document:
   - a sub-division inherits from its parent (33/4 -> 33),
   - village -> hobli/taluk/district, hobli -> taluk, taluk -> district.
3. LLM fallback, only for survey numbers that still have no location. Answers
   are accepted only if the name appears in the supplied context.
4. Last resort: copy from the nearest located survey mention, else the
   document's dominant location.

Names are standardised (extractor.location_standardizer): taluks/districts
against the Karnataka gazetteer, villages/hoblis to the most frequent
spelling in the document. A missing district is filled from the taluk via
the gazetteer.

Every input survey number always gets exactly one result row.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from extractor.location_parser import LocationResponseParser
from extractor.location_prompt import build_location_prompt
from extractor.location_standardizer import (
    PlaceNameStandardizer,
    district_of_taluk,
)
from extractor.normalizer import Normalizer, clean_location_name
from extractor.survey_extractor import SurveyExtractor

logger = logging.getLogger(__name__)

FIELDS = ("village", "hobli", "taluk", "district")
RANK = {name: index for index, name in enumerate(FIELDS)}


# ==========================================================
# PATTERNS
# ==========================================================

# A place-name token: capitalised word, may contain "-" or "." joins
# ("Nandur-K", "K.Hoskote"), or initials ("C." in "C. Kodigehalli",
# "T.N." in "T.N. Kote"). Case-sensitive, word-bounded on both sides.
_TOKEN = r"(?:[A-Z](?:\.[A-Z])*\.|[A-Z][A-Za-z]*(?:[.\-][A-Za-z]+)*)"
_NAME = (
    rf"(?-i:(?<![A-Za-z]){_TOKEN}(?:[ \t]+{_TOKEN}){{0,3}}(?![A-Za-z]))"
)
_GAP = r"[\s:\-\"'‘’“”]*"  # separators between a label and a name

# Up to two OCR line breaks between a name and its label
# ("of Attikatti\n\nVillage", "Beejagatta\nvillage").
_BREAK = r"(?:[ \t]*,?[ \t]*\n){0,2}[ \t]*"

_LABELS = {
    "village": r"Village|Vill\.|Grama",
    "hobli": r"Hob\s?li|Hobii",
    "taluk": r"Taluka|Taluku|Taluk|Tq\.?",
    "district": r"District|Distriet|Dist\.?|Dt\.",
}

_ANY_LABEL = "|".join(_LABELS.values())

# Words that follow a label when it is not a location
# ("Village Accountant", "District Court", "Taluk Surveyor").
_ROLE_WORDS = (
    r"Accountant|Panchayat\w*|Surveyor|Office\w*|Court|Level|Map|Chavadi|"
    r"Judge|Commissioner|Registrar|Collector|Magistrate|Board|Executive|"
    r"Sessions|Forest|Records|Planning|Head|Extract|Limits?|Committee|"
    r"Administration|Authority|Development|Hospital|Police|Treasury|"
    r"Industries|Health|Education|Central|Bank|Co-?operative|Union"
)

_NOT_ROLE = rf"(?![ \t]*[,:]?[ \t]*(?:{_ROLE_WORDS})\b)"

# "Mysore Taluk and District", "Hasana Taluku and Dist."
_COMBINED_NAME_FIRST = re.compile(
    rf"(?P<name>{_NAME})[ \t]*,?[ \t]*\b(?:Taluka|Taluku|Taluk|Tq)\b\.?"
    rf"[\s\"'.]*(?:&|and)\s*[\"']?\s*(?:District|Dist)\b\.?",
    re.IGNORECASE,
)

# "Tq & Dist: Ballari", "Taluka and District Kalaburagi"
_COMBINED_LABEL_FIRST = re.compile(
    rf"\b(?:Taluka|Taluku|Taluk|Tq)\b\.?\s*(?:&|and)\s*"
    rf"(?:District|Dist)\b\.?{_GAP}(?P<name>{_NAME})",
    re.IGNORECASE,
)

# "Rayasandra Village", "Jamakhandi taluk"
_NAME_FIRST = {
    kind: re.compile(
        # A label followed by ":" introduces the next name
        # ("Tq: Mudhol,\nDist: Bagalkot"), so it cannot end this one.
        rf"(?P<name>{_NAME}){_BREAK}(?:{label})\b(?![ \t]*:){_NOT_ROLE}",
        re.IGNORECASE,
    )
    for kind, label in _LABELS.items()
}

# "village Nandur-K", "Tq: Lingasugur", "Dist. Raichur"
# (no "." separator: "District. This Hon'ble court" is a sentence break)
_LABEL_FIRST = {
    kind: re.compile(
        rf"\b(?:{label})(?:\b|(?<=\.))[ \t]*[:\-]?{_BREAK}"
        rf"(?P<name>{_NAME})"
        rf"(?![ \t]*,?[ \t]*(?:{_ANY_LABEL})\b)",
        re.IGNORECASE,
    )
    for kind, label in _LABELS.items()
}

# Words that can never be (part of) a place name. When a candidate name
# contains one, only the words after the last stop word are kept
# ("The Tahsildar Sakleshpur" -> "Sakleshpur").
_STOP_WORDS = {
    "a", "all", "and", "at", "bearing", "block", "by", "for", "from", "in",
    "is", "its", "lying", "near", "of", "on", "or", "said", "same", "the",
    "this", "to", "with", "within", "situated", "situate", "land", "lands",
    "sy", "no", "nos", "survey", "r/at", "r/o", "rat", "resident",
    "residing", "office", "tahsildar", "tahasildar", "tahsilgar",
    "commissioner", "deputy", "assistant", "asst", "additional", "addl",
    "special", "director", "government", "govt", "revenue", "inspector",
    "surveyor", "accountant", "judge", "court", "sessions", "civil",
    "division", "sub-division", "post", "road", "city", "town", "opp",
    "respondent", "respondents", "petitioner", "petitioners", "appellant",
    "ward", "project", "limits", "nearby", "concerned", "respective",
    "schedule", "property", "ref", "dated", "dt", "annexure", "officer",
    "village", "vill", "grama", "hobli", "hobii", "taluk", "taluka",
    "taluku", "tq", "district", "dist", "state", "karnataka", "mr", "sri",
    "smt", "late", "son", "daughter", "wife", "aged", "years", "pin",
    "tahsil", "panchayat", "panchayath", "notification", "order", "copy",
    "sketch", "report", "mahazar", "rtc", "page", "memorandum", "writ",
    "rural", "urban", "there", "that", "which", "where", "when", "then",
    "his", "her", "their", "our", "as", "also", "both", "any", "other",
    "such", "hon", "hon'ble", "honble", "it", "was", "were", "has", "had",
}

# Words that are allowed only as part of a longer name
# ("Bangalore North", "Bengaluru Rural", "Dakshina Kannada").
_TRAILING_OK = {"north", "south", "east", "west", "rural", "urban"}

# Text right before a chain that marks it as a party/office address.
_ADDRESS_BEFORE = re.compile(
    r"(?:R\s*/\s*(?:at|o|a)\b|Rat\b|resid\w*|aged\s+about|years|"
    r"Tah\w*sildar|Commissioner|Office\w*\s+(?:of|at)?|Inspector|"
    r"Director|Officer|Judge|Court|Division|Conservator|Secretary|PIN)"
    r"[\s\W]{0,6}$",
    re.IGNORECASE,
)

# End of a sentence ("Village. Later the site in Sy.No.33/5").
_SENTENCE_BREAK = re.compile(r"[.;]\s+[A-Z]")

# PIN code right after a chain ("Hassan District - 573 201").
_ADDRESS_AFTER = re.compile(r"^[\s\W]{0,4}\d{3}\s?\d{3}\b")

# Linking windows (characters).
FORWARD_PREFERRED = 150
BACKWARD_MAX = 60
FORWARD_MAX = 300
CHAIN_GAP_MAX = 45
NEIGHBOUR_MAX = 800


# ==========================================================
# DATA
# ==========================================================


@dataclass
class _Place:
    kind: str
    name: str
    start: int
    end: int
    also: tuple[str, ...] = ()  # extra kinds (combined "Taluk & District")
    names: dict[str, str] = field(default_factory=dict)  # standardised

    @property
    def kinds(self) -> tuple[str, ...]:
        return (self.kind, *self.also)


@dataclass
class _Chain:
    places: list[_Place] = field(default_factory=list)
    is_address: bool = False

    @property
    def start(self) -> int:
        return self.places[0].start

    @property
    def end(self) -> int:
        return self.places[-1].end

    def location(self) -> dict:
        location = dict.fromkeys(FIELDS)

        for place in self.places:
            for kind in place.kinds:
                if location[kind] is None:
                    location[kind] = place.names.get(kind, place.name)

        return location


@dataclass
class _Mention:
    numbers: list[str]
    start: int
    end: int
    chain: _Chain | None = None


# ==========================================================
# HELPERS
# ==========================================================


def _key(value: str | None) -> str:
    return re.sub(r"[^a-z0-9]", "", (value or "").casefold())


def _score(location: dict) -> int:
    return sum(location.get(f) is not None for f in FIELDS)


def _empty(survey_number: str) -> dict:
    return {"survey_number": survey_number, **dict.fromkeys(FIELDS)}


def _is_stop(words: list[str], index: int) -> bool:
    word = _key(words[index])

    # "North" / "Rural" are fine after a name ("Bangalore North").
    if word in _TRAILING_OK:
        return index == 0

    return word in _STOP_WORDS


def _clean_name(raw: str, label_first: bool) -> str | None:
    """
    Trim a candidate name to the words next to its label: the words right
    after the label for "Village Nandur-K", right before it for
    "Rayasandra Village" ("The Tahsildar Sakleshpur" -> "Sakleshpur").
    """
    words = raw.split()

    if label_first:
        cut = next(
            (i for i in range(len(words)) if _is_stop(words, i)),
            len(words),
        )
        words = words[:cut]
    else:
        cut = next(
            (i for i in range(len(words) - 1, -1, -1) if _is_stop(words, i)),
            -1,
        )
        words = words[cut + 1:]

    if (
        not words
        or any(ch.isdigit() for ch in "".join(words))
        or all(_key(word) in _TRAILING_OK for word in words)
    ):
        return None

    name = clean_location_name(" ".join(words))

    if not name or len(_key(name)) < 3:
        return None

    return name


def _parent_number(number: str) -> str | None:
    if "/" not in number:
        return None

    return number.rsplit("/", 1)[0]


# ==========================================================
# EXTRACTOR
# ==========================================================


class SurveyLocationExtractor:

    def __init__(self, text: str, use_llm: bool = True):
        self.text = text
        self.use_llm = use_llm
        self._client = None
        self._standardize = PlaceNameStandardizer({})

    # ------------------------------------------------------
    # PUBLIC
    # ------------------------------------------------------

    def extract(self, survey_numbers: list[str]) -> list[dict]:
        survey_numbers = list(dict.fromkeys(survey_numbers))

        if not survey_numbers:
            return []

        wanted = set(survey_numbers)
        chains = self._find_chains()
        linkable = [chain for chain in chains if not chain.is_address]
        mentions = self._find_mentions(wanted)

        for mention in mentions:
            mention.chain = self._link(mention, linkable)

        results = {
            number: self._vote(number, mentions)
            for number in survey_numbers
        }

        self._fill_from_parents(results)
        self._fill_hierarchy(results, chains)

        missing = [n for n in survey_numbers if _score(results[n]) == 0]

        if missing and self.use_llm:
            for number, location in self._llm_locations(missing, mentions):
                results[number].update(location)

            self._fill_hierarchy(results, chains)

        self._fill_from_neighbours(results, mentions)
        self._fill_from_document(results)
        self._fill_hierarchy(results, chains)

        for number, location in results.items():
            if location["district"] is None:
                location["district"] = district_of_taluk(location["taluk"])

        located = sum(_score(loc) > 0 for loc in results.values())
        logger.info(
            "Survey locations: %d/%d located",
            located,
            len(results),
        )

        return [results[number] for number in survey_numbers]

    # ------------------------------------------------------
    # MENTIONS
    # ------------------------------------------------------

    def _find_mentions(self, wanted: set[str]) -> list[_Mention]:
        mentions = []

        for match in SurveyExtractor(self.text).iter_mentions():
            numbers = [
                number
                for number in Normalizer.normalize_survey_numbers(
                    [match.group(1)]
                )
                if number in wanted
            ]

            if numbers:
                mentions.append(
                    _Mention(numbers, match.start(), match.end())
                )

        return mentions

    # ------------------------------------------------------
    # PLACES & CHAINS
    # ------------------------------------------------------

    def _find_places(self) -> list[_Place]:
        text = self.text
        places: list[_Place] = []
        taken: list[tuple[int, int]] = []

        def overlaps(start: int, end: int) -> bool:
            return any(start < e and s < end for s, e in taken)

        def add(match: re.Match, kind: str, label_first: bool, also=()):
            name = _clean_name(match.group("name"), label_first)

            if not name or overlaps(match.start(), match.end()):
                return

            taken.append((match.start(), match.end()))
            places.append(
                _Place(kind, name, match.start(), match.end(), tuple(also))
            )

        # Most specific patterns first; later ones skip overlapping spans.
        # "Village Nandur-K Taluka and District Kalaburagi": the label-first
        # combined form must win over "Nandur-K Taluka and District".
        for pattern, label_first in (
            (_COMBINED_LABEL_FIRST, True),
            (_COMBINED_NAME_FIRST, False),
        ):
            for match in pattern.finditer(text):
                add(match, "taluk", label_first, ("district",))

        # Blank out combined spans so "Village Nandur-K" is not rejected
        # for being followed by a label.
        masked = list(text)
        for start, end in taken:
            masked[start:end] = " " * (end - start)
        masked = "".join(masked)

        for patterns, label_first in (
            (_NAME_FIRST, False),
            (_LABEL_FIRST, True),
        ):
            for kind, pattern in patterns.items():
                for match in pattern.finditer(masked):
                    add(match, kind, label_first)

        self._standardize = PlaceNameStandardizer(
            {
                kind: [p.name for p in places if kind in p.kinds]
                for kind in FIELDS
            }
        )

        for place in places:
            place.names = {
                kind: self._standardize(kind, place.name)
                for kind in place.kinds
            }

        return sorted(places, key=lambda p: p.start)

    def _find_chains(self) -> list[_Chain]:
        chains: list[_Chain] = []

        for place in self._find_places():
            current = chains[-1] if chains else None

            if (
                current
                and place.start - current.end <= CHAIN_GAP_MAX
                and RANK[place.kind] > max(
                    RANK[k] for k in current.places[-1].kinds
                )
            ):
                current.places.append(place)
            else:
                chains.append(_Chain([place]))

        for chain in chains:
            before = self.text[max(0, chain.start - 40):chain.start]
            after = self.text[chain.end:chain.end + 12]
            chain.is_address = bool(
                _ADDRESS_BEFORE.search(before)
                or _ADDRESS_AFTER.search(after)
            )

        return chains

    # ------------------------------------------------------
    # LINKING
    # ------------------------------------------------------

    def _link(self, mention: _Mention, chains: list[_Chain]) -> _Chain | None:
        # "Kuragod village in Savalagi Hobli in Sy.No.85 ... in Jamakhandi
        # taluk": the chain wraps around the mention.
        enclosing = next(
            (
                c for c in chains
                if c.start <= mention.start and mention.end <= c.end
            ),
            None,
        )

        if enclosing:
            return enclosing

        forward = next(
            (
                c for c in chains
                if mention.end <= c.start <= mention.end + FORWARD_MAX
            ),
            None,
        )
        backward = next(
            (
                c for c in reversed(chains)
                if mention.start - BACKWARD_MAX <= c.end <= mention.start
                # same sentence only ("Savalagi hobli in Sy.No.6")
                and not _SENTENCE_BREAK.search(
                    self.text[c.end:mention.start]
                )
            ),
            None,
        )

        if forward and forward.start - mention.end <= FORWARD_PREFERRED:
            return forward

        return backward or forward

    @staticmethod
    def _vote(number: str, mentions: list[_Mention]) -> dict:
        locations = [
            m.chain.location()
            for m in mentions
            if m.chain and number in m.numbers
        ]

        result = _empty(number)

        if not locations:
            return result

        villages = Counter(
            _key(loc["village"]) for loc in locations if loc["village"]
        )

        if villages:
            top_village = villages.most_common(1)[0][0]
            locations = [
                loc for loc in locations
                if not loc["village"] or _key(loc["village"]) == top_village
            ]

        for f in FIELDS:
            values = Counter(loc[f] for loc in locations if loc[f])

            if values:
                result[f] = values.most_common(1)[0][0]

        return result

    # ------------------------------------------------------
    # GAP FILLING
    # ------------------------------------------------------

    @staticmethod
    def _copy_missing(target: dict, source: dict) -> None:
        for f in FIELDS:
            if target[f] is None and source.get(f):
                target[f] = source[f]

    def _fill_from_parents(self, results: dict[str, dict]) -> None:
        for number, location in results.items():
            parent = _parent_number(number)

            while parent and _score(location) < len(FIELDS):
                if parent in results:
                    self._copy_missing(location, results[parent])

                parent = _parent_number(parent)

    @staticmethod
    def _fill_hierarchy(
        results: dict[str, dict],
        chains: list[_Chain],
    ) -> None:
        """
        Use parent units stated anywhere in the document (including party
        addresses): a village belongs to one hobli/taluk/district, etc.
        """
        known: dict[tuple[str, str], dict[str, Counter]] = {}

        observations = [chain.location() for chain in chains]
        observations += list(results.values())

        for loc in observations:
            for i, child in enumerate(FIELDS):
                if not loc.get(child):
                    continue

                for parent in FIELDS[i + 1:]:
                    if loc.get(parent):
                        known.setdefault(
                            (child, _key(loc[child])), {}
                        ).setdefault(parent, Counter())[loc[parent]] += 1

        for location in results.values():
            for i, child in enumerate(FIELDS):
                if not location[child]:
                    continue

                parents = known.get((child, _key(location[child])), {})

                for parent in FIELDS[i + 1:]:
                    if location[parent] is None and parents.get(parent):
                        location[parent] = (
                            parents[parent].most_common(1)[0][0]
                        )

    @classmethod
    def _fill_from_neighbours(
        cls,
        results: dict[str, dict],
        mentions: list[_Mention],
    ) -> None:
        located = [
            m for m in mentions
            if any(_score(results[n]) > 0 for n in m.numbers)
        ]

        for number, location in results.items():
            if _score(location) > 0:
                continue

            best, best_distance = None, NEIGHBOUR_MAX + 1

            for mention in (m for m in mentions if number in m.numbers):
                for other in located:
                    distance = min(
                        abs(other.start - mention.end),
                        abs(mention.start - other.end),
                    )

                    if distance < best_distance:
                        best, best_distance = other, distance

            if best:
                source = max(
                    (results[n] for n in best.numbers),
                    key=_score,
                )
                cls._copy_missing(location, source)

    @classmethod
    def _fill_from_document(cls, results: dict[str, dict]) -> None:
        located = [loc for loc in results.values() if _score(loc) > 0]

        if not located:
            return

        villages = Counter(
            _key(loc["village"]) for loc in located if loc["village"]
        )
        top_village = villages.most_common(1)[0][0] if villages else None

        best = max(
            located,
            key=lambda loc: (
                top_village is None or _key(loc["village"]) == top_village,
                _score(loc),
            ),
        )

        for location in results.values():
            if _score(location) == 0:
                cls._copy_missing(location, best)

    # ------------------------------------------------------
    # LLM FALLBACK
    # ------------------------------------------------------

    def _llm_locations(
        self,
        numbers: list[str],
        mentions: list[_Mention],
    ) -> list[tuple[str, dict]]:

        jobs = []

        for number in numbers:
            contexts = [
                self.text[max(0, m.start - 200):m.end + 300]
                for m in mentions
                if number in m.numbers
            ][:3]

            if contexts:
                jobs.append((number, contexts))

        if not jobs:
            return []

        try:
            if self._client is None:
                from extractor.llm.factory import get_llm_client

                self._client = get_llm_client()
        except Exception as exc:
            logger.warning("LLM unavailable for survey locations: %s", exc)
            return []

        def run(job):
            number, contexts = job

            try:
                response = self._client.generate(
                    build_location_prompt(number, contexts)
                )
                location = LocationResponseParser.parse(response)
            except Exception as exc:
                logger.warning("LLM location failed for %s: %s", number, exc)
                return number, {}

            # Accept only names that literally appear in the context.
            haystack = _key(" ".join(contexts))
            cleaned = {}

            for f in FIELDS:
                name = clean_location_name(location.get(f))

                if name and _key(name) and _key(name) in haystack:
                    cleaned[f] = self._standardize(f, name)

            return number, cleaned

        with ThreadPoolExecutor(max_workers=min(4, len(jobs))) as pool:
            return [
                (number, location)
                for number, location in pool.map(run, jobs)
                if location
            ]
