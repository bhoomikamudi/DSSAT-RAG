"""Extract a location mention and radius from a question.

Recognizes phrases such as "near Kitale", "around Eldoret, Kenya",
"within 50 km of Kisumu", "30 mile radius around Bungoma" and explicit
coordinates ("near 0.79, 35.04", "at lat 0.79 lon 35.04").

This module only finds the text; resolving a place name to coordinates is
done by GeocodingService, and distance filtering by the statistics layer.
The location phrase is removed from the question so the rest of the planner
(variables, cultivars, filters) never sees place names.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from app.services.analysis_vocabulary import cultivar_alias_pairs, variable_synonym_pairs

KM_PER_MILE = 1.609344

# Longer unit words first so "miles" is not read as "mi" + "les".
_UNIT_WORDS = r"(?:kilomet(?:er|re)s?|kms|km|miles?|mi)\b"
_UNIT = rf"(?P<unit>{_UNIT_WORDS})"
_NUMBER = r"(?P<radius>\d+(?:\.\d+)?)"

# "within 50 km of", "50 km around", "50-mile radius around", "within 50 km"
RADIUS_PATTERNS = [
    re.compile(
        rf"\bwithin\s+(?:a\s+)?{_NUMBER}\s*-?\s*{_UNIT}(?:\s+radius)?(?:\s+(?:of|from|around))?",
        re.I,
    ),
    re.compile(rf"\b{_NUMBER}\s*-?\s*{_UNIT}\s+(?:radius\s+)?(?:of|from|around)\b", re.I),
    re.compile(rf"\b(?:radius|range)\s+(?:of\s+)?{_NUMBER}\s*-?\s*{_UNIT}", re.I),
]

# Words that introduce a place when followed by a name.
TRIGGER = r"(?:near(?:by)?|around|close\s+to|in\s+the\s+vicinity\s+of|surrounding)"
_RADIUS_TRIGGER = (
    rf"within\s+(?:a\s+)?\d+(?:\.\d+)?\s*-?\s*{_UNIT_WORDS}(?:\s+radius)?\s+(?:of|from|around)"
    rf"|\d+(?:\.\d+)?\s*-?\s*{_UNIT_WORDS}\s+(?:radius\s+)?(?:of|from|around)"
)

_COORD_NUMBER = r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)"
_PAIR = rf"\(?\s*(?P<lat>{_COORD_NUMBER})\s*,\s*(?P<lon>{_COORD_NUMBER})(?![\w.])\s*\)?"
COORDINATE_PATTERNS = [
    # Explicit labels accept integers too. Capture the whole number so range
    # validation rejects invalid coordinates instead of matching a suffix.
    re.compile(
        rf"(?:\b(?:at|near|around|of|from)\s+)?\blat(?:itude)?\s*[:=]?\s*(?P<lat>{_COORD_NUMBER})"
        rf"\s*[,;]?\s*lon(?:g(?:itude)?)?\s*[:=]?\s*(?P<lon>{_COORD_NUMBER})(?![\w.])",
        re.I,
    ),
    re.compile(
        r"(?:\b(?:at|near|around|of|from)\s+)?"
        r"(?:\b(?:these|those|the|following|my|this)\s+)?"
        r"\b(?:coordinates?|coords|point|location|position|lat\s*/\s*lon)\b\s*(?:is|are)?\s*[:=]?\s*"
        + _PAIR,
        re.I,
    ),
    re.compile(
        rf"\b(?:{TRIGGER}|at|of|from)\s+\(?\s*(?P<lat>[-+]?\d+\.\d+)\s*,\s*(?P<lon>[-+]?\d+\.\d+)(?![\w.])\s*\)?",
        re.I,
    ),
    # "these coordinates: 0.79, 35.04", "point (0.79, 35.04)", or a bare
    # decimal pair. Both numbers need a decimal point, so "2010, 2015" or
    # "25, 50" are never read as coordinates.
    re.compile(
        r"(?:\b(?:these|those|the|following|my|this)\s+)?"
        r"(?:\b(?:coordinates?|coords|point|location|position|lat\s*/\s*lon)\b\s*(?:is|are)?\s*[:=]?\s*)?"
        r"\(?\s*(?<![\w.+-])(?P<lat>[-+]?\d+\.\d+)\s*,\s*(?P<lon>[-+]?\d+\.\d+)(?![\w.])\s*\)?",
        re.I,
    ),
]

# Words that refer to the selected map point rather than naming a place.
SELECTED_POINT_PATTERN = re.compile(
    r"\b(here|this (?:point|location|place|spot|area|site)|that (?:point|location|place|spot|area|site)|"
    r"(?:these|those) coordinates|over there|(?:the )?(?:selected|chosen|current) (?:map )?(?:point|location|area|spot)|"
    r"my (?:location|point|area))\b",
    re.I,
)


def mentions_selected_point(query: str) -> bool:
    """True when the question refers to "here" / the selected point."""
    return bool(SELECTED_POINT_PATTERN.search(query))

# A place is up to 5 words, optionally followed by ", Region[, Country]".
# It stops at punctuation or at words that start the rest of the question.
_STOP = (
    r"for|in|during|from|between|with|under|using|by|per|and|or|when|where|"
    r"if|that|which|who|to|on|at|of|each|every|separately|since|before|after|"
    r"over|across|vs|versus|than|compared|relative|is|are|was|were|has|have"
)
_WORD = rf"(?!(?:{_STOP})\b)[A-Za-z][\w'\-.]*"
PLACE_PATTERN = re.compile(
    rf"\b(?:{TRIGGER}|{_RADIUS_TRIGGER})\s+"
    r"(?:the\s+(?:town|city|village|county|area|region)\s+of\s+|the\s+)?"
    rf"(?P<place>{_WORD}(?:\s+{_WORD}){{0,4}}(?:\s*,\s*{_WORD}(?:\s+{_WORD}){{0,3}}){{0,2}})",
    re.I,
)

# Requests to drop any location filter and use the whole dataset.
LOCATION_RESET_PATTERN = re.compile(
    r"\b(all (?:the )?data|all (?:the )?simulations|all records|all locations|all areas|"
    r"(?:whole|entire|full) dataset|everywhere|"
    r"without (?:a |the |any )?(?:location|spatial|area|radius)(?: filter)?|"
    r"no (?:location|spatial|area|radius) filter|"
    r"(?:remove|clear|drop|ignore) (?:the )?(?:location|spatial|area|radius|map)(?: filter| point| selection)?)\b",
    re.I,
)


def mentions_location_reset(query: str) -> bool:
    """True when the user asks to stop filtering by location."""
    return bool(LOCATION_RESET_PATTERN.search(query))


# Common words that follow "near/around" but are not places
# ("yield near zero", "around the normal date", "near the peak").
NOT_A_PLACE = {
    "zero", "normal", "average", "mean", "median", "maximum", "minimum", "max",
    "min", "optimum", "optimal", "peak", "planting", "harvest", "maturity", "end",
    "start", "beginning", "middle", "full", "half", "here", "there", "it", "this",
    "that", "them", "values", "value", "level", "levels", "threshold", "record",
    "records", "data", "same", "expected", "typical", "baseline",
    # Words that point at the selected location instead of naming a place;
    # these must never be sent to the geocoder.
    "these", "those", "coordinates", "coordinate", "coords", "point", "points",
    "location", "position", "spot", "site", "selected", "chosen", "current", "my", "me", "us",
}


@dataclass
class LocationMention:
    place: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    radius_km: Optional[float] = None
    cleaned_query: str = ""

    @property
    def has_location(self) -> bool:
        return self.place is not None or self.latitude is not None


def _known_vocabulary() -> set:
    words = {phrase for phrase, _ in variable_synonym_pairs()}
    words |= {phrase for phrase, _ in cultivar_alias_pairs()}
    words |= {"base", "lng", "sht", "vlng", "vsht"}
    return words


def _is_place(place: str) -> bool:
    if not place:
        return False
    lowered = place.lower()
    vocabulary = _known_vocabulary()
    return (
        lowered.split()[0] not in NOT_A_PLACE
        and lowered not in vocabulary
        and not any(lowered.startswith(v + " ") for v in vocabulary)
    )


def _tidy(text: str) -> str:
    text = re.sub(r"\s+([?.!,;])", r"\1", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip(" ,;")


def _without_spans(text: str, spans: List[Tuple[int, int]]) -> str:
    keep = [True] * len(text)
    for start, end in spans:
        for index in range(start, end):
            keep[index] = False
    return _tidy("".join(ch if keep[i] else " " for i, ch in enumerate(text)))


def extract_location(query: str) -> LocationMention:
    """Find a place or coordinates and an optional radius in the question."""
    mention = LocationMention(cleaned_query=query)
    spans: List[Tuple[int, int]] = []

    for pattern in COORDINATE_PATTERNS:
        match = pattern.search(query)
        if match:
            mention.latitude = float(match.group("lat"))
            mention.longitude = float(match.group("lon"))
            spans.append(match.span())
            break

    if mention.latitude is None:
        match = PLACE_PATTERN.search(query)
        if match:
            place = match.group("place").strip(" ,.")
            if _is_place(place):
                mention.place = place
                spans.append((match.start(), match.end("place")))

    for pattern in RADIUS_PATTERNS:
        match = pattern.search(query)
        if match:
            radius = float(match.group("radius"))
            if match.group("unit").lower().startswith("mi"):
                radius *= KM_PER_MILE
            mention.radius_km = round(radius, 3)
            spans.append(match.span())
            break

    for match in SELECTED_POINT_PATTERN.finditer(query):
        start = match.start()
        prefix = re.search(r"\b(?:near|around|at|in)\s*$", query[:start], re.I)
        spans.append((prefix.start() if prefix else start, match.end()))
    if spans:
        mention.cleaned_query = _without_spans(query, spans)
    return mention
