"""Place-name lookup (geocoding) via OpenStreetMap Nominatim.

Only turns a place name into coordinates; distance filtering lives in the
statistics/spatial services. Uses the free Nominatim API (or any compatible
self-hosted server) through httpx, which the project already depends on.

A lookup never silently picks an unrelated result:
- no results, or results whose name does not match the query -> "not_found"
- several plausible, far-apart places with no clear winner    -> "ambiguous"
- network/HTTP errors                                        -> "error"
"""
from __future__ import annotations

import asyncio
import logging
import math
import re
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

import httpx

from app.core.config import get_settings

logger = logging.getLogger(__name__)

DEFAULT_USER_AGENT = "DSSAT-RAG/1.0 (set GEOCODER_USER_AGENT with a contact)"

# Two candidates farther apart than this are considered different places.
DISTINCT_PLACE_KM = 50.0
# A top result counts as a clear winner when its importance leads by this much.
IMPORTANCE_MARGIN = 0.15


@dataclass
class GeocodeCandidate:
    name: str
    latitude: float
    longitude: float
    importance: float = 0.0
    place_type: Optional[str] = None


@dataclass
class GeocodeResult:
    status: Literal["ok", "ambiguous", "not_found", "error", "disabled"]
    query: str
    match: Optional[GeocodeCandidate] = None
    candidates: List[GeocodeCandidate] = field(default_factory=list)
    message: str = ""


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower()).strip()


def _haversine_km(a: GeocodeCandidate, b: GeocodeCandidate) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (a.latitude, a.longitude, b.latitude, b.longitude))
    h = (math.sin((lat2 - lat1) / 2) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2)
    return 2 * 6371.0088 * math.asin(math.sqrt(h))


def _name_matches(query: str, candidate_name: str) -> bool:
    """The candidate's name must contain the query's most specific part.

    For "Kitale, Kenya" every significant word of "Kitale" must appear in the
    result's name, so a fuzzy match to some other place is rejected.
    """
    primary = _normalize(query.split(",")[0])
    name_words = set(_normalize(candidate_name).split())
    words = [w for w in primary.split() if len(w) > 2] or primary.split()
    return bool(words) and all(word in name_words for word in words)


class GeocodingService:
    """Async Nominatim client with caching and polite rate limiting."""

    _cache: Dict[str, GeocodeResult] = {}
    _lock: Optional[asyncio.Lock] = None
    _last_request: float = 0.0

    def __init__(self, transport: Optional[httpx.AsyncBaseTransport] = None):
        settings = get_settings()
        self.enabled = settings.GEOCODER_ENABLED
        self.url = settings.GEOCODER_URL
        self.user_agent = settings.GEOCODER_USER_AGENT or DEFAULT_USER_AGENT
        self.timeout = settings.GEOCODER_TIMEOUT_SECONDS
        self.min_interval = settings.GEOCODER_MIN_INTERVAL_SECONDS
        self.transport = transport

    @classmethod
    def clear_cache(cls) -> None:
        cls._cache.clear()

    async def geocode(
        self,
        place: str,
        prefer_bounds: Optional[Dict[str, float]] = None,
    ) -> GeocodeResult:
        """Resolve a place name to coordinates.

        Args:
            place: Place text from the user, e.g. "Kitale" or "Kitale, Kenya".
            prefer_bounds: Optional data-coverage box (min/max lat/lon). When
                several distinct places share the name, the single one inside
                this box is chosen, and the result says so.
        """
        query = " ".join(place.split())
        if not query:
            return GeocodeResult("not_found", place, message="No place name was given.")
        if not self.enabled:
            return GeocodeResult(
                "disabled", query,
                message="Place lookup is disabled on this server; select a point on the map or give coordinates.",
            )

        cache_key = _normalize(query)
        cached = self._cache.get(cache_key)
        if cached is not None:
            # Matching candidates are cached; the choice depends on prefer_bounds.
            if cached.status == "ok":
                return self._choose(query, cached.candidates, prefer_bounds)
            return cached

        try:
            raw = await self._request(query)
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning("Geocoding failed for %r: %s", query, exc)
            return GeocodeResult(
                "error", query,
                message=f"The place lookup service could not be reached, so '{query}' was not resolved. "
                        "Try again, select a point on the map, or give coordinates.",
            )

        candidates = []
        for item in raw:
            try:
                candidates.append(
                    GeocodeCandidate(
                        name=str(item.get("display_name") or item.get("name") or ""),
                        latitude=float(item["lat"]),
                        longitude=float(item["lon"]),
                        importance=float(item.get("importance") or 0.0),
                        place_type=item.get("type") or item.get("addresstype"),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue

        matching = [c for c in candidates if _name_matches(query, c.name)]
        if not matching:
            result = GeocodeResult(
                "not_found", query,
                candidates=candidates[:3],
                message=(
                    f"No place named '{query}' was found."
                    + (f" The closest lookup result was '{candidates[0].name}', which does not match, so it was not used."
                       if candidates else "")
                    + " Check the spelling, add a region or country, or select a point on the map."
                ),
            )
            self._cache[cache_key] = result
            return result

        self._cache[cache_key] = GeocodeResult("ok", query, candidates=matching)
        return self._choose(query, matching, prefer_bounds)

    def _choose(
        self,
        query: str,
        candidates: List[GeocodeCandidate],
        prefer_bounds: Optional[Dict[str, float]],
    ) -> GeocodeResult:
        """Pick one candidate only when that choice is unambiguous."""
        top = candidates[0]
        distinct = [c for c in candidates[1:] if _haversine_km(top, c) > DISTINCT_PLACE_KM]

        if not distinct or top.importance - max(c.importance for c in distinct) >= IMPORTANCE_MARGIN:
            return GeocodeResult("ok", query, match=top, candidates=candidates,
                                 message=f"'{query}' resolved to {top.name}.")

        options = [top] + distinct
        if prefer_bounds:
            inside = [c for c in options if _inside(c, prefer_bounds)]
            if len(inside) == 1:
                chosen = inside[0]
                return GeocodeResult(
                    "ok", query, match=chosen, candidates=options,
                    message=(f"'{query}' matches several places; using {chosen.name} "
                             "because it is the only one inside the data coverage area."),
                )

        listed = "; ".join(f"{c.name} ({c.latitude:.3f}, {c.longitude:.3f})" for c in options[:5])
        return GeocodeResult(
            "ambiguous", query, candidates=options,
            message=(f"'{query}' matches several different places: {listed}. "
                     "Please add a region or country, or select the point on the map."),
        )

    async def _request(self, query: str) -> List[Dict[str, Any]]:
        params = {"q": query, "format": "jsonv2", "limit": 5, "addressdetails": 0}
        headers = {"User-Agent": self.user_agent, "Accept-Language": "en"}

        if GeocodingService._lock is None:
            GeocodingService._lock = asyncio.Lock()
        async with GeocodingService._lock:
            wait = self.min_interval - (time.monotonic() - GeocodingService._last_request)
            if wait > 0:
                await asyncio.sleep(wait)
            try:
                async with httpx.AsyncClient(
                    timeout=self.timeout, transport=self.transport
                ) as client:
                    response = await client.get(self.url, params=params, headers=headers)
            finally:
                GeocodingService._last_request = time.monotonic()

        response.raise_for_status()
        data = response.json()
        if not isinstance(data, list):
            raise ValueError("Unexpected geocoder response")
        return data


def _inside(candidate: GeocodeCandidate, bounds: Dict[str, float]) -> bool:
    return (
        bounds["min_lat"] <= candidate.latitude <= bounds["max_lat"]
        and bounds["min_lon"] <= candidate.longitude <= bounds["max_lon"]
    )
