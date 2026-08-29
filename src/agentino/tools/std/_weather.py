"""Open-Meteo weather helper — private, used by get_weather + get_weather_forecast.

Why Open-Meteo:
  - free, no API key
  - clean JSON (hourly arrays for temp/wind/precip)
  - built-in geocoding (city name → lat/lon)

Replaces ad-hoc scraping via fetch_web_data for weather queries: structured,
deterministic, fast (single HTTP per call vs LLM-judge loop on flaky pages).
"""

from __future__ import annotations

import logging

import httpx

log = logging.getLogger(__name__)

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HTTP_TIMEOUT = 10.0


# ----------------------------------------------------------------------------
# Geocoding
# ----------------------------------------------------------------------------


async def geocode(location: str) -> dict | None:
    """Resolve a free-form location string to {lat, lon, name, country, timezone}.

    Open-Meteo's geocoder is fuzzy on city names but rejects full street
    addresses. To make this robust for tenants that store a full address
    (e.g. "82 Akropoleos Street, Paphos, Cyprus"), we try in order:

      1. The exact input.
      2. Last 2 comma-segments (typical "City, Country").
      3. Last segment only (just the country).

    First non-empty hit wins. Returns None if all attempts miss.
    """
    if not location or not location.strip():
        return None

    candidates = [location.strip()]
    parts = [p.strip() for p in location.split(",") if p.strip()]
    if len(parts) >= 2:
        candidates.append(", ".join(parts[-2:]))
    if len(parts) >= 1 and parts[-1] not in candidates:
        candidates.append(parts[-1])

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as c:
        for cand in candidates:
            try:
                r = await c.get(
                    GEOCODE_URL,
                    params={
                        "name": cand,
                        "count": 1,
                        "language": "en",
                        "format": "json",
                    },
                )
                r.raise_for_status()
            except Exception as e:
                log.warning("geocode HTTP failed for %r: %s", cand, e)
                continue
            data = r.json() or {}
            results = data.get("results") or []
            if results:
                hit = results[0]
                return {
                    "lat": hit.get("latitude"),
                    "lon": hit.get("longitude"),
                    "name": hit.get("name"),
                    "country": hit.get("country"),
                    "timezone": hit.get("timezone"),
                }
    return None


# ----------------------------------------------------------------------------
# Forecast
# ----------------------------------------------------------------------------


async def fetch_hourly(lat: float, lon: float, *, days: int = 2) -> dict | None:
    """Fetch hourly + current weather for the next N days (default 2 — today +
    tomorrow). Returns the full Open-Meteo response (nested dict)."""
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as c:
        try:
            r = await c.get(
                FORECAST_URL,
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "current": "temperature_2m,wind_speed_10m,precipitation,weather_code",
                    "hourly": "temperature_2m,precipitation,wind_speed_10m,weather_code",
                    "timezone": "auto",
                    "forecast_days": days,
                },
            )
            r.raise_for_status()
        except Exception as e:
            log.warning("forecast fetch failed for (%s, %s): %s", lat, lon, e)
            return None
    return r.json() or None


# ----------------------------------------------------------------------------
# Formatting / aggregation
# ----------------------------------------------------------------------------

# Compact text labels for the WMO weather codes we care about. Open-Meteo codes:
# https://open-meteo.com/en/docs (search "weather_code")
_WCODE = {
    0: "clear",
    1: "mostly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "fog (rime)",
    51: "drizzle (light)",
    53: "drizzle",
    55: "drizzle (heavy)",
    61: "rain (light)",
    63: "rain",
    65: "rain (heavy)",
    66: "freezing rain (light)",
    67: "freezing rain (heavy)",
    71: "snow (light)",
    73: "snow",
    75: "snow (heavy)",
    77: "snow grains",
    80: "rain showers (light)",
    81: "rain showers",
    82: "rain showers (violent)",
    85: "snow showers (light)",
    86: "snow showers (heavy)",
    95: "thunderstorm",
    96: "thunderstorm w/ hail",
    99: "thunderstorm w/ heavy hail",
}


def label_code(code: int | None) -> str:
    if code is None:
        return ""
    return _WCODE.get(int(code), f"code {code}")


def format_current(payload: dict, location_name: str = "") -> str:
    cur = (payload or {}).get("current") or {}
    if not cur:
        return f"weather data unavailable for {location_name}".strip()
    parts = []
    if location_name:
        parts.append(f"**{location_name}** now:")
    t = cur.get("temperature_2m")
    w = cur.get("wind_speed_10m")
    p = cur.get("precipitation")
    code = cur.get("weather_code")
    detail = []
    if t is not None:
        detail.append(f"{t:.1f}°C")
    if w is not None:
        detail.append(f"wind {w:.1f} km/h")
    if p is not None:
        detail.append(f"precip {p:.1f} mm")
    label = label_code(code)
    if label:
        detail.append(label)
    parts.append(", ".join(detail) or "no data")
    return " ".join(parts)


def format_window(
    payload: dict, *, target_date: str, hour_from: int, hour_to: int, location_name: str = ""
) -> str:
    """Aggregate the hourly arrays into a single line for [hour_from, hour_to]
    on `target_date` (YYYY-MM-DD). Min/max temp, max wind, total precip,
    dominant condition (most common label). Returns "" if no overlap."""
    hourly = (payload or {}).get("hourly") or {}
    times = hourly.get("time") or []
    temps = hourly.get("temperature_2m") or []
    winds = hourly.get("wind_speed_10m") or []
    precs = hourly.get("precipitation") or []
    codes = hourly.get("weather_code") or []

    sel = []
    for i, ts in enumerate(times):
        # Open-Meteo returns ISO local time strings: "2026-04-26T18:00"
        if not ts.startswith(target_date):
            continue
        try:
            hour = int(ts[11:13])
        except Exception:
            continue
        if hour_from <= hour <= hour_to:
            sel.append(i)

    if not sel:
        return ""

    def _pick(arr, idxs):
        return [arr[i] for i in idxs if i < len(arr) and arr[i] is not None]

    t = _pick(temps, sel)
    w = _pick(winds, sel)
    p = _pick(precs, sel)
    c = _pick(codes, sel)

    bits = []
    if location_name:
        bits.append(f"**{location_name}**")
    bits.append(f"{target_date} {hour_from:02d}-{hour_to:02d}h:")
    detail = []
    if t:
        detail.append(f"{min(t):.0f}-{max(t):.0f}°C")
    if w:
        detail.append(f"wind ≤{max(w):.0f} km/h")
    if p:
        detail.append(f"precip {sum(p):.1f} mm total")
    if c:
        # Most common label in the window.
        from collections import Counter

        common, _ = Counter(label_code(int(x)) for x in c).most_common(1)[0]
        if common:
            detail.append(common)
    bits.append(", ".join(detail) or "no data")
    return " ".join(bits)


# ----------------------------------------------------------------------------
# Tenant default location
# ----------------------------------------------------------------------------


def resolve_default_location() -> str:
    """If the caller didn't pass a location, fall back to the tenant's
    configured city. Lookup order:
      1. `tenant.weather_location` — explicit setting in tenant config (best
         precision; owner sets it to "Paphos, Cyprus" or similar).
      2. `tenant.address` — the full street address. Open-Meteo's geocoder
         accepts "82 Akropoleos Street, Paphos, Cyprus" just fine, so we pass
         the whole string. No fragile last-segment heuristic.
    Returns "" if nothing's configured — caller surfaces a friendly error.
    """
    try:
        from agentino.core.context import get_context
    except ImportError:
        return ""
    tenant = get_context("tenant")
    if not tenant:
        return ""
    loc = getattr(tenant, "weather_location", None)
    if loc and str(loc).strip():
        return str(loc).strip()
    addr = getattr(tenant, "address", None)
    if addr and str(addr).strip():
        return str(addr).strip()
    return ""
