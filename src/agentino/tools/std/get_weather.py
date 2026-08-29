"""get_weather — current weather + same-day outlook via Open-Meteo.

Replaces the previous fetch_web_data path for weather, which was unreliable
(JS-heavy pages, LLM judge struggled with hourly forecast tables).
"""

from __future__ import annotations

from agentino.core.tool import tool

from ._weather import (
    fetch_hourly,
    format_current,
    format_window,
    geocode,
    resolve_default_location,
)


@tool(is_read_only=True)
async def get_weather(location: str = "", hours_ahead: int = 6) -> str:
    """Fetch CURRENT weather + a short outlook (next N hours) for a city.

    Use this for any "what's the weather now / today / tomorrow morning"
    question. Do NOT use fetch_web_data for weather — this tool is structured
    and reliable.

    Args:
        location: Free-form city/region ("Paphos, Cyprus", "Limassol",
            "London"). Empty string = use the tenant's configured city.
        hours_ahead: How many hours of outlook to summarize after the current
            reading. Default 6 (covers "rest of today" without bloat).

    Returns a compact line: "Paphos now: 19.2°C, wind 12.0 km/h, precip 0 mm,
    partly cloudy. Next 6h: 17–20°C, wind ≤14 km/h, precip 0 mm total, clear."
    """
    loc = (location or "").strip() or resolve_default_location()
    if not loc:
        return "weather: no location provided and no tenant default configured"

    geo = await geocode(loc)
    if not geo:
        return f"weather: could not geocode location {loc!r}"

    payload = await fetch_hourly(geo["lat"], geo["lon"], days=1)
    if not payload:
        return f"weather: forecast fetch failed for {geo['name']}"

    name = geo.get("name") or loc
    parts = [format_current(payload, location_name=name)]

    # Same-day outlook for [now+1h, now+hours_ahead].
    cur_time = (payload.get("current") or {}).get("time", "")
    if cur_time:
        try:
            cur_hour = int(cur_time[11:13])
            target_date = cur_time[:10]
            window = format_window(
                payload,
                target_date=target_date,
                hour_from=min(23, cur_hour + 1),
                hour_to=min(23, cur_hour + max(1, hours_ahead)),
            )
            if window:
                parts.append(f"\nOutlook: {window}")
        except Exception:
            pass

    return " ".join(parts).strip()
