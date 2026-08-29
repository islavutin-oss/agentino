"""get_weather_forecast — forecast for a specific date + hour window.

For the daily-closing "tomorrow's dinner service 18-23h" use case + any
"what'll the weather be at X o'clock on Y?" query.
"""

from __future__ import annotations

from agentino.core.tool import tool

from ._weather import (
    fetch_hourly,
    format_window,
    geocode,
    resolve_default_location,
)


@tool(is_read_only=True)
async def get_weather_forecast(
    date: str,
    hour_from: int = 18,
    hour_to: int = 23,
    location: str = "",
) -> str:
    """Aggregate hourly forecast for a window [hour_from, hour_to] on `date`
    (YYYY-MM-DD). Returns one compact line: temp range, max wind, total
    precipitation, dominant condition.

    Use for terrace/dinner-service planning ("will it rain tomorrow evening?")
    and the daily-closing "tomorrow outlook" line.

    Args:
        date: Target date YYYY-MM-DD (must be within ~7 days; Open-Meteo
            forecast horizon).
        hour_from: Window start hour (0-23, local time).
        hour_to: Window end hour (0-23, local time, inclusive).
        location: City/region — empty string uses the tenant default.

    Returns: "Paphos 2026-04-26 18-23h: 17-20°C, wind ≤14 km/h, precip 0 mm
    total, clear" — or a clear failure line.
    """
    loc = (location or "").strip() or resolve_default_location()
    if not loc:
        return "weather: no location provided and no tenant default configured"

    if hour_from < 0 or hour_to > 23 or hour_from > hour_to:
        return f"weather: invalid hour window [{hour_from}-{hour_to}]"

    geo = await geocode(loc)
    if not geo:
        return f"weather: could not geocode location {loc!r}"

    payload = await fetch_hourly(geo["lat"], geo["lon"], days=2)
    if not payload:
        return f"weather: forecast fetch failed for {geo['name']}"

    line = format_window(
        payload,
        target_date=date,
        hour_from=hour_from,
        hour_to=hour_to,
        location_name=geo.get("name") or loc,
    )
    if not line:
        return (
            f"weather: no forecast data in window for {date} "
            f"{hour_from:02d}-{hour_to:02d}h ({geo.get('name')}) — "
            f"date may be outside the forecast horizon"
        )
    return line
