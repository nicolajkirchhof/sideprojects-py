"""
finance.apps.assistant._calendar
=================================
Economic calendar — fetch upcoming high-impact events for risk assessment.

Uses the free FairEconomy ForexFactory calendar API.
Migrated from finance.apps.analyst._calendar.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

import requests

log = logging.getLogger(__name__)

_CALENDAR_URLS = {
    "this_week": "https://nfs.faireconomy.media/ff_calendar_thisweek.json",
    "next_week": "https://nfs.faireconomy.media/ff_calendar_nextweek.json",
}

# Events that trigger a NO-GO or size reduction per the playbook
NO_GO_KEYWORDS = {"FOMC", "Fed Rate", "Interest Rate Decision", "CPI", "Non-Farm", "NFP"}

REQUEST_TIMEOUT = 10


@dataclass
class EconomicEvent:
    title: str
    country: str
    date: datetime
    impact: str  # "Low" | "Medium" | "High"
    forecast: str = ""
    previous: str = ""


def fetch_upcoming_events(
    days_ahead: int = 5,
    impact_filter: str = "High",
) -> list[EconomicEvent]:
    """Fetch upcoming economic events, filtered by impact and date range.

    Args:
        days_ahead: How many days ahead to look.
        impact_filter: Minimum impact level ("High", "Medium", "Low").
    """
    from datetime import timezone
    events: list[EconomicEvent] = []
    now = datetime.now(timezone.utc)
    cutoff = now + timedelta(days=days_ahead)
    impact_levels = _impact_levels(impact_filter)

    for label, url in _CALENDAR_URLS.items():
        try:
            resp = requests.get(url, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            log.warning("Failed to fetch %s calendar", label)
            continue

        for entry in data:
            impact = entry.get("impact", "Low")
            if impact not in impact_levels:
                continue

            try:
                dt = datetime.fromisoformat(entry["date"].replace("Z", "+00:00"))
            except (ValueError, KeyError):
                continue

            if dt < now:
                continue  # already passed
            if dt > cutoff:
                continue

            events.append(EconomicEvent(
                title=entry.get("title", ""),
                country=entry.get("country", ""),
                date=dt,
                impact=impact,
                forecast=entry.get("forecast", ""),
                previous=entry.get("previous", ""),
            ))

    events.sort(key=lambda e: e.date)
    usd_high = [e for e in events if e.country == "USD" and e.impact == "High"]
    log.info("Fetched %d upcoming events (%d high-impact USD)", len(events), len(usd_high))
    return events


def events_to_dicts(events: list[EconomicEvent]) -> list[dict]:
    """Serialise a list of EconomicEvent to JSON-compatible dicts."""
    return [
        {
            "title": e.title,
            "country": e.country,
            "date": e.date.isoformat(),
            "impact": e.impact,
            "forecast": e.forecast,
            "previous": e.previous,
        }
        for e in events
    ]


def check_macro_risk(events: list[EconomicEvent]) -> tuple[bool, list[str]]:
    """Check if any upcoming events should trigger caution or NO-GO.

    Returns:
        (has_risk, list of risk descriptions)
    """
    from datetime import timezone
    now = datetime.now(timezone.utc)
    risks: list[str] = []

    for event in events:
        if event.country != "USD" or event.impact != "High":
            continue

        hours_away = (event.date - now).total_seconds() / 3600
        if hours_away < 0:
            continue  # already passed

        is_nogo_event = any(kw.lower() in event.title.lower() for kw in NO_GO_KEYWORDS)

        if hours_away <= 48 and is_nogo_event:
            risks.append(f"NO-GO: {event.title} in {hours_away:.0f}h ({event.date.strftime('%a %b %d %H:%M')})")
        elif hours_away <= 48:
            risks.append(f"CAUTION: {event.title} in {hours_away:.0f}h ({event.date.strftime('%a %b %d %H:%M')})")
        elif hours_away <= 120:
            risks.append(f"UPCOMING: {event.title} on {event.date.strftime('%a %b %d')}")

    return len(risks) > 0, risks


def check_macro_risk_from_dicts(event_dicts: list[dict]) -> tuple[bool, list[str]]:
    """Like check_macro_risk but accepts serialised event dicts from cache."""
    events = []
    for d in event_dicts:
        try:
            dt = datetime.fromisoformat(d["date"])
            events.append(EconomicEvent(
                title=d.get("title", ""),
                country=d.get("country", ""),
                date=dt,
                impact=d.get("impact", "High"),
                forecast=d.get("forecast", ""),
                previous=d.get("previous", ""),
            ))
        except (ValueError, KeyError):
            continue
    return check_macro_risk(events)


def _impact_levels(minimum: str) -> set[str]:
    levels = ["Low", "Medium", "High"]
    idx = levels.index(minimum) if minimum in levels else 0
    return set(levels[idx:])
