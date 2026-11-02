"""A once-a-day "still watching" message to the family, so silence means nothing happened, not that
the monitor died. Sent the first time the clock passes config alerts.heartbeat.hour each day."""
from __future__ import annotations

from datetime import date, datetime


def heartbeat_due(now: datetime, hour: int, last_sent: date | None) -> bool:
    return now.hour >= hour and last_sent != now.date()


def heartbeat_start_state(now: datetime, hour: int) -> date | None:
    """What `last_sent` should be when the monitor starts: today if the hour has already passed, so a
    restart (or a crash loop) never messages the family; the first heartbeat is tomorrow at `hour`."""
    return now.date() if now.hour >= hour else None


def heartbeat_text(person: str, online_since: datetime, now: datetime, events_24h: int, reconnects: int) -> str:
    hours = max(0, int((now - online_since).total_seconds() // 3600))
    up = f"{hours // 24}d {hours % 24}h" if hours >= 24 else f"{hours}h"
    ev = "no events" if events_24h == 0 else f"{events_24h} event{'s' if events_24h != 1 else ''}"
    rc = "" if reconnects == 0 else f", {reconnects} camera reconnect{'s' if reconnects != 1 else ''}"
    return f"👀 Remy is watching {person}. Camera online for {up}, {ev} in the last 24 h{rc}."
