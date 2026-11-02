from datetime import date, datetime

from remy.alerts.heartbeat import heartbeat_due, heartbeat_start_state, heartbeat_text


def test_due_once_per_day_at_or_after_the_hour():
    assert heartbeat_due(datetime(2026, 11, 2, 9, 5), 9, None) is True
    assert heartbeat_due(datetime(2026, 11, 2, 8, 59), 9, None) is False
    assert heartbeat_due(datetime(2026, 11, 2, 9, 5), 9, date(2026, 11, 2)) is False
    assert heartbeat_due(datetime(2026, 11, 2, 23, 0), 9, date(2026, 11, 1)) is True


def test_text_reads_plainly():
    since = datetime(2026, 11, 1, 19, 0)
    t = heartbeat_text("Nani", since, datetime(2026, 11, 2, 9, 0), 0, 0)
    assert t == "👀 Remy is watching Nani. Camera online for 14h, no events in the last 24 h."
    t = heartbeat_text("Nani", since, datetime(2026, 11, 3, 9, 0), 2, 1)
    assert "1d 14h" in t and "2 events" in t and "1 camera reconnect." in t


def test_a_restart_after_the_hour_waits_for_tomorrow():
    assert heartbeat_start_state(datetime(2026, 11, 2, 19, 45), 9) == date(2026, 11, 2)
    assert heartbeat_due(datetime(2026, 11, 2, 23, 0), 9, heartbeat_start_state(datetime(2026, 11, 2, 19, 45), 9)) is False
    assert heartbeat_due(datetime(2026, 11, 3, 9, 0), 9, heartbeat_start_state(datetime(2026, 11, 2, 19, 45), 9)) is True
    assert heartbeat_start_state(datetime(2026, 11, 2, 7, 0), 9) is None
