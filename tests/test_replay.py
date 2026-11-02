from pathlib import Path

from remy.replay import replay

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "event_gmdcsa_S1_Fall_12"


def test_example_event_replays_and_rescores_to_the_stored_verdict(capsys):
    assert replay(EXAMPLE, show_log=False) == 0
    out = capsys.readouterr().out
    assert "flag=True" in out and "matches the stored score.json: yes" in out
    assert "triage: urgency=" in out and "check-in round 1" in out and "outcome: no_answer" in out
