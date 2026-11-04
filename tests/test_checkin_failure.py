"""A confirmed fall must always reach the family. Before Nov 4 an error in the check-in ended the event with no
alert: `except Exception` let SystemExit through (a missing voice file, no camera credentials) and ended the
thread in silence, and every other error was logged and dropped."""
import copy
from types import SimpleNamespace

import pytest

import remy.alerts
import remy.checkin
from remy import monitor


class FakeWhatsApp:
    fail_init = False
    fail_send = 0
    sent: list = []

    def __init__(self, dry_run=False):
        if FakeWhatsApp.fail_init:
            raise SystemExit("WHATSAPP_TOKEN is not set.")
        self.dry = dry_run
        self.last_text = None

    def _send(self, kind, *a):
        if FakeWhatsApp.fail_send:
            FakeWhatsApp.fail_send -= 1
            raise RuntimeError("the network is down")
        FakeWhatsApp.sent.append((kind, a))
        self.last_text = str(a)
        return [{"_http": 200}]

    def urgent(self, summary, said, when, snapshot, reason, video=None):
        return self._send("urgent", reason, summary)

    def calm(self, summary, said, when, video=None):
        return self._send("calm", said, summary)


@pytest.fixture
def make_flow(tmp_path, cfg, monkeypatch):
    FakeWhatsApp.fail_init = False; FakeWhatsApp.fail_send = 0; FakeWhatsApp.sent = []
    monkeypatch.setattr(remy.alerts, "WhatsApp", FakeWhatsApp)
    monkeypatch.setattr(monitor, "EVENTS", tmp_path)
    c = copy.deepcopy(cfg); c["monitor"]["still_down_s"] = -5        # do not hold the flow open after the alert

    def build():
        args = SimpleNamespace(label="t", audio_out="laptop", alerts_dry_run=True, no_api=True, no_checkin=False, cooldown_s=30.0,
                               buffer_fps=10.0, reply=None)
        f = monitor.Flow("floor_still", monitor.State(None, 5.0, 480), args, c, (640, 480),
                         drop={"zone": "rug", "ar": 0.5, "move": 0.01, "for_s": 20.0})
        f.triage = {"summary_for_family": "On the rug."}             # no model call, no wait for one
        f.watch_still_down = lambda: None
        return f
    return build


@pytest.mark.parametrize("err", [SystemExit("TAPO_IP is not set."), RuntimeError("the classifier is down")])
def test_a_check_in_that_raises_still_sends_the_urgent_alert(make_flow, monkeypatch, err):
    def boom(*a, **k):
        raise err
    monkeypatch.setattr(remy.checkin, "ask_once", boom)
    f = make_flow(); f.run()
    assert f.result["outcome"] == "checkin_failed" and f.result["alert"]["kind"] == "urgent"
    kind, (reason, _) = FakeWhatsApp.sent[0][0], FakeWhatsApp.sent[0][1]
    assert kind == "urgent" and "could not finish" in reason
    assert str(err) in f.result["checkin_error"]


def test_a_second_round_that_raises_keeps_what_the_first_round_heard(make_flow, monkeypatch):
    calls = []

    def once(*a, **k):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("the mic died")
        return {"status": "no_answer", "transcript": {"transcript": ""}}
    monkeypatch.setattr(remy.checkin, "ask_once", once)
    f = make_flow(); f.run()
    assert len(calls) == 2 and f.result["alert"]["kind"] == "urgent" and len(f.result["checkin_rounds"]) == 1


def test_an_alert_that_raises_is_tried_once_more_without_the_extras(make_flow, monkeypatch):
    monkeypatch.setattr(remy.checkin, "ask_once", lambda *a, **k: {"status": "no_answer", "transcript": {"transcript": ""}})
    FakeWhatsApp.fail_send = 1
    f = make_flow(); f.run()
    assert f.result["alert"]["kind"] == "urgent" and f.result["alert"].get("last_resort") is True
    assert len(FakeWhatsApp.sent) == 1 and "error" in f.result


def test_no_way_to_send_is_logged_loudly_and_does_not_crash(make_flow, capsys):
    FakeWhatsApp.fail_init = True
    f = make_flow(); f.run()
    assert "alert" not in f.result and f.done
    assert "ALERT FAILED" in capsys.readouterr().out


def test_no_alert_when_no_event_was_raised(make_flow, monkeypatch):
    f = make_flow()
    monkeypatch.setattr(f, "raise_event", lambda: (_ for _ in ()).throw(RuntimeError("before the event")))
    f.run()
    assert FakeWhatsApp.sent == [] and f.result["event"] is False


def test_a_good_check_in_is_not_changed(make_flow, monkeypatch):
    monkeypatch.setattr(remy.checkin, "ask_once", lambda *a, **k: {"status": "needs_help", "transcript": {"transcript": "help me"}})
    f = make_flow(); f.run()
    assert f.result["alert"]["kind"] == "urgent" and f.result["alert"]["reason"] == "She asked for help."
    assert f.result["said"] == "help me" and "checkin_error" not in f.result
