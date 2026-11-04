"""What happens when a part of the system is down: SAM, the WhatsApp send, the speaker or mic, the camera. The
rule throughout is that the family hears what is known, and that nothing claims a message went out when it did not."""
import asyncio
import copy
import time
from types import SimpleNamespace

import numpy as np
import pytest
import requests

import remy.alerts
import remy.alerts.whatsapp as wa_mod
import remy.checkin
from remy import monitor
from remy.alerts.whatsapp import WhatsApp, delivered
from remy.camera import capture as cap_mod
from remy.camera.capture import Capture
from remy.vision import sam

OK, REFUSED = {"_http": 200}, {"_http": 401, "error": {"message": "expired"}}


class FakeWA:
    answer = OK
    sent: list = []

    def __init__(self, dry_run=False):
        self.dry = dry_run; self.last_text = None

    def notice(self, text, video=None, snapshot=None):
        FakeWA.sent.append(("notice", text, video)); self.last_text = text
        return [FakeWA.answer]

    def urgent(self, summary, said, when, snapshot, reason, video=None):
        FakeWA.sent.append(("urgent", reason, video)); self.last_text = reason
        return [FakeWA.answer]

    def calm(self, summary, said, when, video=None):
        FakeWA.sent.append(("calm", said, video)); self.last_text = said
        return [FakeWA.answer]


@pytest.fixture
def make_flow(tmp_path, cfg, monkeypatch):
    FakeWA.answer = OK; FakeWA.sent = []
    monkeypatch.setattr(remy.alerts, "WhatsApp", FakeWA)
    monkeypatch.setattr(monitor, "EVENTS", tmp_path)
    c = copy.deepcopy(cfg); c["monitor"]["still_down_s"] = -5
    state = monitor.State(None, 5.0, 480)
    n = [0]

    def build(kind="drop", state_=None, **over):
        n[0] += 1
        args = SimpleNamespace(label=f"t{n[0]}", audio_out="laptop", alerts_dry_run=True, no_api=False, no_checkin=False, cooldown_s=30.0,
                               buffer_fps=10.0, reply=None)
        drop = {"drop_v": 100.0, "drop_wall": time.time(), "obj": 1, "best": {}} if kind == "drop" else {"zone": "rug", "ar": 0.5, "move": 0.01, "for_s": 20.0}
        f = monitor.Flow(kind, state_ or state, args, c, (640, 480), drop=drop)
        f.triage = {"summary_for_family": "On the rug."}
        f.watch_still_down = lambda: None
        f.say_after = lambda key: f.result.setdefault("said_lines", []).append(key)
        for k, v in over.items():
            setattr(f, k, v)
        return f
    return build


def sam_down(*a, **k):
    raise ConnectionError("SAM is down")


# ---- SAM down: a caution, not silence and not a false "fall confirmed"
def test_sam_down_sends_a_caution_and_raises_no_event(make_flow):
    f = make_flow(confirm_drop=sam_down); f.run()
    assert f.result["event"] is False and f.result["alert"]["kind"] == "caution"
    kind, text, _ = FakeWA.sent[0]
    assert kind == "notice" and "could not confirm" in text and "temporarily down" in text and "Nani" in text


@pytest.mark.parametrize("err", [SystemExit("META_API_KEY is not set"), OSError("disk full")])
def test_any_error_before_the_decision_is_also_a_caution(make_flow, err):
    def boom():
        raise err
    f = make_flow(confirm_drop=boom); f.run()
    assert f.result["alert"]["kind"] == "caution"


def test_a_person_who_is_up_again_gets_no_caution(make_flow):
    f = make_flow(confirm_drop=sam_down, person_got_up=lambda t: True); f.run()
    assert FakeWA.sent == [] and "alert" not in f.result


def test_an_outage_does_not_flood_the_family(make_flow):
    make_flow(confirm_drop=sam_down).run()
    f2 = make_flow(confirm_drop=sam_down); f2.run()
    assert len(FakeWA.sent) == 1 and "alert" not in f2.result
    f2.state.caution_wall = time.time() - 601                  # the window has passed: a new drop is worth saying again
    make_flow(confirm_drop=sam_down).run()
    assert len(FakeWA.sent) == 2


def test_a_caution_that_did_not_go_out_is_not_counted(make_flow):
    FakeWA.answer = REFUSED
    f = make_flow(confirm_drop=sam_down); f.run()
    assert f.result["alert"]["delivered"] is False and f.state.tracking_down is False
    FakeWA.answer = OK
    make_flow(confirm_drop=sam_down).run()
    assert len(FakeWA.sent) == 2


def test_the_family_hears_when_tracking_is_back(make_flow):
    f = make_flow(confirm_drop=sam_down); f.run()
    assert f.state.tracking_down is True
    f2 = make_flow(); f2.tracking_recovered(); f2.tracking_recovered()
    deadline = time.time() + 2
    while len(FakeWA.sent) < 2 and time.time() < deadline:
        time.sleep(0.01)
    assert len(FakeWA.sent) == 2 and "working again" in FakeWA.sent[1][1] and f.state.tracking_down is False


def test_a_hung_sam_call_hits_the_deadline(make_flow, monkeypatch):
    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

    async def hang(*a, **k):
        await asyncio.sleep(30)
    monkeypatch.setattr(sam, "make_client", lambda timeout=0: Client())
    monkeypatch.setattr(sam, "segment_video", hang)
    f = make_flow(); f.cfg["monitor"]["sam_deadline_s"] = 0.05
    t0 = time.time()
    with pytest.raises(asyncio.TimeoutError):
        f.call_sam("clip.mp4")
    assert time.time() - t0 < 2


def test_a_floor_event_is_not_a_caution(make_flow, monkeypatch):
    monkeypatch.setattr(remy.checkin, "ask_once", lambda *a, **k: {"status": "okay", "transcript": {"transcript": "fine"}})
    f = make_flow("floor_still", recheck_after_okay=lambda wa, snap: None); f.run()
    assert f.result["alert"]["kind"] == "calm"


# ---- the send: a refused message is not a sent message
def real_wa(monkeypatch, script):
    monkeypatch.setenv("WHATSAPP_TOKEN", "t"); monkeypatch.setenv("WHATSAPP_PHONE_NUMBER_ID", "1")
    monkeypatch.setattr(wa_mod, "RETRY_WAITS_S", (0, 0))
    wa = WhatsApp(); wa.cfg = {**wa.cfg, "person_name": "Nani"}; wa.channel = "whatsapp"; wa.dry = False
    wa.log_path = wa_mod.Path(monkeypatch._tmp) / "alerts.log"
    wa.recipients = lambda: [{"name": "x", "number": "+1"}]
    calls = []

    def post(path, body=None, files=None, data=None):
        calls.append(path if body is None else body.get("type"))
        r = script.pop(0) if script else REFUSED
        if isinstance(r, Exception):
            raise r
        return r
    wa._post = post
    return wa, calls


@pytest.fixture
def tmpmp(monkeypatch, tmp_path):
    monkeypatch._tmp = tmp_path
    return monkeypatch


def test_delivered_means_a_2xx_or_a_dry_run_for_everyone():
    assert delivered([OK]) and delivered([{"dry": True}]) and delivered([{"_http": 201}])
    assert not delivered([OK, REFUSED]) and not delivered([]) and not delivered([{"_http": None}])


def test_a_refused_video_falls_back_to_text(tmpmp, tmp_path):
    clip = tmp_path / "a.mp4"; clip.write_bytes(b"x")
    wa, calls = real_wa(tmpmp, [{"id": "m", "_http": 200}, REFUSED, OK])
    assert delivered(wa.notice("hello", video=clip)) and calls == ["media", "video", "text"]


def test_network_errors_are_retried_and_never_raise(tmpmp):
    wa, calls = real_wa(tmpmp, [requests.ConnectionError("down"), requests.ConnectionError("down"), OK])
    assert delivered(wa.notice("hello")) and calls == ["text", "text", "text"]


def test_a_send_that_always_fails_reports_it(tmpmp):
    wa, calls = real_wa(tmpmp, [])
    resp = wa.notice("hello")
    assert not delivered(resp) and len(calls) == 3


def test_an_expired_token_is_found_at_startup(tmpmp):
    wa, _ = real_wa(tmpmp, [])
    tmpmp.setattr(requests, "get", lambda *a, **k: SimpleNamespace(status_code=401, text="expired"))
    assert "expired" in wa.token_problem()
    tmpmp.setattr(requests, "get", lambda *a, **k: SimpleNamespace(status_code=200, text="{}"))
    assert wa.token_problem() is None

    def offline(*a, **k):
        raise requests.ConnectionError("no network")
    tmpmp.setattr(requests, "get", offline)
    assert wa.token_problem() is None                          # a network blip must not stop the monitor starting


# ---- do not tell her the family knows when they do not
@pytest.mark.parametrize("status,line", [("needs_help", "ack_sent"), ("okay", "ack_okay")])
def test_she_is_told_the_family_knows_only_if_they_do(make_flow, monkeypatch, status, line):
    monkeypatch.setattr(remy.checkin, "ask_once", lambda *a, **k: {"status": status, "transcript": {"transcript": "x"}})
    f = make_flow("floor_still", recheck_after_okay=lambda wa, snap: None); f.run()
    assert f.result["said_lines"] == [line] and f.result["alert"]["delivered"] is True
    FakeWA.answer = REFUSED
    f = make_flow("floor_still", recheck_after_okay=lambda wa, snap: None); f.run()
    assert "said_lines" not in f.result and f.result["alert"]["delivered"] is False


def test_the_log_says_when_the_family_was_not_told(make_flow, monkeypatch, capsys):
    FakeWA.answer = REFUSED
    monkeypatch.setattr(remy.checkin, "ask_once", lambda *a, **k: {"status": "needs_help", "transcript": {"transcript": "x"}})
    make_flow("floor_still").run()
    assert "ALERT FAILED" in capsys.readouterr().out


# ---- a broken speaker or mic is not "she did not respond"
@pytest.mark.parametrize("broken", [{"speak": {"ok": False}}, {"record": {"ok": False}}])
def test_a_broken_speaker_or_mic_changes_the_reason(make_flow, monkeypatch, broken):
    monkeypatch.setattr(remy.checkin, "ask_once", lambda *a, **k: {"status": "no_answer", "transcript": {"transcript": ""}, **broken})
    f = make_flow("floor_still"); f.run()
    assert "speaker or microphone" in f.result["alert"]["reason"]


def test_a_real_silence_still_says_she_did_not_respond(make_flow, monkeypatch):
    monkeypatch.setattr(remy.checkin, "ask_once", lambda *a, **k: {"status": "no_answer", "transcript": {"transcript": ""},
                                                                   "speak": {"ok": True}, "record": {"ok": True}})
    f = make_flow("floor_still"); f.run()
    assert "did not respond" in f.result["alert"]["reason"]


# ---- the camera
class Stream:
    plan: list = []

    def __init__(self, url, max_width=None):
        good = Stream.plan.pop(0) if Stream.plan else None
        if good is None:
            raise RuntimeError("no video stream")
        self.left = good; self.src_fps = 15.0

    def read(self):
        if self.left > 0:
            self.left -= 1
            return True, np.zeros((8, 8, 3), np.uint8)
        return False, None

    def close(self):
        pass


def clocked_capture(monkeypatch, plan, **kw):
    monkeypatch.setattr(cap_mod, "FFmpegFrames", Stream)
    monkeypatch.setattr(cap_mod, "tapo_rtsp", lambda s=2: "rtsp://x/stream2", raising=False)
    Stream.plan = plan
    seen = []
    c = Capture("rtsp://x/stream2", log=lambda m: None, on_outage=lambda kind, s: seen.append((kind, s)), **kw)
    now = [0.0]
    c._now = lambda: now[0]
    c._sleep = lambda s: now.__setitem__(0, now[0] + s)
    return c, seen


def test_a_long_outage_tells_the_family_once_and_then_that_it_is_back(monkeypatch):
    c, seen = clocked_capture(monkeypatch, [1, None, None, None, None, None, None, 5], outage_notice_s=60, max_outage_s=600)
    assert c.read()[0] and c.read()[0]
    assert [k for k, _ in seen] == ["down", "back"]


def test_a_short_blip_tells_nobody(monkeypatch):
    c, seen = clocked_capture(monkeypatch, [1, 5], outage_notice_s=60)
    assert c.read()[0] and c.read()[0] and seen == []


def test_giving_up_is_reported(monkeypatch):
    c, seen = clocked_capture(monkeypatch, [1], outage_notice_s=60, max_outage_s=120, backoff_max_s=30)
    assert c.read()[0] and c.read()[0] is False
    assert [k for k, _ in seen] == ["down", "gave_up"]


def test_a_broken_notice_does_not_stop_the_reconnect(monkeypatch):
    c, _ = clocked_capture(monkeypatch, [1, None, None, None, 5], outage_notice_s=1)

    def boom(kind, s):
        raise SystemExit("no token")
    c.on_outage = boom
    assert c.read()[0] and c.read()[0]


def test_the_camera_notices_name_her_and_the_time():
    assert "stopped watching Nani" in monitor.camera_notice_text("gave_up", "Nani", 600)
    assert "10 min" in monitor.camera_notice_text("gave_up", "Nani", 600)
    assert "back" in monitor.camera_notice_text("back", "Nani", 180) and "Nani" in monitor.camera_notice_text("down", "Nani", 120)
