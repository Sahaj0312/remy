"""The check-in records from the monitor's open room-audio stream instead of opening a second camera
connection: the Tapo C200 serves three RTSP sessions at a time, and a refused fourth one made a live
check-in hear nothing (Nov 3). A shared stream is left open; a dead one is replaced for the round."""
import numpy as np
import pytest

import remy.checkin as checkin
from remy.camera.audio_in import record_until_silence

RATE = 16000
CHUNK = RATE // 20          # 50 ms, what MicStream delivers


def chunks(seconds, db):
    amp = 10 ** (db / 20) * np.sqrt(2)
    t = np.arange(CHUNK) / RATE
    return [(amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32) for _ in range(int(seconds * 20))]


class FakeMic:
    """Yields its chunks once, then reports the stream as ended."""

    def __init__(self, parts=None, alive=True, err=""):
        self.parts = parts or []
        self.alive = alive
        self.err = err
        self.first_wall = 1.0 if alive else None
        self.closed = False

    def wait_ready(self, timeout=5.0):
        return self.first_wall is not None

    def chunks_since(self, since_wall, until_wall=None):
        for x in self.parts:
            yield since_wall, x
        self.alive = False

    def close(self):
        self.closed = True


def test_speech_then_silence_stops_early_and_leaves_a_shared_mic_open(tmp_path, cfg):
    mic = FakeMic(chunks(0.6, -25) + chunks(4.0, -70))
    r = record_until_silence(tmp_path / "a.wav", 15.0, cfg["checkin"]["silence"], mic=mic, since_wall=0.0)
    assert r["ok"] and r["stopped_early"] and r["speech_at_s"] is not None
    assert (tmp_path / "a.wav").exists() and mic.closed is False


def test_a_dead_mic_reports_the_error_and_writes_nothing(tmp_path, cfg):
    mic = FakeMic(alive=False, err="Error opening input: Operation not permitted")
    r = record_until_silence(tmp_path / "b.wav", 15.0, cfg["checkin"]["silence"], mic=mic, since_wall=0.0)
    assert r["ok"] is False and "not permitted" in r["err"] and not (tmp_path / "b.wav").exists()


@pytest.fixture
def quiet_round(monkeypatch, tmp_path):
    """ask_once with no camera, speaker, Kokoro or ASR: returns a backend that says okay."""
    import remy.camera.audio_out as audio_out
    prompt = tmp_path / "ask.wav"; prompt.write_bytes(b"RIFF")
    monkeypatch.setattr(checkin, "prompt_path", lambda key, lang: prompt)
    monkeypatch.setattr(audio_out, "speak", lambda wav, out, wait=True: {"device": "fake", "took_s": 0.0})
    monkeypatch.setattr(checkin, "transcribe_wav", lambda p: {"transcript": "yeah I'm okay", "latency_s": 0.0})
    monkeypatch.setattr(checkin, "record_mic", lambda s, dst: pytest.fail("the plain recording must not be reached"))

    class Okay:
        def classify_reply(self, t, q):
            return {"status": "okay", "quote": t, "reason": "fake", "rule": False, "latency_s": 0.0, "cost_usd": 0.0}
    return Okay()


def test_ask_once_records_from_the_shared_mic_and_leaves_it_open(tmp_path, cfg, quiet_round, monkeypatch):
    shared = FakeMic(chunks(0.6, -25) + chunks(4.0, -70))
    monkeypatch.setattr(checkin, "MicStream", lambda *a, **k: pytest.fail("a second camera connection was opened"))
    r = checkin.ask_once(cfg["checkin"], "laptop", tmp_path, "r1", log=lambda s: None, backend=quiet_round, mic=shared)
    assert r["status"] == "okay" and r["record"]["ok"] and shared.closed is False


def test_ask_once_replaces_a_dead_shared_mic_for_the_round(tmp_path, cfg, quiet_round, monkeypatch):
    dead = FakeMic(alive=False, err="Operation not permitted")
    fresh = FakeMic(chunks(0.6, -25) + chunks(4.0, -70))
    monkeypatch.setattr(checkin, "MicStream", lambda *a, **k: fresh)
    lines = []
    r = checkin.ask_once(cfg["checkin"], "laptop", tmp_path, "r1", log=lines.append, backend=quiet_round, mic=dead)
    assert r["record"]["ok"] and fresh.closed is True and dead.closed is False
    assert any("room audio stream is down" in l for l in lines)
