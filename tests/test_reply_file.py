"""The no-camera path: `remy run --reply` answers the check-in from a file, and a fresh clone has what it needs to
speak (examples/voice). A canned answer must never stand in for a real person, so it is limited to video files."""
import pytest

import remy.camera.audio_out as audio_out
import remy.checkin as checkin
import remy.checkin.tts as tts
from remy import monitor
from remy.camera.audio_in import reply_from_file
from remy.common import ROOT

REPLIES = ROOT / "examples" / "replies"
CLIP = ROOT / "examples" / "clips" / "gmdcsa_S1_Fall_12.mp4"


@pytest.mark.parametrize("name,hears_speech", [("okay", True), ("help", True), ("silent", False)])
def test_the_sample_replies_read_as_speech_or_silence(tmp_path, cfg, name, hears_speech):
    r = reply_from_file(REPLIES / f"{name}.wav", tmp_path / "r.wav", 15, cfg["checkin"]["silence"])
    assert r["ok"] and (r["speech_at_s"] is not None) == hears_speech
    assert (tmp_path / "r.wav").stat().st_size > 16000     # 16 kHz mono, at least half a second


def test_an_unreadable_reply_is_a_failed_recording_not_a_crash(tmp_path, cfg):
    r = reply_from_file(tmp_path / "nope.wav", tmp_path / "r.wav", 15, cfg["checkin"]["silence"])
    assert r["ok"] is False and r["err"] and r["speech_at_s"] is None


def test_a_reply_round_opens_no_microphone(tmp_path, cfg, monkeypatch):
    def no_mic(*a, **k):
        raise AssertionError("a reply round must not touch the camera")
    monkeypatch.setattr(checkin, "MicStream", no_mic)
    monkeypatch.setattr(checkin, "record_mic", no_mic)
    monkeypatch.setattr(audio_out, "speak", lambda *a, **k: {"device": "stub", "ok": True, "took_s": 0.0})
    monkeypatch.setattr(checkin, "transcribe_wav", lambda wav: {"transcript": "help me", "latency_s": 0})
    r = checkin.ask_once(cfg["checkin"], "laptop", tmp_path, "t", reply=REPLIES / "help.wav")
    assert r["status"] == "needs_help" and r["record"]["speech_at_s"] is not None


def test_a_fresh_clone_has_every_line_and_local_recordings_win(tmp_path, cfg, monkeypatch):
    monkeypatch.setattr(tts, "AUDIO_DIR", tmp_path)
    for key in tts.LINE_KEYS:
        assert tts.prompt_path(key, "en") == tts.SHIPPED_VOICE_DIR / f"{key}_en.wav" and tts.prompt_path(key, "en").exists()
    (tmp_path / "ask_en.wav").write_bytes(b"x")
    assert tts.prompt_path("ask", "en") == tmp_path / "ask_en.wav"
    assert tts.prompt_path("ask", "hi") == tmp_path / "ask_hi.wav"    # not shipped: make_prompts will be asked for it


@pytest.mark.parametrize("source", ["tapo", "0", "rtsp://camera/stream1"])
def test_reply_is_refused_on_a_live_source(source, capsys):
    with pytest.raises(SystemExit) as e:
        monitor.main(["--source", source, "--reply", str(REPLIES / "okay.wav")])
    assert e.value.code == 2 and "only with a video file" in capsys.readouterr().err


def test_a_missing_reply_file_stops_at_startup(capsys):
    with pytest.raises(SystemExit) as e:
        monitor.main(["--source", str(CLIP), "--reply", "no_such.wav"])
    assert e.value.code == 2 and "no such file" in capsys.readouterr().err


def test_no_send_credentials_stop_a_live_run_at_startup(monkeypatch, capsys):
    import remy.alerts

    class NoToken:
        def __init__(self, *a, **k):
            raise SystemExit("WHATSAPP_TOKEN is not set.")
    monkeypatch.setattr(monitor, "api_key", lambda: "k")
    monkeypatch.setattr(remy.alerts, "WhatsApp", NoToken)
    with pytest.raises(SystemExit) as e:
        monitor.main(["--source", str(CLIP)])
    assert e.value.code == 2 and "--alerts-dry-run" in capsys.readouterr().err


def test_laptop_speech_without_a_player_reports_it(tmp_path, monkeypatch):
    monkeypatch.setattr(audio_out.shutil, "which", lambda name: None)
    r = audio_out.speak(tmp_path / "a.wav", "laptop")
    assert r["ok"] is False and "no audio player" in r["detail"]
