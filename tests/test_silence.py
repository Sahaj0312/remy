"""The silence stop on synthetic audio shaped like the real recordings: gated floor at -72 dBFS, rustle at
-45, speech at -25."""
import numpy as np

from remy.camera.silence import SilenceStop, cut_point
from remy.checkin import classify

RATE = 16000


def tone(seconds, db, freq=220.0):
    t = np.arange(int(RATE * seconds)) / RATE
    return (10 ** (db / 20)) * np.sqrt(2) * np.sin(2 * np.pi * freq * t)


def noise(seconds, db):
    rng = np.random.default_rng(0)
    return (10 ** (db / 20)) * rng.standard_normal(int(RATE * seconds))


def test_silence_runs_to_the_end():
    r = cut_point(noise(15.0, -72), RATE)
    assert r["speech_at_s"] is None and r["stopped_early"] is False and r["duration_s"] == 15.0


def test_a_yeah_stops_the_recording_two_seconds_after_it():
    x = np.concatenate([noise(3.5, -72), tone(0.4, -25), noise(11.1, -72)])
    r = cut_point(x, RATE)
    assert r["speech_at_s"] is not None and 3.4 <= r["speech_at_s"] <= 3.6
    assert r["stopped_early"] and 5.8 <= r["duration_s"] <= 6.1       # 3.9 s end of speech + 2.0 s silence


def test_never_stops_before_min_s():
    x = np.concatenate([tone(0.4, -25), noise(14.6, -72)])
    r = cut_point(x, RATE)
    assert r["stopped_early"] and r["duration_s"] == 4.0


def test_a_short_quiet_transient_is_not_speech():
    x = np.concatenate([tone(0.1, -38), noise(14.9, -72)])
    r = cut_point(x, RATE)
    assert r["speech_at_s"] is None and r["stopped_early"] is False


def test_a_short_loud_word_counts_as_speech():
    x = np.concatenate([noise(3.6, -72), tone(0.15, -25), noise(11.25, -72)])
    r = cut_point(x, RATE)
    assert r["speech_at_s"] is not None and 3.5 <= r["speech_at_s"] <= 3.7
    assert r["stopped_early"] and 5.6 <= r["duration_s"] <= 5.9


def test_rustle_after_speech_keeps_listening():
    x = np.concatenate([noise(1.0, -72), tone(0.4, -25), noise(5.0, -45), noise(8.6, -72)])
    r = cut_point(x, RATE)
    assert r["speech_at_s"] is not None
    assert r["duration_s"] >= 8.4                                     # rustle at -45 is not silence; stop ~2 s after it ends


def test_incremental_feed_matches_the_offline_cut():
    x = np.concatenate([noise(2.0, -72), tone(0.5, -30), noise(12.5, -72)])
    det = SilenceStop(0.05)
    hop = int(RATE * 0.05); stopped_at = None
    for i in range(0, len(x) - hop + 1, hop):
        if det.feed(x[i:i + hop]):
            stopped_at = det.t; break
    assert round(stopped_at, 2) == cut_point(x, RATE)["duration_s"]


def test_heard_speech_without_words_is_unclear_not_no_answer(cfg):
    ck = cfg["checkin"]
    assert classify("", ck, "q", backend=None, heard_speech=True)["status"] == "unclear"
    assert classify("", ck, "q", backend=None, heard_speech=False)["status"] == "no_answer"
    assert classify("", {**ck, "escalate_on_unheard_speech": False}, "q", backend=None, heard_speech=True)["status"] == "no_answer"
