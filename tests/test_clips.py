import shutil

import numpy as np
import pytest

from remy.camera.clips import concat_clips, freeze_clip, probe, write_clip

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
RATE = 16000


def frames(seconds, fps=10.0, w=64, h=48, shade=0):
    return [(k / fps, np.full((h, w, 3), shade, np.uint8)) for k in range(int(seconds * fps))]


def tone(seconds, db=-25.0):
    t = np.arange(int(RATE * seconds)) / RATE
    return (10 ** (db / 20)) * np.sin(2 * np.pi * 220 * t)


def test_clip_with_audio_has_both_streams(tmp_path):
    p = write_clip(frames(3.0), 10.0, tmp_path / "a.mp4", audio=tone(3.0), rate=RATE)
    info = probe(p)
    assert info["has_audio"] and info["width"] == 64 and 2.8 <= info["duration_s"] <= 3.3


def test_clip_without_audio_gets_a_silent_track(tmp_path):
    p = write_clip(frames(2.0), 10.0, tmp_path / "s.mp4")
    info = probe(p)
    assert info["has_audio"] and 1.8 <= info["duration_s"] <= 2.3


def test_freeze_frame_lasts_as_long_as_the_audio(tmp_path):
    p = freeze_clip(np.zeros((48, 64, 3), np.uint8), tone(4.0), tmp_path / "f.mp4")
    assert 3.8 <= probe(p)["duration_s"] <= 4.3


def test_concat_joins_clips_of_different_sizes(tmp_path):
    a = write_clip(frames(2.0, w=64, h=48), 10.0, tmp_path / "a.mp4", audio=tone(2.0))
    b = write_clip(frames(3.0, w=96, h=72, shade=200), 10.0, tmp_path / "b.mp4")
    out = concat_clips([a, b], tmp_path / "ab.mp4")
    info = probe(out)
    assert info["has_audio"] and info["width"] == 64 and 4.7 <= info["duration_s"] <= 5.4
