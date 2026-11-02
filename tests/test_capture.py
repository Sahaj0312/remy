"""Capture reconnects a dropped stream with back-off and gives up after max_outage_s. FFmpegFrames is
replaced by a scripted fake, so no camera is needed."""
import numpy as np
import pytest

from remy.camera import capture as cap_mod
from remy.camera.capture import Capture, kind_of

FRAME = np.zeros((8, 8, 3), np.uint8)


class FakeStream:
    """Each instance yields `good` frames then fails. `plan` is a list consumed per construction;
    an entry of None makes the constructor raise (camera unreachable)."""
    plan: list = []
    made = 0

    def __init__(self, url, max_width=None):
        FakeStream.made += 1
        good = FakeStream.plan.pop(0) if FakeStream.plan else 0
        if good is None:
            raise RuntimeError("no video stream")
        self.left = good
        self.src_fps = 15.0

    def read(self):
        if self.left > 0:
            self.left -= 1
            return True, FRAME
        return False, None

    def close(self):
        pass


@pytest.fixture
def fake(monkeypatch):
    monkeypatch.setattr(cap_mod, "FFmpegFrames", FakeStream)
    FakeStream.made = 0
    return FakeStream


def make(plan, **kw):
    FakeStream.plan = list(plan)
    c = Capture("rtsp://cam/stream2", log=lambda *_: None, **kw)
    slept = []
    c._sleep = slept.append
    clock = {"t": 1000.0}
    c._now = lambda: clock["t"]
    return c, slept, clock


def test_kind_of():
    assert kind_of("tapo") == "rtsp" and kind_of("rtsp://u:p@1.2.3.4:554/s") == "rtsp"
    assert kind_of("0") == "webcam" and kind_of("eval/clips/x.mp4") == "file"


def test_reads_frames_then_reconnects_once(fake):
    c, slept, _ = make([2, 3])
    assert c.read()[0] and c.read()[0]
    ok, frame = c.read()                    # third read fails on stream #1, stream #2 answers
    assert ok and frame is FRAME
    assert c.reconnects == 1 and fake.made == 2
    assert slept == [2.0]
    assert c.take_reconnected() is True and c.take_reconnected() is False


def test_backoff_doubles_and_an_unreachable_camera_is_retried(fake):
    c, slept, _ = make([1, None, None, 5])
    assert c.read()[0]
    assert c.read()[0]                      # two failed constructions, then success
    assert slept == [2.0, 4.0, 8.0]
    assert c.reconnects == 1


def test_gives_up_after_max_outage(fake):
    c, slept, clock = make([1, None, None, None], max_outage_s=10.0)
    c._sleep = lambda s: (slept.append(s), clock.__setitem__("t", clock["t"] + s))
    assert c.read()[0]
    ok, frame = c.read()
    assert ok is False and frame is None
    assert sum(slept) >= 10.0


def test_a_file_source_does_not_reconnect(tmp_path, monkeypatch):
    import cv2
    p = tmp_path / "clip.mp4"
    w = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
    for _ in range(5):
        w.write(np.zeros((48, 64, 3), np.uint8))
    w.release()
    if not p.exists() or p.stat().st_size == 0:
        pytest.skip("OpenCV cannot write mp4 here")
    c = Capture(str(p), log=lambda *_: None)
    assert c.is_file and c.kind == "file" and c.reconnect is False
    n = 0
    while c.read()[0]:
        n += 1
    assert n == 5 and c.reconnects == 0
