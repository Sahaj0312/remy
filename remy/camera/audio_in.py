"""The Tapo microphone as mono 16 kHz PCM WAV (what the ASR accepts): a fixed N seconds, or until she has
answered (remy.camera.silence).

The camera sends 8 kHz A-law, so the resample adds no detail. ffmpeg does the pull because Python sockets
cannot reach LAN hosts on a Mac with Local Network privacy on (see docs/hardware-setup.md).
"""
from __future__ import annotations

import subprocess
import threading
import time
import wave
from collections import deque
from pathlib import Path

import numpy as np

from remy.camera.rtsp import FFmpegAudio
from remy.camera.silence import SilenceStop
from remy.common import tapo_rtsp


def record_mic(seconds: float, dst: Path, stream: int = 1) -> dict:
    t0 = time.time()
    r = subprocess.run(["ffmpeg", "-y", "-v", "error", "-rtsp_transport", "tcp", "-i", tapo_rtsp(stream), "-t", str(seconds),
                        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", "-map_metadata", "-1", str(dst)],
                       capture_output=True, text=True, timeout=seconds + 60)
    return {"ok": r.returncode == 0 and dst.exists(), "took_s": round(time.time() - t0, 2), "err": r.stderr.strip()[:200]}


class MicStream:
    """The camera mic as a background stream of (wall time, 50 ms chunk). Opened before the prompt is spoken,
    so the 2 s the RTSP connect takes is not paid after the question; the recorder then takes only the chunks
    that arrived after a given moment."""

    def __init__(self, stream: int = 1, rate: int = 16000, keep_s: float = 30.0):
        self.rate = rate
        self.au = FFmpegAudio(tapo_rtsp(stream), rate=rate, chunk_ms=50)
        self.buf: deque = deque(maxlen=int(keep_s / 0.05))
        self.err = ""
        self.alive = True
        self.first_wall = None
        self._th = threading.Thread(target=self._reader, daemon=True)
        self._th.start()

    def _reader(self):
        try:
            for wall, x in self.au:
                if self.first_wall is None:
                    self.first_wall = wall
                self.buf.append((wall, x))
        except Exception as e:
            self.err = f"{type(e).__name__}: {e}"[:200]
        finally:
            self.alive = False

    def wait_ready(self, timeout: float = 5.0) -> bool:
        t0 = time.time()
        while self.first_wall is None and self.alive and time.time() - t0 < timeout:
            time.sleep(0.05)
        return self.first_wall is not None

    def chunks_since(self, since_wall: float):
        """Yield chunks that arrived at or after since_wall, then keep yielding live ones."""
        seen = 0
        while True:
            items = list(self.buf)
            while seen < len(items):
                wall, x = items[seen]; seen += 1
                if wall >= since_wall:
                    yield wall, x
            if not self.alive:
                return
            time.sleep(0.02)
            # the deque may have dropped old items; re-sync by wall time
            items2 = list(self.buf)
            if items2 and items and items2[0][0] > items[0][0]:
                seen = sum(1 for w, _ in items2 if w < (items[seen - 1][0] if seen else since_wall)) + (1 if seen else 0)
                seen = min(seen, len(items2))

    def slice(self, w0: float, w1: float, offset_s: float = 0.0) -> np.ndarray:
        """Samples whose chunks arrived between wall times w0 and w1 (shifted by offset_s): the room's sound
        for a clip cut from the frame buffer over the same wall-clock span."""
        parts = [x for wall, x in list(self.buf) if w0 + offset_s <= wall < w1 + offset_s]
        return np.concatenate(parts) if parts else np.zeros(0, np.float32)

    def close(self):
        self.au.close()


def record_until_silence(dst: Path, max_s: float, params: dict | None = None, stream: int = 1, rate: int = 16000,
                         mic: MicStream | None = None, since_wall: float | None = None) -> dict:
    """Record from the camera mic and stop once speech has been followed by silence (see remy.camera.silence).
    With `mic` (an already-open MicStream) the recording starts from the chunks that arrived after `since_wall`,
    so no connect time is spent after the prompt. Returns record_mic's keys plus duration_s, speech_at_s,
    stopped_early, peak_db."""
    t0 = time.time()
    det = SilenceStop(0.05, **{**(params or {}), "max_s": max_s})
    chunks: list[np.ndarray] = []
    own = mic is None
    if own:
        mic = MicStream(stream, rate)
        since_wall = 0.0
    err = ""
    try:
        if not mic.wait_ready():
            err = mic.err or "no audio from the camera"
        else:
            for _, x in mic.chunks_since(since_wall if since_wall is not None else t0):
                chunks.append(x)
                if det.feed(x):
                    break
            if mic.err and not chunks:
                err = mic.err
    except Exception as e:
        err = f"{type(e).__name__}: {e}"[:200]
    finally:
        if own:
            mic.close()
    if chunks:
        pcm = (np.clip(np.concatenate(chunks), -1, 1) * 32767).astype(np.int16)
        dst.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(dst), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate); w.writeframes(pcm.tobytes())
    ok = bool(chunks) and dst.exists() and det.t >= 1.0
    return {"ok": ok, "took_s": round(time.time() - t0, 2), "err": err if not ok else "", **det.summary()}
