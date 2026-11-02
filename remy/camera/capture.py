"""One frame source for the monitor: a Tapo/RTSP/HTTP stream through ffmpeg, a webcam, or a video file.

A camera stream drops now and then (Wi-Fi, a camera reboot, the router). For a safety monitor that must not
be the end of the run, so a failed read on a stream reopens it with a growing back-off (2, 4, 8 ... 60 s) and
keeps going; the caller sees a `take_reconnected()` flag and resets its short-term state. A file ends when it
ends. After `max_outage_s` without a frame the read returns False and the monitor exits on purpose, so a
supervisor (launchd, systemd) can restart it and the gap is in the log.
"""
from __future__ import annotations

import time
from pathlib import Path

from remy.camera.rtsp import FFmpegFrames


def kind_of(source: str) -> str:
    """'file' | 'webcam' | 'rtsp'."""
    if source == "tapo" or source.startswith(("rtsp://", "rtsps://", "http://", "https://")):
        return "rtsp"
    if source.isdigit():
        return "webcam"
    return "file"


class Capture:
    def __init__(self, source: str, max_width: int = 640, reconnect: bool = True, max_outage_s: float = 600.0,
                 backoff_max_s: float = 60.0, log=print):
        self.source = source
        self.kind = kind_of(source)
        self.max_width = max_width
        self.reconnect = reconnect and self.kind == "rtsp"
        self.max_outage_s = max_outage_s
        self.backoff_max_s = backoff_max_s
        self.log = log
        self.reconnects = 0
        self._reconnected = False
        self._sleep = time.sleep          # swapped in tests
        self._now = time.time
        self._reader = None
        self.src_fps = 0.0
        self.is_file = self.kind == "file"
        self._open(first=True)

    # ---- opening
    def _url(self) -> str:
        if self.source == "tapo":
            from remy.common import tapo_rtsp
            return tapo_rtsp(2)
        return self.source

    def _open(self, first: bool = False) -> None:
        if self.kind == "rtsp":
            rd = FFmpegFrames(self._url(), max_width=self.max_width)
            self.src_fps = rd.src_fps if 0 < rd.src_fps <= 60 else 15.0
            self._reader = rd
            return
        import cv2
        if self.kind == "file" and not Path(self.source).exists():
            raise SystemExit(f"cannot open source {self.source}: no such file")
        cap = cv2.VideoCapture(int(self.source) if self.kind == "webcam" else self.source)
        if not cap.isOpened():
            raise SystemExit(f"cannot open source {self.source}")
        self.src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._reader = cap

    def _close_reader(self) -> None:
        rd, self._reader = self._reader, None
        if rd is None:
            return
        try:
            rd.close() if hasattr(rd, "close") else rd.release()
        except Exception:
            pass

    # ---- reading
    def read(self):
        """(ok, frame). On a stream, a failed read reconnects until max_outage_s has passed."""
        if self._reader is not None:
            ok, frame = self._reader.read()
            if ok:
                return True, frame
        if not self.reconnect:
            return False, None
        return self._reconnect_loop()

    def _reconnect_loop(self):
        t_lost = self._now()
        delay = 2.0
        self.log(f"stream lost, reconnecting (will give up after {self.max_outage_s:.0f}s)")
        while True:
            self._close_reader()
            if self._now() - t_lost >= self.max_outage_s:
                self.log(f"stream still down after {self.max_outage_s:.0f}s; giving up")
                return False, None
            self._sleep(delay)
            try:
                self._open()
                ok, frame = self._reader.read()
            except Exception as e:
                ok, frame = False, None
                self.log(f"reconnect failed: {type(e).__name__}: {str(e)[:120]}")
            if ok:
                self.reconnects += 1
                self._reconnected = True
                self.log(f"stream back after {self._now() - t_lost:.0f}s (reconnect #{self.reconnects})")
                return True, frame
            delay = min(delay * 2, self.backoff_max_s)

    def take_reconnected(self) -> bool:
        """True once after each successful reconnect; the caller resets its tracks and buffer."""
        flag, self._reconnected = self._reconnected, False
        return flag

    def release(self) -> None:
        self._close_reader()
