"""Read video frames from an RTSP URL (or a file) through an ffmpeg subprocess.

Why: on this Mac, Python's own sockets are blocked from LAN hosts by the macOS Local Network
privacy setting (EHOSTUNREACH), while ffmpeg is allowed. Piping raw BGR frames from ffmpeg also
gives a predictable low-latency decode path for the live pipeline.

    for t_wall, frame in FFmpegFrames(url, max_width=640):   # frame: HxWx3 uint8 BGR
        ...
"""
from __future__ import annotations

import json
import subprocess
import time
from typing import Iterator

import numpy as np


def probe(url: str, timeout: float = 30) -> dict:
    cmd = ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,codec_name,profile,width,height,r_frame_rate,sample_rate,channels",
           "-of", "json"]
    if url.startswith("rtsp"):
        cmd += ["-rtsp_transport", "tcp"]
    out = subprocess.run(cmd + [url], capture_output=True, text=True, timeout=timeout).stdout
    return json.loads(out) if out else {}


class FFmpegFrames:
    def __init__(self, url: str, max_width: int | None = None, fps: float | None = None, low_delay: bool = True,
                 realtime_file: bool = False):
        self.url = url
        info = probe(url)
        vs = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), None)
        if vs is None:
            raise RuntimeError(f"no video stream in {url}")
        self.src_w, self.src_h = int(vs["width"]), int(vs["height"])
        num, den = vs.get("r_frame_rate", "0/1").split("/")
        self.src_fps = float(num) / float(den) if float(den) else 0.0
        self.audio = next((s for s in info.get("streams", []) if s.get("codec_type") == "audio"), None)
        self.w, self.h = self.src_w, self.src_h
        if max_width and self.src_w > max_width:
            self.w = max_width
            self.h = int(round(self.src_h * max_width / self.src_w / 2)) * 2
        vf = []
        if fps:
            vf.append(f"fps={fps:g}")
        if (self.w, self.h) != (self.src_w, self.src_h):
            vf.append(f"scale={self.w}:{self.h}")
        cmd = ["ffmpeg", "-v", "error", "-nostdin"]
        if url.startswith("rtsp"):
            cmd += ["-rtsp_transport", "tcp"]
            if low_delay:
                cmd += ["-fflags", "nobuffer", "-flags", "low_delay", "-probesize", "500000", "-analyzeduration", "0"]
        elif realtime_file:
            cmd += ["-re"]
        cmd += ["-i", url, "-an"]
        if vf:
            cmd += ["-vf", ",".join(vf)]
        cmd += ["-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
        self.cmd = cmd
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=self.w * self.h * 3 * 4)
        self.t_start = time.time()
        self.n = 0

    def __iter__(self) -> Iterator[tuple[float, np.ndarray]]:
        size = self.w * self.h * 3
        while True:
            buf = self.proc.stdout.read(size)
            if len(buf) < size:
                break
            self.n += 1
            yield time.time(), np.frombuffer(buf, np.uint8).reshape(self.h, self.w, 3)
        self.close()

    def read(self):
        size = self.w * self.h * 3
        buf = self.proc.stdout.read(size)
        if len(buf) < size:
            return False, None
        self.n += 1
        return True, np.frombuffer(buf, np.uint8).reshape(self.h, self.w, 3)

    def close(self):
        if self.proc.poll() is None:
            self.proc.kill()
        try:
            self.proc.wait(timeout=5)
        except Exception:
            pass

    def stderr(self) -> str:
        try:
            return self.proc.stderr.read().decode(errors="replace") if self.proc.poll() is not None else ""
        except Exception:
            return ""


class FFmpegAudio:
    """Mono 16-bit PCM chunks from the RTSP audio track via ffmpeg."""

    def __init__(self, url: str, rate: int = 16000, chunk_ms: int = 50):
        self.rate, self.chunk = rate, int(rate * chunk_ms / 1000)
        cmd = ["ffmpeg", "-v", "error", "-nostdin", "-rtsp_transport", "tcp", "-fflags", "nobuffer", "-flags", "low_delay",
               "-i", url, "-vn", "-ac", "1", "-ar", str(rate), "-f", "s16le", "-"]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=self.chunk * 2 * 8)

    def __iter__(self):
        size = self.chunk * 2
        while True:
            buf = self.proc.stdout.read(size)
            if len(buf) < size:
                break
            yield time.time(), np.frombuffer(buf, np.int16).astype(np.float32) / 32768

    def close(self):
        if self.proc.poll() is None:
            self.proc.kill()
