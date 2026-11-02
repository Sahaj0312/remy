"""The Tapo microphone: record N seconds from the RTSP audio track as mono 16 kHz PCM WAV (what the ASR accepts).

The camera sends 8 kHz A-law, so the resample adds no detail. ffmpeg does the pull because Python sockets
cannot reach LAN hosts on a Mac with Local Network privacy on (see docs/hardware-setup.md).
"""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

from remy.common import tapo_rtsp


def record_mic(seconds: float, dst: Path, stream: int = 1) -> dict:
    t0 = time.time()
    r = subprocess.run(["ffmpeg", "-y", "-v", "error", "-rtsp_transport", "tcp", "-i", tapo_rtsp(stream), "-t", str(seconds),
                        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", "-map_metadata", "-1", str(dst)],
                       capture_output=True, text=True, timeout=seconds + 60)
    return {"ok": r.returncode == 0 and dst.exists(), "took_s": round(time.time() - t0, 2), "err": r.stderr.strip()[:200]}
