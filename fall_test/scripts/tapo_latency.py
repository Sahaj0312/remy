"""Audio round-trip latency of the Tapo stream: play a beep on this Mac at a known time and
detect it in the camera's RTSP audio. Reports how far the stream lags real time (audio path).
  python scripts/tapo_latency.py --stream 2 --repeats 3
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, tapo_rtsp  # noqa: E402
from rtsp_reader import FFmpegAudio  # noqa: E402

BEEP = ROOT / "audio" / "beep.wav"


def make_beep():
    if not BEEP.exists():
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=1500:duration=0.4",
                        "-af", "volume=0.9", "-ar", "22050", "-ac", "1", str(BEEP)], check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stream", type=int, default=2)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--gap", type=float, default=4.0)
    args = p.parse_args()
    make_beep()
    au = FFmpegAudio(tapo_rtsp(args.stream))
    chunks: list[tuple[float, float]] = []      # (wall time, rms dBFS)
    stop = threading.Event()

    def reader():
        for t, x in au:
            rms = float(np.sqrt(np.mean(x ** 2)))
            chunks.append((t, 20 * np.log10(max(rms, 1e-6))))
            if stop.is_set():
                break

    th = threading.Thread(target=reader, daemon=True); th.start()
    t0 = time.time()
    while len(chunks) < 10 and time.time() - t0 < 15:
        time.sleep(0.1)
    if len(chunks) < 10:
        raise SystemExit("no audio from the stream")
    print(f"audio flowing ({len(chunks)} chunks after {time.time() - t0:.1f}s). Mac speaker will beep {args.repeats}x.", flush=True)
    time.sleep(2.0)
    lats = []
    for k in range(args.repeats):
        base = np.median([d for _, d in chunks[-40:]])
        t_play = time.time()
        subprocess.Popen(["afplay", str(BEEP)])
        lat = None
        while time.time() - t_play < args.gap:
            new = [(t, d) for (t, d) in chunks if t > t_play and d > base + 12]
            if new:
                lat = new[0][0] - t_play; break
            time.sleep(0.01)
        print(f"  beep {k + 1}: baseline {base:.0f} dBFS -> {'not detected' if lat is None else f'detected after {lat:.2f}s'}", flush=True)
        if lat is not None:
            lats.append(lat)
        time.sleep(args.gap)
    stop.set(); au.close()
    if lats:
        print(f"audio-path lag behind real time: median {np.median(lats):.2f}s (min {min(lats):.2f}, max {max(lats):.2f}); "
              f"includes ~0.1-0.2s of afplay start-up, so the stream lag is slightly less.")
    else:
        print("beep never detected: is the Mac speaker audible to the camera?")


if __name__ == "__main__":
    main()
