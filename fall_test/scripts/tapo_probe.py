"""Probe the Tapo C200 RTSP streams: resolution, fps, time to first frame, frame jitter,
glass-to-glass latency (flash test), and snapshots.

  python scripts/tapo_probe.py                      # both streams, 10 s each, snapshots to tapo/
  python scripts/tapo_probe.py --streams 2 --tag night --seconds 4   # one night snapshot
  python scripts/tapo_probe.py --flash              # latency: point the camera at this laptop's screen first
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, redact, tapo_rtsp, write_json  # noqa: E402
from rtsp_reader import FFmpegFrames, probe as ffprobe_json  # noqa: E402

OUT_DIR = ROOT / "tapo"


def probe(stream: int, seconds: float, tag: str, flash: bool, flash_duration: float) -> dict:
    url = tapo_rtsp(stream)
    res = {"stream": f"stream{stream}", "url": redact(url), "tag": tag}
    print(f"== {res['url']}", flush=True)
    t0 = time.time()
    try:
        res["ffprobe"] = ffprobe_json(url)
        res["ffprobe_s"] = round(time.time() - t0, 2)
        t1 = time.time()
        rd = FFmpegFrames(url)
    except Exception as e:
        res["error"] = repr(e); print("  cannot open:", e, flush=True); return res
    res["reported"] = {"width": rd.src_w, "height": rd.src_h, "fps": round(rd.src_fps, 2),
                       "audio": None if rd.audio is None else f"{rd.audio.get('codec_name')} {rd.audio.get('sample_rate')} Hz"}
    ok, frame = rd.read()
    res["first_frame_s"] = round(time.time() - t1, 2)
    if not ok:
        res["error"] = "no frame: " + rd.stderr()[:300]; print("  no frame", flush=True); return res
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUT_DIR.mkdir(exist_ok=True)
    snap = OUT_DIR / f"snap_stream{stream}_{tag}_{stamp}.jpg"
    cv2.imwrite(str(snap), frame)
    res["snapshot"] = str(snap.relative_to(ROOT))
    print(f"  {frame.shape[1]}x{frame.shape[0]} reported {rd.src_fps:g} fps, first frame {res['first_frame_s']}s after open "
          f"(ffprobe took {res['ffprobe_s']}s), snapshot {snap.name}", flush=True)

    ts = [time.time()]
    while time.time() - ts[0] < seconds:
        ok, frame = rd.read()
        if not ok:
            break
        ts.append(time.time())
    iv = np.diff(ts)
    res["measured"] = {"frames": len(ts) - 1, "fps": round((len(ts) - 1) / max(1e-6, ts[-1] - ts[0]), 2),
                       "interval_ms_mean": round(float(iv.mean() * 1000), 1) if len(iv) else None,
                       "interval_ms_p95": round(float(np.percentile(iv, 95) * 1000), 1) if len(iv) else None,
                       "interval_ms_max": round(float(iv.max() * 1000), 1) if len(iv) else None}
    print(f"  measured {res['measured']['fps']} fps over {seconds:g}s (interval mean {res['measured']['interval_ms_mean']} ms, "
          f"p95 {res['measured']['interval_ms_p95']} ms, max {res['measured']['interval_ms_max']} ms)", flush=True)

    if flash:
        win = "flash"
        cv2.namedWindow(win, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        black = np.zeros((600, 1000, 3), np.uint8); white = np.full((600, 1000, 3), 255, np.uint8)
        cv2.imshow(win, black); cv2.waitKey(1)
        base = []; t_end = time.time() + 2.0
        while time.time() < t_end:
            ok, frame = rd.read(); cv2.waitKey(1)
            if ok:
                base.append(float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean()))
        b0 = float(np.median(base)) if base else 0.0
        cv2.imshow(win, white); cv2.waitKey(1)
        t_flash = time.time(); lat = None
        while time.time() - t_flash < flash_duration:
            ok, frame = rd.read(); cv2.waitKey(1)
            if not ok:
                continue
            m = float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean())
            if m > b0 + 12:
                lat = time.time() - t_flash; break
        cv2.destroyWindow(win); cv2.waitKey(1)
        res["flash_latency_s"] = None if lat is None else round(lat, 3)
        res["flash_baseline_brightness"] = round(b0, 1)
        print(f"  flash latency: {'not detected (is the camera looking at this screen?)' if lat is None else f'{lat:.2f} s'}", flush=True)
    rd.close()
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--streams", default="1,2")
    p.add_argument("--seconds", type=float, default=10)
    p.add_argument("--tag", default="day")
    p.add_argument("--flash", action="store_true", help="glass-to-glass latency via a screen flash")
    p.add_argument("--flash-duration", type=float, default=8.0)
    args = p.parse_args()
    results = [probe(int(s), args.seconds, args.tag, args.flash, args.flash_duration) for s in args.streams.split(",")]
    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / f"probe_{args.tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    write_json(out, results)
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
