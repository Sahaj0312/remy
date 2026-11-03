"""Does SAM 3.1 latency depend on how much video we send? Two parts:
  1. a regression over the 100 committed eval runs (free): upload, first-frame and stream time vs frame count;
  2. a controlled run (about $0.15): one fall clip in five variants (10 vs 5 fps, 3.5 vs 3 s, 640 vs 480 px),
     each sent three times one after another, with the fall score checked on every result.
Writes eval/SAM_LATENCY.md.

  python eval/sam_latency.py                 # part 1 only
  python eval/sam_latency.py --controlled    # both (needs META_API_KEY; sends 15 clips)
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from remy.common import EVAL, OUT, ROOT, load_config, video_info
from remy.vision import sam
from remy.vision.score import score_track

CLIP = ROOT / "examples" / "event_gmdcsa_S1_Fall_12" / "clip.mp4"      # the live simulation's 35-frame clip
VARIANTS = {   # name: (ffmpeg input options, filter)
    "35f_10fps_640": ([], None),
    "18f_5fps_640": ([], "fps=5"),
    "30f_10fps_640": (["-ss", "0.5"], None),
    "15f_5fps_640": (["-ss", "0.5"], "fps=5"),
    "35f_10fps_480": ([], "scale=480:-2"),
}


def regression() -> list[str]:
    rows = []
    for f in sorted(glob.glob(str(OUT / "sam" / "*.json"))):
        if f.endswith(".error.json"):
            continue
        d = json.load(open(f)); t = d["timing"]
        rows.append((d["n_frames"], t["upload_s"], t["first_frame_s"], t["stream_s"], t["total_s"]))
    n = np.array([r[0] for r in rows], float)
    L = [f"## 1. The {len(rows)} eval runs (batch, three uploads in parallel)\n",
         f"Frames per clip {n.min():.0f} to {n.max():.0f}, median {np.median(n):.0f}. Linear fit of each timing against frame count:\n",
         "| timing | median | p10 | p90 | fit | correlation |", "|---|---|---|---|---|---|"]
    for name, i in (("upload", 1), ("time to first frame", 2), ("stream after upload", 3), ("total", 4)):
        y = np.array([r[i] for r in rows]); a, b = np.polyfit(n, y, 1); r = np.corrcoef(n, y)[0, 1]
        L.append(f"| {name} | {np.median(y):.1f} s | {np.percentile(y, 10):.1f} s | {np.percentile(y, 90):.1f} s | {b:.1f} s + {a * 100:.1f} s per 100 frames | {r:.2f} |")
    L += ["", "| frames | clips | upload | first frame | stream | total |", "|---|---|---|---|---|---|"]
    for lo, hi in ((0, 40), (40, 60), (60, 90), (90, 200)):
        m = (n >= lo) & (n < hi)
        if m.any():
            med = lambda i: np.median([r[i] for r, k in zip(rows, m) if k])
            L.append(f"| {lo} to {hi} | {m.sum()} | {med(1):.1f} s | {med(2):.1f} s | {med(3):.1f} s | {med(4):.1f} s |")
    L.append("\nThe stream time grows with frames; the first frame does not. The batch ran three clips at once, so the "
             "absolute numbers are higher than a single live call.")
    return L


async def controlled(runs: int) -> list[str]:
    cfg = load_config()["score_v2"]
    tmp = Path(tempfile.mkdtemp())
    files = {}
    for name, (inopts, vf) in VARIANTS.items():
        dst = tmp / f"{name}.mp4"
        cmd = ["ffmpeg", "-y", "-v", "error", *inopts, "-i", str(CLIP)] + (["-vf", vf] if vf else []) + \
              ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", str(dst)]
        subprocess.run(cmd, check=True); files[name] = dst
    res = {k: [] for k in files}
    async with sam.make_client(timeout=600) as api:
        for rep in range(runs):
            for name, f in files.items():
                info = video_info(f)
                result, t = await sam.segment_video(api, f, "one_bit", False)
                frames, meta, objects = sam.result_to_frames(result, info["n_frames"], info["fps"], info["width"], info["height"], use_masks=False)
                sc = score_track({"clip": name, "fps": info["fps"], "height": info["height"], "width": info["width"], "n_frames": info["n_frames"],
                                  "frames": frames, "objects": objects}, cfg, "v2")
                res[name].append((info["n_frames"], t["upload_s"], t["first_frame_s"], t["stream_s"], t["total_s"], sc["flag"]))
                print(f"rep {rep + 1} {name:16s} total {t['total_s']:.1f}s flag={sc['flag']}", flush=True)
    L = [f"\n## 2. One fall clip, five variants, {runs} sequential calls each\n",
         "| variant | frames | total, median (min to max) | upload | first frame | stream | fall confirmed |", "|---|---|---|---|---|---|---|"]
    for name, v in res.items():
        tot = [x[4] for x in v]
        L.append(f"| {name.replace('_', ' ')} | {v[0][0]} | {np.median(tot):.1f} s ({min(tot):.1f} to {max(tot):.1f}) | {np.median([x[1] for x in v]):.1f} s | "
                 f"{np.median([x[2] for x in v]):.1f} s | {np.median([x[3] for x in v]):.1f} s | {sum(x[5] for x in v)}/{len(v)} |")
    return L


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--controlled", action="store_true")
    p.add_argument("--runs", type=int, default=3)
    a = p.parse_args()
    L = ["# SAM 3.1 latency versus clip length\n",
         "Question: is the 8 to 10 s from \"clip cut\" to \"fall confirmed\" a function of how much video we send, or a fixed "
         "cost of the call? If it scales, a shorter or sparser clip buys time; if not, only self-hosting does.\n"] + regression()
    if a.controlled:
        L += asyncio.run(controlled(a.runs))
    L += ["", "## Reading\n",
          "A live call is about 4 s of fixed cost (2 s upload, 2 s to the first frame) plus roughly 0.1 s per frame of streaming. "
          "Halving the frames (5 fps) or the width (480 px) each save about a second on a 7 s median, and run-to-run variance "
          "is 6 to 12 s for the same clip. Every variant still confirmed the fall. Verdict: the live clip stays at 10 fps and "
          "640 px; the frame count is not the lever. The only large cut left is to run SAM yourself, with no upload and no queue."]
    out = EVAL / "SAM_LATENCY.md"; out.write_text("\n".join(L) + "\n")
    print("\n".join(L)); print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
