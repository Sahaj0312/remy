"""Annotated GIFs for the report: box + centroid trail + state overlay, title with labels/verdicts.
  python eval/make_gifs.py gmdcsa_S1_Fall_12 --out eval/out/gifs/gmdcsa_S1_Fall_12.gif --label fall
Needs the eval extra (imageio): pip install -e ".[eval]"."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import imageio.v3 as iio
import numpy as np

from remy.common import CLIPS, OUT, read_json

FONT = cv2.FONT_HERSHEY_SIMPLEX


def put(img, text, org, scale=0.45, color=(255, 255, 255), thick=1):
    cv2.putText(img, text, org, FONT, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, color, thick, cv2.LINE_AA)


def make_gif(name: str, score_dir: str, out_path: Path, title: str | None, width: int, label: str):
    tracker = score_dir.split("_")[0]
    track = read_json(OUT / tracker / f"{name}.json")
    score_p = OUT / "scores" / score_dir / f"{name}.json"
    score = read_json(score_p) if score_p.exists() else {}
    spark_p = OUT / "spark" / score_dir / f"{name}.json"
    spark = read_json(spark_p) if spark_p.exists() else None
    best = score.get("best") or {}
    fps = track["fps"]
    frames_src = track["frames"]
    oid = score.get("object_id")
    if oid and oid in (track.get("objects") or {}):
        frames_src = track["objects"][oid]
    s_i, e_i = best.get("start_i", -1), best.get("end_i", -1)
    hold_end = e_i + int(round(best.get("no_getup_observed_s", best.get("hold_s", 0)) * fps))

    cap = cv2.VideoCapture(str(CLIPS / f"{name}.mp4"))
    frames_out, trail = [], []
    i = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        H, W = img.shape[:2]
        f = frames_src[i] if i < len(frames_src) else {"present": False}
        state, color = "", (80, 200, 80)
        if s_i <= i <= e_i:
            state, color = "DROP", (40, 40, 230)
        elif e_i < i <= hold_end:
            state, color = "DOWN", (30, 150, 255)
        if f.get("present"):
            x1, y1, x2, y2 = [int(round(v)) for v in f["box"]]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cx, cy = int(f["cx"] * W), int(f["cy"] * H)
            trail.append((cx, cy))
            for k in range(1, len(trail[-12:])):
                a, b = trail[-12:][k - 1], trail[-12:][k]
                cv2.line(img, a, b, (0, 255, 255), 2)
            cv2.circle(img, (cx, cy), 5, (0, 255, 255), -1)
            put(img, f"AR {f['ar']:.2f}  cy {f['cy']:.2f}", (x1, max(14, y1 - 6)), 0.45, (255, 255, 255))
        else:
            put(img, "no person", (8, H - 10), 0.45, (200, 200, 200))
        put(img, f"t={i / fps:4.1f}s {state}", (8, 18), 0.5, color if state else (255, 255, 255), 1)
        # title bar
        bar = np.zeros((44, W, 3), dtype=np.uint8)
        t1 = title or f"{name}  true={label}"
        flag = score.get("flag")
        t2 = f"{score_dir} flag={'YES' if flag else 'no'}" + (f" hold={best.get('hold')}" if best else "")
        if spark:
            v = spark["verdict"]
            t2 += f" | Spark: {v.get('urgency')}, {'lying still' if v.get('is_lying_still') else 'moving'}"
        put(bar, t1, (8, 17), 0.5, (255, 255, 255), 1)
        put(bar, t2, (8, 37), 0.45, (180, 220, 255), 1)
        img = np.vstack([bar, img])
        scale = width / img.shape[1]
        img = cv2.resize(img, (width, int(img.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        frames_out.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        i += 1
    cap.release()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(out_path, frames_out, duration=int(1000 / fps), loop=0)
    print(f"wrote {out_path} ({len(frames_out)} frames, {out_path.stat().st_size / 1e6:.1f} MB)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("clip")
    p.add_argument("--score-dir", default="sam_v2")
    p.add_argument("--out", required=True)
    p.add_argument("--title", default=None)
    p.add_argument("--label", default="?")
    p.add_argument("--width", type=int, default=420)
    a = p.parse_args()
    make_gif(a.clip, a.score_dir, Path(a.out), a.title, a.width, a.label)


if __name__ == "__main__":
    main()
