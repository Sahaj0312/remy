"""Eval batch: YOLO pose (ultralytics) over every clip in eval/labels.csv, emitting the same per-frame
track schema as eval/sam_tracks.py into eval/out/yolo/.
Detections are linked into tracks with the greedy IoU/overlap matcher (remy.vision.tracking.link_detections);
ByteTrack was tried and rejected: its Kalman-smoothed boxes lag the fall and the ID breaks mid-fall.

cx, cy = mean of confident keypoints (falls back to box center); ar = box h/w.
Every tracked person is kept under "objects"; "frames" is the primary (most-moving) one.
"""
from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from remy.common import CLIPS, OUT, load_config, load_labels, video_info, write_json
from remy.vision.tracking import empty_frame, finalize_objects, link_detections, suppress_duplicates
from remy.vision.yolo import load_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("clips", nargs="*")
    p.add_argument("--force", action="store_true")
    p.add_argument("--device", default=None, help="cpu | mps | 0")
    args = p.parse_args()
    full_cfg = load_config()
    cfg = full_cfg["yolo"]
    tcfg = full_cfg["tracking"]
    model = load_model(cfg)
    names = args.clips or [r["clip"] for r in load_labels()]
    if not args.force:
        names = [n for n in names if not (OUT / "yolo" / f"{n}.json").exists()]
    for name in names:
        path = CLIPS / f"{name}.mp4"
        info = video_info(path)
        W, H, fps = info["width"], info["height"], info["fps"]
        cap = cv2.VideoCapture(str(path))
        frame_dets: dict[int, list[dict]] = {}
        per_frame_ms = []
        i = 0
        t_clip = time.perf_counter()
        while True:
            ok, img = cap.read()
            if not ok:
                break
            t0 = time.perf_counter()
            res = model.predict(img, verbose=False, conf=cfg["conf"], imgsz=cfg["imgsz"],
                                device=args.device, classes=[0])[0]
            per_frame_ms.append((time.perf_counter() - t0) * 1000)
            if res.boxes is not None and len(res.boxes):
                boxes = res.boxes.xyxy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
                for j in range(len(boxes)):
                    x1, y1, x2, y2 = boxes[j]
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    n_kp = 0
                    if res.keypoints is not None and len(res.keypoints) > j:
                        kxy = res.keypoints.xy[j].cpu().numpy()
                        kc = res.keypoints.conf[j].cpu().numpy() if res.keypoints.conf is not None else np.ones(len(kxy))
                        good = kc >= cfg["kp_conf"]
                        n_kp = int(good.sum())
                        if n_kp >= 3:
                            cx, cy = kxy[good, 0].mean(), kxy[good, 1].mean()
                    frame_dets.setdefault(i, []).append({
                        "box": [float(v) for v in (x1, y1, x2, y2)], "cx_px": float(cx), "cy_px": float(cy),
                        "mask_area_px": None, "conf": round(float(confs[j]), 3), "n_keypoints": n_kp,
                    })
            i += 1
        cap.release()
        frame_dets = {k: suppress_duplicates(v, tcfg["dup_max_overlap"]) for k, v in frame_dets.items()}
        by_obj = link_detections(frame_dets, int(round(tcfg["max_gap_s"] * fps)), tcfg["min_link_score"])
        objects, primary = finalize_objects(by_obj, i, fps, W, H)
        frames = objects[primary] if primary else [empty_frame(k, fps) for k in range(i)]
        doc = {
            "clip": name, "source_video": f"clips/{name}.mp4", "tracker": cfg["model"].replace(".pt", "") + "+iou-link",
            **info,
            "timing": {"total_s": round(time.perf_counter() - t_clip, 2),
                       "per_frame_ms_mean": round(float(np.mean(per_frame_ms)), 1),
                       "per_frame_ms_p95": round(float(np.percentile(per_frame_ms, 95)), 1)},
            "cost_usd": 0.0,
            "meta": {"n_objects": len(objects), "primary_object": primary,
                     "frames_per_object": {k: sum(f["present"] for f in v) for k, v in objects.items()}},
            "frames": frames, "objects": objects,
        }
        write_json(OUT / "yolo" / f"{name}.json", doc)
        present = sum(f["present"] for f in frames)
        print(f"{name}: primary {present}/{len(frames)} frames, {len(objects)} tracked persons, "
              f"{doc['timing']['per_frame_ms_mean']:.0f} ms/frame", flush=True)


if __name__ == "__main__":
    main()
