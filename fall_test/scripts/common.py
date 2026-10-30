"""Shared helpers: paths, config, API key, track-file schema."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLIPS = ROOT / "clips"
OUT = ROOT / "out"
CONFIG_PATH = ROOT / "config.json"
LABELS_PATH = ROOT / "labels.csv"

META_BASE_URL = "https://api.meta.ai/v1"
SAM_MODEL = "sam-3.1"
SPARK_MODEL = "muse-spark-1.3"

# Pricing from https://dev.meta.ai/docs/pricing-rate-limits (checked 2026-10-29)
SAM_USD_PER_VIDEO_FRAME = 0.20 / 1000
SAM_MIN_BILLED_FRAMES = 50
SPARK_USD_PER_INPUT_TOKEN = 1.25 / 1_000_000
SPARK_USD_PER_OUTPUT_TOKEN = 4.25 / 1_000_000


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return json.load(f)


def load_labels() -> list[dict]:
    with open(LABELS_PATH, newline="") as f:
        return list(csv.DictReader(f))


def env_get(*names: str, required: bool = True) -> str | None:
    """First of `names` found in the environment, else in fall_test/.env."""
    for name in names:
        if os.environ.get(name):
            return os.environ[name]
    env_file = ROOT / ".env"
    if env_file.exists():
        vals = {}
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            vals[k.strip()] = v.strip().strip('"').strip("'")
        for name in names:
            if vals.get(name):
                return vals[name]
    if required:
        raise SystemExit(f"{names[0]} is not set. Export it or put {names[0]}=... in fall_test/.env")
    return None


def api_key() -> str:
    """META_API_KEY (or MODEL_API_KEY) from env, else from fall_test/.env."""
    return env_get("META_API_KEY", "MODEL_API_KEY")


def tapo_rtsp(stream: int = 2) -> str:
    """rtsp://user:pass@ip:554/stream<N> from TAPO_IP / TAPO_USER / TAPO_PASS."""
    from urllib.parse import quote
    ip, user, pw = env_get("TAPO_IP"), env_get("TAPO_USER"), env_get("TAPO_PASS")
    return f"rtsp://{quote(user, safe='')}:{quote(pw, safe='')}@{ip}:554/stream{stream}"


def redact(url: str) -> str:
    import re
    return re.sub(r"//[^@/]+@", "//<user>:<pass>@", url)


def video_info(path: Path) -> dict:
    """width, height, fps, n_frames via ffprobe."""
    import subprocess

    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-count_frames",
            "-show_entries", "stream=width,height,r_frame_rate,nb_read_frames",
            "-of", "json", str(path),
        ],
        capture_output=True, text=True, check=True,
    ).stdout
    s = json.loads(out)["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    return {
        "width": int(s["width"]),
        "height": int(s["height"]),
        "fps": float(num) / float(den),
        "n_frames": int(s["nb_read_frames"]),
    }


def sam_cost_usd(n_frames: int) -> float:
    return max(n_frames, SAM_MIN_BILLED_FRAMES) * SAM_USD_PER_VIDEO_FRAME


def empty_frame(i: int, fps: float) -> dict:
    return {"i": i, "t": round(i / fps, 3), "present": False}


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0, 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter), inter / max(1e-6, min(area_a, area_b))


def link_score(a, b):
    """Association score between two boxes: IoU, or half the overlap relative to the
    smaller box (so a box that collapses from tall to wide during a fall still links)."""
    iou, ov = _iou(a, b)
    return max(iou, 0.5 * ov)


def suppress_duplicates(dets: list[dict], max_overlap: float = 0.5) -> list[dict]:
    """Drop detections that mostly lie inside a higher-confidence one (YOLO's IoU NMS
    keeps partial-body duplicates whose IoU is small but whose overlap/min-area is large)."""
    keep: list[dict] = []
    for d in sorted(dets, key=lambda d: -(d.get("conf") or 0.0)):
        if all(_iou(d["box"], k["box"])[1] < max_overlap for k in keep):
            keep.append(d)
    return keep


def link_detections(frame_dets: dict[int, list[dict]], max_gap: int, min_score: float) -> dict[str, dict[int, dict]]:
    """Greedy frame-to-frame association of raw detections into tracks.
    frame_dets: {frame_i: [det dicts with "box"]} -> {track_id: {frame_i: det}}"""
    tracks: dict[str, dict[int, dict]] = {}
    last: dict[str, tuple[int, list]] = {}      # id -> (last frame, last box)
    next_id = 1
    for i in sorted(frame_dets):
        dets = frame_dets[i]
        cands = []
        for tid, (li, lb) in last.items():
            if i - li > max_gap:
                continue
            for j, d in enumerate(dets):
                sc = link_score(lb, d["box"])
                if sc >= min_score:
                    cands.append((sc, tid, j))
        cands.sort(reverse=True)
        used_t, used_d = set(), set()
        for sc, tid, j in cands:
            if tid in used_t or j in used_d:
                continue
            used_t.add(tid); used_d.add(j)
            tracks[tid][i] = dets[j]; last[tid] = (i, dets[j]["box"])
        for j, d in enumerate(dets):
            if j in used_d:
                continue
            tid = str(next_id); next_id += 1
            tracks[tid] = {i: d}; last[tid] = (i, d["box"])
    return tracks


def merge_broken_tracks(by_obj: dict[str, dict[int, dict]], max_gap: int, min_score: float) -> tuple[dict, dict]:
    """Merge track B into track A when A ends, B starts within max_gap frames, they
    don't overlap in time, and their boundary boxes link. Returns (merged, id_map)."""
    order = sorted(by_obj, key=lambda k: min(by_obj[k]))
    merged: dict[str, dict[int, dict]] = {}
    id_map: dict[str, str] = {}
    for oid in order:
        frames = by_obj[oid]
        if not frames:
            continue
        start = min(frames)
        best, best_sc = None, 0.0
        for mid, mframes in merged.items():
            end = max(mframes)
            if 0 < start - end <= max_gap:
                sc = link_score(mframes[end]["box"], frames[start]["box"])
                if sc >= min_score and sc > best_sc:
                    best, best_sc = mid, sc
        if best is None:
            merged[oid] = dict(frames); id_map[oid] = oid
        else:
            merged[best].update(frames); id_map[oid] = best
    return merged, id_map


def finalize_objects(by_obj: dict, n_frames: int, fps: float, W: int, H: int) -> tuple[dict, str | None]:
    """by_obj: {object_id: {frame_i: {box, cx_px, cy_px, ar, mask_area_px, conf}}}
    -> ({object_id: [frame dicts for every i]}, primary object id).
    Primary = the object whose normalized centroid moves the most (std cx + std cy),
    which prefers the subject over a seated bystander."""
    import numpy as np

    objects: dict[str, list[dict]] = {}
    motion: dict[str, float] = {}
    for oid, per in by_obj.items():
        frames = []
        cxs, cys = [], []
        for i in range(n_frames):
            d = per.get(i)
            if not d:
                frames.append(empty_frame(i, fps))
                continue
            x1, y1, x2, y2 = d["box"]
            bw, bh = max(x2 - x1, 1e-6), max(y2 - y1, 1e-6)
            cx, cy = d["cx_px"] / W, d["cy_px"] / H
            cxs.append(cx); cys.append(cy)
            frames.append({
                "i": i, "t": round(i / fps, 3), "present": True,
                "box": [round(float(v), 1) for v in (x1, y1, x2, y2)],
                "cx": round(float(cx), 4), "cy": round(float(cy), 4),
                "ar": round(float(bh / bw), 3),
                "mask_area": None if d.get("mask_area_px") is None else round(d["mask_area_px"] / (W * H), 4),
                "conf": d.get("conf"),
            })
        objects[str(oid)] = frames
        motion[str(oid)] = float(np.std(cxs) + np.std(cys)) if len(cxs) >= 2 else 0.0
    primary = max(motion, key=motion.get) if motion else None
    return objects, primary


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)


def read_json(path: Path):
    with open(path) as f:
        return json.load(f)
