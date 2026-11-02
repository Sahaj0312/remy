"""Track-file schema and the greedy box linker shared by the YOLO and SAM paths.

A track is a list of per-frame dicts {i, t, present, box, cx, cy, ar, mask_area, conf} (cx, cy normalised
to the frame). ByteTrack was tried and rejected: its Kalman-smoothed boxes lag the fall and the ID breaks
mid-fall, so detections are linked frame to frame by IoU / overlap instead.
"""
from __future__ import annotations

from collections import deque

import numpy as np


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


class OnlineTracks:
    """Live version of the linker: feeds detections tick by tick and keeps the last keep_s seconds per track."""

    def __init__(self, fps, H, W, max_gap_s, min_score, keep_s):
        self.fps, self.H, self.W = fps, H, W
        self.max_gap = int(round(max_gap_s * fps)); self.min_score = min_score; self.keep = int(round(keep_s * fps))
        self.tracks: dict[str, dict] = {}; self.next_id = 1; self.tick = 0

    def update(self, dets):
        i = self.tick
        cands = sorted([(link_score(tr["last_box"], d["box"]), tid, j) for tid, tr in self.tracks.items()
                        if i - tr["last_i"] <= self.max_gap for j, d in enumerate(dets)], reverse=True)
        used_t, used_d = set(), set()
        for sc, tid, j in cands:
            if sc < self.min_score or tid in used_t or j in used_d:
                continue
            used_t.add(tid); used_d.add(j); self._push(tid, i, dets[j])
        for j, d in enumerate(dets):
            if j not in used_d:
                tid = str(self.next_id); self.next_id += 1
                self.tracks[tid] = {"frames": deque(maxlen=self.keep), "last_i": i, "last_box": d["box"]}
                self._push(tid, i, d)
        for tid in list(self.tracks):
            tr = self.tracks[tid]
            if tr["last_i"] != i:
                tr["frames"].append(empty_frame(i, self.fps))
            if i - tr["last_i"] > self.max_gap:
                del self.tracks[tid]
        self.tick += 1

    def _push(self, tid, i, d):
        tr = self.tracks[tid]; x1, y1, x2, y2 = d["box"]
        tr["frames"].append({"i": i, "t": round(i / self.fps, 3), "present": True, "box": d["box"], "cx": d["cx_px"] / self.W,
                             "cy": d["cy_px"] / self.H, "ar": (y2 - y1) / max(x2 - x1, 1e-6), "mask_area": None, "conf": d["conf"]})
        tr["last_i"] = i; tr["last_box"] = d["box"]

    def as_track(self, name="live"):
        objects = {}
        for tid, tr in self.tracks.items():
            fr = list(tr["frames"])
            if fr:
                base = fr[0]["i"]
                objects[tid] = [{**f, "i": f["i"] - base, "t": round((f["i"] - base) / self.fps, 3)} for f in fr]
        primary = max(objects, key=lambda k: sum(f["present"] for f in objects[k])) if objects else None
        return {"clip": name, "fps": self.fps, "height": self.H, "width": self.W, "tracker": "live-yolo",
                "frames": objects.get(primary, []), "objects": objects}
