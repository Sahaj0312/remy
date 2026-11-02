"""Rule-based fall scores over a per-frame track (SAM or YOLO).

v2 (default) works in body lengths, has no "tall before" requirement, and replaces the fixed
hold with a get-up check:
  DROP : centroid y falls >= drop_rel_min body lengths in total, with a peak speed >= drop_speed_rel_min
         body lengths/s measured over drop_window_s. Body length = median of max(box h, box w) in
         the body_len_window_s before the drop (so a person lying on a bed still has a sane scale).
  SHAPE: median box aspect ratio (h/w) in the post_window_s after the drop <= ar_wide_max (ends wide).
  HOLD : "ok" if no get-up within no_getup_s after the drop, "failed" if the person gets up
         (ar >= getup_ar_min or centroid rises >= getup_rise_rel body lengths), "unknown" if the clip
         ends first. A clip is flagged when DROP and SHAPE pass and HOLD is ok or unknown.
v1 is the first-round score (absolute drop + tall->wide + fixed 3 s low hold), kept for before/after.
trigger is v2's DROP check alone with looser thresholds: what the always-on YOLO stage uses to call SAM.
Thresholds live in config.json under score_v1 / score_v2 / trigger. The batch runner over the eval set is eval/score_tracks.py.
"""
from __future__ import annotations

import math

import numpy as np


# ----------------------------------------------------------------------------- shared helpers
def arrays(frames: list[dict], fps: float, max_gap_s: float):
    """t, cy, ar, body (max(box h, box w) as fraction of frame height), present; short gaps interpolated."""
    n = len(frames)
    t = np.array([f["t"] for f in frames], dtype=float)
    cy = np.full(n, np.nan); ar = np.full(n, np.nan); body = np.full(n, np.nan)
    for k, f in enumerate(frames):
        if f.get("present"):
            cy[k] = f["cy"]; ar[k] = f["ar"]
            if "box" in f and "_H" in f:
                x1, y1, x2, y2 = f["box"]
                body[k] = max(y2 - y1, x2 - x1) / f["_H"]
    present = ~np.isnan(cy)
    max_gap = int(round(max_gap_s * fps))
    for arr in (cy, ar, body):
        idx = np.where(~np.isnan(arr))[0]
        for k in range(len(idx) - 1):
            a, b = idx[k], idx[k + 1]
            if 1 < b - a <= max_gap + 1:
                arr[a:b + 1] = np.linspace(arr[a], arr[b], b - a + 1)
    return t, cy, ar, body, present


def nanmedian(x):
    x = x[~np.isnan(x)]
    return float(np.median(x)) if len(x) else float("nan")


def merge_events(cand: np.ndarray, k: int) -> list[tuple[int, int]]:
    events = []
    if len(cand):
        start = prev = int(cand[0])
        for c in cand[1:]:
            c = int(c)
            if c - prev <= 1:
                prev = c
            else:
                events.append((start, prev + k)); start = prev = c
        events.append((start, prev + k))
    return events


# ----------------------------------------------------------------------------- v2
def score_v2(track: dict, frames: list[dict], cfg: dict, trigger_only: bool = False) -> dict:
    fps = track["fps"]; H = track["height"]
    for f in frames:
        f["_H"] = H
    n = len(frames)
    t, cy, ar, body, present = arrays(frames, fps, cfg["max_gap_s"])
    out = {"clip": track["clip"], "tracker": track.get("tracker"), "scorer": "trigger" if trigger_only else "v2",
           "coverage": round(float(present.mean()) if n else 0.0, 3), "n_frames": n,
           "flag": False, "hold": None, "events": [], "best": None}
    if present.sum() < max(3, int(cfg["min_coverage"] * n)):
        out["reason"] = "insufficient tracking coverage"; return out

    k = max(1, int(round(cfg["drop_window_s"] * fps)))
    bw = max(1, int(round(cfg["body_len_window_s"] * fps)))
    clip_body = max(nanmedian(body), cfg["body_len_floor"])
    # rolling body length from the preceding window (falls back to the clip median)
    body_ref = np.full(n, clip_body)
    for i in range(n):
        m = nanmedian(body[max(0, i - bw):i + 1])
        if not math.isnan(m):
            body_ref[i] = max(m, cfg["body_len_floor"])
    vel = np.full(n, np.nan)
    for i in range(n - k):
        if not (math.isnan(cy[i]) or math.isnan(cy[i + k])):
            vel[i] = (cy[i + k] - cy[i]) / (k / fps) / body_ref[i]     # body lengths per second, + = down
    cand = np.where(vel >= cfg["drop_speed_rel_min"])[0]
    events = merge_events(cand, k)

    post_w = max(1, int(round(cfg["post_window_s"] * fps)))
    hold_n = int(round(cfg["no_getup_s"] * fps))
    results = []
    for (s, e) in events:
        e = min(e, n - 1)
        bl = body_ref[s]
        total_drop = float(cy[e] - cy[s]) if not (math.isnan(cy[e]) or math.isnan(cy[s])) else float("nan")
        drop_rel = total_drop / bl if not math.isnan(total_drop) else float("nan")
        vmax = float(np.nanmax(vel[s:e + 1]))
        ar_post = nanmedian(ar[e:min(n, e + post_w + 1)])
        cy_post = nanmedian(cy[e:min(n, e + post_w + 1)])
        drop_ok = (not math.isnan(drop_rel)) and drop_rel >= cfg["drop_rel_min"] and vmax >= cfg["drop_speed_rel_min"]
        shape_ok = (not math.isnan(ar_post)) and ar_post <= cfg["ar_wide_max"]
        # get-up check
        hold, getup_i = "unknown", None
        tail_end = min(n, e + 1 + hold_n)
        for j in range(e + 1, tail_end):
            if math.isnan(ar[j]):
                continue
            if ar[j] >= cfg["getup_ar_min"] or cy[j] <= cy_post - cfg["getup_rise_rel"] * bl:
                hold, getup_i = "failed", j; break
        if hold == "unknown" and (tail_end - (e + 1)) >= hold_n:
            hold = "ok"
        observed_s = ((getup_i if getup_i is not None else tail_end) - (e + 1)) / fps
        flag = drop_ok and (trigger_only or (shape_ok and hold in ("ok", "unknown")))
        soft = (min(1.0, vmax / cfg["drop_speed_rel_min"]),
                min(1.0, drop_rel / cfg["drop_rel_min"]) if not math.isnan(drop_rel) and drop_rel > 0 else 0.0,
                min(1.0, cfg["ar_wide_max"] / ar_post) if not math.isnan(ar_post) and ar_post > 0 else 0.0,
                1.0 if hold == "ok" else (0.5 if hold == "unknown" else 0.0))
        results.append({
            "start_i": int(s), "end_i": int(e), "start_t": round(float(t[s]), 2), "end_t": round(float(t[e]), 2),
            "time_to_floor_s": round(float(t[e] - t[s]), 2),
            "body_len": round(float(bl), 3), "drop_rel": round(drop_rel, 3), "drop_speed_rel_max": round(vmax, 3),
            "total_drop": round(total_drop, 3), "ar_post": round(ar_post, 3), "cy_post": round(cy_post, 3),
            "hold": hold, "getup_t": None if getup_i is None else round(float(t[getup_i]), 2),
            "no_getup_observed_s": round(observed_s, 2),
            "drop_ok": bool(drop_ok), "shape_ok": bool(shape_ok), "flag": bool(flag),
            "soft_score": round(float(np.mean(soft)), 3),
        })
    out["events"] = results
    if results:
        flagged = [r for r in results if r["flag"]]
        best = max(flagged or results, key=lambda r: r["soft_score"])
        out["best"] = best; out["flag"] = bool(flagged); out["hold"] = best["hold"]
    out["summary"] = {"cy_min": round(float(np.nanmin(cy)), 3), "cy_max": round(float(np.nanmax(cy)), 3),
                      "ar_min": round(float(np.nanmin(ar)), 3), "ar_max": round(float(np.nanmax(ar)), 3),
                      "body_len_clip": round(clip_body, 3),
                      "vel_rel_max": round(float(np.nanmax(vel)), 3) if np.any(~np.isnan(vel)) else None}
    return out


# ----------------------------------------------------------------------------- v1 (first round)
def score_v1(track: dict, frames: list[dict], cfg: dict) -> dict:
    fps = track["fps"]; n = len(frames)
    t, cy, ar, _body, present = arrays(frames, fps, cfg["max_gap_s"])
    out = {"clip": track["clip"], "tracker": track.get("tracker"), "scorer": "v1",
           "coverage": round(float(present.mean()) if n else 0.0, 3), "n_frames": n,
           "flag": False, "hold": None, "events": [], "best": None}
    if present.sum() < max(3, int(cfg["min_coverage"] * n)):
        out["reason"] = "insufficient tracking coverage"; return out
    k = max(1, int(round(cfg["drop_window_s"] * fps)))
    vel = np.full(n, np.nan)
    for i in range(n - k):
        if not (math.isnan(cy[i]) or math.isnan(cy[i + k])):
            vel[i] = (cy[i + k] - cy[i]) / (k / fps)
    events = merge_events(np.where(vel >= cfg["drop_speed_min"])[0], k)
    pre_w = max(1, int(round(cfg["pre_window_s"] * fps))); post_w = max(1, int(round(cfg["post_window_s"] * fps)))
    hold_n = int(round(cfg["low_hold_s"] * fps)); min_post_n = int(round(cfg["min_post_s"] * fps))
    results = []
    for (s, e) in events:
        e = min(e, n - 1)
        total_drop = float(cy[e] - cy[s]) if not (math.isnan(cy[e]) or math.isnan(cy[s])) else float("nan")
        vmax = float(np.nanmax(vel[s:e + 1]))
        ar_pre = nanmedian(ar[max(0, s - pre_w):s + 1]); ar_post = nanmedian(ar[e:min(n, e + post_w + 1)])
        cy_post = nanmedian(cy[e:min(n, e + post_w + 1)])
        tail = slice(e + 1, min(n, e + 1 + hold_n)); tail_len = tail.stop - tail.start
        low = [(ar[j] <= cfg["low_ar_max"]) or (cy[j] >= cy_post - cfg["low_cy_margin"])
               for j in range(tail.start, tail.stop) if not math.isnan(ar[j])]
        low_frac = float(np.mean(low)) if low else 0.0
        if tail_len == 0 and cfg["allow_clip_end"] and not math.isnan(ar[e]):
            low_frac = 1.0 if ar[e] <= cfg["low_ar_max"] else 0.0
        drop_ok = (not math.isnan(total_drop)) and total_drop >= cfg["drop_min"] and vmax >= cfg["drop_speed_min"]
        shape_ok = (not math.isnan(ar_pre)) and (not math.isnan(ar_post)) and ar_pre >= cfg["ar_tall_min"] and ar_post <= cfg["ar_wide_max"]
        hold_enough = (tail_len >= hold_n) or (cfg["allow_clip_end"] and (tail_len >= min_post_n or tail_len == 0))
        hold_ok = hold_enough and low_frac >= cfg["low_hold_frac"]
        results.append({"start_i": int(s), "end_i": int(e), "start_t": round(float(t[s]), 2), "end_t": round(float(t[e]), 2),
                        "drop_speed_max": round(vmax, 3), "total_drop": round(total_drop, 3),
                        "ar_pre": round(ar_pre, 3), "ar_post": round(ar_post, 3), "low_frac": round(low_frac, 3),
                        "hold_s": round(tail_len / fps, 2), "clip_truncated_hold": bool(tail_len < hold_n),
                        "hold": "ok" if hold_ok else "failed",
                        "drop_ok": bool(drop_ok), "shape_ok": bool(shape_ok), "hold_ok": bool(hold_ok),
                        "flag": bool(drop_ok and shape_ok and hold_ok),
                        "soft_score": round(float(np.mean([min(1.0, vmax / cfg["drop_speed_min"]), low_frac])), 3)})
    out["events"] = results
    if results:
        flagged = [r for r in results if r["flag"]]
        best = max(flagged or results, key=lambda r: r["soft_score"])
        out["best"] = best; out["flag"] = bool(flagged); out["hold"] = best["hold"]
    return out


# ----------------------------------------------------------------------------- per-clip wrapper
def score_track(track: dict, cfg: dict, scorer: str) -> dict:
    """Score every tracked person; the clip is flagged if any of them is."""
    objects = track.get("objects") or {"primary": track["frames"]}
    fn = {"v1": lambda tr, fr: score_v1(tr, fr, cfg),
          "v2": lambda tr, fr: score_v2(tr, fr, cfg),
          "trigger": lambda tr, fr: score_v2(tr, fr, cfg, trigger_only=True)}[scorer]
    per_obj = {oid: fn(track, frames) for oid, frames in objects.items()
               if sum(f.get("present", False) for f in frames) >= 3}
    if not per_obj:
        res = fn(track, track["frames"]); res["object_id"] = None; res["objects_scored"] = {}; return res
    def key(oid):
        r = per_obj[oid]
        return (bool(r["flag"]), (r.get("best") or {}).get("soft_score", 0.0), r["coverage"])
    best_oid = max(per_obj, key=key)
    res = dict(per_obj[best_oid]); res["object_id"] = best_oid
    res["objects_scored"] = {oid: {"flag": r["flag"], "coverage": r["coverage"],
                                   "soft_score": (r.get("best") or {}).get("soft_score")} for oid, r in per_obj.items()}
    return res
