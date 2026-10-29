"""Rule-based fall score over a per-frame track (SAM or YOLO).

A candidate fall needs, in order:
  1. DROP : centroid y (normalized, 0=top) rises fast: velocity >= drop_speed_min
           over a drop_window_s window, and the total drop across the merged
           event >= drop_min (fraction of frame height).
  2. SHAPE: box aspect ratio (h/w) median in the pre_window_s before the drop
           >= ar_tall_min, and median in the post_window_s after >= ... <= ar_wide_max.
  3. HOLD : for low_hold_s after the drop ends, at least low_hold_frac of the
           frames are "low" (ar <= low_ar_max, or cy within low_cy_margin of
           the post-drop centroid). If the clip ends first and allow_clip_end
           is true, the available tail is used as long as it's >= min_post_s.
All thresholds live in config.json["score"].
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, load_config, load_labels, read_json, write_json  # noqa: E402


def interpolate(frames: list[dict], fps: float, max_gap_s: float):
    """Return arrays t, cy, ar, present with short gaps linearly filled."""
    n = len(frames)
    t = np.array([f["t"] for f in frames], dtype=float)
    cy = np.full(n, np.nan)
    ar = np.full(n, np.nan)
    for k, f in enumerate(frames):
        if f.get("present"):
            cy[k] = f["cy"]
            ar[k] = f["ar"]
    present = ~np.isnan(cy)
    max_gap = int(round(max_gap_s * fps))
    for arr in (cy, ar):
        idx = np.where(~np.isnan(arr))[0]
        if len(idx) < 2:
            continue
        k = 0
        while k < len(idx) - 1:
            a, b = idx[k], idx[k + 1]
            if 1 < b - a <= max_gap + 1:
                arr[a:b + 1] = np.linspace(arr[a], arr[b], b - a + 1)
            k += 1
    return t, cy, ar, present


def nanmedian(x):
    x = x[~np.isnan(x)]
    return float(np.median(x)) if len(x) else float("nan")


def score_frames(track: dict, frames: list[dict], cfg: dict) -> dict:
    fps = track["fps"]
    n = len(frames)
    t, cy, ar, present = interpolate(frames, fps, cfg["max_gap_s"])
    coverage = float(present.mean()) if n else 0.0
    out = {
        "clip": track["clip"], "tracker": track.get("tracker"),
        "coverage": round(coverage, 3), "n_frames": n,
        "flag": False, "events": [], "best": None,
    }
    if present.sum() < max(3, int(cfg["min_coverage"] * n)):
        out["reason"] = "insufficient tracking coverage"
        return out

    k = max(1, int(round(cfg["drop_window_s"] * fps)))
    vel = np.full(n, np.nan)
    for i in range(n - k):
        if not (math.isnan(cy[i]) or math.isnan(cy[i + k])):
            vel[i] = (cy[i + k] - cy[i]) / (k / fps)  # frame-heights per second, + = down
    cand = np.where(vel >= cfg["drop_speed_min"])[0]

    # merge consecutive candidate start frames into events
    events = []
    if len(cand):
        start = prev = int(cand[0])
        for c in cand[1:]:
            c = int(c)
            if c - prev <= 1:
                prev = c
            else:
                events.append((start, prev + k))
                start = prev = c
        events.append((start, prev + k))

    pre_w = max(1, int(round(cfg["pre_window_s"] * fps)))
    post_w = max(1, int(round(cfg["post_window_s"] * fps)))
    hold_n = int(round(cfg["low_hold_s"] * fps))
    min_post_n = int(round(cfg["min_post_s"] * fps))

    results = []
    for (s, e) in events:
        e = min(e, n - 1)
        total_drop = float(cy[e] - cy[s]) if not (math.isnan(cy[e]) or math.isnan(cy[s])) else float("nan")
        vmax = float(np.nanmax(vel[s:e + 1])) if e >= s else float("nan")
        ar_pre = nanmedian(ar[max(0, s - pre_w):s + 1])
        pre_h = [ (frames[j]["box"][3] - frames[j]["box"][1]) / track["height"]
                  for j in range(max(0, s - pre_w), s + 1) if frames[j].get("present") ]
        drop_rel = (total_drop / float(np.median(pre_h))) if pre_h and not math.isnan(total_drop) else float("nan")
        ar_post = nanmedian(ar[e:min(n, e + post_w + 1)])
        cy_post = nanmedian(cy[e:min(n, e + post_w + 1)])
        # hold window
        tail = slice(e + 1, min(n, e + 1 + hold_n))
        tail_len = tail.stop - tail.start
        clip_truncated = tail_len < hold_n
        low = np.zeros(tail_len, dtype=bool)
        for j, idx in enumerate(range(tail.start, tail.stop)):
            if math.isnan(ar[idx]):
                continue
            low[j] = (ar[idx] <= cfg["low_ar_max"]) or (cy[idx] >= cy_post - cfg["low_cy_margin"])
        low_frac = float(low.mean()) if tail_len else 0.0
        if tail_len == 0 and cfg["allow_clip_end"] and not math.isnan(ar[e]):
            # clip ends on the last drop frame: judge "low" from that frame alone
            low_frac = 1.0 if ar[e] <= cfg["low_ar_max"] else 0.0
        # stand-up check: centroid coming back up strongly within the hold window ends the "low" state
        drop_ok = (not math.isnan(total_drop)) and total_drop >= cfg["drop_min"] and vmax >= cfg["drop_speed_min"]
        shape_ok = (not math.isnan(ar_pre)) and (not math.isnan(ar_post)) and ar_pre >= cfg["ar_tall_min"] and ar_post <= cfg["ar_wide_max"]
        hold_enough = (tail_len >= hold_n) or (cfg["allow_clip_end"] and (tail_len >= min_post_n or tail_len == 0))
        hold_ok = hold_enough and low_frac >= cfg["low_hold_frac"]
        soft = (
            min(1.0, vmax / cfg["drop_speed_min"]) if not math.isnan(vmax) else 0.0,
            min(1.0, total_drop / cfg["drop_min"]) if not math.isnan(total_drop) and total_drop > 0 else 0.0,
            min(1.0, ar_pre / cfg["ar_tall_min"]) if not math.isnan(ar_pre) else 0.0,
            min(1.0, cfg["ar_wide_max"] / ar_post) if not math.isnan(ar_post) and ar_post > 0 else 0.0,
            low_frac,
        )
        results.append({
            "start_i": int(s), "end_i": int(e),
            "start_t": round(float(t[s]), 2), "end_t": round(float(t[e]), 2),
            "drop_speed_max": round(vmax, 3), "total_drop": round(total_drop, 3),
            "drop_rel_to_height": round(drop_rel, 3),
            "ar_pre": round(ar_pre, 3), "ar_post": round(ar_post, 3), "cy_post": round(cy_post, 3),
            "hold_frames": tail_len, "hold_s": round(tail_len / fps, 2),
            "clip_truncated_hold": bool(clip_truncated), "low_frac": round(low_frac, 3),
            "drop_ok": bool(drop_ok), "shape_ok": bool(shape_ok), "hold_ok": bool(hold_ok),
            "flag": bool(drop_ok and shape_ok and hold_ok),
            "soft_score": round(float(np.mean(soft)), 3),
        })
    out["events"] = results
    if results:
        flagged = [r for r in results if r["flag"]]
        best = max(flagged or results, key=lambda r: r["soft_score"])
        out["best"] = best
        out["flag"] = bool(flagged)
    out["summary"] = {
        "cy_min": round(float(np.nanmin(cy)), 3), "cy_max": round(float(np.nanmax(cy)), 3),
        "ar_max": round(float(np.nanmax(ar)), 3), "ar_min": round(float(np.nanmin(ar)), 3),
        "vel_max": round(float(np.nanmax(vel)), 3) if np.any(~np.isnan(vel)) else None,
    }
    return out


def score_track(track: dict, cfg: dict) -> dict:
    """Score every tracked person; the clip is flagged if any of them is. Returns the
    best object's result plus a per-object summary."""
    objects = track.get("objects") or {"primary": track["frames"]}
    per_obj = {}
    for oid, frames in objects.items():
        if sum(f.get("present", False) for f in frames) < 3:
            continue
        per_obj[oid] = score_frames(track, frames, cfg)
    if not per_obj:
        res = score_frames(track, track["frames"], cfg)
        res["object_id"] = track.get("meta", {}).get("primary_object")
        res["objects_scored"] = {}
        return res
    def key(oid):
        r = per_obj[oid]
        return (bool(r["flag"]), (r.get("best") or {}).get("soft_score", 0.0), r["coverage"])
    best_oid = max(per_obj, key=key)
    res = dict(per_obj[best_oid])
    res["object_id"] = best_oid
    res["objects_scored"] = {
        oid: {"flag": r["flag"], "coverage": r["coverage"],
              "soft_score": (r.get("best") or {}).get("soft_score")}
        for oid, r in per_obj.items()
    }
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tracker", default="sam", choices=["sam", "yolo"])
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()
    cfg = load_config()["score"]
    labels = {r["clip"]: r for r in load_labels()}
    rows = []
    for name, lab in labels.items():
        tp = OUT / args.tracker / f"{name}.json"
        if not tp.exists():
            continue
        res = score_track(read_json(tp), cfg)
        res["label"] = lab["label"]
        res["source"] = lab["source"]
        write_json(OUT / "scores" / args.tracker / f"{name}.json", res)
        rows.append(res)
    falls = [r for r in rows if r["label"] == "fall"]
    adls = [r for r in rows if r["label"] == "adl"]
    caught = sum(r["flag"] for r in falls)
    fa = sum(r["flag"] for r in adls)
    summary = {
        "tracker": args.tracker, "n_scored": len(rows),
        "falls_total": len(falls), "falls_caught": caught,
        "adl_total": len(adls), "adl_false_alarms": fa,
        "config": cfg,
    }
    write_json(OUT / "scores" / args.tracker / "_summary.json", summary)
    if not args.quiet:
        for r in rows:
            b = r.get("best") or {}
            print(f"{r['clip']:22s} {r['label']:4s} flag={str(r['flag']):5s} obj={str(r.get('object_id'))}/{len(r.get('objects_scored') or {})} cov={r['coverage']:.2f} "
                  f"v={b.get('drop_speed_max','-')} drop={b.get('total_drop','-')} "
                  f"arpre={b.get('ar_pre','-')} arpost={b.get('ar_post','-')} low={b.get('low_frac','-')} "
                  f"hold={b.get('hold_s','-')}s{'*' if b.get('clip_truncated_hold') else ''} "
                  f"[{'D' if b.get('drop_ok') else '.'}{'S' if b.get('shape_ok') else '.'}{'H' if b.get('hold_ok') else '.'}]")
    print(json.dumps({k: v for k, v in summary.items() if k != "config"}))


if __name__ == "__main__":
    main()
