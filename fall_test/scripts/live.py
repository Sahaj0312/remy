"""Live pipeline skeleton: webcam / RTSP / file -> always-on YOLO pose -> rolling buffer ->
on a candidate drop: wait no_getup_s (cancel if the person gets up) -> SAM 3.1 on the buffered clip ->
fall score v2 -> Spark triage -> print the event, save clip + snapshot + json under events/.

  python scripts/live.py --source 0                                   # laptop webcam
  python scripts/live.py --source rtsp://user:pass@ip:554/stream2     # Tapo C200
  python scripts/live.py --source clips/gmdcsa_S1_Fall_12.mp4 --realtime   # simulate with a file
  --no-api  : stop after the YOLO trigger (no SAM/Spark calls, no cost)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    META_BASE_URL, ROOT, api_key, empty_frame, link_score, load_config, sam_cost_usd,
    suppress_duplicates, video_info, write_json,
)
from fall_score import score_track  # noqa: E402

EVENTS = ROOT / "events"


def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] {msg}", flush=True)


class OnlineTracks:
    """Greedy IoU/overlap linker fed one YOLO tick at a time; every track gets a frame per tick."""

    def __init__(self, fps: float, H: int, W: int, max_gap_s: float, min_score: float, keep_s: float):
        self.fps, self.H, self.W = fps, H, W
        self.max_gap = int(round(max_gap_s * fps)); self.min_score = min_score
        self.keep = int(round(keep_s * fps))
        self.tracks: dict[str, dict] = {}       # id -> {"frames": deque, "last_i": int, "last_box": list}
        self.next_id = 1; self.tick = 0

    def update(self, dets: list[dict]) -> None:
        i = self.tick
        cands = [(link_score(tr["last_box"], d["box"]), tid, j)
                 for tid, tr in self.tracks.items() if i - tr["last_i"] <= self.max_gap
                 for j, d in enumerate(dets)]
        cands = [c for c in cands if c[0] >= self.min_score]
        cands.sort(reverse=True)
        used_t, used_d = set(), set()
        for sc, tid, j in cands:
            if tid in used_t or j in used_d:
                continue
            used_t.add(tid); used_d.add(j)
            self._push(tid, i, dets[j])
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
        tr = self.tracks[tid]
        x1, y1, x2, y2 = d["box"]
        tr["frames"].append({"i": i, "t": round(i / self.fps, 3), "present": True, "box": d["box"],
                             "cx": d["cx_px"] / self.W, "cy": d["cy_px"] / self.H,
                             "ar": (y2 - y1) / max(x2 - x1, 1e-6), "mask_area": None, "conf": d["conf"]})
        tr["last_i"] = i; tr["last_box"] = d["box"]

    def as_track(self, name: str) -> dict:
        """Uniform-time track dict for the scorer (frames re-indexed from 0 per object)."""
        objects = {}
        for tid, tr in self.tracks.items():
            frames = list(tr["frames"])
            if not frames:
                continue
            base = frames[0]["i"]
            objects[tid] = [{**f, "i": f["i"] - base, "t": round((f["i"] - base) / self.fps, 3)} for f in frames]
        primary = max(objects, key=lambda k: sum(f["present"] for f in objects[k])) if objects else None
        return {"clip": name, "fps": self.fps, "height": self.H, "width": self.W, "tracker": "live-yolo",
                "frames": objects.get(primary, []), "objects": objects}


def yolo_detect(model, img, ycfg, tcfg) -> list[dict]:
    res = model.predict(img, verbose=False, conf=ycfg["conf"], imgsz=ycfg["imgsz"], classes=[0])[0]
    dets = []
    if res.boxes is not None and len(res.boxes):
        boxes = res.boxes.xyxy.cpu().numpy(); confs = res.boxes.conf.cpu().numpy()
        for j in range(len(boxes)):
            x1, y1, x2, y2 = boxes[j]; cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            if res.keypoints is not None and len(res.keypoints) > j and res.keypoints.conf is not None:
                kxy = res.keypoints.xy[j].cpu().numpy(); kc = res.keypoints.conf[j].cpu().numpy()
                good = kc >= ycfg["kp_conf"]
                if good.sum() >= 3:
                    cx, cy = kxy[good, 0].mean(), kxy[good, 1].mean()
            dets.append({"box": [float(v) for v in (x1, y1, x2, y2)], "cx_px": float(cx), "cy_px": float(cy),
                         "mask_area_px": None, "conf": round(float(confs[j]), 3)})
    return suppress_duplicates(dets, tcfg["dup_max_overlap"])


def resample(frames: list[tuple[float, np.ndarray]], fps: float) -> list[tuple[float, np.ndarray]]:
    """Nearest-frame resample of (t, frame) pairs onto a uniform fps grid (sources may be slower or jittery)."""
    if len(frames) < 2:
        return frames
    t0, t1 = frames[0][0], frames[-1][0]
    ts = np.array([t for t, _ in frames])
    out = []
    g = t0
    while g <= t1 + 1e-6:
        j = int(np.argmin(np.abs(ts - g)))
        out.append((g, frames[j][1])); g += 1 / fps
    return out


def write_clip(frames: list[tuple[float, np.ndarray]], fps: float, dst: Path):
    frames = resample(frames, fps)
    h, w = frames[0][1].shape[:2]
    proc = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
                             "-r", f"{fps:g}", "-i", "-", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)], stdin=subprocess.PIPE)
    for _, fr in frames:
        proc.stdin.write(fr.tobytes())
    proc.stdin.close(); proc.wait()


class Pipeline:
    def __init__(self, args, cfg):
        self.args, self.cfg = args, cfg
        self.lock = threading.Lock()
        self.busy = False
        self.events: list[dict] = []

    def handle(self, clip_frames, fps, drop_wall: float, drop_virtual: float, snapshot, label: str):
        """Runs in a worker thread: SAM -> score -> Spark -> event."""
        from openai import AsyncOpenAI, OpenAI
        import sam_track
        from spark_verify import run_triage
        t_trigger = time.time()
        ev_dir = EVENTS / f"{datetime.fromtimestamp(drop_wall).strftime('%Y%m%d_%H%M%S')}_{label}"
        ev_dir.mkdir(parents=True, exist_ok=True)
        clip = ev_dir / "clip.mp4"
        write_clip(clip_frames, fps, clip)
        cv2.imwrite(str(ev_dir / "snapshot.jpg"), snapshot)
        timings = {"drop_wall": drop_wall, "trigger_wall": t_trigger, "clip_s": len(clip_frames) / fps}
        info = video_info(clip)
        if self.args.no_api:
            log(f"[dry run] clip saved to {ev_dir} ({info['n_frames']} frames); SAM/Spark skipped")
            write_json(ev_dir / "event.json", {"label": label, "dry_run": True, "timings": timings})
            return
        try:
            async def run_sam():
                async with AsyncOpenAI(base_url=META_BASE_URL, api_key=api_key(), timeout=600) as api:
                    return await sam_track.segment_video(api, clip, "one_bit", False)
            result, sam_timing = asyncio.run(run_sam())
            timings["sam_done_wall"] = time.time()
            frames, meta, objects = sam_track.result_to_frames(result, info["n_frames"], info["fps"], info["width"], info["height"])
            track = {"clip": ev_dir.name, "fps": info["fps"], "height": info["height"], "width": info["width"],
                     "tracker": "sam-3.1", "frames": frames, "objects": objects, "meta": meta}
            write_json(ev_dir / "sam_track.json", track)
            score = score_track(track, self.cfg["score_v2"], "v2")
            write_json(ev_dir / "score.json", score)
            timings["score_done_wall"] = time.time()
            cost = sam_cost_usd(info["n_frames"])
            if not score["flag"]:
                log(f"SAM says no fall (hold={score.get('hold')}, best={score.get('best')}) -> no event. "
                    f"SAM {sam_timing['total_s']:.1f}s, ${cost:.4f}")
                write_json(ev_dir / "event.json", {"label": label, "fall": False, "score": score, "timings": timings, "cost_usd": cost})
                return
            client = OpenAI(base_url=META_BASE_URL, api_key=api_key(), timeout=300)
            event_time = datetime.fromtimestamp(drop_wall).strftime("%-I:%M%p").lower()
            try:
                triage = run_triage(client, self.cfg["spark"], score["best"], info["fps"], info["n_frames"], clip, event_time, "video")
            except Exception as e:
                triage = {"verdict": {"urgency": "unknown", "summary_for_family": f"(Spark failed: {e})",
                                      "what_happened": "", "is_lying_still": None}, "cost_usd": 0.0, "latency_s": None}
            timings["spark_done_wall"] = time.time()
            timings["event_wall"] = time.time()
            timings["drop_to_event_s"] = round(timings["event_wall"] - drop_wall, 2)
            timings["sam_s"] = round(sam_timing["total_s"], 2); timings["spark_s"] = triage.get("latency_s")
            cost += triage.get("cost_usd", 0.0)
            v = triage["verdict"]
            event = {"label": label, "fall": True, "time": datetime.fromtimestamp(drop_wall).isoformat(timespec="seconds"),
                     "urgency": v.get("urgency"), "summary_for_family": v.get("summary_for_family"),
                     "what_happened": v.get("what_happened"), "is_lying_still": v.get("is_lying_still"),
                     "score": score["best"], "hold": score.get("hold"), "timings": timings, "cost_usd": round(cost, 4),
                     "spark": {k: triage.get(k) for k in ("latency_s", "usage", "cost_usd", "raw_text")}}
            write_json(ev_dir / "event.json", event)
            log(f"EVENT  {event['time']}  urgency={event['urgency']}  lying_still={event['is_lying_still']}  "
                f"drop->event {timings['drop_to_event_s']}s (wait {t_trigger - drop_wall:.1f}s, SAM {timings['sam_s']}s, "
                f"Spark {timings['spark_s']}s)  ${cost:.4f}\n        {event['summary_for_family']}")
            self.events.append(event)
        except Exception as e:
            log(f"pipeline error: {type(e).__name__}: {e}")
            write_json(ev_dir / "event.json", {"label": label, "error": repr(e), "timings": timings})
        finally:
            with self.lock:
                self.busy = False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, help="device index, RTSP/HTTP URL, or video file")
    p.add_argument("--realtime", action="store_true", help="pace a file source at its native fps")
    p.add_argument("--yolo-fps", type=float, default=5.0)
    p.add_argument("--buffer-fps", type=float, default=10.0)
    p.add_argument("--buffer-s", type=float, default=10.0)
    p.add_argument("--pre-s", type=float, default=4.0, help="seconds of buffer before the drop to send to SAM")
    p.add_argument("--post-s", type=float, default=None, help="wait after the drop before calling SAM (default: no_getup_s)")
    p.add_argument("--cooldown-s", type=float, default=20.0)
    p.add_argument("--max-width", type=int, default=640)
    p.add_argument("--duration", type=float, default=0, help="stop after N seconds (0 = until the source ends / Ctrl-C)")
    p.add_argument("--no-api", action="store_true")
    p.add_argument("--display", action="store_true")
    p.add_argument("--label", default="live")
    args = p.parse_args()
    cfg = load_config()
    post_s = args.post_s if args.post_s is not None else cfg["score_v2"]["no_getup_s"]
    trig_cfg = {**cfg["score_v2"], **cfg["trigger"]}

    src = int(args.source) if args.source.isdigit() else args.source
    is_file = isinstance(src, str) and Path(src).exists()
    cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG) if (isinstance(src, str) and src.startswith(("rtsp", "http"))) else cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"cannot open source {args.source}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    log(f"source opened: {args.source} ({src_fps:.1f} fps reported), yolo {args.yolo_fps:g} fps, buffer {args.buffer_s:g}s @ {args.buffer_fps:g} fps, "
        f"wait {post_s:g}s after a drop, {'DRY RUN' if args.no_api else 'SAM+Spark live'}")
    from ultralytics import YOLO
    model = YOLO(cfg["yolo"]["model"])
    EVENTS.mkdir(exist_ok=True)

    pipe = Pipeline(args, cfg)
    buffer: deque = deque(maxlen=int(args.buffer_s * args.buffer_fps) + 5)
    tracks = None
    last_buf = last_yolo = -1e9
    pending = None            # {"drop_v": virtual t, "drop_wall": wall t, "track_id": id}
    cooldown_until = -1e9
    t0_wall = time.time(); i = 0; yolo_ms = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t_v = i / src_fps if is_file else time.time() - t0_wall       # virtual clock for files
        if is_file and args.realtime:
            lag = t_v - (time.time() - t0_wall)
            if lag > 0:
                time.sleep(lag)
        wall = time.time()
        if frame.shape[1] > args.max_width:
            s = args.max_width / frame.shape[1]
            frame = cv2.resize(frame, (args.max_width, int(frame.shape[0] * s)), interpolation=cv2.INTER_AREA)
        H, W = frame.shape[:2]
        if tracks is None:
            tracks = OnlineTracks(args.yolo_fps, H, W, cfg["tracking"]["max_gap_s"], cfg["tracking"]["min_link_score"], keep_s=8.0)
        if t_v - last_buf >= 1 / args.buffer_fps - 1e-6:
            buffer.append((t_v, frame.copy())); last_buf = t_v
        if t_v - last_yolo >= 1 / args.yolo_fps - 1e-6:
            last_yolo = t_v
            ty = time.perf_counter()
            dets = yolo_detect(model, frame, cfg["yolo"], cfg["tracking"])
            yolo_ms.append((time.perf_counter() - ty) * 1000)
            tracks.update(dets)
            if args.display:
                for d in dets:
                    x1, y1, x2, y2 = [int(v) for v in d["box"]]; cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.imshow("live", frame); cv2.waitKey(1)
            # ---- trigger / get-up logic on the YOLO tracks
            tr = tracks.as_track("live")
            if pending is None and t_v >= cooldown_until and not pipe.busy and tr["objects"]:
                res = score_track(tr, trig_cfg, "trigger")
                if res["flag"]:
                    b = res["best"]
                    drop_v = t_v - (tr["objects"][res["object_id"]][-1]["t"] - b["end_t"])
                    pending = {"drop_v": drop_v, "drop_wall": wall - (t_v - drop_v), "obj": res["object_id"], "best": b}
                    log(f"candidate drop on track {res['object_id']}: drop_rel {b['drop_rel']:.2f} body lengths, "
                        f"{b['drop_speed_rel_max']:.2f} bl/s, ended {t_v - drop_v:.1f}s ago; waiting {post_s:g}s for a get-up")
            elif pending is not None:
                res = score_track(tr, {**cfg["score_v2"]}, "v2")
                if res.get("hold") == "failed" and res["best"] and abs(res["best"]["end_t"] - (pending["drop_v"] - (t_v - tr["frames"][-1]["t"] if tr["frames"] else 0))) < 2.0:
                    log("person got back up within the wait -> no event (YOLO)")
                    pending = None; cooldown_until = t_v + 5
                elif t_v - pending["drop_v"] >= post_s:
                    start_v = pending["drop_v"] - args.pre_s
                    clip_frames = [(t, f) for (t, f) in buffer if t >= start_v]
                    snap = min(buffer, key=lambda tf: abs(tf[0] - pending["drop_v"]))[1]
                    log(f"sending {clip_frames[-1][0] - clip_frames[0][0]:.1f}s clip ({len(clip_frames)} buffered frames) to SAM")
                    with pipe.lock:
                        pipe.busy = True
                    threading.Thread(target=pipe.handle, args=(clip_frames, args.buffer_fps, pending["drop_wall"], pending["drop_v"], snap, args.label), daemon=True).start()
                    pending = None; cooldown_until = t_v + args.cooldown_s
        i += 1
        if args.duration and t_v >= args.duration:
            break
    cap.release()
    # let the worker finish
    while pipe.busy:
        time.sleep(0.2)
    log(f"done: {i} frames read, YOLO {np.mean(yolo_ms):.0f} ms/frame mean, {len(pipe.events)} event(s)")
    for e in pipe.events:
        print(json.dumps({k: e[k] for k in ("time", "urgency", "summary_for_family", "is_lying_still", "cost_usd")}, indent=None))
        print("   timings:", {k: v for k, v in e["timings"].items() if k.endswith("_s")})


if __name__ == "__main__":
    main()
