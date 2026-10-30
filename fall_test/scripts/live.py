"""Live fall pipeline v2.

capture (webcam / RTSP / file) -> YOLO pose 5 fps + greedy tracks -> rolling 10 fps buffer
  -> candidate drop: clip cut at once, SAM 3.1 runs while YOLO watches for a get-up (no fixed wait)
     -> score v2 on the SAM track -> EVENT
  -> floor-zone rule: on floor/rug (not sofa/bed), low, barely moving for N s -> EVENT without a drop
EVENT -> Spark triage (inline base64 burst, async) + still-down watcher (YOLO + zones)
      -> spoken check-in (Kokoro prompt on camera or laptop) -> 15 s mic -> transcribe -> Spark classify
      -> okay: calm WhatsApp; needs_help / twice unclear or no answer: urgent WhatsApp with snapshot
      -> still down 60 s after an okay: check in again.
Every step is timestamped in events/<id>/log.txt and event.json.

  python scripts/live.py --source tapo
  python scripts/live.py --source clips/gmdcsa_S1_Fall_12.mp4 --realtime --audio-out laptop --alerts-dry-run
"""
from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import costs  # noqa: E402
from common import (  # noqa: E402
    META_BASE_URL, ROOT, api_key, empty_frame, link_score, load_config, redact, sam_cost_usd, suppress_duplicates,
    video_info, write_json,
)
from fall_score import score_track  # noqa: E402
from zones import Zones  # noqa: E402

EVENTS = ROOT / "events"


def now_str() -> str:
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


# ----------------------------------------------------------------------------- tracking (from v1)
class OnlineTracks:
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


def yolo_detect(model, img, ycfg, tcfg):
    res = model.predict(img, verbose=False, conf=ycfg["conf"], imgsz=ycfg["imgsz"], classes=[0])[0]
    dets = []
    if res.boxes is not None and len(res.boxes):
        boxes = res.boxes.xyxy.cpu().numpy(); confs = res.boxes.conf.cpu().numpy()
        for j in range(len(boxes)):
            x1, y1, x2, y2 = boxes[j]; cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            if res.keypoints is not None and len(res.keypoints) > j and res.keypoints.conf is not None:
                kxy = res.keypoints.xy[j].cpu().numpy(); kc = res.keypoints.conf[j].cpu().numpy(); good = kc >= ycfg["kp_conf"]
                if good.sum() >= 3:
                    cx, cy = kxy[good, 0].mean(), kxy[good, 1].mean()
            dets.append({"box": [float(v) for v in (x1, y1, x2, y2)], "cx_px": float(cx), "cy_px": float(cy), "mask_area_px": None,
                         "conf": round(float(confs[j]), 3)})
    return suppress_duplicates(dets, tcfg["dup_max_overlap"])


def resample(frames, fps):
    if len(frames) < 2:
        return frames
    ts = np.array([t for t, _ in frames]); out = []; g = frames[0][0]
    while g <= frames[-1][0] + 1e-6:
        out.append((g, frames[int(np.argmin(np.abs(ts - g)))][1])); g += 1 / fps
    return out


def write_clip(frames, fps, dst: Path):
    frames = resample(frames, fps); h, w = frames[0][1].shape[:2]
    proc = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{fps:g}",
                             "-i", "-", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
                             "-movflags", "+faststart", str(dst)], stdin=subprocess.PIPE)
    for _, fr in frames:
        proc.stdin.write(fr.tobytes())
    proc.stdin.close(); proc.wait()


# ----------------------------------------------------------------------------- shared state
class State:
    """What the capture loop knows right now; flows read it under the lock."""

    def __init__(self, zones: Zones | None, yolo_fps: float, H: int):
        self.lock = threading.Lock(); self.zones = zones; self.fps = yolo_fps; self.H = H
        self.history: deque = deque(maxlen=int(60 * yolo_fps))     # per YOLO tick
        self.buffer: deque | None = None; self.t = 0.0; self.wall = time.time(); self.t0_wall = time.time()

    def push(self, t, wall, det):
        rec = {"t": t, "wall": wall, "present": det is not None}
        if det is not None:
            x1, y1, x2, y2 = det["box"]
            rec.update(box=det["box"], ar=(y2 - y1) / max(x2 - x1, 1e-6), cy=det["cy_px"] / self.H, cx=det["cx_px"] / (self.zones.w if self.zones else 1),
                       body=max(y2 - y1, x2 - x1) / self.H, zone=self.zones.classify_box(det["box"]) if self.zones else None)
        with self.lock:
            self.history.append(rec); self.t = t; self.wall = wall

    def recent(self, seconds: float) -> list[dict]:
        with self.lock:
            return [r for r in self.history if r["t"] >= self.t - seconds]

    def latest(self):
        with self.lock:
            return self.history[-1] if self.history else None

    def movement_bl(self, seconds: float = 2.0) -> float | None:
        rs = [r for r in self.recent(seconds) if r["present"]]
        if len(rs) < 3:
            return None
        xs = np.array([r["cx"] for r in rs]); ys = np.array([r["cy"] for r in rs]); body = max(np.median([r["body"] for r in rs]), 0.15)
        return float(max(xs.max() - xs.min(), ys.max() - ys.min()) / body)

    def down_now(self, low_ar_max: float) -> bool:
        r = self.latest()
        return bool(r and r["present"] and r["ar"] <= low_ar_max and not (r.get("zone") or {}).get("on_furniture"))


# ----------------------------------------------------------------------------- event flow
class Flow(threading.Thread):
    def __init__(self, kind, state: State, args, cfg, model_info, drop=None, snapshot=None):
        super().__init__(daemon=True)
        self.kind, self.state, self.args, self.cfg, self.drop, self.snapshot = kind, state, args, cfg, drop, snapshot
        self.W, self.H = model_info
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.dir = EVENTS / f"{stamp}_{args.label}_{kind}"; self.dir.mkdir(parents=True, exist_ok=True)
        self.t_start = time.time(); self.lines = []; self.timings = {}; self.cost = 0.0
        self.triage = None; self.got_up_at = None; self.still_down_at = None; self.done = False
        self.result = {"kind": kind, "event": False}

    # ---- logging
    def log(self, msg):
        line = f"[{now_str()}] [+{time.time() - self.t_start:6.2f}s] {msg}"
        print(line, flush=True); self.lines.append(line)
        with open(self.dir / "log.txt", "a") as f:
            f.write(line + "\n")

    def mark(self, key):
        self.timings[key] = round(time.time() - self.t_start, 2)

    def save(self):
        self.result.update(timings=self.timings, cost_usd=round(self.cost, 4), triage=self.triage, got_up_at_s=self.got_up_at,
                           still_down_at_s=self.still_down_at)
        write_json(self.dir / "event.json", self.result)

    # ---- helpers on the shared state
    def person_got_up(self, since_t: float) -> bool:
        rs = [r for r in self.state.recent(60) if r["present"] and r["t"] >= since_t]
        if len(rs) < 4:
            return False
        head = rs[: max(2, len(rs) // 4)]
        down_cy = float(np.median([r["cy"] for r in head])); body = max(float(np.median([r["body"] for r in head])), 0.15)
        n = self.cfg["monitor"]["getup_confirm_ticks"]; run = 0
        for r in rs[len(head):]:
            up = r["cy"] <= down_cy - self.cfg["score_v2"]["getup_rise_rel"] * body or r["ar"] >= 2.0
            run = run + 1 if up else 0
            if run >= n:
                return True
        return False

    def clip_from_buffer(self, start_t, end_t=None):
        with self.state.lock:
            fr = [(t, f) for (t, f) in self.state.buffer if t >= start_t and (end_t is None or t <= end_t)]
        return fr

    # ---- main
    def run(self):
        try:
            if self.kind == "drop":
                ok = self.confirm_drop()
                if not ok:
                    self.save(); self.done = True; return
            else:
                self.log(f"floor-zone rule: on {self.drop['zone']} , low (ar {self.drop['ar']:.2f}), moved {self.drop['move']:.2f} body lengths "
                         f"in 2 s, for {self.drop['for_s']:.0f}s -> EVENT without a drop")
            self.raise_event()
        except Exception as e:
            self.log(f"flow error: {type(e).__name__}: {e}")
            self.result["error"] = repr(e)
        finally:
            self.save(); self.done = True

    def confirm_drop(self) -> bool:
        d = self.drop
        self.log(f"candidate drop: {d['best']['drop_rel']:.2f} body lengths at {d['best']['drop_speed_rel_max']:.2f} bl/s (track {d['obj']})")
        # let post_s of post-drop frames into the buffer (the ends-wide check needs ~1 s), then cut and send at once
        while self.state.t < d["drop_v"] + self.args.post_s:
            time.sleep(0.05)
        frames = self.clip_from_buffer(d["drop_v"] - self.args.pre_s)
        clip = self.dir / "clip.mp4"; write_clip(frames, self.args.buffer_fps, clip); self.mark("clip_written")
        cv2.imwrite(str(self.dir / "snapshot.jpg"), self.snapshot)
        info = video_info(clip)
        self.log(f"clip {info['n_frames']} frames ({info['n_frames']/info['fps']:.1f}s) written, SAM upload starting while YOLO watches for a get-up")
        if self.args.no_api:
            self.log("[dry run] SAM skipped"); self.result.update(dry_run=True); return False
        import sam_track

        async def run_sam():
            async with sam_track.AsyncOpenAI(base_url=META_BASE_URL, api_key=api_key(), timeout=600) as api:
                return await sam_track.segment_video(api, clip, "one_bit", False)
        self.mark("sam_call_start")
        result, st = asyncio.run(run_sam())
        self.timings["sam_upload_s"] = round(st["upload_s"], 2); self.timings["sam_total_s"] = round(st["total_s"], 2); self.mark("sam_done")
        self.cost += sam_cost_usd(info["n_frames"]); costs.add("sam-live", sam_cost_usd(info["n_frames"]), self.dir.name)
        frames_t, meta, objects = sam_track.result_to_frames(result, info["n_frames"], info["fps"], info["width"], info["height"], use_masks=False)
        self.mark("sam_parsed")
        track = {"clip": self.dir.name, "fps": info["fps"], "height": info["height"], "width": info["width"], "n_frames": info["n_frames"],
                 "tracker": "sam-3.1", "frames": frames_t, "objects": objects, "meta": meta}
        score = score_track(track, self.cfg["score_v2"], "v2"); self.mark("scored")
        threading.Thread(target=lambda: (write_json(self.dir / "sam_track.json", track), write_json(self.dir / "score.json", score)), daemon=True).start()
        self.result["score"] = score.get("best"); self.result["sam_flag"] = score["flag"]
        if self.person_got_up(d["drop_v"]):
            self.log(f"SAM {'confirms' if score['flag'] else 'rejects'} the drop, but the person is already up -> no event")
            return False
        if not score["flag"]:
            self.log(f"SAM: not a fall (best={ {k: (score.get('best') or {}).get(k) for k in ('drop_rel','drop_speed_rel_max','ar_post')} }) -> no event")
            return False
        self.log(f"SAM confirms: drop {score['best']['drop_rel']:.2f} bl at {score['best']['drop_speed_rel_max']:.2f} bl/s, ends wide (ar {score['best']['ar_post']:.2f})")
        return True

    def raise_event(self):
        self.mark("event"); self.result["event"] = True
        self.event_wall = time.time(); self.event_t = self.state.t
        self.result["time"] = datetime.fromtimestamp(self.drop.get("drop_wall", self.event_wall)).isoformat(timespec="seconds")
        when = datetime.fromtimestamp(self.drop.get("drop_wall", self.event_wall)).strftime("%-I:%M%p").lower()
        self.when = when
        if not (self.dir / "snapshot.jpg").exists() and self.snapshot is not None:
            cv2.imwrite(str(self.dir / "snapshot.jpg"), self.snapshot)
        self.log(f"EVENT raised ({self.kind}) at {when}; drop->event {self.timings['event']:.2f}s")
        threading.Thread(target=self.run_triage, daemon=True).start()
        threading.Thread(target=self.watch_still_down, daemon=True).start()
        self.run_checkin_and_alert()

    # ---- Spark triage (async)
    def run_triage(self):
        try:
            if self.args.no_api:
                return
            from spark_verify import run_triage
            from openai import OpenAI
            clip = self.dir / "clip.mp4"
            if not clip.exists():        # floor-still: cut the last 6 s from the buffer
                write_clip(self.clip_from_buffer(self.state.t - 6.0), self.args.buffer_fps, clip)
            info = video_info(clip)
            best = self.result.get("score") or {"start_i": max(0, info["n_frames"] - int(3 * info["fps"])), "end_i": info["n_frames"] - 1,
                                                "drop_rel": 0.0, "drop_speed_rel_max": 0.0, "time_to_floor_s": 0.0, "shape_ok": True,
                                                "ar_post": 0.0, "hold": "unknown", "no_getup_observed_s": float(self.drop.get("for_s", 0))}
            spark_cfg = dict(self.cfg["spark"]); spark_cfg["burst_after_s"] = spark_cfg.get("live_burst_after_s", 1.0)
            client = OpenAI(base_url=META_BASE_URL, api_key=api_key(), timeout=120)
            t0 = time.time()
            doc = run_triage(client, spark_cfg, best, info["fps"], info["n_frames"], clip, self.when, "inline")
            self.triage = doc["verdict"]; self.timings["spark_total_s"] = round(time.time() - t0, 2); self.mark("triage_done")
            self.cost += doc["cost_usd"]; costs.add("spark-triage", doc["cost_usd"], self.dir.name)
            self.log(f"Spark triage ({self.timings['spark_total_s']}s incl. burst cut, inline base64): urgency={self.triage['urgency']} "
                     f"lying_still={self.triage['is_lying_still']} :: {self.triage['summary_for_family']}")
        except Exception as e:
            self.log(f"Spark triage failed: {type(e).__name__}: {e}")
            self.triage = {"urgency": "unknown", "summary_for_family": "", "what_happened": "", "is_lying_still": None}

    def summary_text(self, wait_s=12.0) -> str:
        t0 = time.time()
        while self.triage is None and time.time() - t0 < wait_s:
            time.sleep(0.2)
        if self.triage and self.triage.get("summary_for_family"):
            return self.triage["summary_for_family"]
        return f"Possible fall detected at {self.when}."

    # ---- still-down watcher
    def watch_still_down(self):
        m = self.cfg["monitor"]; t_event = self.event_t
        while not self.done:
            time.sleep(0.5)
            if self.got_up_at is None and self.person_got_up(self.drop.get("drop_v", t_event)):
                self.got_up_at = round(time.time() - self.t_start, 2); self.mark("got_up")
                self.log(f"person got back up ({self.state.t - t_event:.0f}s after the event)")
                return
            down_for = self.state.t - t_event
            if self.still_down_at is None and down_for >= m["still_down_s"] and self.state.down_now(m["low_ar_max"]):
                self.still_down_at = round(time.time() - self.t_start, 2); self.mark("still_down")
                self.log(f"STILL DOWN: no get-up {m['still_down_s']}s after the event (movement {self.state.movement_bl():.2f} bl / 2 s)")

    # ---- check-in + alerts
    def run_checkin_and_alert(self):
        from alerts import WhatsApp
        from checkin import ask_once
        ck = self.cfg["checkin"]; wa = WhatsApp(dry_run=self.args.alerts_dry_run)
        said, outcome, rounds = None, None, []
        if self.args.no_checkin:
            self.log("check-in skipped (--no-checkin)")
            outcome = "skipped"
        else:
            for attempt in range(1, ck["retries"] + 2):
                self.mark(f"checkin{attempt}_start")
                r = ask_once(ck, self.args.audio_out, self.dir, f"r{attempt}", log=self.log); rounds.append(r)
                self.mark(f"checkin{attempt}_done")
                if r["transcript"].get("transcript"):
                    said = r["transcript"]["transcript"]
                if r["status"] in ("okay", "needs_help"):
                    outcome = r["status"]; break
                if self.got_up_at is not None:
                    outcome = "got_up"; break
                outcome = r["status"]
                if attempt <= ck["retries"]:
                    self.log(f"check-in {attempt}: {r['status']} -> asking once more")
        self.result.update(checkin_rounds=rounds, said=said, outcome=outcome)
        summary = self.summary_text()
        snap = self.dir / "snapshot.jpg"
        if outcome == "okay":
            resp = wa.calm(summary, said, self.when); self.mark("alert_sent")
            self.log(f"ALERT calm -> {len(resp)} recipient(s){' [dry run]' if wa.dry else ''}: she says she's okay")
            self.result["alert"] = {"kind": "calm", "responses": resp}
            self.recheck_after_okay(wa, snap)
        elif outcome == "got_up":
            resp = wa.calm(summary + " She got back up on her own.", said, self.when); self.mark("alert_sent")
            self.log(f"ALERT calm -> {len(resp)} recipient(s){' [dry run]' if wa.dry else ''}: got up")
            self.result["alert"] = {"kind": "calm", "responses": resp}
        elif outcome == "skipped":
            self.result["alert"] = None
        else:
            reason = {"needs_help": "She asked for help.", "no_answer": "No answer to two check-ins.",
                      "unclear": "Two check-ins, no clear answer."}.get(outcome, outcome)
            resp = wa.urgent(summary, said, self.when, snap, reason); self.mark("alert_sent")
            self.log(f"ALERT urgent -> {len(resp)} recipient(s){' [dry run]' if wa.dry else ''}: {reason}")
            self.result["alert"] = {"kind": "urgent", "reason": reason, "responses": resp}
        self.save()
        # keep the watcher alive a while for the still-down line, then finish
        t0 = time.time()
        while time.time() - t0 < self.cfg["monitor"]["still_down_s"] + 5 and self.got_up_at is None and self.still_down_at is None:
            time.sleep(0.5)

    def recheck_after_okay(self, wa, snap):
        from checkin import ask_once
        ck = self.cfg["checkin"]; limit = ck["recheck_after_s"]
        while self.got_up_at is None:
            if self.state.t - self.event_t >= limit:
                if not self.state.down_now(self.cfg["monitor"]["low_ar_max"]):
                    self.log(f"{limit}s after the event: not down any more -> no re-check"); return
                self.log(f"still down {limit}s after saying okay -> checking in again")
                r = ask_once(ck, self.args.audio_out, self.dir, "recheck", log=self.log)
                self.result.setdefault("checkin_rounds", []).append(r)
                if r["status"] == "okay":
                    self.log("re-check: okay")
                    wa.send_text  # no new alert on a second okay
                else:
                    reason = "Still on the floor a minute after saying she was okay; " + (
                        "asked for help." if r["status"] == "needs_help" else "no clear answer this time.")
                    resp = wa.urgent(self.summary_text(), r["transcript"].get("transcript") or self.result.get("said"), self.when, snap, reason)
                    self.log(f"ALERT urgent (re-check) -> {len(resp)} recipient(s){' [dry run]' if wa.dry else ''}: {reason}")
                    self.result["alert_recheck"] = {"kind": "urgent", "reason": reason, "responses": resp}
                return
            time.sleep(0.5)


# ----------------------------------------------------------------------------- main loop
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--realtime", action="store_true")
    p.add_argument("--yolo-fps", type=float, default=5.0)
    p.add_argument("--buffer-fps", type=float, default=10.0)
    p.add_argument("--buffer-s", type=float, default=12.0)
    p.add_argument("--pre-s", type=float, default=4.0)
    p.add_argument("--post-s", type=float, default=1.0, help="seconds after the drop end to include before cutting the SAM clip (shape check needs ~1 s)")
    p.add_argument("--cooldown-s", type=float, default=30.0)
    p.add_argument("--max-width", type=int, default=640)
    p.add_argument("--duration", type=float, default=0)
    p.add_argument("--no-api", action="store_true")
    p.add_argument("--no-checkin", action="store_true")
    p.add_argument("--audio-out", default=None, help="camera | laptop (default: config audio.output)")
    p.add_argument("--alerts-dry-run", action="store_true")
    p.add_argument("--label", default="live")
    p.add_argument("--save-candidates", action="store_true")
    p.add_argument("--no-zones", action="store_true", help="ignore zones.json (e.g. when simulating with a clip from another room)")
    args = p.parse_args()
    cfg = load_config(); trig_cfg = {**cfg["score_v2"], **cfg["trigger"]}; mon = cfg["monitor"]

    src = int(args.source) if args.source.isdigit() else args.source
    if src == "tapo":
        from common import tapo_rtsp
        src = tapo_rtsp(2)
    is_file = isinstance(src, str) and Path(src).exists()
    if isinstance(src, str) and src.startswith(("rtsp", "http")):
        from rtsp_reader import FFmpegFrames
        rd = FFmpegFrames(src, max_width=args.max_width); src_fps = rd.src_fps if 0 < rd.src_fps <= 60 else 15.0
        class _Cap:
            def read(self): return rd.read()
            def release(self): rd.close()
        cap = _Cap()
    else:
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise SystemExit(f"cannot open source {args.source}")
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    from ultralytics import YOLO
    model = YOLO(cfg["yolo"]["model"]); EVENTS.mkdir(exist_ok=True)
    print(f"[{now_str()}] source {redact(str(args.source))} ({src_fps:.1f} fps), yolo {args.yolo_fps:g} fps, buffer {args.buffer_s:g}s, "
          f"{'DRY RUN' if args.no_api else 'SAM+Spark live'}, audio out {args.audio_out or cfg['audio']['output']}, "
          f"alerts {'DRY' if args.alerts_dry_run else cfg['alerts']['channel']}, floor rule {mon['floor_still_s']}s, still-down {mon['still_down_s']}s", flush=True)

    state = None; tracks = None; buffer = deque(maxlen=int(args.buffer_s * args.buffer_fps) + 5)
    last_buf = last_yolo = -1e9; last_drop_v = -1e9; last_cancelled_v = None; cooldown_until = -1e9
    pending = None; flows: list[Flow] = []; floor_since = None; t0_wall = time.time(); i = 0; yolo_ms = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t_v = i / src_fps if is_file else time.time() - t0_wall
        if is_file and args.realtime:
            lag = t_v - (time.time() - t0_wall)
            if lag > 0:
                time.sleep(lag)
        wall = time.time()
        if frame.shape[1] > args.max_width:
            s = args.max_width / frame.shape[1]; frame = cv2.resize(frame, (args.max_width, int(frame.shape[0] * s)), interpolation=cv2.INTER_AREA)
        H, W = frame.shape[:2]
        if state is None:
            zpath = ROOT / mon["zones_file"]
            zones = Zones(zpath, W, H) if (zpath.exists() and not args.no_zones) else None
            state = State(zones, args.yolo_fps, H); state.buffer = buffer
            tracks = OnlineTracks(args.yolo_fps, H, W, cfg["tracking"]["max_gap_s"], cfg["tracking"]["min_link_score"], keep_s=8.0)
            print(f"[{now_str()}] zones: {list(zones.polys) if zones else 'none (floor rule off)'}", flush=True)
        if t_v - last_buf >= 1 / args.buffer_fps - 1e-6:
            buffer.append((t_v, frame.copy())); last_buf = t_v
        if t_v - last_yolo >= 1 / args.yolo_fps - 1e-6:
            last_yolo = t_v; ty = time.perf_counter()
            dets = yolo_detect(model, frame, cfg["yolo"], cfg["tracking"]); yolo_ms.append((time.perf_counter() - ty) * 1000)
            tracks.update(dets)
            main_det = max(dets, key=lambda d: (d["box"][2] - d["box"][0]) * (d["box"][3] - d["box"][1])) if dets else None
            state.push(t_v, wall, main_det)
            active = [f for f in flows if not f.done]
            tr = tracks.as_track()
            # ---- drop trigger
            if not active and t_v >= cooldown_until and tr["objects"]:
                res = score_track(tr, trig_cfg, "trigger")
                if res["flag"]:
                    b = res["best"]; drop_v = t_v - (tr["objects"][res["object_id"]][-1]["t"] - b["end_t"])
                    if drop_v <= last_drop_v + 0.5 and last_cancelled_v is None:
                        res["flag"] = False
                if res["flag"]:
                    if last_cancelled_v is not None and drop_v - last_cancelled_v < 12.0:
                        drop_v = last_cancelled_v
                    last_drop_v = max(last_drop_v, drop_v); last_cancelled_v = None
                    snap = min(buffer, key=lambda tf: abs(tf[0] - drop_v))[1]
                    fl = Flow("drop", state, args, cfg, (W, H), drop={"drop_v": drop_v, "drop_wall": wall - (t_v - drop_v), "obj": res["object_id"], "best": b}, snapshot=snap)
                    flows.append(fl); fl.start(); cooldown_until = t_v + args.cooldown_s
            # ---- floor-zone rule (no drop needed)
            if state.zones is not None and not active and t_v >= cooldown_until:
                r = state.latest()
                down = r and r["present"] and r["zone"]["on_floor"] and r["ar"] <= mon["low_ar_max"]
                if down:
                    floor_since = floor_since if floor_since is not None else t_v
                    mv = state.movement_bl(2.0)
                    if t_v - floor_since >= mon["floor_still_s"] and mv is not None and mv <= mon["still_move_max_bl"]:
                        snap = buffer[-1][1]
                        fl = Flow("floor_still", state, args, cfg, (W, H),
                                  drop={"drop_v": floor_since, "drop_wall": wall - (t_v - floor_since), "zone": r["zone"]["foot_zone"], "ar": r["ar"],
                                        "move": mv, "for_s": t_v - floor_since}, snapshot=snap)
                        flows.append(fl); fl.start(); cooldown_until = t_v + args.cooldown_s; floor_since = None
                else:
                    floor_since = None
        i += 1
        if args.duration and t_v >= args.duration:
            break
    cap.release()
    for f in flows:
        f.join(timeout=240)
    print(f"[{now_str()}] done: {i} frames, YOLO {np.mean(yolo_ms):.0f} ms/frame, {sum(f.result.get('event', False) for f in flows)} event(s), "
          f"session cost ${sum(f.cost for f in flows):.4f}, ledger total ${costs.total():.2f}", flush=True)
    for f in flows:
        print(json.dumps({k: f.result.get(k) for k in ("kind", "event", "time", "outcome", "said")}, ensure_ascii=False), "timings", f.timings)


if __name__ == "__main__":
    main()
