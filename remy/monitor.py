"""The live monitor: watch a camera (or a clip) and handle a fall end to end.

capture (webcam / RTSP / file) -> YOLO pose 5 fps + greedy tracks -> rolling 10 fps buffer
  -> candidate drop: clip cut at once, SAM 3.1 runs while YOLO watches for a get-up (no fixed wait)
     -> score v2 on the SAM track -> EVENT
  -> floor-zone rule: on floor/rug (not sofa/bed), low, barely moving for N s -> EVENT without a drop
EVENT -> Spark triage (inline base64 burst, async) + still-down watcher (YOLO + zones)
      -> spoken check-in (Kokoro prompt on camera or laptop) -> 15 s mic -> transcribe -> Spark classify
      -> okay: calm WhatsApp; needs_help / twice unclear or no answer: urgent WhatsApp with snapshot
      -> still down 60 s after an okay: check in again.
Every step is timestamped in events/<id>/log.txt and event.json. A dropped stream is reopened with back-off
(remy.camera.capture) and a daily heartbeat tells the family the monitor is alive (config alerts.heartbeat).

  remy run --source tapo
  remy run --source examples/clips/gmdcsa_S1_Fall_12.mp4 --realtime --no-zones --reply examples/replies/okay.wav
  remy run --source 0 --no-api                       # laptop webcam, YOLO trigger only, no API calls
"""
from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from remy import costs
from remy.alerts.heartbeat import heartbeat_due, heartbeat_start_state, heartbeat_text
from remy.camera.audio_in import MicStream
from remy.camera.capture import Capture
from remy.camera.clips import concat_clips, freeze_clip, write_clip
from remy.alerts.whatsapp import delivered
from remy.common import EVENTS, ROOT, api_key, env_get, load_config, redact, sam_cost_usd, video_info, write_json
from remy.vision.score import score_track
from remy.vision.tracking import OnlineTracks
from remy.vision.yolo import detect as yolo_detect, load_model
from remy.vision.zones import Zones
from remy.web import WebView, annotate


def now_str() -> str:
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


# ----------------------------------------------------------------------------- shared state
class State:
    """What the capture loop knows right now; flows read it under the lock."""

    def __init__(self, zones: Zones | None, yolo_fps: float, H: int):
        self.lock = threading.Lock(); self.zones = zones; self.fps = yolo_fps; self.H = H
        self.history: deque = deque(maxlen=int(60 * yolo_fps))     # per YOLO tick
        self.buffer: deque | None = None; self.t = 0.0; self.wall = time.time(); self.t0_wall = time.time()
        self.cooldown_until_wall = 0.0      # set by a flow when it raises an event; the main loop holds the trigger until then
        self.caution_wall = 0.0             # when the family last got a caution, so an outage does not flood them
        self.tracking_down = False          # they were told tracking is down and have not yet heard it is back
        self.last_event_t = None            # stream time of the last raised event, for de-duplicating one fall
        self.audio: MicStream | None = None # rolling room audio over the same span as the frame buffer (camera sources only)
        self.web: WebView | None = None     # the local page, when --web is on

    def reset(self):
        """After a camera reconnect: forget the short-term history, the event bookkeeping stays."""
        with self.lock:
            self.history.clear()
            if self.buffer is not None:
                self.buffer.clear()

    def push(self, t, wall, det):
        rec = {"t": t, "wall": wall, "present": det is not None}
        if det is not None:
            x1, y1, x2, y2 = det["box"]
            rec.update(box=det["box"], ar=(y2 - y1) / max(x2 - x1, 1e-6), cy=det["cy_px"] / self.H, cx=det["cx_px"] / (self.zones.w if self.zones else 1),
                       body=max(y2 - y1, x2 - x1) / self.H, zone=self.zones.classify_box(det["box"], det.get("hip_px")) if self.zones else None)
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

    def on_furniture_since(self, t_v: float, max_s: float = 3.0) -> tuple[int, int]:
        """Of the YOLO records from stream time t_v on (at most max_s of them): how many put the hips on the
        sofa or bed, and how many saw a body at all. The drop path asks this: a fast sit-down ends on furniture.
        The hips only: a body lying on the floor next to the sofa has its box bottom on the sofa polygon."""
        with self.lock:
            rs = [r for r in self.history if r["present"] and t_v <= r["t"] <= t_v + max_s]
        return sum(1 for r in rs if (r.get("zone") or {}).get("body_on_furniture")), len(rs)


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
        self.released = False               # the main loop lifts the trigger cooldown once, when a flow ends without an event
        self.result = {"kind": kind, "event": False}

    # ---- logging
    def log(self, msg):
        line = f"[{now_str()}] [+{time.time() - self.t_start:6.2f}s] {msg}"
        print(line, flush=True); self.lines.append(line)
        if self.state.web is not None:
            self.state.web.line(line)
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
            fr = [(t, f) for (t, f, _w) in self.state.buffer if t >= start_t and (end_t is None or t <= end_t)]
        return fr

    def clip_from_buffer_wall(self, w0, w1):
        """Frames whose wall time lies in [w0, w1], as [(t, frame)]; the span's wall limits for the audio buffer."""
        with self.state.lock:
            fr = [(t, f, w) for (t, f, w) in self.state.buffer if w0 <= w <= w1]
        return [(t, f) for t, f, _ in fr], (fr[0][2], fr[-1][2]) if fr else (None, None)

    def room_audio(self, w0, w1):
        au = self.state.audio
        if au is None or w0 is None:
            return None
        off = float((self.cfg["alerts"].get("video") or {}).get("audio_offset_s", 0.0))
        x = au.slice(w0, w1, off)
        return x if len(x) else None

    def fall_with_sound(self, frames):
        """The fall clip for the family: the same frames as clip.mp4 plus the room's sound. Written in the
        background so SAM is not delayed; the alert waits for it."""
        if not frames:
            return
        with self.state.lock:
            walls = [w for (_t, _f, w) in self.state.buffer if frames[0][0] <= _t <= frames[-1][0]]
        w0, w1 = (walls[0], walls[-1] + 0.1) if walls else (None, None)

        def work():
            try:
                write_clip(frames, self.args.buffer_fps, self.dir / "fall_with_sound.mp4", self.room_audio(w0, w1))
            except Exception as e:
                self.log(f"fall clip with sound failed: {type(e).__name__}: {e}")
        th = threading.Thread(target=work, daemon=True); th.start(); self._fall_thread = th

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
        except (Exception, SystemExit) as e:     # SystemExit too: a missing credential or voice file must not end this in silence
            self.log(f"flow error: {type(e).__name__}: {e}")
            self.result["error"] = repr(e)
            self.last_resort_alert()
            self.caution_alert()
        finally:
            self.save(); self.done = True

    def last_resort_alert(self):
        """An event was raised and something broke before the family heard: tell them anyway, with text only."""
        if not self.result.get("event") or "alert" in self.result:
            return
        try:
            from remy.alerts import WhatsApp
            wa = WhatsApp(dry_run=self.args.alerts_dry_run)
            when = getattr(self, "when", "just now")
            reason = "The system hit an error before it could finish checking on her."
            resp = wa.urgent(f"Possible fall detected at {when}.", self.result.get("said"), when, None, reason)
            ok = self.note_alert("urgent", resp, wa, reason)
            self.result["alert"] = {"kind": "urgent", "reason": reason, "responses": resp, "text": wa.last_text, "last_resort": True, "delivered": ok}
        except (Exception, SystemExit) as e:
            self.log(f"ALERT FAILED, the family was not told: {type(e).__name__}: {e}")

    def note_alert(self, kind: str, resp: list[dict], wa, why: str) -> bool:
        """Log what really happened to a message. Returns True if it was accepted for everyone."""
        ok = delivered(resp)
        if ok:
            self.log(f"ALERT {kind} -> {len(resp)} recipient(s){' [dry run]' if wa.dry else ''}: {why}")
        else:
            errs = "; ".join(str((r.get("error") or {}).get("message", r.get("error")) if isinstance(r.get("error"), dict) else r.get("error"))[:120] for r in resp) or "no recipients"
            self.log(f"ALERT FAILED, the family was not told ({errs}). It was {kind}: {why}")
        return ok

    def caution_alert(self):
        """Something failed before this drop got a verdict, most likely SAM. It cannot be called a fall and it
        cannot be called nothing, so the family hears what is known and sees the clip. The checks that need no
        model still apply, and an outage gets at most one message per alerts.caution_every_s."""
        if self.kind != "drop" or self.result.get("event") or not self.drop:
            return
        try:
            st, d = self.state, self.drop
            last_ev = st.last_event_t
            if last_ev is not None and abs(d["drop_v"] - last_ev) < 20.0:
                self.log("no verdict on this drop, but an event was already raised for it -> no caution"); return
            if self.person_got_up(d["drop_v"]):
                self.log("no verdict on this drop, but the person is already up -> no caution"); return
            if st.zones is not None:
                n_f, n = st.on_furniture_since(d["drop_v"], 8.0)
                if n and n_f * 2 >= n:
                    self.log(f"no verdict on this drop, but the body is on the sofa or bed ({n_f}/{n} frames) -> no caution"); return
            alerts = self.cfg["alerts"]
            if time.time() - st.caution_wall < float(alerts.get("caution_every_s", 600)):
                self.log("no verdict on this drop; the family had a caution recently -> no second one yet"); return
            st.caution_wall = time.time()
            from remy.alerts import WhatsApp
            wa = WhatsApp(dry_run=self.args.alerts_dry_run)
            th = getattr(self, "_fall_thread", None)
            if th is not None:
                th.join(timeout=20)
            video = next((p for p in (self.dir / "fall_with_sound.mp4", self.dir / "clip.mp4") if p.exists()), None)
            resp = wa.notice(caution_text(alerts.get("caution_prefix", "\u26a0\ufe0f Caution:"), alerts["person_name"],
                                          datetime.fromtimestamp(d.get("drop_wall", time.time())).strftime("%-I:%M%p").lower(), st.zones is not None),
                             video=video, snapshot=self.dir / "snapshot.jpg")
            ok = self.note_alert("caution", resp, wa, "no verdict on this drop, the family was asked to check")
            self.result["alert"] = {"kind": "caution", "responses": resp, "text": wa.last_text, "delivered": ok}
            if ok:
                st.tracking_down = True
            else:
                st.caution_wall = 0.0            # it did not go out, so the next drop should try again
        except (Exception, SystemExit) as e:
            self.log(f"ALERT FAILED, the caution was not sent: {type(e).__name__}: {e}")

    def tracking_recovered(self):
        """The first good SAM call after a caution: close the loop with the family, once."""
        if not self.state.tracking_down:
            return
        self.state.tracking_down = False
        self.log("SAM 3.1 answered again after an outage")

        def work():
            try:
                from remy.alerts import WhatsApp
                wa = WhatsApp(dry_run=self.args.alerts_dry_run)
                self.note_alert("notice", wa.notice(f"\u2705 Fall tracking is working again. Remy is back to normal for {self.cfg['alerts']['person_name']}."), wa, "tracking is back")
            except (Exception, SystemExit) as e:
                self.log(f"could not send the recovery notice: {type(e).__name__}: {e}")
        threading.Thread(target=work, daemon=True).start()

    def call_sam(self, clip: Path):
        """SAM 3.1 with a deadline. A live call takes 5 to 14 s; past the deadline it counts as down, because a
        hung call would hold the caution back and, while this flow runs, keep the floor rule from starting."""
        from remy.vision import sam
        deadline = float(self.cfg["monitor"].get("sam_deadline_s", 45))

        async def run_sam():
            async with sam.make_client(timeout=deadline) as api:
                return await sam.segment_video(api, clip, "one_bit", False)
        return asyncio.run(asyncio.wait_for(run_sam(), deadline))

    def confirm_drop(self) -> bool:
        d = self.drop
        self.log(f"candidate drop: {d['best']['drop_rel']:.2f} body lengths at {d['best']['drop_speed_rel_max']:.2f} bl/s (track {d['obj']})")
        # let post_s of post-drop frames into the buffer (the ends-wide check needs ~1 s), then cut and send at once
        while self.state.t < d["drop_v"] + self.args.post_s:
            time.sleep(0.05)
        frames = self.clip_from_buffer(d["drop_v"] - self.args.pre_s)
        clip = self.dir / "clip.mp4"; write_clip(frames, self.args.buffer_fps, clip); self.mark("clip_written")
        self.fall_with_sound(frames)
        cv2.imwrite(str(self.dir / "snapshot.jpg"), self.snapshot)
        info = video_info(clip)
        self.log(f"clip {info['n_frames']} frames ({info['n_frames']/info['fps']:.1f}s) written, SAM 3.1 upload starting while YOLO watches for a get-up")
        if self.args.no_api:
            self.log("[dry run] SAM skipped"); self.result.update(dry_run=True); return False
        from remy.vision import sam
        self.mark("sam_call_start")
        result, st = self.call_sam(clip)
        self.tracking_recovered()
        self.timings["sam_upload_s"] = round(st["upload_s"], 2); self.timings["sam_total_s"] = round(st["total_s"], 2); self.mark("sam_done")
        self.cost += sam_cost_usd(info["n_frames"]); costs.add("sam-live", sam_cost_usd(info["n_frames"]), self.dir.name)
        frames_t, meta, objects = sam.result_to_frames(result, info["n_frames"], info["fps"], info["width"], info["height"], use_masks=False)
        self.mark("sam_parsed")
        track = {"clip": self.dir.name, "fps": info["fps"], "height": info["height"], "width": info["width"], "n_frames": info["n_frames"],
                 "tracker": "sam-3.1", "frames": frames_t, "objects": objects, "meta": meta}
        score = score_track(track, self.cfg["score_v2"], "v2"); self.mark("scored")
        threading.Thread(target=lambda: (write_json(self.dir / "sam_track.json", track), write_json(self.dir / "score.json", score)), daemon=True).start()
        self.result["score"] = score.get("best"); self.result["sam_flag"] = score["flag"]
        last_ev = self.state.last_event_t
        if last_ev is not None and abs(d["drop_v"] - last_ev) < 20.0:
            self.log(f"SAM 3.1 {'confirms' if score['flag'] else 'rejects'} the drop, but an event was already raised for this fall -> duplicate, no event")
            return False
        if self.person_got_up(d["drop_v"]):
            self.log(f"SAM 3.1 {'confirms' if score['flag'] else 'rejects'} the drop, but the person is already up -> no event")
            return False
        if self.state.zones is not None:
            n_f, n = self.state.on_furniture_since(d["drop_v"], 8.0)
            if n and n_f * 2 >= n:
                self.log(f"SAM 3.1 {'confirms' if score['flag'] else 'rejects'} the drop, but the body is on the sofa or bed ({n_f}/{n} frames) -> no event")
                return False
        if not score["flag"]:
            self.log(f"SAM 3.1: not a fall (best={ {k: (score.get('best') or {}).get(k) for k in ('drop_rel','drop_speed_rel_max','ar_post')} }) -> no event")
            return False
        self.log(f"SAM 3.1 confirms: drop {score['best']['drop_rel']:.2f} bl at {score['best']['drop_speed_rel_max']:.2f} bl/s, ends wide (ar {score['best']['ar_post']:.2f})")
        return True

    def raise_event(self):
        self.mark("event"); self.result["event"] = True
        self.state.cooldown_until_wall = time.time() + self.args.cooldown_s
        self.state.last_event_t = self.drop.get("drop_v", self.state.t)
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
            from remy.triage import get_backend
            clip = self.dir / "clip.mp4"
            if not clip.exists():        # floor-still: cut the last 6 s from the buffer
                fr = self.clip_from_buffer(self.state.t - 6.0)
                write_clip(fr, self.args.buffer_fps, clip); self.fall_with_sound(fr)
            info = video_info(clip)
            best = self.result.get("score") or {"start_i": max(0, info["n_frames"] - int(3 * info["fps"])), "end_i": info["n_frames"] - 1,
                                                "drop_rel": 0.0, "drop_speed_rel_max": 0.0, "time_to_floor_s": 0.0, "shape_ok": True,
                                                "ar_post": 0.0, "hold": "unknown", "no_getup_observed_s": float(self.drop.get("for_s", 0))}
            spark_cfg = dict(self.cfg["spark"]); spark_cfg["burst_after_s"] = spark_cfg.get("live_burst_after_s", 1.0)
            backend = get_backend({**self.cfg, "spark": spark_cfg}, timeout=120)
            t0 = time.time()
            doc = backend.triage_fall(best, info["fps"], info["n_frames"], clip, self.when, "inline" if backend.name == "spark" else "frames")
            self.triage = doc["verdict"]; self.timings["spark_total_s"] = round(time.time() - t0, 2); self.mark("triage_done")
            self.cost += doc["cost_usd"]; costs.add("spark-triage", doc["cost_usd"], self.dir.name)
            self.result["triage_backend"] = backend.label
            self.log(f"{backend.label} triage ({self.timings['spark_total_s']}s incl. burst cut): urgency={self.triage['urgency']} "
                     f"lying_still={self.triage['is_lying_still']} :: {self.triage['summary_for_family']}")
        except (Exception, SystemExit) as e:
            self.log(f"triage failed: {type(e).__name__}: {e}")
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
        from remy.alerts import WhatsApp
        from remy.checkin import ask_once
        ck = self.cfg["checkin"]; wa = WhatsApp(dry_run=self.args.alerts_dry_run)
        said, outcome, rounds = None, None, []
        self.answer_clips: list[Path] = []
        vcfg = self.cfg["alerts"].get("video") or {}

        def on_recorded(wav: Path, rec_wall: float, r: dict):
            """Cut the frames of her answer while they are still in the buffer: from answer_lead_s before the
            first speech to the end of the recording, with the recording's audio over the same span."""
            if r.get("speech_at_s") is None or not vcfg.get("enabled", True):
                return
            import wave as _wave
            lead = float(vcfg.get("answer_lead_s", 1.0))
            a0 = max(0.0, r["speech_at_s"] - lead); a1 = float(r["duration_s"])
            with _wave.open(str(wav)) as w:
                rate = w.getframerate(); x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
            audio = x[int(a0 * rate): int(a1 * rate)]
            frames, _ = self.clip_from_buffer_wall(rec_wall + a0, rec_wall + a1)
            dst = self.dir / f"answer_{wav.stem.split('_')[1]}.mp4"
            if frames:
                write_clip(frames, self.args.buffer_fps, dst, audio, rate)
            else:                                   # the moment has scrolled out of the buffer (file sources): hold the last frame
                with self.state.lock:
                    last = self.state.buffer[-1][1] if self.state.buffer else None
                if last is None:
                    return
                freeze_clip(last, audio, dst, rate, self.args.buffer_fps)
            self.answer_clips.append(dst)
            self.log(f"answer clip: {dst.name} ({a1 - a0:.1f}s from {a0:.1f}s into the recording)")

        if self.args.no_checkin:
            self.log("check-in skipped (--no-checkin)")
            outcome = "skipped"
        else:
            retries = ck.get("retries_no_answer", 1)
            for attempt in range(1, retries + 2):
                self.mark(f"checkin{attempt}_start")
                line = ("ask_floor" if self.kind == "floor_still" else "ask") if attempt == 1 else "ask_again"
                try:
                    r = ask_once(ck, self.args.audio_out, self.dir, f"r{attempt}", log=self.log, on_recorded=on_recorded, line=line,
                                 mic=self.state.audio, reply=self.args.reply)
                except (Exception, SystemExit) as e:
                    self.log(f"check-in failed: {type(e).__name__}: {e} -> escalating")
                    self.result["checkin_error"] = f"{type(e).__name__}: {e}"[:300]
                    outcome = "checkin_failed"
                    break
                rounds.append(r)
                self.mark(f"checkin{attempt}_done")
                if r["transcript"].get("transcript"):
                    said = r["transcript"]["transcript"]
                outcome = r["status"]
                if outcome in ("okay", "needs_help"):
                    break
                if outcome == "unclear" and ck.get("escalate_on_unclear", True):
                    self.log("check-in: reply was not a clear okay -> escalating")
                    break
                if self.got_up_at is not None:
                    outcome = "got_up"; break
                if attempt <= retries:
                    self.log(f"check-in {attempt}: {outcome} -> asking once more")
        self.result.update(checkin_rounds=rounds, said=said, outcome=outcome)
        summary = self.summary_text()
        # what the watcher knows now beats what the clip showed
        if self.got_up_at is not None:
            summary += f" She got back up about {int(self.state.t - self.event_t)}s after the fall."
        elif self.state.down_now(self.cfg["monitor"]["low_ar_max"]) and self.state.t - self.event_t >= 5:
            summary += f" Still down {int(self.state.t - self.event_t)}s after the fall."
        snap = self.dir / "snapshot.jpg"
        video = self.alert_video() if vcfg.get("enabled", True) and outcome != "skipped" else None
        if outcome == "okay":
            resp = wa.calm(summary, said, self.when, video=video); self.mark("alert_sent")
            ok = self.note_alert("calm", resp, wa, "she says she's okay")
            self.result["alert"] = {"kind": "calm", "responses": resp, "text": wa.last_text, "delivered": ok}
            if ok:                                   # "I've let the family know" must be true
                self.say_after("ack_okay")
            self.recheck_after_okay(wa, snap)
        elif outcome == "got_up":
            resp = wa.calm(summary + " She got back up on her own.", said, self.when, video=video); self.mark("alert_sent")
            ok = self.note_alert("calm", resp, wa, "got up")
            self.result["alert"] = {"kind": "calm", "responses": resp, "text": wa.last_text, "delivered": ok}
        elif outcome == "skipped":
            self.result["alert"] = None
        else:
            name = self.cfg["alerts"]["person_name"]
            reason = {"needs_help": "She asked for help.", "no_answer": f"{name} did not respond when asked if she was okay.",
                      "unclear": "She answered, but not clearly.",
                      "checkin_failed": f"The check-in could not finish, so we do not know if {name} is okay."}.get(outcome, outcome)
            if outcome == "no_answer" and any(not (r.get("speak") or {}).get("ok", True) or not (r.get("record") or {}).get("ok", True) for r in rounds):
                reason = "The camera's speaker or microphone failed, so she may not have been asked or heard."
            resp = wa.urgent(summary, said, self.when, snap, reason, video=video); self.mark("alert_sent")
            ok = self.note_alert("urgent", resp, wa, reason)
            self.result["alert"] = {"kind": "urgent", "reason": reason, "responses": resp, "text": wa.last_text, "delivered": ok}
            if ok:                                   # "I've sent a message to the family" must be true
                self.say_after("ack_sent")
        self.save()
        # keep the watcher alive a while for the still-down line, then finish
        t0 = time.time()
        while time.time() - t0 < self.cfg["monitor"]["still_down_s"] + 5 and self.got_up_at is None and self.still_down_at is None:
            time.sleep(0.5)

    def say_after(self, key: str):
        """Tell her what just happened (ack_okay / ack_sent), after the alert is out, without holding anything up."""
        if self.args.no_checkin or not (self.cfg["checkin"].get("lines") or {}).get(key):
            return
        from remy.checkin import say_line

        def work():
            try:
                say_line(self.cfg["checkin"], key, self.args.audio_out, log=self.log)
            except (Exception, SystemExit) as e:
                self.log(f"could not say {key}: {type(e).__name__}: {e}")
        threading.Thread(target=work, daemon=True).start()

    def alert_video(self) -> Path | None:
        """fall_with_sound.mp4 (or the silent clip) followed by her answer(s), as one MP4 for WhatsApp."""
        th = getattr(self, "_fall_thread", None)
        if th is not None:
            th.join(timeout=20)
        fall = self.dir / "fall_with_sound.mp4"
        if not fall.exists():
            fall = self.dir / "clip.mp4"
        parts = ([fall] if fall.exists() else []) + list(self.answer_clips)
        if not parts:
            return None
        try:
            t0 = time.time()
            out = concat_clips(parts, self.dir / "alert.mp4")
            self.log(f"alert video: {' + '.join(p.name for p in parts)} -> {out.name} ({out.stat().st_size / 1e3:.0f} KB, {time.time() - t0:.1f}s)")
            return out
        except Exception as e:
            self.log(f"alert video failed: {type(e).__name__}: {e}")
            return None

    def recheck_after_okay(self, wa, snap):
        from remy.checkin import ask_once
        ck = self.cfg["checkin"]; limit = ck["recheck_after_s"]
        while self.got_up_at is None:
            # stream time drives this; the wall clock is the guard for when the stream has stopped (end of a file)
            if self.state.t - self.event_t >= limit or time.time() - self.event_wall >= limit + 10:
                if not self.state.down_now(self.cfg["monitor"]["low_ar_max"]):
                    self.log(f"{limit}s after the event: not down any more -> no re-check"); return
                self.log(f"still down {limit}s after saying okay -> checking in again")
                try:
                    r = ask_once(ck, self.args.audio_out, self.dir, "recheck", log=self.log, line="recheck", mic=self.state.audio,
                                 reply=self.args.reply)
                except (Exception, SystemExit) as e:
                    self.log(f"re-check failed: {type(e).__name__}: {e} -> escalating")
                    r = {"status": "checkin_failed", "transcript": {}}
                self.result.setdefault("checkin_rounds", []).append(r)
                if r["status"] == "okay":
                    self.log("re-check: okay (no new alert)")
                else:
                    reason = "Still on the floor a minute after saying she was okay; " + (
                        "asked for help." if r["status"] == "needs_help" else "no clear answer this time.")
                    resp = wa.urgent(self.summary_text(), r["transcript"].get("transcript") or self.result.get("said"), self.when, snap, reason)
                    ok = self.note_alert("urgent (re-check)", resp, wa, reason)
                    self.result["alert_recheck"] = {"kind": "urgent", "reason": reason, "responses": resp, "text": wa.last_text, "delivered": ok}
                return
            time.sleep(0.5)


# ----------------------------------------------------------------------------- messages about the system itself
def caution_text(prefix: str, name: str, when: str, has_floor_rule: bool) -> str:
    still = " Until it is back, Remy may miss a fall" + (", but it will still ask her if she stays on the floor." if has_floor_rule else ".")
    return f"{prefix} Remy saw a possible fall at {when} but could not confirm it, because fall tracking may be temporarily down. Please check on {name}.{still}"


def camera_notice_text(kind: str, name: str, seconds: float, prefix: str = "\u26a0\ufe0f Caution:") -> str:
    mins = max(1, round(seconds / 60))
    return {"down": f"{prefix} Remy lost the camera {mins} min ago and is trying to reconnect. It cannot see {name} until then.",
            "back": f"\u2705 The camera is back after {mins} min. Remy is watching {name} again.",
            "gave_up": f"{prefix} Remy has stopped watching {name}. The camera has been down for {mins} min, so it needs a restart."}[kind]


# ----------------------------------------------------------------------------- the page's view of the flows
def web_status(flows: list, wall: float):
    """(state, since_wall, last alert text, events) for the local page, read off the flows."""
    n_ev = sum(1 for f in flows if f.result.get("event"))
    alert_text = None
    for f in reversed(flows):
        al = f.result.get("alert")
        if al and al.get("text"):
            alert_text = al["text"]; break
    for f in reversed(flows):
        if not f.result.get("event"):
            continue
        since = f.drop.get("drop_wall", f.t_start) if f.drop else f.t_start
        if "alert_sent" in f.timings:
            if wall - (f.t_start + f.timings["alert_sent"]) < 90:
                return "alert", since, alert_text, n_ev
            break
        if not f.done:
            return ("checkin" if any(k.startswith("checkin") for k in f.timings) else "event"), since, alert_text, n_ev
    for f in flows:
        if not f.done and not f.result.get("event"):
            since = f.drop.get("drop_wall", f.t_start) if f.drop else f.t_start
            return "candidate", since, alert_text, n_ev
    return "watching", None, alert_text, n_ev


# ----------------------------------------------------------------------------- main loop
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="remy run", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True, help="tapo | <rtsp url> | <webcam index> | <video file>")
    p.add_argument("--realtime", action="store_true")
    p.add_argument("--yolo-fps", type=float, default=5.0)
    p.add_argument("--buffer-fps", type=float, default=10.0)
    p.add_argument("--buffer-s", type=float, default=12.0)
    p.add_argument("--pre-s", type=float, default=4.0)
    p.add_argument("--post-s", type=float, default=1.0, help="seconds after the drop end to include before cutting the SAM clip (shape check needs ~1 s)")
    p.add_argument("--cooldown-s", type=float, default=30.0, help="after a raised event")
    p.add_argument("--reject-cooldown-s", type=float, default=2.0, help="after a candidate SAM rejected")
    p.add_argument("--max-width", type=int, default=640)
    p.add_argument("--duration", type=float, default=0)
    p.add_argument("--no-api", action="store_true")
    p.add_argument("--no-checkin", action="store_true")
    p.add_argument("--audio-out", default=None, help="camera | laptop (default: config audio.output)")
    p.add_argument("--alerts-dry-run", action="store_true")
    p.add_argument("--reply", metavar="WAV", help="video-file sources only: use this recording as her answer, not a microphone. "
                                                  "Implies --alerts-dry-run and, unless set, --audio-out laptop")
    p.add_argument("--label", default="live")
    p.add_argument("--save-candidates", action="store_true")
    p.add_argument("--no-zones", action="store_true", help="ignore zones.json (e.g. when simulating with a clip from another room)")
    p.add_argument("--no-reconnect", action="store_true", help="exit when the stream drops instead of reconnecting")
    p.add_argument("--max-outage-s", type=float, default=600.0, help="give up reconnecting after this long without a frame")
    p.add_argument("--web", type=int, default=8790, metavar="PORT", help="serve the local page (camera view, state, log) on 127.0.0.1:PORT")
    p.add_argument("--no-web", action="store_true", help="do not serve the local page")
    args = p.parse_args(argv)
    if args.reply:
        # a canned "I'm okay" must never stand in for a real person, or reach a real family
        if not Path(args.source).is_file():
            p.error("--reply works only with a video file as --source")
        if not Path(args.reply).is_file():
            p.error(f"--reply: no such file: {args.reply}")
        args.reply = Path(args.reply); args.alerts_dry_run = True; args.audio_out = args.audio_out or "laptop"
    if not args.no_api:                          # find a missing credential now, not after a fall
        api_key()
        if not args.alerts_dry_run:
            from remy.alerts import WhatsApp
            try:
                problem = WhatsApp().token_problem()
            except SystemExit as e:
                problem = str(e)
            if problem:
                p.error(f"{problem}. To run without sending, add --alerts-dry-run.")
    cfg = load_config(); trig_cfg = {**cfg["score_v2"], **cfg["trigger"]}; mon = cfg["monitor"]

    web = None if args.no_web else WebView()

    def log(msg):
        line = f"[{now_str()}] {msg}"
        print(line, flush=True)
        if web is not None:
            web.line(line)

    if web is not None:
        try:
            log(f"web view: {web.serve(args.web)}")
        except OSError as e:
            log(f"web view not started ({e}); use --web <port> or --no-web"); web = None

    if Path(args.source).is_file() and not args.reply and not args.no_checkin and not env_get("TAPO_IP", required=False):
        log("no camera set up, so the check-in cannot listen and will escalate; to answer from a file add --reply examples/replies/okay.wav")
    def camera_notice(kind: str, seconds: float):
        """Tell the family about a long camera outage. "gave_up" is sent before the process exits; the others do
        not hold up the reconnect."""
        def send():
            try:
                from remy.alerts import WhatsApp
                wa = WhatsApp(dry_run=args.alerts_dry_run)
                resp = wa.notice(camera_notice_text(kind, cfg["alerts"]["person_name"], seconds, cfg["alerts"].get("caution_prefix", "\u26a0\ufe0f Caution:")))
                log(f"camera notice ({kind}) -> {len(resp)} recipient(s){' [dry run]' if wa.dry else ''}" if delivered(resp) else f"ALERT FAILED, the camera notice ({kind}) was not sent")
            except (Exception, SystemExit) as e:
                log(f"ALERT FAILED, the camera notice ({kind}) was not sent: {type(e).__name__}: {e}")
        if kind == "gave_up":
            send()
        else:
            threading.Thread(target=send, daemon=True).start()

    cap = Capture(args.source, max_width=args.max_width, reconnect=not args.no_reconnect, max_outage_s=args.max_outage_s, log=log,
                  on_outage=camera_notice, outage_notice_s=float(mon.get("camera_notice_s", 120)))
    is_file, src_fps = cap.is_file, cap.src_fps
    model = load_model(cfg["yolo"]); EVENTS.mkdir(exist_ok=True)
    hb_cfg = cfg["alerts"].get("heartbeat") or {}
    hb_on = bool(hb_cfg.get("enabled")) and not is_file
    log(f"source {redact(str(args.source))} ({src_fps:.1f} fps), yolo {args.yolo_fps:g} fps, buffer {args.buffer_s:g}s, "
        f"{'DRY RUN' if args.no_api else 'SAM+Spark live'}, audio out {args.audio_out or cfg['audio']['output']}, "
        f"alerts {'DRY' if args.alerts_dry_run else cfg['alerts']['channel']}, floor rule {mon['floor_still_s']}s, still-down {mon['still_down_s']}s, "
        f"heartbeat {'at %02d:00' % hb_cfg.get('hour', 9) if hb_on else 'off'}")

    state = None; tracks = None; buffer = deque(maxlen=int(args.buffer_s * args.buffer_fps) + 5)
    last_buf = last_yolo = -1e9; last_drop_v = -1e9; last_cancelled_v = None; cooldown_until = -1e9
    flows: list[Flow] = []; floor_since = None; t0_wall = time.time(); i = 0; yolo_ms = []
    last_dets: list = []; last_main = None
    started = datetime.now(); hb_last_date = heartbeat_start_state(started, int(hb_cfg.get('hour', 9))); hb_next_check = 0.0; wa_hb = None; audio_retry_at = 0.0; furniture_log_at = 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if cap.take_reconnected():
            # the stream was down: the buffer and tracks span the gap, so start them again
            if state is not None:
                state.reset(); tracks = OnlineTracks(args.yolo_fps, state.H, tracks.W, cfg["tracking"]["max_gap_s"], cfg["tracking"]["min_link_score"], keep_s=8.0)
            floor_since = None; last_buf = last_yolo = -1e9
            log("tracks and buffer reset after the reconnect")
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
            state = State(zones, args.yolo_fps, H); state.buffer = buffer; state.web = web
            if cap.kind == "rtsp" and (cfg["alerts"].get("video") or {}).get("enabled", True):
                try:
                    state.audio = MicStream(stream=1, keep_s=args.buffer_s + 10)
                    log("room audio buffer on (camera mic, in memory only)")
                except Exception as e:
                    log(f"room audio buffer failed to start ({type(e).__name__}); clips will be silent")
            tracks = OnlineTracks(args.yolo_fps, H, W, cfg["tracking"]["max_gap_s"], cfg["tracking"]["min_link_score"], keep_s=8.0)
            print(f"[{now_str()}] zones: {list(zones.polys) if zones else 'none (floor rule off)'}", flush=True)
        if t_v - last_buf >= 1 / args.buffer_fps - 1e-6:
            buffer.append((t_v, frame.copy(), wall)); last_buf = t_v
            if web is not None:
                st, since, alert_text, n_ev = web_status(flows, wall)
                web.status(st, since, alert_text, n_ev)
                web.frame(annotate(frame, last_dets, last_main, state.zones, st, None if since is None else wall - since))
        if t_v - last_yolo >= 1 / args.yolo_fps - 1e-6:
            last_yolo = t_v; ty = time.perf_counter()
            dets = yolo_detect(model, frame, cfg["yolo"], cfg["tracking"]); yolo_ms.append((time.perf_counter() - ty) * 1000)
            last_dets = dets
            tracks.update(dets)
            main_det = max(dets, key=lambda d: (d["box"][2] - d["box"][0]) * (d["box"][3] - d["box"][1])) if dets else None
            state.push(t_v, wall, main_det); last_main = main_det
            active = [f for f in flows if not f.done]
            # a flow that finished without an event only blocks the trigger for a couple of seconds
            for f in flows:
                if f.done and not f.result.get("event") and not f.released:
                    f.released = True; cooldown_until = min(cooldown_until, t_v + args.reject_cooldown_s)
            tr = tracks.as_track()
            # ---- drop trigger (allowed while another flow is still waiting on SAM, so a real fall right after
            #      a false candidate is not lost; never while an event is being handled)
            handling = [f for f in active if f.result.get("event")]
            if not handling and len(active) < 2 and t_v >= cooldown_until and tr["objects"]:
                res = score_track(tr, trig_cfg, "trigger")
                if res["flag"]:
                    b = res["best"]; drop_v = t_v - (tr["objects"][res["object_id"]][-1]["t"] - b["end_t"])
                    if drop_v <= last_drop_v + 0.5 and last_cancelled_v is None:
                        res["flag"] = False
                if res["flag"] and state.zones is not None:
                    n_f, n = state.on_furniture_since(drop_v)
                    if n < 3 and t_v - drop_v < 1.0:
                        res["flag"] = False              # too early to tell where the body ended: look again next tick
                    elif n and n_f * 2 >= n:             # a sit-down or a flop onto the sofa or bed, not a fall
                        res["flag"] = False; last_drop_v = max(last_drop_v, drop_v)
                        if wall >= furniture_log_at:
                            furniture_log_at = wall + 10
                            log(f"candidate drop ends on the sofa or bed ({n_f}/{n} frames) -> no SAM call")
                if res["flag"] and active and any(abs(fa.drop["drop_v"] - drop_v) < 6.0 for fa in active if fa.drop):
                    res["flag"] = False      # same drop the running flow is already checking
                if res["flag"]:
                    if last_cancelled_v is not None and drop_v - last_cancelled_v < 12.0:
                        drop_v = last_cancelled_v
                    last_drop_v = max(last_drop_v, drop_v); last_cancelled_v = None
                    snap = min(buffer, key=lambda tf: abs(tf[0] - drop_v))[1]
                    fl = Flow("drop", state, args, cfg, (W, H), drop={"drop_v": drop_v, "drop_wall": wall - (t_v - drop_v), "obj": res["object_id"], "best": b}, snapshot=snap)
                    flows.append(fl); fl.start(); cooldown_until = t_v + args.reject_cooldown_s
            if state.cooldown_until_wall > wall:          # an event was raised: hold the trigger
                cooldown_until = max(cooldown_until, t_v + (state.cooldown_until_wall - wall))
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
        # ---- the room audio stream: reopen it if the camera dropped it (the check-in records from it too)
        if state is not None and state.audio is not None and not state.audio.alive and wall >= audio_retry_at:
            audio_retry_at = wall + 15
            log(f"room audio stream down ({state.audio.err or 'ended'}); reopening")
            state.audio.close()
            try:
                state.audio = MicStream(stream=1, keep_s=args.buffer_s + 10)
            except Exception as e:
                log(f"room audio stream failed to reopen ({type(e).__name__}: {e})")
        # ---- daily heartbeat (checked twice a minute, sent once a day)
        if hb_on and wall >= hb_next_check:
            hb_next_check = wall + 30
            now = datetime.now()
            if heartbeat_due(now, int(hb_cfg.get("hour", 9)), hb_last_date):
                hb_last_date = now.date()
                try:
                    if wa_hb is None:
                        from remy.alerts import WhatsApp
                        wa_hb = WhatsApp(dry_run=args.alerts_dry_run)
                    n_ev = sum(1 for f in flows if f.result.get("event") and wall - getattr(f, "event_wall", 0) < 86400)
                    text = heartbeat_text(cfg["alerts"]["person_name"], started, now, n_ev, cap.reconnects)
                    resp = [wa_hb.send_text(r["number"], text) for r in wa_hb.recipients()]
                    log(f"heartbeat -> {len(resp)} recipient(s){' [dry run]' if wa_hb.dry else ''}: {text}" if delivered(resp)
                        else f"heartbeat FAILED, nothing reached the family: {text}")
                except Exception as e:
                    log(f"heartbeat failed: {type(e).__name__}: {e}")
        if args.duration and t_v >= args.duration:
            break
    cap.release()
    if state is not None and state.audio is not None:
        state.audio.close()
    if web is not None:
        web.close()
    for f in flows:
        f.join(timeout=240)
    print(f"[{now_str()}] done: {i} frames, YOLO {np.mean(yolo_ms) if yolo_ms else 0:.0f} ms/frame, {sum(f.result.get('event', False) for f in flows)} event(s), "
          f"{cap.reconnects} camera reconnect(s), session cost ${sum(f.cost for f in flows):.4f}, ledger total ${costs.total():.2f}", flush=True)
    for f in flows:
        print(json.dumps({k: f.result.get(k) for k in ("kind", "event", "time", "outcome", "said")}, ensure_ascii=False), "timings", f.timings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
