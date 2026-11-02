"""Muse Spark as the triage backend (no veto): urgency + a family summary, and the check-in reply classifier.

triage_fall: a short 10 fps burst around the drop (video file via the Files API, inline base64 on the live
path, or JPEG frames) plus the tracker numbers -> JSON
  {"urgency": "high"|"medium"|"low", "summary_for_family": "...", "what_happened": "...", "is_lying_still": bool}
classify_reply: the transcript of the 15 s after "are you okay?" -> okay | needs_help | unclear | no_answer.
The batch runner over the eval set is eval/spark_triage.py.
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import subprocess
import tempfile
import time
from pathlib import Path

import cv2
from openai import OpenAI

from remy import costs
from remy.common import META_BASE_URL, SPARK_MODEL, SPARK_USD_PER_INPUT_TOKEN, SPARK_USD_PER_OUTPUT_TOKEN, api_key
from remy.triage.base import TriageBackend

SYSTEM = """You are the triage step of a home fall-alert system. A motion detector has already decided that
the person fell; you do NOT decide whether it was a fall. Your job is to judge how serious it looks and to
write a short note for the family.

You get a short video (or frames) from a fixed indoor camera around the moment of the fall, plus numbers
from a tracker: how far the body centroid dropped (in body lengths), how fast, how long the drop took, and how
long the person has been down without getting up. The floor may have a mat, rug or mattress on it; that does
not make the event less of a fall.

Urgency:
- "high": hard or fast fall, head/trunk impact likely, or the person is not moving afterwards, or they fell
  from a height (bed, chair) onto the floor.
- "medium": clear fall but softer or slower, or the person is moving, shifting or trying to get up.
- "low": very soft, slow or partial fall, and the person is clearly moving or already getting up.

Return only the JSON object requested. summary_for_family must be one or two plain sentences a worried
relative can read in three seconds, e.g. "Went down quickly near the bed at 3:42pm and hasn't moved for 40
seconds." Use the event time you are given, name the place in the room if visible, and say whether they are
moving. what_happened is a one-sentence factual description of the motion."""

SCHEMA = {
    "type": "object",
    "properties": {
        "urgency": {"type": "string", "enum": ["high", "medium", "low"]},
        "summary_for_family": {"type": "string"},
        "what_happened": {"type": "string"},
        "is_lying_still": {"type": "boolean"},
    },
    "required": ["urgency", "summary_for_family", "what_happened", "is_lying_still"],
    "additionalProperties": False,
}


def burst_window(best: dict, n_frames: int, fps: float, cfg: dict) -> tuple[int, int]:
    s = max(0, best["start_i"] - int(round(cfg["burst_before_s"] * fps)))
    e = min(n_frames - 1, best["end_i"] + int(round(cfg["burst_after_s"] * fps)))
    return s, e


def cut_clip(src: Path, s: int, e: int, fps: float, dst: Path):
    """Frame-exact cut [s, e] re-encoded at the clip fps."""
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(src),
                    "-vf", f"select='between(n\\,{s}\\,{e})',setpts=N/{fps:g}/TB", "-r", f"{fps:g}",
                    "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", str(dst)], check=True, capture_output=True)


def grab_frames(path: Path, indices: list[int], quality: int) -> dict[int, str]:
    cap = cv2.VideoCapture(str(path)); want = set(indices); frames = {}; i = 0
    while True:
        ok, img = cap.read()
        if not ok or i > max(indices):
            break
        if i in want:
            ok2, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
            frames[i] = base64.b64encode(buf.tobytes()).decode("ascii")
        i += 1
    cap.release()
    return frames


def parse_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            return json.loads(m.group(0))
        raise


def numbers_text(best: dict, event_time: str) -> str:
    return (f"Tracker numbers: the body centroid dropped {best['drop_rel']:.2f} body lengths, peak speed "
            f"{best['drop_speed_rel_max']:.2f} body lengths per second, time to floor {best['time_to_floor_s']:.1f} s, "
            f"body ended {'wide/horizontal' if best['shape_ok'] else 'not clearly horizontal'} "
            f"(height/width {best['ar_post']:.2f}). Time down without getting up so far: "
            f"{best['no_getup_observed_s']:.1f} s"
            + (" (clip ended, still down)" if best['hold'] == 'unknown' else
               (" (then got up)" if best['hold'] == 'failed' else " (no get-up seen)")) +
            f". Event time: {event_time}.")


def run_triage(client: OpenAI, cfg: dict, best: dict, fps: float, n_frames: int, src: Path,
               event_time: str, mode: str = "video") -> dict:
    """Core call: burst around `best` from `src` + the tracker numbers -> triage JSON. No file I/O."""
    s, e = burst_window(best, n_frames, fps, cfg)
    text = (f"A {(e - s + 1) / fps:.1f} s burst at {fps:g} fps around the fall. " + numbers_text(best, event_time))
    content = [{"type": "input_text", "text": text}]
    file_id = None
    if mode in ("video", "inline"):
        with tempfile.TemporaryDirectory() as tmp:
            burst = Path(tmp) / "burst.mp4"
            cut_clip(src, s, e, fps, burst)
            if mode == "inline":                 # no second upload: base64 data URL in the request body
                data = base64.b64encode(burst.read_bytes()).decode("ascii")
                content.append({"type": "input_video", "video_url": f"data:video/mp4;base64,{data}"})
            else:
                up = client.files.create(file=(burst.name, burst.read_bytes(), "video/mp4"), purpose="user_data")
                file_id = up.id
                content.append({"type": "input_video", "file_id": file_id})
    else:
        step = max(1, int(round(fps / cfg["frames_fps"])))
        idx = list(range(s, e + 1, step))
        frames = grab_frames(src, idx, cfg["jpeg_quality"])
        for i in idx:
            content.append({"type": "input_text", "text": f"t={i / fps:.1f}s"})
            content.append({"type": "input_image", "image_url": f"data:image/jpeg;base64,{frames[i]}"})
    content.append({"type": "input_text", "text": 'Return JSON: {"urgency": "high"|"medium"|"low", '
                    '"summary_for_family": "...", "what_happened": "...", "is_lying_still": true|false}'})
    req = dict(model=cfg["model"], instructions=SYSTEM,
               input=[{"type": "message", "role": "user", "content": content}],
               max_output_tokens=cfg["max_output_tokens"], reasoning={"effort": cfg.get("live_reasoning_effort", cfg["reasoning_effort"]) if mode == "inline" else cfg["reasoning_effort"]},
               text={"format": {"type": "json_schema", "name": "fall_triage", "schema": SCHEMA, "strict": True}})
    t0 = time.perf_counter()
    resp = client.responses.create(**req)
    if resp.status != "completed" or not resp.output_text.strip():
        req["max_output_tokens"] = cfg["max_output_tokens"] * 2          # reasoning ate the cap: retry once
        resp = client.responses.create(**req)
    latency = time.perf_counter() - t0
    text_out = resp.output_text
    usage = resp.usage
    doc = {"model": cfg["model"], "mode": mode, "file_id": file_id, "burst_frames": [s, e],
           "burst_s": round((e - s + 1) / fps, 2), "event_time": event_time,
           "numbers": {k: best[k] for k in ("drop_rel", "drop_speed_rel_max", "time_to_floor_s", "ar_post", "hold", "no_getup_observed_s")},
           "raw_text": text_out, "status": resp.status, "latency_s": round(latency, 2),
           "usage": {"input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens,
                     "reasoning_tokens": getattr(getattr(usage, "output_tokens_details", None), "reasoning_tokens", None)},
           "cost_usd": round(usage.input_tokens * SPARK_USD_PER_INPUT_TOKEN + usage.output_tokens * SPARK_USD_PER_OUTPUT_TOKEN, 5),
           "response_id": resp.id}
    doc["verdict"] = parse_json(text_out)       # raises on bad JSON; caller keeps doc via exception args
    return doc


# ----------------------------------------------------------------------------- check-in reply
CLASSIFY_SYSTEM = """You classify the spoken reply of an elderly person who was just asked whether they are okay after a
possible fall at home. The transcript comes from a room microphone and may contain background noise, TV, or other
people. Reply with JSON only.
- "okay": they clearly say they are fine / okay / don't need help (any language, e.g. "I'm okay", "theek hoon", "haan theek hai").
- "needs_help": they ask for help, say they are hurt, cannot get up, or are in pain (e.g. "help", "bachao", "madad", "I can't get up").
- "unclear": words were heard but do not answer the question (TV, unrelated speech, a question back, mumbling).
- "no_answer": nothing meaningful was said.
Return {"status": "...", "quote": "the words that decided it, or empty", "reason": "one short sentence"}."""

CLASSIFY_SCHEMA = {"type": "object",
                   "properties": {"status": {"type": "string", "enum": ["okay", "needs_help", "unclear", "no_answer"]},
                                  "quote": {"type": "string"}, "reason": {"type": "string"}},
                   "required": ["status", "quote", "reason"], "additionalProperties": False}


def classify_reply(client: OpenAI, model: str, transcript: str, question: str) -> dict:
    """Spark judges the transcript with the question as context. Rules (empty, help words) run before this."""
    text = (transcript or "").strip()
    t0 = time.perf_counter()
    resp = client.responses.create(
        model=model, instructions=CLASSIFY_SYSTEM,
        input=f'Question asked: "{question}"\nTranscript of the next 15 seconds: "{text}"',
        reasoning={"effort": "low"}, max_output_tokens=1500,
        text={"format": {"type": "json_schema", "name": "checkin", "schema": CLASSIFY_SCHEMA, "strict": True}})
    lat = time.perf_counter() - t0
    usd = resp.usage.input_tokens * SPARK_USD_PER_INPUT_TOKEN + resp.usage.output_tokens * SPARK_USD_PER_OUTPUT_TOKEN
    costs.add("spark-classify", usd, text[:40])
    try:
        out = json.loads(resp.output_text)
    except Exception:
        out = {"status": "unclear", "quote": "", "reason": f"bad JSON: {resp.output_text[:80]}"}
    out.update(rule=False, latency_s=round(lat, 2), cost_usd=round(usd, 5))
    return out


# ----------------------------------------------------------------------------- the backend
class SparkTriage(TriageBackend):
    """Muse Spark through the Meta Model API (OpenAI-compatible Responses endpoint).

    cfg is the "spark" section of config.json: model, burst_before_s / burst_after_s, frames_fps, jpeg_quality,
    reasoning_effort, live_reasoning_effort, max_output_tokens.
    """
    name = "spark"

    def __init__(self, cfg: dict, timeout: float = 300, client: OpenAI | None = None):
        self.cfg = cfg
        self.client = client or OpenAI(base_url=META_BASE_URL, api_key=api_key(), timeout=timeout)

    def triage_fall(self, best: dict, fps: float, n_frames: int, src: Path, event_time: str, mode: str = "video") -> dict:
        return run_triage(self.client, self.cfg, best, fps, n_frames, src, event_time, mode)

    def classify_reply(self, transcript: str, question: str) -> dict:
        return classify_reply(self.client, self.cfg.get("model", SPARK_MODEL), transcript, question)
