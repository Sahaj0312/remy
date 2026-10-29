"""Verify flagged clips with Muse Spark: send ~6 frames spanning before / during /
after the detected drop and ask for a strict-JSON verdict.

Reads out/scores/<tracker>/<clip>.json (needs flag=True and best event),
writes out/spark/<tracker>/<clip>.json.
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    CLIPS, META_BASE_URL, OUT, SPARK_USD_PER_INPUT_TOKEN, SPARK_USD_PER_OUTPUT_TOKEN,
    api_key, load_config, load_labels, read_json, write_json,
)
from openai import OpenAI  # noqa: E402

SYSTEM = """You are a careful reviewer for a home fall-detection system. You will see a few
still frames from a single fixed indoor camera, in time order, with their timestamps.
Decide whether the person FELL in this clip.

A FALL is an uncontrolled, unintended collapse to the floor or ground (or against furniture),
usually fast, ending with the person lying or slumped on the floor.

These are NOT falls, even though the person ends up low or horizontal:
- sitting down on a chair, bed, sofa or the floor on purpose
- bending over, crouching, kneeling, picking something up from the floor
- lying down or sleeping on a bed, sofa or even on the floor on purpose, with a controlled motion
- exercising: push-ups, planks, yoga, stretching, sit-ups, bending exercises
- getting up from bed or from the floor

Judge the MOTION between frames (controlled vs. uncontrolled, speed, where the body ends up)
and the context (bed vs. open floor). If the frames are ambiguous, say so with a lower confidence.
Answer only with the JSON object requested."""

SCHEMA = {
    "type": "object",
    "properties": {
        "fall": {"type": "boolean", "description": "true only if the person fell"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reason": {"type": "string", "description": "one or two sentences"},
    },
    "required": ["fall", "confidence", "reason"],
    "additionalProperties": False,
}


def pick_frames(score: dict, n_frames: int, fps: float, offsets) -> list[int]:
    b = score["best"]
    s, e = b["start_i"], b["end_i"]
    idx = []
    for off in offsets:
        if off == "mid":
            i = (s + e) // 2
        else:
            i = e + int(round(off * fps))
        idx.append(min(max(i, 0), n_frames - 1))
    # de-duplicate, then pad by splitting the largest gap until we have len(offsets) frames
    out = sorted(set(idx))
    while len(out) < len(offsets):
        gaps = [(out[k + 1] - out[k], k) for k in range(len(out) - 1)]
        if not gaps or max(gaps)[0] < 2:
            break
        g, k = max(gaps)
        out.insert(k + 1, out[k] + g // 2)
    return out


def grab_frames(path: Path, indices: list[int], quality: int) -> dict[int, str]:
    cap = cv2.VideoCapture(str(path))
    want = set(indices)
    frames: dict[int, str] = {}
    i = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if i in want:
            ok2, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
            frames[i] = base64.b64encode(buf.tobytes()).decode("ascii")
        i += 1
        if i > max(indices):
            break
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


def verify_clip(client: OpenAI, name: str, tracker: str, cfg: dict, args) -> dict:
    score = read_json(OUT / "scores" / tracker / f"{name}.json")
    track = read_json(OUT / tracker / f"{name}.json")
    fps, n = track["fps"], track["n_frames"]
    indices = pick_frames(score, n, fps, cfg["offsets_s"])
    frames = grab_frames(CLIPS / f"{name}.mp4", indices, cfg["jpeg_quality"])
    e_t = score["best"]["end_t"]
    content = [{"type": "input_text",
                "text": f"{len(indices)} frames from a {n/fps:.1f}s clip at {fps:g} fps. "
                        f"Our motion detector flagged a possible fall ending around t={e_t:.1f}s. "
                        "Frames are in time order."}]
    for i in indices:
        rel = i / fps - e_t
        phase = "before" if rel < -0.3 else ("during" if rel <= 0.1 else "after")
        content.append({"type": "input_text", "text": f"Frame at t={i/fps:.1f}s ({phase} the flagged motion):"})
        content.append({"type": "input_image", "image_url": f"data:image/jpeg;base64,{frames[i]}"})
    content.append({"type": "input_text", "text": 'Return JSON: {"fall": true|false, "confidence": 0-1, "reason": "..."}'})

    req = dict(
        model=cfg["model"],
        instructions=SYSTEM,
        input=[{"type": "message", "role": "user", "content": content}],
        max_output_tokens=cfg["max_output_tokens"],
        reasoning={"effort": cfg["reasoning_effort"]},
    )
    structured = not args.no_schema
    if structured:
        req["text"] = {"format": {"type": "json_schema", "name": "fall_verdict",
                                  "schema": SCHEMA, "strict": True}}
    t0 = time.perf_counter()
    try:
        resp = client.responses.create(**req)
    except Exception as e:
        if structured and getattr(e, "status_code", None) == 400:
            # fall back to prompt-only JSON
            req.pop("text")
            structured = False
            resp = client.responses.create(**req)
        else:
            raise
    latency = time.perf_counter() - t0
    text = resp.output_text
    verdict = parse_json(text)
    usage = resp.usage
    cost = usage.input_tokens * SPARK_USD_PER_INPUT_TOKEN + usage.output_tokens * SPARK_USD_PER_OUTPUT_TOKEN
    doc = {
        "clip": name, "tracker": tracker, "model": cfg["model"],
        "frames_sent": indices, "frame_times_s": [round(i / fps, 2) for i in indices],
        "verdict": verdict, "raw_text": text, "structured_output": structured,
        "latency_s": round(latency, 2),
        "usage": {"input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens,
                  "reasoning_tokens": getattr(getattr(usage, "output_tokens_details", None), "reasoning_tokens", None)},
        "cost_usd": round(cost, 5),
        "response_id": resp.id,
    }
    write_json(OUT / "spark" / tracker / f"{name}.json", doc)
    print(f"{name:28s} spark fall={verdict.get('fall')} conf={verdict.get('confidence')} "
          f"{latency:.1f}s in={usage.input_tokens} out={usage.output_tokens} ${cost:.4f} :: {verdict.get('reason','')[:90]}",
          flush=True)
    return doc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("clips", nargs="*", help="default: every flagged clip")
    p.add_argument("--tracker", default="sam", choices=["sam", "yolo"])
    p.add_argument("--all", action="store_true", help="verify every scored clip, not only flagged")
    p.add_argument("--force", action="store_true")
    p.add_argument("--no-schema", action="store_true")
    args = p.parse_args()
    cfg = load_config()["spark"]
    names = args.clips
    if not names:
        names = []
        for r in load_labels():
            sp = OUT / "scores" / args.tracker / f"{r['clip']}.json"
            if not sp.exists():
                continue
            sc = read_json(sp)
            if (sc.get("flag") or args.all) and sc.get("best"):
                names.append(r["clip"])
    if not args.force:
        names = [n for n in names if not (OUT / "spark" / args.tracker / f"{n}.json").exists()]
    print(f"{len(names)} clips to verify with {cfg['model']}", flush=True)
    client = OpenAI(base_url=META_BASE_URL, api_key=api_key(), timeout=300)
    for n in names:
        try:
            verify_clip(client, n, args.tracker, cfg, args)
        except Exception as e:
            print(f"{n}: FAILED {type(e).__name__}: {e}", flush=True)
            write_json(OUT / "spark" / args.tracker / f"{n}.error.json", {"clip": n, "error": repr(e)})


if __name__ == "__main__":
    main()
