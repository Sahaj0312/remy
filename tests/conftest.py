import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def cfg() -> dict:
    """The tracked example config, not the per-home one, so tests do not depend on local tuning."""
    return json.load(open(REPO / "config.example.json"))


def make_frames(spec, fps=10.0, W=640, H=480):
    """spec: list of (seconds, cx_px, cy_px, box_w, box_h) segments or None for 'not present' segments.
    Produces per-frame dicts in the shared track schema, linearly interpolating inside each segment
    from the previous segment's end pose (so a 0.4 s fall segment is a 0.4 s move)."""
    frames = []
    i = 0
    prev = None
    for seg in spec:
        n = int(round(seg[0] * fps))
        if seg[1] is None:
            for _ in range(n):
                frames.append({"i": i, "t": round(i / fps, 3), "present": False}); i += 1
            continue
        _, cx, cy, bw, bh = seg
        start = prev if prev else (cx, cy, bw, bh)
        for k in range(n):
            a = (k + 1) / n
            px = start[0] + (cx - start[0]) * a; py = start[1] + (cy - start[1]) * a
            w = start[2] + (bw - start[2]) * a; h = start[3] + (bh - start[3]) * a
            box = [px - w / 2, py - h / 2, px + w / 2, py + h / 2]
            frames.append({"i": i, "t": round(i / fps, 3), "present": True, "box": box,
                           "cx": px / W, "cy": py / H, "ar": h / w, "mask_area": None, "conf": 0.9})
            i += 1
        prev = (cx, cy, bw, bh)
    return frames


def make_track(spec, fps=10.0, W=640, H=480, name="synthetic"):
    frames = make_frames(spec, fps, W, H)
    return {"clip": name, "fps": fps, "height": H, "width": W, "tracker": "synthetic", "n_frames": len(frames),
            "frames": frames, "objects": {"1": frames}}
