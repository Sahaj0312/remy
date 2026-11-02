"""SAM 3.1 video tracking through the Meta Model API (Files API upload + streamed Responses).

segment_video() uploads a clip and streams the segmentation; result_to_frames() turns the records into
the shared track schema (remy.vision.tracking). Follows the official cookbook pattern
(meta-model-cookbook/07_sam/01_api_basics/python/sam_api.py). The batch runner over the eval set is
eval/sam_tracks.py; the live monitor calls segment_video() on the buffered clip.
"""
from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import AsyncIterable, AsyncIterator
from contextlib import aclosing
from pathlib import Path
from typing import Any

import numpy as np
from meta_sam_parser import (
    ResponsesEventLike,
    VideoSegmentationResult,
    decode_mask_to_raster,
    parse_responses_stream,
    video_segmentation_format,
)
from openai import AsyncOpenAI

from remy.common import META_BASE_URL, SAM_MODEL, api_key, load_config
from remy.vision.tracking import empty_frame, finalize_objects, merge_broken_tracks

PROMPT = "person"


async def stream_events(stream: AsyncIterable[Any]) -> AsyncIterator[ResponsesEventLike]:
    """SDK events -> plain dicts; synthesize response.output_text.done from
    response.content_part.done (the API doesn't emit the former)."""
    completed: set[tuple] = set()
    async for event in stream:
        data = event.model_dump() if hasattr(event, "model_dump") else dict(event)
        lane = (data.get("item_id"), data.get("output_index"), data.get("content_index"))
        if data.get("type") == "response.output_text.done":
            if lane not in completed:
                completed.add(lane)
                yield data
            continue
        yield data
        part = data.get("part")
        if (
            data.get("type") == "response.content_part.done"
            and lane not in completed
            and isinstance(part, dict)
            and part.get("type") == "output_text"
            and isinstance(part.get("text"), str)
        ):
            completed.add(lane)
            yield {
                "type": "response.output_text.done",
                "item_id": data.get("item_id"),
                "output_index": data.get("output_index"),
                "content_index": data.get("content_index"),
                "text": part["text"],
            }


class Timed:
    """Wrap the event stream to record time-to-first-frame."""

    def __init__(self, events: AsyncIterable[ResponsesEventLike], t0: float):
        self.events = events
        self.t0 = t0
        self.first_delta: float | None = None
        self.n_deltas = 0

    async def __aiter__(self):
        async for ev in self.events:
            if ev.get("type") == "response.output_text.delta":
                self.n_deltas += 1
                if self.first_delta is None:
                    self.first_delta = time.perf_counter() - self.t0
            yield ev


async def segment_video(api: AsyncOpenAI, path: Path, mask_encoding: str,
                        include_confidence: bool) -> tuple[VideoSegmentationResult, dict]:
    timing: dict[str, float] = {}
    t0 = time.perf_counter()
    uploaded = await api.files.create(
        file=(path.name, path.read_bytes(), "video/mp4"), purpose="user_data",
    )
    timing["upload_s"] = time.perf_counter() - t0
    metadata = {"mask_encoding": mask_encoding}
    if include_confidence:
        metadata["include_confidence"] = "true"
    t1 = time.perf_counter()
    stream = await api.responses.create(
        model=SAM_MODEL,
        stream=True,
        input=[{
            "type": "message", "role": "user",
            "content": [
                {"type": "input_text", "text": PROMPT},
                {"type": "input_video", "file_id": uploaded.id},
            ],
        }],
        metadata=metadata,
    )
    async with stream, aclosing(stream_events(stream)) as events:
        timed = Timed(events, t1)
        parsed = parse_responses_stream(timed, video_segmentation_format())
        async with parsed:
            async for _snapshot in parsed:
                pass
            result = await parsed.final_result()
    timing["first_frame_s"] = timed.first_delta if timed.first_delta is not None else -1.0
    timing["stream_s"] = time.perf_counter() - t1
    timing["total_s"] = time.perf_counter() - t0
    timing["n_delta_events"] = timed.n_deltas
    timing["file_id"] = uploaded.id
    return result, timing


def mask_stats(record) -> tuple[float, float, int]:
    """Centroid (frame coords) and area of a box-local mask raster."""
    raster = np.frombuffer(decode_mask_to_raster(record.mask), dtype=np.uint8)
    h, w = record.mask.height, record.mask.width
    raster = raster.reshape(h, w)
    ys, xs = np.nonzero(raster)
    if len(xs) == 0:
        return float("nan"), float("nan"), 0
    b = record.bounds
    # bounds are half-open source-media coords; raster spans them
    sx = (b.right - b.left) / w
    sy = (b.bottom - b.top) / h
    cx = b.left + (xs.mean() + 0.5) * sx
    cy = b.top + (ys.mean() + 0.5) * sy
    area = int(len(xs) * sx * sy)
    return float(cx), float(cy), area


def result_to_frames(result: VideoSegmentationResult, n_frames: int, fps: float,
                     W: int, H: int, use_masks: bool = True) -> tuple[list[dict], dict, dict]:
    """use_masks=False skips the pure-Python mask decode (~70 ms per mask) and uses box centers: the live path."""
    by_obj: dict[str, dict[int, dict]] = defaultdict(dict)
    for rec in result.records:
        if rec.frame is None:
            continue
        fi = rec.frame.frame_index
        slot = by_obj[rec.object_id].setdefault(fi, {})
        if rec.kind == "box":
            slot["box"] = [rec.left, rec.top, rec.right, rec.bottom]
            slot["conf"] = rec.confidence
        elif rec.kind == "mask":
            slot["mask_rec"] = rec
    meta = {
        "n_objects": len(by_obj),
        "outcome": result.outcome.status,
        "diagnostics": [
            {"severity": d.severity, "code": d.code, "line": d.line, "message": d.message}
            for d in result.diagnostics
        ],
    }
    if not by_obj:
        return [empty_frame(i, fps) for i in range(n_frames)], meta, {}
    clean: dict[str, dict[int, dict]] = {}
    for oid, per in by_obj.items():
        clean[oid] = {}
        for fi, slot in per.items():
            if "box" not in slot or fi >= n_frames:
                continue
            x1, y1, x2, y2 = slot["box"]
            cx, cy, area = (x1 + x2) / 2, (y1 + y2) / 2, None
            if use_masks and "mask_rec" in slot:
                mcx, mcy, marea = mask_stats(slot["mask_rec"])
                if not np.isnan(mcx):
                    cx, cy, area = mcx, mcy, marea
            clean[oid][fi] = {"box": slot["box"], "cx_px": cx, "cy_px": cy,
                              "mask_area_px": area, "conf": slot.get("conf")}
    tcfg = load_config()["tracking"]
    clean, id_map = merge_broken_tracks(clean, int(round(tcfg["max_gap_s"] * fps)), tcfg["min_link_score"])
    objects, primary = finalize_objects(clean, n_frames, fps, W, H)
    meta.update({
        "sam_id_map": id_map,
        "primary_object": primary,
        "frames_per_object": {k: sum(f["present"] for f in v) for k, v in objects.items()},
        "max_frame_index": max(max(p) for p in by_obj.values()),
    })
    return objects[primary], meta, objects


def make_client(timeout: float = 600) -> AsyncOpenAI:
    return AsyncOpenAI(base_url=META_BASE_URL, api_key=api_key(), timeout=timeout)
