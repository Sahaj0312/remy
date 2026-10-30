"""Track the person in each 5 fps clip with SAM 3.1 (Meta Model API).

Writes out/sam/<clip>.json with per-frame box, mask centroid, mask area,
aspect ratio, plus latency and cost. Follows the official cookbook pattern
(meta-model-cookbook/07_sam/01_api_basics/python/sam_api.py).
"""
from __future__ import annotations

import argparse
import asyncio
import sys
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    CLIPS, META_BASE_URL, OUT, SAM_MODEL, api_key, empty_frame, finalize_objects, load_config,
    load_labels, merge_broken_tracks,
    sam_cost_usd, video_info, write_json,
)

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


async def run_clip(api: AsyncOpenAI, name: str, args) -> dict:
    path = CLIPS / f"{name}.mp4"
    info = video_info(path)
    result, timing = await segment_video(api, path, args.mask_encoding, args.confidence)
    frames, meta, objects = result_to_frames(result, info["n_frames"], info["fps"],
                                             info["width"], info["height"])
    doc = {
        "clip": name, "source_video": str(path.relative_to(path.parents[1])),
        "tracker": SAM_MODEL, "prompt": PROMPT,
        **info,
        "billed_frames": max(info["n_frames"], 50),
        "cost_usd": round(sam_cost_usd(info["n_frames"]), 5),
        "timing": timing, "meta": meta,
        "raw_output_chars": len(result.raw_output),
        "frames": frames,
        "objects": objects,
    }
    write_json(OUT / "sam" / f"{name}.json", doc)
    if args.save_raw:
        (OUT / "sam" / f"{name}.raw.txt").write_text(result.raw_output)
    present = sum(f["present"] for f in frames)
    print(f"{name}: {present}/{info['n_frames']} frames tracked, objects={meta['n_objects']}, "
          f"upload {timing['upload_s']:.1f}s, first-frame {timing['first_frame_s']:.1f}s, "
          f"total {timing['total_s']:.1f}s, ${doc['cost_usd']:.4f}", flush=True)
    return doc


async def main_async(args):
    names = args.clips or [r["clip"] for r in load_labels()]
    if args.limit:
        names = names[: args.limit]
    if not args.force:
        names = [n for n in names if not (OUT / "sam" / f"{n}.json").exists()]
    total_cost = sum(sam_cost_usd(video_info(CLIPS / f"{n}.mp4")["n_frames"]) for n in names)
    print(f"{len(names)} clips to run; estimated SAM cost ${total_cost:.3f}", flush=True)
    if args.estimate_only:
        return
    sem = asyncio.Semaphore(args.parallel)
    async with AsyncOpenAI(base_url=META_BASE_URL, api_key=api_key(), timeout=600) as api:
        async def one(n):
            async with sem:
                try:
                    return await run_clip(api, n, args)
                except Exception as e:  # keep going; record the failure
                    print(f"{n}: FAILED {type(e).__name__}: {e}", flush=True)
                    write_json(OUT / "sam" / f"{n}.error.json", {"clip": n, "error": repr(e)})
        await asyncio.gather(*(one(n) for n in names))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("clips", nargs="*")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--parallel", type=int, default=1)
    p.add_argument("--force", action="store_true")
    p.add_argument("--estimate-only", action="store_true")
    p.add_argument("--save-raw", action="store_true")
    p.add_argument("--confidence", action="store_true", help="request include_confidence")
    p.add_argument("--mask-encoding", default="one_bit", choices=["one_bit", "lossless"])
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
