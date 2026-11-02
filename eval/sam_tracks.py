"""Eval batch: track the person in every clip with SAM 3.1 (Meta Model API) -> eval/out/sam/<clip>.json
with per-frame box, mask centroid, mask area, aspect ratio, plus latency and cost.

  python eval/sam_tracks.py --estimate-only
  python eval/sam_tracks.py --parallel 3
  python eval/sam_tracks.py gmdcsa_S1_Fall_12 --force
"""
from __future__ import annotations

import argparse
import asyncio

from openai import AsyncOpenAI

from remy.common import CLIPS, OUT, SAM_MODEL, api_key, load_labels, sam_cost_usd, video_info, write_json
from remy.common import META_BASE_URL
from remy.vision.sam import PROMPT, result_to_frames, segment_video


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
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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
