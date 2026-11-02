"""Define floor and bed zones for a fixed camera from one frame, using SAM 3.1 concept prompts.

  remy setup-zones --stream 2                                   # grab a frame from the Tapo stream2 -> zones.json + zones_overlay.jpg
  remy setup-zones --frame tapo/zones_frame.jpg                 # from a saved frame
  remy setup-zones --frame f.jpg --prompts floor,rug,bed,sofa   # extra zones
Each prompt is one SAM 3.1 image request ($2.50 per 1,000 images). Masks become polygons in zones.json
(pixel coords of the frame size used); the overlay shows floor in green, bed in blue, others in orange.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
from contextlib import aclosing
from pathlib import Path

import cv2
import numpy as np
from meta_sam_parser import decode_mask_to_raster, image_segmentation_format, parse_responses_stream
from openai import AsyncOpenAI

from remy.common import META_BASE_URL, ROOT, SAM_MODEL, api_key, tapo_rtsp, write_json
from remy.vision.sam import stream_events

COLORS = {"floor": (80, 200, 80), "bed": (230, 120, 40), "sofa": (40, 160, 255), "couch": (40, 160, 255)}
SYNONYMS = {"floor": ["floor", "rug", "carpet"], "bed": ["bed", "mattress", "cot"], "sofa": ["sofa", "couch"]}


async def segment_image(api: AsyncOpenAI, jpg_bytes: bytes, concept: str):
    data_url = "data:image/jpeg;base64," + base64.b64encode(jpg_bytes).decode("ascii")
    stream = await api.responses.create(
        model=SAM_MODEL, stream=True,
        input=[{"type": "message", "role": "user",
                "content": [{"type": "input_text", "text": concept}, {"type": "input_image", "image_url": data_url}]}],
        metadata={"mask_encoding": "one_bit", "include_confidence": "true"})
    async with stream, aclosing(stream_events(stream)) as events:
        parsed = parse_responses_stream(events, image_segmentation_format())
        async with parsed:
            async for _ in parsed:
                pass
            return await parsed.final_result()


def masks_to_polys(result, W: int, H: int) -> list[dict]:
    out = []
    for rec in result.records:
        if rec.kind != "mask":
            continue
        raster = np.frombuffer(decode_mask_to_raster(rec.mask), dtype=np.uint8).reshape(rec.mask.height, rec.mask.width)
        b = rec.bounds
        full = np.zeros((H, W), np.uint8)
        x1, y1 = int(round(b.left)), int(round(b.top)); x2, y2 = int(round(b.right)), int(round(b.bottom))
        patch = cv2.resize(raster, (max(1, x2 - x1), max(1, y2 - y1)), interpolation=cv2.INTER_NEAREST)
        full[y1:y1 + patch.shape[0], x1:x1 + patch.shape[1]] = patch[: H - y1, : W - x1]
        cnts, _ = cv2.findContours(full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        c = cv2.approxPolyDP(c, 2.0, True)
        out.append({"object_id": rec.object_id, "confidence": rec.confidence, "area_frac": round(float(full.mean()), 4),
                    "polygon": [[int(p[0][0]), int(p[0][1])] for p in c], "mask": full})
    return out


async def main_async(args):
    if args.frame:
        img = cv2.imread(args.frame)
        src = args.frame
    else:
        from remy.camera.rtsp import FFmpegFrames
        url = tapo_rtsp(args.stream)
        rd = FFmpegFrames(url)
        ok, img = rd.read(); rd.close()
        if not ok:
            raise SystemExit("could not grab a frame from the stream")
        src = f"tapo stream{args.stream}"
        (ROOT / "tapo").mkdir(exist_ok=True)
        cv2.imwrite(str(ROOT / "tapo" / "zones_frame.jpg"), img)
    H, W = img.shape[:2]
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    zones = {"source": src, "width": W, "height": H, "zones": {}}
    overlay = img.copy()
    async with AsyncOpenAI(base_url=META_BASE_URL, api_key=api_key(), timeout=120) as api:
        for concept in args.prompts.split(","):
            concept = concept.strip()
            polys = []
            for phrase in SYNONYMS.get(concept, [concept]):          # fall back to synonyms when a prompt finds nothing
                res = await segment_image(api, buf.tobytes(), phrase)
                polys = masks_to_polys(res, W, H)
                if polys:
                    if phrase != concept:
                        print(f"  ('{concept}' found nothing, '{phrase}' did)", flush=True)
                    break
            polys.sort(key=lambda z: -z["area_frac"])
            keep = polys[: args.max_per_prompt]
            zones["zones"][concept] = [{k: v for k, v in z.items() if k != "mask"} for z in keep]
            color = COLORS.get(concept, (0, 165, 255))
            for z in keep:
                tint = overlay.copy(); tint[z["mask"] > 0] = color
                overlay = cv2.addWeighted(overlay, 0.55, tint, 0.45, 0)
                cv2.polylines(overlay, [np.array(z["polygon"], np.int32)], True, color, 2)
                x, y = z["polygon"][0]
                cv2.putText(overlay, f"{concept} {z['confidence'] if z['confidence'] is not None else ''}", (x, max(14, y - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
            print(f"{concept}: {len(polys)} match(es), kept {len(keep)}: " +
                  ", ".join(f"{z['area_frac']*100:.1f}% of frame (conf {z['confidence']})" for z in keep), flush=True)
    out_json = Path(args.out); out_json.parent.mkdir(parents=True, exist_ok=True)
    write_json(out_json, zones)
    out_img = out_json.with_name(out_json.stem + "_overlay.jpg")
    cv2.imwrite(str(out_img), overlay)
    print(f"wrote {out_json} and {out_img}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="remy setup-zones", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frame")
    p.add_argument("--stream", type=int, default=2)
    p.add_argument("--prompts", default="floor,bed")
    p.add_argument("--max-per-prompt", type=int, default=2)
    p.add_argument("--out", default=str(ROOT / "zones.json"))
    asyncio.run(main_async(p.parse_args(argv)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
