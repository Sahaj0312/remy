# fall_test — feasibility test for camera-based fall detection

Measures whether "track the person, score the drop, verify with a VLM" works on public footage.
Not a product. Everything lives in this folder.

## Layout
- `labels.csv` — 40 clips: 20 falls, 20 ADL (hard negatives: sitting, crouching, lying in bed, push-ups).
- `config.json` — all thresholds (score, Spark prompt frames, YOLO settings).
- `raw/` — downloads (UR Fall cam0 RGB zips + extracted PNGs; GMDCSA24 mp4s). Untrusted data, nothing executes from here.
- `clips/` — 5 fps H.264 clips, max width 640 (what gets uploaded to SAM).
- `scripts/`
  - `prepare_clips.py` — raw → `clips/`
  - `sam_track.py` — SAM 3.1 via Meta Model API (Files API upload + streamed Responses), per-frame box / mask centroid / aspect ratio → `out/sam/`
  - `yolo_track.py` — same output schema from `yolo11n-pose` locally → `out/yolo/`
  - `fall_score.py --tracker sam|yolo` — rule score (drop + tall→wide + stays low) → `out/scores/<tracker>/`
  - `spark_verify.py --tracker sam` — Muse Spark 1.3 JSON verdict on ~6 frames for every flagged clip → `out/spark/<tracker>/`
  - `make_gifs.py` — annotated GIFs → `out/gifs/`
  - `report.py` — `REPORT.md`
- `run_all.sh` — the whole chain.

## Run
```sh
export META_API_KEY=...        # Meta Model API key (MODEL_API_KEY also accepted, or put it in fall_test/.env)
./run_all.sh
```
Pricing used for estimates (dev.meta.ai/docs/pricing-rate-limits, 2026-10-29): SAM 3.1 $0.20 per 1,000 video frames,
50-frame minimum per request; Muse Spark 1.3 $1.25 / $4.25 per 1M input / output tokens.

## Data sources
- UR Fall Detection Dataset (Kwolek & Kępski 2014), CC BY-NC-SA 4.0 — https://fenix.ur.edu.pl/mkepski/ds/uf.html
- GMDCSA24 (Alam et al. 2024), MIT — https://github.com/ekramalam/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos

## Results (2026-10-29)
See `REPORT.md` for the per-clip table, metrics, latency and cost. Annotated GIFs are written to `out/gifs/`
and are not committed (they are frames from the datasets). Total API spend for the full run was about $0.52.
