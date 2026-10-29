# fall_test — feasibility test for camera-based fall detection

Measures whether "track the person, score the drop, triage with a VLM" works on public footage,
and holds a skeleton of the live pipeline. Not a product. Everything lives in this folder.

## Layout
- `labels.csv` — 100 clips: 40 falls, 60 non-falls, with `type` (fall:standing, fall:chair, fall:bed, fall:floor-sit,
  fall:slow, fall:onto-bed, adl:lie-bed, adl:lie-floor, adl:sit, adl:bend-pick, adl:exercise-floor, adl:exercise-stand,
  adl:get-up, adl:sit-walk) and `lighting` (day / night / lights-off). `scripts/build_labels.py` regenerates it.
- `config.json` — every threshold: `score_v2` (current), `score_v1` (round 1), `trigger` (YOLO stage), `spark`, `yolo`, `tracking`.
- `raw/` — downloads (UR Fall cam0 RGB zips, GMDCSA24 mp4s). Untrusted data, nothing executes from here. Not committed.
- `clips/` — 10 fps H.264 clips, max width 640 (what SAM sees). `clips_5fps/` holds the round-1 versions. Not committed.
- `scripts/`
  - `prepare_clips.py` — raw → `clips/`
  - `sam_track.py` — SAM 3.1 via Meta Model API (Files API upload + streamed Responses) → `out/sam/`
  - `yolo_track.py` — same schema from `yolo11n-pose` locally, greedy IoU linking → `out/yolo/`
  - `fall_score.py --tracker sam|yolo --scorer v2|v1|trigger` → `out/scores/<tracker>_<scorer>/`
  - `spark_verify.py` — Muse Spark 1.3 triage of flagged clips (urgency, family summary, is_lying_still) → `out/spark/<dir>/`
  - `live.py` — live skeleton (see below)
  - `make_gifs.py`, `report.py` → `out/gifs/`, `REPORT.md`
- `run_all.sh` — the eval chain. `events/` — live events (clip, snapshot, json). Not committed.

## Score v2 (in body lengths)
Body length = median of max(box height, box width) in the second before the drop. A clip is flagged when
the centroid drops ≥ `drop_rel_min` body lengths with peak speed ≥ `drop_speed_rel_min` body lengths/s,
the box ends wide (`ar_wide_max`), and the person does not get up within `no_getup_s`. If the clip ends
before `no_getup_s`, the hold is `unknown` and the clip still counts as flagged; the report says how many.
No "tall before" requirement, so falls from a bed or from sitting on the floor can trigger.

## Live pipeline
```sh
export META_API_KEY=...                          # or put it in fall_test/.env
.venv/bin/python scripts/live.py --source 0      # laptop webcam
.venv/bin/python scripts/live.py --source rtsp://user:pass@ip:554/stream2
.venv/bin/python scripts/live.py --source clips/gmdcsa_S1_Fall_12.mp4 --realtime   # simulate with a file
.venv/bin/python scripts/live.py --source 0 --no-api                                # YOLO trigger only, no cost
```
YOLO pose runs at 5 fps on the local machine and keeps a 10 s buffer at 10 fps. When the trigger rule
sees a drop it waits `no_getup_s` (default 3 s); if the person gets back up the event is dropped. Otherwise the
buffered clip (4 s before the drop to now) goes to SAM 3.1, score v2 runs on the SAM track, and if it is a fall
Muse Spark returns urgency and a family summary. The event is printed and saved under `events/<time>_<label>/`
with `clip.mp4`, `snapshot.jpg`, `sam_track.json`, `score.json`, `event.json` (timings inside).
On macOS the terminal app needs Camera permission (System Settings → Privacy & Security → Camera) or
OpenCV reports "not authorized to capture video".

## Run the eval
```sh
export META_API_KEY=...
./run_all.sh
```
Pricing used (dev.meta.ai/docs/pricing-rate-limits, 2026-10-29): SAM 3.1 $0.20 per 1,000 video frames,
50-frame minimum per request; Muse Spark 1.3 $1.25 / $4.25 per 1M input / output tokens.

## Results
See `REPORT.md`. Annotated GIFs are written to `out/gifs/` and are not committed (they are dataset frames).

## Data sources
- UR Fall Detection Dataset (Kwolek & Kępski 2014), CC BY-NC-SA 4.0 — https://fenix.ur.edu.pl/mkepski/ds/uf.html
- GMDCSA24 (Alam et al. 2024), MIT — https://github.com/ekramalam/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos
