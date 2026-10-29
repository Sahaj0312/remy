#!/bin/sh
# Full pipeline. Needs META_API_KEY in the environment (or in fall_test/.env).
set -e
cd "$(dirname "$0")"
PY=.venv/bin/python
[ -d clips ] && [ "$(ls clips | wc -l)" -ge 40 ] || $PY scripts/prepare_clips.py
echo "== SAM 3.1: estimate, then 4 pilot clips, then the rest =="
$PY scripts/sam_track.py --estimate-only
$PY scripts/sam_track.py urfall_fall-01 urfall_adl-10 gmdcsa_S1_Fall_12 gmdcsa_S2_ADL_12
$PY scripts/sam_track.py --parallel 2
echo "== rule score on SAM tracks =="
$PY scripts/fall_score.py --tracker sam
echo "== Muse Spark verification of flagged clips =="
$PY scripts/spark_verify.py --tracker sam
echo "== YOLO baseline (local, free) =="
$PY scripts/yolo_track.py
$PY scripts/fall_score.py --tracker yolo --quiet
echo "== report =="
$PY scripts/report.py > /dev/null
echo "wrote REPORT.md"
