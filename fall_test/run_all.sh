#!/bin/sh
# Full eval chain. Needs META_API_KEY in the environment (or in fall_test/.env).
set -e
cd "$(dirname "$0")"
PY=.venv/bin/python
$PY scripts/build_labels.py
$PY scripts/prepare_clips.py
echo "== SAM 3.1 =="; $PY scripts/sam_track.py --estimate-only; $PY scripts/sam_track.py --parallel 3
echo "== scores =="
$PY scripts/fall_score.py --tracker sam --scorer v2 --quiet
$PY scripts/fall_score.py --tracker sam --scorer v1 --quiet
echo "== Spark triage of flagged clips =="; $PY scripts/spark_verify.py --score-dir sam_v2
echo "== YOLO baseline + trigger =="
$PY scripts/yolo_track.py
$PY scripts/fall_score.py --tracker yolo --scorer trigger --quiet
$PY scripts/fall_score.py --tracker yolo --scorer v2 --quiet
echo "== report =="; $PY scripts/report.py > /dev/null; echo "wrote REPORT.md"
