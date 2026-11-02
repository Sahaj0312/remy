#!/bin/sh
# Full eval chain over eval/labels.csv. Needs META_API_KEY (environment or .env) and `pip install -e .` done.
set -e
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
$PY eval/build_labels.py
$PY eval/prepare_clips.py
echo "== SAM 3.1 =="; $PY eval/sam_tracks.py --estimate-only; $PY eval/sam_tracks.py --parallel 3
echo "== scores =="
$PY eval/score_tracks.py --tracker sam --scorer v2 --quiet
$PY eval/score_tracks.py --tracker sam --scorer v1 --quiet
echo "== Spark triage of flagged clips =="; $PY eval/spark_triage.py --score-dir sam_v2
echo "== YOLO baseline + trigger =="
$PY eval/yolo_tracks.py
$PY eval/score_tracks.py --tracker yolo --scorer trigger --quiet
$PY eval/score_tracks.py --tracker yolo --scorer v2 --quiet
echo "== report =="; $PY eval/report.py > /dev/null; echo "wrote eval/REPORT.md"
