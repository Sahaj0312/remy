"""How sensitive is score v2 to its two drop thresholds? Re-scores the committed SAM tracks for a grid of
drop_speed_rel_min x drop_rel_min (and ar_wide_max alone) and writes eval/SENSITIVITY.md. Free: no API calls.

  python eval/sensitivity.py
"""
from __future__ import annotations

import json
from pathlib import Path

from remy.common import EVAL, OUT, load_labels, read_json
from remy.vision.score import score_track

SPEEDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
DROPS = [0.10, 0.125, 0.15, 0.175, 0.20, 0.25]
ARS = [0.6, 0.7, 0.8, 0.9, 1.0, 1.2]


def run(tracks: dict[str, dict], labels: dict[str, str], cfg: dict) -> tuple[int, int]:
    caught = fa = 0
    for name, tr in tracks.items():
        flag = score_track(tr, cfg, "v2")["flag"]
        if labels[name] == "fall":
            caught += flag
        else:
            fa += flag
    return caught, fa


def main():
    base = json.load(open(EVAL.parent / "config.example.json"))["score_v2"]
    labels = {r["clip"]: r["label"] for r in load_labels()}
    tracks = {n: read_json(OUT / "sam" / f"{n}.json") for n in labels if (OUT / "sam" / f"{n}.json").exists()}
    n_f = sum(v == "fall" for v in labels.values()); n_a = len(labels) - n_f
    L = ["# Score v2 threshold sensitivity\n",
         f"Re-scored the committed SAM 3.1 tracks ({len(tracks)} clips: {n_f} falls, {n_a} non-falls). Each cell is "
         "falls caught / false alarms. The chosen operating point is in bold. The thresholds were tuned on this same set, "
         "so these numbers are in-sample; the table shows how flat the neighbourhood is, not how a new room will do.\n",
         f"## drop_speed_rel_min (rows) x drop_rel_min (columns), ar_wide_max = {base['ar_wide_max']}\n",
         "| speed \\ drop | " + " | ".join(f"{d:g}" for d in DROPS) + " |", "|---|" + "---|" * len(DROPS)]
    for s in SPEEDS:
        cells = []
        for d in DROPS:
            c, f = run(tracks, labels, {**base, "drop_speed_rel_min": s, "drop_rel_min": d})
            cell = f"{c}/{f}"
            if s == base["drop_speed_rel_min"] and d == base["drop_rel_min"]:
                cell = f"**{cell}**"
            cells.append(cell)
        L.append(f"| {s:g} | " + " | ".join(cells) + " |")
    L += ["", f"## ar_wide_max alone (speed {base['drop_speed_rel_min']}, drop {base['drop_rel_min']})\n",
          "| ar_wide_max | " + " | ".join(f"{a:g}" for a in ARS) + " |", "|---|" + "---|" * len(ARS)]
    cells = []
    for a in ARS:
        c, f = run(tracks, labels, {**base, "ar_wide_max": a})
        cell = f"{c}/{f}"
        cells.append(f"**{cell}**" if a == base["ar_wide_max"] else cell)
    L.append("| falls / FA | " + " | ".join(cells) + " |")
    L += ["", "Read across a row: a looser drop threshold adds few falls and few false alarms, because the misses are slow "
          "falls that no drop threshold separates from lying down. Read down a column: the speed threshold is the knob that "
          "trades false alarms for the slow falls."]
    out = EVAL / "SENSITIVITY.md"
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L)); print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
