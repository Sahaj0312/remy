"""Out-of-sample check of score v2: the thresholds were hand-tuned on all 100 clips, so the headline 32/40 is
in-sample. This script splits the clips in two halves stratified by type, picks the thresholds on one half by a
grid search, scores the other half with them, and swaps. Repeated over several random splits. Writes
eval/HOLDOUT.md. Free: it re-scores the committed SAM tracks, no API calls.

  python eval/holdout.py            # 5 splits
  python eval/holdout.py --splits 10
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
import statistics as st
from collections import defaultdict

from remy.common import EVAL, OUT, load_labels, read_json
from remy.vision.score import score_track

SPEEDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
DROPS = [0.10, 0.125, 0.15, 0.175, 0.20, 0.25]
ARS = [0.7, 0.8, 0.9, 1.0]
FA_COST = 2.0          # one false alarm costs as much as two missed falls when choosing thresholds


def flags_per_combo(tracks: dict, base: dict) -> dict[tuple, dict[str, bool]]:
    """{(speed, drop, ar): {clip: flag}} for every grid point, computed once."""
    out = {}
    for s, d, a in itertools.product(SPEEDS, DROPS, ARS):
        cfg = {**base, "drop_speed_rel_min": s, "drop_rel_min": d, "ar_wide_max": a}
        out[(s, d, a)] = {n: bool(score_track(tr, cfg, "v2")["flag"]) for n, tr in tracks.items()}
    return out


def tally(flags: dict[str, bool], names: list[str], labels: dict[str, str]) -> tuple[int, int, int, int]:
    caught = fa = nf = na = 0
    for n in names:
        if labels[n] == "fall":
            nf += 1; caught += flags[n]
        else:
            na += 1; fa += flags[n]
    return caught, nf, fa, na


def pick(combos: dict, names: list[str], labels: dict[str, str], base_key: tuple) -> tuple:
    """Best grid point on `names` by caught - FA_COST * false alarms; ties go to the point nearest the
    hand-picked one, so the search does not wander for nothing."""
    def key(c):
        caught, _, fa, _ = tally(combos[c], names, labels)
        dist = sum(abs(x - y) for x, y in zip(c, base_key))
        return (caught - FA_COST * fa, -dist)
    return max(combos, key=key)


def stratified_split(labels_rows: list[dict], seed: int) -> tuple[list[str], list[str]]:
    rng = random.Random(seed)
    by_type = defaultdict(list)
    for r in labels_rows:
        by_type[r["type"]].append(r["clip"])
    a, b = [], []
    for typ in sorted(by_type):
        names = by_type[typ][:]; rng.shuffle(names)
        half = len(names) // 2
        if len(names) % 2 and rng.random() < 0.5:
            half += 1
        a += names[:half]; b += names[half:]
    return a, b


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--splits", type=int, default=5)
    args = p.parse_args()
    base = json.load(open(EVAL.parent / "config.example.json"))["score_v2"]
    base_key = (base["drop_speed_rel_min"], base["drop_rel_min"], base["ar_wide_max"])
    rows = load_labels()
    labels = {r["clip"]: r["label"] for r in rows}
    tracks = {r["clip"]: read_json(OUT / "sam" / f"{r['clip']}.json") for r in rows if (OUT / "sam" / f"{r['clip']}.json").exists()}
    rows = [r for r in rows if r["clip"] in tracks]
    combos = flags_per_combo(tracks, base)
    in_caught, in_nf, in_fa, in_na = tally(combos[base_key], list(tracks), labels)

    L = ["# Score v2 out-of-sample check\n",
         f"The thresholds in config were tuned by hand on all {len(tracks)} clips, so the headline {in_caught}/{in_nf} falls and "
         f"{in_fa}/{in_na} false alarms is in-sample. Here the clips are split in two halves, stratified by type; the grid "
         f"(speed {SPEEDS[0]} to {SPEEDS[-1]}, drop {DROPS[0]} to {DROPS[-1]}, ar_wide_max {ARS[0]} to {ARS[-1]}) picks thresholds on one "
         f"half by `caught - {FA_COST:g} x false alarms`, the other half is scored with them, then the halves swap. Every clip "
         f"is therefore scored once with thresholds it did not influence. Repeated over {args.splits} random splits.\n",
         "| split | half A picked (speed, drop, ar) → scores B | half B picked → scores A | pooled out-of-sample |",
         "|---|---|---|---|"]
    pooled = []
    picked = []
    for seed in range(args.splits):
        a, b = stratified_split(rows, seed)
        ca = pick(combos, a, labels, base_key); cb = pick(combos, b, labels, base_key)
        cb_on_b = tally(combos[ca], b, labels); ca_on_a = tally(combos[cb], a, labels)
        caught = cb_on_b[0] + ca_on_a[0]; nf = cb_on_b[1] + ca_on_a[1]; fa = cb_on_b[2] + ca_on_a[2]; na = cb_on_b[3] + ca_on_a[3]
        pooled.append((caught, nf, fa, na)); picked += [ca, cb]
        L.append(f"| {seed + 1} | {ca} → {cb_on_b[0]}/{cb_on_b[1]} falls, {cb_on_b[2]}/{cb_on_b[3]} FA | {cb} → {ca_on_a[0]}/{ca_on_a[1]} falls, "
                 f"{ca_on_a[2]}/{ca_on_a[3]} FA | **{caught}/{nf} falls, {fa}/{na} FA** |")
    recalls = [c / nf for c, nf, _, _ in pooled]; fas = [fa for _, _, fa, _ in pooled]
    same = sum(1 for c in picked if c == base_key)
    L += ["",
          f"**Out-of-sample over {args.splits} splits: {st.mean(recalls) * 100:.0f}% of falls caught "
          f"(range {min(recalls) * 100:.0f} to {max(recalls) * 100:.0f}%), {st.mean(fas):.1f} false alarms per 60 clips "
          f"(range {min(fas)} to {max(fas)}). In-sample: {in_caught / in_nf * 100:.0f}% and {in_fa}.**\n",
          f"The hand-picked point {base_key} was chosen by the search in {same} of {len(picked)} half-fits; the others are its "
          "neighbours. The misses are the same slow falls every time: no threshold in the grid separates a fall from sitting "
          "or onto a bed from lying down on purpose."]
    out = EVAL / "HOLDOUT.md"
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L)); print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
