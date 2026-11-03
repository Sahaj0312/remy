"""Head-to-head of the triage backends on the flagged eval clips and on a fixed set of check-in replies.
Writes eval/TRIAGE_COMPARE.md.

  python eval/triage_compare.py --run spark_open           # Spark Open (Fireworks), frames mode -> out/triage/spark_open/
  python eval/triage_compare.py --run spark --frames       # hosted Spark 1.3 in frames mode -> out/triage/spark_frames/
  python eval/triage_compare.py --replies spark,spark_open # the reply classifier on REPLIES
  python eval/triage_compare.py --report                   # compare whatever exists (incl. the committed video-mode Spark verdicts)
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from remy.common import CLIPS, EVAL, OUT, load_config, load_labels, read_json, write_json
from remy.triage import get_backend

# (transcript, expected status). Expected = what a careful listener would say; the rule layer is not applied here,
# so "help" phrases test the model, not the keyword list.
REPLIES = [
    ("Yeah.", "okay"), ("I'm fine", "okay"), ("yes yes I'm alright", "okay"), ("no no I'm fine, just slipped", "okay"),
    ("all good", "okay"), ("haan main theek hoon", "okay"), ("theek hai", "okay"), ("I'm okay, don't worry", "okay"),
    ("help", "needs_help"), ("help me please", "needs_help"), ("bachao", "needs_help"), ("I can't get up", "needs_help"),
    ("my hip hurts a lot", "needs_help"), ("call someone", "needs_help"), ("it hurts", "needs_help"),
    ("Hello?", "unclear"), ("what time is it", "unclear"), ("and now the weather for tomorrow", "unclear"),
    ("hmm", "unclear"), ("who is that", "unclear"), ("the news at six", "unclear"),
    ("I'm not okay", "needs_help"), ("not really", "needs_help"), ("ow ow ow", "needs_help"),
]
QUESTION = "Are you okay? Can you hear me?"


def flagged_clips() -> list[str]:
    d = OUT / "scores" / "sam_v2"
    return [r["clip"] for r in load_labels() if (d / f"{r['clip']}.json").exists() and read_json(d / f"{r['clip']}.json").get("flag")]


def run_backend(name: str, frames: bool, parallel: int, force: bool):
    cfg = load_config()
    out_dir = OUT / "triage" / (f"{name}_frames" if (name == "spark" and frames) else name)
    names = [c for c in flagged_clips() if force or not (out_dir / f"{c}.json").exists()]
    print(f"{len(names)} clips -> {out_dir.relative_to(EVAL)}", flush=True)
    backend = get_backend({**cfg, "triage": {"backend": name}}, timeout=300)

    def one(c):
        score = read_json(OUT / "scores" / "sam_v2" / f"{c}.json"); track = read_json(OUT / "sam" / f"{c}.json")
        best = score["best"]
        try:
            doc = backend.triage_fall(best, track["fps"], track["n_frames"], CLIPS / f"{c}.mp4", f"{best['end_t']:.1f} s into the clip",
                                      "frames" if (frames or name != "spark") else "video")
            write_json(out_dir / f"{c}.json", {"clip": c, **doc})
            v = doc["verdict"]
            print(f"{c:22s} {v['urgency']:6s} still={str(v['is_lying_still']):5s} {doc['latency_s']:5.1f}s ${doc['cost_usd']:.4f} :: {v['summary_for_family'][:80]}", flush=True)
        except Exception as e:
            print(f"{c}: FAILED {type(e).__name__}: {str(e)[:160]}", flush=True)
            write_json(out_dir / f"{c}.error.json", {"clip": c, "error": repr(e)})
    with ThreadPoolExecutor(parallel) as ex:
        list(ex.map(one, names))


def run_replies(names: list[str]):
    cfg = load_config()
    out = {}
    for name in names:
        backend = get_backend({**cfg, "triage": {"backend": name}}, timeout=120)
        rows = []
        for text, expected in REPLIES:
            r = backend.classify_reply(text, QUESTION)
            rows.append({"transcript": text, "expected": expected, "status": r["status"], "reason": r.get("reason", ""),
                         "latency_s": r.get("latency_s"), "cost_usd": r.get("cost_usd")})
            print(f"{name:11s} {text!r:40s} -> {r['status']:10s} ({'ok' if r['status'] == expected else 'expected ' + expected}) {r.get('latency_s')}s", flush=True)
        out[name] = rows
        write_json(OUT / "triage" / f"replies_{name}.json", rows)
    return out


def load_set(d):
    p = OUT / "triage" / d if d != "spark_video" else OUT / "spark" / "sam_v2"
    return {f.stem: read_json(f) for f in p.glob("*.json") if not f.name.endswith(".error.json")} if p.exists() else {}


def report():
    by_clip = {r["clip"]: r for r in load_labels()}
    sets = {k: v for k, v in {"spark_video": load_set("spark_video"), "spark_frames": load_set("spark_frames"), "spark_open": load_set("spark_open")}.items() if v}
    label = {"spark_video": "Spark 1.3, video burst (Meta API)", "spark_frames": "Spark 1.3, frames (Meta API)", "spark_open": "Spark Open, frames (Fireworks)"}
    L = ["# Triage backends compared\n",
         "The same flagged clips, the same system prompt and the same tracker numbers. Hosted Spark 1.3 sees the burst as a video "
         "file (the live path) or as JPEG frames; Spark Open on Fireworks sees frames, because the partner endpoint takes images "
         "only. Urgency is a judgment call, so agreement is the interesting number, not accuracy.\n",
         "| backend | clips | high / medium / low | lying still | median latency | mean cost | mean tokens in / out |", "|---|---|---|---|---|---|---|"]
    for k, docs in sets.items():
        urg = Counter(d["verdict"]["urgency"] for d in docs.values()); still = sum(bool(d["verdict"]["is_lying_still"]) for d in docs.values())
        lat = st.median(d["latency_s"] for d in docs.values()); cost = st.mean(d["cost_usd"] for d in docs.values())
        tin = st.mean(d["usage"]["input_tokens"] for d in docs.values()); tout = st.mean(d["usage"]["output_tokens"] for d in docs.values())
        L.append(f"| {label[k]} | {len(docs)} | {urg['high']} / {urg['medium']} / {urg['low']} | {still}/{len(docs)} | {lat:.1f} s | ${cost:.4f} | {tin:.0f} / {tout:.0f} |")
    keys = list(sets)
    if len(keys) >= 2:
        L += ["", "## Agreement\n", "| pair | same urgency | within one step | same lying-still |", "|---|---|---|---|"]
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a, b = sets[keys[i]], sets[keys[j]]; common = sorted(set(a) & set(b))
                if not common:
                    continue
                order = {"low": 0, "medium": 1, "high": 2}
                same = sum(a[c]["verdict"]["urgency"] == b[c]["verdict"]["urgency"] for c in common)
                near = sum(abs(order[a[c]["verdict"]["urgency"]] - order[b[c]["verdict"]["urgency"]]) <= 1 for c in common)
                still = sum(bool(a[c]["verdict"]["is_lying_still"]) == bool(b[c]["verdict"]["is_lying_still"]) for c in common)
                L.append(f"| {label[keys[i]]} vs {label[keys[j]]} | {same}/{len(common)} | {near}/{len(common)} | {still}/{len(common)} |")
        L += ["", "## Per clip\n", "| clip | type | " + " | ".join(label[k] for k in keys) + " |", "|---|---|" + "---|" * len(keys)]
        for c in sorted(set().union(*[set(s) for s in sets.values()])):
            cells = []
            for k in keys:
                d = sets[k].get(c)
                cells.append(f"{d['verdict']['urgency']}, {'still' if d['verdict']['is_lying_still'] else 'moving'}" if d else "-")
            L.append(f"| {c} | {by_clip[c]['type']} | " + " | ".join(cells) + " |")
        L += ["", "## Family notes, side by side (first five)\n"]
        for c in sorted(set.intersection(*[set(s) for s in sets.values()]))[:5]:
            L.append(f"- **{c}**")
            for k in keys:
                L.append(f"  - {label[k]}: \"{sets[k][c]['verdict']['summary_for_family']}\"")
    # replies
    reps = {k: read_json(OUT / "triage" / f"replies_{k}.json") for k in ("spark", "spark_open") if (OUT / "triage" / f"replies_{k}.json").exists()}
    if reps:
        L += ["", "## The reply classifier on 24 fixed transcripts\n",
              "What the model decides from the words alone (the help-word rule is bypassed here). Expected labels are a careful listener's.\n",
              "| backend | agree with expected | median latency | mean cost |", "|---|---|---|---|"]
        for k, rows in reps.items():
            ok = sum(r["status"] == r["expected"] for r in rows)
            L.append(f"| {'Spark 1.3 (Meta API)' if k == 'spark' else 'Spark Open (Fireworks)'} | {ok}/{len(rows)} | {st.median(r['latency_s'] for r in rows):.1f} s | ${st.mean(r['cost_usd'] for r in rows):.4f} |")
        L += ["", "| transcript | expected | " + " | ".join(reps) + " |", "|---|---|" + "---|" * len(reps)]
        for i, (text, expected) in enumerate(REPLIES):
            cells = [reps[k][i]["status"] + ("" if reps[k][i]["status"] == expected else " ✗") for k in reps]
            L.append(f"| {text} | {expected} | " + " | ".join(cells) + " |")
    out = EVAL / "TRIAGE_COMPARE.md"
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L)); print(f"\nwrote {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", help="backend name to run over the flagged clips: spark | spark_open")
    p.add_argument("--frames", action="store_true", help="with --run spark: send frames instead of a video file")
    p.add_argument("--replies", help="comma-separated backends to run over the fixed reply list")
    p.add_argument("--report", action="store_true")
    p.add_argument("--parallel", type=int, default=3)
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    if a.run:
        run_backend(a.run, a.frames, a.parallel, a.force)
    if a.replies:
        run_replies(a.replies.split(","))
    if a.report or not (a.run or a.replies):
        report()


if __name__ == "__main__":
    main()
