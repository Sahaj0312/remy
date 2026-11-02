"""Eval batch: Muse Spark triage of every clip the rule score flagged.
Reads eval/out/scores/<tracker>_<scorer>/<clip>.json, writes eval/out/spark/<tracker>_<scorer>/<clip>.json.

  python eval/spark_triage.py --score-dir sam_v2
  python eval/spark_triage.py gmdcsa_S1_Fall_12 --force
"""
from __future__ import annotations

import argparse

from remy.common import CLIPS, OUT, load_config, load_labels, read_json, write_json
from remy.triage.spark import SparkTriage


def triage(backend: SparkTriage, name: str, score_dir: str, args) -> dict:
    score = read_json(OUT / "scores" / score_dir / f"{name}.json")
    tracker = score_dir.split("_")[0]
    track = read_json(OUT / tracker / f"{name}.json")
    best = score["best"]
    mode = "frames" if args.frames else "video"
    doc = backend.triage_fall(best, track["fps"], track["n_frames"], CLIPS / f"{name}.mp4",
                              f"{best['end_t']:.1f} s into the clip", mode)
    doc = {"clip": name, "score_dir": score_dir, **doc}
    write_json(OUT / "spark" / score_dir / f"{name}.json", doc)
    v, u = doc["verdict"], doc["usage"]
    print(f"{name:22s} {v.get('urgency','?'):6s} still={str(v.get('is_lying_still')):5s} {doc['latency_s']:.1f}s "
          f"in={u['input_tokens']} out={u['output_tokens']} ${doc['cost_usd']:.4f} :: {v.get('summary_for_family','')[:100]}", flush=True)
    return doc


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("clips", nargs="*", help="default: every flagged clip")
    p.add_argument("--score-dir", default="sam_v2", help="eval/out/scores/<dir> to read flags from")
    p.add_argument("--all", action="store_true", help="every clip with an event, not only flagged")
    p.add_argument("--force", action="store_true")
    p.add_argument("--frames", action="store_true", help="send JPEG frames instead of a video file")
    args = p.parse_args()
    cfg = load_config()["spark"]
    names = args.clips
    if not names:
        names = [r["clip"] for r in load_labels()
                 if (OUT / "scores" / args.score_dir / f"{r['clip']}.json").exists()
                 and (lambda sc: (sc.get("flag") or args.all) and sc.get("best"))(read_json(OUT / "scores" / args.score_dir / f"{r['clip']}.json"))]
    if not args.force:
        names = [n for n in names if not (OUT / "spark" / args.score_dir / f"{n}.json").exists()]
    print(f"{len(names)} clips to triage with {cfg['model']} ({'frames' if args.frames else 'video'})", flush=True)
    backend = SparkTriage(cfg, timeout=300)
    for n in names:
        try:
            triage(backend, n, args.score_dir, args)
        except Exception as e:
            print(f"{n}: FAILED {type(e).__name__}: {e}", flush=True)
            write_json(OUT / "spark" / args.score_dir / f"{n}.error.json", {"clip": n, "error": repr(e)})


if __name__ == "__main__":
    main()
