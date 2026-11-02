"""Eval batch: run a rule score over every track in eval/out/<tracker>/ -> eval/out/scores/<tracker>_<scorer>/.

  python eval/score_tracks.py --tracker sam --scorer v2
  python eval/score_tracks.py --tracker yolo --scorer trigger --quiet
Scorers live in remy.vision.score; thresholds in config.json (score_v1 / score_v2 / trigger).
"""
from __future__ import annotations

import argparse
import json

from remy.common import OUT, load_config, load_labels, read_json, write_json
from remy.vision.score import score_track


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tracker", default="sam", help="track dir under eval/out/, e.g. sam, yolo, sam_5fps")
    p.add_argument("--scorer", default="v2", choices=["v1", "v2", "trigger"])
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()
    full = load_config()
    cfg = {"v1": full["score_v1"], "v2": full["score_v2"], "trigger": {**full["score_v2"], **full["trigger"]}}[args.scorer]
    out_dir = OUT / "scores" / f"{args.tracker}_{args.scorer}"
    rows = []
    for lab in load_labels():
        tp = OUT / args.tracker / f"{lab['clip']}.json"
        if not tp.exists():
            continue
        res = score_track(read_json(tp), cfg, args.scorer)
        res.update(label=lab["label"], source=lab["source"], type=lab.get("type"))
        write_json(out_dir / f"{lab['clip']}.json", res)
        rows.append(res)
    falls = [r for r in rows if r["label"] == "fall"]; adls = [r for r in rows if r["label"] == "adl"]
    summary = {"tracker": args.tracker, "scorer": args.scorer, "n_scored": len(rows),
               "falls_total": len(falls), "falls_caught": sum(r["flag"] for r in falls),
               "adl_total": len(adls), "adl_false_alarms": sum(r["flag"] for r in adls),
               "hold_unknown_flagged": sum(1 for r in rows if r["flag"] and r.get("hold") == "unknown"),
               "config": cfg}
    write_json(out_dir / "_summary.json", summary)
    if not args.quiet:
        for r in rows:
            b = r.get("best") or {}
            if args.scorer == "v1":
                detail = (f"v={b.get('drop_speed_max','-')} drop={b.get('total_drop','-')} arpre={b.get('ar_pre','-')} "
                          f"arpost={b.get('ar_post','-')} low={b.get('low_frac','-')} [{'D' if b.get('drop_ok') else '.'}"
                          f"{'S' if b.get('shape_ok') else '.'}{'H' if b.get('hold_ok') else '.'}]")
            else:
                detail = (f"vrel={b.get('drop_speed_rel_max','-')} droprel={b.get('drop_rel','-')} body={b.get('body_len','-')} "
                          f"arpost={b.get('ar_post','-')} hold={b.get('hold','-')}({b.get('no_getup_observed_s','-')}s) "
                          f"[{'D' if b.get('drop_ok') else '.'}{'S' if b.get('shape_ok') else '.'}]")
            print(f"{r['clip']:22s} {r['label']:4s} {str(r.get('type','')):20s} flag={str(r['flag']):5s} cov={r['coverage']:.2f} {detail}")
    print(json.dumps({k: v for k, v in summary.items() if k != "config"}))


if __name__ == "__main__":
    main()
