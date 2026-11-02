"""Assemble eval/REPORT.md: per-type results, before/after tuning, YOLO trigger rate, Spark triage, latency, cost."""
from __future__ import annotations

import statistics as st
from collections import Counter, defaultdict

from remy.common import EVAL, OUT, load_labels, read_json


def load_scores(d: str) -> dict[str, dict]:
    p = OUT / "scores" / d
    return {f.stem: read_json(f) for f in p.glob("*.json") if not f.name.startswith("_")} if p.exists() else {}


def med(xs):
    xs = [x for x in xs if x is not None and x >= 0]
    return (st.median(xs), st.mean(xs), max(xs)) if xs else (None, None, None)


def fmt(x, nd=1):
    return "-" if x is None else f"{x:.{nd}f}"


def tf(v):
    return "-" if v is None else ("YES" if v else "no")


def main():
    labels = load_labels()
    by_clip = {r["clip"]: r for r in labels}
    S = {d: load_scores(d) for d in ("sam_v2", "sam_v1", "sam_5fps_v1", "sam_5fps_v2", "yolo_v2", "yolo_trigger", "yolo_5fps_v1")}
    spark = {f.stem: read_json(f) for f in (OUT / "spark" / "sam_v2").glob("*.json") if not f.name.endswith(".error.json")} if (OUT / "spark" / "sam_v2").exists() else {}
    sam_docs = [read_json(f) for f in (OUT / "sam").glob("*.json") if not f.name.endswith(".error.json")] if (OUT / "sam").exists() else []
    yolo_docs = [read_json(f) for f in (OUT / "yolo").glob("*.json")] if (OUT / "yolo").exists() else []
    L = ["# Fall-detection feasibility test, round 2\n"]
    n_f = sum(r["label"] == "fall" for r in labels); n_a = len(labels) - n_f
    L.append(f"{len(labels)} clips: {n_f} falls, {n_a} non-falls ({sum(r['source']=='urfall' for r in labels)} UR Fall, "
             f"{sum(r['source']=='gmdcsa24' for r in labels)} GMDCSA24), 10 fps, max width 640. "
             f"Lighting: {dict(Counter(r['lighting'] for r in labels))}.\n")

    # ---------------- before / after on the original 40
    first40 = [c for c in S["sam_5fps_v1"]]
    if first40:
        L.append("## Before / after tuning on the first-round 40 clips\n")
        L.append("| tracks | score | falls caught | false alarms | flagged with hold unknown |")
        L.append("|---|---|---|---|---|")
        for d, name in (("sam_5fps_v1", "5 fps, v1 (round 1)"), ("sam_5fps_v2", "5 fps, v2"), ("sam_v1", "10 fps, v1"), ("sam_v2", "10 fps, v2 (round 2)")):
            sc = {c: S[d][c] for c in first40 if c in S[d]}
            if not sc:
                continue
            f = [r for r in sc.values() if r["label"] == "fall"]; a = [r for r in sc.values() if r["label"] == "adl"]
            unk = sum(1 for r in sc.values() if r["flag"] and r.get("hold") == "unknown")
            L.append(f"| {name} | | {sum(r['flag'] for r in f)}/{len(f)} | {sum(r['flag'] for r in a)}/{len(a)} | {unk} |")
        L.append("\nv1 = absolute drop + tall-before + fixed 3 s low hold. v2 = drop in body lengths, ends-wide, no get-up within "
                 "no_getup_s (hold 'unknown' when the clip ends first, still flagged).\n")

    # ---------------- per type on the full set (SAM v2)
    sv2 = S["sam_v2"]
    if sv2:
        L.append("## Per-type results on the full set (SAM 3.1 tracks, score v2)\n")
        L.append("| type | clips | flagged by score | hold unknown | note |")
        L.append("|---|---|---|---|---|")
        per = defaultdict(list)
        for c, r in sv2.items():
            per[by_clip[c]["type"]].append(r)
        for typ in sorted(per, key=lambda t: (not t.startswith("fall"), t)):
            rs = per[typ]; flagged = sum(r["flag"] for r in rs); unk = sum(1 for r in rs if r["flag"] and r.get("hold") == "unknown")
            note = "caught" if typ.startswith("fall") else "false alarms"
            L.append(f"| {typ} | {len(rs)} | {flagged}/{len(rs)} | {unk} | {note} |")
        f = [r for r in sv2.values() if r["label"] == "fall"]; a = [r for r in sv2.values() if r["label"] == "adl"]
        unk_all = sum(1 for r in sv2.values() if r["flag"] and r.get("hold") == "unknown")
        unk_ok = sum(1 for r in sv2.values() if r["flag"] and r.get("hold") == "ok")
        L.append(f"\n**Overall: {sum(r['flag'] for r in f)}/{len(f)} falls caught, {sum(r['flag'] for r in a)}/{len(a)} false alarms.** "
                 f"Hold was 'unknown' (clip ended before {sv2[next(iter(sv2))].get('config', {}).get('no_getup_s', 'N')} s) on {unk_all} of the "
                 f"{unk_all + unk_ok} flagged clips.\n")
        missed = [c for c, r in sv2.items() if r['label'] == 'fall' and not r['flag']]
        fas = [c for c, r in sv2.items() if r['label'] == 'adl' and r['flag']]
        if missed:
            L.append("Missed falls: " + ", ".join(f"{c} ({by_clip[c]['type']})" for c in missed) + "\n")
        if fas:
            L.append("False alarms: " + ", ".join(f"{c} ({by_clip[c]['type']})" for c in fas) + "\n")

    # ---------------- YOLO trigger rate
    tr = S["yolo_trigger"]
    if tr:
        L.append("## YOLO always-on filter: would it have called SAM?\n")
        f = [r for r in tr.values() if r["label"] == "fall"]; a = [r for r in tr.values() if r["label"] == "adl"]
        L.append(f"YOLO pose + trigger rule (drop only, loose thresholds) fires on **{sum(r['flag'] for r in f)}/{len(f)} falls** "
                 f"and {sum(r['flag'] for r in a)}/{len(a)} non-falls (each firing = one SAM call).\n")
        per = defaultdict(list)
        for c, r in tr.items():
            per[by_clip[c]["type"]].append(r)
        L.append("| type | trigger fired |"); L.append("|---|---|")
        for typ in sorted(per, key=lambda t: (not t.startswith("fall"), t)):
            rs = per[typ]; L.append(f"| {typ} | {sum(r['flag'] for r in rs)}/{len(rs)} |")
        missed = [c for c, r in tr.items() if r["label"] == "fall" and not r["flag"]]
        if missed:
            L.append("\nFalls the YOLO trigger missed (never reach SAM): " + ", ".join(f"{c} ({by_clip[c]['type']}, coverage {tr[c]['coverage']:.2f})" for c in missed) + "\n")
        yv2 = S["yolo_v2"]
        if yv2:
            f = [r for r in yv2.values() if r["label"] == "fall"]; a = [r for r in yv2.values() if r["label"] == "adl"]
            L.append(f"\nFor reference, YOLO tracks + the full v2 score: {sum(r['flag'] for r in f)}/{len(f)} falls, {sum(r['flag'] for r in a)}/{len(a)} false alarms.\n")

    # ---------------- Spark triage
    if spark:
        L.append("## Muse Spark triage of flagged clips (urgency + family summary)\n")
        urg = Counter(d["verdict"].get("urgency") for d in spark.values())
        still = Counter(d["verdict"].get("is_lying_still") for d in spark.values())
        L.append(f"{len(spark)} flagged clips triaged. Urgency: {dict(urg)}. is_lying_still: {dict(still)}.\n")
        per = defaultdict(Counter)
        for c, d in spark.items():
            per[by_clip[c]["type"]][d["verdict"].get("urgency")] += 1
        L.append("| type | high | medium | low |"); L.append("|---|---|---|---|")
        for typ in sorted(per, key=lambda t: (not t.startswith("fall"), t)):
            L.append(f"| {typ} | {per[typ]['high']} | {per[typ]['medium']} | {per[typ]['low']} |")
        L.append("\nSample summaries:\n")
        for c in list(spark)[:6]:
            v = spark[c]["verdict"]
            L.append(f"- {c} ({by_clip[c]['type']}): **{v.get('urgency')}** — \"{v.get('summary_for_family')}\"")
        L.append("")

    # ---------------- per clip table
    L.append("## Per-clip results\n")
    L.append("| clip | type | lighting | true | SAM v2 (D/S/hold) | YOLO trigger | Spark urgency | lying still |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in labels:
        c = r["clip"]; s = sv2.get(c); t = tr.get(c); sp = spark.get(c)
        if s:
            b = s.get("best") or {}
            sam = f"{tf(s['flag'])} {'D' if b.get('drop_ok') else '.'}{'S' if b.get('shape_ok') else '.'}/{b.get('hold', '-') if b else '-'}"
        else:
            sam = "-"
        L.append(f"| {c} | {r['type']} | {r['lighting']} | {r['label']} | {sam} | {tf(t['flag']) if t else '-'} | "
                 f"{sp['verdict'].get('urgency') if sp else ('-' if not s or not s['flag'] else 'pending')} | "
                 f"{tf(sp['verdict'].get('is_lying_still')) if sp else '-'} |")
    L.append("")

    # ---------------- latency & cost
    if sam_docs:
        L.append("## SAM 3.1 latency and cost\n")
        up = med([d["timing"]["upload_s"] for d in sam_docs]); ff = med([d["timing"]["first_frame_s"] for d in sam_docs])
        tot = med([d["timing"]["total_s"] for d in sam_docs]); fr = [d["n_frames"] for d in sam_docs]
        L.append("| metric | median | mean | max |"); L.append("|---|---|---|---|")
        L.append(f"| upload (s) | {fmt(up[0])} | {fmt(up[1])} | {fmt(up[2])} |")
        L.append(f"| time to first frame (s) | {fmt(ff[0])} | {fmt(ff[1])} | {fmt(ff[2])} |")
        L.append(f"| total per clip (s) | {fmt(tot[0])} | {fmt(tot[1])} | {fmt(tot[2])} |")
        L.append(f"| frames per clip | {st.median(fr):.0f} | {st.mean(fr):.0f} | {max(fr)} |")
        sam_cost = sum(d["cost_usd"] for d in sam_docs)
        L.append(f"\nSAM: {len(sam_docs)} clips, {sum(fr)} frames sent, {sum(d['billed_frames'] for d in sam_docs)} billed, **${sam_cost:.3f}**.\n")
    else:
        sam_cost = 0.0
    if spark:
        lat = med([d["latency_s"] for d in spark.values()])
        ti = sum(d["usage"]["input_tokens"] for d in spark.values()); to = sum(d["usage"]["output_tokens"] for d in spark.values())
        sp_cost = sum(d["cost_usd"] for d in spark.values())
        L.append("## Muse Spark latency and cost\n")
        L.append(f"{len(spark)} calls ({Counter(d['mode'] for d in spark.values())}), latency median {fmt(lat[0])}s / mean {fmt(lat[1])}s / max {fmt(lat[2])}s. "
                 f"Tokens: {ti} in, {to} out. Cost **${sp_cost:.4f}** (${sp_cost/len(spark):.4f} per call).\n")
    else:
        sp_cost = 0.0
    r1 = 0.519
    L.append(f"**Round-2 API cost: ${sam_cost + sp_cost:.3f}. Round 1 was ${r1:.3f}. Running total ${sam_cost + sp_cost + r1:.2f}.**\n")
    if yolo_docs:
        ms = [d["timing"]["per_frame_ms_mean"] for d in yolo_docs]
        L.append(f"YOLO pose: mean {st.mean(ms):.0f} ms/frame on this Mac's CPU.\n")
    (EVAL / "REPORT.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
